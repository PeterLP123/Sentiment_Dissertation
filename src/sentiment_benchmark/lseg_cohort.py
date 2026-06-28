from __future__ import annotations

import csv
import io
import json
import random
import re
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .artifact_io import atomic_write_json, atomic_write_jsonl, atomic_write_text, canonical_json, sha256_file, sha256_text
from .lseg_corpus import load_verified_lseg_corpus
from .lseg_source import LsegNewsError, utc_now

LSEG_COHORT_SCHEMA_VERSION = 1
DECISIONS = {"include", "review", "exclude"}


@dataclass(frozen=True)
class CohortConfig:
    output_dir: Path
    timezone: str = "America/New_York"
    development_fraction: float = 0.7
    development_size: int = 2100
    holdout_size: int = 900
    l3_size: int = 500
    seed: int = 42
    lead_window_chars: int = 2000
    min_body_mentions: int = 2
    overrides_path: Path | None = None

    @property
    def cohort_size(self) -> int:
        return self.development_size + self.holdout_size


@dataclass(frozen=True)
class LsegCohortResult:
    output_dir: Path
    manifest_path: Path
    cohort_path: Path
    screening_path: Path
    coverage_path: Path
    l3_path: Path
    cohort_count: int


def load_cohort_config(path: str | Path) -> CohortConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    raw = payload.get("cohort")
    if not isinstance(raw, dict):
        raise LsegNewsError("analysis config must contain a [cohort] table")
    output = str(raw.get("output_dir") or "").strip()
    if not output:
        raise LsegNewsError("cohort.output_dir is required")
    overrides = str(raw.get("overrides_path") or "").strip()
    config = CohortConfig(
        output_dir=Path(output),
        timezone=str(raw.get("timezone") or "America/New_York"),
        development_fraction=float(raw.get("development_fraction", 0.7)),
        development_size=int(raw.get("development_size", 2100)),
        holdout_size=int(raw.get("holdout_size", 900)),
        l3_size=int(raw.get("l3_size", 500)),
        seed=int(raw.get("seed", 42)),
        lead_window_chars=int(raw.get("lead_window_chars", 2000)),
        min_body_mentions=int(raw.get("min_body_mentions", 2)),
        overrides_path=Path(overrides) if overrides else None,
    )
    if not 0 < config.development_fraction < 1:
        raise LsegNewsError("cohort.development_fraction must be between 0 and 1")
    if min(config.development_size, config.holdout_size, config.l3_size) < 1:
        raise LsegNewsError("cohort sizes must be positive")
    if config.l3_size > config.development_size:
        raise LsegNewsError("cohort.l3_size cannot exceed development_size")
    ZoneInfo(config.timezone)
    return config


def _aliases(manifest: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    companies = ((manifest.get("config") or {}).get("companies") or [])
    result: dict[str, tuple[str, ...]] = {}
    for company in companies:
        if not isinstance(company, dict):
            continue
        symbol = str(company.get("symbol") or "").upper()
        values = [str(value).strip() for value in company.get("aliases") or [] if str(value).strip()]
        name = str(company.get("name") or "").strip()
        if name:
            values.append(name)
        if symbol:
            result[symbol] = tuple(dict.fromkeys(values))
    return result


def _alias_matches(text: str, aliases: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for alias in aliases:
        flags = 0 if alias.isupper() and len(alias) <= 5 else re.IGNORECASE
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, flags):
            matches.append(alias)
    return matches


def _relevance(record: dict[str, Any], aliases: tuple[str, ...], config: CohortConfig) -> tuple[str, str, int]:
    headline = str(record.get("headline") or "")
    body = str(record.get("clean_text") or "")
    headline_matches = _alias_matches(headline, aliases)
    body_matches = _alias_matches(body, aliases)
    lead_matches = _alias_matches(body[: config.lead_window_chars], aliases)
    mention_count = sum(
        len(
            re.findall(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                body,
                0 if alias.isupper() and len(alias) <= 5 else re.IGNORECASE,
            )
        )
        for alias in aliases
    )
    if headline_matches:
        return "include", "target_alias_in_headline", mention_count
    if lead_matches and mention_count >= config.min_body_mentions:
        return "include", "target_alias_in_lead_and_body", mention_count
    if body_matches:
        return "review", "target_alias_only_in_body", mention_count
    return "exclude", "no_target_alias", 0


def _load_overrides(path: Path | None) -> dict[tuple[str, str], tuple[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise LsegNewsError(f"cohort overrides file does not exist: {path}")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for index, item in enumerate(payload.get("overrides") or [], start=1):
        family = str(item.get("story_family_id") or "").strip()
        symbol = str(item.get("symbol") or "").strip().upper()
        decision = str(item.get("decision") or "").strip().lower()
        reason = str(item.get("reason") or "").strip()
        if not family or not symbol or decision not in DECISIONS or not reason:
            raise LsegNewsError(f"invalid cohort override {index}")
        result[(family, symbol)] = decision, reason
    return result


def _local_date(timestamp: str, timezone: str) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise LsegNewsError(f"version_created lacks timezone: {timestamp}")
    return parsed.astimezone(ZoneInfo(timezone)).date().isoformat()


def _sample_stratified(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < size:
        raise LsegNewsError(f"requested {size} cohort rows, but only {len(rows)} are available")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["symbol"]), str(row["news_date"]))].append(row)
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    for values in groups.values():
        generator.shuffle(values)
    selected: list[dict[str, Any]] = []
    while len(selected) < size:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: (row["news_date"], row["symbol"], row["story_family_id"]))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return atomic_write_text(path, stream.getvalue())


def build_lseg_analysis_cohort(corpus_manifest: str | Path, analysis_config: str | Path) -> LsegCohortResult:
    config = load_cohort_config(analysis_config)
    if config.output_dir.exists():
        raise LsegNewsError(f"refusing to overwrite cohort output directory: {config.output_dir}")
    manifest, records = load_verified_lseg_corpus(corpus_manifest)
    aliases_by_symbol = _aliases(manifest)
    overrides = _load_overrides(config.overrides_path)
    screening: list[dict[str, Any]] = []
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("scoring_eligible") or not record.get("version_created"):
            continue
        family = str(record.get("story_family_id") or "").strip()
        for symbol_value in record.get("matched_symbols") or []:
            symbol = str(symbol_value).upper()
            aliases = aliases_by_symbol.get(symbol, (symbol,))
            decision, reason, mentions = _relevance(record, aliases, config)
            if (family, symbol) in overrides:
                decision, override_reason = overrides[(family, symbol)]
                reason = f"manual_override:{override_reason}"
            row = {
                "article_id": record.get("article_id"),
                "revision_id": record.get("revision_id"),
                "story_id": record.get("story_id"),
                "story_family_id": family,
                "symbol": symbol,
                "headline": record.get("headline") or "",
                "clean_text": record.get("clean_text") or "",
                "clean_text_sha256": record.get("clean_text_sha256") or "",
                "version_created": record.get("version_created"),
                "news_date": _local_date(str(record["version_created"]), config.timezone),
                "relevance_decision": decision,
                "relevance_reason": reason,
                "entity_mention_count": mentions,
                "selected_revision": False,
            }
            screening.append(row)
            if decision == "include":
                candidates[(family, symbol)].append(row)
    selected: list[dict[str, Any]] = []
    for values in candidates.values():
        chosen = min(values, key=lambda row: (str(row["version_created"]), str(row["story_id"])))
        chosen["selected_revision"] = True
        selected.append(chosen)
    dates = sorted({str(row["news_date"]) for row in selected})
    split_at = max(1, min(len(dates) - 1, round(len(dates) * config.development_fraction)))
    development_dates = set(dates[:split_at])
    development = _sample_stratified(
        [row for row in selected if row["news_date"] in development_dates],
        config.development_size,
        config.seed,
    )
    holdout = _sample_stratified(
        [row for row in selected if row["news_date"] not in development_dates],
        config.holdout_size,
        config.seed + 1,
    )
    for row in development:
        row["chronological_split"] = "development"
    for row in holdout:
        row["chronological_split"] = "holdout"
    cohort = sorted([*development, *holdout], key=lambda row: (row["news_date"], row["symbol"], row["story_family_id"]))
    l3 = _sample_stratified(development, config.l3_size, config.seed + 2)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    cohort_path = atomic_write_jsonl(config.output_dir / "cohort.jsonl", cohort)
    atomic_write_jsonl(config.output_dir / "development.jsonl", development)
    atomic_write_jsonl(config.output_dir / "holdout.jsonl", holdout)
    l3_path = atomic_write_jsonl(config.output_dir / "l3_subset.jsonl", l3)
    screen_fields = [
        "story_family_id",
        "story_id",
        "revision_id",
        "symbol",
        "version_created",
        "news_date",
        "relevance_decision",
        "relevance_reason",
        "entity_mention_count",
        "selected_revision",
    ]
    screening_path = _write_csv(config.output_dir / "screening_index.csv", screening, screen_fields)
    coverage_rows: list[dict[str, Any]] = []
    coverage = Counter((row["symbol"], row["news_date"], row.get("chronological_split", "not_selected")) for row in cohort)
    for (symbol, news_date, split), count in sorted(coverage.items()):
        coverage_rows.append({"symbol": symbol, "news_date": news_date, "split": split, "events": count})
    coverage_path = _write_csv(config.output_dir / "coverage_by_symbol_date.csv", coverage_rows, ["symbol", "news_date", "split", "events"])
    files = {path.name: {"path": path.name, "sha256": sha256_file(path)} for path in (cohort_path, screening_path, coverage_path, l3_path)}
    manifest_path = config.output_dir / "manifest.json"
    fingerprint_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }
    atomic_write_json(manifest_path, {
        "schema_version": LSEG_COHORT_SCHEMA_VERSION,
        "status": "completed",
        "created_at": utc_now(),
        "source_corpus_manifest": str(Path(corpus_manifest)),
        "source_corpus_manifest_sha256": sha256_file(corpus_manifest),
        "analysis_config": str(Path(analysis_config)),
        "analysis_config_sha256": sha256_file(analysis_config),
        "config_fingerprint": sha256_text(canonical_json(fingerprint_payload)),
        "revision_policy": "earliest_relevance_passing_revision_per_family_symbol",
        "counts": {
            "screened": len(screening),
            "family_symbol_includes": len(selected),
            "development": len(development),
            "holdout": len(holdout),
            "cohort": len(cohort),
            "l3": len(l3),
        },
        "chronological_split": {
            "development_fraction": config.development_fraction,
            "development_dates": sorted(development_dates),
            "holdout_dates": sorted(set(dates) - development_dates),
        },
        "sampling": {
            "seed": config.seed,
            "development_size": config.development_size,
            "holdout_size": config.holdout_size,
            "l3_size": config.l3_size,
        },
        "files": files,
        "sharing": {"redistribute": False, "licensed_full_text": True},
    })
    return LsegCohortResult(config.output_dir, manifest_path, cohort_path, screening_path, coverage_path, l3_path, len(cohort))


def load_verified_lseg_cohort(manifest_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(manifest_path)
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "completed" or manifest.get("schema_version") != LSEG_COHORT_SCHEMA_VERSION:
        raise LsegNewsError(f"not a completed schema-v{LSEG_COHORT_SCHEMA_VERSION} cohort: {path}")
    entry = (manifest.get("files") or {}).get("cohort.jsonl")
    if not isinstance(entry, dict):
        raise LsegNewsError("cohort manifest is missing cohort.jsonl")
    cohort_path = path.parent / str(entry.get("path") or "")
    if not cohort_path.exists() or sha256_file(cohort_path) != entry.get("sha256"):
        raise LsegNewsError(f"cohort hash mismatch: {cohort_path}")
    rows = [json.loads(line) for line in cohort_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return manifest, rows
