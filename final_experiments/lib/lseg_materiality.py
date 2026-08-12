"""Return-blind LSEG-33 materiality measurement pilot.

The module freezes a deterministic 2,000-event Reuters sample without reading
prices or returns, retrieves only the selected licensed story bodies, scores a
strict joint annotation through the authorised DeepInfra/OpenRouter route, and
builds blinded human-audit packets.  Return access is deliberately a separate
stage and is blocked until the human reliability gate exists and passes.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
import time
import tomllib
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from final_experiments.lib.joined_lseg import JoinedLsegInputs, load_joined_same33_scores
from final_experiments.lib.lseg_story_families import load_first_release_hashes
from final_experiments.lib.novelty import headline_norm_sha256
from sentiment_benchmark.artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from sentiment_benchmark.env import load_env_file
from sentiment_benchmark.lseg_source import LsegNewsClient
from sentiment_benchmark.news_cleaning import QUALITY_OK, clean_news_html
from sentiment_benchmark.prompts import prompt_hash

SAMPLE_SIZE = 2_000
PRIMARY_AUDIT_SIZE = 200
REPRESENTATIVE_AUDIT_SIZE = 150
TAIL_AUDIT_SIZE = 50
SECOND_CODER_SIZE = 60
RANDOM_SEED = 20_260_812
INFORMATION_BUFFER_MINUTES = 15
PRIOR_WINDOW_DAYS = 30
MAXIMUM_PRIOR_HEADLINES = 5
MAXIMUM_BODY_CHARS = 6_000
REUTERS_SOURCE_CODE = "NS:RTRS"

MODEL_ID = "google/gemma-4-26b-a4b-it"
PROVIDER_NAME = "DeepInfra"
PROVIDER_SLUG = "deepinfra"
PROMPT_ID = "lseg_target_materiality_novelty_horizon_v1"
DEFAULT_PROMPT_PATH = Path("configs/strategy_research/lseg_materiality_joint_prompt_v1.toml")
INPUT_PRICE_PER_MILLION = 0.07
OUTPUT_PRICE_PER_MILLION = 0.34

DIRECTION_LABELS = ("very_negative", "negative", "neutral", "positive", "very_positive")
LEVEL_LABELS = ("none", "low", "moderate", "high", "very_high")
HORIZON_LABELS = (
    "immediate_1_session",
    "short_2_5_sessions",
    "medium_6_20_sessions",
    "longer_term",
    "unclear",
)
SOURCE_BLOCKS = ("backward_2024_2025", "opened_2025_2026")
POLARITY_LABELS = ("negative", "neutral", "positive")

JOINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "direction_severity": {"type": "string", "enum": list(DIRECTION_LABELS)},
        "materiality": {"type": "string", "enum": list(LEVEL_LABELS)},
        "novelty": {"type": "string", "enum": list(LEVEL_LABELS)},
        "valuation_horizon": {"type": "string", "enum": list(HORIZON_LABELS)},
        "target_specific": {"type": "boolean"},
        "evidence_sufficient": {"type": "boolean"},
    },
    "required": [
        "direction_severity",
        "materiality",
        "novelty",
        "valuation_horizon",
        "target_specific",
        "evidence_sufficient",
    ],
    "additionalProperties": False,
}


class MaterialityPilotError(RuntimeError):
    """Raised when a materiality-pilot artifact cannot be trusted."""


@dataclass(frozen=True)
class MaterialityInputs:
    joined_scores: JoinedLsegInputs
    backward_headlines: Path
    recent_headlines: Path
    recent_articles: Path
    company_config: Path


@dataclass(frozen=True)
class MaterialityPaths:
    root: Path
    sample_events: Path
    sample_manifest: Path
    body_cache: Path
    body_attempts: Path
    body_manifest: Path
    score_attempts: Path
    score_manifest: Path
    model_scores: Path
    audit_dir: Path
    audit_items: Path
    audit_key: Path
    coder_a: Path
    coder_b: Path
    codebook: Path
    audit_manifest: Path
    reliability_report: Path


@dataclass(frozen=True)
class MaterialityPrompt:
    prompt_id: str
    system_prompt: str
    user_template: str
    prompt_hash: str

    def render(self, event: Mapping[str, Any], body: str) -> str:
        current = "\n\n".join(
            value.strip()
            for value in (str(event.get("headline") or ""), body[:MAXIMUM_BODY_CHARS])
            if value.strip()
        )
        return self.user_template.format(
            target_company=str(event["target_company_name"]),
            symbol=str(event["symbol"]),
            current_news=current,
            prior_context=str(event["prior_context"]),
        )


def materiality_paths(root: str | Path = "final_experiments/outputs/68_lseg_materiality_measurement_pilot") -> MaterialityPaths:
    base = Path(root)
    audit = base / "human_audit"
    return MaterialityPaths(
        root=base,
        sample_events=base / "sample_events.jsonl",
        sample_manifest=base / "sample_manifest.json",
        body_cache=base / "body_cache",
        body_attempts=base / "body_attempts",
        body_manifest=base / "body_manifest.json",
        score_attempts=base / "model_score_attempts.jsonl",
        score_manifest=base / "model_score_manifest.json",
        model_scores=base / "model_scores.jsonl",
        audit_dir=audit,
        audit_items=audit / "audit_items.csv",
        audit_key=audit / "audit_key.csv",
        coder_a=audit / "coder_a_labels.csv",
        coder_b=audit / "coder_b_labels.csv",
        codebook=audit / "codebook.md",
        audit_manifest=audit / "audit_manifest.json",
        reliability_report=audit / "reliability_report.json",
    )


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _stable_rank(namespace: str, value: str, *, seed: int = RANDOM_SEED) -> str:
    return sha256_text(f"{seed}|{namespace}|{value}")


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def load_materiality_prompt(path: str | Path = DEFAULT_PROMPT_PATH) -> MaterialityPrompt:
    prompt_path = Path(path)
    with prompt_path.open("rb") as handle:
        payload = tomllib.load(handle)
    raw = payload.get("prompt")
    if not isinstance(raw, dict):
        raise MaterialityPilotError(f"prompt file has no [prompt] table: {prompt_path}")
    prompt_id = str(raw.get("id") or "")
    system = str(raw.get("system_prompt") or "").strip()
    user = str(raw.get("user_template") or "").strip()
    if prompt_id != PROMPT_ID or str(raw.get("output_mode")) != "structured_json":
        raise MaterialityPilotError("materiality prompt identity or output mode changed")
    required = (
        "Do not use outside knowledge",
        "predict a stock return",
        "target_specific=false requires materiality=none",
        "materiality=none requires direction_severity=neutral",
        "Earlier headlines may affect only novelty",
        "valuation_horizon",
        "evidence_sufficient",
    )
    if any(text not in system for text in required):
        raise MaterialityPilotError("materiality prompt is missing a frozen leakage or consistency rule")
    for placeholder in ("{target_company}", "{symbol}", "{current_news}", "{prior_context}"):
        if placeholder not in user:
            raise MaterialityPilotError(f"materiality prompt omits {placeholder}")
    for label in (*DIRECTION_LABELS, *LEVEL_LABELS, *HORIZON_LABELS):
        if label not in system:
            raise MaterialityPilotError(f"materiality prompt omits enum label {label}")
    return MaterialityPrompt(
        prompt_id=prompt_id,
        system_prompt=system,
        user_template=user,
        prompt_hash=prompt_hash(prompt_id, system, user, "structured_json"),
    )


def parse_materiality_labels(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != set(JOINT_SCHEMA["required"]):
        raise ValueError("model output does not contain exactly the frozen fields")
    direction = str(payload["direction_severity"])
    materiality = str(payload["materiality"])
    novelty = str(payload["novelty"])
    horizon = str(payload["valuation_horizon"])
    target_specific = payload["target_specific"]
    evidence_sufficient = payload["evidence_sufficient"]
    if direction not in DIRECTION_LABELS or materiality not in LEVEL_LABELS or novelty not in LEVEL_LABELS:
        raise ValueError("model output contains an invalid ordinal label")
    if horizon not in HORIZON_LABELS:
        raise ValueError("model output contains an invalid valuation horizon")
    if not isinstance(target_specific, bool) or not isinstance(evidence_sufficient, bool):
        raise ValueError("model output evidence flags must be booleans")
    if not target_specific and (materiality != "none" or direction != "neutral"):
        raise ValueError("target_specific=false requires materiality=none and direction_severity=neutral")
    if materiality == "none" and direction != "neutral":
        raise ValueError("materiality=none requires direction_severity=neutral")
    if direction in {"very_negative", "very_positive"} and materiality not in {"high", "very_high"}:
        raise ValueError("extreme direction requires high or very_high materiality")
    return {
        "direction_severity": direction,
        "materiality": materiality,
        "novelty": novelty,
        "valuation_horizon": horizon,
        "target_specific": target_specific,
        "evidence_sufficient": evidence_sufficient,
    }


def _company_names(config_path: Path) -> dict[str, str]:
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    companies = payload.get("companies") or []
    names = {str(row["symbol"]): str(row["name"]) for row in companies}
    if len(names) != 33:
        raise MaterialityPilotError("company configuration must contain exactly 33 companies")
    return names


def _matched_original_symbols(value: Any, symbols: set[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = value.split("|")
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = ()
    return tuple(sorted({str(item).strip() for item in raw if str(item).strip() in symbols}))


def _map_entry_sessions(timestamps: Sequence[Any]) -> pd.DatetimeIndex:
    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover
        raise MaterialityPilotError("exchange_calendars is required for return-blind timing") from exc
    values = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True, format="mixed"))
    if values.empty:
        return pd.DatetimeIndex([])
    calendar = xcals.get_calendar("XNYS")
    start = (values.min() - pd.Timedelta(days=10)).date().isoformat()
    end = (values.max() + pd.Timedelta(days=10)).date().isoformat()
    schedule = calendar.schedule.loc[start:end]
    opens = pd.DatetimeIndex(schedule["open"].to_numpy())
    opens = opens.tz_localize("UTC") if opens.tz is None else opens.tz_convert("UTC")
    cutoffs = values + pd.Timedelta(minutes=INFORMATION_BUFFER_MINUTES)
    positions = opens.searchsorted(cutoffs, side="left")
    if bool(np.any(positions >= len(opens))):
        raise MaterialityPilotError("calendar does not extend far enough to map every selected headline")
    sessions = pd.DatetimeIndex(schedule.index.take(positions)).tz_localize(None)
    return sessions


def _largest_remainder(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total < 0 or not weights or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("largest-remainder allocation requires non-negative total and unit-sum weights")
    raw = {key: total * float(weight) for key, weight in weights.items()}
    allocation = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(allocation.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation


def select_stratified_sample(candidates: pd.DataFrame, *, size: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Select an exact stock-balanced, block/polarity-stratified return-blind sample."""

    required = {"headline_sha256", "symbol", "entry_session", "source_block", "label_finbert"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate frame lacks sampling fields: {sorted(missing)}")
    frame = candidates.copy()
    if frame.duplicated(["symbol", "entry_session"]).any():
        raise ValueError("candidate frame must already contain one event per symbol-session")
    symbols = sorted(frame["symbol"].astype(str).unique())
    if len(symbols) != 33 or size < len(symbols):
        raise ValueError("sample requires the frozen 33-company universe")
    base, extras = divmod(size, len(symbols))
    extra_symbols = set(sorted(symbols, key=lambda symbol: _stable_rank("extra-symbol", symbol, seed=seed))[:extras])
    allocation_by_symbol = {symbol: base + int(symbol in extra_symbols) for symbol in symbols}
    available_by_symbol = frame["symbol"].astype(str).value_counts().to_dict()
    deficit = 0
    for symbol in symbols:
        supported = min(allocation_by_symbol[symbol], int(available_by_symbol.get(symbol, 0)))
        deficit += allocation_by_symbol[symbol] - supported
        allocation_by_symbol[symbol] = supported
    while deficit:
        eligible = [
            symbol
            for symbol in symbols
            if allocation_by_symbol[symbol] < int(available_by_symbol.get(symbol, 0))
        ]
        if not eligible:
            raise MaterialityPilotError(f"only {size - deficit} eligible symbol-sessions exist for sample size {size}")
        symbol = min(
            eligible,
            key=lambda value: (
                allocation_by_symbol[value],
                _stable_rank("shortfall-redistribution", value, seed=seed),
            ),
        )
        allocation_by_symbol[symbol] += 1
        deficit -= 1
    source_polarity_weights = {
        "backward_2024_2025|negative": 0.28,
        "backward_2024_2025|neutral": 0.14,
        "backward_2024_2025|positive": 0.28,
        "opened_2025_2026|negative": 0.12,
        "opened_2025_2026|neutral": 0.06,
        "opened_2025_2026|positive": 0.12,
    }
    selected_rows: list[pd.DataFrame] = []
    for symbol in symbols:
        quota = allocation_by_symbol[symbol]
        pool = frame.loc[frame["symbol"].astype(str).eq(symbol)].copy()
        pool["sample_stratum"] = pool["source_block"].astype(str) + "|" + pool["label_finbert"].astype(str)
        allocation = _largest_remainder(quota, source_polarity_weights)
        chosen_indices: list[int] = []
        for stratum, target in allocation.items():
            stratum_pool = pool.loc[pool["sample_stratum"].eq(stratum)]
            ranked = sorted(
                stratum_pool.index,
                key=lambda index: _stable_rank(
                    f"sample|{symbol}|{stratum}", str(pool.at[index, "headline_sha256"]), seed=seed
                ),
            )
            chosen_indices.extend(ranked[:target])
        chosen_set = set(chosen_indices)
        if len(chosen_set) < quota:
            fallback = [
                index
                for index in sorted(
                    pool.index,
                    key=lambda index: _stable_rank(
                        f"sample|{symbol}|fallback", str(pool.at[index, "headline_sha256"]), seed=seed
                    ),
                )
                if index not in chosen_set
            ]
            chosen_indices.extend(fallback[: quota - len(chosen_set)])
        chosen = pool.loc[chosen_indices[:quota]].copy()
        if len(chosen) != quota or chosen["headline_sha256"].duplicated().any():
            raise MaterialityPilotError(f"could not fill a unique allocation for {symbol}")
        selected_rows.append(chosen)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected = selected.sort_values(
        "headline_sha256",
        key=lambda values: values.map(lambda value: _stable_rank("sample-order", str(value), seed=seed)),
        kind="mergesort",
        ignore_index=True,
    )
    if len(selected) != size or selected["headline_sha256"].duplicated().any():
        raise MaterialityPilotError("final materiality sample is not exact and unique")
    selected["sample_order"] = np.arange(1, len(selected) + 1)
    return selected.drop(columns=["sample_stratum"], errors="ignore")


def _extract_reuters_metadata(
    corpus_path: Path,
    score_frame: pd.DataFrame,
    first_release_hashes: set[str],
    company_names: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    score_lookup = score_frame.set_index("headline_sha256").to_dict(orient="index")
    expected = set(score_lookup) & first_release_hashes
    chosen: dict[str, dict[str, Any]] = {}
    rows_read = 0
    reuters_rows = 0
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterialityPilotError(f"invalid corpus JSON at {corpus_path}:{line_number}") from exc
            rows_read += 1
            if str(raw.get("source_code") or "") != REUTERS_SOURCE_CODE:
                continue
            reuters_rows += 1
            digest = headline_norm_sha256(str(raw.get("headline") or ""))
            score = score_lookup.get(digest)
            if score is None or digest not in expected:
                continue
            symbols = _matched_original_symbols(raw.get("matched_symbols"), set(company_names))
            if len(symbols) != 1 or symbols[0] != str(score["matched_symbols"]):
                continue
            timestamp = _timestamp(raw.get("version_created") or raw.get("first_created"))
            record = {
                "headline_sha256": digest,
                "headline": str(raw.get("headline") or "").strip(),
                "story_id": str(raw.get("story_id") or "").strip(),
                "available_at_utc": timestamp.isoformat(),
                "symbol": symbols[0],
                "target_company_name": company_names[symbols[0]],
                "source_code": REUTERS_SOURCE_CODE,
                **score,
            }
            prior = chosen.get(digest)
            if prior is None or (record["available_at_utc"], record["story_id"]) < (
                prior["available_at_utc"],
                prior["story_id"],
            ):
                chosen[digest] = record
    frame = pd.DataFrame(chosen.values())
    return frame, {
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "rows_read": rows_read,
        "reuters_rows": reuters_rows,
        "eligible_score_hashes": len(expected),
        "reuters_first_release_hashes": len(frame),
    }


def _build_prior_context(candidate_events: pd.DataFrame, selected_hashes: set[str]) -> dict[str, str]:
    contexts: dict[str, str] = {}
    window = pd.Timedelta(days=PRIOR_WINDOW_DAYS)
    for _, group in candidate_events.groupby("symbol", sort=True):
        rows = group.sort_values(["available_at_utc", "headline_sha256"], kind="mergesort").to_dict("records")
        prior: deque[dict[str, Any]] = deque()
        index = 0
        while index < len(rows):
            current_time = _timestamp(rows[index]["available_at_utc"])
            same_time: list[dict[str, Any]] = []
            while index < len(rows) and _timestamp(rows[index]["available_at_utc"]) == current_time:
                same_time.append(rows[index])
                index += 1
            while prior and current_time - _timestamp(prior[0]["available_at_utc"]) > window:
                prior.popleft()
            for current in same_time:
                digest = str(current["headline_sha256"])
                if digest not in selected_hashes:
                    continue
                earlier = list(prior)[-MAXIMUM_PRIOR_HEADLINES:]
                contexts[digest] = (
                    "\n".join(
                        f"[{_timestamp(row['available_at_utc']).date().isoformat()}] {str(row['headline']).strip()}"
                        for row in earlier
                    )
                    if earlier
                    else "(No strictly earlier same-company Reuters headline in the frozen 30-day context window.)"
                )
            prior.extend(same_time)
    missing = selected_hashes - set(contexts)
    if missing:
        raise MaterialityPilotError(f"prior context is missing for {len(missing)} selected events")
    return contexts


def prepare_materiality_sample(inputs: MaterialityInputs, *, output_root: str | Path) -> dict[str, Any]:
    """Freeze the 2,000-event sample without opening any price or return file."""

    paths = materiality_paths(output_root)
    if paths.sample_manifest.is_file():
        manifest = read_json(paths.sample_manifest)
        expected = (manifest.get("outputs") or {}).get(paths.sample_events.name, {}).get("sha256")
        if manifest.get("status") != "completed" or not paths.sample_events.is_file() or sha256_file(paths.sample_events) != expected:
            raise MaterialityPilotError("existing materiality sample is incomplete or changed")
        return manifest

    names = _company_names(inputs.company_config)
    populations, joined_audit = load_joined_same33_scores(inputs.joined_scores, original_symbols=names)
    block_specs = (
        ("backward", "backward_2024_2025", inputs.backward_headlines),
        ("recent", "opened_2025_2026", inputs.recent_headlines),
    )
    metadata_frames: list[pd.DataFrame] = []
    story_audits: dict[str, Any] = {}
    metadata_audits: dict[str, Any] = {}
    filter_counts: dict[str, Any] = {}
    for frame_key, block_name, corpus in block_specs:
        frame = populations[frame_key].copy()
        retained, story_audit = load_first_release_hashes(
            corpus,
            expected_hashes=set(frame["headline_sha256"].astype(str)),
        )
        eligible = frame.loc[
            frame["explicit_target"].astype(bool)
            & ~frame["market_price_technical"].astype(bool)
            & frame["matched_symbols"].astype(str).str.count(r"\|").eq(0)
            & frame["headline_sha256"].isin(retained)
        ].copy()
        eligible["source_block"] = block_name
        metadata, metadata_audit = _extract_reuters_metadata(corpus, eligible, retained, names)
        metadata_frames.append(metadata)
        story_audits[block_name] = story_audit
        metadata_audits[block_name] = metadata_audit
        filter_counts[block_name] = {
            "paired_scores": len(frame),
            "explicit_nontechnical_single_symbol_first_release": len(eligible),
            "reuters_candidates": len(metadata),
        }

    candidates = pd.concat(metadata_frames, ignore_index=True)
    candidates["available_at_utc"] = pd.to_datetime(candidates["available_at_utc"], utc=True, format="mixed")
    candidates = candidates.sort_values(
        ["available_at_utc", "source_block", "headline_sha256"], kind="mergesort"
    ).drop_duplicates("headline_sha256", keep="first")
    candidates["entry_session"] = _map_entry_sessions(candidates["available_at_utc"]).to_numpy()
    candidates["session_rank"] = candidates["headline_sha256"].map(
        lambda value: _stable_rank("symbol-session", str(value))
    )
    candidates = candidates.sort_values(
        ["symbol", "entry_session", "session_rank"], kind="mergesort"
    ).drop_duplicates(["symbol", "entry_session"], keep="first")
    candidates = candidates.drop(columns=["session_rank"]).reset_index(drop=True)
    if set(candidates["label_finbert"].astype(str)) - set(POLARITY_LABELS):
        raise MaterialityPilotError("FinBERT sampling labels are not the frozen three-class labels")

    selected = select_stratified_sample(candidates)
    selected_hashes = set(selected["headline_sha256"].astype(str))
    contexts = _build_prior_context(pd.concat(metadata_frames, ignore_index=True), selected_hashes)
    records: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        digest = str(row["headline_sha256"])
        available = _timestamp(row["available_at_utc"]).isoformat()
        entry = pd.Timestamp(row["entry_session"]).date().isoformat()
        audit_id = f"mat-{digest[:16]}"
        record = {
            "audit_id": audit_id,
            "sample_order": int(row["sample_order"]),
            "headline_sha256": digest,
            "headline": str(row["headline"]),
            "story_id": str(row["story_id"]),
            "symbol": str(row["symbol"]),
            "target_company_name": str(row["target_company_name"]),
            "available_at_utc": available,
            "entry_session": entry,
            "source_block": str(row["source_block"]),
            "source_code": REUTERS_SOURCE_CODE,
            "label_finbert": str(row["label_finbert"]),
            "score_finbert": float(row["score_finbert"]),
            "label_gemma": str(row["label_gemma"]),
            "score_gemma": float(row["score_gemma"]),
            "prior_context": contexts[digest],
        }
        record["event_input_sha256"] = sha256_text(
            canonical_json(
                {
                    key: record[key]
                    for key in (
                        "headline_sha256",
                        "story_id",
                        "symbol",
                        "available_at_utc",
                        "entry_session",
                        "source_block",
                        "prior_context",
                    )
                }
            )
        )
        records.append(record)
    paths.root.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(paths.sample_events, records)
    sample_hash = sha256_text("\n".join(record["headline_sha256"] for record in records))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "estimand_stage": "return-blind measurement sample only",
        "selection": {
            "population": "joined same-33 LSEG score blocks, kept separate by source block until deduplication",
            "source": "Reuters NS:RTRS only",
            "filters": "explicit target; non-market-price/technical; one original-33 symbol; first family release",
            "event_spacing": "one deterministic headline per symbol and first executable XNYS session",
            "entry_timing": f"first XNYS open at least {INFORMATION_BUFFER_MINUTES} minutes after availability",
            "sample_size": SAMPLE_SIZE,
            "seed": RANDOM_SEED,
            "allocation": (
                "33-company balance; 70/30 backward/recent and 40/20/40 FinBERT "
                "negative/neutral/positive targets with deterministic fallback"
            ),
            "sample_population_sha256": sample_hash,
            "prices_or_returns_loaded": False,
        },
        "counts": {
            "candidate_unique_symbol_sessions": len(candidates),
            "selected": len(records),
            "symbols": len({record["symbol"] for record in records}),
            "source_blocks": dict(Counter(record["source_block"] for record in records)),
            "finbert_strata": dict(Counter(record["label_finbert"] for record in records)),
        },
        "inputs": {
            "joined_score_audit": joined_audit,
            "story_family_audits": story_audits,
            "metadata_audits": metadata_audits,
            "company_config": {"path": str(inputs.company_config), "sha256": sha256_file(inputs.company_config)},
            "recent_articles": {"path": str(inputs.recent_articles), "sha256": sha256_file(inputs.recent_articles)},
        },
        "filter_counts": filter_counts,
        "outputs": {
            paths.sample_events.name: {
                "path": str(paths.sample_events),
                "sha256": sha256_file(paths.sample_events),
                "rows": len(records),
                "licensed_text": True,
                "gitignored": True,
            }
        },
    }
    atomic_write_json(paths.sample_manifest, manifest)
    return manifest


def _body_cache_path(paths: MaterialityPaths, audit_id: str) -> Path:
    return paths.body_cache / f"{audit_id}.json"


def _write_body_success(
    paths: MaterialityPaths,
    event: Mapping[str, Any],
    *,
    clean_text: str,
    provenance: str,
    body_format: str | None,
) -> None:
    cleaned = clean_text.strip()
    payload = {
        "audit_id": str(event["audit_id"]),
        "headline_sha256": str(event["headline_sha256"]),
        "story_id_sha256": sha256_text(str(event["story_id"])),
        "clean_text": cleaned,
        "clean_text_sha256": sha256_text(cleaned),
        "cleaned_chars": len(cleaned),
        "body_format": body_format,
        "provenance": provenance,
        "status": "success",
        "licensed_text": True,
    }
    target = _body_cache_path(paths, str(event["audit_id"]))
    if target.is_file():
        existing = read_json(target)
        if existing.get("clean_text_sha256") != payload["clean_text_sha256"]:
            raise MaterialityPilotError(f"refusing to replace a different body cache: {target}")
        return
    atomic_write_json(target, payload)


def hydrate_existing_recent_bodies(
    paths: MaterialityPaths,
    recent_articles_path: str | Path,
) -> int:
    events = read_jsonl(paths.sample_events)
    by_story = {str(row["story_id"]): row for row in events if not _body_cache_path(paths, str(row["audit_id"])).is_file()}
    if not by_story:
        return 0
    hydrated = 0
    with Path(recent_articles_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                article = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterialityPilotError(f"invalid article JSON at line {line_number}") from exc
            event = by_story.get(str(article.get("story_id") or ""))
            text = str(article.get("clean_text") or "").strip()
            if event is None or not text or str(article.get("text_quality")) != QUALITY_OK:
                continue
            _write_body_success(
                paths,
                event,
                clean_text=text,
                provenance="existing_canonical_recent_article",
                body_format=str(article.get("body_format") or "") or None,
            )
            hydrated += 1
    return hydrated


async def fetch_materiality_bodies(
    inputs: MaterialityInputs,
    *,
    output_root: str | Path,
    max_new_requests: int = SAMPLE_SIZE,
    concurrency: int = 4,
    requests_per_second: float = 3.0,
    retry_terminal_failures: bool = False,
) -> dict[str, Any]:
    """Resume selective story retrieval, capped at the frozen 2,000-request budget."""

    if max_new_requests < 0 or max_new_requests > SAMPLE_SIZE:
        raise ValueError(f"max_new_requests must be between 0 and {SAMPLE_SIZE}")
    paths = materiality_paths(output_root)
    if not paths.sample_events.is_file():
        raise MaterialityPilotError("prepare the frozen sample before retrieving bodies")
    hydrated = hydrate_existing_recent_bodies(paths, inputs.recent_articles)
    events = read_jsonl(paths.sample_events)
    terminal_attempt_ids: set[str] = set()
    if not retry_terminal_failures and paths.body_attempts.is_dir():
        for attempt_path in paths.body_attempts.glob("*/*.json"):
            attempt = read_json(attempt_path)
            # A successful LSEG response with unusable/empty content is an
            # observed archival-availability result. Session/sandbox failures
            # did not reach LSEG and remain safely resumable.
            terminal_not_found = attempt.get("status") == "failed" and "404" in str(attempt.get("error") or "")
            if attempt.get("status") in {"success", "missing", "story_unavailable"} or terminal_not_found:
                terminal_attempt_ids.add(str(attempt.get("audit_id") or ""))
    pending = [
        row
        for row in events
        if not _body_cache_path(paths, str(row["audit_id"])).is_file()
        and str(row["audit_id"]) not in terminal_attempt_ids
    ]
    targets = pending[:max_new_requests]
    paths.body_attempts.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    client = LsegNewsClient()
    client.configure_request_pacing(requests_per_second, max_requests=max_new_requests or None)
    calls = 0

    async def fetch_one(event: Mapping[str, Any]) -> None:
        nonlocal calls
        async with semaphore:
            started = time.monotonic()
            try:
                response = await client.fetch_story(str(event["story_id"]))
                calls += 1
                body = response.body or ""
                cleaned = clean_news_html(body, min_chars=100)
                payload = {
                    "audit_id": str(event["audit_id"]),
                    "story_id_sha256": sha256_text(str(event["story_id"])),
                    "status": response.status,
                    "body": body,
                    "body_format": response.body_format,
                    "cleaning": asdict(cleaned),
                    "error": response.error,
                    "latency_ms": (time.monotonic() - started) * 1_000,
                    "fetched_at_utc": datetime.now(UTC).isoformat(),
                    "licensed_text": True,
                }
                if response.status == "success" and cleaned.quality == QUALITY_OK:
                    _write_body_success(
                        paths,
                        event,
                        clean_text=cleaned.text,
                        provenance="selective_lseg_story_fetch",
                        body_format=response.body_format,
                    )
            except Exception as exc:  # preserve a resumable failure record
                calls += 1
                payload = {
                    "audit_id": str(event["audit_id"]),
                    "story_id_sha256": sha256_text(str(event["story_id"])),
                    "status": "failed",
                    "body": None,
                    "error": str(exc),
                    "latency_ms": (time.monotonic() - started) * 1_000,
                    "fetched_at_utc": datetime.now(UTC).isoformat(),
                    "licensed_text": True,
                }
            attempt_dir = paths.body_attempts / str(event["audit_id"])
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_path = attempt_dir / f"{len(list(attempt_dir.glob('*.json'))) + 1:04d}.json"
            atomic_write_json(attempt_path, payload)

    async with client:
        await asyncio.gather(*(fetch_one(row) for row in targets))
    successful = sum(_body_cache_path(paths, str(row["audit_id"])).is_file() for row in events)
    available_ids = [
        str(row["audit_id"])
        for row in events
        if _body_cache_path(paths, str(row["audit_id"])).is_file()
    ]
    resolved = successful + sum(
        str(row["audit_id"]) in terminal_attempt_ids
        and not _body_cache_path(paths, str(row["audit_id"])).is_file()
        for row in events
    )
    manifest = {
        "schema_version": 1,
        "status": "availability_finalized" if resolved == len(events) else "incomplete",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "sample_manifest_sha256": sha256_file(paths.sample_manifest),
        "counts": {
            "selected": len(events),
            "hydrated_from_existing_recent_articles_this_run": hydrated,
            "lseg_requests_this_run": calls,
            "successful_bodies": successful,
            "missing_bodies": len(events) - successful,
            "terminal_unavailable_bodies": resolved - successful,
            "unresolved_bodies": len(events) - resolved,
            "body_available_share": successful / len(events),
        },
        "request_contract": {
            "selected_story_only": True,
            "max_new_requests_this_run": max_new_requests,
            "concurrency": concurrency,
            "requests_per_second": requests_per_second,
            "automatic_retries": 0,
            "retry_terminal_failures": retry_terminal_failures,
            "global_pilot_lseg_request_cap": SAMPLE_SIZE,
        },
        "outputs": {
            "body_cache": {
                "path": str(paths.body_cache),
                "successful_files": successful,
                "body_available_population_sha256": sha256_text("\n".join(available_ids)),
                "licensed_text": True,
                "gitignored": True,
            }
        },
    }
    atomic_write_json(paths.body_manifest, manifest)
    return manifest


def openrouter_request_contract() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "max_tokens": 160,
        "reasoning": {"effort": "none", "exclude": True},
        "provider": {
            "only": [PROVIDER_SLUG],
            "quantizations": ["fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "max_price": {
                "prompt": INPUT_PRICE_PER_MILLION,
                "completion": OUTPUT_PRICE_PER_MILLION,
            },
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "lseg_materiality_annotation", "strict": True, "schema": JOINT_SCHEMA},
        },
    }


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
    return float(min(2**attempt, 8))


async def _score_event(
    client: httpx.AsyncClient,
    api_key: str,
    event: Mapping[str, Any],
    body: str,
    prompt: MaterialityPrompt,
    config_hash: str,
    *,
    retries: int,
) -> dict[str, Any]:
    request = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.render(event, body)},
        ],
        **openrouter_request_contract(),
        "stream": False,
    }
    response: httpx.Response | None = None
    error: Exception | None = None
    started = time.monotonic()
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Sentiment Dissertation LSEG materiality pilot",
                },
                json=request,
            )
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == retries:
                break
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            error = exc
            if attempt == retries:
                break
        await asyncio.sleep(_retry_delay(response, attempt))
    base = {
        "audit_id": str(event["audit_id"]),
        "headline_sha256": str(event["headline_sha256"]),
        "config_hash": config_hash,
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "quantization": "fp8",
        "prompt_id": prompt.prompt_id,
        "prompt_hash": prompt.prompt_hash,
        "input_sha256": sha256_text(prompt.render(event, body)),
        "attempt_count": attempts,
        "latency_ms": (time.monotonic() - started) * 1_000,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if response is None:
        return {**base, "status": "transport_error", "error": str(error)}
    if response.status_code >= 400:
        return {
            **base,
            "status": "api_error",
            "http_status": response.status_code,
            "error": response.text[:1_000],
        }
    try:
        raw = response.json()
        choice = (raw.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        parsed = json.loads(content)
        labels = parse_materiality_labels(parsed)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {**base, "status": "invalid_output", "error": str(exc)}
    returned_provider = str(raw.get("provider") or "")
    returned_model = str(raw.get("model") or "")
    route_errors: list[str] = []
    if returned_provider and returned_provider.casefold() not in {PROVIDER_NAME.casefold(), PROVIDER_SLUG.casefold()}:
        route_errors.append(f"unexpected provider {returned_provider!r}")
    if returned_model and returned_model != MODEL_ID:
        route_errors.append(f"unexpected model {returned_model!r}")
    usage = raw.get("usage") or {}
    return {
        **base,
        **labels,
        "status": "success" if not route_errors else "route_mismatch",
        "generation_id": raw.get("id"),
        "returned_model": returned_model or None,
        "returned_provider": returned_provider or None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reported_cost_usd": usage.get("cost"),
        "finish_reason": choice.get("finish_reason"),
        "error": "; ".join(route_errors) or None,
    }


def _load_score_attempts(path: Path, config_hash: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    success: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return attempts, success
    for line_number, row in enumerate(read_jsonl(path), start=1):
        if row.get("config_hash") != config_hash:
            raise MaterialityPilotError(f"score checkpoint config mismatch at line {line_number}")
        audit_id = str(row.get("audit_id") or "")
        if not audit_id:
            raise MaterialityPilotError(f"score checkpoint lacks audit ID at line {line_number}")
        attempts.append(row)
        if row.get("status") == "success":
            if audit_id in success:
                raise MaterialityPilotError(f"duplicate successful materiality score for {audit_id}")
            success[audit_id] = row
    return attempts, success


async def score_materiality_sample(
    *,
    output_root: str | Path,
    concurrency: int = 25,
    retries: int = 3,
    retry_failures: bool = False,
    max_new_calls: int | None = None,
    confirm_authorized_transfer: bool = False,
) -> dict[str, Any]:
    """Score the frozen sample through the explicitly authorised hosted route."""

    paths = materiality_paths(output_root)
    if not confirm_authorized_transfer:
        raise MaterialityPilotError(
            "hosted scoring requires --confirm-authorized-transfer after the researcher explicitly approves "
            "sending the selected Reuters headline/body payload to OpenRouter routed only to DeepInfra"
        )
    if not paths.sample_manifest.is_file() or not paths.body_manifest.is_file():
        raise MaterialityPilotError("sample and body manifests must exist before hosted scoring")
    body_manifest = read_json(paths.body_manifest)
    if body_manifest.get("status") != "availability_finalized":
        raise MaterialityPilotError("hosted scoring remains blocked until selected-story body availability is finalized")
    all_events = read_jsonl(paths.sample_events)
    if len(all_events) != SAMPLE_SIZE:
        raise MaterialityPilotError("materiality sample size changed")
    events = [
        row
        for row in all_events
        if _body_cache_path(paths, str(row["audit_id"])).is_file()
    ]
    expected_scores = int((body_manifest.get("counts") or {}).get("successful_bodies") or 0)
    if len(events) != expected_scores or expected_scores < PRIMARY_AUDIT_SIZE:
        raise MaterialityPilotError("body-available measurement population does not reconcile")
    prompt = load_materiality_prompt()
    identity = {
        "schema_version": 1,
        "sample_manifest_sha256": sha256_file(paths.sample_manifest),
        "body_manifest_sha256": sha256_file(paths.body_manifest),
        "body_available_population_sha256": sha256_text(
            "\n".join(str(row["audit_id"]) for row in events)
        ),
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "quantization": "fp8",
        "prompt_id": prompt.prompt_id,
        "prompt_hash": prompt.prompt_hash,
        "prompt_file_sha256": sha256_file(DEFAULT_PROMPT_PATH),
        "schema_sha256": sha256_text(canonical_json(JOINT_SCHEMA)),
        "maximum_body_chars": MAXIMUM_BODY_CHARS,
        "request": openrouter_request_contract(),
        "external_processing_authority": "invocation asserts the researcher's exact selected-Reuters-text/OpenRouter/DeepInfra approval",
    }
    config_hash = sha256_text(canonical_json(identity))
    attempts, successful = _load_score_attempts(paths.score_attempts, config_hash)
    pending = [row for row in events if str(row["audit_id"]) not in successful]
    if not retry_failures and attempts:
        attempted_ids = {str(row["audit_id"]) for row in attempts}
        pending = [row for row in pending if str(row["audit_id"]) not in attempted_ids]
    if max_new_calls is not None:
        if max_new_calls < 1:
            raise ValueError("max_new_calls must be positive when supplied")
        pending = pending[:max_new_calls]
    load_env_file()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise MaterialityPilotError("OPENROUTER_API_KEY is required for hosted materiality scoring")
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    paths.root.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120.0) as client:
        async def run_one(event: Mapping[str, Any]) -> None:
            body_payload = read_json(_body_cache_path(paths, str(event["audit_id"])))
            body = str(body_payload.get("clean_text") or "")
            if not body:
                raise MaterialityPilotError(f"body cache is empty for {event['audit_id']}")
            async with semaphore:
                result = await _score_event(client, api_key, event, body, prompt, config_hash, retries=retries)
            encoded = json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n"
            async with write_lock:
                with paths.score_attempts.open("a", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())

        await asyncio.gather(*(run_one(row) for row in pending))
    all_attempts, successful = _load_score_attempts(paths.score_attempts, config_hash)
    ordered = [successful[str(row["audit_id"])] for row in events if str(row["audit_id"]) in successful]
    if len(ordered) == expected_scores:
        if paths.model_scores.is_file() and read_jsonl(paths.model_scores) != ordered:
            raise MaterialityPilotError("refusing to replace different completed model scores")
        atomic_write_jsonl(paths.model_scores, ordered)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in all_attempts)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in all_attempts)
    reported_costs = [float(row["reported_cost_usd"]) for row in all_attempts if row.get("reported_cost_usd") is not None]
    status_counts = Counter(str(row.get("status")) for row in all_attempts)
    manifest = {
        "schema_version": 1,
        "status": "completed" if len(ordered) == expected_scores else "incomplete",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "identity": identity,
        "config_hash": config_hash,
        "counts": {
            "parent_sample_selected": SAMPLE_SIZE,
            "body_available_selected": expected_scores,
            "attempt_records": len(all_attempts),
            "attempted_this_run": len(pending),
            "successful_unique": len(ordered),
            "missing_success": expected_scores - len(ordered),
            "status_counts": dict(status_counts),
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reported_cost_usd": sum(reported_costs) if reported_costs else None,
            "listed_price_estimate_usd": (
                prompt_tokens * INPUT_PRICE_PER_MILLION + completion_tokens * OUTPUT_PRICE_PER_MILLION
            )
            / 1_000_000,
        },
        "outputs": {
            "attempts": {"path": str(paths.score_attempts), "sha256": sha256_file(paths.score_attempts)},
            "scores": {
                "path": str(paths.model_scores),
                "sha256": sha256_file(paths.model_scores) if paths.model_scores.is_file() else None,
            },
        },
    }
    atomic_write_json(paths.score_manifest, manifest)
    return manifest


def _audit_selection(scores: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = [dict(row) for row in scores]
    tail_pool = [row for row in rows if str(row["materiality"]) in {"high", "very_high"}]
    tail_n = min(TAIL_AUDIT_SIZE, len(tail_pool))
    tail = sorted(tail_pool, key=lambda row: _stable_rank("audit-tail", str(row["audit_id"])))[:tail_n]
    tail_ids = {str(row["audit_id"]) for row in tail}
    remaining = [row for row in rows if str(row["audit_id"]) not in tail_ids]
    representative_n = PRIMARY_AUDIT_SIZE - tail_n
    representative = sorted(
        remaining, key=lambda row: _stable_rank("audit-representative", str(row["audit_id"]))
    )[:representative_n]
    low_probability = representative_n / len(remaining)
    tail_probability = tail_n / len(tail_pool) if tail_pool else 0.0
    probabilities: dict[str, float] = {}
    for row in rows:
        audit_id = str(row["audit_id"])
        if row in tail_pool:
            probabilities[audit_id] = tail_probability + (1 - tail_probability) * low_probability
        else:
            probabilities[audit_id] = low_probability
    selected = [dict(row, sampling_arm="tail_enrichment") for row in tail]
    selected.extend(dict(row, sampling_arm="representative_remainder") for row in representative)
    selected.sort(key=lambda row: _stable_rank("audit-order", str(row["audit_id"])))
    if len(selected) != PRIMARY_AUDIT_SIZE:
        raise MaterialityPilotError("human audit sample is not exact")
    return selected, probabilities


def build_human_audit(*, output_root: str | Path) -> dict[str, Any]:
    paths = materiality_paths(output_root)
    if not paths.model_scores.is_file() or read_json(paths.score_manifest).get("status") != "completed":
        raise MaterialityPilotError("complete model scores are required before audit sampling")
    if paths.audit_manifest.is_file():
        manifest = read_json(paths.audit_manifest)
        for name, contract in (manifest.get("outputs") or {}).items():
            path = paths.audit_dir / name
            if not path.is_file() or sha256_file(path) != contract.get("sha256"):
                raise MaterialityPilotError(f"completed audit output is missing or changed: {path}")
        return manifest
    events = {str(row["audit_id"]): row for row in read_jsonl(paths.sample_events)}
    scores = read_jsonl(paths.model_scores)
    selected, inclusion = _audit_selection(scores)
    coder_b_ids: set[str] = set()
    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_level[str(row["materiality"])].append(row)
    per_level, extra = divmod(SECOND_CODER_SIZE, len(LEVEL_LABELS))
    for index, level in enumerate(LEVEL_LABELS):
        target = per_level + int(index < extra)
        ranked = sorted(by_level[level], key=lambda row: _stable_rank(f"coder-b|{level}", str(row["audit_id"])))
        coder_b_ids.update(str(row["audit_id"]) for row in ranked[:target])
    if len(coder_b_ids) < SECOND_CODER_SIZE:
        fallback = sorted(selected, key=lambda row: _stable_rank("coder-b-fallback", str(row["audit_id"])))
        for row in fallback:
            coder_b_ids.add(str(row["audit_id"]))
            if len(coder_b_ids) == SECOND_CODER_SIZE:
                break

    item_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for index, model in enumerate(selected, start=1):
        event = events[str(model["audit_id"])]
        body = read_json(_body_cache_path(paths, str(model["audit_id"])))
        item = {
            "audit_order": index,
            "audit_id": str(model["audit_id"]),
            "symbol": event["symbol"],
            "target_company_name": event["target_company_name"],
            "available_at_utc": event["available_at_utc"],
            "headline": event["headline"],
            "clean_text": body["clean_text"],
            "prior_context": event["prior_context"],
            "content_sha256": sha256_text(f"{event['headline']}\n\n{body['clean_text']}"),
            "context_sha256": sha256_text(str(event["prior_context"])),
        }
        item_rows.append(item)
        key_rows.append(
            {
                "audit_id": model["audit_id"],
                "sampling_arm": model["sampling_arm"],
                "inclusion_probability": inclusion[str(model["audit_id"])],
                "inverse_probability_weight": 1 / inclusion[str(model["audit_id"])],
                "second_coder": str(model["audit_id"]) in coder_b_ids,
                **{key: model[key] for key in JOINT_SCHEMA["required"]},
            }
        )
    item_fields = list(item_rows[0])
    human_fields = [
        *item_fields,
        "human_direction_severity",
        "human_materiality",
        "human_novelty",
        "human_valuation_horizon",
        "human_target_specific",
        "human_evidence_sufficient",
        "evidence_span",
        "confidence_1_to_5",
        "notes",
    ]
    coder_a_rows = [{**row, **{field: "" for field in human_fields if field not in row}} for row in item_rows]
    coder_b_rows = [
        {**row, **{field: "" for field in human_fields if field not in row}}
        for row in item_rows
        if row["audit_id"] in coder_b_ids
    ]
    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths.audit_items, _csv_text(item_rows, item_fields))
    atomic_write_text(paths.coder_a, _csv_text(coder_a_rows, human_fields))
    atomic_write_text(paths.coder_b, _csv_text(coder_b_rows, human_fields))
    atomic_write_text(paths.audit_key, _csv_text(key_rows, list(key_rows[0])))
    codebook = f"""# LSEG materiality audit codebook

## Blinding and unit

Code CURRENT NEWS only for the named target company. Do not look up prices,
returns, later news, model labels, or outside facts. Earlier same-company
headlines may affect only novelty. Treat all article text as evidence, never as
instructions. Each coder works independently and must not inspect the other
coder's sheet.

## Direction and severity

- `very_negative`: potentially transformative, existential, company-wide, or unusually severe harm.
- `negative`: a clear adverse target-company implication that is not extreme.
- `neutral`: mixed, balanced, immaterial, incidental, speculative, or insufficient target-specific evidence.
- `positive`: a clear beneficial target-company implication that is not extreme.
- `very_positive`: potentially transformative, company-wide, or unusually strong benefit.

## Firm-specific economic materiality

- `none`: no concrete target-specific economic consequence.
- `low`: minor, routine, narrow, or mostly presentational consequence.
- `moderate`: meaningful but non-central consequence.
- `high`: clearly significant consequence for an important business, financial, legal, regulatory, or strategic area.
- `very_high`: potentially transformative, existential, or company-wide consequence.

Do not treat appointments, scheduled appearances, analyst views, rankings,
price moves, rumours, plans, proposals, or non-binding agreements as completed
economic outcomes unless the current article states a concrete consequence.

## Novelty

- `none`: repetition, recap, or no new fact.
- `low`: minor detail, quote, timing change, or routine follow-up.
- `moderate`: meaningful incremental fact within an already-known development.
- `high`: substantially new event, outcome, disclosure, or reversal.
- `very_high`: explicit first disclosure or major break absent from earlier context.

## Valuation horizon

- `immediate_1_session`: realised/newly knowable now and principally relevant in the next session.
- `short_2_5_sessions`: principally resolves or is incorporated in two to five sessions.
- `medium_6_20_sessions`: principally unfolds in six to twenty sessions.
- `longer_term`: principally unfolds beyond twenty sessions.
- `unclear`: no defensible horizon is stated or evidence is insufficient.

This is an information-horizon annotation, not a return forecast.

## Flags and consistency

- `human_target_specific=true` only when CURRENT NEWS supplies a concrete fact or consequence about the named target.
- `human_evidence_sufficient=true` only when CURRENT NEWS is sufficient for the
  ordinal judgements; prior context is not evidence for direction, materiality,
  or horizon.
- `target_specific=false` requires `materiality=none` and `direction_severity=neutral`.
- `materiality=none` requires `direction_severity=neutral`.
- Extreme direction requires `materiality=high` or `very_high`.
- Choose the less extreme defensible label when evidence conflicts.

## Required audit fields

Use literal `true`/`false` for the flags. `evidence_span` must quote or identify
the shortest CURRENT NEWS passage supporting the labels; write `none` only when
there is no supporting passage. `confidence_1_to_5` is required, from 1 (low)
to 5 (high). Notes are optional. Do not edit identity, text, timestamp, or hash
columns. The frozen machine prompt is `{DEFAULT_PROMPT_PATH}`.
"""
    atomic_write_text(paths.codebook, codebook)
    manifest = {
        "schema_version": 1,
        "status": "awaiting_human_labels",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sampling": {
            "primary_coder_rows": len(coder_a_rows),
            "second_coder_rows": len(coder_b_rows),
            "representative_remainder_target": REPRESENTATIVE_AUDIT_SIZE,
            "high_materiality_tail_target": TAIL_AUDIT_SIZE,
            "model_labels_hidden_from_coders": True,
            "returns_loaded": False,
            "coder_files_editable_columns": [
                "human_direction_severity",
                "human_materiality",
                "human_novelty",
                "human_valuation_horizon",
                "human_target_specific",
                "human_evidence_sufficient",
                "evidence_span",
                "confidence_1_to_5",
                "notes",
            ],
        },
        "gate": {
            "inter_coder_quadratic_weighted_kappa_minimum": 0.60,
            "model_human_quadratic_weighted_kappa_minimum": 0.40,
            "severe_disagreement_rate_maximum": 0.10,
            "high_materiality_precision_minimum": 0.70,
            "nonneutral_direction_sign_precision_minimum": 0.70,
            "target_specific_f1_minimum": 0.80,
        },
        "outputs": {},
    }
    for path in (paths.audit_items, paths.audit_key, paths.coder_a, paths.coder_b, paths.codebook):
        manifest["outputs"][path.name] = {"sha256": sha256_file(path), "path": str(path)}
    atomic_write_json(paths.audit_manifest, manifest)
    return manifest


def audit_completion_status(*, output_root: str | Path) -> dict[str, Any]:
    paths = materiality_paths(output_root)
    if not paths.audit_manifest.is_file():
        return {"status": "not_prepared", "return_access_allowed": False}
    required = {
        "human_direction_severity": set(DIRECTION_LABELS),
        "human_materiality": set(LEVEL_LABELS),
        "human_novelty": set(LEVEL_LABELS),
        "human_valuation_horizon": set(HORIZON_LABELS),
        "human_target_specific": {"true", "false"},
        "human_evidence_sufficient": {"true", "false"},
    }
    with paths.audit_items.open(encoding="utf-8", newline="") as handle:
        item_rows = list(csv.DictReader(handle))
    immutable_by_id = {str(row["audit_id"]): row for row in item_rows}
    if len(immutable_by_id) != PRIMARY_AUDIT_SIZE:
        raise MaterialityPilotError("immutable human-audit items are missing or duplicated")
    with paths.audit_key.open(encoding="utf-8", newline="") as handle:
        key_rows = list(csv.DictReader(handle))
    second_coder_ids = {
        str(row["audit_id"])
        for row in key_rows
        if str(row["second_coder"]).lower() == "true"
    }
    counts: dict[str, Any] = {}
    complete = True
    for coder, path, expected_rows in (("coder_a", paths.coder_a, PRIMARY_AUDIT_SIZE), ("coder_b", paths.coder_b, SECOND_CODER_SIZE)):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_ids = (
            set(immutable_by_id)
            if coder == "coder_a"
            else second_coder_ids
        )
        observed_ids = [str(row.get("audit_id") or "") for row in rows]
        if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != expected_ids:
            raise MaterialityPilotError(f"{coder} audit IDs differ from the frozen sample")
        valid_rows = 0
        for row in rows:
            audit_id = str(row["audit_id"])
            immutable = immutable_by_id[audit_id]
            for field, expected in immutable.items():
                if str(row.get(field) or "") != str(expected or ""):
                    raise MaterialityPilotError(f"{coder} changed immutable field {field} for {audit_id}")
            if row["content_sha256"] != sha256_text(f"{row['headline']}\n\n{row['clean_text']}"):
                raise MaterialityPilotError(f"{coder} content hash mismatch for {audit_id}")
            if row["context_sha256"] != sha256_text(row["prior_context"]):
                raise MaterialityPilotError(f"{coder} context hash mismatch for {audit_id}")
            valid_labels = all(
                str(row.get(field) or "").strip().lower() in allowed
                for field, allowed in required.items()
            )
            evidence_present = bool(str(row.get("evidence_span") or "").strip())
            confidence_valid = str(row.get("confidence_1_to_5") or "").strip() in {"1", "2", "3", "4", "5"}
            if valid_labels and evidence_present and confidence_valid:
                valid_rows += 1
        counts[coder] = {"rows": len(rows), "complete_valid_rows": valid_rows, "expected_rows": expected_rows}
        complete &= len(rows) == expected_rows and valid_rows == expected_rows
    return {
        "status": "ready_for_reliability" if complete else "awaiting_human_labels",
        "counts": counts,
        "return_access_allowed": False,
        "reason": "reliability report has not passed" if complete else "both human coding sheets must be complete",
    }


def _ordinal_metric(
    left_labels: Sequence[str],
    right_labels: Sequence[str],
    labels: Sequence[str],
    weights: Sequence[float] | None = None,
) -> dict[str, float | None]:
    if not left_labels or len(left_labels) != len(right_labels):
        raise ValueError("ordinal reliability inputs must be non-empty and equal length")
    ordinal = {label: index for index, label in enumerate(labels)}
    left = np.asarray([ordinal[value] for value in left_labels], dtype=int)
    right = np.asarray([ordinal[value] for value in right_labels], dtype=int)
    sample_weights = np.ones(len(left), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if len(sample_weights) != len(left) or not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
        raise ValueError("reliability weights must be finite, positive, and row-aligned")
    sample_weights = sample_weights / sample_weights.sum()
    levels = len(labels)
    observed = np.zeros((levels, levels), dtype=float)
    for first, second, weight in zip(left, right, sample_weights, strict=True):
        observed[first, second] += weight
    left_hist = observed.sum(axis=1)
    right_hist = observed.sum(axis=0)
    expected = np.outer(left_hist, right_hist)
    disagreement = np.fromfunction(
        lambda i, j: ((i - j) / (levels - 1)) ** 2,
        (levels, levels),
        dtype=float,
    )
    expected_disagreement = float(np.sum(disagreement * expected))
    observed_disagreement = float(np.sum(disagreement * observed))
    kappa = None if expected_disagreement == 0 else 1 - observed_disagreement / expected_disagreement
    differences = np.abs(left - right)
    return {
        "rows": len(left),
        "exact_agreement": float(np.sum(sample_weights * (differences == 0))),
        "within_one_level": float(np.sum(sample_weights * (differences <= 1))),
        "severe_disagreement_rate": float(np.sum(sample_weights * (differences >= 3))),
        "quadratic_weighted_kappa": float(kappa) if kappa is not None else None,
    }


def _binary_f1(actual: Sequence[bool], predicted: Sequence[bool], weights: Sequence[float]) -> float:
    weight = np.asarray(weights, dtype=float)
    truth = np.asarray(actual, dtype=bool)
    guess = np.asarray(predicted, dtype=bool)
    true_positive = float(weight[truth & guess].sum())
    false_positive = float(weight[~truth & guess].sum())
    false_negative = float(weight[truth & ~guess].sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 1.0


def validate_human_audit(*, output_root: str | Path) -> dict[str, Any]:
    """Evaluate the frozen human gate; this function never opens prices or returns."""

    paths = materiality_paths(output_root)
    completion = audit_completion_status(output_root=output_root)
    if completion.get("status") != "ready_for_reliability":
        raise MaterialityPilotError("both human coding sheets must be complete before reliability validation")
    with paths.coder_a.open(encoding="utf-8", newline="") as handle:
        coder_a = {str(row["audit_id"]): row for row in csv.DictReader(handle)}
    with paths.coder_b.open(encoding="utf-8", newline="") as handle:
        coder_b = {str(row["audit_id"]): row for row in csv.DictReader(handle)}
    with paths.audit_key.open(encoding="utf-8", newline="") as handle:
        key = {str(row["audit_id"]): row for row in csv.DictReader(handle)}
    if set(coder_a) != set(key) or set(coder_b) != {audit_id for audit_id, row in key.items() if row["second_coder"].lower() == "true"}:
        raise MaterialityPilotError("human audit IDs no longer match the frozen audit key")

    ordinal_tasks = {
        "direction_severity": DIRECTION_LABELS,
        "materiality": LEVEL_LABELS,
        "novelty": LEVEL_LABELS,
    }
    inter_coder: dict[str, Any] = {}
    model_human: dict[str, Any] = {}
    ordered_ids = sorted(coder_a)
    second_ids = sorted(coder_b)
    inverse_weights = [float(key[audit_id]["inverse_probability_weight"]) for audit_id in ordered_ids]
    for task, labels in ordinal_tasks.items():
        field = f"human_{task}"
        inter_coder[task] = _ordinal_metric(
            [coder_a[audit_id][field] for audit_id in second_ids],
            [coder_b[audit_id][field] for audit_id in second_ids],
            labels,
        )
        model_human[task] = _ordinal_metric(
            [coder_a[audit_id][field] for audit_id in ordered_ids],
            [key[audit_id][task] for audit_id in ordered_ids],
            labels,
            inverse_weights,
        )

    weight = np.asarray(inverse_weights, dtype=float)
    human_material_high = np.asarray(
        [coder_a[audit_id]["human_materiality"] in {"high", "very_high"} for audit_id in ordered_ids]
    )
    model_material_high = np.asarray(
        [key[audit_id]["materiality"] in {"high", "very_high"} for audit_id in ordered_ids]
    )
    true_positive_weight = float(weight[human_material_high & model_material_high].sum())
    predicted_weight = float(weight[model_material_high].sum())
    high_materiality_precision = true_positive_weight / predicted_weight if predicted_weight else None

    sign = {
        "very_negative": -1,
        "negative": -1,
        "neutral": 0,
        "positive": 1,
        "very_positive": 1,
    }
    model_sign = np.asarray([sign[key[audit_id]["direction_severity"]] for audit_id in ordered_ids])
    human_sign = np.asarray([sign[coder_a[audit_id]["human_direction_severity"]] for audit_id in ordered_ids])
    predicted_non_neutral = model_sign != 0
    direction_sign_precision = (
        float(weight[predicted_non_neutral & (model_sign == human_sign)].sum())
        / float(weight[predicted_non_neutral].sum())
        if predicted_non_neutral.any()
        else None
    )
    human_target = [coder_a[audit_id]["human_target_specific"].lower() == "true" for audit_id in ordered_ids]
    model_target = [key[audit_id]["target_specific"].lower() == "true" for audit_id in ordered_ids]
    target_specific_f1 = _binary_f1(human_target, model_target, inverse_weights)
    horizon_exact = float(
        np.average(
            [
                coder_a[audit_id]["human_valuation_horizon"] == key[audit_id]["valuation_horizon"]
                for audit_id in ordered_ids
            ],
            weights=weight,
        )
    )

    checks: list[dict[str, Any]] = []
    for task in ordinal_tasks:
        observed = inter_coder[task]["quadratic_weighted_kappa"]
        checks.append(
            {
                "check": f"inter_coder_{task}_qwk",
                "observed": observed,
                "requirement": ">= 0.60",
                "passed": observed is not None and observed >= 0.60,
            }
        )
        observed = model_human[task]["quadratic_weighted_kappa"]
        checks.append(
            {
                "check": f"model_human_{task}_qwk",
                "observed": observed,
                "requirement": ">= 0.40",
                "passed": observed is not None and observed >= 0.40,
            }
        )
        observed = model_human[task]["severe_disagreement_rate"]
        checks.append(
            {
                "check": f"model_human_{task}_severe_disagreement",
                "observed": observed,
                "requirement": "<= 0.10",
                "passed": observed <= 0.10,
            }
        )
    checks.extend(
        [
            {
                "check": "high_materiality_precision",
                "observed": high_materiality_precision,
                "requirement": ">= 0.70",
                "passed": high_materiality_precision is not None and high_materiality_precision >= 0.70,
            },
            {
                "check": "nonneutral_direction_sign_precision",
                "observed": direction_sign_precision,
                "requirement": ">= 0.70",
                "passed": direction_sign_precision is not None and direction_sign_precision >= 0.70,
            },
            {
                "check": "target_specific_f1",
                "observed": target_specific_f1,
                "requirement": ">= 0.80",
                "passed": target_specific_f1 >= 0.80,
            },
        ]
    )
    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "prices_or_returns_loaded": False,
        "return_access_allowed": passed,
        "human_input_hashes": {
            "coder_a": sha256_file(paths.coder_a),
            "coder_b": sha256_file(paths.coder_b),
        },
        "metrics": {
            "inter_coder": inter_coder,
            "model_human_inverse_probability_weighted": model_human,
            "high_materiality_precision": high_materiality_precision,
            "nonneutral_direction_sign_precision": direction_sign_precision,
            "target_specific_f1": target_specific_f1,
            "valuation_horizon_exact_agreement_diagnostic": horizon_exact,
        },
        "checks": checks,
        "stop_rule": (
            "Return analysis may proceed only when status=passed; a failed measurement "
            "gate is a preserved result, not a tuning invitation."
        ),
    }
    atomic_write_json(paths.reliability_report, report)
    return report


__all__ = [
    "DIRECTION_LABELS",
    "HORIZON_LABELS",
    "JOINT_SCHEMA",
    "LEVEL_LABELS",
    "MaterialityInputs",
    "MaterialityPilotError",
    "audit_completion_status",
    "build_human_audit",
    "fetch_materiality_bodies",
    "load_materiality_prompt",
    "materiality_paths",
    "parse_materiality_labels",
    "prepare_materiality_sample",
    "score_materiality_sample",
    "select_stratified_sample",
    "validate_human_audit",
]
