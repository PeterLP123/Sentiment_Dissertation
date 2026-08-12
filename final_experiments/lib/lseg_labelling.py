"""Reproducible label inheritance for the expanded LSEG headline corpus."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sentiment_benchmark.artifact_io import sha256_file, sha256_text
from sentiment_benchmark.headline_return_study import BASELINE_COLUMNS
from sentiment_benchmark.headline_value import ScorableHeadline, collect_scorable_headline_records

FINBERT_MODEL_ID = "ProsusAI/finbert"
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
LEGACY_EXECUTION_SCORERS = ("finbert", "vader")


@dataclass(frozen=True)
class BaselineSeedSummary:
    target_unique_headlines: int
    reusable_unique_headlines: int
    missing_unique_headlines: int
    inherited_score_rows: int
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class FinbertExportSummary:
    unique_headlines: int
    output_path: Path
    manifest_path: Path


def _population(collection_root: Path) -> tuple[dict[str, ScorableHeadline], str]:
    records = {record.headline_sha256: record for record in collect_scorable_headline_records(collection_root)}
    hashes = set(records)
    return records, sha256_text("\n".join(sorted(hashes)))


def _validate_prior_manifest(scores_path: Path) -> dict:
    manifest_path = scores_path.with_suffix(scores_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"prior score manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    finbert = (manifest.get("models") or {}).get("finbert") or {}
    if manifest.get("status") != "completed":
        raise ValueError("prior score manifest is not completed")
    if finbert.get("model_id") != FINBERT_MODEL_ID or finbert.get("revision") != FINBERT_REVISION:
        raise ValueError("prior scores do not use the frozen FinBERT model and revision")
    if finbert.get("revision_enforced") is not True:
        raise ValueError("prior FinBERT revision was not enforced")
    expected_hash = str((manifest.get("output") or {}).get("sha256") or "")
    if expected_hash != sha256_file(scores_path):
        raise ValueError("prior score artifact does not match its manifest hash")
    return manifest


def _existing_summary(output_path: Path, manifest_path: Path) -> BaselineSeedSummary | None:
    if not output_path.exists() and not manifest_path.exists():
        return None
    if not output_path.is_file() or not manifest_path.is_file():
        raise ValueError("incomplete inherited-score seed artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("output", {}).get("sha256") != sha256_file(output_path):
        raise ValueError("existing inherited-score seed has changed; it may already be in active scoring")
    summary = manifest.get("summary") or {}
    return BaselineSeedSummary(
        target_unique_headlines=int(summary["target_unique_headlines"]),
        reusable_unique_headlines=int(summary["reusable_unique_headlines"]),
        missing_unique_headlines=int(summary["missing_unique_headlines"]),
        inherited_score_rows=int(summary["inherited_score_rows"]),
        output_path=output_path,
        manifest_path=manifest_path,
    )


def prepare_reusable_baseline_seed(
    collection_root: str | Path,
    prior_scores_path: str | Path,
    output_path: str | Path,
) -> BaselineSeedSummary:
    """Copy only compatible prior labels that belong to the expanded population.

    The output matches the paired resume contract used by the already-completed
    2026-08-03 compatibility run. It must not initiate a new VADER run; current
    downstream work uses :func:`export_finbert_only_scores`.
    """
    root = Path(collection_root)
    prior = Path(prior_scores_path)
    output = Path(output_path)
    seed_manifest = output.with_suffix(output.suffix + ".seed.json")
    existing = _existing_summary(output, seed_manifest)
    if existing is not None:
        return existing
    if not prior.is_file():
        raise ValueError(f"prior score artifact is missing: {prior}")
    _validate_prior_manifest(prior)
    prior_manifest_path = prior.with_suffix(prior.suffix + ".manifest.json")
    target_records, population_sha256 = _population(root)
    target_hashes = set(target_records)
    output.parent.mkdir(parents=True, exist_ok=True)

    pairs: set[tuple[str, str]] = set()
    reusable_hashes: set[str] = set()
    metadata_changes: dict[str, set[str]] = {}
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    try:
        writer = csv.DictWriter(temporary_handle, fieldnames=BASELINE_COLUMNS)
        writer.writeheader()
        with prior.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != BASELINE_COLUMNS:
                raise ValueError("prior score artifact has an incompatible schema")
            for row in reader:
                headline_hash = str(row.get("headline_sha256") or "")
                scorer = str(row.get("baseline") or "")
                if headline_hash not in target_hashes or scorer not in LEGACY_EXECUTION_SCORERS or row.get("status") != "success":
                    continue
                pair = (headline_hash, scorer)
                if pair in pairs:
                    raise ValueError(f"prior score artifact duplicates {headline_hash}/{scorer}")
                pairs.add(pair)
                reusable_hashes.add(headline_hash)
                record = target_records[headline_hash]
                canonical_metadata = {
                    "headline": record.headline,
                    "matched_symbols": "|".join(record.matched_symbols),
                    "first_timestamp": record.first_timestamp,
                    "explicit_target": str(record.explicit_target),
                    "contextual": str(record.contextual),
                    "market_price_technical": str(record.market_price_technical),
                }
                for field, value in canonical_metadata.items():
                    if row.get(field) != value:
                        metadata_changes.setdefault(headline_hash, set()).add(field)
                    row[field] = value
                writer.writerow(row)
        temporary_handle.flush()
        os.fsync(temporary_handle.fileno())
        temporary_handle.close()
        incomplete = sorted(
            headline_hash
            for headline_hash in reusable_hashes
            if any((headline_hash, scorer) not in pairs for scorer in LEGACY_EXECUTION_SCORERS)
        )
        if incomplete:
            raise ValueError(f"prior labels are incomplete for {len(incomplete)} reusable headlines")
        if len(pairs) != len(LEGACY_EXECUTION_SCORERS) * len(reusable_hashes):
            raise ValueError("inherited score rows do not reconcile to reusable headlines")
        os.replace(temporary_path, output)
    except BaseException:
        temporary_handle.close()
        temporary_path.unlink(missing_ok=True)
        raise

    summary = BaselineSeedSummary(
        target_unique_headlines=len(target_hashes),
        reusable_unique_headlines=len(reusable_hashes),
        missing_unique_headlines=len(target_hashes - reusable_hashes),
        inherited_score_rows=len(pairs),
        output_path=output,
        manifest_path=seed_manifest,
    )
    metadata_change_counts = Counter(
        field
        for fields in metadata_changes.values()
        for field in fields
    )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "built_at": datetime.now(UTC).isoformat(),
        "contract": {
            "estimand": "headline-level financial sentiment probabilities and p_positive minus p_negative",
            "unit": "unique normalized headline text",
            "label_timing": "headline text only; no returns or post-publication fields",
            "scorers": list(LEGACY_EXECUTION_SCORERS),
            "finbert_model_id": FINBERT_MODEL_ID,
            "finbert_revision": FINBERT_REVISION,
            "reuse_rule": "inherit only successful exact normalized-headline hashes with both frozen scorers",
            "metadata_rule": "rewrite inherited headline metadata from the target snapshot; inherit probabilities only",
        },
        "input": {
            "collection_root": str(root),
            "population_sha256": population_sha256,
            "prior_scores_path": str(prior),
            "prior_scores_sha256": sha256_file(prior),
            "prior_manifest_sha256": sha256_file(prior_manifest_path),
        },
        "summary": {
            key: value
            for key, value in asdict(summary).items()
            if key not in {"output_path", "manifest_path"}
        },
        "metadata_reconciliation": {
            "changed_headlines": len(metadata_changes),
            "changed_headlines_by_field": dict(sorted(metadata_change_counts.items())),
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "sharing": {"contains_licensed_headline_text": True, "source_control": False, "redistribute": False},
    }
    temporary_manifest = seed_manifest.with_name(f".{seed_manifest.name}.tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, seed_manifest)
    return summary


def export_finbert_only_scores(
    combined_scores_path: str | Path,
    output_path: str | Path,
) -> FinbertExportSummary:
    """Export the frozen FinBERT rows and exclude VADER from downstream work."""
    combined = Path(combined_scores_path)
    output = Path(output_path)
    combined_manifest_path = combined.with_suffix(combined.suffix + ".manifest.json")
    if not combined.is_file() or not combined_manifest_path.is_file():
        raise ValueError("combined score artifact or manifest is missing")
    combined_manifest = json.loads(combined_manifest_path.read_text(encoding="utf-8"))
    finbert = (combined_manifest.get("models") or {}).get("finbert") or {}
    if combined_manifest.get("status") != "completed":
        raise ValueError("combined score manifest is not completed")
    if finbert.get("model_id") != FINBERT_MODEL_ID or finbert.get("revision") != FINBERT_REVISION:
        raise ValueError("combined scores do not use the frozen FinBERT model and revision")
    if finbert.get("revision_enforced") is not True:
        raise ValueError("combined FinBERT revision was not enforced")
    if (combined_manifest.get("output") or {}).get("sha256") != sha256_file(combined):
        raise ValueError("combined score artifact does not match its manifest hash")
    expected = int((combined_manifest.get("input") or {}).get("unique_headlines") or 0)
    counts = combined_manifest.get("counts") or {}
    if int(counts.get("finbert_successes") or 0) != expected:
        raise ValueError("combined score manifest does not report complete FinBERT coverage")
    if int(counts.get("vader_successes") or 0) != expected:
        raise ValueError("combined score manifest does not reconcile the excluded VADER rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    seen: set[str] = set()
    try:
        writer = csv.DictWriter(temporary_handle, fieldnames=BASELINE_COLUMNS)
        writer.writeheader()
        with combined.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != BASELINE_COLUMNS:
                raise ValueError("combined score artifact has an incompatible schema")
            for row in reader:
                if row.get("baseline") != "finbert":
                    continue
                if row.get("status") != "success":
                    raise ValueError("combined score artifact contains a failed FinBERT row")
                headline_hash = str(row.get("headline_sha256") or "")
                if not headline_hash or headline_hash in seen:
                    raise ValueError(f"combined score artifact has an invalid FinBERT hash: {headline_hash!r}")
                probabilities = [float(row[name]) for name in ("p_positive", "p_negative", "p_neutral")]
                if any(value < 0.0 or value > 1.0 for value in probabilities):
                    raise ValueError(f"FinBERT probabilities are outside [0, 1] for {headline_hash}")
                if abs(sum(probabilities) - 1.0) > 1e-4:
                    raise ValueError(f"FinBERT probabilities do not sum to one for {headline_hash}")
                if abs(float(row["score"]) - (probabilities[0] - probabilities[1])) > 1e-4:
                    raise ValueError(f"FinBERT score does not match its probabilities for {headline_hash}")
                seen.add(headline_hash)
                writer.writerow(row)
        if len(seen) != expected:
            raise ValueError(f"FinBERT export has {len(seen)} rows; expected {expected}")
        temporary_handle.flush()
        os.fsync(temporary_handle.fileno())
        temporary_handle.close()
        os.replace(temporary_path, output)
    except BaseException:
        temporary_handle.close()
        temporary_path.unlink(missing_ok=True)
        raise

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    payload = {
        "schema_version": 1,
        "status": "completed",
        "built_at": datetime.now(UTC).isoformat(),
        "decision": "VADER excluded from downstream analysis by researcher decision on 2026-08-03",
        "contract": {
            "unit": "unique normalized headline text",
            "scorer": "finbert",
            "model_id": FINBERT_MODEL_ID,
            "revision": FINBERT_REVISION,
            "revision_enforced": True,
        },
        "input": {
            "combined_scores_path": str(combined),
            "combined_scores_sha256": sha256_file(combined),
            "combined_manifest_sha256": sha256_file(combined_manifest_path),
        },
        "summary": {"unique_headlines": len(seen), "excluded_vader_rows": len(seen)},
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "sharing": {"contains_licensed_headline_text": True, "source_control": False, "redistribute": False},
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return FinbertExportSummary(len(seen), output, manifest_path)


__all__ = [
    "BaselineSeedSummary",
    "FinbertExportSummary",
    "FINBERT_MODEL_ID",
    "FINBERT_REVISION",
    "export_finbert_only_scores",
    "prepare_reusable_baseline_seed",
]
