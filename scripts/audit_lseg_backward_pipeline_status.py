#!/usr/bin/env python3
"""Audit aggregate download, FinBERT, and Gemma status for a backward snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.backward_validation import audit_backward_collection  # noqa: E402
from final_experiments.lib.openrouter_validation import (  # noqa: E402
    MODEL_ID,
    PROMPT_HASH,
    PROVIDER_NAME,
)
from sentiment_benchmark.artifact_io import atomic_write_json, sha256_file  # noqa: E402

DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / (
    "Data/collections/lseg_us_sector_33_backward_20240101_20251026_headlines/"
    "derived/incremental_finbert/snapshot_20260810_second_account_quota_stop"
)
DEFAULT_GEMMA_OUTPUTS = (
    REPO_ROOT
    / (
        "Data/collections/lseg_us_sector_33_backward_20240101_20251026_headlines/"
        "derived/incremental_gemma/snapshot_20260809_quota_stop/"
        "headline_scores_gemma4_26b_a4b_it_deepinfra_fp8_investor_headline_soft_label_v1.csv"
    ),
    REPO_ROOT
    / (
        "Data/collections/lseg_us_sector_33_backward_20240101_20251026_headlines/"
        "derived/incremental_gemma/snapshot_20260810_second_account_quota_stop/"
        "headline_scores_gemma4_26b_a4b_it_deepinfra_fp8_investor_headline_soft_label_v1_delta.csv"
    ),
)
DEFAULT_FINBERT_COMPLETION = DEFAULT_SNAPSHOT_ROOT / "finbert_verified_completion.json"
DEFAULT_FINBERT_STAGING = REPO_ROOT / (
    "Data/collections/lseg_us_sector_33_backward_20240101_20251026_headlines/"
    "derived/incremental_finbert_seed/snapshot_20260810_second_account_quota_stop/finbert_staging_record.json"
)
DEFAULT_STATUS_OUTPUT = DEFAULT_SNAPSHOT_ROOT / "pipeline_status.json"
BACKWARD_SPEC = REPO_ROOT / "final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--gemma-output", type=Path, action="append", default=None)
    parser.add_argument("--finbert-completion", type=Path, default=DEFAULT_FINBERT_COMPLETION)
    parser.add_argument("--finbert-staging", type=Path, default=DEFAULT_FINBERT_STAGING)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATUS_OUTPUT)
    parser.add_argument("--daily-request-budget", type=int, default=10_000)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _gemma_status(outputs: tuple[Path, ...], expected_population: int) -> dict[str, Any]:
    available = [output for output in outputs if output.is_file()]
    if not available:
        return {
            "status": "not_started",
            "outputs": [str(output) for output in outputs],
            "population": expected_population,
            "unique_successes": 0,
            "remaining": expected_population,
        }
    attempt_rows = 0
    statuses: Counter[str] = Counter()
    success_hashes: set[str] = set()
    duplicate_successes = 0
    invalid_success_probabilities = 0
    route_mismatches = 0
    config_hashes: dict[str, Counter[str]] = {}
    reported_cost_usd = 0.0
    artifacts: list[dict[str, Any]] = []
    manifest_running = False
    for output in available:
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"Gemma output exists without its manifest: {manifest_path}")
        manifest = _load_json(manifest_path)
        manifest_status = str(manifest.get("status") or "unknown")
        manifest_running |= manifest_status == "running"
        artifact_attempts = 0
        artifact_cost = 0.0
        artifact_configs: Counter[str] = Counter()
        with output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "headline_sha256", "model_id", "provider", "quantization", "prompt_hash",
                "p_positive", "p_negative", "p_neutral", "status", "reported_cost_usd", "config_hash",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Gemma output is missing required columns: {sorted(missing)}")
            for row in reader:
                attempt_rows += 1
                artifact_attempts += 1
                status = str(row["status"])
                statuses[status] += 1
                artifact_configs[str(row["config_hash"])] += 1
                row_cost = float(row["reported_cost_usd"] or 0.0)
                reported_cost_usd += row_cost
                artifact_cost += row_cost
                if status != "success":
                    continue
                headline_hash = str(row["headline_sha256"])
                if headline_hash in success_hashes:
                    duplicate_successes += 1
                success_hashes.add(headline_hash)
                try:
                    probabilities = [float(row[name]) for name in ("p_positive", "p_negative", "p_neutral")]
                except ValueError:
                    invalid_success_probabilities += 1
                else:
                    if not (
                        all(0.0 <= value <= 1.0 for value in probabilities)
                        and math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
                    ):
                        invalid_success_probabilities += 1
                route_mismatches += not (
                    row["model_id"] == MODEL_ID
                    and row["provider"].casefold() == PROVIDER_NAME.casefold()
                    and row["quantization"] == "fp8"
                    and row["prompt_hash"] == PROMPT_HASH
                )
        config_hashes[str(output)] = artifact_configs
        expected_hash = manifest.get("output_sha256")
        artifacts.append(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "manifest_status": manifest_status,
                "attempt_rows": artifact_attempts,
                "reported_cost_usd": artifact_cost,
                "output_sha256_verified": sha256_file(output) == expected_hash if expected_hash else None,
            }
        )

    unique_successes = len(success_hashes)
    if unique_successes > expected_population:
        raise ValueError("Gemma successes exceed the frozen snapshot population")
    validation_pass = (
        duplicate_successes == 0
        and invalid_success_probabilities == 0
        and route_mismatches == 0
        and all(len(values) <= 1 for values in config_hashes.values())
    )
    status = (
        "completed"
        if unique_successes == expected_population and validation_pass
        else "running"
        if manifest_running
        else "partial"
    )

    return {
        "status": status,
        "artifacts": artifacts,
        "population": expected_population,
        "attempt_rows": attempt_rows,
        "status_counts": dict(sorted(statuses.items())),
        "unique_successes": unique_successes,
        "coverage_pct": 100.0 * unique_successes / expected_population,
        "remaining": expected_population - unique_successes,
        "duplicate_successes": duplicate_successes,
        "invalid_success_probabilities": invalid_success_probabilities,
        "route_mismatches": route_mismatches,
        "config_hashes_by_artifact": {path: dict(values) for path, values in config_hashes.items()},
        "reported_cost_usd": reported_cost_usd,
        "validation_pass": validation_pass,
        "model": MODEL_ID,
        "provider": PROVIDER_NAME,
        "quantization": "fp8",
        "prompt_hash": PROMPT_HASH,
    }


def _finbert_status(completion: Path, staging: Path, snapshot: dict[str, Any], population: int) -> dict[str, Any]:
    if completion.is_file():
        finbert = _load_json(completion)
        if finbert.get("population_sha256") != snapshot["population_sha256"]:
            raise ValueError("verified FinBERT completion belongs to a different population")
        if int(finbert.get("finbert_successes", -1)) != population:
            raise ValueError("verified FinBERT completion does not cover the snapshot population")
        return finbert
    if not staging.is_file():
        return {"status": "not_started", "population": population, "remaining": population}
    record = _load_json(staging)
    seed = record["resume_seed"]
    reusable = int(seed["inherited_unique_headlines"])
    return {
        "status": record["status"],
        "population": population,
        "population_sha256": snapshot["population_sha256"],
        "reusable_successes": reusable,
        "coverage_pct": 100.0 * reusable / population,
        "remaining": population - reusable,
        "staging_record": str(staging),
        "remote_snapshot_root": record["snapshot"]["remote_root"],
        "remote_seed_path": seed["remote_path"],
        "metadata_changed_headlines": seed["metadata_changed_headlines"],
        "watcher": record.get("watcher"),
    }


def main() -> None:
    args = parse_args()
    snapshot_manifest_path = args.snapshot_root / "manifest.json"
    snapshot_manifest = _load_json(snapshot_manifest_path)
    snapshot = snapshot_manifest["snapshot"]
    population = int(snapshot["unique_scorable_headlines"])

    collection_audit = audit_backward_collection(
        REPO_ROOT,
        BACKWARD_SPEC,
        daily_request_budget=args.daily_request_budget,
    )
    progress = collection_audit.summary.iloc[0]
    projection = collection_audit.projection.iloc[0]
    frontier_companies = progress["frontier_companies_completed"]

    finbert = _finbert_status(args.finbert_completion, args.finbert_staging, snapshot, population)

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "return_blind": True,
        "prices_or_returns_loaded": False,
        "snapshot": {
            "id": args.snapshot_root.name,
            "manifest": str(snapshot_manifest_path),
            "manifest_sha256": sha256_file(snapshot_manifest_path),
            "population_sha256": snapshot["population_sha256"],
            "unique_scorable_headlines": population,
            "completed_company_dates": int(snapshot["completed_company_dates"]),
            "source_pages": int(snapshot["source_pages"]),
            "partial_collection": bool(snapshot["partial_collection"]),
        },
        "lseg_downloads": {
            "status": str(progress["status"]),
            "completed_company_dates": int(progress["completed_windows"]),
            "expected_company_dates": int(progress["expected_windows"]),
            "coverage_pct": float(progress["window_completion_pct"]),
            "saved_pages": int(progress["headline_pages"]),
            "complete_dates": int(progress["complete_dates"]),
            "complete_date_prefix": int(progress["complete_date_prefix"]),
            "frontier_date": progress["frontier_date"],
            "frontier_companies_completed": (
                None if frontier_companies is None else int(frontier_companies)
            ),
            "pagination_anomalies": int(progress["pagination_anomalies"]),
            "estimated_remaining_full_quota_days": int(projection["estimated_remaining_quota_days"]),
        },
        "finbert": finbert,
        "gemma": _gemma_status(tuple(args.gemma_output or DEFAULT_GEMMA_OUTPUTS), population),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
