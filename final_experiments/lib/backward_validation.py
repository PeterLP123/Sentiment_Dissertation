"""Return-blind gates for the frozen LSEG backward replication.

This module may inspect acquisition metadata and sentiment scores. It does not
import a price loader, construct a return, or execute a portfolio.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment_benchmark.artifact_io import canonical_json, read_json, sha256_file, sha256_text
from sentiment_benchmark.lseg_source import collection_windows, load_lseg_collection_config

SAFE_PAGINATION_ANOMALY_REASONS = frozenset({"repeated_cursor_duplicate_page"})


class BackwardValidationError(ValueError):
    """A frozen backward-replication contract or input failed closed."""


@dataclass(frozen=True)
class BackwardCollectionAudit:
    """Licence-safe progress and gate evidence for the headline collection."""

    summary: pd.DataFrame
    date_progress: pd.DataFrame
    projection: pd.DataFrame
    gate_table: pd.DataFrame
    collection_gate_pass: bool


@dataclass(frozen=True)
class GemmaDriftAudit:
    """Pre-return stability evidence for the exact-endpoint Gemma trigger."""

    metrics: pd.DataFrame
    gate_table: pd.DataFrame
    drift_gate_pass: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BackwardValidationError(message)


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackwardValidationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BackwardValidationError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve(root: Path, value: object, *, label: str) -> Path:
    raw = str(value or "").strip()
    _require(bool(raw), f"{label} path is missing")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _checkpoint_progress(
    pages_dir: Path,
    *,
    symbols: frozenset[str],
    maximum: int,
) -> tuple[set[tuple[str, int]], dict[int, int], dict[int, int], int, int]:
    completed: set[tuple[str, int]] = set()
    page_requests_by_window: dict[int, int] = {}
    page_bytes_by_window: dict[int, int] = {}
    page_count = 0
    page_bytes = 0
    for page_path in pages_dir.glob("*.json"):
        page_count += 1
        checkpoint_bytes = page_path.stat().st_size
        page_bytes += checkpoint_bytes
        payload = read_json(page_path)
        symbol = str(payload.get("symbol") or "").strip().upper()
        try:
            window_index = int(payload.get("window_index"))
        except (TypeError, ValueError):
            continue
        if symbol not in symbols or not 1 <= window_index <= maximum:
            continue
        page_requests_by_window[window_index] = page_requests_by_window.get(window_index, 0) + 1
        page_bytes_by_window[window_index] = page_bytes_by_window.get(window_index, 0) + checkpoint_bytes
        terminal = not str(payload.get("cursor_out") or "").strip() or bool(str(payload.get("pagination_terminal_reason") or "").strip())
        if terminal:
            completed.add((symbol, window_index))
    return completed, page_requests_by_window, page_bytes_by_window, page_count, page_bytes


def _directory_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def audit_backward_collection(
    repo_root: str | Path,
    spec_path: str | Path,
    *,
    daily_request_budget: int = 10_000,
) -> BackwardCollectionAudit:
    """Audit frozen identities and collection progress without reading returns."""

    _require(daily_request_budget > 0, "daily request budget must be positive")
    root = Path(repo_root).resolve()
    spec_file = _resolve(root, spec_path, label="backward specification")
    spec = _read_object(spec_file, label="backward specification")
    _require(spec.get("status") == "frozen_awaiting_lseg_collection", "backward specification is not frozen")

    parent_entry = spec.get("parent_strategy_spec")
    _require(isinstance(parent_entry, dict), "parent strategy entry is missing")
    parent_path = _resolve(root, parent_entry.get("path"), label="parent strategy specification")
    _require(sha256_file(parent_path) == parent_entry.get("sha256"), "parent strategy hash mismatch")

    collection_entry = spec.get("collection")
    _require(isinstance(collection_entry, dict), "collection entry is missing")
    config_path = _resolve(root, collection_entry.get("config_path"), label="collection config")
    _require(sha256_file(config_path) == collection_entry.get("config_file_sha256"), "collection config file hash mismatch")
    config = load_lseg_collection_config(config_path)
    _require(config.config_sha256 == collection_entry.get("parsed_config_sha256"), "parsed collection config hash mismatch")
    _require(config.collection_id == collection_entry.get("collection_id"), "collection id mismatch")
    _require(config.fetch_story_bodies is False, "backward collection must remain headline-only")
    _require(config.window_days == 1, "backward collection must retain one-day windows")
    _require(config.max_requests_per_run == 9500, "backward collection must retain the 9500-request cap")

    expected_symbols = frozenset(company.symbol for company in config.companies)
    _require(len(expected_symbols) == 33, "backward collection must contain exactly 33 unique symbols")
    windows = collection_windows(config)
    window_count = len(windows)
    expected_windows = len(expected_symbols) * window_count
    raw_dir = root / config.raw_dir
    pages_dir = raw_dir / "headline_pages"
    manifest_path = raw_dir / "manifest.json"

    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = _read_object(manifest_path, label="collection manifest")
        _require(manifest.get("config_sha256") == config.config_sha256, "collection manifest config hash mismatch")
        _require(
            canonical_json(manifest.get("config")) == canonical_json(config.to_payload()),
            "collection manifest config payload mismatch",
        )

    if pages_dir.is_dir():
        completed, page_requests_by_window, page_bytes_by_window, page_count, page_bytes = _checkpoint_progress(
            pages_dir,
            symbols=expected_symbols,
            maximum=window_count,
        )
    else:
        completed = set()
        page_requests_by_window = {}
        page_bytes_by_window = {}
        page_count = 0
        page_bytes = 0

    completed_companies_by_window: dict[int, int] = {}
    for _, window_index in completed:
        completed_companies_by_window[window_index] = completed_companies_by_window.get(window_index, 0) + 1
    date_progress = pd.DataFrame(
        [
            {
                "window_index": window.index,
                "window_start": window.start,
                "window_end": window.end,
                "completed_companies": completed_companies_by_window.get(window.index, 0),
                "expected_companies": len(expected_symbols),
                "company_completion_pct": (100.0 * completed_companies_by_window.get(window.index, 0) / len(expected_symbols)),
                "complete_date": completed_companies_by_window.get(window.index, 0) == len(expected_symbols),
                "page_requests": page_requests_by_window.get(window.index, 0),
                "checkpoint_mib": page_bytes_by_window.get(window.index, 0) / 2**20,
            }
            for window in windows
        ]
    )
    complete_dates = int(date_progress["complete_date"].sum())
    complete_prefix_dates = 0
    for is_complete in date_progress["complete_date"]:
        if not bool(is_complete):
            break
        complete_prefix_dates += 1
    frontier = date_progress.iloc[complete_prefix_dates] if complete_prefix_dates < window_count else None
    partial_dates = int(((date_progress["completed_companies"] > 0) & ~date_progress["complete_date"]).sum())
    untouched_dates = int((date_progress["completed_companies"] == 0).sum())

    observed_requests_per_completed_window = page_count / len(completed) if completed else float("nan")
    if complete_dates:
        observed_requests_per_complete_date = float(date_progress.loc[date_progress["complete_date"], "page_requests"].mean())
        estimated_total_requests = observed_requests_per_complete_date * window_count
        projection_basis = "mean page requests across completed 33-company dates"
    elif completed:
        observed_requests_per_complete_date = float("nan")
        estimated_total_requests = observed_requests_per_completed_window * expected_windows
        projection_basis = "mean page requests per completed company-window; company-biased until the first complete date"
    else:
        observed_requests_per_complete_date = float("nan")
        estimated_total_requests = float(expected_windows)
        projection_basis = "one-request-per-company-window minimum; no completed checkpoints yet"
    minimum_remaining_requests = max(expected_windows - len(completed), 0)
    estimated_remaining_requests = max(math.ceil(estimated_total_requests - page_count), minimum_remaining_requests)
    raw_storage_bytes = _directory_bytes(raw_dir)
    storage_probe = raw_dir if raw_dir.exists() else root
    free_disk_bytes = shutil.disk_usage(storage_probe).free
    observed_bytes_per_page = page_bytes / page_count if page_count else float("nan")
    projected_additional_storage_bytes = (
        estimated_remaining_requests * observed_bytes_per_page if np.isfinite(observed_bytes_per_page) else float("nan")
    )
    projected_free_after_completion_bytes = (
        free_disk_bytes - projected_additional_storage_bytes if np.isfinite(projected_additional_storage_bytes) else float("nan")
    )
    projection = pd.DataFrame(
        [
            {
                "projection_basis": projection_basis,
                "daily_request_budget": daily_request_budget,
                "observed_requests_per_completed_company_window": observed_requests_per_completed_window,
                "observed_requests_per_complete_date": observed_requests_per_complete_date,
                "minimum_remaining_requests": minimum_remaining_requests,
                "estimated_remaining_requests": estimated_remaining_requests,
                "minimum_remaining_quota_days": math.ceil(minimum_remaining_requests / daily_request_budget),
                "estimated_remaining_quota_days": math.ceil(estimated_remaining_requests / daily_request_budget),
                "current_raw_storage_gib": raw_storage_bytes / 2**30,
                "observed_checkpoint_mib_per_page": observed_bytes_per_page / 2**20,
                "projected_additional_storage_gib": projected_additional_storage_bytes / 2**30,
                "free_disk_gib": free_disk_bytes / 2**30,
                "projected_free_after_completion_gib": projected_free_after_completion_bytes / 2**30,
            }
        ]
    )
    manifest_status = str(manifest.get("status") or "not_started")
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    anomalies = manifest.get("pagination_anomalies") if isinstance(manifest.get("pagination_anomalies"), list) else []
    anomaly_reasons = {str(item.get("reason") or "") for item in anomalies if isinstance(item, dict) and str(item.get("reason") or "")}
    unsafe_anomalies = anomaly_reasons - SAFE_PAGINATION_ANOMALY_REASONS
    headlines_path = raw_dir / "headlines.jsonl"
    stories_path = raw_dir / "stories.jsonl"
    completed_manifest = manifest_status == "completed"
    all_windows = len(completed) == expected_windows
    headline_file_ready = completed_manifest and headlines_path.is_file()
    headline_hash_matches = False
    story_hash_matches = False
    if completed_manifest and isinstance(manifest.get("files"), dict):
        files = manifest["files"]
        headline_entry = files.get("headlines_jsonl") if isinstance(files.get("headlines_jsonl"), dict) else {}
        story_entry = files.get("stories_jsonl") if isinstance(files.get("stories_jsonl"), dict) else {}
        headline_hash_matches = headline_file_ready and sha256_file(headlines_path) == headline_entry.get("sha256")
        story_hash_matches = stories_path.is_file() and sha256_file(stories_path) == story_entry.get("sha256")

    no_unsafe_anomalies = not unsafe_anomalies
    no_story_bodies = completed_manifest and int(counts.get("stories", -1)) == 0
    collection_gate_pass = all(
        [
            completed_manifest,
            all_windows,
            headline_hash_matches,
            story_hash_matches,
            no_unsafe_anomalies,
            no_story_bodies,
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "status": manifest_status,
                "companies": len(expected_symbols),
                "windows_per_company": window_count,
                "completed_windows": len(completed),
                "expected_windows": expected_windows,
                "window_completion_pct": 100.0 * len(completed) / expected_windows,
                "headline_pages": page_count,
                "complete_dates": complete_dates,
                "complete_date_prefix": complete_prefix_dates,
                "partial_dates": partial_dates,
                "untouched_dates": untouched_dates,
                "frontier_date": frontier["window_start"] if frontier is not None else None,
                "frontier_companies_completed": int(frontier["completed_companies"]) if frontier is not None else None,
                "unique_headlines": counts.get("headlines"),
                "raw_headline_rows": counts.get("raw_headline_rows"),
                "pagination_anomalies": len(anomalies),
                "unsafe_pagination_reasons": ",".join(sorted(unsafe_anomalies)),
                "started_at": manifest.get("started_at"),
                "completed_at": manifest.get("completed_at"),
            }
        ]
    )
    gate_table = pd.DataFrame(
        [
            {"gate": "manifest_completed", "passed": completed_manifest},
            {"gate": "all_company_windows_completed", "passed": all_windows},
            {"gate": "headline_file_hash_matches", "passed": headline_hash_matches},
            {"gate": "empty_story_file_hash_matches", "passed": story_hash_matches},
            {"gate": "no_unsafe_pagination_anomalies", "passed": no_unsafe_anomalies},
            {"gate": "headline_only_zero_stories", "passed": no_story_bodies},
            {"gate": "collection_ready_for_drift_audit", "passed": collection_gate_pass},
        ]
    )
    return BackwardCollectionAudit(
        summary=summary,
        date_progress=date_progress,
        projection=projection,
        gate_table=gate_table,
        collection_gate_pass=collection_gate_pass,
    )


def select_gemma_drift_sample(
    reference_csv: str | Path,
    *,
    expected_successes: int = 888_155,
    expected_endpoints: int = 774,
    nonendpoint_count: int = 1_000,
    seed_prefix: str = "20260806:",
) -> pd.DataFrame:
    """Select the frozen endpoint census plus deterministic non-endpoint controls."""

    path = Path(reference_csv)
    successes = 0
    seen: set[str] = set()
    endpoint_rows: list[dict[str, object]] = []
    nonendpoint_heap: list[tuple[int, str, dict[str, object]]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"headline_sha256", "headline", "score", "status"}
        _require(required.issubset(reader.fieldnames or []), "reference score CSV schema is incompatible")
        for line_number, row in enumerate(reader, start=2):
            if row.get("status") != "success":
                continue
            successes += 1
            headline_hash = str(row.get("headline_sha256") or "").strip()
            headline = str(row.get("headline") or "")
            _require(bool(headline_hash), f"successful reference row lacks a hash at line {line_number}")
            _require(headline_hash not in seen, f"duplicate successful reference hash at line {line_number}: {headline_hash}")
            seen.add(headline_hash)
            try:
                score = float(row.get("score") or "")
            except ValueError as exc:
                raise BackwardValidationError(f"invalid successful reference score at line {line_number}") from exc
            _require(np.isfinite(score) and -1.0 <= score <= 1.0, f"out-of-range reference score at line {line_number}")
            selected = {
                "headline_sha256": headline_hash,
                "headline": headline,
                "reference_score": score,
                "stratum": "endpoint" if score in {-1.0, 1.0} else "nonendpoint_control",
            }
            if score in {-1.0, 1.0}:
                endpoint_rows.append(selected)
                continue
            key = int(sha256_text(f"{seed_prefix}{headline_hash}"), 16)
            heap_item = (-key, headline_hash, selected)
            if len(nonendpoint_heap) < nonendpoint_count:
                heapq.heappush(nonendpoint_heap, heap_item)
            elif key < -nonendpoint_heap[0][0]:
                heapq.heapreplace(nonendpoint_heap, heap_item)

    _require(successes == expected_successes, f"expected {expected_successes:,} successful reference rows, found {successes:,}")
    _require(len(endpoint_rows) == expected_endpoints, f"expected {expected_endpoints:,} exact endpoints, found {len(endpoint_rows):,}")
    _require(len(nonendpoint_heap) == nonendpoint_count, "reference population has too few non-endpoint controls")
    controls = [item[2] for item in sorted(nonendpoint_heap, key=lambda item: (-item[0], item[1]))]
    sample = pd.DataFrame(endpoint_rows + controls).sort_values(["stratum", "headline_sha256"], kind="stable")
    return sample.reset_index(drop=True)


def evaluate_gemma_drift(
    reference_sample: pd.DataFrame,
    repeated_scores: pd.DataFrame,
    *,
    continuous_spearman_minimum: float = 0.95,
    continuous_mean_absolute_difference_maximum: float = 0.10,
    direction_agreement_minimum: float = 0.90,
    endpoint_same_direction_retention_minimum: float = 0.80,
    endpoint_opposite_direction_count_maximum: int = 0,
    nonendpoint_to_endpoint_rate_maximum: float = 0.05,
) -> GemmaDriftAudit:
    """Evaluate the frozen Gemma stability thresholds without any market data."""

    reference_required = {"headline_sha256", "reference_score", "stratum"}
    repeat_required = {"headline_sha256", "score", "status"}
    _require(reference_required.issubset(reference_sample.columns), "reference sample schema is incompatible")
    _require(repeat_required.issubset(repeated_scores.columns), "repeated-score schema is incompatible")
    repeats = repeated_scores.loc[repeated_scores["status"].eq("success")].copy()
    _require(not repeats["headline_sha256"].duplicated().any(), "repeated scores contain duplicate successes")
    _require(len(repeats) == len(reference_sample), "repeated-score coverage is incomplete")
    merged = reference_sample.merge(
        repeats[["headline_sha256", "score"]],
        on="headline_sha256",
        how="left",
        validate="one_to_one",
    )
    _require(merged["score"].notna().all(), "repeated scores do not cover every frozen sample hash")
    merged["reference_score"] = pd.to_numeric(merged["reference_score"], errors="coerce")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    _require(np.isfinite(merged[["reference_score", "score"]].to_numpy()).all(), "drift scores must be finite")
    _require(merged["score"].between(-1.0, 1.0).all(), "repeated scores must lie in [-1, 1]")

    endpoints = merged["stratum"].eq("endpoint")
    controls = merged["stratum"].eq("nonendpoint_control")
    _require(endpoints.any() and controls.any(), "both drift strata are required")
    reference_direction = np.sign(merged["reference_score"])
    repeated_direction = np.sign(merged["score"])
    spearman = float(merged["reference_score"].corr(merged["score"], method="spearman"))
    mean_absolute_difference = float((merged["reference_score"] - merged["score"]).abs().mean())
    direction_agreement = float((reference_direction == repeated_direction).mean())
    endpoint_same_direction_retention = float((merged.loc[endpoints, "reference_score"] == merged.loc[endpoints, "score"]).mean())
    endpoint_opposite_direction_count = int((reference_direction.loc[endpoints] == -repeated_direction.loc[endpoints]).sum())
    nonendpoint_to_endpoint_rate = float(merged.loc[controls, "score"].isin([-1.0, 1.0]).mean())

    gate_rows = [
        {
            "gate": "continuous_spearman",
            "value": spearman,
            "threshold": continuous_spearman_minimum,
            "passed": spearman >= continuous_spearman_minimum,
        },
        {
            "gate": "continuous_mean_absolute_difference",
            "value": mean_absolute_difference,
            "threshold": continuous_mean_absolute_difference_maximum,
            "passed": mean_absolute_difference <= continuous_mean_absolute_difference_maximum,
        },
        {
            "gate": "direction_agreement",
            "value": direction_agreement,
            "threshold": direction_agreement_minimum,
            "passed": direction_agreement >= direction_agreement_minimum,
        },
        {
            "gate": "endpoint_same_direction_retention",
            "value": endpoint_same_direction_retention,
            "threshold": endpoint_same_direction_retention_minimum,
            "passed": endpoint_same_direction_retention >= endpoint_same_direction_retention_minimum,
        },
        {
            "gate": "endpoint_opposite_direction_count",
            "value": endpoint_opposite_direction_count,
            "threshold": endpoint_opposite_direction_count_maximum,
            "passed": endpoint_opposite_direction_count <= endpoint_opposite_direction_count_maximum,
        },
        {
            "gate": "nonendpoint_to_endpoint_rate",
            "value": nonendpoint_to_endpoint_rate,
            "threshold": nonendpoint_to_endpoint_rate_maximum,
            "passed": nonendpoint_to_endpoint_rate <= nonendpoint_to_endpoint_rate_maximum,
        },
    ]
    gate_table = pd.DataFrame(gate_rows)
    metrics = pd.DataFrame(
        [
            {
                "sample_rows": len(merged),
                "endpoint_rows": int(endpoints.sum()),
                "nonendpoint_control_rows": int(controls.sum()),
                "continuous_spearman": spearman,
                "continuous_mean_absolute_difference": mean_absolute_difference,
                "direction_agreement": direction_agreement,
                "endpoint_same_direction_retention": endpoint_same_direction_retention,
                "endpoint_opposite_direction_count": endpoint_opposite_direction_count,
                "nonendpoint_to_endpoint_rate": nonendpoint_to_endpoint_rate,
            }
        ]
    )
    drift_gate_pass = bool(gate_table["passed"].all())
    return GemmaDriftAudit(metrics=metrics, gate_table=gate_table, drift_gate_pass=drift_gate_pass)
