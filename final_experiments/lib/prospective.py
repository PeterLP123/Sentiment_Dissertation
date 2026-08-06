"""Fail-closed population audit for the frozen prospective LSEG replay.

The registry is append-only and source controlled; licensed LSEG headline files
remain local and ignored.  This module streams those files, retains only story
IDs and normalized-headline hashes, and returns aggregate audit tables.  It
never invokes a scorer, loads prices, or computes returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from final_experiments.lib.novelty import headline_norm_sha256, normalize_headline
from sentiment_benchmark.artifact_io import sha256_file, sha256_text
from sentiment_benchmark.lseg_source import load_lseg_collection_config

SAFE_PAGINATION_ANOMALY_REASONS = frozenset({"repeated_cursor_duplicate_page"})


class ProspectivePopulationError(ValueError):
    """The frozen prospective source contract or population audit failed."""


@dataclass(frozen=True)
class ProspectivePopulationAudit:
    """Licence-safe aggregate evidence from one append-only batch registry."""

    batch_audit: pd.DataFrame
    cumulative_audit: pd.DataFrame
    gate_table: pd.DataFrame
    source_hashes: pd.DataFrame
    population_sha256: str
    story_population_sha256: str
    earliest_required_session: str
    calendar_gate_pass: bool


@dataclass(frozen=True)
class _ScannedBatch:
    story_ids: frozenset[str]
    eligible_story_ids: frozenset[str]
    headline_hashes: frozenset[str]
    eligible_headline_hashes: frozenset[str]
    matched_symbols: frozenset[str]
    headline_rows: int
    duplicate_story_ids: int
    duplicate_normalized_headlines: int
    empty_normalized_headline_rows: int
    strictly_post_cutoff_headlines: int
    exact_cutoff_headlines: int
    before_cutoff_headlines: int
    first_event_timestamp_utc: str
    last_event_timestamp_utc: str


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectivePopulationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ProspectivePopulationError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve(repo_root: Path, value: object, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ProspectivePopulationError(f"{label} path is missing")
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _utc(value: object, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ProspectivePopulationError(f"invalid {label}: {value!r}") from exc
    if pd.isna(timestamp):
        raise ProspectivePopulationError(f"invalid {label}: {value!r}")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProspectivePopulationError(message)


def _expected_symbols(parent_spec: dict[str, Any]) -> frozenset[str]:
    sectors = parent_spec.get("universe", {}).get("sectors")
    _require(isinstance(sectors, dict) and sectors, "parent universe sectors are missing")
    symbols = {str(symbol).strip().upper() for members in sectors.values() for symbol in (members if isinstance(members, list) else [])}
    expected_count = int(parent_spec.get("universe", {}).get("companies", 0))
    _require(len(symbols) == expected_count, "parent universe company count is inconsistent")
    return frozenset(symbols)


def _scan_headlines(
    path: Path,
    *,
    batch_start: pd.Timestamp,
    batch_end: pd.Timestamp,
    prospective_cutoff: pd.Timestamp,
    expected_symbols: frozenset[str],
) -> _ScannedBatch:
    story_ids: set[str] = set()
    eligible_story_ids: set[str] = set()
    headline_hashes: set[str] = set()
    eligible_headline_hashes: set[str] = set()
    matched_symbols: set[str] = set()
    duplicate_story_ids = 0
    duplicate_normalized_headlines = 0
    empty_normalized_headline_rows = 0
    strictly_post_cutoff = 0
    exact_cutoff = 0
    before_cutoff = 0
    timestamps: list[pd.Timestamp] = []

    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ProspectivePopulationError(f"cannot read headline source: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProspectivePopulationError(f"invalid headline JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ProspectivePopulationError(f"headline JSONL row is not an object at {path}:{line_number}")

            story_id = str(row.get("story_id") or "").strip()
            if not story_id:
                raise ProspectivePopulationError(f"headline row is missing story_id at {path}:{line_number}")
            if story_id in story_ids:
                duplicate_story_ids += 1
            story_ids.add(story_id)

            normalized = normalize_headline(str(row.get("headline") or ""))
            headline_hash: str | None = None
            if not normalized:
                empty_normalized_headline_rows += 1
            else:
                headline_hash = headline_norm_sha256(normalized)
                if headline_hash in headline_hashes:
                    duplicate_normalized_headlines += 1
                headline_hashes.add(headline_hash)

            row_symbols = {str(symbol).strip().upper() for symbol in (row.get("matched_symbols") or []) if str(symbol).strip()}
            unknown_symbols = row_symbols - expected_symbols
            if unknown_symbols:
                raise ProspectivePopulationError(f"headline row contains symbols outside the frozen universe: {sorted(unknown_symbols)}")
            matched_symbols.update(row_symbols)

            event_value = row.get("first_created") or row.get("version_created")
            event_timestamp = _utc(event_value, label="headline event timestamp")
            if event_timestamp < batch_start or event_timestamp >= batch_end:
                raise ProspectivePopulationError(
                    "headline event timestamp falls outside its frozen batch interval: "
                    f"{event_timestamp.isoformat()} not in "
                    f"[{batch_start.isoformat()}, {batch_end.isoformat()})"
                )
            timestamps.append(event_timestamp)
            if event_timestamp > prospective_cutoff:
                strictly_post_cutoff += 1
                eligible_story_ids.add(story_id)
                if headline_hash is not None:
                    eligible_headline_hashes.add(headline_hash)
            elif event_timestamp == prospective_cutoff:
                exact_cutoff += 1
            else:
                before_cutoff += 1

    if not timestamps:
        raise ProspectivePopulationError(f"headline source is empty: {path}")
    return _ScannedBatch(
        story_ids=frozenset(story_ids),
        eligible_story_ids=frozenset(eligible_story_ids),
        headline_hashes=frozenset(headline_hashes),
        eligible_headline_hashes=frozenset(eligible_headline_hashes),
        matched_symbols=frozenset(matched_symbols),
        headline_rows=len(timestamps),
        duplicate_story_ids=duplicate_story_ids,
        duplicate_normalized_headlines=duplicate_normalized_headlines,
        empty_normalized_headline_rows=empty_normalized_headline_rows,
        strictly_post_cutoff_headlines=strictly_post_cutoff,
        exact_cutoff_headlines=exact_cutoff,
        before_cutoff_headlines=before_cutoff,
        first_event_timestamp_utc=min(timestamps).isoformat(),
        last_event_timestamp_utc=max(timestamps).isoformat(),
    )


def _calendar_gate(*, cutoff: pd.Timestamp, end_exclusive: pd.Timestamp, required_sessions: int) -> tuple[int, pd.Timestamp]:
    search_end = max(
        pd.Timestamp(end_exclusive.to_pydatetime() + timedelta(days=7)),
        pd.Timestamp(cutoff.to_pydatetime() + timedelta(days=max(730, required_sessions * 4))),
    )
    calendar = xcals.get_calendar("XNYS", start=cutoff.date(), end=search_end.date())
    sessions = calendar.sessions
    cutoff_date = cutoff.tz_localize(None).normalize()
    end_date = end_exclusive.tz_localize(None).normalize()
    post_cutoff = sessions[sessions > cutoff_date]
    _require(
        len(post_cutoff) >= required_sessions,
        "exchange calendar does not extend to the frozen session minimum",
    )
    eligible = post_cutoff[post_cutoff < end_date]
    return len(eligible), post_cutoff[required_sessions - 1]


def audit_prospective_registry(
    repo_root: str | Path,
    registry_path: str | Path,
) -> ProspectivePopulationAudit:
    """Audit every registered batch without opening scores, prices, or returns."""

    root = Path(repo_root).resolve()
    registry_file = _resolve(root, registry_path, label="registry")
    registry = _read_object(registry_file, label="batch registry")
    _require(registry.get("status") == "append_only", "batch registry is not append-only")

    parent_entry = registry.get("parent_strategy_spec")
    _require(isinstance(parent_entry, dict), "registry parent strategy spec is missing")
    parent_path = _resolve(root, parent_entry.get("path"), label="parent strategy spec")
    parent_hash = sha256_file(parent_path)
    _require(
        parent_hash == parent_entry.get("sha256"),
        "parent strategy spec hash does not match the registry",
    )
    parent_spec = _read_object(parent_path, label="parent strategy spec")
    _require(
        parent_spec.get("status") == "frozen_awaiting_genuinely_new_lseg_dates",
        "parent strategy spec is not frozen for prospective dates",
    )
    cutoff = _utc(
        parent_spec.get("claim_boundary", {}).get("prospective_start_utc"),
        label="prospective cutoff",
    )
    expected_symbols = _expected_symbols(parent_spec)

    batch_entries = registry.get("batches")
    _require(isinstance(batch_entries, list) and batch_entries, "registry has no batches")
    batch_rows: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = [
        {"path": str(registry_file.relative_to(root)), "sha256": sha256_file(registry_file)},
        {"path": str(parent_path.relative_to(root)), "sha256": parent_hash},
    ]
    seen_story_ids: set[str] = set()
    seen_eligible_story_ids: set[str] = set()
    seen_headline_hashes: set[str] = set()
    seen_eligible_headline_hashes: set[str] = set()
    seen_symbols: set[str] = set()
    previous_end: pd.Timestamp | None = None
    last_end: pd.Timestamp | None = None

    for expected_order, entry in enumerate(batch_entries, start=1):
        _require(isinstance(entry, dict), f"batch {expected_order} registry entry is invalid")
        _require(
            int(entry.get("order", -1)) == expected_order,
            "batch registry order must be contiguous and one-based",
        )
        acquisition_path = _resolve(root, entry.get("acquisition_spec_path"), label="acquisition spec")
        acquisition_hash = sha256_file(acquisition_path)
        _require(
            acquisition_hash == entry.get("acquisition_spec_sha256"),
            f"acquisition spec hash mismatch for batch {expected_order}",
        )
        acquisition = _read_object(acquisition_path, label="acquisition spec")
        _require(
            acquisition.get("status") == "frozen_before_retrieval",
            f"batch {expected_order} was not frozen before retrieval",
        )
        _require(
            acquisition.get("parent_strategy_spec", {}).get("sha256") == parent_hash,
            f"batch {expected_order} does not reference the frozen parent hash",
        )
        _require(
            acquisition.get("data_separation", {}).get("pool_with_fnspid") is False
            and acquisition.get("data_separation", {}).get("pool_with_opened_lseg_window") is False,
            f"batch {expected_order} violates source separation",
        )
        _require(
            acquisition.get("scoring_authorisation", {}).get("paid_external_gemma_scoring_permitted_by_this_spec") is False,
            f"batch {expected_order} unexpectedly authorises paid Gemma scoring",
        )

        config_entry = acquisition.get("collection_config")
        _require(isinstance(config_entry, dict), f"batch {expected_order} config entry is missing")
        config_path = _resolve(root, config_entry.get("path"), label="collection config")
        _require(
            sha256_file(config_path) == config_entry.get("file_sha256"),
            f"collection config file hash mismatch for batch {expected_order}",
        )
        config = load_lseg_collection_config(config_path)
        _require(
            config.config_sha256 == config_entry.get("parsed_config_sha256"),
            f"parsed collection config hash mismatch for batch {expected_order}",
        )
        _require(
            config.collection_id == config_entry.get("collection_id"),
            f"collection id mismatch for batch {expected_order}",
        )
        _require(config.fetch_story_bodies is False, "prospective collection requested story bodies")
        _require(
            {company.symbol.upper() for company in config.companies} == expected_symbols,
            f"collection universe drift in batch {expected_order}",
        )

        interval = acquisition.get("closed_interval")
        _require(isinstance(interval, dict), f"batch {expected_order} interval is missing")
        batch_start = _utc(interval.get("start_inclusive_utc"), label="batch start")
        batch_end = _utc(interval.get("end_exclusive_utc"), label="batch end")
        _require(batch_start < batch_end, f"batch {expected_order} interval is empty")
        _require(
            batch_start == _utc(config.start, label="config start") and batch_end == _utc(config.end, label="config end"),
            f"batch {expected_order} spec and config intervals differ",
        )
        if previous_end is None:
            _require(batch_start == cutoff, "first batch must begin at the prospective cutoff")
        else:
            _require(
                batch_start == previous_end,
                "prospective batches must be contiguous with no gaps or interval overlap",
            )
        previous_end = batch_end
        last_end = batch_end

        raw_dir = root / config.raw_dir
        raw_manifest_path = raw_dir / "manifest.json"
        headlines_path = raw_dir / "headlines.jsonl"
        raw_manifest = _read_object(raw_manifest_path, label="raw collection manifest")
        _require(
            raw_manifest.get("status") == "completed",
            f"raw collection is not complete for batch {expected_order}",
        )
        _require(
            raw_manifest.get("config_sha256") == config.config_sha256,
            f"raw manifest config hash mismatch for batch {expected_order}",
        )
        counts = raw_manifest.get("counts", {})
        _require(counts.get("fetch_story_bodies") is False, "raw manifest reports story retrieval")
        _require(
            int(counts.get("stories", -1)) == 0 and int(counts.get("failed_stories", -1)) == 0,
            "prospective batch contains story requests or failures",
        )
        anomalies = raw_manifest.get("pagination_anomalies") or []
        unexpected_anomalies = {str(item.get("reason")) for item in anomalies if isinstance(item, dict)} - SAFE_PAGINATION_ANOMALY_REASONS
        _require(
            not unexpected_anomalies,
            f"unexpected pagination anomalies: {sorted(unexpected_anomalies)}",
        )
        expected_headline_hash = raw_manifest.get("files", {}).get("headlines_jsonl", {}).get("sha256")
        _require(
            sha256_file(headlines_path) == expected_headline_hash,
            f"headline source hash mismatch for batch {expected_order}",
        )

        scanned = _scan_headlines(
            headlines_path,
            batch_start=batch_start,
            batch_end=batch_end,
            prospective_cutoff=cutoff,
            expected_symbols=expected_symbols,
        )
        _require(
            scanned.headline_rows == int(counts.get("headlines", -1)),
            f"headline row count mismatch for batch {expected_order}",
        )
        _require(
            scanned.duplicate_story_ids == 0,
            f"duplicate story IDs within batch {expected_order}",
        )
        _require(
            scanned.before_cutoff_headlines == 0,
            f"batch {expected_order} contains pre-cutoff headlines",
        )

        overlap_story_ids = scanned.story_ids & seen_story_ids
        overlap_headline_hashes = scanned.headline_hashes & seen_headline_hashes
        overlap_eligible_story_ids = scanned.eligible_story_ids & seen_eligible_story_ids
        overlap_eligible_headline_hashes = scanned.eligible_headline_hashes & seen_eligible_headline_hashes
        new_eligible_story_ids = scanned.eligible_story_ids - seen_eligible_story_ids
        new_eligible_headline_hashes = scanned.eligible_headline_hashes - seen_eligible_headline_hashes
        seen_story_ids.update(scanned.story_ids)
        seen_eligible_story_ids.update(scanned.eligible_story_ids)
        seen_headline_hashes.update(scanned.headline_hashes)
        seen_eligible_headline_hashes.update(scanned.eligible_headline_hashes)
        seen_symbols.update(scanned.matched_symbols)

        source_rows.extend(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
            for path in (acquisition_path, config_path, raw_manifest_path, headlines_path)
        )
        batch_rows.append(
            {
                "batch_order": expected_order,
                "collection_id": config.collection_id,
                "start_inclusive_utc": batch_start.isoformat(),
                "end_exclusive_utc": batch_end.isoformat(),
                "headline_rows": scanned.headline_rows,
                "unique_story_ids": len(scanned.story_ids),
                "unique_normalized_headlines": len(scanned.headline_hashes),
                "eligible_unique_story_ids": len(scanned.eligible_story_ids),
                "eligible_unique_normalized_headlines": len(scanned.eligible_headline_hashes),
                "duplicate_normalized_headline_rows": scanned.duplicate_normalized_headlines,
                "empty_normalized_headline_rows": scanned.empty_normalized_headline_rows,
                "cross_batch_story_id_overlap": len(overlap_story_ids),
                "cross_batch_normalized_headline_overlap": len(overlap_headline_hashes),
                "cross_batch_eligible_story_id_overlap": len(overlap_eligible_story_ids),
                "cross_batch_eligible_normalized_headline_overlap": len(overlap_eligible_headline_hashes),
                "new_eligible_unique_story_ids": len(new_eligible_story_ids),
                "new_eligible_unique_normalized_headlines": len(new_eligible_headline_hashes),
                "matched_symbols": len(scanned.matched_symbols),
                "strictly_post_cutoff_headlines": scanned.strictly_post_cutoff_headlines,
                "exact_cutoff_headlines": scanned.exact_cutoff_headlines,
                "first_event_timestamp_utc": scanned.first_event_timestamp_utc,
                "last_event_timestamp_utc": scanned.last_event_timestamp_utc,
                "story_bodies_requested": False,
                "unexpected_pagination_anomalies": 0,
                "licensed_headline_text_exported": False,
            }
        )

        required_sessions = int(parent_spec["minimum_evaluation_population"]["complete_price_calendar_sessions"])
        calendar_upper_bound, _ = _calendar_gate(
            cutoff=cutoff,
            end_exclusive=batch_end,
            required_sessions=required_sessions,
        )
        cumulative_rows.append(
            {
                "through_batch_order": expected_order,
                "through_end_exclusive_utc": batch_end.isoformat(),
                "calendar_session_upper_bound": calendar_upper_bound,
                "calendar_sessions_required": required_sessions,
                "calendar_gate_pass": calendar_upper_bound >= required_sessions,
                "headline_rows_cumulative": sum(row["headline_rows"] for row in batch_rows),
                "unique_story_ids_cumulative": len(seen_story_ids),
                "unique_normalized_headlines_cumulative": len(seen_headline_hashes),
                "eligible_unique_story_ids_cumulative": len(seen_eligible_story_ids),
                "eligible_unique_normalized_headlines_cumulative": len(seen_eligible_headline_hashes),
                "matched_symbols_cumulative": len(seen_symbols),
            }
        )

    _require(last_end is not None, "registry has no completed batch interval")
    required_sessions = int(parent_spec["minimum_evaluation_population"]["complete_price_calendar_sessions"])
    required_active = int(parent_spec["minimum_evaluation_population"]["primary_active_sessions"])
    required_active_half = int(parent_spec["minimum_evaluation_population"]["active_sessions_each_chronological_half"])
    calendar_upper_bound, earliest_required = _calendar_gate(
        cutoff=cutoff,
        end_exclusive=last_end,
        required_sessions=required_sessions,
    )
    calendar_pass = calendar_upper_bound >= required_sessions
    gate_table = pd.DataFrame(
        [
            {
                "gate": "complete_price_calendar_sessions",
                "required": required_sessions,
                "observed_or_upper_bound": calendar_upper_bound,
                "status": "PASS_TO_SCORING_STAGE" if calendar_pass else "FAIL_BY_CALENDAR_UPPER_BOUND",
                "reason": (
                    "Population may advance to authorised scoring and complete-price checks."
                    if calendar_pass
                    else "No scores, prices, or returns loaded; continue immutable accumulation."
                ),
            },
            {
                "gate": "primary_active_sessions",
                "required": required_active,
                "observed_or_upper_bound": pd.NA,
                "status": "NOT_EVALUATED_BEFORE_SCORING",
                "reason": "Requires frozen Gemma and FinBERT event construction.",
            },
            {
                "gate": "active_sessions_each_chronological_half",
                "required": required_active_half,
                "observed_or_upper_bound": pd.NA,
                "status": "NOT_EVALUATED_BEFORE_SCORING",
                "reason": "Requires the scored primary strategy schedule.",
            },
        ]
    )
    source_hashes = (
        pd.DataFrame(source_rows).drop_duplicates("path", keep="last").sort_values("path", kind="mergesort").reset_index(drop=True)
    )
    return ProspectivePopulationAudit(
        batch_audit=pd.DataFrame(batch_rows),
        cumulative_audit=pd.DataFrame(cumulative_rows),
        gate_table=gate_table,
        source_hashes=source_hashes,
        population_sha256=sha256_text("\n".join(sorted(seen_eligible_headline_hashes))),
        story_population_sha256=sha256_text("\n".join(sorted(seen_eligible_story_ids))),
        earliest_required_session=str(earliest_required.date()),
        calendar_gate_pass=calendar_pass,
    )


__all__ = [
    "ProspectivePopulationAudit",
    "ProspectivePopulationError",
    "audit_prospective_registry",
]
