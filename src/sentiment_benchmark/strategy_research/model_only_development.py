"""Frozen model-only development universe and filtered strategy artifacts.

This module records the explicit decision to treat local model annotations as a
research assumption rather than human-validated labels.  It never reads an
evaluation-period return and deliberately stops before evaluation scoring.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from .run_validation import (
    CompletedRunValidationError,
    load_completed_run_snapshot,
    validate_completed_stage_output,
)
from .schemas import load_strategy_events
from .signal_quality import (
    DEFAULT_JOINT_PROMPT_PATH,
    JOINT_PROMPT_ID,
    JOINT_SIGNAL_SCHEMA,
    TASK_LABELS,
    TASKS,
    AuditItem,
    JointSignalRecord,
    SignalAuditSpec,
    SignalQualityError,
    _assert_loopback_endpoint,
    _expected_record_identity,
    build_prior_context,
    load_joint_signal_prompt,
)

MODEL_ONLY_CONTRACT = "model_only_filtered_stock_strategy_v1"
MINIMUM_LEVEL = "moderate"
LEVELS = ("none", "low", "moderate", "high", "very_high")
DIRECTION_SCORE = {
    "very_negative": -1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
    "very_positive": 1.0,
}


class ModelOnlyDevelopmentError(RuntimeError):
    """Raised when the frozen model-only experiment cannot proceed safely."""


@dataclass(frozen=True)
class ModelOnlyUniverse:
    universe_id: str
    output_dir: Path
    items_path: Path
    assumption_path: Path
    manifest_path: Path
    event_count: int
    reused: bool


@dataclass(frozen=True)
class FilteredStrategyScores:
    strategy_id: str
    output_dir: Path
    scores_path: Path
    manifest_path: Path
    event_count: int
    eligible_event_count: int
    reused: bool


def _csv_rows(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _load_items(path: Path) -> tuple[AuditItem, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(AuditItem.from_row(row) for row in csv.DictReader(handle))
    if not rows or len({row.audit_id for row in rows}) != len(rows):
        raise ModelOnlyDevelopmentError("model-only items are empty or contain duplicate audit IDs")
    if len({row.event_id for row in rows}) != len(rows):
        raise ModelOnlyDevelopmentError("model-only items contain duplicate event IDs")
    return rows


def _assumption_text(universe_id: str, evaluation_start: str, event_count: int) -> str:
    return f"""# Model-only development assumption: {universe_id}

## Decision

The researcher explicitly accepts the frozen local model annotations as a modelling assumption without claiming
human validation. This is not evidence that the labels are ground truth or interchangeable with human judgement.

## Frozen boundary

- Development events: {event_count}
- Evaluation start: `{evaluation_start}`
- Only events whose eligible execution session is strictly before the boundary are present.
- No price, return, portfolio, or evaluation outcome enters the model prompt or scoring cache.
- Evaluation-period events remain unscored until the development controls produce a locked go decision.

## Frozen candidate rule

- Direction must be non-neutral.
- Materiality must be `moderate`, `high`, or `very_high`.
- Novelty must be `moderate`, `high`, or `very_high`.
- Eligible negative directions map to `-1`; eligible positive directions map to `+1`.
- Ineligible but successfully classified events map to an explicit filtered score of `0`; missing scores never do.
- Multiple events are averaged once per stock-session.
- The action threshold is `0.75`.
- Positions use a non-refreshing 10-session hold; same-direction news does not restart the clock.
- Costs are 10 bps per side and stocks are tested independently in development controls.

## Interpretation limit

This experiment can show whether the frozen model-defined filter improves development diagnostics. It cannot validate
the annotations, restore a pristine holdout, establish causality, or demonstrate deployable alpha.
"""


def validate_model_only_scoring_universe(universe_dir: str | Path) -> None:
    """Fail closed before the model-only command can make a local model call."""

    root = Path(universe_dir)
    manifest_path = root / "sample_manifest.json"
    items_path = root / "audit_items.csv"
    assumption_path = root / "model_only_assumption.md"
    if any(not path.is_file() for path in (manifest_path, items_path, assumption_path)):
        raise ModelOnlyDevelopmentError("model-only scoring universe is incomplete")
    manifest = read_json(manifest_path)
    identity = manifest.get("identity")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or not isinstance(identity, dict)
        or identity.get("contract") != MODEL_ONLY_CONTRACT
        or manifest.get("identity_sha256") != sha256_text(canonical_json(identity))
    ):
        raise ModelOnlyDevelopmentError("scoring input is not the completed model-only contract")
    for path in (items_path, assumption_path):
        expected_hash = manifest.get("outputs", {}).get(path.name, {}).get("sha256")
        if expected_hash != sha256_file(path):
            raise ModelOnlyDevelopmentError(f"completed model-only scoring input changed: {path.name}")


def prepare_model_only_development(
    run_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    prior_window_days: int = 30,
    maximum_prior_headlines: int = 5,
) -> ModelOnlyUniverse:
    """Freeze every development event into an audit-compatible scoring universe."""

    root = Path(run_dir)
    try:
        snapshot = load_completed_run_snapshot(root)
    except CompletedRunValidationError as exc:
        raise ModelOnlyDevelopmentError(str(exc)) from exc
    report = snapshot.report
    config = snapshot.config
    evaluation_start = str(config["run"]["evaluation_start"])
    derived_run = Path(str(config["outputs"]["derived_root"])) / root.name
    events_path = derived_run / "events" / "events.jsonl"
    if not events_path.is_file():
        raise ModelOnlyDevelopmentError(f"source event artifact is missing: {events_path}")
    try:
        validate_completed_stage_output(
            snapshot,
            stage="events",
            output_name="events",
            output_path=events_path,
            root_output_suffix=f"{root.name}/events/events.jsonl",
        )
    except CompletedRunValidationError as exc:
        raise ModelOnlyDevelopmentError(str(exc)) from exc

    spec = SignalAuditSpec(
        sample_per_symbol=1,
        prior_window_days=prior_window_days,
        maximum_prior_headlines=maximum_prior_headlines,
    )
    identity = {
        "schema_version": 1,
        "contract": MODEL_ONLY_CONTRACT,
        "source_run_id": root.name,
        "source_run_identity": report.get("run_identity_sha256"),
        "evaluation_start": evaluation_start,
        "events_sha256": sha256_file(events_path),
        "model": config["scoring"]["model"],
        "model_digest": config["scoring"]["model_digest"],
        "human_validation_status": "waived_by_researcher",
        "label_status": "model_generated_research_assumption",
        "prior_window_days": prior_window_days,
        "maximum_prior_headlines": maximum_prior_headlines,
        "selection": "all events with eligible_execution_session strictly before evaluation_start",
        "strategy_rule": {
            "minimum_materiality": MINIMUM_LEVEL,
            "minimum_novelty": MINIMUM_LEVEL,
            "direction_mapping": DIRECTION_SCORE,
            "stock_session_aggregation": "mean",
            "threshold": 0.75,
            "hold_sessions": 10,
            "same_direction_refresh": False,
            "cost_bps_per_side": 10.0,
        },
    }
    identity_hash = sha256_text(canonical_json(identity))
    universe_id = f"model-only-development-{identity_hash[:12]}"
    output_dir = Path(output_root) if output_root is not None else derived_run / "signal_quality" / universe_id
    items_path = output_dir / "audit_items.csv"
    assumption_path = output_dir / "model_only_assumption.md"
    manifest_path = output_dir / "sample_manifest.json"
    expected = (items_path, assumption_path)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("schema_version") != 1 or existing.get("status") != "completed":
            raise ModelOnlyDevelopmentError(f"refusing to reuse incomplete model-only universe: {output_dir}")
        if (
            existing.get("identity_sha256") != identity_hash
            or canonical_json(existing.get("identity")) != canonical_json(identity)
        ):
            raise ModelOnlyDevelopmentError(f"existing model-only universe identity mismatch: {output_dir}")
        for path in expected:
            expected_hash = existing.get("outputs", {}).get(path.name, {}).get("sha256")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ModelOnlyDevelopmentError(f"completed model-only universe output changed: {path}")
        return ModelOnlyUniverse(
            universe_id,
            output_dir,
            items_path,
            assumption_path,
            manifest_path,
            int(existing["row_counts"]["audit_items"]),
            True,
        )
    if output_dir.exists():
        raise ModelOnlyDevelopmentError(f"refusing to reuse incomplete model-only universe: {output_dir}")

    try:
        events = [event.to_payload() for event in load_strategy_events(events_path)]
    except ValueError as exc:
        raise ModelOnlyDevelopmentError(f"source strategy events are invalid: {exc}") from exc
    development = [dict(row) for row in events if str(row["eligible_execution_session"]) < evaluation_start]
    development.sort(key=lambda row: (str(row.get("available_at_utc") or ""), str(row.get("event_id") or "")))
    event_ids = [str(row.get("event_id") or "") for row in development]
    if not development or any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
        raise ModelOnlyDevelopmentError("development universe is empty or has invalid event identities")
    contexts = build_prior_context(events, event_ids, evaluation_start, spec)
    items: list[AuditItem] = []
    symbols: Counter[str] = Counter()
    for index, event in enumerate(development, start=1):
        event_id = str(event["event_id"])
        headline = str(event.get("headline") or "")
        body = str(event.get("lead_or_body") or "")
        current_text = "\n\n".join(value.strip() for value in (headline, body) if value.strip())
        if sha256_text(current_text) != str(event.get("text_sha256") or ""):
            raise ModelOnlyDevelopmentError(f"source event text hash mismatch: {event_id}")
        symbol = str(event.get("symbol") or "").upper()
        context = contexts[event_id]
        items.append(
            AuditItem(
                audit_id=f"MD-{index:05d}",
                event_id=event_id,
                symbol=symbol,
                target_company_name=str(event.get("target_company_name") or symbol),
                available_at_utc=str(event["available_at_utc"]),
                eligible_execution_session=str(event["eligible_execution_session"]),
                headline=headline,
                lead_or_body=body,
                prior_context=context,
                content_hash=str(event["text_sha256"]),
                context_hash=sha256_text(context),
            )
        )
        symbols[symbol] += 1

    output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(AuditItem.__dataclass_fields__)
    atomic_write_text(items_path, _csv_rows([item.to_row() for item in items], fields))
    atomic_write_text(assumption_path, _assumption_text(universe_id, evaluation_start, len(items)))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": universe_id,
        "identity_sha256": identity_hash,
        "identity": identity,
        "row_counts": {"audit_items": len(items), "symbols": len(symbols)},
        "symbol_counts": dict(sorted(symbols.items())),
        "outputs": {
            path.name: {"path": path.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in expected
        },
        "warning": "Model-only development assumption; human validation was explicitly waived and evaluation events are absent.",
    }
    atomic_write_json(manifest_path, manifest)
    return ModelOnlyUniverse(
        universe_id,
        output_dir,
        items_path,
        assumption_path,
        manifest_path,
        len(items),
        False,
    )


def _filtered_score(record: JointSignalRecord) -> tuple[float, str]:
    direction = record.direction_severity
    materiality = record.materiality
    novelty = record.novelty
    if direction not in DIRECTION_SCORE or materiality not in LEVELS or novelty not in LEVELS:
        raise ModelOnlyDevelopmentError(f"invalid completed model-only labels: {record.audit_id}")
    if direction == "neutral":
        return 0.0, "direction_neutral"
    if LEVELS.index(materiality) < LEVELS.index(MINIMUM_LEVEL):
        return 0.0, "materiality_below_moderate"
    if LEVELS.index(novelty) < LEVELS.index(MINIMUM_LEVEL):
        return 0.0, "novelty_below_moderate"
    return DIRECTION_SCORE[direction], "eligible"


def _validate_completed_score_contract(
    universe_manifest: Mapping[str, Any],
    items_path: Path,
    score_manifest: Mapping[str, Any],
    records: Sequence[JointSignalRecord],
    items: Sequence[AuditItem],
) -> None:
    universe_identity = universe_manifest["identity"]
    score_identity = score_manifest.get("identity")
    if not isinstance(score_identity, dict):
        raise ModelOnlyDevelopmentError("model-only scoring manifest has no canonical identity")
    try:
        endpoint = _assert_loopback_endpoint(str(score_identity.get("endpoint") or ""))
    except (SignalQualityError, ValueError) as exc:
        raise ModelOnlyDevelopmentError("model-only scoring identity is not local-only") from exc
    prompt = load_joint_signal_prompt(DEFAULT_JOINT_PROMPT_PATH)
    expected_identity = {
        "schema_version": 2,
        "sample_identity_sha256": universe_manifest["identity_sha256"],
        "audit_items_sha256": sha256_file(items_path),
        "model": universe_identity["model"],
        "model_digest": universe_identity["model_digest"],
        "endpoint": endpoint,
        "prompt_path": DEFAULT_JOINT_PROMPT_PATH.as_posix(),
        "prompt_file_sha256": sha256_file(DEFAULT_JOINT_PROMPT_PATH),
        "prompt_id": JOINT_PROMPT_ID,
        "prompt_hash": prompt.prompt_hash,
        "schema_sha256": sha256_text(canonical_json(JOINT_SIGNAL_SCHEMA)),
        "labels": {task: list(TASK_LABELS[task]) for task in TASKS},
        "call_contract": "one joint structured response per event",
        "request": {
            "temperature": 0.0,
            "max_completion_tokens": 64,
            "ollama_think": False,
            "structured_json_output": True,
            "retries": 3,
        },
    }
    if (
        canonical_json(score_identity) != canonical_json(expected_identity)
        or score_manifest.get("identity_sha256") != sha256_text(canonical_json(score_identity))
    ):
        raise ModelOnlyDevelopmentError("model-only scoring identity does not match the frozen scoring contract")
    item_by_id = {item.audit_id: item for item in items}
    for record in records:
        item = item_by_id.get(record.audit_id)
        if item is None:
            raise ModelOnlyDevelopmentError(f"model-only score has an unknown audit ID: {record.audit_id}")
        expected_record = _expected_record_identity(
            item,
            prompt,
            model_id=str(expected_identity["model"]),
            model_digest=str(expected_identity["model_digest"]),
            endpoint=endpoint,
        )
        if any(getattr(record, field) != value for field, value in expected_record.items()):
            raise ModelOnlyDevelopmentError(f"model-only score metadata violates the frozen contract: {record.audit_id}")


def build_model_only_filtered_scores(
    universe_dir: str | Path,
    score_dir: str | Path,
    *,
    output_root: str | Path | None = None,
) -> FilteredStrategyScores:
    """Map completed joint labels into the one predeclared filtered signal."""

    universe_root = Path(universe_dir)
    validate_model_only_scoring_universe(universe_root)
    universe_manifest = read_json(universe_root / "sample_manifest.json")
    items_path = universe_root / "audit_items.csv"
    expected_items_hash = universe_manifest.get("outputs", {}).get(items_path.name, {}).get("sha256")
    if (
        universe_manifest.get("schema_version") != 1
        or universe_manifest.get("status") != "completed"
        or not items_path.is_file()
    ):
        raise ModelOnlyDevelopmentError("model-only development universe is incomplete")
    if sha256_file(items_path) != expected_items_hash:
        raise ModelOnlyDevelopmentError("model-only development items changed")
    if universe_manifest.get("identity", {}).get("contract") != MODEL_ONLY_CONTRACT:
        raise ModelOnlyDevelopmentError("input universe is not the frozen model-only contract")

    scoring_root = Path(score_dir)
    score_manifest = read_json(scoring_root / "scores_manifest.json")
    model_scores_path = scoring_root / "model_scores.jsonl"
    expected_scores_hash = score_manifest.get("outputs", {}).get(model_scores_path.name, {}).get("sha256")
    if (
        score_manifest.get("schema_version") != 1
        or score_manifest.get("status") != "completed"
        or not model_scores_path.is_file()
    ):
        raise ModelOnlyDevelopmentError("model-only scoring manifest is incomplete or its identity changed")
    if sha256_file(model_scores_path) != expected_scores_hash:
        raise ModelOnlyDevelopmentError("completed model-only scores changed")
    if score_manifest.get("identity", {}).get("sample_identity_sha256") != universe_manifest.get("identity_sha256"):
        raise ModelOnlyDevelopmentError("model scores do not belong to the frozen development universe")

    items = _load_items(items_path)
    records = tuple(JointSignalRecord.from_payload(row) for row in read_jsonl(model_scores_path))
    _validate_completed_score_contract(universe_manifest, items_path, score_manifest, records, items)
    record_by_id = {record.audit_id: record for record in records}
    if len(records) != len(items) or len(record_by_id) != len(records):
        raise ModelOnlyDevelopmentError("completed model-only score cardinality is invalid")
    if set(record_by_id) != {item.audit_id for item in items}:
        raise ModelOnlyDevelopmentError("completed model-only score identities do not match the universe")

    identity = {
        "schema_version": 1,
        "contract": MODEL_ONLY_CONTRACT,
        "universe_identity_sha256": universe_manifest["identity_sha256"],
        "score_identity_sha256": score_manifest["identity_sha256"],
        "audit_items_sha256": sha256_file(items_path),
        "model_scores_sha256": sha256_file(model_scores_path),
        "minimum_materiality": MINIMUM_LEVEL,
        "minimum_novelty": MINIMUM_LEVEL,
        "direction_mapping": DIRECTION_SCORE,
        "ineligible_successful_score": 0.0,
        "missing_score_policy": "fail; never coerce to neutral",
    }
    identity_hash = sha256_text(canonical_json(identity))
    strategy_id = f"model-only-filtered-scores-{identity_hash[:12]}"
    output_dir = Path(output_root) if output_root is not None else universe_root / "filtered_strategy" / strategy_id
    scores_path = output_dir / "scores.jsonl"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("schema_version") != 1 or existing.get("status") != "completed":
            raise ModelOnlyDevelopmentError(f"refusing to reuse incomplete filtered strategy output: {output_dir}")
        if (
            existing.get("identity_sha256") != identity_hash
            or canonical_json(existing.get("identity")) != canonical_json(identity)
        ):
            raise ModelOnlyDevelopmentError(f"existing filtered strategy identity mismatch: {output_dir}")
        expected_hash = existing.get("outputs", {}).get(scores_path.name, {}).get("sha256")
        if not scores_path.is_file() or sha256_file(scores_path) != expected_hash:
            raise ModelOnlyDevelopmentError("completed filtered strategy scores are missing or changed")
        return FilteredStrategyScores(
            strategy_id,
            output_dir,
            scores_path,
            manifest_path,
            int(existing["row_counts"]["events"]),
            int(existing["row_counts"]["eligible_events"]),
            True,
        )
    if output_dir.exists():
        raise ModelOnlyDevelopmentError(f"refusing to reuse incomplete filtered strategy output: {output_dir}")

    output_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for item in items:
        record = record_by_id[item.audit_id]
        if record.status != "success":
            raise ModelOnlyDevelopmentError(f"model-only score is not successful: {item.audit_id}")
        score, reason = _filtered_score(record)
        reasons[reason] += 1
        output_rows.append(
            {
                "event_id": item.event_id,
                "status": "success",
                "score": score,
                "raw_label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
                "model_direction_severity": record.direction_severity,
                "model_materiality": record.materiality,
                "model_novelty": record.novelty,
                "eligible": reason == "eligible",
                "filter_reason": reason,
                "model_id": record.model_id,
                "model_digest": record.model_digest,
                "prompt_id": record.prompt_id,
                "prompt_hash": record.prompt_hash,
                "source_cache_key": record.cache_key,
            }
        )
    output_rows.sort(key=lambda row: str(row["event_id"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_jsonl(scores_path, output_rows)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "strategy_id": strategy_id,
        "identity_sha256": identity_hash,
        "identity": identity,
        "row_counts": {
            "events": len(output_rows),
            "eligible_events": reasons["eligible"],
            "filtered_events": len(output_rows) - reasons["eligible"],
        },
        "filter_reasons": dict(sorted(reasons.items())),
        "outputs": {
            scores_path.name: {
                "path": scores_path.as_posix(),
                "sha256": sha256_file(scores_path),
                "size_bytes": scores_path.stat().st_size,
            }
        },
        "warning": "Development-only model-defined signal; not human validated and not an evaluation result.",
    }
    atomic_write_json(manifest_path, manifest)
    return FilteredStrategyScores(
        strategy_id,
        output_dir,
        scores_path,
        manifest_path,
        len(output_rows),
        reasons["eligible"],
        False,
    )
