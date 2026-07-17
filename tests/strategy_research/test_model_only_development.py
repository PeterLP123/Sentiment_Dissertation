from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from sentiment_benchmark.strategy_research.model_only_development import (
    MODEL_ONLY_CONTRACT,
    ModelOnlyDevelopmentError,
    build_model_only_filtered_scores,
    prepare_model_only_development,
    validate_model_only_scoring_universe,
)
from sentiment_benchmark.strategy_research.signal_quality import (
    DEFAULT_JOINT_PROMPT_PATH,
    JOINT_PROMPT_ID,
    JOINT_SIGNAL_SCHEMA,
    TASK_LABELS,
    TASKS,
    AuditItem,
    _expected_record_identity,
    load_joint_signal_prompt,
)


def _event(event_id: str, symbol: str, day: int, *, evaluation: bool = False) -> dict[str, object]:
    headline = f"{symbol} headline {event_id}"
    body = f"Target-specific body for {event_id}."
    timestamp = datetime(2026, 1 if not evaluation else 2, day, 12, tzinfo=UTC)
    return {
        "event_id": event_id,
        "story_family_id": f"family-{event_id}",
        "revision_id": f"revision-{event_id}",
        "symbol": symbol,
        "target_company_name": f"{symbol} Corp",
        "version_created_utc": (timestamp - timedelta(minutes=15)).isoformat(),
        "available_at_utc": timestamp.isoformat(),
        "source": "lseg",
        "headline": headline,
        "lead_or_body": body,
        "text_sha256": sha256_text(f"{headline}\n\n{body}"),
        "target_relevance": "include",
        "event_session": timestamp.date().isoformat(),
        "eligible_execution_session": timestamp.date().isoformat(),
        "exclusion_reason": None,
        "input_manifest_hash": "manifest",
    }


def _source_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "source-run"
    derived = tmp_path / "derived"
    events = [
        _event("AAA-1", "AAA", 1),
        _event("AAA-2", "AAA", 2),
        _event("BBB-1", "BBB", 3),
        _event("AAA-evaluation", "AAA", 2, evaluation=True),
    ]
    events_path = derived / run_dir.name / "events" / "events.jsonl"
    atomic_write_jsonl(events_path, events)
    report = {
        "manifest_schema_version": 1,
        "pipeline_schema_version": 1,
        "status": "completed",
        "run_id": run_dir.name,
        "run_identity_sha256": "source-identity",
    }
    config = {
        "run": {"evaluation_start": "2026-02-01"},
        "outputs": {"derived_root": derived.as_posix()},
        "scoring": {"model": "gemma4:12b", "model_digest": "frozen-digest"},
    }
    config_hash = sha256_text(canonical_json(config))
    root_manifest = {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_dir.name,
        "run_identity_sha256": report["run_identity_sha256"],
        "config": config,
        "config_sha256": config_hash,
        "input_identities": {},
        "outputs": {events_path.as_posix(): {"sha256": sha256_file(events_path)}},
    }
    atomic_write_json(run_dir / "manifest.json", root_manifest)
    report["config"] = config
    report["config_sha256"] = config_hash
    report["outputs"] = {"manifest": {"sha256": sha256_file(run_dir / "manifest.json")}}
    atomic_write_json(run_dir / "manifests" / "report.json", report)
    atomic_write_json(
        run_dir / "manifests" / "events.json",
        {
            "manifest_schema_version": 1,
            "pipeline_schema_version": 1,
            "status": "completed",
            "run_id": run_dir.name,
            "run_identity_sha256": report["run_identity_sha256"],
            "outputs": {"events": {"sha256": sha256_file(events_path)}},
        },
    )
    return run_dir


def _completed_scores(universe_dir: Path, labels: list[tuple[str, str, str]]) -> Path:
    universe_manifest = read_json(universe_dir / "sample_manifest.json")
    with (universe_dir / "audit_items.csv").open(encoding="utf-8", newline="") as handle:
        items = list(csv.DictReader(handle))
    score_dir = universe_dir / "scoring" / "test-scores"
    prompt = load_joint_signal_prompt(DEFAULT_JOINT_PROMPT_PATH)
    endpoint = "http://127.0.0.1:11435"
    rows = []
    for item_row, (direction, materiality, novelty) in zip(items, labels, strict=True):
        item = AuditItem.from_row(item_row)
        expected_record = _expected_record_identity(
            item,
            prompt,
            model_id="gemma4:12b",
            model_digest="frozen-digest",
            endpoint=endpoint,
        )
        rows.append(
            {
                **expected_record,
                "direction_severity": direction,
                "materiality": materiality,
                "novelty": novelty,
                "status": "success",
                "attempt_count": 1,
                "latency_ms": 5.0,
                "created_at_utc": "2026-07-17T12:00:00+00:00",
                "error": None,
            }
        )
    scores_path = score_dir / "model_scores.jsonl"
    atomic_write_jsonl(scores_path, rows)
    universe_identity = universe_manifest["identity"]
    score_identity = {
        "schema_version": 2,
        "sample_identity_sha256": universe_manifest["identity_sha256"],
        "audit_items_sha256": sha256_file(universe_dir / "audit_items.csv"),
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
    atomic_write_json(
        score_dir / "scores_manifest.json",
        {
            "schema_version": 1,
            "status": "completed",
            "identity_sha256": sha256_text(canonical_json(score_identity)),
            "identity": score_identity,
            "outputs": {
                scores_path.name: {
                    "path": scores_path.as_posix(),
                    "sha256": sha256_file(scores_path),
                }
            },
        },
    )
    return score_dir


def test_model_only_universe_is_all_development_events_and_immutable(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)

    first = prepare_model_only_development(run_dir)
    second = prepare_model_only_development(run_dir)

    assert first.event_count == 3
    assert second.reused
    manifest = read_json(first.manifest_path)
    assert manifest["identity"]["contract"] == MODEL_ONLY_CONTRACT
    assert manifest["identity"]["human_validation_status"] == "waived_by_researcher"
    assert "not evidence that the labels are ground truth" in first.assumption_path.read_text(encoding="utf-8")
    with first.items_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["event_id"] for row in rows] == ["AAA-1", "AAA-2", "BBB-1"]
    assert all(row["eligible_execution_session"] < "2026-02-01" for row in rows)
    assert rows[1]["prior_context"].count("AAA headline AAA-1") == 1
    assert "AAA headline AAA-2" not in rows[1]["prior_context"]


def test_model_only_universe_rejects_missing_execution_session(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    report = read_json(run_dir / "manifests" / "report.json")
    events_path = Path(report["config"]["outputs"]["derived_root"]) / run_dir.name / "events" / "events.jsonl"
    events = read_jsonl(events_path)
    events[0]["eligible_execution_session"] = ""
    atomic_write_jsonl(events_path, events)
    events_manifest_path = run_dir / "manifests" / "events.json"
    events_manifest = read_json(events_manifest_path)
    events_manifest["outputs"]["events"]["sha256"] = sha256_file(events_path)
    atomic_write_json(events_manifest_path, events_manifest)
    root_manifest_path = run_dir / "manifest.json"
    root_manifest = read_json(root_manifest_path)
    output_key = next(key for key in root_manifest["outputs"] if key.endswith("/events/events.jsonl"))
    root_manifest["outputs"][output_key]["sha256"] = sha256_file(events_path)
    atomic_write_json(root_manifest_path, root_manifest)
    report["outputs"]["manifest"]["sha256"] = sha256_file(root_manifest_path)
    atomic_write_json(run_dir / "manifests" / "report.json", report)

    with pytest.raises(ModelOnlyDevelopmentError, match="strategy events are invalid"):
        prepare_model_only_development(run_dir)


def test_model_only_universe_refuses_changed_events_stage(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    report = read_json(run_dir / "manifests" / "report.json")
    events_path = Path(report["config"]["outputs"]["derived_root"]) / run_dir.name / "events" / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ModelOnlyDevelopmentError, match="events stage.*changed"):
        prepare_model_only_development(run_dir)


def test_model_only_universe_refuses_incomplete_manifest_reuse(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    universe = prepare_model_only_development(run_dir)
    manifest = read_json(universe.manifest_path)
    manifest["status"] = "in_progress"
    atomic_write_json(universe.manifest_path, manifest)

    with pytest.raises(ModelOnlyDevelopmentError, match="incomplete model-only universe"):
        prepare_model_only_development(run_dir)


def test_model_only_scoring_validation_fails_before_changed_assumption_can_be_used(tmp_path: Path) -> None:
    universe = prepare_model_only_development(_source_run(tmp_path))
    validate_model_only_scoring_universe(universe.output_dir)
    with universe.assumption_path.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ModelOnlyDevelopmentError, match="scoring input changed"):
        validate_model_only_scoring_universe(universe.output_dir)


def test_model_only_scoring_validation_rejects_an_ordinary_audit_contract(tmp_path: Path) -> None:
    universe = prepare_model_only_development(_source_run(tmp_path))
    manifest = read_json(universe.manifest_path)
    manifest["identity"]["contract"] = "ordinary_signal_audit"
    manifest["identity_sha256"] = sha256_text(canonical_json(manifest["identity"]))
    atomic_write_json(universe.manifest_path, manifest)

    with pytest.raises(ModelOnlyDevelopmentError, match="not the completed model-only contract"):
        validate_model_only_scoring_universe(universe.output_dir)


def test_frozen_filter_maps_only_supported_non_neutral_events(tmp_path: Path) -> None:
    universe = prepare_model_only_development(_source_run(tmp_path))
    score_dir = _completed_scores(
        universe.output_dir,
        [
            ("positive", "moderate", "moderate"),
            ("negative", "low", "high"),
            ("neutral", "none", "high"),
        ],
    )

    first = build_model_only_filtered_scores(universe.output_dir, score_dir)
    second = build_model_only_filtered_scores(universe.output_dir, score_dir)

    assert first.event_count == 3
    assert first.eligible_event_count == 1
    assert second.reused
    rows = read_jsonl(first.scores_path)
    assert [(row["event_id"], row["score"], row["filter_reason"]) for row in rows] == [
        ("AAA-1", 1.0, "eligible"),
        ("AAA-2", 0.0, "materiality_below_moderate"),
        ("BBB-1", 0.0, "direction_neutral"),
    ]


def test_filtered_strategy_refuses_changed_completed_scores(tmp_path: Path) -> None:
    universe = prepare_model_only_development(_source_run(tmp_path))
    score_dir = _completed_scores(
        universe.output_dir,
        [("positive", "high", "high"), ("negative", "high", "high"), ("neutral", "none", "low")],
    )
    with (score_dir / "model_scores.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ModelOnlyDevelopmentError, match="scores changed"):
        build_model_only_filtered_scores(universe.output_dir, score_dir)


def test_filtered_strategy_revalidates_the_waiver_assumption(tmp_path: Path) -> None:
    universe = prepare_model_only_development(_source_run(tmp_path))
    score_dir = _completed_scores(
        universe.output_dir,
        [("positive", "high", "high"), ("negative", "high", "high"), ("neutral", "none", "low")],
    )
    with universe.assumption_path.open("a", encoding="utf-8") as handle:
        handle.write("changed after scoring\n")

    with pytest.raises(ModelOnlyDevelopmentError, match="scoring input changed"):
        build_model_only_filtered_scores(universe.output_dir, score_dir)


def test_filtered_strategy_rejects_a_noncanonical_score_identity(tmp_path: Path) -> None:
    universe = prepare_model_only_development(_source_run(tmp_path))
    score_dir = _completed_scores(
        universe.output_dir,
        [("positive", "high", "high"), ("negative", "high", "high"), ("neutral", "none", "low")],
    )
    manifest_path = score_dir / "scores_manifest.json"
    manifest = read_json(manifest_path)
    manifest["identity"]["unexpected_change"] = True
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ModelOnlyDevelopmentError, match="frozen scoring contract"):
        build_model_only_filtered_scores(universe.output_dir, score_dir)


def test_filtered_strategy_rejects_a_different_model_even_with_a_rehashed_identity(tmp_path: Path) -> None:
    universe = prepare_model_only_development(_source_run(tmp_path))
    score_dir = _completed_scores(
        universe.output_dir,
        [("positive", "high", "high"), ("negative", "high", "high"), ("neutral", "none", "low")],
    )
    manifest_path = score_dir / "scores_manifest.json"
    manifest = read_json(manifest_path)
    manifest["identity"]["model"] = "different-model:latest"
    manifest["identity"]["model_digest"] = "different-digest"
    manifest["identity_sha256"] = sha256_text(canonical_json(manifest["identity"]))
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ModelOnlyDevelopmentError, match="frozen scoring contract"):
        build_model_only_filtered_scores(universe.output_dir, score_dir)


def test_filtered_strategy_refuses_incomplete_manifest_reuse(tmp_path: Path) -> None:
    universe = prepare_model_only_development(_source_run(tmp_path))
    score_dir = _completed_scores(
        universe.output_dir,
        [("positive", "high", "high"), ("negative", "high", "high"), ("neutral", "none", "low")],
    )
    filtered = build_model_only_filtered_scores(universe.output_dir, score_dir)
    manifest = read_json(filtered.manifest_path)
    manifest["status"] = "in_progress"
    atomic_write_json(filtered.manifest_path, manifest)

    with pytest.raises(ModelOnlyDevelopmentError, match="incomplete filtered strategy"):
        build_model_only_filtered_scores(universe.output_dir, score_dir)
