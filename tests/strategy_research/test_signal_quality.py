from __future__ import annotations

import asyncio
import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import sentiment_benchmark.strategy_research.signal_quality as signal_quality
from sentiment_benchmark.artifact_io import atomic_write_jsonl, sha256_file, sha256_text
from sentiment_benchmark.models import ModelConfig, StructuredJSONResponseRecord
from sentiment_benchmark.strategy_research.signal_quality import (
    JOINT_SIGNAL_SCHEMA,
    SignalAuditSpec,
    SignalQualityError,
    build_prior_context,
    load_joint_signal_prompt,
    parse_joint_signal_labels,
    prepare_signal_audit,
    score_signal_audit,
    select_audit_events,
    validate_signal_audit,
)


def _event(event_id: str, symbol: str, label: str, day: int, *, evaluation: bool = False) -> tuple[dict, dict]:
    headline = f"{symbol} {label} development {event_id}"
    body = f"Target-specific body for {event_id}."
    timestamp = datetime(2026, 1 if not evaluation else 2, day, 12, tzinfo=UTC)
    event = {
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
    score = {"event_id": event_id, "raw_label": label, "status": "success", "score": 0.0}
    return event, score


def _source_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "source-run"
    derived = tmp_path / "derived"
    event_rows: list[dict] = []
    score_rows: list[dict] = []
    for symbol in ("AAA", "BBB"):
        for index, label in enumerate(("negative", "neutral", "positive"), start=1):
            event, score = _event(f"{symbol}-{label}", symbol, label, index)
            event_rows.append(event)
            score_rows.append(score)
        event, score = _event(f"{symbol}-evaluation", symbol, "positive", 2, evaluation=True)
        event_rows.append(event)
        score_rows.append(score)
    derived_run = derived / run_dir.name
    atomic_write_jsonl(derived_run / "events" / "events.jsonl", event_rows)
    atomic_write_jsonl(derived_run / "scores" / "scores.jsonl", score_rows)
    report = {
        "status": "completed",
        "run_identity_sha256": "source-identity",
        "config": {
            "run": {"evaluation_start": "2026-02-01"},
            "outputs": {"derived_root": derived.as_posix()},
            "scoring": {"model": "gemma4:12b", "model_digest": "frozen-digest"},
        },
    }
    (run_dir / "manifests").mkdir(parents=True)
    (run_dir / "manifests" / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_dir


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_models(self):
        return [
            ModelConfig(
                model_id="gemma4:12b",
                name="gemma4:12b",
                raw_metadata={"digest": "frozen-digest"},
            )
        ]

    async def generate_structured_json(self, model_id, **kwargs):
        self.calls.append((model_id, kwargs))
        even = (kwargs["row_number"] - 1) % 2 == 0
        labels = {
            "direction_severity": "negative" if even else "positive",
            "materiality": "high" if even else "very_high",
            "novelty": "moderate" if even else "high",
        }
        return StructuredJSONResponseRecord(
            row_number=kwargs["row_number"],
            model_id=model_id,
            prompt_hash=kwargs["prompt_hash"],
            raw_content=json.dumps(labels, sort_keys=True),
            parsed_json=labels,
            parse_status="valid",
            status="success",
            latency_ms=5.0,
            attempt_count=1,
        )


class MismatchedOwnedClient(FakeOllamaClient):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def list_models(self):
        models = await super().list_models()
        models[0].raw_metadata["digest"] = "different-digest"
        return models

    async def close(self) -> None:
        self.closed = True


def test_sampling_is_deterministic_balanced_and_development_only() -> None:
    events: list[dict] = []
    scores: list[dict] = []
    for symbol in ("AAA", "BBB"):
        for index, label in enumerate(("negative", "neutral", "positive"), start=1):
            event, score = _event(f"{symbol}-{label}", symbol, label, index)
            events.append(event)
            scores.append(score)
        event, score = _event(f"{symbol}-evaluation", symbol, "positive", 2, evaluation=True)
        events.append(event)
        scores.append(score)
    spec = SignalAuditSpec(sample_per_symbol=3, seed=7)

    first = select_audit_events(events, scores, "2026-02-01", spec)
    second = select_audit_events(events, scores, "2026-02-01", spec)

    assert first == second
    assert len(first) == 6
    assert {label for _, label in first} == {"negative", "neutral", "positive"}
    assert all(event["eligible_execution_session"] < "2026-02-01" for event, _ in first)


def test_prior_context_is_strictly_earlier_and_windowed() -> None:
    old, _ = _event("old", "AAA", "neutral", 1)
    current, _ = _event("current", "AAA", "positive", 3)
    same_time = dict(current, event_id="same-time", headline="Must not appear")
    future, _ = _event("future", "AAA", "positive", 4)

    context = build_prior_context(
        [old, current, same_time, future],
        ["current"],
        "2026-02-01",
        SignalAuditSpec(prior_window_days=30),
    )["current"]

    assert old["headline"] in context
    assert "Must not appear" not in context
    assert future["headline"] not in context


def test_prepare_packet_is_blind_immutable_and_has_strict_prompts(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    first = prepare_signal_audit(run_dir, spec=SignalAuditSpec(sample_per_symbol=2, seed=9))
    before = sha256_file(first.items_path)
    second = prepare_signal_audit(run_dir, spec=SignalAuditSpec(sample_per_symbol=2, seed=9))

    assert first.item_count == 4
    assert second.reused
    assert sha256_file(second.items_path) == before
    with first.items_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "raw_label" not in (reader.fieldnames or [])
        assert "source_label" not in (reader.fieldnames or [])
        assert all(row["eligible_execution_session"] < "2026-02-01" for row in reader)
    prompt = load_joint_signal_prompt()
    assert prompt.prompt_id == "strategy_target_signal_quality_joint_v2"
    assert "predict a stock return" in prompt.system_prompt
    assert "materiality=none requires direction_severity=neutral" in prompt.system_prompt
    assert set(JOINT_SIGNAL_SCHEMA["required"]) == {
        "direction_severity",
        "materiality",
        "novelty",
    }


def test_joint_label_parser_enforces_cross_field_consistency() -> None:
    valid = parse_joint_signal_labels(
        {"direction_severity": "neutral", "materiality": "none", "novelty": "high"}
    )
    assert valid["novelty"] == "high"
    with pytest.raises(ValueError, match="materiality=none"):
        parse_joint_signal_labels(
            {"direction_severity": "positive", "materiality": "none", "novelty": "high"}
        )
    with pytest.raises(ValueError, match="extreme direction"):
        parse_joint_signal_labels(
            {"direction_severity": "very_negative", "materiality": "moderate", "novelty": "low"}
        )


def test_local_scoring_resumes_then_human_validation_passes(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    prepared = prepare_signal_audit(run_dir, spec=SignalAuditSpec(sample_per_symbol=2, seed=11))
    first_client = FakeOllamaClient()
    first = asyncio.run(
        score_signal_audit(
            prepared.output_dir,
            endpoint="http://127.0.0.1:11435",
            max_new_calls=2,
            allow_calls=True,
            client=first_client,
        )
    )
    assert not first.complete
    assert first.calls_made == 2
    assert first.successful_scores == 2

    second_client = FakeOllamaClient()
    second = asyncio.run(
        score_signal_audit(
            prepared.output_dir,
            endpoint="http://127.0.0.1:11435",
            allow_calls=True,
            client=second_client,
        )
    )
    assert second.complete
    assert second.successful_scores == 4
    assert second.calls_made == 2
    scores_hash = sha256_file(second.scores_path)
    reused = asyncio.run(
        score_signal_audit(
            prepared.output_dir,
            endpoint="http://127.0.0.1:11435",
            allow_calls=True,
            client=FakeOllamaClient(),
        )
    )
    assert reused.reused
    assert sha256_file(reused.scores_path) == scores_hash

    human_path = prepared.output_dir / "human_labels.csv"
    with prepared.human_template_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    model_rows = [json.loads(line) for line in second.scores_path.read_text(encoding="utf-8").splitlines()]
    model_lookup = {row["audit_id"]: row for row in model_rows}
    for row in rows:
        audit_id = row["audit_id"]
        row["human_direction_severity"] = model_lookup[audit_id]["direction_severity"]
        row["human_materiality"] = model_lookup[audit_id]["materiality"]
        row["human_novelty"] = model_lookup[audit_id]["novelty"]
    with human_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    validation = validate_signal_audit(second.output_dir, human_path, output_root=tmp_path / "validation")

    assert all(check.passed for check in validation.acceptance)
    assert all(metric.quadratic_weighted_kappa == 1 for metric in validation.metrics)


def test_scoring_rejects_nonloopback_endpoint_without_calls(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    prepared = prepare_signal_audit(run_dir, spec=SignalAuditSpec(sample_per_symbol=1))
    client = FakeOllamaClient()

    with pytest.raises(SignalQualityError, match="loopback"):
        asyncio.run(
            score_signal_audit(
                prepared.output_dir,
                endpoint="https://external.example/v1",
                allow_calls=True,
                client=client,
            )
        )

    assert client.calls == []


def test_owned_client_closes_when_frozen_model_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _source_run(tmp_path)
    prepared = prepare_signal_audit(run_dir, spec=SignalAuditSpec(sample_per_symbol=1))
    client = MismatchedOwnedClient()
    monkeypatch.setattr(signal_quality, "OllamaClient", lambda **_: client)

    with pytest.raises(SignalQualityError, match="digest mismatch"):
        asyncio.run(
            score_signal_audit(
                prepared.output_dir,
                endpoint="http://127.0.0.1:11435",
                allow_calls=True,
            )
        )

    assert client.calls == []
    assert client.closed
