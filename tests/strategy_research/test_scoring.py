from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from sentiment_benchmark.artifact_io import sha256_text
from sentiment_benchmark.models import LLMResponseRecord
from sentiment_benchmark.strategy_research.prompts import strategy_prompt
from sentiment_benchmark.strategy_research.schemas import StrategyEvent, load_score_records
from sentiment_benchmark.strategy_research.score_mapping import FIVE_LEVEL_LABELS, map_score_label
from sentiment_benchmark.strategy_research.scoring import (
    ScoringIdentity,
    import_completed_score_cache,
    inspect_score_cache,
    materialize_scores,
    score_cache_key,
    score_events,
)


def _event(event_id: str = "event-1", symbol: str = "ACME") -> StrategyEvent:
    headline = "Acme wins a material contract"
    body = "Acme expects higher annual cash flow."
    return StrategyEvent(
        event_id=event_id,
        story_family_id=f"family-{event_id}",
        revision_id=f"revision-{event_id}",
        symbol=symbol,
        target_company_name="Acme Corp",
        version_created_utc=datetime(2026, 3, 6, 13, 0, tzinfo=UTC),
        available_at_utc=datetime(2026, 3, 6, 13, 15, tzinfo=UTC),
        source="lseg",
        headline=headline,
        lead_or_body=body,
        text_sha256=sha256_text(f"{headline}\n\n{body}"),
        target_relevance="include",
        event_session=date(2026, 3, 6),
        eligible_execution_session=date(2026, 3, 6),
        exclusion_reason=None,
        input_manifest_hash="manifest",
    )


class FakeClient:
    def __init__(self, label: str | None) -> None:
        self.label = label
        self.calls = []

    async def classify(self, model_id, prompt, example, **kwargs):
        self.calls.append((model_id, prompt, example, kwargs))
        return LLMResponseRecord(
            row_number=example.row_number,
            model_id=model_id,
            prompt_hash=prompt.prompt_hash,
            raw_content=self.label,
            normalized_label=self.label,
            parse_status="valid" if self.label else "invalid",
            status="success",
            latency_ms=12.5,
            attempt_count=2,
        )


def test_score_mappings_and_prompts_are_strict_and_separate() -> None:
    expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert [map_score_label(label, "five_level") for label in FIVE_LEVEL_LABELS] == expected
    assert [map_score_label(label, "three_class") for label in ("negative", "neutral", "positive")] == [-1.0, 0.0, 1.0]
    three = strategy_prompt("three_class")
    five = strategy_prompt("five_level")
    assert three.prompt_hash != five.prompt_hash
    assert "Do not predict a stock return" in five.system_prompt
    assert five.output_mode == "label_only"


def test_hosted_score_identity_binds_the_exact_endpoint() -> None:
    first = ScoringIdentity(
        "three_class",
        "cerebras",
        "gemma-4-31b",
        endpoint="https://api.cerebras.ai/v1",
    )
    second = ScoringIdentity(
        "three_class",
        "cerebras",
        "gemma-4-31b",
        endpoint="https://example.invalid/v1",
    )

    assert first.resolved_endpoint == "https://api.cerebras.ai/v1"
    assert first.config_hash != second.config_hash


def test_ollama_score_identity_binds_disabled_thinking() -> None:
    identity = ScoringIdentity(
        "three_class",
        "ollama",
        "gemma4:12b",
        model_digest="digest",
        endpoint="http://127.0.0.1:11435",
        ollama_think=False,
    )

    assert identity.ollama_think is False
    with pytest.raises(ValueError, match="requires ollama_think=False"):
        replace(identity, ollama_think=None)


def test_score_cache_is_target_specific_resumable_and_deterministic(tmp_path) -> None:
    identity = ScoringIdentity("five_level", "cerebras", "gemma-4-31b")
    first_event = _event()
    second_target = replace(first_event, event_id="event-2", symbol="OTHER", target_company_name="Other Corp")
    cache = tmp_path / "cache"
    assert score_cache_key(first_event, identity) != score_cache_key(second_target, identity)
    assert score_cache_key(first_event, identity) != score_cache_key(
        first_event,
        replace(identity, max_completion_tokens=32),
    )

    before = inspect_score_cache([first_event], identity, cache)
    assert before.expected_score_calls == 1
    assert not cache.exists()
    client = FakeClient("very_positive")
    first = asyncio.run(score_events([first_event], identity, cache, client, allow_calls=True))
    second = asyncio.run(score_events([first_event], identity, cache, FakeClient("negative"), allow_calls=False))

    assert first.calls_made == 1
    assert first.records[0].score == 1.0
    assert first.records[0].attempt_count == 2
    assert second.cache_hits == 1
    assert second.records == first.records
    output = materialize_scores(tmp_path / "scores.jsonl", first)
    first_bytes = output.read_bytes()
    materialize_scores(output, second)
    assert output.read_bytes() == first_bytes
    assert load_score_records(output) == first.records


def test_invalid_score_is_not_neutral_and_is_retried_with_cumulative_attempts(tmp_path) -> None:
    identity = ScoringIdentity("three_class", "cerebras", "gemma-4-31b")
    event = _event()
    cache = tmp_path / "cache"

    first = asyncio.run(score_events([event], identity, cache, FakeClient(None), allow_calls=True))
    second = asyncio.run(score_events([event], identity, cache, FakeClient(None), allow_calls=True))

    assert first.records[0].status == "invalid"
    assert first.records[0].score is None
    assert first.missing_event_ids == (event.event_id,)
    assert second.records[0].attempt_count == 4
    assert inspect_score_cache([event], identity, cache).cache_hits == 0


def test_score_cache_rejects_incompatible_identity_or_mapping(tmp_path) -> None:
    identity = ScoringIdentity("three_class", "cerebras", "gemma-4-31b")
    event = _event()
    cache = tmp_path / "cache"
    asyncio.run(score_events([event], identity, cache, FakeClient("positive"), allow_calls=True))
    completed = next((cache / "completed").rglob("*.json"))
    payload = json.loads(completed.read_text(encoding="utf-8"))
    payload["score"] = 0.5
    completed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid numeric mapping"):
        inspect_score_cache([event], identity, cache)


def test_successful_attempt_journal_recovers_without_a_second_model_call(tmp_path) -> None:
    identity = ScoringIdentity("three_class", "cerebras", "gemma-4-31b")
    event = _event()
    cache = tmp_path / "cache"
    first_client = FakeClient("positive")
    asyncio.run(score_events([event], identity, cache, first_client, allow_calls=True))
    completed = next((cache / "completed").rglob("*.json"))
    completed.unlink()

    inspection = inspect_score_cache([event], identity, cache)
    assert inspection.cache_hits == 1
    assert completed.exists() is False
    second_client = FakeClient("negative")
    recovered = asyncio.run(score_events([event], identity, cache, second_client, allow_calls=False))

    assert recovered.records[0].score == 1.0
    assert recovered.calls_made == 0
    assert second_client.calls == []
    assert completed.exists()


def test_bounded_scoring_stops_after_new_call_limit_and_resumes(tmp_path) -> None:
    identity = ScoringIdentity("three_class", "cerebras", "gemma-4-31b")
    events = [_event(f"event-{index}") for index in range(4)]
    cache = tmp_path / "cache"
    first_client = FakeClient("positive")

    first = asyncio.run(
        score_events(
            events,
            identity,
            cache,
            first_client,
            allow_calls=True,
            max_new_scores=2,
        )
    )
    second_client = FakeClient("negative")
    second = asyncio.run(
        score_events(
            events,
            identity,
            cache,
            second_client,
            allow_calls=True,
            max_new_scores=1,
        )
    )

    assert first.calls_made == 2
    assert first.cache_hits == 0
    assert len(first.missing_event_ids) == 2
    assert len(first_client.calls) == 2
    assert second.calls_made == 1
    assert second.cache_hits == 2
    assert len(second.missing_event_ids) == 1
    assert len(second_client.calls) == 1
    assert inspect_score_cache(events, identity, cache).cache_hits == 3


def test_bounded_scoring_rejects_nonpositive_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_new_scores must be positive"):
        asyncio.run(
            score_events(
                [_event()],
                ScoringIdentity("three_class", "cerebras", "gemma-4-31b"),
                tmp_path / "cache",
                FakeClient("positive"),
                allow_calls=True,
                max_new_scores=0,
            )
        )


def test_complete_score_cache_can_be_imported_across_run_identities(tmp_path) -> None:
    identity = ScoringIdentity("three_class", "cerebras", "gemma-4-31b")
    events = [_event(f"event-{index}") for index in range(3)]
    source = tmp_path / "source"
    target = tmp_path / "target"
    first = asyncio.run(score_events(events, identity, source, FakeClient("positive"), allow_calls=True))
    assert first.calls_made == 3

    provenance = import_completed_score_cache(events, identity, source, target)
    reused = asyncio.run(score_events(events, identity, target, allow_calls=False))

    assert provenance["imported_records"] == 3
    assert provenance["source_completed_sha256"]
    assert reused.cache_hits == 3
    assert reused.calls_made == 0
    assert reused.missing_event_ids == ()


def test_incomplete_score_cache_import_is_rejected_before_target_writes(tmp_path) -> None:
    identity = ScoringIdentity("three_class", "cerebras", "gemma-4-31b")
    events = [_event("one"), _event("two")]
    source = tmp_path / "source"
    target = tmp_path / "target"
    asyncio.run(score_events(events[:1], identity, source, FakeClient("neutral"), allow_calls=True))

    with pytest.raises(ValueError, match="source score cache is incomplete"):
        import_completed_score_cache(events, identity, source, target)

    assert not target.exists()
