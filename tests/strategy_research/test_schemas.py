from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from sentiment_benchmark.artifact_io import sha256_text
from sentiment_benchmark.strategy_research.schemas import StrategyEvent, load_strategy_events, write_strategy_events


def _event() -> StrategyEvent:
    headline = "Acme wins a material contract"
    body = "Acme expects the agreement to increase annual cash flow."
    return StrategyEvent(
        event_id="event-1",
        story_family_id="story-family",
        revision_id="revision-1",
        symbol="ACME",
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
        input_manifest_hash="manifest-hash",
    )


def test_strategy_event_requires_aware_timestamps_and_matching_text_hash() -> None:
    event = _event()
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(event, version_created_utc=datetime(2026, 3, 6, 13, 0))
    with pytest.raises(ValueError, match="text_sha256"):
        replace(event, text_sha256="wrong")


def test_strategy_event_jsonl_is_deterministic_and_immutable(tmp_path) -> None:
    later = replace(
        _event(),
        event_id="event-2",
        available_at_utc=datetime(2026, 3, 9, 13, 15, tzinfo=UTC),
        eligible_execution_session=date(2026, 3, 9),
    )
    path = tmp_path / "events.jsonl"

    write_strategy_events(path, [later, _event()])
    first = path.read_bytes()
    write_strategy_events(path, [_event(), later])

    assert path.read_bytes() == first
    assert [event.event_id for event in load_strategy_events(path)] == ["event-1", "event-2"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_strategy_events(path, [_event()])
