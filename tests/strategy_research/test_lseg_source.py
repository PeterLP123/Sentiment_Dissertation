from __future__ import annotations

import json
from pathlib import Path

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.strategy_research.schemas import load_strategy_events
from sentiment_benchmark.strategy_research.sources import build_strategy_events, write_event_build_artifacts


def _record(
    story_id: str,
    family: str,
    timestamp: str,
    *,
    headline: str = "Alpha Corp announces an update",
    body: str = "Alpha Corp expects growth. Alpha Corp has signed a material contract.",
    symbols: list[str] | None = None,
    eligible: bool = True,
) -> dict:
    return {
        "article_id": family,
        "revision_id": story_id.replace(":", "-"),
        "story_id": story_id,
        "story_family_id": family,
        "headline": headline,
        "clean_text": body,
        "clean_text_sha256": "source-clean-hash",
        "version_created": timestamp,
        "matched_symbols": ["AAA"] if symbols is None else symbols,
        "scoring_eligible": eligible,
        "max_scoring_chars": 12000,
    }


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    rows = [
        # The first revision is available before Friday's open. The later
        # revision must never replace it.
        _record("urn:alpha:contract:1", "urn:alpha:contract", "2026-03-06T13:55:00+00:00"),
        _record("urn:alpha:contract:2", "urn:alpha:contract", "2026-03-06T15:00:00+00:00"),
        # Processing crosses the Friday open and therefore executes Monday,
        # after the US daylight-saving transition.
        _record("urn:alpha:cross:1", "urn:alpha:cross", "2026-03-06T14:15:00+00:00"),
        _record("urn:alpha:after-hours:1", "urn:alpha:after-hours", "2026-03-06T22:00:00+00:00"),
        # One story creates one target-specific event for each named company.
        _record(
            "urn:joint:1",
            "urn:joint",
            "2026-03-07T12:00:00+00:00",
            headline="Alpha Corp and Beta Corp sign a joint agreement",
            body="Alpha Corp and Beta Corp signed an agreement affecting both companies.",
            symbols=["AAA", "BBB"],
        ),
        # July 3 is the observed Independence Day market holiday in 2026.
        _record("urn:alpha:holiday:1", "urn:alpha:holiday", "2026-07-03T12:00:00+00:00"),
        _record("urn:invalid:1", "urn:invalid", "2026-03-06T12:00:00"),
        _record("urn:no-symbol:1", "urn:no-symbol", "2026-03-06T12:00:00+00:00", symbols=[]),
        _record("urn:no-text:1", "urn:no-text", "2026-03-06T12:00:00+00:00", body="", eligible=False),
        _record(
            "urn:irrelevant:1",
            "urn:irrelevant",
            "2026-03-06T12:00:00+00:00",
            headline="Macro update",
            body="No named company is discussed.",
        ),
    ]
    # Reproduce a real-corpus boundary where truncation lands on whitespace.
    # Event hashing must use the same normalized text exposed by StrategyEvent.
    rows[0]["max_scoring_chars"] = 11
    articles = root / "articles.jsonl"
    articles.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    screening = root / "screening_index.csv"
    screening.write_text("article_id,revision_id\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "config": {
            "companies": [
                {"symbol": "AAA", "name": "Alpha Corp", "aliases": ["Alpha Corp"]},
                {"symbol": "BBB", "name": "Beta Corp", "aliases": ["Beta Corp"]},
            ]
        },
        "files": {
            "articles_jsonl": {"path": articles.name, "sha256": sha256_file(articles)},
            "screening_index_csv": {"path": screening.name, "sha256": sha256_file(screening)},
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_build_strategy_events_uses_point_in_time_revisions_and_xnys_opens(tmp_path: Path) -> None:
    result = build_strategy_events(_corpus(tmp_path))

    assert len(result.events) == 6
    contract = next(event for event in result.events if event.story_family_id == "urn:alpha:contract")
    crossed = next(event for event in result.events if event.story_family_id == "urn:alpha:cross")
    joint = [event for event in result.events if event.story_family_id == "urn:joint"]
    after_hours = next(event for event in result.events if event.story_family_id == "urn:alpha:after-hours")
    holiday = next(event for event in result.events if event.story_family_id == "urn:alpha:holiday")
    assert contract.revision_id == "urn-alpha-contract-1"
    assert contract.available_at_utc.isoformat() == "2026-03-06T14:10:00+00:00"
    assert contract.eligible_execution_session.isoformat() == "2026-03-06"
    assert crossed.eligible_execution_session.isoformat() == "2026-03-09"
    assert crossed.available_at_utc.isoformat() == "2026-03-06T14:30:00+00:00"
    assert after_hours.eligible_execution_session.isoformat() == "2026-03-09"
    assert holiday.eligible_execution_session.isoformat() == "2026-07-06"
    assert {event.symbol for event in joint} == {"AAA", "BBB"}
    assert all(event.eligible_execution_session.isoformat() == "2026-03-09" for event in joint)
    assert [event.event_id for event in result.events] == [
        event.event_id for event in sorted(result.events, key=lambda value: (value.available_at_utc, value.event_id))
    ]
    assert result.attrition["duplicate_revision"] == 1
    assert result.attrition["invalid_timestamp"] == 1
    assert result.attrition["missing_symbol"] == 1
    assert result.attrition["missing_usable_text"] == 1
    assert result.attrition["relevance_exclusion"] == 1
    assert result.attrition["missing_score"] == 0
    assert result.event_rules["processing_buffer_minutes"] == 15
    assert len(result.event_rules_hash) == 64


def test_event_artifacts_are_replayable_and_refuse_different_content(tmp_path: Path) -> None:
    result = build_strategy_events(_corpus(tmp_path))
    output = tmp_path / "events"

    paths = write_event_build_artifacts(output, result)
    write_event_build_artifacts(output, result)

    assert len(load_strategy_events(paths[0])) == 6
    assert json.loads(paths[2].read_text(encoding="utf-8"))["counts"]["selected_events"] == 6
