from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from sentiment_benchmark.strategy_research.paper import PaperJournal, PaperJournalError, prospective_availability


def _clock(*values: str):
    timestamps = iter(datetime.fromisoformat(value.replace("Z", "+00:00")) for value in values)
    return lambda: next(timestamps)


def _race_paper_append(path: str, event_id: str, symbol: str, start: Any, results: Any) -> None:
    journal = PaperJournal(path, clock=lambda: datetime(2026, 7, 16, 10, 0, 2, tzinfo=UTC))
    start.wait()
    try:
        record = journal.append_event(
            event_id=event_id,
            source="synthetic",
            source_url=f"https://example.test/news/{event_id}",
            target_symbol=symbol,
            published_at_utc="2026-07-16T09:55:00Z",
            first_seen_at_utc="2026-07-16T10:00:00Z",
            retrieved_at_utc="2026-07-16T10:00:01Z",
            content_sha256=("a" if symbol == "AAPL" else "b") * 64,
        )
        results.put(("ok", record.sequence))
    except PaperJournalError as exc:
        results.put(("error", str(exc)))


def _append_event(journal: PaperJournal) -> None:
    journal.append_event(
        event_id="evt-1",
        source="synthetic",
        source_url="https://example.test/news/1",
        target_symbol="AAPL",
        published_at_utc="2026-07-16T09:55:00Z",
        first_seen_at_utc="2026-07-16T10:00:00Z",
        retrieved_at_utc="2026-07-16T10:00:01Z",
        content_sha256="a" * 64,
    )


def test_verify_absent_journal_does_not_create_it(tmp_path: Path) -> None:
    path = tmp_path / "paper/journal.jsonl"

    assert PaperJournal(path).verify() == ()
    assert not path.exists()
    assert not path.parent.exists()


def test_paper_journal_appends_hash_chained_event_proposal_and_outcome(tmp_path: Path) -> None:
    journal = PaperJournal(
        tmp_path / "paper.jsonl",
        clock=_clock("2026-07-16T10:00:02Z", "2026-07-16T10:03:01Z", "2026-07-17T13:30:01Z"),
    )
    _append_event(journal)
    proposal = journal.append_proposed_order(
        proposal_id="p-1",
        event_id="evt-1",
        scoring_completed_at_utc="2026-07-16T10:03:00Z",
        proposed_at_utc="2026-07-16T10:03:00Z",
        target_weight=0.04,
        order_delta=0.04,
        scorer_identity="fixture-v1",
        prompt_hash="b" * 64,
    )
    outcome = journal.append_outcome(
        outcome_id="o-1",
        proposal_id="p-1",
        observed_at_utc="2026-07-17T13:30:00Z",
        gross_return=0.01,
        net_return=0.0099,
    )

    records = journal.verify()

    assert [record.record_type for record in records] == ["event", "proposed_order", "outcome"]
    assert proposal.previous_record_sha256 == records[0].record_sha256
    assert outcome.previous_record_sha256 == proposal.record_sha256
    assert outcome.payload["proposal_record_id"] == "proposal:p-1"


def test_prospective_availability_is_later_of_first_seen_and_scoring_completion() -> None:
    result = prospective_availability("2026-07-16T10:05:00Z", "2026-07-16T10:03:00Z")

    assert result == datetime(2026, 7, 16, 10, 5, tzinfo=UTC)


def test_proposed_order_cannot_predate_prospective_availability(tmp_path: Path) -> None:
    journal = PaperJournal(tmp_path / "paper.jsonl", clock=_clock("2026-07-16T10:00:02Z"))
    _append_event(journal)

    with pytest.raises(PaperJournalError, match="prospective event availability"):
        journal.append_proposed_order(
            proposal_id="early",
            event_id="evt-1",
            scoring_completed_at_utc="2026-07-16T10:05:00Z",
            proposed_at_utc="2026-07-16T10:04:59Z",
            target_weight=0.01,
            order_delta=0.01,
            scorer_identity="fixture-v1",
            prompt_hash="b" * 64,
        )


def test_duplicate_records_never_rewrite_or_append(tmp_path: Path) -> None:
    journal = PaperJournal(tmp_path / "paper.jsonl", clock=_clock("2026-07-16T10:00:02Z"))
    _append_event(journal)
    before = journal.path.read_bytes()

    with pytest.raises(PaperJournalError, match="already contains record_id"):
        _append_event(journal)

    assert journal.path.read_bytes() == before
    assert len(journal.verify()) == 1


def test_tampering_fails_closed(tmp_path: Path) -> None:
    journal = PaperJournal(tmp_path / "paper.jsonl", clock=_clock("2026-07-16T10:00:02Z"))
    _append_event(journal)
    journal.path.write_text(journal.path.read_text(encoding="utf-8").replace("AAPL", "MSFT"), encoding="utf-8")

    with pytest.raises(PaperJournalError, match="record hash mismatch"):
        journal.verify()


def test_naive_timestamps_are_rejected_without_creating_journal(tmp_path: Path) -> None:
    journal = PaperJournal(tmp_path / "paper.jsonl")

    with pytest.raises(PaperJournalError, match="timezone-aware"):
        journal.append_event(
            event_id="evt-1",
            source="synthetic",
            source_url="https://example.test/news/1",
            target_symbol="AAPL",
            published_at_utc="2026-07-16T09:55:00",
            first_seen_at_utc="2026-07-16T10:00:00Z",
            retrieved_at_utc="2026-07-16T10:00:01Z",
            content_sha256="a" * 64,
        )

    assert not journal.path.exists()


def test_recording_clock_cannot_move_backwards(tmp_path: Path) -> None:
    journal = PaperJournal(
        tmp_path / "paper.jsonl",
        clock=_clock("2026-07-16T11:00:00Z", "2026-07-16T10:00:00Z"),
    )
    _append_event(journal)

    with pytest.raises(PaperJournalError, match="must be monotonic"):
        journal.append_event(
            event_id="evt-2",
            source="synthetic",
            source_url="https://example.test/news/2",
            target_symbol="MSFT",
            published_at_utc="2026-07-16T09:00:00Z",
            first_seen_at_utc="2026-07-16T09:01:00Z",
            retrieved_at_utc="2026-07-16T09:01:01Z",
            content_sha256="c" * 64,
        )

    assert len(journal.verify()) == 1


def test_proposal_requires_a_sha256_prompt_identity(tmp_path: Path) -> None:
    journal = PaperJournal(tmp_path / "paper.jsonl", clock=_clock("2026-07-16T10:00:02Z"))
    _append_event(journal)

    with pytest.raises(PaperJournalError, match="prompt_hash must be a 64-character"):
        journal.append_proposed_order(
            proposal_id="invalid-prompt",
            event_id="evt-1",
            scoring_completed_at_utc="2026-07-16T10:03:00Z",
            proposed_at_utc="2026-07-16T10:03:00Z",
            target_weight=0.01,
            order_delta=0.01,
            scorer_identity="fixture-v1",
            prompt_hash="not-a-hash",
        )


def test_concurrent_appenders_produce_one_valid_hash_chain(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    path = tmp_path / "paper.jsonl"
    processes = [
        context.Process(target=_race_paper_append, args=(str(path), event_id, symbol, start, results))
        for event_id, symbol in (("evt-a", "AAPL"), ("evt-b", "MSFT"))
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=5), results.get(timeout=5)]
    results.close()

    assert sorted(outcome[0] for outcome in outcomes) == ["ok", "ok"]
    assert sorted(outcome[1] for outcome in outcomes) == [1, 2]
    records = PaperJournal(path).verify()
    assert [record.sequence for record in records] == [1, 2]
    assert records[1].previous_record_sha256 == records[0].record_sha256
