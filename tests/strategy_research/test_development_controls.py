from __future__ import annotations

from collections import Counter, defaultdict

from sentiment_benchmark.strategy_research.development_controls import (
    build_control_exposures,
    shuffle_scores_within_stock,
)
from sentiment_benchmark.strategy_research.development_tests import SessionSignal


def signal(session: str, symbol: str, score: float) -> SessionSignal:
    return SessionSignal(
        session=session,
        symbol=symbol,
        event_count=1,
        mean_score=score,
        positive_events=int(score > 0),
        neutral_events=int(score == 0),
        negative_events=int(score < 0),
    )


def test_non_refreshing_hold_ignores_same_direction_and_reverses_opposite() -> None:
    sessions = tuple(f"2026-01-{day:02d}" for day in range(1, 9))
    signals = (
        signal(sessions[0], "AAA", 1.0),
        signal(sessions[2], "AAA", 1.0),
        signal(sessions[4], "AAA", -1.0),
    )

    exposures = build_control_exposures(
        sessions,
        ("AAA",),
        signals,
        threshold=0.75,
        hold_sessions=4,
        control="sentiment_v2",
    )["AAA"]

    assert exposures == (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0)


def test_refresh_control_restarts_same_direction_clock() -> None:
    sessions = tuple(f"2026-01-{day:02d}" for day in range(1, 8))
    signals = (signal(sessions[0], "AAA", 1.0), signal(sessions[2], "AAA", 1.0))

    non_refreshing = build_control_exposures(
        sessions,
        ("AAA",),
        signals,
        threshold=0.75,
        hold_sessions=4,
        control="sentiment_v2",
    )["AAA"]
    refreshing = build_control_exposures(
        sessions,
        ("AAA",),
        signals,
        threshold=0.75,
        hold_sessions=4,
        control="refreshing_signal",
    )["AAA"]

    assert non_refreshing == (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert refreshing == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)


def test_inverted_and_news_timing_controls_ignore_or_reverse_direction() -> None:
    sessions = ("2026-01-01", "2026-01-02", "2026-01-03")
    signals = (signal(sessions[0], "AAA", 1.0), signal(sessions[1], "BBB", 0.0))

    inverted = build_control_exposures(
        sessions,
        ("AAA", "BBB"),
        signals,
        threshold=0.75,
        hold_sessions=2,
        control="inverted_sentiment",
    )
    timing = build_control_exposures(
        sessions,
        ("AAA", "BBB"),
        signals,
        threshold=0.75,
        hold_sessions=2,
        control="news_timing_long",
    )

    assert inverted["AAA"] == (-1.0, -1.0, 0.0)
    assert inverted["BBB"] == (0.0, 0.0, 0.0)
    assert timing["AAA"] == (1.0, 1.0, 0.0)
    assert timing["BBB"] == (0.0, 1.0, 1.0)


def test_shuffle_preserves_stock_score_multisets_and_session_event_counts() -> None:
    events = [
        {"event_id": "a1", "eligible_execution_session": "2026-01-01", "symbol": "AAA"},
        {"event_id": "a2", "eligible_execution_session": "2026-01-01", "symbol": "AAA"},
        {"event_id": "a3", "eligible_execution_session": "2026-01-02", "symbol": "AAA"},
        {"event_id": "b1", "eligible_execution_session": "2026-01-01", "symbol": "BBB"},
        {"event_id": "b2", "eligible_execution_session": "2026-01-02", "symbol": "BBB"},
        {"event_id": "b3", "eligible_execution_session": "2026-01-02", "symbol": "BBB"},
    ]
    scores = [
        {"event_id": "a1", "status": "success", "score": -1.0},
        {"event_id": "a2", "status": "success", "score": 0.0},
        {"event_id": "a3", "status": "success", "score": 1.0},
        {"event_id": "b1", "status": "success", "score": 1.0},
        {"event_id": "b2", "status": "success", "score": 1.0},
        {"event_id": "b3", "status": "invalid", "score": 0.0},
    ]

    first = shuffle_scores_within_stock(events, scores, ("2026-01-01", "2026-01-02"), seed=7)
    second = shuffle_scores_within_stock(events, scores, ("2026-01-01", "2026-01-02"), seed=7)

    assert first == second
    counts = {(row.session, row.symbol): row.event_count for row in first}
    assert counts == {
        ("2026-01-01", "AAA"): 2,
        ("2026-01-02", "AAA"): 1,
        ("2026-01-01", "BBB"): 1,
        ("2026-01-02", "BBB"): 1,
    }
    shuffled_multisets: dict[str, Counter[float]] = defaultdict(Counter)
    for row in first:
        shuffled_multisets[row.symbol].update(
            {
                -1.0: row.negative_events,
                0.0: row.neutral_events,
                1.0: row.positive_events,
            }
        )
    assert shuffled_multisets["AAA"] == Counter({-1.0: 1, 0.0: 1, 1.0: 1})
    assert shuffled_multisets["BBB"] == Counter({1.0: 2})
