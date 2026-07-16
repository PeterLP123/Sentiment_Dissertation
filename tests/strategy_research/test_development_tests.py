from __future__ import annotations

import pytest

from sentiment_benchmark.strategy_research.development_tests import (
    DevelopmentTestSpec,
    aggregate_stock_session_scores,
    calculate_label_horizon_results,
    evaluate_candidate_folds,
)
from sentiment_benchmark.strategy_research.market import OpenToOpenReturn


def test_stock_session_aggregation_removes_event_volume_duplication() -> None:
    events = [
        {"event_id": "e1", "eligible_execution_session": "2026-01-02", "symbol": "AAA"},
        {"event_id": "e2", "eligible_execution_session": "2026-01-02", "symbol": "AAA"},
        {"event_id": "e3", "eligible_execution_session": "2026-01-02", "symbol": "BBB"},
        {"event_id": "e4", "eligible_execution_session": "2026-01-02", "symbol": "BBB"},
    ]
    scores = [
        {"event_id": "e1", "status": "success", "score": 1.0},
        {"event_id": "e2", "status": "success", "score": -1.0},
        {"event_id": "e3", "status": "success", "score": 1.0},
        {"event_id": "e4", "status": "invalid", "score": 0.0},
    ]

    rows = aggregate_stock_session_scores(events, scores, ("2026-01-02",))

    assert len(rows) == 2
    assert rows[0].symbol == "AAA"
    assert rows[0].event_count == 2
    assert rows[0].mean_score == 0.0
    assert rows[0].positive_events == rows[0].negative_events == 1


def test_label_horizon_returns_never_reach_evaluation_boundary() -> None:
    sessions = ("2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07")
    symbols = ("AAA", "BBB")
    prices = {
        ("AAA", "2026-01-02"): 100.0,
        ("AAA", "2026-01-05"): 110.0,
        ("AAA", "2026-01-06"): 121.0,
        ("AAA", "2026-01-07"): 1000.0,
        ("BBB", "2026-01-02"): 100.0,
        ("BBB", "2026-01-05"): 100.0,
        ("BBB", "2026-01-06"): 100.0,
        ("BBB", "2026-01-07"): 1.0,
    }
    events = [
        {"event_id": "e1", "eligible_execution_session": "2026-01-02", "symbol": "AAA"},
        {"event_id": "e2", "eligible_execution_session": "2026-01-06", "symbol": "AAA"},
    ]
    scores = [
        {"event_id": "e1", "status": "success", "score": 1.0},
        {"event_id": "e2", "status": "success", "score": -1.0},
    ]
    aggregated = aggregate_stock_session_scores(events, scores, sessions[:-1])

    labels, coefficients = calculate_label_horizon_results(
        events,
        scores,
        aggregated,
        prices,
        sessions,
        symbols,
        "2026-01-07",
        (1,),
    )

    stock_session = [row for row in labels if row.aggregation == "stock_session_mean"]
    assert len(stock_session) == 1
    assert stock_session[0].label == "positive"
    assert stock_session[0].observations == 1
    assert stock_session[0].mean_raw_return == pytest.approx(0.10)
    assert all(row.observations == 1 for row in coefficients)


def test_candidate_folds_use_training_center_and_exclude_boundary_interval() -> None:
    sessions = (
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
    )
    symbols = ("AAA", "BBB")
    returns = []
    for symbol in symbols:
        values = (0.0, 0.0, 0.0, 0.02 if symbol == "AAA" else -0.01, 10.0)
        returns.extend(
            OpenToOpenReturn(symbol, current, following, value)
            for current, following, value in zip(sessions[:-1], sessions[1:], values, strict=True)
        )
    events = [
        {"event_id": "a1", "eligible_execution_session": "2026-01-02", "symbol": "AAA"},
        {"event_id": "b1", "eligible_execution_session": "2026-01-02", "symbol": "BBB"},
        {"event_id": "a2", "eligible_execution_session": "2026-01-07", "symbol": "AAA"},
        {"event_id": "b2", "eligible_execution_session": "2026-01-07", "symbol": "BBB"},
    ]
    scores = [
        {"event_id": "a1", "status": "success", "score": 1.0},
        {"event_id": "b1", "status": "success", "score": -1.0},
        {"event_id": "a2", "status": "success", "score": 1.0},
        {"event_id": "b2", "status": "success", "score": -1.0},
    ]
    signals = aggregate_stock_session_scores(events, scores, sessions[:-1])
    folds = [
        {
            "fold_index": 0,
            "training_sessions": list(sessions[:3]),
            "validation_sessions": list(sessions[3:]),
        }
    ]
    spec = DevelopmentTestSpec(
        horizons=(1,),
        thresholds=(0.0,),
        minimum_supported_stocks=1,
        minimum_active_fraction=0.0,
    )

    fold_rows, candidates = evaluate_candidate_folds(
        signals,
        returns,
        sessions[:-1],
        symbols,
        folds,
        "2026-01-09",
        spec,
    )

    assert len(fold_rows) == 2
    assert all(row.validation_end == "2026-01-07" for row in fold_rows)
    assert all(row.eligible_intervals == 1 for row in fold_rows)
    assert all(row.valid for row in candidates)
