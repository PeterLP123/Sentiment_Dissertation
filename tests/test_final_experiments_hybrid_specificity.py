from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from final_experiments.lib.hybrid_specificity import (
    FILTER_ID,
    apply_high_specificity_filter,
    audit_topic_metadata,
    build_capped_hybrid_targets,
)


def test_specificity_filter_keeps_only_explicit_single_company_nontechnical() -> None:
    scores = pd.DataFrame(
        {
            "headline_sha256": ["a", "b", "c", "d"],
            "matched_symbols": ["AAA", "AAA|BBB", "CCC", "DDD"],
            "explicit_target": [True, True, False, True],
            "contextual": [False, False, True, False],
            "market_price_technical": [False, False, False, True],
            "score": [0.9, 0.8, -0.3, 0.1],
        }
    )

    filtered, audit = apply_high_specificity_filter(scores)

    assert filtered["headline_sha256"].tolist() == ["a"]
    assert filtered.attrs["specificity_filter_id"] == FILTER_ID
    assert audit["input_unique_headlines"] == 4
    assert audit["retained_unique_headlines"] == 1
    assert audit["input_company_associations"] == 5
    assert audit["retained_company_associations"] == 1
    assert audit["sentiment_or_return_used"] is False


def test_specificity_filter_rejects_inconsistent_contextual_flag() -> None:
    scores = pd.DataFrame(
        {
            "headline_sha256": ["a"],
            "matched_symbols": ["AAA"],
            "explicit_target": [True],
            "contextual": [True],
            "market_price_technical": [False],
        }
    )

    with pytest.raises(ValueError, match="logical complement"):
        apply_high_specificity_filter(scores)


def test_topic_audit_counts_only_nonempty_metadata(tmp_path) -> None:
    path = tmp_path / "headlines.jsonl"
    rows = [
        {"headline": "licensed one", "subjects": [], "entities": []},
        {"headline": "licensed two", "subjects": ["TOPIC"], "entities": []},
        {"headline": "licensed three", "subjects": [], "entities": [{"id": "X"}]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert audit_topic_metadata(path) == {
        "corpus_rows": 3,
        "rows_with_subjects": 1,
        "rows_with_entities": 1,
    }


def test_capped_hybrid_uses_gemma_counts_and_finbert_ranks() -> None:
    sessions = pd.to_datetime(["2026-01-02", "2026-01-05"])
    symbols = ("AAA", "BBB", "CCC")
    gemma = pd.DataFrame(
        [
            {"session_date": sessions[0], "symbol": "AAA", "strongest_event": 1.0},
            {"session_date": sessions[0], "symbol": "BBB", "strongest_event": 0.2},
            {"session_date": sessions[0], "symbol": "CCC", "strongest_event": -1.0},
            {"session_date": sessions[1], "symbol": "AAA", "strongest_event": 1.0},
            {"session_date": sessions[1], "symbol": "BBB", "strongest_event": 0.2},
            {"session_date": sessions[1], "symbol": "CCC", "strongest_event": -0.8},
        ]
    )
    finbert = pd.DataFrame(
        [
            {"session_date": session, "symbol": symbol, "strongest_event": score}
            for session in sessions
            for symbol, score in (("AAA", 0.1), ("BBB", 0.9), ("CCC", -0.7))
        ]
    )

    targets, audit = build_capped_hybrid_targets(gemma, finbert, symbols)

    first = {position.symbol: position.target_weight for position in targets[0].positions}
    second = np.asarray([position.target_weight for position in targets[1].positions])
    assert first == {"AAA": 0.0, "BBB": 0.25, "CCC": -0.25}
    assert np.array_equal(second, np.zeros(3))
    assert audit["eligible"].tolist() == [True, False]
    assert targets[0].gross_exposure == pytest.approx(0.5)
    assert targets[1].gross_exposure == 0.0


def test_capped_hybrid_requires_complete_finbert_ranks() -> None:
    panel = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "symbol": ["AAA", "BBB"],
            "strongest_event": [1.0, -1.0],
        }
    )
    finbert = panel.copy()
    finbert.loc[0, "strongest_event"] = np.nan

    with pytest.raises(ValueError, match="must be complete"):
        build_capped_hybrid_targets(panel, finbert, ("AAA", "BBB"))
