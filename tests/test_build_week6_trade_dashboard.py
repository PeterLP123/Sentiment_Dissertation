from __future__ import annotations

import math
import runpy
from pathlib import Path

import pandas as pd

_DASHBOARD = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "build_week6_trade_dashboard.py"))
HeadlineContribution = _DASHBOARD["HeadlineContribution"]
_decision_payload = _DASHBOARD["_decision_payload"]


def test_model_decision_payload_distinguishes_unscored_from_zero() -> None:
    reference = ("ABC", "2026-01-02")
    contributions = {
        ("ABC", "2026-01-02", "positive"): HeadlineContribution(
            symbol="ABC",
            news_date="2026-01-02",
            norm_hash="positive",
            headline="Positive",
            score=0.5,
            raw_count=1,
        ),
        ("ABC", "2026-01-02", "zero"): HeadlineContribution(
            symbol="ABC",
            news_date="2026-01-02",
            norm_hash="zero",
            headline="Neutral",
            score=0.0,
            raw_count=1,
        ),
    }
    signals = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "news_date": "2026-01-02",
                "scorer_id": "llm/example",
                "article_count": 3,
                "valid_count": 2,
                "mean_score": 0.25,
            }
        ]
    )

    payload = _decision_payload(
        contributions,
        {reference: 3},
        signals,
        {reference},
        scorer_id="llm/example",
        nonzero_examples=3,
        neutral_examples=1,
    )["ABC|2026-01-02"]

    assert payload["raw_n"] == 3
    assert payload["n"] == 2
    assert payload["x"] == 1
    assert payload["p"] == 1
    assert payload["m"] == 0
    assert payload["z"] == 1
    assert math.isclose(payload["mu"], 0.25)
