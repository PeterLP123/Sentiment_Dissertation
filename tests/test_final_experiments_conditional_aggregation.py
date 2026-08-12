from __future__ import annotations

import numpy as np
import pandas as pd

from final_experiments.lib.conditional_aggregation import (
    conditional_negative_share_test,
    daily_conditional_rank_coefficients,
)


def test_conditional_rank_test_recovers_negative_daily_effect() -> None:
    rows = []
    for date_number, session in enumerate(pd.date_range("2020-01-02", periods=12, freq="B")):
        for name in range(20):
            mean = ((name * 7 + date_number) % 20) / 19
            negative_share = name / 19
            count = 1 + ((name * 3 + date_number) % 9)
            outcome = -2.0 * negative_share + 0.4 * mean + 0.01 * name
            rows.append(
                {
                    "session_date": session,
                    "mean_continuous": mean,
                    "negative_share": negative_share,
                    "n": count,
                    "return": outcome,
                }
            )
    frame = pd.DataFrame(rows)
    summary, daily, audit = conditional_negative_share_test(
        frame,
        date_col="session_date",
        outcome_col="return",
        min_names=10,
        hac_lags=2,
    )

    assert len(daily) == 12
    assert audit["dates_used"] == 12
    assert summary["estimate"] < 0
    assert summary["ci_high"] < 0
    assert summary["direction_matches"] is True


def test_rank_deficient_and_small_sessions_are_counted() -> None:
    complete = pd.DataFrame(
        {
            "session_date": pd.Timestamp("2020-01-02"),
            "return": np.arange(10, dtype=float),
            "mean_continuous": np.array([3, 8, 1, 9, 4, 0, 7, 2, 6, 5], dtype=float),
            "negative_share": np.linspace(0, 1, 10),
            "n": np.array([2, 9, 1, 7, 4, 10, 3, 8, 5, 6]),
        }
    )
    constant = complete.assign(
        session_date=pd.Timestamp("2020-01-03"),
        negative_share=0.0,
    )
    small = complete.iloc[:5].assign(session_date=pd.Timestamp("2020-01-06"))
    daily, audit = daily_conditional_rank_coefficients(
        pd.concat([complete, constant, small], ignore_index=True),
        date_col="session_date",
        outcome_col="return",
        mean_col="mean_continuous",
        negative_share_col="negative_share",
        count_col="n",
        min_names=10,
    )

    assert len(daily) == 1
    assert audit == {
        "dates_total": 3,
        "dates_used": 1,
        "dates_below_min_names": 1,
        "dates_constant_variable": 1,
        "dates_rank_deficient": 0,
    }
