from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.lib.conditional_aggregation import (
    conditional_negative_share_test,
    daily_conditional_rank_coefficients,
    hac_regime_mean_contrast,
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


def test_hac_regime_contrast_estimates_comparison_minus_reference() -> None:
    development_dates = pd.date_range("2019-11-01", periods=20, freq="B")
    evaluation_dates = pd.date_range("2020-01-02", periods=20, freq="B")
    common_variation = np.sin(np.arange(20)) * 0.002
    daily = pd.DataFrame(
        {
            "session_date": [*development_dates, *evaluation_dates],
            "regime": ["development"] * 20 + ["evaluation"] * 20,
            "beta_negative_share": [
                *(-0.01 + common_variation),
                *(-0.03 + common_variation),
            ],
        }
    )

    result = hac_regime_mean_contrast(daily, hac_lags=2)

    assert result["contrast"] == "evaluation_minus_development"
    assert result["reference_estimate"] == pytest.approx(
        float((-0.01 + common_variation).mean())
    )
    assert result["comparison_estimate"] == pytest.approx(
        float((-0.03 + common_variation).mean())
    )
    assert result["estimate"] == pytest.approx(-0.02)
    assert result["ci_high"] < 0
    assert result["n_reference"] == 20
    assert result["n_comparison"] == 20


def test_conditional_rank_test_can_omit_constant_story_count_control() -> None:
    rows = []
    for date_number, session in enumerate(pd.date_range("2020-01-02", periods=12, freq="B")):
        for name in range(20):
            mean = ((name * 7 + date_number) % 20) / 19
            negative_share = name / 19
            outcome = -2.0 * negative_share + 0.4 * mean + 0.01 * name
            rows.append(
                {
                    "session_date": session,
                    "mean_continuous": mean,
                    "negative_share": negative_share,
                    "n": 1,
                    "return": outcome,
                }
            )
    frame = pd.DataFrame(rows)

    summary, daily, audit = conditional_negative_share_test(
        frame,
        date_col="session_date",
        outcome_col="return",
        count_col=None,
        min_names=10,
        hac_lags=2,
    )

    assert len(daily) == 12
    assert audit["dates_used"] == 12
    assert summary["estimate"] < 0
    assert summary["ci_high"] < 0
    assert daily["beta_log_story_count"].isna().all()
