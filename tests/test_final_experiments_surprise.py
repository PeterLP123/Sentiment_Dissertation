"""Focused tests for sentiment-surprise demeaning and OOS horse race."""

from __future__ import annotations

import numpy as np
import pandas as pd

from final_experiments.lib.surprise import (
    apply_demeaning_stack,
    nested_oos_horse_race,
    oos_r2_comparison,
    prior_session_return,
    trailing_firm_baseline,
    trailing_market_model_ar,
)


def test_trailing_firm_baseline_excludes_today() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * 6,
            "session_date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-08",
                    "2020-01-09",
                ]
            ),
            "level": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    base = trailing_firm_baseline(frame, level_col="level", news_days=3, min_periods=2)
    assert pd.isna(base.iloc[0])
    assert pd.isna(base.iloc[1])  # only 1 prior < min_periods 2
    assert abs(base.iloc[2] - 0.5) < 1e-12  # mean(0,1)
    assert abs(base.iloc[5] - 3.0) < 1e-12  # mean(2,3,4)


def test_demeaning_stack_peels_firm_then_cs() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "session_date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]
            ),
            "mean_continuous": [1.0, 3.0, 1.0, 5.0],
            "split": ["development"] * 4,
        }
    )
    out = apply_demeaning_stack(frame)
    # Day 2 CS mean of surprise_firm after firm baseline with min_periods=5 →
    # early rows have NaN baseline; use longer series:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * 6 + ["BBB"] * 6,
            "session_date": pd.to_datetime(
                [f"2020-01-{d:02d}" for d in range(2, 8)] * 2
            ),
            "mean_continuous": [1, 1, 1, 1, 1, 3] + [1, 1, 1, 1, 1, 7],
            "split": ["development"] * 12,
        }
    )
    out = apply_demeaning_stack(frame)
    last = out.loc[out["session_date"] == "2020-01-07"]
    assert last["surprise_full"].notna().all()
    # Idiosyncratic sums to ~0 cross-sectionally on that day.
    assert abs(float(last["surprise_cs"].sum())) < 1e-10


def test_market_model_recovers_known_beta() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", periods=200)
    mkt = rng.normal(0, 0.01, size=len(dates))
    stock = 0.001 + 1.5 * mkt + rng.normal(0, 0.001, size=len(dates))
    # Build price levels from returns.
    spy_px = 100 * np.cumprod(1 + mkt)
    aaa_px = 50 * np.cumprod(1 + stock)
    prices = pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates) + ["AAA"] * len(dates),
            "session_date": list(dates) * 2,
            "adjusted_open": np.concatenate([spy_px, aaa_px]),
        }
    )
    news = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "session_date": [dates[-2]],
            "mean_continuous": [0.1],
        }
    )
    out = trailing_market_model_ar(
        news, prices, window=120, gap=21, min_obs=80, market_symbol="SPY"
    )
    assert out["ar_mm_h1"].notna().iloc[0]
    assert abs(out["market_beta"].iloc[0] - 1.5) < 0.15


def test_market_model_outcome_starts_at_the_news_session() -> None:
    dates = pd.bdate_range("2020-01-02", periods=8)
    # Prior returns have beta one. The event-day forward stock move is +10%
    # while the market is flat, so the abnormal outcome must reflect +10%.
    spy = [100.0, 101.0, 100.0, 102.0, 101.0, 100.0, 100.0, 100.0]
    aaa = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 115.5, 115.5]
    prices = pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates) + ["AAA"] * len(dates),
            "session_date": list(dates) * 2,
            "adjusted_open": spy + aaa,
        }
    )
    news = pd.DataFrame({"symbol": ["AAA"], "session_date": [dates[5]]})
    out = trailing_market_model_ar(
        news,
        prices,
        window=4,
        gap=1,
        min_obs=4,
        market_symbol="SPY",
    )
    assert abs(float(out.loc[0, "ret_h1"]) - 0.10) < 1e-12
    assert out.loc[0, "return_end_date"] == dates[6]


def test_prior_return_ends_at_the_news_session() -> None:
    dates = pd.bdate_range("2020-01-02", periods=4)
    prices = pd.DataFrame(
        {
            "symbol": ["SPY"] * 4 + ["AAA"] * 4,
            "session_date": list(dates) * 2,
            "adjusted_open": [100.0] * 4 + [100.0, 102.0, 112.2, 112.2],
        }
    )
    news = pd.DataFrame({"symbol": ["AAA"], "session_date": [dates[2]]})
    control = prior_session_return(news, prices)
    # dates[1] -> dates[2] is +10%; dates[0] -> dates[1] was only +2%.
    assert abs(float(control.iloc[0]) - 0.10) < 1e-12


def test_oos_horse_race_prefers_true_signal() -> None:
    rng = np.random.default_rng(1)
    n = 400
    dates = pd.bdate_range("2018-01-01", periods=n)
    level = rng.normal(size=n)
    surprise = rng.normal(size=n)
    # Outcome depends on surprise, not level; split mid-sample.
    y = 0.01 * surprise + rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame(
        {
            "session_date": dates,
            "symbol": ["AAA"] * n,
            "split": ["development"] * 250 + ["evaluation"] * 150,
            "ret_lag1": rng.normal(scale=0.01, size=n),
            "level": level,
            "surprise_full": surprise,
            "surprise_firm": surprise,
            "ar_mm_h1": y,
        }
    )
    result, meta = nested_oos_horse_race(frame)
    assert meta["evaluation_date_clusters"] == 150
    assert meta["eval_clusters_pass"] is False  # 150 < 280 gate
    oos = result.drop_duplicates("model").set_index("model")["oos_r2"]
    assert oos["M_surprise"] > oos["M_level"]
    assert set(result["coefficient_sample"]) == {"development"}
    # The null nest must be present, or none of the other R2 values are attributable.
    assert oos["M_surprise"] > oos["M_control_only"]


def test_oos_r2_comparison_separates_real_gain_from_noise() -> None:
    rng = np.random.default_rng(3)
    n = 900
    dates = pd.bdate_range("2018-01-01", periods=n)
    surprise = rng.normal(size=n)
    noise_level = rng.normal(size=n)
    y = 0.02 * surprise + rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame(
        {
            "session_date": dates,
            "symbol": ["AAA"] * n,
            "split": ["development"] * 500 + ["evaluation"] * 400,
            "ret_lag1": rng.normal(scale=0.01, size=n),
            "level": noise_level,
            "surprise_full": surprise,
            "surprise_firm": surprise,
            "ar_mm_h1": y,
        }
    )
    table = oos_r2_comparison(frame, replications=299, seed=11).set_index("model")
    assert table.loc["M_control_only", "delta_vs_baseline"] == 0.0
    # A genuine predictor's gain over the null nest excludes zero...
    assert table.loc["M_surprise", "delta_ci_low"] > 0.0
    # ...while a pure-noise predictor's does not.
    assert table.loc["M_level", "delta_ci_low"] < 0.0 < table.loc["M_level", "delta_ci_high"]
    # p is floored at the bootstrap's own resolution, never reported as exactly 0.
    assert table.loc["M_surprise", "delta_p_two_sided"] >= 1.0 / 299
