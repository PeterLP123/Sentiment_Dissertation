from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.lib.risk_overlay import (
    MomentumOverlayConfig,
    apply_sentiment_brake,
    backtest_weight_panel,
    build_adjusted_open_returns,
    build_market_negative_pressure,
    build_market_risk_exposure,
    build_monthly_momentum_weights,
)


def _synthetic_prices() -> pd.DataFrame:
    sessions = pd.bdate_range("2020-01-02", periods=80)
    rates = {"AAA": 0.004, "BBB": 0.002, "CCC": -0.001, "DDD": -0.003, "SPY": 0.001}
    rows = []
    for symbol, rate in rates.items():
        for i, session in enumerate(sessions):
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": session,
                    "adjusted_open": 100.0 * (1.0 + rate) ** i,
                }
            )
    return pd.DataFrame(rows)


def test_monthly_momentum_is_lagged_dense_and_dollar_neutral() -> None:
    prices = _synthetic_prices()
    sessions = pd.DatetimeIndex(sorted(prices.loc[prices["symbol"].eq("SPY"), "session_date"].unique()))
    config = MomentumOverlayConfig(
        lookback_sessions=5,
        skip_sessions=1,
        tail_fraction=0.25,
        min_universe=4,
        bootstrap_replications=9,
    )

    weights, formations = build_monthly_momentum_weights(
        prices,
        sessions,
        config=config,
    )

    assert not weights.empty
    formed = formations.loc[formations["status"].eq("formed")]
    assert (formed["lookback_end"] < formed["formation_date"]).all()
    assert (formed["lookback_start"] < formed["lookback_end"]).all()
    by_day = weights.groupby("session_date")["weight"].agg(
        net="sum", gross=lambda values: values.abs().sum()
    )
    assert np.allclose(by_day["net"], 0.0)
    assert np.allclose(by_day["gross"], 1.0)
    assert set(weights.loc[weights["weight"].gt(0), "symbol"]) == {"AAA"}
    assert set(weights.loc[weights["weight"].lt(0), "symbol"]) == {"DDD"}


def test_single_asset_return_uses_the_immediate_dense_session() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "session_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "adjusted_open": [100.0, 102.0, 101.0],
        }
    )

    returns = build_adjusted_open_returns(prices, symbol="SPY", return_col="spy_return")

    assert returns["return_end_date"].tolist() == pd.to_datetime(["2020-01-03", "2020-01-06"]).tolist()
    assert np.allclose(returns["spy_return"], [0.02, 101.0 / 102.0 - 1.0])


def test_sentiment_brakes_preserve_neutrality_without_levering_up() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    base = pd.DataFrame(
        [
            {"session_date": date, "symbol": symbol, "weight": weight}
            for date in dates
            for symbol, weight in [("AAA", -0.25), ("BBB", -0.25), ("CCC", 0.25), ("DDD", 0.25)]
        ]
    )
    flags = pd.DataFrame(
        {
            "session_date": [dates[0], dates[0]],
            "symbol": ["AAA", "CCC"],
            "risk_flag": [True, True],
        }
    )

    symmetric = apply_sentiment_brake(base, flags, mode="symmetric")
    long_only = apply_sentiment_brake(base, flags, mode="long_only")

    for frame in (symmetric, long_only):
        summary = frame.groupby("session_date")["weight"].agg(
            net="sum", gross=lambda values: values.abs().sum()
        )
        assert np.allclose(summary["net"], 0.0)
        assert summary["gross"].le(1.0 + 1e-12).all()
        assert (frame["weight"].abs() <= frame["base_weight"].abs() + 1e-12).all()
    assert symmetric.loc[symmetric["session_date"].eq(dates[0]), "weight"].abs().sum() == 0.5
    assert long_only.loc[long_only["session_date"].eq(dates[0]), "weight"].abs().sum() == 0.5
    assert symmetric.loc[symmetric["session_date"].eq(dates[1]), "weight"].abs().sum() == 1.0


def test_backtest_charges_entry_and_final_liquidation() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    weights = pd.DataFrame(
        [
            {"session_date": date, "symbol": symbol, "weight": weight}
            for date in dates
            for symbol, weight in [("AAA", 0.5), ("BBB", -0.5)]
        ]
    )
    returns = pd.DataFrame(
        [
            {"session_date": date, "symbol": symbol, "ar": 0.0}
            for date in dates
            for symbol in ["AAA", "BBB"]
        ]
    )

    daily = backtest_weight_panel(weights, returns, cost_bps_per_side=10.0)

    assert np.allclose(daily["turnover"], [0.5, 0.5])
    assert np.allclose(daily["cost"], [0.001, 0.001])
    assert np.isclose(daily["net_return"].sum(), -0.002)


def test_market_pressure_uses_only_prior_sessions_for_its_baseline() -> None:
    sessions = pd.bdate_range("2020-01-02", periods=6)
    firm_day = pd.DataFrame(
        {
            "session_date": sessions,
            "negative_share": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "n": [10] * 6,
        }
    )

    pressure = build_market_negative_pressure(
        firm_day,
        sessions,
        trailing_window=3,
        min_periods=2,
    )

    fourth = pressure.iloc[3]
    assert np.isclose(fourth["trailing_mean"], np.mean([0.0, 0.2, 0.4]))
    assert np.isclose(fourth["negative_pressure"], 0.6)
    assert fourth["negative_pressure_z"] > 0


def test_market_hysteresis_waits_for_exit_threshold() -> None:
    pressure = pd.DataFrame(
        {
            "session_date": pd.bdate_range("2020-01-02", periods=6),
            "negative_pressure_z": [np.nan, 1.6, 1.0, 0.6, 0.5, 2.0],
        }
    )

    one_day = build_market_risk_exposure(pressure, mode="one_day")
    hysteresis = build_market_risk_exposure(pressure, mode="hysteresis")

    assert one_day["risk_off"].tolist() == [False, True, False, False, False, True]
    assert hysteresis["risk_off"].tolist() == [False, True, True, True, False, True]
    assert set(hysteresis["exposure"]) == {0.25, 1.0}
