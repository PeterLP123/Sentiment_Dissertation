"""Focused tests for the overlapping-tranche strategy backtest.

Only the mechanics that would silently corrupt every strategy number are
covered: tranche weighting, turnover, market adjustment, and drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from final_experiments.lib.strategy import (
    StrategyConfig,
    _tranche_weights,
    build_daily_ar_panel,
    event_time_car,
    event_time_spread_inference,
    max_drawdown,
    monthly_return_matrix,
    run_backtest,
    select_frozen_spec,
    yearly_table,
)


def _prices(n_sessions: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2019-09-02", periods=n_sessions)
    rows = []
    for symbol, drift in (("SPY", 0.0), ("AAA", 0.002), ("BBB", -0.002)):
        level = 100.0
        for date in dates:
            rows.append({"symbol": symbol, "session_date": date, "adjusted_open": level})
            level *= 1.0 + drift
    return pd.DataFrame(rows)


def test_tranche_weights_average_the_last_h_books() -> None:
    book = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert np.allclose(_tranche_weights(book, 1), book)
    three = _tranche_weights(book, 3)
    # Row 0 sees one real book and two empty pre-history rows.
    assert np.allclose(three[0], [1 / 3, 0.0])
    assert np.allclose(three[1], [1 / 3, 1 / 3])
    assert np.allclose(three[2], [2 / 3, 2 / 3])


def test_daily_ar_panel_removes_the_market() -> None:
    ar = build_daily_ar_panel(_prices(), market_symbol="SPY")
    assert set(ar["symbol"]) == {"AAA", "BBB"}
    # SPY is flat, so AAA's abnormal return is its own drift.
    assert abs(float(ar.loc[ar["symbol"] == "AAA", "ar"].iloc[0]) - 0.002) < 1e-9
    assert abs(float(ar.loc[ar["symbol"] == "BBB", "ar"].iloc[0]) + 0.002) < 1e-9
    assert ar["ar"].notna().all()
    assert (ar["return_end_date"] > ar["session_date"]).all()


def test_daily_ar_does_not_jump_over_a_missing_exchange_session() -> None:
    dates = pd.bdate_range("2019-01-02", periods=4)
    prices = pd.DataFrame(
        [{"symbol": "SPY", "session_date": d, "adjusted_open": 100.0} for d in dates]
        + [
            {"symbol": "AAA", "session_date": dates[0], "adjusted_open": 100.0},
            {"symbol": "AAA", "session_date": dates[2], "adjusted_open": 110.0},
            {"symbol": "AAA", "session_date": dates[3], "adjusted_open": 110.0},
        ]
    )
    ar = build_daily_ar_panel(prices)
    assert not ((ar["symbol"] == "AAA") & (ar["session_date"] == dates[0])).any()


def _firm_day(n_sessions: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2019-09-02", periods=n_sessions)
    rows = []
    for date in dates:
        # AAA always ranks high, BBB always low, so the book is stable.
        rows.append({"symbol": "AAA", "session_date": date, "signal": 1.0, "split": "development"})
        rows.append({"symbol": "BBB", "session_date": date, "signal": -1.0, "split": "development"})
    return pd.DataFrame(rows)


def test_backtest_earns_the_spread_and_is_dollar_neutral() -> None:
    config = StrategyConfig(signal="signal", horizon=1, cost_bps_per_side=0.0, min_names=2)
    result = run_backtest(_firm_day(), build_daily_ar_panel(_prices()), config, split="development")
    daily = result.daily
    assert not daily.empty
    # Long AAA (+0.002) / short BBB (-0.002) at half weight each => +0.002 a day.
    assert abs(float(daily["gross_return"].iloc[5]) - 0.002) < 1e-9
    assert abs(float(daily["net_exposure"].abs().max())) < 1e-12
    assert result.summary["sharpe_net"] > 0
    assert result.summary["max_missing_weight_share"] == 0.0


def test_longer_horizon_cuts_turnover() -> None:
    # The book must actually churn for this to mean anything: flip the ranking
    # every session so a one-day hold reverses the whole book daily.
    dates = pd.bdate_range("2019-09-02", periods=60)
    rows = []
    for i, date in enumerate(dates):
        flip = 1.0 if i % 2 == 0 else -1.0
        rows.append({"symbol": "AAA", "session_date": date, "signal": flip, "split": "development"})
        rows.append({"symbol": "BBB", "session_date": date, "signal": -flip, "split": "development"})
    firm_day = pd.DataFrame(rows)
    ar = build_daily_ar_panel(_prices(60))

    turnovers = {}
    for horizon in (1, 5):
        config = StrategyConfig(
            signal="signal", horizon=horizon, cost_bps_per_side=0.0, min_names=2
        )
        result = run_backtest(firm_day, ar, config, split="development")
        # Ignore the ramp: the first `horizon` sessions are still building the book.
        turnovers[horizon] = float(result.daily["turnover"].iloc[horizon + 1 :].mean())
    assert turnovers[1] > 0.0
    assert turnovers[5] < turnovers[1]


def test_tied_signal_with_breadth_cut_stays_neutral_or_does_not_trade() -> None:
    """A big tie block must not produce a one-sided book.

    `dispersion` is exactly 0 on every single-story firm-day — half the panel.
    Oriented negative, that tie sits at the top of the centred rank, so a breadth
    cut keeps only names below it. Scaling by total gross would have produced a
    ~100% short book; per-leg normalisation must give a neutral book or no trade.
    """
    dates = pd.bdate_range("2019-09-02", periods=30)
    rows = []
    for date in dates:
        for i in range(12):  # 12 names tied at zero
            rows.append({"symbol": f"T{i}", "session_date": date, "signal": 0.0,
                         "split": "development"})
        for i, value in enumerate([0.1, 0.4, 0.9]):  # 3 names with real dispersion
            rows.append({"symbol": f"D{i}", "session_date": date, "signal": value,
                         "split": "development"})
    firm_day = pd.DataFrame(rows)

    price_rows = []
    for symbol in ["SPY"] + [f"T{i}" for i in range(12)] + [f"D{i}" for i in range(3)]:
        level = 100.0
        for date in dates:
            price_rows.append({"symbol": symbol, "session_date": date, "adjusted_open": level})
            level *= 1.001
    ar = build_daily_ar_panel(pd.DataFrame(price_rows), market_symbol="SPY")

    for breadth in (0.0, 0.5, 0.8):
        config = StrategyConfig(
            signal="signal", horizon=1, breadth=breadth, orient=-1.0,
            cost_bps_per_side=0.0, min_names=2,
        )
        result = run_backtest(firm_day, ar, config, split="development")
        if result.daily.empty:
            continue
        traded = result.daily.loc[result.daily["gross_exposure"] > 0]
        assert float(traded["net_exposure"].abs().max() if len(traded) else 0.0) < 1e-9, (
            f"breadth={breadth} produced a directional book"
        )


def test_max_drawdown_matches_a_hand_computed_curve() -> None:
    # Peak 1.2, trough 0.6 => -50%.
    assert abs(max_drawdown(np.array([1.0, 1.2, 0.6, 0.9])) + 0.5) < 1e-12
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0


def test_select_frozen_spec_applies_eligibility_before_ranking() -> None:
    sweep = pd.DataFrame(
        [
            # Best Sharpe but far too few sessions to be a portfolio.
            {"signal": "a", "horizon": 1, "breadth": 0.0, "sharpe_net": 9.0,
             "n_sessions": 10, "mean_positions": 50.0, "annualized_turnover": 100.0,
             "mean_net": 0.01, "breakeven_bps_per_side": 50.0},
            {"signal": "b", "horizon": 5, "breadth": 0.5, "sharpe_net": 1.0,
             "n_sessions": 900, "mean_positions": 40.0, "annualized_turnover": 50.0,
             "mean_net": 0.001, "breakeven_bps_per_side": 20.0},
        ]
    )
    spec = select_frozen_spec(sweep, min_sessions=250, min_positions=5.0)
    assert spec["signal"] == "b"
    assert spec["horizon"] == 5
    assert spec["n_cells_eligible"] == 1
    assert spec["selection_split"] == "development"


def test_development_horizon_never_uses_evaluation_returns() -> None:
    dates = pd.bdate_range("2019-12-16", "2020-01-10")
    price_rows = []
    for symbol in ("SPY", "AAA", "BBB"):
        level = 100.0
        for date in dates:
            price_rows.append({"symbol": symbol, "session_date": date, "adjusted_open": level})
            level *= 1.001
    ar = build_daily_ar_panel(pd.DataFrame(price_rows))
    rows = []
    for date in dates:
        split = "development" if date <= pd.Timestamp("2019-12-31") else "evaluation"
        rows.extend(
            [
                {"symbol": "AAA", "session_date": date, "signal": 1.0, "split": split},
                {"symbol": "BBB", "session_date": date, "signal": -1.0, "split": split},
            ]
        )
    result = run_backtest(
        pd.DataFrame(rows),
        ar,
        StrategyConfig(signal="signal", horizon=3, cost_bps_per_side=0.0, min_names=2),
        split="development",
    )
    assert not result.daily.empty
    assert result.daily["session_date"].max() < pd.Timestamp("2020-01-01")


def test_event_time_car_recovers_drift_that_stops() -> None:
    """A planted 3-session drift must show up in the top quantile and then flat-line.

    GOOD drifts +0.4%/session for its first three held sessions then tracks the
    market; BAD mirrors it. If lag alignment were off by one the CAR would keep
    climbing past lag 3 or start at zero.
    """
    n = 60
    dates = pd.bdate_range("2019-01-02", periods=n)
    drift_days = 3
    step = 0.004

    price_rows = []
    for symbol in ("SPY", "GOOD", "BAD", "FLATA", "FLATB"):
        level = 100.0
        for date in dates:
            price_rows.append({"symbol": symbol, "session_date": date, "adjusted_open": level})
            # Signal is formed every session, so drift has to be a constant per-session
            # rate for the event-time average to have a clean expected value.
            if symbol == "GOOD":
                level *= 1.0 + step
            elif symbol == "BAD":
                level *= 1.0 - step
    prices = pd.DataFrame(price_rows)

    firm_rows = []
    for date in dates:
        for symbol, value in (("GOOD", 2.0), ("FLATA", 0.5), ("FLATB", -0.5), ("BAD", -2.0)):
            firm_rows.append(
                {"symbol": symbol, "session_date": date, "signal": value, "split": "development"}
            )
    firm_day = pd.DataFrame(firm_rows)

    car = event_time_car(
        firm_day,
        build_daily_ar_panel(prices),
        signal="signal",
        n_quantiles=4,
        max_lag=6,
        split="development",
    )
    top = car.loc[car["quantile"] == 4].sort_values("lag")
    bottom = car.loc[car["quantile"] == 1].sort_values("lag")

    # Top quantile is GOOD: roughly +step per session, every session.
    assert abs(float(top.loc[top["lag"] == 1, "cum_ar"].iloc[0]) - step) < 5e-4
    assert abs(float(top.loc[top["lag"] == drift_days, "cum_ar"].iloc[0]) - drift_days * step) < 2e-3
    # Bottom quantile is BAD and mirrors it.
    assert float(bottom.loc[bottom["lag"] == drift_days, "cum_ar"].iloc[0]) < 0
    # Monotone in quantile at every lag: 4 above 1.
    for lag in (1, 3, 6):
        assert (
            float(top.loc[top["lag"] == lag, "cum_ar"].iloc[0])
            > float(bottom.loc[bottom["lag"] == lag, "cum_ar"].iloc[0])
        )


def test_event_time_spread_has_block_bootstrap_uncertainty() -> None:
    n = 80
    dates = pd.bdate_range("2019-01-02", periods=n)
    prices_rows = []
    for symbol, drift in (("SPY", 0.0), ("GOOD", 0.003), ("BAD", -0.003)):
        level = 100.0
        for date in dates:
            prices_rows.append(
                {"symbol": symbol, "session_date": date, "adjusted_open": level}
            )
            level *= 1.0 + drift
    firm_day = pd.DataFrame(
        [
            {"symbol": symbol, "session_date": date, "signal": value, "split": "development"}
            for date in dates
            for symbol, value in (("GOOD", 1.0), ("BAD", -1.0))
        ]
    )
    inference = event_time_spread_inference(
        firm_day,
        build_daily_ar_panel(pd.DataFrame(prices_rows)),
        signal="signal",
        n_quantiles=2,
        max_lag=3,
        block_length=5,
        replications=99,
        seed=7,
    )
    assert len(inference) == 3
    assert (inference["spread_cum_ar"] > 0).all()
    assert (inference["ci_low"] > 0).all()
    assert (inference["p_centered_block"] >= 1 / 99).all()


def test_monthly_return_matrix_compounds_within_month() -> None:
    daily = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03"]),
            "net_return": [0.10, 0.10, -0.05],
        }
    )
    matrix = monthly_return_matrix(daily)
    assert abs(float(matrix.loc[2020, 1]) - 0.21) < 1e-12  # 1.1 * 1.1 - 1
    assert abs(float(matrix.loc[2020, 2]) + 0.05) < 1e-12


def test_yearly_table_splits_by_calendar_year() -> None:
    daily = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2020-06-01", "2020-06-02", "2021-06-01"]),
            "net_return": [0.01, -0.01, 0.02],
            "turnover": [0.5, 0.5, 0.5],
        }
    )
    table = yearly_table(daily)
    assert table["year"].tolist() == [2020, 2021]
    assert table.loc[table["year"] == 2021, "sessions"].iloc[0] == 1
