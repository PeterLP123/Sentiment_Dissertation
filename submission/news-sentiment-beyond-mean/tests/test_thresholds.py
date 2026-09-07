"""Focused tests for threshold selection and portfolio accounting."""

import numpy as np
import pandas as pd
import pytest

from experiments.lib.evaluate import PortfolioInsolvencyError
from experiments.lib.thresholds import (
    choose_fixed_band,
    choose_probability_cutoff,
    gate_daily_portfolio,
)


def test_probability_cutoff_cannot_win_by_almost_never_trading() -> None:
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "session_date": np.repeat(dates, 10),
            "symbol": [f"S{i:02d}" for _ in dates for i in range(10)],
            "oriented_rank": np.tile(np.linspace(-1.0, 1.0, 10), len(dates)),
            "ar_open_h1": np.tile(np.linspace(-0.01, 0.01, 10), len(dates)),
            "ret_open_h1": np.tile(np.linspace(-0.01, 0.01, 10), len(dates)),
        }
    )
    probabilities = np.full(len(frame), 0.51)
    cutoff, sweep = choose_probability_cutoff(frame, probabilities)
    assert cutoff == 0.50
    assert bool(sweep.loc[sweep["cutoff"] == 0.50, "selection_eligible"].iloc[0])
    assert not bool(sweep.loc[sweep["cutoff"] == 0.525, "selection_eligible"].iloc[0])


def test_threshold_gate_remains_dollar_neutral_after_row_selection() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02"] * 5),
            "symbol": list("ABCDE"),
            "oriented_rank": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "ar_open_h1": [0.01, 0.005, 0.0, -0.005, -0.01],
            "ret_open_h1": [0.01, 0.005, 0.0, -0.005, -0.01],
        }
    )
    daily = gate_daily_portfolio(frame, np.array([True, False, False, True, True]))
    assert daily.loc[0, "n_active"] == 3
    assert daily.loc[0, "net_exposure"] == pytest.approx(0.0, abs=1e-12)


def test_threshold_gate_charges_final_liquidation() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02"] * 4),
            "symbol": list("ABCD"),
            "oriented_rank": [-1.0, -0.5, 0.5, 1.0],
            "ar_open_h1": [0.0, 0.0, 0.0, 0.0],
            "ret_open_h1": [0.0, 0.0, 0.0, 0.0],
        }
    )
    daily = gate_daily_portfolio(frame, np.ones(4, dtype=bool), cost_bps_per_side=10.0)
    assert daily.loc[0, "turnover"] == pytest.approx(1.0)
    assert daily.loc[0, "cost"] == pytest.approx(0.002)
    assert daily.loc[0, "net_return"] == pytest.approx(-0.002)


def test_fixed_band_uses_the_same_activity_floor_as_learned_gates() -> None:
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "session_date": np.repeat(dates, 10),
            "symbol": [f"S{i:02d}" for _ in dates for i in range(10)],
            "oriented_rank": np.tile(np.linspace(-1.0, 1.0, 10), len(dates)),
            "abs_oriented_rank": np.tile(np.abs(np.linspace(-1.0, 1.0, 10)), len(dates)),
            "ar_open_h1": np.tile(np.linspace(-0.01, 0.01, 10), len(dates)),
            "ret_open_h1": np.tile(np.linspace(-0.01, 0.01, 10), len(dates)),
        }
    )
    _, sweep = choose_fixed_band(frame, require_activity=True)
    assert bool(sweep.loc[sweep["band"] == 0.0, "selection_eligible"].iloc[0])
    assert not bool(sweep.loc[sweep["band"] == 0.8, "selection_eligible"].iloc[0])


def test_threshold_gate_does_not_trade_one_sided_selection() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02"] * 4),
            "symbol": list("ABCD"),
            "oriented_rank": [-1.0, -0.5, 0.5, 1.0],
            "ar_open_h1": [0.01, 0.005, -0.005, -0.01],
            "ret_open_h1": [0.01, 0.005, -0.005, -0.01],
        }
    )
    daily = gate_daily_portfolio(frame, np.array([False, False, True, True]))
    assert daily.loc[0, "n_active"] == 0
    assert daily.loc[0, "net_exposure"] == 0.0


def test_threshold_gate_reports_the_insolvency_session() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02"] * 2),
            "symbol": ["A", "B"],
            "oriented_rank": [1.0, -1.0],
            "ar_open_h1": [0.0, -3.0],
            "ret_open_h1": [0.0, 3.0],
        }
    )
    with pytest.raises(PortfolioInsolvencyError) as error:
        gate_daily_portfolio(frame, np.ones(2, dtype=bool))
    assert error.value.session == pd.Timestamp("2024-01-02")
    assert error.value.wealth_growth == pytest.approx(-0.5)
