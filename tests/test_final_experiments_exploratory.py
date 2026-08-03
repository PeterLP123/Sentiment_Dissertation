from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from final_experiments.lib.earnings import attach_earnings_distance, map_earnings_sessions
from final_experiments.lib.lseg_robustness import attach_next_open_returns, lseg_ic_table
from final_experiments.lib.thresholds import (
    choose_fixed_band,
    choose_probability_cutoff,
    gate_daily_portfolio,
)


def test_earnings_mapping_and_event_time_sign() -> None:
    sessions = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    calendar = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": ["2024-01-03"],
            "session_rule": ["same_session"],
            "timing_flag": ["BMO"],
        }
    )
    mapped = map_earnings_sessions(calendar, sessions)
    assert mapped.loc[0, "earnings_session"] == pd.Timestamp("2024-01-03")

    firm_day = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "session_date": sessions,
        }
    )
    attached = attach_earnings_distance(firm_day, mapped).sort_values("session_date")
    assert attached["sessions_to_earnings"].tolist() == [-1, 0, 1]
    assert attached["earnings_window"].tolist() == ["pre", "event", "post"]


def test_next_session_rule_is_strict() -> None:
    sessions = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    calendar = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": ["2024-01-03"],
            "session_rule": ["next_session"],
            "timing_flag": ["AMC"],
        }
    )
    mapped = map_earnings_sessions(calendar, sessions)
    assert mapped.loc[0, "earnings_session"] == pd.Timestamp("2024-01-04")


def test_lseg_date_only_news_uses_strictly_later_open() -> None:
    panel = pd.DataFrame({"symbol": ["AAA"], "news_date": ["2024-01-02"]})
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "session_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "adjusted_open": [100.0, 110.0, 121.0],
        }
    )
    attached = attach_next_open_returns(panel, prices)
    assert attached.loc[0, "entry_session"] == pd.Timestamp("2024-01-03")
    assert attached.loc[0, "raw_open_h1"] == pytest.approx(0.10)


def test_probability_cutoff_cannot_win_by_almost_never_trading() -> None:
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "session_date": np.repeat(dates, 10),
            "symbol": [f"S{i:02d}" for _ in dates for i in range(10)],
            "oriented_rank": np.tile(np.linspace(-1.0, 1.0, 10), len(dates)),
            "ar_open_h1": np.tile(np.linspace(-0.01, 0.01, 10), len(dates)),
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
        }
    )
    daily = gate_daily_portfolio(frame, np.ones(4, dtype=bool), cost_bps_per_side=10.0)
    # Half-L1 turnover is 0.5 to open and 0.5 to close. With the codebase's
    # per-side convention this is a 20 bps round trip on gross notional.
    assert daily.loc[0, "turnover"] == pytest.approx(1.0)
    assert daily.loc[0, "cost"] == pytest.approx(0.002)
    assert daily.loc[0, "net_return"] == pytest.approx(-0.002)


def test_fixed_band_can_use_the_same_activity_floor_as_learned_gates() -> None:
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "session_date": np.repeat(dates, 10),
            "symbol": [f"S{i:02d}" for _ in dates for i in range(10)],
            "oriented_rank": np.tile(np.linspace(-1.0, 1.0, 10), len(dates)),
            "abs_oriented_rank": np.tile(np.abs(np.linspace(-1.0, 1.0, 10)), len(dates)),
            "ar_open_h1": np.tile(np.linspace(-0.01, 0.01, 10), len(dates)),
        }
    )
    _, sweep = choose_fixed_band(frame, require_activity=True)
    assert bool(sweep.loc[sweep["band"] == 0.0, "selection_eligible"].iloc[0])
    assert not bool(sweep.loc[sweep["band"] == 0.8, "selection_eligible"].iloc[0])


def test_threshold_gate_does_not_trade_a_one_sided_selection() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02"] * 4),
            "symbol": list("ABCD"),
            "oriented_rank": [-1.0, -0.5, 0.5, 1.0],
            "ar_open_h1": [0.01, 0.005, -0.005, -0.01],
        }
    )
    daily = gate_daily_portfolio(frame, np.array([False, False, True, True]))
    assert daily.loc[0, "n_active"] == 0
    assert daily.loc[0, "net_exposure"] == 0.0


def test_lseg_ic_excludes_dense_panel_no_news_rows() -> None:
    frame = pd.DataFrame(
        {
            "entry_session": pd.to_datetime(["2024-01-03"] * 4 + ["2024-01-04"] * 4),
            "headline_count": [1, 1, 1, 0, 1, 1, 1, 0],
            "mean_sentiment_score": [-1.0, 0.0, 1.0, 99.0] * 2,
            "raw_open_h1": [-0.01, 0.0, 0.01, -0.99] * 2,
        }
    )
    result = lseg_ic_table(
        frame,
        arms=("mean_sentiment_score",),
        family="test family",
    )
    assert int(result.loc[0, "n"]) == 6
    assert int(result.loc[0, "n_clusters"]) == 2
