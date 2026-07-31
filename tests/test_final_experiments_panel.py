"""Focused tests for final_experiments panel joins — silent errors corrupt everything downstream."""

from __future__ import annotations

import pandas as pd

from final_experiments.lib.panel import attach_open_returns, map_earnings_to_sessions


def test_open_returns_use_the_next_session_not_the_next_news_day() -> None:
    """The stock leg must come from the exchange calendar.

    AAA has news on day 0 and again only on day 4. Shifting inside the panel
    would make ``ret_open_h1`` a four-session return while SPY's leg stays one
    session, so the "abnormal return" would be a horizon mismatch.
    """
    dates = pd.bdate_range("2020-01-06", periods=5)
    prices = pd.DataFrame(
        [{"symbol": "AAA", "session_date": d, "adjusted_open": 100.0 + i} for i, d in enumerate(dates)]
        + [{"symbol": "SPY", "session_date": d, "adjusted_open": 200.0} for d in dates]
    )
    panel = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "session_date": [dates[0], dates[3]],
            "adjusted_open": [100.0, 103.0],
            "spy_adjusted_open": [200.0, 200.0],
        }
    )
    out = attach_open_returns(panel, prices).sort_values("session_date").reset_index(drop=True)
    # Day 0 -> day 1 is 100 -> 101, not 100 -> 103.
    assert abs(float(out.loc[0, "next_adjusted_open"]) - 101.0) < 1e-12
    assert abs(float(out.loc[0, "ret_open_h1"]) - 0.01) < 1e-12
    # SPY is flat, so the abnormal return is the stock's own one-session move.
    assert abs(float(out.loc[0, "spy_ret_open_h1"])) < 1e-12
    assert abs(float(out.loc[0, "ar_open_h1"]) - 0.01) < 1e-12


def test_open_returns_do_not_jump_over_a_missing_exchange_session() -> None:
    dates = pd.bdate_range("2020-01-06", periods=4)
    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "session_date": dates[0], "adjusted_open": 100.0},
            # AAA is missing on the exchange's next session.
            {"symbol": "AAA", "session_date": dates[2], "adjusted_open": 103.0},
        ]
        + [{"symbol": "SPY", "session_date": d, "adjusted_open": 200.0} for d in dates]
    )
    panel = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "session_date": [dates[0]],
            "adjusted_open": [100.0],
            "spy_adjusted_open": [200.0],
        }
    )
    out = attach_open_returns(panel, prices)
    assert out.loc[0, "return_end_date"] == dates[1]
    assert pd.isna(out.loc[0, "next_adjusted_open"])
    assert pd.isna(out.loc[0, "ret_open_h1"])


def test_earnings_bmo_maps_to_same_trading_session() -> None:
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": ["2020-01-06"],  # Monday
            "timing_flag": ["BMO"],
            "session_rule": ["same_session"],
            "fiscal_period": ["Q4 2019"],
        }
    )
    mapped = map_earnings_to_sessions(earnings)
    assert len(mapped) == 1
    assert mapped.iloc[0]["session_date"] == pd.Timestamp("2020-01-06")


def test_earnings_amc_maps_to_next_session() -> None:
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "report_date": ["2020-01-06"],  # Monday
            "timing_flag": ["AMC"],
            "session_rule": ["next_session"],
            "fiscal_period": ["Q4 2019"],
        }
    )
    mapped = map_earnings_to_sessions(earnings)
    assert mapped.iloc[0]["session_date"] == pd.Timestamp("2020-01-07")


def test_earnings_dedupes_same_symbol_report_date() -> None:
    earnings = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "report_date": ["2020-01-06", "2020-01-06"],
            "timing_flag": ["BMO", "BMO"],
            "session_rule": ["same_session", "same_session"],
            "fiscal_period": ["Q4 2019", "Q4 2019 dup"],
        }
    )
    mapped = map_earnings_to_sessions(earnings)
    assert len(mapped) == 1


def test_adjusted_open_formula() -> None:
    open_, close, adj_close = 100.0, 200.0, 50.0
    adjusted_open = open_ * (adj_close / close)
    assert adjusted_open == 25.0
