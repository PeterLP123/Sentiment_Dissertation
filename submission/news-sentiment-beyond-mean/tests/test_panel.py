"""Focused tests for experiments panel joins — silent errors corrupt everything downstream."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.lib.panel import (
    attach_close_returns,
    attach_open_returns,
    map_earnings_to_sessions,
    session_open_close_return_legs,
)


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


def test_close_returns_use_the_next_session_not_the_next_news_day() -> None:
    dates = pd.bdate_range("2020-01-06", periods=5)
    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "session_date": d, "adjusted_close": 100.0 + i}
            for i, d in enumerate(dates)
        ]
        + [{"symbol": "SPY", "session_date": d, "adjusted_close": 200.0} for d in dates]
    )
    panel = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "session_date": [dates[0], dates[3]],
        }
    )
    out = attach_close_returns(panel, prices).sort_values("session_date").reset_index(drop=True)
    assert abs(float(out.loc[0, "next_adjusted_close"]) - 101.0) < 1e-12
    assert abs(float(out.loc[0, "ret_close_h1"]) - 0.01) < 1e-12
    assert abs(float(out.loc[0, "ar_close_h1"]) - 0.01) < 1e-12


def test_close_returns_do_not_jump_over_a_missing_exchange_session() -> None:
    dates = pd.bdate_range("2020-01-06", periods=4)
    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "session_date": dates[0], "adjusted_close": 100.0},
            {"symbol": "AAA", "session_date": dates[2], "adjusted_close": 103.0},
        ]
        + [{"symbol": "SPY", "session_date": d, "adjusted_close": 200.0} for d in dates]
    )
    panel = pd.DataFrame({"symbol": ["AAA"], "session_date": [dates[0]]})
    out = attach_close_returns(panel, prices)
    assert out.loc[0, "return_end_date"] == dates[1]
    assert pd.isna(out.loc[0, "next_adjusted_close"])
    assert pd.isna(out.loc[0, "ret_close_h1"])


def test_open_close_legs_use_market_sessions_and_reconcile() -> None:
    dates = pd.bdate_range("2020-01-06", periods=3)
    prices = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session_date": dates[0],
                "adjusted_open": 99.0,
                "adjusted_close": 100.0,
            },
            {
                "symbol": "AAA",
                "session_date": dates[1],
                "adjusted_open": 102.0,
                "adjusted_close": 103.0,
            },
            {
                "symbol": "AAA",
                "session_date": dates[2],
                "adjusted_open": 105.0,
                "adjusted_close": 104.0,
            },
            *[
                {
                    "symbol": "SPY",
                    "session_date": date,
                    "adjusted_open": 200.0 + index,
                    "adjusted_close": 200.5 + index,
                }
                for index, date in enumerate(dates)
            ],
        ]
    )

    legs = session_open_close_return_legs(prices)
    row = legs.loc[
        legs["symbol"].eq("AAA") & legs["session_date"].eq(dates[1])
    ].iloc[0]
    assert row["ret_previous_open_to_assigned_open"] == pytest.approx(102.0 / 99.0 - 1.0)
    assert row["ret_pre_open_overnight"] == pytest.approx(102.0 / 100.0 - 1.0)
    assert row["ret_assigned_session_intraday"] == pytest.approx(103.0 / 102.0 - 1.0)
    assert row["ret_post_close_overnight"] == pytest.approx(105.0 / 103.0 - 1.0)
    assert row["ret_assigned_open_to_next_open"] == pytest.approx(105.0 / 102.0 - 1.0)
    compounded = (
        (1.0 + row["ret_assigned_session_intraday"])
        * (1.0 + row["ret_post_close_overnight"])
        - 1.0
    )
    assert row["ret_assigned_open_to_next_open"] == pytest.approx(compounded)
    assert row["ar_assigned_session_intraday"] == pytest.approx(
        row["ret_assigned_session_intraday"] - row["spy_ret_assigned_session_intraday"]
    )


def test_open_close_legs_do_not_jump_over_missing_next_session() -> None:
    dates = pd.bdate_range("2020-01-06", periods=3)
    prices = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session_date": dates[0],
                "adjusted_open": 100.0,
                "adjusted_close": 101.0,
            },
            {
                "symbol": "AAA",
                "session_date": dates[2],
                "adjusted_open": 104.0,
                "adjusted_close": 105.0,
            },
            *[
                {
                    "symbol": "SPY",
                    "session_date": date,
                    "adjusted_open": 200.0,
                    "adjusted_close": 200.0,
                }
                for date in dates
            ],
        ]
    )

    row = session_open_close_return_legs(prices).loc[
        lambda frame: frame["symbol"].eq("AAA") & frame["session_date"].eq(dates[0])
    ].iloc[0]
    assert row["next_session_date"] == dates[1]
    assert pd.isna(row["next_adjusted_open"])
    assert pd.isna(row["ret_post_close_overnight"])
    assert pd.isna(row["ret_assigned_open_to_next_open"])
