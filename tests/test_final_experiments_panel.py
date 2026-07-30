"""Focused tests for final_experiments panel joins — silent errors corrupt everything downstream."""

from __future__ import annotations

import pandas as pd

from final_experiments.lib.panel import map_earnings_to_sessions


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
