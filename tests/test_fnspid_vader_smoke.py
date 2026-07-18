from __future__ import annotations

import math

import pandas as pd

from sentiment_benchmark.fnspid_vader_smoke import (
    _benjamini_hochberg,
    _SessionMapper,
    attach_abnormal_returns,
    is_price_recap,
)


def test_price_recap_regex_covers_frozen_examples() -> None:
    positives = [
        "Shares surge after earnings",
        "Stock movers: Acme drops 7%",
        "Premarket top gainers",
        "Acme hits a 52-week high",
        "Market wrap: banks rise",
        "Acme shares are down after hours",
    ]
    assert all(is_price_recap(value) for value in positives)
    assert not is_price_recap("Acme appoints a new chief financial officer")


def test_session_mapper_is_strict_for_date_only_and_close_aware_for_timed_rows() -> None:
    mapper = _SessionMapper()
    assert mapper.map("2020-01-03 00:00:00 UTC") == ("2020-01-06", False)
    assert mapper.map("2020-01-03 15:00:00 UTC") == ("2020-01-03", True)
    assert mapper.map("2020-01-03 22:00:00 UTC") == ("2020-01-06", True)


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    adjusted = _benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert adjusted == [0.04, 0.05333333333333334, 0.05333333333333334, 0.2]


def test_abnormal_return_horizon_uses_future_market_session() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    prices = {
        "SPY": pd.DataFrame({"session_date": dates, "close": [1, 1, 1, 1], "adjusted_close": [100, 101, 102, 103]}),
        "AAA": pd.DataFrame({"session_date": dates, "close": [1, 1, 1, 1], "adjusted_close": [50, 51, 53, 54]}),
    }
    sessions = pd.DataFrame({"symbol": ["AAA"], "session_date": [pd.Timestamp("2020-01-03")], "vader_mean": [0.5]})
    result = attach_abnormal_returns(sessions, prices, market_symbol="SPY", horizons=(1, 2))
    expected_h1 = math.log(53 / 51) - math.log(102 / 101)
    expected_h2 = math.log(54 / 53) - math.log(103 / 102)
    assert math.isclose(result.iloc[0]["ar_h1"], expected_h1)
    assert math.isclose(result.iloc[0]["ar_h2"], expected_h2)
