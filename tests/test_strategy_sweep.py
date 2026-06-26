"""Tests for the parameter sweep, focusing on train/test selection discipline."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sentiment_benchmark.backtest import DailySignal
from sentiment_benchmark.prices import PriceRow
from sentiment_benchmark.strategy_sweep import (
    ParameterGrid,
    chronological_split_date,
    load_prices_csv,
    load_signals_csv,
    sweep,
)

# Per-date horizon-1 return is driven by the entry session's open->close move.
_PRICE_MOVES = {
    "2026-06-08": (100.0, 100.0),
    "2026-06-09": (100.0, 110.0),  # entry for the 06-08 signal: +10%
    "2026-06-10": (100.0, 102.0),  # entry for the 06-09 signal: +2%
    "2026-06-11": (100.0, 100.0),
    "2026-06-12": (100.0, 100.0),
    "2026-06-15": (100.0, 100.0),
    "2026-06-16": (100.0, 102.0),  # entry for the 06-15 signal: +2%
    "2026-06-17": (100.0, 110.0),  # entry for the 06-16 signal: +10%
    "2026-06-18": (100.0, 100.0),
    "2026-06-19": (100.0, 100.0),
}


def _prices() -> list[PriceRow]:
    return [
        PriceRow("AAPL", day, open_, max(open_, close), min(open_, close), close, 1_000.0, False)
        for day, (open_, close) in _PRICE_MOVES.items()
    ]


def _signal(news_date: str, mean_score: float) -> DailySignal:
    return DailySignal("AAPL", news_date, "model", 3, 3, mean_score, "positive", 1)


def _signals() -> list[DailySignal]:
    # Train (before 06-15): big move pairs with low score; Test: big move pairs with high score.
    return [
        _signal("2026-06-08", 0.5),  # +10%, traded only at low threshold
        _signal("2026-06-09", 2.0),  # +2%, traded at both thresholds
        _signal("2026-06-15", 0.5),  # +2%, test
        _signal("2026-06-16", 2.0),  # +10%, test
    ]


def test_parameter_grid_enumerates_full_product() -> None:
    grid = ParameterGrid(thresholds=(0.0, 0.5), horizons=(1, 3), min_valid_stories=(1, 2))
    assert len(list(grid.points())) == 2 * 2 * 2 * 1


def test_chronological_split_date_by_fraction() -> None:
    dates = [f"2026-06-{day:02d}" for day in range(1, 11)]  # 10 distinct dates
    assert chronological_split_date(dates, 0.7) == "2026-06-08"  # index round(10*0.7)=7
    with pytest.raises(ValueError):
        chronological_split_date(["2026-06-01"], 0.5)  # need >= 2 distinct dates


def test_sweep_selects_on_train_not_test() -> None:
    grid = ParameterGrid(thresholds=(0.0, 1.0), horizons=(1,))
    results = sweep(
        _signals(),
        _prices(),
        grid,
        split_date="2026-06-15",
        notional_usd=10_000.0,
        metric="mean_return",
    )

    assert len(results) == 2
    by_threshold = {result.threshold: result for result in results}

    # threshold 0.0 trades both train events -> train mean (0.10+0.02)/2 = 0.06.
    low = by_threshold[0.0]
    assert low.n_train == 2
    assert low.train_metric == pytest.approx(0.06)
    # threshold 1.0 gates out the low-score event -> only the +2% train trade.
    high = by_threshold[1.0]
    assert high.n_train == 1
    assert high.train_metric == pytest.approx(0.02)

    # Selection maximises TRAIN metric -> threshold 0.0, even though threshold 1.0
    # has the better held-out test metric (0.10 vs 0.06).
    selected = [result for result in results if result.selected]
    assert len(selected) == 1
    assert selected[0].threshold == 0.0
    assert high.test_metric > low.test_metric


def test_sweep_selects_nothing_when_no_train_trades() -> None:
    # Threshold above every score -> no train trades -> no selection.
    grid = ParameterGrid(thresholds=(9.0,), horizons=(1,))
    results = sweep(_signals(), _prices(), grid, split_date="2026-06-15", notional_usd=10_000.0, metric="mean_return")
    assert all(not result.selected for result in results)
    assert results[0].n_train == 0


def test_csv_loaders_round_trip_with_none_handling(tmp_path: Path) -> None:
    signals_csv = tmp_path / "daily_signals.csv"
    with signals_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol", "news_date", "scorer_id", "article_count", "valid_count",
                "mean_score", "signal", "signal_value", "availability_timestamp",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "symbol": "AAPL", "news_date": "2026-06-08", "scorer_id": "m", "article_count": "3",
            "valid_count": "3", "mean_score": "1.5", "signal": "positive", "signal_value": "1",
            "availability_timestamp": "",
        })
        writer.writerow({  # a no-signal row: empty optionals become None
            "symbol": "AAPL", "news_date": "2026-06-09", "scorer_id": "m", "article_count": "0",
            "valid_count": "0", "mean_score": "", "signal": "", "signal_value": "",
            "availability_timestamp": "",
        })
    signals = load_signals_csv(signals_csv)
    assert signals[0].mean_score == 1.5
    assert signals[0].signal_value == 1
    assert signals[1].mean_score is None
    assert signals[1].signal is None
    assert signals[1].signal_value is None

    prices_csv = tmp_path / "prices.csv"
    with prices_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["symbol", "session_date", "open", "high", "low", "close", "volume", "repaired"],
        )
        writer.writeheader()
        writer.writerow({
            "symbol": "AAPL", "session_date": "2026-06-08", "open": "100.0", "high": "105.0",
            "low": "95.0", "close": "101.0", "volume": "1000.0", "repaired": "True",
        })
    prices = load_prices_csv(prices_csv)
    assert prices[0].close == 101.0
    assert prices[0].repaired is True
