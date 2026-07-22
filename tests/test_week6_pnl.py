"""Focused synthetic tests for the leak-controlled Week 6 daily P&L path."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from sentiment_benchmark.week6_pnl import (
    Week6PnlConfig,
    Week6PnlError,
    align_signals_to_sessions,
    annualized_sharpe,
    build_manual_timing_check,
    build_portfolio_daily,
    build_stock_daily_pnl,
    build_strategy_grid,
    compounded_return,
    development_correlations,
    load_and_validate_inputs,
    maximum_drawdown,
    run_week6_pnl,
    select_low_correlation_stocks,
    select_stable_strategy,
)


def _prices(symbols: tuple[str, ...] = ("AAA",), closes: tuple[float, ...] | None = None) -> pd.DataFrame:
    dates = (
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
        "2026-01-20",
        "2026-01-21",
        "2026-01-22",
        "2026-01-23",
    )
    values = closes or tuple(100.0 + index for index in range(len(dates)))
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for session_date, close in zip(dates, values, strict=True):
            adjusted = close * (1.0 + symbol_index * 0.001)
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": session_date,
                    "open": adjusted,
                    "high": adjusted * 1.01,
                    "low": adjusted * 0.99,
                    "close": adjusted,
                    "volume": 1000.0,
                    "repaired": False,
                }
            )
    return pd.DataFrame(rows)


def _signals(rows: list[tuple[str, str, float | None]], scorer: str = "headline/sentiment_all") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "news_date": news_date,
                "scorer_id": scorer,
                "article_count": 2,
                "valid_count": 0 if score is None else 2,
                "mean_score": score,
                "signal": "",
                "signal_value": "",
                "availability_timestamp": "",
            }
            for symbol, news_date, score in rows
        ]
    )


def test_no_lookahead_and_weekend_alignment() -> None:
    signals = _signals([("AAA", "2026-01-02", 1.0)])
    prices = _prices(closes=(90, 100, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122))
    aligned, _ = align_signals_to_sessions(signals, prices, threshold=0.0, exchange_timezone="America/New_York")
    assert aligned.iloc[0]["entry_date"] == "2026-01-05"

    daily, _ = build_stock_daily_pnl(
        signals,
        prices,
        threshold=0.0,
        holding_period=1,
        split_date="2026-01-16",
        starting_capital=100_000,
        transaction_cost_bps_per_side=0,
    )
    entry = daily.loc[daily["date"] == "2026-01-05"].iloc[0]
    assert entry["entry_price"] == 100
    assert entry["exit_price"] == 110
    assert entry["realised_forward_return"] == pytest.approx(0.10)
    assert entry["gross_return"] == pytest.approx(0.10)
    assert not daily.loc[daily["date"] == "2026-01-02", "active_trade"].iloc[0]


def test_timestamp_can_use_same_session_close_only_after_availability() -> None:
    signals = _signals([("AAA", "2026-01-05", 1.0)])
    signals.loc[0, "availability_timestamp"] = "2026-01-05T14:00:00Z"  # 09:00 New York
    aligned, _ = align_signals_to_sessions(signals, _prices(), threshold=0.0, exchange_timezone="America/New_York")
    assert aligned.iloc[0]["entry_date"] == "2026-01-05"
    signals.loc[0, "availability_timestamp"] = "2026-01-05T22:00:00Z"  # after close
    aligned, _ = align_signals_to_sessions(signals, _prices(), threshold=0.0, exchange_timezone="America/New_York")
    assert aligned.iloc[0]["entry_date"] == "2026-01-06"


def test_long_short_flat_signs_turnover_and_costs() -> None:
    signals = _signals(
        [
            ("AAA", "2026-01-02", 1.0),  # long from Jan 5
            ("AAA", "2026-01-05", -1.0),  # direct short from Jan 6
            ("AAA", "2026-01-06", 0.0),  # flat from Jan 7
        ]
    )
    prices = _prices(closes=(99, 100, 110, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110))
    daily, _ = build_stock_daily_pnl(
        signals,
        prices,
        threshold=0.0,
        holding_period=3,
        split_date="2026-01-16",
        starting_capital=100_000,
        transaction_cost_bps_per_side=10,
    )
    long = daily.loc[daily["date"] == "2026-01-05"].iloc[0]
    short = daily.loc[daily["date"] == "2026-01-06"].iloc[0]
    flat = daily.loc[daily["date"] == "2026-01-07"].iloc[0]
    assert long["gross_pnl"] == pytest.approx(10_000)
    assert short["gross_pnl"] == pytest.approx(10_000)  # short earns 10% on 110 -> 99
    assert flat["gross_pnl"] == 0
    assert short["turnover"] == 2
    assert short["transaction_cost"] == pytest.approx(200)
    assert flat["turnover"] == 1
    assert flat["transaction_cost"] == pytest.approx(100)


def test_compounding_sharpe_and_drawdown() -> None:
    assert compounded_return([0.10, -0.10]) == pytest.approx(-0.01)
    expected = (0.01 / math.sqrt(0.0002)) * math.sqrt(252)
    assert annualized_sharpe([0.0, 0.02]) == pytest.approx(expected)
    assert maximum_drawdown([0.10, -0.20, 0.05]) == pytest.approx(-0.20)


def test_split_integrity_and_boundary_attrition() -> None:
    signals = _signals([("AAA", "2026-01-13", 1.0), ("AAA", "2026-01-16", -1.0)])
    daily, counts = build_stock_daily_pnl(
        signals,
        _prices(),
        threshold=0.0,
        holding_period=5,
        split_date="2026-01-16",
        starting_capital=100_000,
        transaction_cost_bps_per_side=10,
    )
    assert counts.boundary_attrited_signals >= 1
    assert (daily.loc[daily["date"] < "2026-01-16", "split"] == "development").all()
    assert (daily.loc[daily["date"] >= "2026-01-16", "split"] == "evaluation").all()
    first_evaluation = daily.loc[daily["split"] == "evaluation"].iloc[0]
    assert first_evaluation["previous_position"] == 0


def test_stable_strategy_selection_ignores_evaluation_columns() -> None:
    grid = pd.DataFrame(
        [
            {
                "threshold": 0.0,
                "holding_period": 1,
                "median_subperiod_sharpe": 0.4,
                "worst_subperiod_sharpe": 0.1,
                "development_net_sharpe": 0.5,
                "development_turnover": 3.0,
                "evaluation_sharpe": -100.0,
            },
            {
                "threshold": 0.05,
                "holding_period": 3,
                "median_subperiod_sharpe": 0.3,
                "worst_subperiod_sharpe": 0.2,
                "development_net_sharpe": 10.0,
                "development_turnover": 1.0,
                "evaluation_sharpe": 100.0,
            },
        ]
    )
    threshold, holding, _ = select_stable_strategy(grid)
    assert (threshold, holding) == (0.0, 1)
    grid["evaluation_sharpe"] *= -1
    assert select_stable_strategy(grid)[:2] == (0.0, 1)


def test_evaluation_signals_cannot_change_development_grid() -> None:
    base = _signals(
        [
            ("AAA", "2026-01-02", 1.0),
            ("AAA", "2026-01-05", -1.0),
            ("AAA", "2026-01-16", 1.0),
        ]
    )
    changed = base.copy()
    changed.loc[changed["news_date"] >= "2026-01-16", "mean_score"] = -1.0
    config = Week6PnlConfig(
        run_id="synthetic",
        thresholds=(0.0, 0.05),
        holding_periods=(1, 3),
        min_joint_active_dates=1,
    )
    grid_a, _, _ = build_strategy_grid(base, _prices(), config=config, split_date="2026-01-16")
    grid_b, _, _ = build_strategy_grid(changed, _prices(), config=config, split_date="2026-01-16")
    development_columns = [column for column in grid_a if column.startswith("development_") or column.startswith("subperiod_")]
    pd.testing.assert_frame_equal(grid_a[development_columns], grid_b[development_columns])
    assert select_stable_strategy(grid_a)[:2] == select_stable_strategy(grid_b)[:2]


def test_duplicate_and_missing_price_validation(tmp_path: Path) -> None:
    signals = _signals([("AAA", "2026-01-02", 1.0)])
    signals_path = tmp_path / "signals.csv"
    prices_path = tmp_path / "prices.csv"
    pd.concat([signals, signals]).to_csv(signals_path, index=False)
    _prices().to_csv(prices_path, index=False)
    with pytest.raises(Week6PnlError, match="duplicate signal"):
        load_and_validate_inputs(signals_path, prices_path, "headline/sentiment_all")

    _signals([("MISSING", "2026-01-02", 1.0)]).to_csv(signals_path, index=False)
    with pytest.raises(Week6PnlError, match="no price panel"):
        load_and_validate_inputs(signals_path, prices_path, "headline/sentiment_all")


def test_correlation_support_counts_and_low_correlation_reproducibility() -> None:
    rows = []
    for index, session_date in enumerate(("2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07")):
        for symbol, value, active in (
            ("AAA", [0.1, -0.1, 0.1, -0.1][index], True),
            ("BBB", [-0.1, 0.1, -0.1, 0.1][index], index < 3),
            ("CCC", [0.1, 0.1, -0.1, -0.1][index], True),
        ):
            rows.append({"date": session_date, "symbol": symbol, "net_return": value, "active_trade": active, "split": "development"})
    daily = pd.DataFrame(rows)
    correlations = development_correlations(daily, min_joint_active_dates=3)
    pair = correlations.loc[(correlations["symbol_a"] == "AAA") & (correlations["symbol_b"] == "BBB")].iloc[0]
    assert pair["overlapping_dates"] == 4
    assert pair["jointly_active_dates"] == 3
    assert pair["pairwise_correlation"] == pytest.approx(-1.0)
    selected_a = select_low_correlation_stocks(correlations, daily, requested_count=2, min_active_days=3)
    selected_b = select_low_correlation_stocks(correlations, daily, requested_count=2, min_active_days=3)
    pd.testing.assert_frame_equal(selected_a, selected_b)


def test_identical_inputs_produce_identical_daily_output() -> None:
    signals = _signals([("AAA", "2026-01-02", 1.0), ("AAA", "2026-01-06", -1.0)])
    kwargs = {
        "threshold": 0.05,
        "holding_period": 3,
        "split_date": "2026-01-16",
        "starting_capital": 100_000,
        "transaction_cost_bps_per_side": 10,
    }
    first, first_counts = build_stock_daily_pnl(signals, _prices(), **kwargs)
    second, second_counts = build_stock_daily_pnl(signals, _prices(), **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert first_counts == second_counts


def test_portfolio_fixed_weight_and_manual_check() -> None:
    signals = _signals([(symbol, news_date, 1.0) for symbol in ("AAA", "BBB") for news_date in ("2026-01-02", "2026-01-16")])
    daily, _ = build_stock_daily_pnl(
        signals,
        _prices(("AAA", "BBB")),
        threshold=0,
        holding_period=3,
        split_date="2026-01-16",
        starting_capital=100_000,
        transaction_cost_bps_per_side=10,
    )
    portfolio = build_portfolio_daily(daily, {"all": ["AAA", "BBB"]}, starting_capital=100_000)
    assert (portfolio["weight_per_stock"] == 0.5).all()
    assert (portfolio["daily_gross_pnl"] - portfolio["transaction_cost"] - portfolio["daily_net_pnl"]).abs().max() < 1e-10
    assert (
        portfolio["fixed_notional_gross_pnl"] - portfolio["fixed_notional_transaction_cost"] - portfolio["fixed_notional_net_pnl"]
    ).abs().max() < 1e-10
    manual = build_manual_timing_check(daily, cost_rate=0.001)
    assert len(manual) == 5
    assert set(manual["pass_fail"]) == {"pass"}


def test_end_to_end_outputs_and_manifest(tmp_path: Path) -> None:
    symbols = ("AAA", "BBB", "CCC")
    signals = _signals(
        [
            (symbol, news_date, 1.0 if index % 2 == 0 else -1.0)
            for symbol in symbols
            for index, news_date in enumerate(("2026-01-02", "2026-01-05", "2026-01-07", "2026-01-12", "2026-01-16", "2026-01-20"))
        ]
    )
    signals_path = tmp_path / "daily_signals.csv"
    prices_path = tmp_path / "prices.csv"
    signals.to_csv(signals_path, index=False)
    _prices(symbols).to_csv(prices_path, index=False)
    config = Week6PnlConfig(
        run_id="integration",
        thresholds=(0.0,),
        holding_periods=(1,),
        min_joint_active_dates=1,
        low_correlation_stock_count=2,
    )
    result = run_week6_pnl(
        signals_path,
        prices_path,
        tmp_path / "results" / "week6_pnl",
        config,
        command="sentiment-bench analyze-week6-pnl synthetic",
        repo_root=tmp_path,
        generated_at="2026-07-13T00:00:00+00:00",
    )
    required = {
        "stock_daily_pnl.csv",
        "portfolio_daily_pnl.csv",
        "stock_performance.csv",
        "portfolio_performance.csv",
        "strategy_grid.csv",
        "pnl_correlations.csv",
        "selected_low_correlation_stocks.csv",
        "manual_timing_check.csv",
        "cumulative_pnl.png",
        "cumulative_pnl_evaluation.png",
        "drawdown.png",
        "pnl_correlation_heatmap.png",
        "summary.md",
        "manifest.json",
    }
    assert required <= {path.name for path in result.output_dir.iterdir()}
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["selected_rule"]["selection_data"] == "development_only"
    assert "manifest.json" not in manifest["output_hashes"]
    with pytest.raises(Week6PnlError, match="refusing to overwrite"):
        run_week6_pnl(signals_path, prices_path, tmp_path / "results" / "week6_pnl", config)


def test_end_to_end_can_skip_optional_low_correlation_portfolio(tmp_path: Path) -> None:
    symbols = ("AAA", "BBB", "CCC")
    signals = _signals(
        [(symbol, "2026-01-02", 1.0 if symbol != "CCC" else -1.0) for symbol in symbols]
        + [(symbol, "2026-01-16", -1.0 if symbol != "CCC" else 1.0) for symbol in symbols]
    )
    signals_path = tmp_path / "daily_signals.csv"
    prices_path = tmp_path / "prices.csv"
    signals.to_csv(signals_path, index=False)
    _prices(symbols).to_csv(prices_path, index=False)
    config = Week6PnlConfig(
        run_id="all-stock-only",
        thresholds=(0.0,),
        holding_periods=(1,),
        min_joint_active_dates=20,
        include_low_correlation_portfolio=False,
    )
    result = run_week6_pnl(
        signals_path,
        prices_path,
        tmp_path / "results" / "week6_pnl",
        config,
        command="sentiment-bench analyze-week6-pnl synthetic --skip-low-correlation-portfolio",
        repo_root=tmp_path,
        generated_at="2026-07-22T00:00:00+00:00",
    )
    performance = pd.read_csv(result.output_dir / "portfolio_performance.csv")
    selected = pd.read_csv(result.output_dir / "selected_low_correlation_stocks.csv")
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert set(performance["portfolio"]) == {"all_stock_equal_weight"}
    assert selected.empty
    assert manifest["configuration"]["include_low_correlation_portfolio"] is False
    assert manifest["selected_stock_set"] == []
