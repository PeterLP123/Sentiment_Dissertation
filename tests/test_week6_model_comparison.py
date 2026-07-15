from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sentiment_benchmark.week6_model_comparison import (
    Week6ModelComparisonConfig,
    development_quantile_thresholds,
    optimize_development_weights,
    profitable_negative_pairs,
    run_week6_model_comparison,
    select_negative_correlation_clique,
)


def _prices(symbols: tuple[str, ...]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", "2026-02-06")
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for index, session in enumerate(dates):
            close = 100.0 + symbol_index * 5 + index * (1 if symbol_index % 2 == 0 else -0.25)
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": session.strftime("%Y-%m-%d"),
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1_000,
                }
            )
    return pd.DataFrame(rows)


def _signals(symbols: tuple[str, ...], scorers: tuple[str, ...]) -> pd.DataFrame:
    news_dates = ("2026-01-02", "2026-01-06", "2026-01-09", "2026-01-16", "2026-01-23", "2026-01-28")
    rows = []
    for scorer_index, scorer in enumerate(scorers):
        for symbol_index, symbol in enumerate(symbols):
            for date_index, news_date in enumerate(news_dates):
                sign = 1 if (date_index + symbol_index + scorer_index) % 2 == 0 else -1
                rows.append(
                    {
                        "symbol": symbol,
                        "news_date": news_date,
                        "scorer_id": scorer,
                        "article_count": 2,
                        "valid_count": 2,
                        "mean_score": sign * (0.2 + 0.1 * scorer_index),
                        "signal": "positive" if sign > 0 else "negative",
                        "signal_value": sign,
                        "availability_timestamp": "",
                    }
                )
    return pd.DataFrame(rows)


def test_development_quantile_thresholds_ignore_evaluation_scores() -> None:
    frame = pd.DataFrame(
        {
            "news_date": ["2026-01-02", "2026-01-03", "2026-02-01"],
            "mean_score": [0.1, -0.5, 0.2],
        }
    )
    changed = frame.copy()
    changed.loc[changed["news_date"] >= "2026-02-01", "mean_score"] = 100.0
    first = development_quantile_thresholds(frame, split_date="2026-02-01", quantiles=(0.0, 0.5))
    second = development_quantile_thresholds(changed, split_date="2026-02-01", quantiles=(0.0, 0.5))
    pd.testing.assert_frame_equal(first, second)
    assert first.set_index("threshold_quantile").loc[0.0, "absolute_threshold"] == 0.0
    assert first.set_index("threshold_quantile").loc[0.5, "absolute_threshold"] == pytest.approx(0.3)


def test_negative_portfolio_requires_profit_support_and_pairwise_negativity() -> None:
    dates = pd.bdate_range("2026-01-02", periods=6).strftime("%Y-%m-%d")
    patterns = {
        "AAA": [0.02, -0.01, 0.02, -0.01, 0.02, -0.01],
        "BBB": [-0.01, 0.02, -0.01, 0.02, -0.01, 0.02],
        "CCC": [0.01, -0.005, 0.01, -0.005, 0.01, -0.005],
    }
    daily_rows = []
    for symbol, values in patterns.items():
        for session_date, value in zip(dates, values, strict=True):
            daily_rows.append(
                {
                    "date": session_date,
                    "symbol": symbol,
                    "net_return": value,
                    "net_pnl": value * 10_000,
                    "active_trade": True,
                    "split": "development",
                }
            )
    daily = pd.DataFrame(daily_rows)
    correlations = pd.DataFrame(
        [
            {
                "symbol_a": "AAA",
                "symbol_b": "BBB",
                "pairwise_correlation": -0.9,
                "overlapping_dates": 6,
                "jointly_active_dates": 6,
                "symbol_a_active_days": 6,
                "symbol_b_active_days": 6,
                "support_status": "supported",
            },
            {
                "symbol_a": "AAA",
                "symbol_b": "CCC",
                "pairwise_correlation": 0.8,
                "overlapping_dates": 6,
                "jointly_active_dates": 6,
                "symbol_a_active_days": 6,
                "symbol_b_active_days": 6,
                "support_status": "supported",
            },
            {
                "symbol_a": "BBB",
                "symbol_b": "CCC",
                "pairwise_correlation": -0.7,
                "overlapping_dates": 6,
                "jointly_active_dates": 6,
                "symbol_a_active_days": 6,
                "symbol_b_active_days": 6,
                "support_status": "supported",
            },
        ]
    )
    evidence = profitable_negative_pairs(correlations, daily, min_active_days=5)
    selected = select_negative_correlation_clique(evidence, requested_count=3)
    assert selected == ("AAA", "BBB")
    weights = optimize_development_weights(daily, selected, maximum_weight=0.6)
    assert weights["weight"].sum() == pytest.approx(1.0)
    assert weights["weight"].max() <= 0.6 + 1e-9


def test_end_to_end_model_comparison_writes_ranked_outputs(tmp_path: Path) -> None:
    symbols = ("AAA", "BBB", "CCC")
    scorers = ("model/gemma", "model/finbert", "model/vader")
    signals_path = tmp_path / "daily_signals.csv"
    prices_path = tmp_path / "prices.csv"
    _signals(symbols, scorers).to_csv(signals_path, index=False, lineterminator="\n")
    _prices(symbols).to_csv(prices_path, index=False, lineterminator="\n")
    config = Week6ModelComparisonConfig(
        run_id="comparison",
        scorer_ids=scorers,
        threshold_quantiles=(0.0, 0.5),
        holding_periods=(1, 3),
        min_joint_active_dates=1,
        min_stock_active_days=1,
    )
    result = run_week6_model_comparison(
        signals_path,
        prices_path,
        tmp_path / "results",
        config,
        command="sentiment-bench compare-week6-models synthetic",
        repo_root=tmp_path,
        generated_at="2026-07-15T00:00:00+00:00",
    )
    required = {
        "model_stock_daily_pnl.csv",
        "model_portfolio_daily_pnl.csv",
        "model_stock_performance.csv",
        "model_portfolio_performance.csv",
        "model_strategy_grid.csv",
        "profitable_negative_pairs.csv",
        "selected_negative_portfolio_weights.csv",
        "best_strategies.md",
        "best_strategies.html",
        "best_strategies_equity.png",
        "manifest.json",
    }
    assert required <= {path.name for path in result.output_dir.iterdir()}
    performance = pd.read_csv(result.output_dir / "model_portfolio_performance.csv")
    assert set(performance["scorer_id"]) == set(scorers)
    assert set(performance["period"]) == {"development", "evaluation"}
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection_contract"]["evaluation"] == "frozen rules and weights evaluated once"
    assert len(manifest["selected_rules"]) == 3
