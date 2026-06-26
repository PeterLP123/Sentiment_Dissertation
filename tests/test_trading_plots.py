"""Tests for sensitivity summaries and the sweep heatmap pivot (pure parts)."""

from __future__ import annotations

from pathlib import Path

from sentiment_benchmark.strategy_sweep import SweepResult
from sentiment_benchmark.trading_plots import (
    summarize_cutoff,
    summarize_masking,
    sweep_heatmap_matrix,
    write_sensitivity_csvs,
)


def _return(scorer_id: str, news_date: str, net_pct: float, horizon: int = 1):
    from sentiment_benchmark.backtest import ReturnRow

    return ReturnRow(
        symbol="AAPL",
        traded_symbol="AAPL",
        index_fallback=False,
        news_date=news_date,
        scorer_id=scorer_id,
        mean_score=1.0,
        signal="positive",
        signal_value=1,
        d_adjusted_close=None,
        entry_date=news_date,
        entry_adjusted_open=100.0,
        horizon=horizon,
        exit_date=news_date,
        exit_adjusted_close=100.0 + net_pct,
        market_return=net_pct / 100,
        strategy_return=net_pct / 100,
        strategy_return_pct=net_pct,
        pnl_usd=net_pct,
        notional_usd=10_000.0,
        action="buy",
        transaction_cost=0.0,
        net_strategy_return=net_pct / 100,
        net_strategy_return_pct=net_pct,
        net_pnl_usd=net_pct,
    )


def test_summarize_masking_splits_arms() -> None:
    returns = [
        _return("openai/m", "2026-06-08", 1.0),
        _return("openai/m", "2026-06-09", -1.0),
        _return("openai/m#masked", "2026-06-08", 0.5),
        _return("openai/m#masked", "2026-06-09", 0.5),
    ]
    rows = summarize_masking(returns)
    by_group = {row.group: row for row in rows}
    assert set(by_group) == {"masked", "unmasked"}
    assert by_group["unmasked"].scorer_id == "openai/m"  # base id, suffix stripped
    assert by_group["unmasked"].mean_net_return_pct == 0.0
    assert by_group["masked"].mean_net_return_pct == 0.5
    assert by_group["masked"].hit_rate == 1.0


def test_summarize_masking_empty_without_masked_arm() -> None:
    assert summarize_masking([_return("openai/m", "2026-06-08", 1.0)]) == []


def test_summarize_cutoff_splits_pre_post_and_baseline_na() -> None:
    models = ("openai/gpt-4o-mini",)  # cutoff 2023-10-01
    returns = [
        _return("openai/gpt-4o-mini", "2026-06-08", 2.0),  # post-cutoff
        _return("openai/gpt-4o-mini", "2023-01-01", -1.0),  # pre-cutoff
        _return("baseline/vader", "2026-06-08", 1.0),  # no cutoff -> n/a
    ]
    groups = {(row.scorer_id, row.group) for row in summarize_cutoff(returns, models)}
    assert ("openai/gpt-4o-mini", "post_cutoff") in groups
    assert ("openai/gpt-4o-mini", "pre_cutoff") in groups
    assert ("baseline/vader", "cutoff_na") in groups


def test_write_sensitivity_csvs_emits_expected_files(tmp_path: Path) -> None:
    returns = [
        _return("openai/gpt-4o-mini", "2026-06-08", 1.0),
        _return("openai/gpt-4o-mini#masked", "2026-06-08", 0.5),
    ]
    written = write_sensitivity_csvs(tmp_path, returns, ("openai/gpt-4o-mini",))
    names = {path.name for path in written}
    assert names == {"sensitivity_cutoff.csv", "sensitivity_masking.csv"}
    assert all(path.exists() for path in written)


def test_sweep_heatmap_matrix_pivots() -> None:
    results = [
        SweepResult(0.0, 1, 1, 0.0, 2, 2, 0.5, 0.4, 0.4, 0.5),
        SweepResult(0.0, 3, 1, 0.0, 2, 2, 0.6, 0.3, 0.3, 0.5),
        SweepResult(0.1, 1, 1, 0.0, 1, 1, 0.2, 0.1, 0.1, 1.0),
        SweepResult(0.1, 3, 1, 0.0, 1, 1, 0.3, 0.2, 0.2, 1.0),
    ]
    thresholds, horizons, matrix = sweep_heatmap_matrix(results, "test_metric")
    assert thresholds == [0.0, 0.1]
    assert horizons == [1, 3]
    assert matrix[0] == [0.4, 0.3]  # threshold 0.0 across horizons 1, 3
    assert matrix[1] == [0.1, 0.2]
