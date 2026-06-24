import numpy as np
import pandas as pd
import pytest

from sentiment_benchmark.trading_effectiveness import (
    EffectivenessResult,
    TradingEffectivenessError,
    benchmark_comparison,
    benjamini_hochberg,
    evaluate_effectiveness,
    format_effectiveness_markdown,
    risk_economics,
    significance_tests,
)


def _event(
    scorer_id: str,
    horizon: int,
    strategy_return_pct: float,
    *,
    signal_value: int = 1,
    market_return: float | None = None,
    symbol: str = "AAPL",
    entry_date: str = "2026-06-11",
) -> dict[str, object]:
    return {
        "scorer_id": scorer_id,
        "horizon": horizon,
        "strategy_return_pct": strategy_return_pct,
        "signal_value": signal_value,
        "market_return": strategy_return_pct / 100.0 if market_return is None else market_return,
        "pnl_usd": strategy_return_pct * 100.0,
        "entry_date": entry_date,
        "symbol": symbol,
    }


def test_benjamini_hochberg_matches_hand_computation() -> None:
    assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.04, 0.04, 0.04])
    assert benjamini_hochberg([0.001, 0.5]) == pytest.approx([0.002, 0.5])


def test_benjamini_hochberg_ignores_non_finite() -> None:
    result = benjamini_hochberg([0.001, float("nan")])
    assert result[0] == pytest.approx(0.001)
    assert np.isnan(result[1])


def test_significance_flags_clear_positive_edge_and_clears_noise() -> None:
    edge = [
        _event("edge", 1, value, symbol=f"S{i}", entry_date=f"2026-06-{11 + i:02d}")
        for i, value in enumerate([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8])
    ]
    noise = [
        _event("noise", 1, value, symbol=f"N{i}", entry_date=f"2026-06-{11 + i:02d}")
        for i, value in enumerate([-2, -1, 0, 1, 2, -2, -1, 0, 1, 2])
    ]
    frame = significance_tests(pd.DataFrame(edge + noise))

    edge_row = frame[frame["scorer_id"] == "edge"].iloc[0]
    noise_row = frame[frame["scorer_id"] == "noise"].iloc[0]
    assert edge_row["mean_return_pct"] == pytest.approx(1.9)
    assert edge_row["t_test_p"] < 0.001
    assert bool(edge_row["significant_bh_5pct"]) is True
    assert edge_row["trade_hit_rate"] == pytest.approx(1.0)
    assert noise_row["t_test_p"] > 0.05
    assert bool(noise_row["significant_bh_5pct"]) is False


def test_benchmark_excess_is_strategy_minus_buy_and_hold() -> None:
    rows = [
        _event("m", 1, 3.0, signal_value=1, market_return=0.03, symbol="A", entry_date="2026-06-11"),
        _event("m", 1, 2.0, signal_value=-1, market_return=-0.02, symbol="B", entry_date="2026-06-12"),
    ]
    frame = benchmark_comparison(pd.DataFrame(rows)).iloc[0]
    assert frame["strategy_mean_pct"] == pytest.approx(2.5)
    assert frame["buy_and_hold_mean_pct"] == pytest.approx(0.5)
    assert frame["excess_vs_buy_and_hold_pp"] == pytest.approx(2.0)


def test_economics_profit_factor_winrate_and_drawdown() -> None:
    rows = [
        _event("m", 1, 2.0, signal_value=1, symbol="A", entry_date="2026-06-11"),
        _event("m", 1, -1.0, signal_value=1, symbol="B", entry_date="2026-06-12"),
        _event("m", 1, 3.0, signal_value=1, symbol="C", entry_date="2026-06-13"),
        _event("m", 1, -1.0, signal_value=1, symbol="D", entry_date="2026-06-14"),
        _event("m", 1, 0.0, signal_value=0, symbol="E", entry_date="2026-06-15"),
    ]
    frame = risk_economics(pd.DataFrame(rows)).iloc[0]
    assert frame["n_events"] == 5
    assert frame["n_trades"] == 4
    assert frame["win_rate"] == pytest.approx(0.5)
    assert frame["avg_win_pct"] == pytest.approx(2.5)
    assert frame["avg_loss_pct"] == pytest.approx(-1.0)
    assert frame["profit_factor"] == pytest.approx(2.5)
    assert frame["mean_return_pct"] == pytest.approx(0.6)
    assert frame["max_drawdown_pct"] == pytest.approx(-1.0)


def test_economics_profit_factor_infinite_without_losses() -> None:
    rows = [_event("m", 1, 1.0, symbol="A"), _event("m", 1, 2.0, symbol="B", entry_date="2026-06-12")]
    frame = risk_economics(pd.DataFrame(rows)).iloc[0]
    assert np.isinf(frame["profit_factor"])


def test_evaluate_and_markdown_render() -> None:
    rows = [
        _event("consensus/majority", horizon, value, symbol=f"S{i}", entry_date=f"2026-06-{11 + i:02d}")
        for horizon in (1, 3)
        for i, value in enumerate([1.0, -0.5, 2.0, -1.0, 0.5])
    ]
    result = evaluate_effectiveness(pd.DataFrame(rows), scorer_display={"consensus/majority": "LLM consensus"})
    assert isinstance(result, EffectivenessResult)
    assert {"t_test_q_bh", "significant_bh_5pct"} <= set(result.significance.columns)
    assert "excess_vs_buy_and_hold_pp" in result.benchmarks.columns
    assert "return_risk_ratio" in result.economics.columns

    markdown = format_effectiveness_markdown(result)
    assert "## Trading effectiveness" in markdown
    assert "LLM consensus" in markdown
    assert "Headline." in markdown


def test_missing_columns_raise() -> None:
    with pytest.raises(TradingEffectivenessError, match="missing columns"):
        significance_tests(pd.DataFrame({"scorer_id": ["m"], "horizon": [1]}))
