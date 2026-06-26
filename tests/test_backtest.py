"""Tests for the pure backtest core: equity-curve aggregation and run_backtest."""

from __future__ import annotations

import pytest

from sentiment_benchmark.backtest import (
    BacktestResult,
    DailySignal,
    DecisionPolicyConfig,
    ReturnRow,
    build_equity_curve,
    run_backtest,
)
from sentiment_benchmark.prices import PriceRow

# Deterministic price ladder for AAPL: each session closes 10 above the prior,
# opens flat at 100 on the entry session, so returns are round numbers.
_SESSIONS = [
    ("2026-06-08", 100.0, 100.0),
    ("2026-06-09", 100.0, 110.0),  # entry session (open 100)
    ("2026-06-10", 110.0, 120.0),
    ("2026-06-11", 120.0, 130.0),
    ("2026-06-12", 130.0, 140.0),
    ("2026-06-15", 140.0, 150.0),
]


def _prices(symbol: str = "AAPL") -> list[PriceRow]:
    return [
        PriceRow(symbol, day, open_, open_ + 5, open_ - 5, close, 1_000.0, False)
        for day, open_, close in _SESSIONS
    ]


def _return_row(*, exit_date: str, net_pnl: float, horizon: int = 1, notional: float = 10_000.0) -> ReturnRow:
    """Minimal ReturnRow for equity-curve tests; only the fields the curve reads matter."""
    return ReturnRow(
        symbol="AAPL",
        traded_symbol="AAPL",
        index_fallback=False,
        news_date="2026-06-08",
        scorer_id="model",
        mean_score=1.0,
        signal="positive",
        signal_value=1,
        d_adjusted_close=None,
        entry_date="2026-06-09",
        entry_adjusted_open=100.0,
        horizon=horizon,
        exit_date=exit_date,
        exit_adjusted_close=110.0,
        market_return=net_pnl / notional,
        strategy_return=net_pnl / notional,
        strategy_return_pct=net_pnl / notional * 100,
        pnl_usd=net_pnl,
        notional_usd=notional,
        action="buy",
        transaction_cost=0.0,
        net_strategy_return=net_pnl / notional,
        net_strategy_return_pct=net_pnl / notional * 100,
        net_pnl_usd=net_pnl,
    )


def test_build_equity_curve_accumulates_by_exit_date() -> None:
    rows = [
        _return_row(exit_date="2026-06-09", net_pnl=100.0),
        _return_row(exit_date="2026-06-10", net_pnl=-40.0),
        _return_row(exit_date="2026-06-11", net_pnl=25.0),
    ]
    curve = build_equity_curve(rows, horizon=1)
    assert [point.date for point in curve] == ["2026-06-09", "2026-06-10", "2026-06-11"]
    assert [point.cumulative_pnl_usd for point in curve] == pytest.approx([100.0, 60.0, 85.0])
    # cumulative_return normalises pooled P&L by the per-trade notional.
    assert curve[-1].cumulative_return == pytest.approx(85.0 / 10_000.0)


def test_build_equity_curve_groups_same_exit_date() -> None:
    rows = [
        _return_row(exit_date="2026-06-09", net_pnl=100.0),
        _return_row(exit_date="2026-06-09", net_pnl=50.0),
    ]
    curve = build_equity_curve(rows, horizon=1)
    assert len(curve) == 1
    assert curve[0].trade_count == 2
    assert curve[0].realised_pnl_usd == pytest.approx(150.0)


def test_build_equity_curve_filters_to_requested_horizon() -> None:
    rows = [
        _return_row(exit_date="2026-06-09", net_pnl=100.0, horizon=1),
        _return_row(exit_date="2026-06-11", net_pnl=999.0, horizon=3),
    ]
    curve = build_equity_curve(rows, horizon=1)
    assert len(curve) == 1
    assert curve[0].cumulative_pnl_usd == pytest.approx(100.0)


def test_build_equity_curve_empty_when_no_match() -> None:
    rows = [_return_row(exit_date="2026-06-09", net_pnl=100.0, horizon=1)]
    assert build_equity_curve(rows, horizon=7) == []


def test_run_backtest_end_to_end_buy_signal() -> None:
    signal = DailySignal("AAPL", "2026-06-08", "model", 3, 3, 1.0, "positive", 1)
    policy = DecisionPolicyConfig(min_valid_stories=1, threshold=0.0)

    result = run_backtest(
        [signal],
        _prices(),
        policy,
        horizons=(1, 3),
        notional_usd=10_000.0,
    )

    assert isinstance(result, BacktestResult)
    assert [d.action for d in result.decisions] == ["buy"]
    by_horizon = {row.horizon: row for row in result.returns}
    # entry open 100 -> H1 exit close 110 (+10%), H3 exit close 130 (+30%).
    assert by_horizon[1].strategy_return == pytest.approx(0.10)
    assert by_horizon[3].strategy_return == pytest.approx(0.30)
    # equity_horizon defaults to max(horizons) == 3.
    assert result.equity_curve[-1].cumulative_pnl_usd == pytest.approx(0.30 * 10_000.0)


def test_run_backtest_hold_signal_makes_no_trades() -> None:
    # Mean below threshold -> hold -> no decisions traded -> empty returns/curve.
    signal = DailySignal("AAPL", "2026-06-08", "model", 3, 3, 0.2, "positive", 1)
    policy = DecisionPolicyConfig(min_valid_stories=1, threshold=0.5)

    result = run_backtest([signal], _prices(), policy, horizons=(1,), notional_usd=10_000.0)

    assert [d.action for d in result.decisions] == ["hold"]
    assert result.returns == []
    assert result.equity_curve == []
