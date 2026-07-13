"""Focused tests for funded cross-sectional portfolio accounting."""

from __future__ import annotations

import math
from dataclasses import replace
from statistics import fmean, stdev

import pytest

from sentiment_benchmark.backtest import (
    DailySignal,
    DecisionPolicyConfig,
    IndexFallback,
    TradingDecision,
    TradingStrategyError,
    make_trading_decisions,
)
from sentiment_benchmark.portfolio import PortfolioConfig, run_portfolio_backtest
from sentiment_benchmark.prices import PriceRow


def _signal(symbol: str, news_date: str, value: int, *, score: float | None = None) -> DailySignal:
    mean_score = float(value) if score is None else score
    return DailySignal(
        symbol=symbol,
        news_date=news_date,
        scorer_id="model",
        article_count=2,
        valid_count=2,
        mean_score=mean_score,
        signal="positive" if value > 0 else "negative" if value < 0 else "neutral",
        signal_value=value,
    )


def _price(symbol: str, session_date: str, open_price: float, close_price: float) -> PriceRow:
    return PriceRow(
        symbol=symbol,
        session_date=session_date,
        open=open_price,
        high=max(open_price, close_price),
        low=min(open_price, close_price),
        close=close_price,
        volume=1_000.0,
        repaired=False,
    )


def test_funded_book_reconciles_stock_pnl_nav_costs_and_borrow() -> None:
    result = run_portfolio_backtest(
        [_signal("LONG", "2026-01-01", 1), _signal("SHORT", "2026-01-01", -1)],
        [
            _price("LONG", "2026-01-02", 100.0, 110.0),
            _price("SHORT", "2026-01-02", 100.0, 90.0),
            # The provider's forward buffer must not become a zero-return day.
            _price("LONG", "2026-01-05", 110.0, 110.0),
            _price("SHORT", "2026-01-05", 90.0, 90.0),
        ],
        DecisionPolicyConfig(transaction_cost_bps_per_side=10.0, short_borrow_bps_per_day=2.0),
        horizons=(1,),
        initial_nav_usd=10_000.0,
    )

    assert len(result.trades) == 2
    assert sorted(trade.signed_entry_notional_usd for trade in result.trades) == [-5_000.0, 5_000.0]
    day = result.daily_portfolio[0]
    assert day.gross_pnl_usd == pytest.approx(1_000.0)
    assert day.transaction_cost_usd == pytest.approx(20.0)  # actual entry + exit notionals
    assert day.borrow_cost_usd == pytest.approx(0.9)  # 2 bps on the short's $4,500 closing value
    assert day.net_pnl_usd == pytest.approx(979.1)
    assert day.end_nav_usd == pytest.approx(10_979.1)
    assert day.turnover_usd == pytest.approx(20_000.0)
    assert sum(row.net_pnl_usd for row in result.daily_stock_pnl) == pytest.approx(day.net_pnl_usd)

    summary = result.summaries[0]
    assert summary.total_profit_usd == pytest.approx(day.net_pnl_usd)
    assert summary.total_return_pct == pytest.approx(9.791)
    assert summary.total_transaction_cost_usd == pytest.approx(20.0)
    assert summary.total_borrow_cost_usd == pytest.approx(0.9)
    assert (summary.start_date, summary.end_date) == ("2026-01-02", "2026-01-02")
    assert summary.observations == 1


def test_multi_day_horizon_uses_staggered_capital_sleeves() -> None:
    dates = ("2026-01-02", "2026-01-05", "2026-01-06")
    prices = [_price(symbol, session_date, 100.0, 100.0) for symbol in ("LONG", "SHORT") for session_date in dates]
    signals = [
        _signal("LONG", "2026-01-01", 1),
        _signal("SHORT", "2026-01-01", -1),
        _signal("LONG", "2026-01-02", 1),
        _signal("SHORT", "2026-01-02", -1),
    ]
    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(),
        horizons=(2,),
        initial_nav_usd=10_000.0,
    )

    assert len(result.trades) == 4
    assert {abs(trade.signed_target_weight) for trade in result.trades} == {0.25}
    days = {row.date: row for row in result.daily_portfolio}
    assert days["2026-01-02"].turnover_usd == pytest.approx(5_000.0)
    assert days["2026-01-02"].gross_exposure_usd == pytest.approx(5_000.0)
    # On the overlap day, one 50%-gross sleeve exits and another enters; the
    # marked book is 100%-gross, never two fresh full-capital event blocks.
    assert days["2026-01-05"].gross_exposure_usd == pytest.approx(10_000.0)
    assert days["2026-01-05"].turnover_usd == pytest.approx(10_000.0)
    assert days["2026-01-06"].gross_exposure_usd == pytest.approx(5_000.0)


def test_h3_daily_marks_reconcile_to_terminal_lot_pnl_and_nav() -> None:
    dates = ("2026-01-02", "2026-01-05", "2026-01-06")
    prices = [
        _price("LONG", dates[0], 100.0, 102.0),
        _price("LONG", dates[1], 102.0, 101.0),
        _price("LONG", dates[2], 101.0, 106.0),
        _price("SHORT", dates[0], 100.0, 98.0),
        _price("SHORT", dates[1], 98.0, 99.0),
        _price("SHORT", dates[2], 99.0, 95.0),
    ]
    result = run_portfolio_backtest(
        [_signal("LONG", "2026-01-01", 1), _signal("SHORT", "2026-01-01", -1)],
        prices,
        DecisionPolicyConfig(short_borrow_bps_per_day=2.0),
        horizons=(3,),
        initial_nav_usd=10_000.0,
        # H3 divides gross exposure by three; 3x makes this isolated cohort 1x.
        portfolio_config=PortfolioConfig(gross_exposure=3.0),
    )

    assert sum(trade.gross_pnl_usd for trade in result.trades) == pytest.approx(550.0)
    assert sum(row.gross_pnl_usd for row in result.daily_portfolio) == pytest.approx(550.0)
    assert sum(row.borrow_cost_usd for row in result.daily_portfolio) == pytest.approx(2.92)
    assert result.summaries[0].final_nav_usd == pytest.approx(10_547.08)
    stock_by_date = {
        session_date: sum(row.net_pnl_usd for row in result.daily_stock_pnl if row.date == session_date)
        for session_date in dates
    }
    assert stock_by_date == pytest.approx(
        {row.date: row.net_pnl_usd for row in result.daily_portfolio}
    )


@pytest.mark.parametrize(
    ("available_at", "expected_entry"),
    [
        ("2026-06-10T13:29:59+00:00", "2026-06-10"),
        ("2026-06-10T13:30:00+00:00", "2026-06-11"),
        ("2026-06-10T15:00:00+00:00", "2026-06-11"),
    ],
)
def test_availability_timestamp_uses_strict_exchange_open_boundary(
    available_at: str,
    expected_entry: str,
) -> None:
    signals = [
        replace(_signal("A", "2026-06-10", 1), availability_timestamp=available_at),
        replace(_signal("B", "2026-06-10", -1), availability_timestamp=available_at),
    ]
    prices = [
        _price(symbol, session_date, 100.0, 101.0 if symbol == "A" else 99.0)
        for symbol in ("A", "B")
        for session_date in ("2026-06-10", "2026-06-11")
    ]

    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(),
        horizons=(1,),
        initial_nav_usd=10_000.0,
    )

    assert {trade.entry_date for trade in result.trades} == {expected_entry}


def test_prices_are_sorted_and_duplicate_or_incomplete_histories_fail() -> None:
    signals = [_signal("A", "2026-01-01", 1), _signal("B", "2026-01-01", -1)]
    prices = [
        _price("A", "2026-01-05", 100.0, 101.0),
        _price("B", "2026-01-05", 100.0, 99.0),
        _price("A", "2026-01-02", 100.0, 101.0),
        _price("B", "2026-01-02", 100.0, 99.0),
    ]
    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(),
        horizons=(1,),
        initial_nav_usd=10_000.0,
    )
    assert {trade.entry_date for trade in result.trades} == {"2026-01-02"}

    with pytest.raises(TradingStrategyError, match="duplicate price row"):
        run_portfolio_backtest(
            signals,
            [*prices, prices[0]],
            DecisionPolicyConfig(),
            horizons=(1,),
            initial_nav_usd=10_000.0,
        )
    with pytest.raises(TradingStrategyError, match="incomplete price horizon"):
        run_portfolio_backtest(
            signals,
            prices[:2],
            DecisionPolicyConfig(),
            horizons=(2,),
            initial_nav_usd=10_000.0,
        )


def test_equal_weighting_respects_cap_and_leaves_unused_capacity_in_cash() -> None:
    signals = [
        _signal("A", "2026-01-01", 1),
        _signal("B", "2026-01-01", 1),
        _signal("C", "2026-01-01", -1),
    ]
    prices = [_price(symbol, "2026-01-02", 100.0, 100.0) for symbol in ("A", "B", "C")]
    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(),
        horizons=(1,),
        initial_nav_usd=10_000.0,
        portfolio_config=PortfolioConfig(weighting="equal", max_abs_weight=0.4),
    )

    weights = {trade.traded_symbol: trade.signed_target_weight for trade in result.trades}
    assert weights == pytest.approx({"A": 0.25, "B": 0.25, "C": -0.4})
    assert result.daily_portfolio[0].gross_exposure_usd == pytest.approx(9_000.0)


def test_index_fallback_collisions_aggregate_to_one_auditable_lot() -> None:
    signals = [_signal("A", "2026-01-01", 1), _signal("B", "2026-01-01", 1)]
    result = run_portfolio_backtest(
        signals,
        [_price("SPY", "2026-01-02", 100.0, 101.0)],
        DecisionPolicyConfig(),
        horizons=(1,),
        initial_nav_usd=10_000.0,
        index_fallback=IndexFallback(symbol="SPY", min_texts=3),
        portfolio_config=PortfolioConfig(
            require_two_sided=False,
            duplicate_entry_policy="aggregate",
        ),
    )

    assert len(result.trades) == 1
    assert result.trades[0].traded_symbol == "SPY"
    assert result.trades[0].source_symbols == "A|B"


def test_concentration_cap_applies_across_overlapping_sleeves() -> None:
    dates = ("2026-01-02", "2026-01-05", "2026-01-06")
    prices = [_price(symbol, session_date, 100.0, 100.0) for symbol in ("LONG", "SHORT") for session_date in dates]
    signals = [
        _signal("LONG", "2026-01-01", 1),
        _signal("SHORT", "2026-01-01", -1),
        _signal("LONG", "2026-01-02", 1),
        _signal("SHORT", "2026-01-02", -1),
    ]
    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(),
        horizons=(2,),
        initial_nav_usd=10_000.0,
        portfolio_config=PortfolioConfig(max_abs_weight=0.25),
    )

    # The first pair consumes the +/-25% name limits.  The overlapping second
    # cohort therefore remains in cash instead of treating the cap per sleeve.
    assert len(result.trades) == 2
    overlap = next(row for row in result.daily_portfolio if row.date == "2026-01-05")
    assert overlap.gross_exposure_usd == pytest.approx(5_000.0)
    assert overlap.entries == 0


def test_daily_metrics_are_annualized_from_funded_returns() -> None:
    signals = [
        _signal("LONG", "2026-01-01", 1),
        _signal("SHORT", "2026-01-01", -1),
        _signal("LONG", "2026-01-02", 1),
        _signal("SHORT", "2026-01-02", -1),
    ]
    prices = [
        _price("LONG", "2026-01-02", 100.0, 110.0),
        _price("SHORT", "2026-01-02", 100.0, 90.0),
        _price("LONG", "2026-01-05", 100.0, 95.0),
        _price("SHORT", "2026-01-05", 100.0, 105.0),
    ]
    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(),
        horizons=(1,),
        initial_nav_usd=10_000.0,
    )
    returns = [0.10, -0.05]
    expected_sharpe = math.sqrt(252) * fmean(returns) / stdev(returns)
    summary = result.summaries[0]

    assert [row.daily_return for row in result.daily_portfolio] == pytest.approx(returns)
    assert summary.final_nav_usd == pytest.approx(10_450.0)
    assert summary.total_return == pytest.approx(0.045)
    assert summary.annualized_sharpe == pytest.approx(expected_sharpe)
    assert summary.annualized_return == pytest.approx(1.045**126 - 1)
    assert summary.max_drawdown == pytest.approx(0.05)


def test_stock_outputs_find_profitable_negatively_correlated_pair() -> None:
    signals = [
        _signal("A", "2026-01-01", 1),
        _signal("B", "2026-01-01", -1),
        _signal("A", "2026-01-02", 1),
        _signal("B", "2026-01-02", -1),
    ]
    prices = [
        _price("A", "2026-01-02", 100.0, 102.0),
        _price("B", "2026-01-02", 100.0, 101.0),
        _price("A", "2026-01-05", 100.0, 99.0),
        _price("B", "2026-01-05", 100.0, 98.0),
    ]
    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(),
        horizons=(1,),
        initial_nav_usd=10_000.0,
    )

    assert {row.symbol for row in result.stock_summaries} == {"A", "B"}
    assert all(row.net_pnl_usd > 0 for row in result.stock_summaries)
    assert result.correlations[0].correlation == pytest.approx(-1.0)
    assert result.correlations[0].simultaneous_active_days == 2
    assert result.correlations[0].both_profitable is True
    assert all(row.sample_covariance is not None for row in result.covariances)
    assert all(row.shrunk_covariance is not None for row in result.covariances)
    # The exported stock panel is dense, including zero-contribution dates, so
    # downstream covariance calculations can reproduce the built-in estimates.
    assert len(result.daily_stock_pnl) == 4  # 2 stocks x 2 sessions


def test_duplicate_entries_require_an_explicit_resolution_policy() -> None:
    signals = [
        _signal("A", "2026-01-01", 1),
        _signal("A", "2026-01-02", 1),  # both map to the next available session
        _signal("B", "2026-01-02", -1),
    ]
    prices = [_price("A", "2026-01-05", 100.0, 101.0), _price("B", "2026-01-05", 100.0, 99.0)]
    with pytest.raises(TradingStrategyError, match="multiple signals map to A entry 2026-01-05"):
        run_portfolio_backtest(
            signals,
            prices,
            DecisionPolicyConfig(),
            horizons=(1,),
            initial_nav_usd=10_000.0,
        )

    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(),
        horizons=(1,),
        initial_nav_usd=10_000.0,
        portfolio_config=PortfolioConfig(duplicate_entry_policy="aggregate"),
    )
    assert len(result.trades) == 2
    aggregate = next(trade for trade in result.trades if trade.traded_symbol == "A")
    assert aggregate.news_dates == "2026-01-01|2026-01-02"


def test_raw_signal_mode_bypasses_policy_gates_but_preserves_policy_decisions() -> None:
    signals = [
        _signal("A", "2026-01-01", 1, score=0.1),
        _signal("B", "2026-01-01", -1, score=-0.1),
    ]
    prices = [_price("A", "2026-01-02", 100.0, 101.0), _price("B", "2026-01-02", 100.0, 99.0)]
    result = run_portfolio_backtest(
        signals,
        prices,
        DecisionPolicyConfig(threshold=0.5),
        horizons=(1,),
        initial_nav_usd=10_000.0,
        use_decision_policy=False,
    )

    assert all(decision.action == "hold" for decision in result.decisions)
    assert len(result.trades) == 2
    assert result.summaries[0].total_profit_usd == pytest.approx(100.0)


def test_inconsistent_action_and_position_sign_is_rejected() -> None:
    signals = [_signal("A", "2026-01-01", 1)]

    def inconsistent(
        values: list[DailySignal],
        policy: DecisionPolicyConfig,
    ) -> list[TradingDecision]:
        return [replace(make_trading_decisions(values, policy)[0], position=-1.0)]

    with pytest.raises(TradingStrategyError, match="buy decision has non-positive position"):
        run_portfolio_backtest(
            signals,
            [_price("A", "2026-01-02", 100.0, 101.0)],
            DecisionPolicyConfig(),
            horizons=(1,),
            initial_nav_usd=10_000.0,
            decision_fn=inconsistent,
        )


def test_require_two_sided_applies_to_non_neutral_construction() -> None:
    result = run_portfolio_backtest(
        [_signal("A", "2026-01-01", 1)],
        [_price("A", "2026-01-02", 100.0, 110.0)],
        DecisionPolicyConfig(),
        horizons=(1,),
        initial_nav_usd=10_000.0,
        portfolio_config=PortfolioConfig(dollar_neutral=False, require_two_sided=True),
    )

    assert result.trades == []
    assert result.daily_portfolio[0].end_nav_usd == pytest.approx(10_000.0)
    assert result.summaries[0].trade_count == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gross_exposure": 0.0}, "gross_exposure"),
        ({"weighting": "rank"}, "weighting"),
        ({"max_positions_per_side": 0}, "max_positions_per_side"),
        ({"duplicate_entry_policy": "silent"}, "duplicate_entry_policy"),
    ],
)
def test_portfolio_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PortfolioConfig(**kwargs)  # type: ignore[arg-type]
