"""Pure backtest core for the news-sentiment trading strategy.

This module owns the strategy *math* — decision policy, per-event return
calculation, and equity-curve aggregation — with no network or file I/O. It is
deliberately decoupled from ``trading_strategy`` (news ingestion, scoring,
orchestration) so the core can be unit-tested in isolation, swept over parameter
grids, and visualised. ``trading_strategy`` imports and re-exports these names,
so existing call sites and tests are unaffected.

Dependency direction: ``prices`` (leaf) ← ``backtest`` ← ``trading_strategy``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time
from itertools import groupby
from zoneinfo import ZoneInfo

from .prices import PriceRow


class TradingStrategyError(RuntimeError):
    """Raised when the trading pilot cannot be configured or completed."""


@dataclass(frozen=True)
class IndexFallback:
    """Trade a broad index on company-days too thin to support an individual stock.

    When a company-day has fewer than ``min_texts`` accepted articles, its (noisy)
    sentiment signal is executed against ``symbol`` instead of the company's own
    stock, following the supervisor's "use a US index if you can't get enough
    texts" guidance. Disabled unless a config supplies ``enabled = true``.
    """

    symbol: str
    min_texts: int


@dataclass(frozen=True)
class DecisionPolicyConfig:
    min_valid_stories: int = 1
    threshold: float = 0.0
    transaction_cost_bps_per_side: float = 0.0
    short_borrow_bps_per_day: float = 0.0
    policy_version: str = "sentiment_threshold_v1"


@dataclass(frozen=True)
class DailySignal:
    symbol: str
    news_date: str
    scorer_id: str
    article_count: int
    valid_count: int
    mean_score: float | None
    signal: str | None
    signal_value: int | None
    availability_timestamp: str | None = None


@dataclass(frozen=True)
class TradingDecision:
    symbol: str
    news_date: str
    scorer_id: str
    article_count: int
    valid_count: int
    mean_score: float | None
    action: str
    action_value: int
    threshold: float
    min_valid_stories: int
    policy_version: str
    reason: str
    availability_timestamp: str | None = None


@dataclass(frozen=True)
class ReturnRow:
    symbol: str
    traded_symbol: str
    index_fallback: bool
    news_date: str
    scorer_id: str
    mean_score: float
    signal: str
    signal_value: int
    d_adjusted_close: float | None
    entry_date: str
    entry_adjusted_open: float
    horizon: int
    exit_date: str
    exit_adjusted_close: float
    market_return: float
    strategy_return: float
    strategy_return_pct: float
    pnl_usd: float
    notional_usd: float
    action: str
    transaction_cost: float
    net_strategy_return: float
    net_strategy_return_pct: float
    net_pnl_usd: float
    availability_timestamp: str | None = None


@dataclass(frozen=True)
class EquityPoint:
    """One realisation step of a fixed-horizon, fixed-notional equity curve."""

    date: str
    trade_count: int
    realised_pnl_usd: float
    cumulative_pnl_usd: float
    cumulative_return: float


@dataclass(frozen=True)
class BacktestResult:
    decisions: list[TradingDecision]
    returns: list[ReturnRow]
    equity_curve: list[EquityPoint]


def make_trading_decisions(
    signals: list[DailySignal],
    policy: DecisionPolicyConfig,
) -> list[TradingDecision]:
    decisions: list[TradingDecision] = []
    for signal in signals:
        if signal.mean_score is None or signal.valid_count == 0:
            action, action_value, reason = "hold", 0, "no_valid_sentiment"
        elif signal.valid_count < policy.min_valid_stories:
            action, action_value, reason = "hold", 0, "insufficient_valid_stories"
        elif signal.mean_score > 0 and signal.mean_score >= policy.threshold:
            action, action_value, reason = "buy", 1, "positive_mean_meets_threshold"
        elif signal.mean_score < 0 and signal.mean_score <= -policy.threshold:
            action, action_value, reason = "sell", -1, "negative_mean_meets_threshold"
        else:
            action, action_value, reason = "hold", 0, "inside_no_trade_band"
        decisions.append(
            TradingDecision(
                symbol=signal.symbol,
                news_date=signal.news_date,
                scorer_id=signal.scorer_id,
                article_count=signal.article_count,
                valid_count=signal.valid_count,
                mean_score=signal.mean_score,
                action=action,
                action_value=action_value,
                threshold=policy.threshold,
                min_valid_stories=policy.min_valid_stories,
                policy_version=policy.policy_version,
                reason=reason,
                availability_timestamp=signal.availability_timestamp,
            )
        )
    return decisions


def calculate_returns(
    signals: list[DailySignal] | list[TradingDecision],
    prices: list[PriceRow],
    *,
    horizons: tuple[int, ...],
    notional_usd: float,
    index_fallback: IndexFallback | None = None,
    transaction_cost_bps_per_side: float = 0.0,
    short_borrow_bps_per_day: float = 0.0,
    timezone: str = "America/New_York",
) -> list[ReturnRow]:
    prices_by_symbol: dict[str, list[PriceRow]] = defaultdict(list)
    for row in prices:
        prices_by_symbol[row.symbol].append(row)
    returns: list[ReturnRow] = []
    for signal in signals:
        if isinstance(signal, TradingDecision):
            if signal.action == "hold" or signal.mean_score is None:
                continue
            signal_name = "positive" if signal.action == "buy" else "negative"
            signal_value = signal.action_value
            action = signal.action
        else:
            if signal.signal is None or signal.signal_value is None or signal.mean_score is None:
                continue
            signal_name = signal.signal
            signal_value = signal.signal_value
            action = "buy" if signal_value > 0 else "sell" if signal_value < 0 else "hold"
        use_fallback = index_fallback is not None and signal.article_count < index_fallback.min_texts
        traded_symbol = index_fallback.symbol if index_fallback is not None and use_fallback else signal.symbol
        symbol_prices = prices_by_symbol.get(traded_symbol, [])
        d_price = next((row for row in symbol_prices if row.session_date == signal.news_date), None)
        if signal.availability_timestamp:
            try:
                available_at = datetime.fromisoformat(signal.availability_timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise TradingStrategyError(
                    f"invalid availability timestamp for {signal.symbol}: {signal.availability_timestamp}"
                ) from exc
            if available_at.tzinfo is None:
                raise TradingStrategyError(
                    f"availability timestamp must include an offset: {signal.availability_timestamp}"
                )
            exchange_zone = ZoneInfo(timezone)
            future = [
                row
                for row in symbol_prices
                if datetime.combine(date.fromisoformat(row.session_date), time(9, 30), exchange_zone) > available_at
            ]
        else:
            future = [row for row in symbol_prices if row.session_date > signal.news_date]
        if len(future) < max(horizons):
            raise TradingStrategyError(
                f"incomplete price horizon for {traded_symbol} on {signal.news_date}: "
                f"need {max(horizons)} future sessions, found {len(future)}"
            )
        entry = future[0]
        if entry.open <= 0:
            raise TradingStrategyError(f"invalid entry price for {traded_symbol} on {entry.session_date}")
        for horizon in horizons:
            exit_row = future[horizon - 1]
            market_return = exit_row.close / entry.open - 1
            strategy_return = signal_value * market_return
            transaction_cost = transaction_cost_bps_per_side * 2 / 10_000
            short_borrow_cost = short_borrow_bps_per_day * horizon / 10_000 if signal_value < 0 else 0.0
            net_strategy_return = strategy_return - transaction_cost - short_borrow_cost
            returns.append(
                ReturnRow(
                    symbol=signal.symbol,
                    traded_symbol=traded_symbol,
                    index_fallback=use_fallback,
                    news_date=signal.news_date,
                    scorer_id=signal.scorer_id,
                    mean_score=signal.mean_score,
                    signal=signal_name,
                    signal_value=signal_value,
                    d_adjusted_close=d_price.close if d_price else None,
                    entry_date=entry.session_date,
                    entry_adjusted_open=entry.open,
                    horizon=horizon,
                    exit_date=exit_row.session_date,
                    exit_adjusted_close=exit_row.close,
                    market_return=market_return,
                    strategy_return=strategy_return,
                    strategy_return_pct=strategy_return * 100,
                    pnl_usd=strategy_return * notional_usd,
                    notional_usd=notional_usd,
                    action=action,
                    transaction_cost=transaction_cost + short_borrow_cost,
                    net_strategy_return=net_strategy_return,
                    net_strategy_return_pct=net_strategy_return * 100,
                    net_pnl_usd=net_strategy_return * notional_usd,
                    availability_timestamp=signal.availability_timestamp,
                )
            )
    return returns


def build_equity_curve(
    returns: list[ReturnRow],
    *,
    horizon: int,
    use_net: bool = True,
) -> list[EquityPoint]:
    """Cumulative fixed-notional P&L for a single horizon, ordered by realisation
    (exit) date. Each event risks one ``notional_usd`` unit; ``cumulative_return``
    normalises pooled P&L by that per-trade notional. Returns one point per exit
    date (P&L summed across events realised that day)."""
    rows = sorted(
        (row for row in returns if row.horizon == horizon),
        key=lambda row: (row.exit_date, row.news_date, row.symbol, row.scorer_id),
    )
    if not rows:
        return []
    base_notional = rows[0].notional_usd or 1.0
    points: list[EquityPoint] = []
    cumulative = 0.0
    for exit_date, group in groupby(rows, key=lambda row: row.exit_date):
        group_rows = list(group)
        step = sum((row.net_pnl_usd if use_net else row.pnl_usd) for row in group_rows)
        cumulative += step
        points.append(
            EquityPoint(
                date=exit_date,
                trade_count=len(group_rows),
                realised_pnl_usd=step,
                cumulative_pnl_usd=cumulative,
                cumulative_return=cumulative / base_notional,
            )
        )
    return points


def run_backtest(
    signals: list[DailySignal],
    prices: list[PriceRow],
    policy: DecisionPolicyConfig,
    *,
    horizons: tuple[int, ...],
    notional_usd: float,
    index_fallback: IndexFallback | None = None,
    timezone: str = "America/New_York",
    use_decision_policy: bool = True,
    equity_horizon: int | None = None,
) -> BacktestResult:
    """Pure end-to-end backtest: signals → decisions → per-event returns →
    equity curve. ``use_decision_policy=False`` reproduces the legacy signal rule
    (returns computed straight off the raw signals)."""
    decisions = make_trading_decisions(signals, policy)
    basis: list[DailySignal] | list[TradingDecision] = decisions if use_decision_policy else signals
    returns = calculate_returns(
        basis,
        prices,
        horizons=horizons,
        notional_usd=notional_usd,
        index_fallback=index_fallback,
        transaction_cost_bps_per_side=policy.transaction_cost_bps_per_side,
        short_borrow_bps_per_day=policy.short_borrow_bps_per_day,
        timezone=timezone,
    )
    eq_horizon = equity_horizon if equity_horizon is not None else max(horizons)
    equity_curve = build_equity_curve(returns, horizon=eq_horizon)
    return BacktestResult(decisions=decisions, returns=returns, equity_curve=equity_curve)
