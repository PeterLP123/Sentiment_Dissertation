"""Funded cross-sectional portfolio accounting for sentiment strategies.

The event-study backtest deliberately treats every event as a fresh fixed-
notional trade.  This module answers a different question: what would have
happened to one finite pool of capital if the daily signals had been traded as
a portfolio?

Each ``(scorer, horizon)`` is evaluated as an independent book.  A horizon of
``h`` sessions divides the requested gross exposure into ``h`` staggered daily
sleeves, preventing overlapping cohorts from silently multiplying the capital
base.  Lots are marked to each session's close; entry and exit costs are charged
on the actual notional traded and short borrow accrues once per holding session.

The module is pure: it performs no file or network I/O.  Its row-oriented result
objects are intentionally suitable for direct CSV serialisation by the trading
runner.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from statistics import fmean, stdev
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.covariance import LedoitWolf

from .backtest import (
    DailySignal,
    DecisionPolicyConfig,
    IndexFallback,
    TradingDecision,
    TradingStrategyError,
    make_trading_decisions,
)
from .prices import PriceRow

WeightingMethod = Literal["equal", "signal"]
DuplicateEntryPolicy = Literal["error", "first", "aggregate"]
DecisionFn = Callable[[list[DailySignal], DecisionPolicyConfig], list[TradingDecision]]


@dataclass(frozen=True)
class PortfolioConfig:
    """Controls portfolio construction and annualisation.

    ``gross_exposure`` is a multiple of start-of-day NAV.  It is divided by the
    holding horizon before each cohort is opened.  With the default value of
    one, a five-session strategy therefore opens a 20%-gross sleeve each day.

    ``dollar_neutral`` splits a sleeve equally between its long and short sides.
    If ``require_two_sided`` is true, an unpaired one-sided cohort remains in
    cash.  ``signal`` weighting uses the absolute decision position as
    conviction; ``equal`` assigns every selected name the same raw weight.
    ``max_abs_weight`` caps the marked net position in a stock across all
    overlapping sleeves; allocation that cannot be placed under the cap stays
    in cash.
    """

    gross_exposure: float = 1.0
    dollar_neutral: bool = True
    require_two_sided: bool = True
    weighting: WeightingMethod = "signal"
    max_abs_weight: float = 1.0
    max_positions_per_side: int | None = None
    duplicate_entry_policy: DuplicateEntryPolicy = "error"
    periods_per_year: int = 252
    annual_risk_free_rate: float = 0.0
    estimate_shrunk_covariance: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.gross_exposure) or self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be a finite positive number")
        if self.weighting not in ("equal", "signal"):
            raise ValueError("weighting must be 'equal' or 'signal'")
        if not math.isfinite(self.max_abs_weight) or self.max_abs_weight <= 0:
            raise ValueError("max_abs_weight must be a finite positive number")
        if self.max_positions_per_side is not None and self.max_positions_per_side < 1:
            raise ValueError("max_positions_per_side must be positive when supplied")
        if self.duplicate_entry_policy not in ("error", "first", "aggregate"):
            raise ValueError("duplicate_entry_policy must be 'error', 'first', or 'aggregate'")
        if self.periods_per_year < 1:
            raise ValueError("periods_per_year must be positive")
        if not math.isfinite(self.annual_risk_free_rate) or self.annual_risk_free_rate <= -1:
            raise ValueError("annual_risk_free_rate must be finite and greater than -1")


@dataclass(frozen=True)
class PortfolioTradeRow:
    """One funded portfolio lot, retained as an end-to-end audit record."""

    scorer_id: str
    horizon: int
    source_symbols: str
    traded_symbol: str
    news_dates: str
    entry_date: str
    exit_date: str
    signed_target_weight: float
    entry_price: float
    exit_price: float
    shares: float
    signed_entry_notional_usd: float
    gross_pnl_usd: float
    transaction_cost_usd: float
    borrow_cost_usd: float
    net_pnl_usd: float


@dataclass(frozen=True)
class DailyStockPnlRow:
    """Daily mark-to-market contribution for one traded stock."""

    scorer_id: str
    horizon: int
    date: str
    symbol: str
    gross_pnl_usd: float
    transaction_cost_usd: float
    borrow_cost_usd: float
    net_pnl_usd: float
    return_contribution: float
    cumulative_net_pnl_usd: float
    gross_exposure_usd: float
    net_exposure_usd: float
    entry_notional_usd: float
    exit_notional_usd: float
    active_lots: int


@dataclass(frozen=True)
class DailyPortfolioRow:
    """One funded book's daily P&L, NAV, exposure, and drawdown."""

    scorer_id: str
    horizon: int
    date: str
    start_nav_usd: float
    gross_pnl_usd: float
    transaction_cost_usd: float
    borrow_cost_usd: float
    net_pnl_usd: float
    daily_return: float
    end_nav_usd: float
    cumulative_profit_usd: float
    cumulative_return: float
    drawdown: float
    gross_exposure_usd: float
    long_exposure_usd: float
    short_exposure_usd: float
    net_exposure_usd: float
    turnover_usd: float
    turnover: float
    entries: int
    exits: int
    active_lots: int


@dataclass(frozen=True)
class PortfolioSummary:
    """Annualised and whole-period metrics for one scorer/horizon book."""

    scorer_id: str
    horizon: int
    initial_nav_usd: float
    final_nav_usd: float
    observations: int
    trade_count: int
    start_date: str | None
    end_date: str | None
    total_profit_usd: float
    total_return: float
    total_return_pct: float
    annualized_return: float | None
    annualized_return_pct: float | None
    annualized_volatility: float | None
    annualized_volatility_pct: float | None
    annualized_sharpe: float | None
    annualized_sortino: float | None
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float | None
    profitable_day_rate: float
    profit_factor: float | None
    total_transaction_cost_usd: float
    total_borrow_cost_usd: float
    average_daily_turnover: float
    annualized_turnover: float


@dataclass(frozen=True)
class StockSummary:
    """Contribution metrics for one stock within a scorer/horizon book."""

    scorer_id: str
    horizon: int
    symbol: str
    observations: int
    active_days: int
    trade_count: int
    gross_pnl_usd: float
    transaction_cost_usd: float
    borrow_cost_usd: float
    net_pnl_usd: float
    annualized_contribution_sharpe: float | None
    correlation_to_portfolio: float | None
    max_peak_to_trough_loss_usd: float


@dataclass(frozen=True)
class CorrelationRow:
    """Sample correlation between two aligned daily stock contributions."""

    scorer_id: str
    horizon: int
    symbol_a: str
    symbol_b: str
    observations: int
    simultaneous_active_days: int
    both_profitable: bool
    correlation: float | None


@dataclass(frozen=True)
class CovarianceRow:
    """Daily return-contribution covariance in long form.

    ``shrunk_covariance`` is the Ledoit-Wolf estimate when enabled and enough
    observations exist; it is otherwise ``None``.
    """

    scorer_id: str
    horizon: int
    symbol_a: str
    symbol_b: str
    observations: int
    sample_covariance: float | None
    shrunk_covariance: float | None


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Complete, serialisable output of a funded portfolio evaluation."""

    decisions: list[TradingDecision]
    trades: list[PortfolioTradeRow]
    daily_stock_pnl: list[DailyStockPnlRow]
    daily_portfolio: list[DailyPortfolioRow]
    summaries: list[PortfolioSummary]
    stock_summaries: list[StockSummary]
    correlations: list[CorrelationRow]
    covariances: list[CovarianceRow]


@dataclass(frozen=True)
class _Candidate:
    scorer_id: str
    traded_symbol: str
    source_symbols: tuple[str, ...]
    news_dates: tuple[str, ...]
    position: float
    path: tuple[PriceRow, ...]

    @property
    def entry_date(self) -> str:
        return self.path[0].session_date


@dataclass
class _Lot:
    candidate: _Candidate
    horizon: int
    signed_target_weight: float
    entry_price: float
    shares: float
    signed_entry_notional_usd: float
    gross_pnl_usd: float = 0.0
    transaction_cost_usd: float = 0.0
    borrow_cost_usd: float = 0.0


@dataclass
class _StockDay:
    gross_pnl: float = 0.0
    transaction_cost: float = 0.0
    borrow_cost: float = 0.0
    gross_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    entry_notional: float = 0.0
    exit_notional: float = 0.0
    active_lots: int = 0

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.transaction_cost - self.borrow_cost


def _position(decision: TradingDecision) -> float:
    return decision.position if decision.position is not None else float(decision.action_value)


def _raw_signal_decisions(signals: Sequence[DailySignal], policy: DecisionPolicyConfig) -> list[TradingDecision]:
    """Convert raw directions to decisions while intentionally ignoring gates."""

    decisions: list[TradingDecision] = []
    for signal in signals:
        value = int(signal.signal_value or 0)
        if signal.mean_score is None or value == 0:
            action, reason = "hold", "raw_signal_not_tradeable"
        else:
            action = "buy" if value > 0 else "sell"
            reason = "raw_signal_direction"
        decisions.append(
            TradingDecision(
                symbol=signal.symbol,
                news_date=signal.news_date,
                scorer_id=signal.scorer_id,
                article_count=signal.article_count,
                valid_count=signal.valid_count,
                mean_score=signal.mean_score,
                action=action,
                action_value=value,
                threshold=policy.threshold,
                min_valid_stories=policy.min_valid_stories,
                policy_version=policy.policy_version,
                reason=reason,
                availability_timestamp=signal.availability_timestamp,
                position=float(value),
            )
        )
    return decisions


def _price_index(prices: Sequence[PriceRow]) -> dict[str, tuple[PriceRow, ...]]:
    grouped: dict[str, list[PriceRow]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in prices:
        key = (row.symbol, row.session_date)
        if key in seen:
            raise TradingStrategyError(f"duplicate price row for {row.symbol} on {row.session_date}")
        seen.add(key)
        grouped[row.symbol].append(row)
    return {symbol: tuple(sorted(rows, key=lambda item: item.session_date)) for symbol, rows in grouped.items()}


def _future_path(
    decision: TradingDecision,
    rows: tuple[PriceRow, ...],
    horizon: int,
    timezone: str,
) -> tuple[PriceRow, ...]:
    if decision.availability_timestamp:
        try:
            available_at = datetime.fromisoformat(decision.availability_timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TradingStrategyError(
                f"invalid availability timestamp for {decision.symbol}: {decision.availability_timestamp}"
            ) from exc
        if available_at.tzinfo is None:
            raise TradingStrategyError(
                f"availability timestamp must include an offset: {decision.availability_timestamp}"
            )
        exchange_zone = ZoneInfo(timezone)
        future = tuple(
            row
            for row in rows
            if datetime.combine(date.fromisoformat(row.session_date), time(9, 30), exchange_zone) > available_at
        )
    else:
        future = tuple(row for row in rows if row.session_date > decision.news_date)
    if len(future) < horizon:
        raise TradingStrategyError(
            f"incomplete price horizon for {decision.symbol} on {decision.news_date}: "
            f"need {horizon} future sessions, found {len(future)}"
        )
    path = future[:horizon]
    for row in path:
        if not all(math.isfinite(value) and value > 0 for value in (row.open, row.close)):
            raise TradingStrategyError(f"invalid price for {row.symbol} on {row.session_date}")
    return path


def _candidates(
    decisions: Sequence[TradingDecision],
    prices_by_symbol: dict[str, tuple[PriceRow, ...]],
    *,
    horizon: int,
    index_fallback: IndexFallback | None,
    timezone: str,
    duplicate_policy: DuplicateEntryPolicy,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for decision in decisions:
        position = _position(decision)
        if not math.isfinite(position):
            raise TradingStrategyError(f"non-finite position for {decision.symbol} on {decision.news_date}")
        if decision.action == "buy" and position <= 0:
            raise TradingStrategyError(
                f"buy decision has non-positive position for {decision.symbol} on {decision.news_date}: {position}"
            )
        if decision.action == "sell" and position >= 0:
            raise TradingStrategyError(
                f"sell decision has non-negative position for {decision.symbol} on {decision.news_date}: {position}"
            )
        if decision.action not in ("buy", "sell", "hold"):
            raise TradingStrategyError(f"unknown action {decision.action!r} for {decision.symbol} on {decision.news_date}")
        if decision.action == "hold" or decision.mean_score is None:
            continue
        use_fallback = index_fallback is not None and decision.article_count < index_fallback.min_texts
        traded_symbol = index_fallback.symbol if use_fallback and index_fallback is not None else decision.symbol
        rows = prices_by_symbol.get(traded_symbol, ())
        path = _future_path(decision, rows, horizon, timezone)
        candidates.append(
            _Candidate(
                scorer_id=decision.scorer_id,
                traded_symbol=traded_symbol,
                source_symbols=(decision.symbol,),
                news_dates=(decision.news_date,),
                position=position,
                path=path,
            )
        )

    grouped: dict[tuple[str, str, str], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.scorer_id, candidate.entry_date, candidate.traded_symbol)].append(candidate)

    resolved: list[_Candidate] = []
    for key in sorted(grouped):
        matches = grouped[key]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        scorer_id, entry_date, traded_symbol = key
        if duplicate_policy == "error":
            raise TradingStrategyError(
                f"multiple signals map to {traded_symbol} entry {entry_date} for scorer {scorer_id}; "
                "set duplicate_entry_policy='first' or 'aggregate' explicitly"
            )
        if duplicate_policy == "first":
            resolved.append(sorted(matches, key=lambda item: (item.news_dates, item.source_symbols))[0])
            continue
        exemplar = matches[0]
        combined_position = sum(item.position for item in matches)
        if combined_position == 0:
            continue
        resolved.append(
            replace(
                exemplar,
                source_symbols=tuple(sorted({symbol for item in matches for symbol in item.source_symbols})),
                news_dates=tuple(sorted({value for item in matches for value in item.news_dates})),
                position=combined_position,
            )
        )
    return sorted(resolved, key=lambda item: (item.scorer_id, item.entry_date, item.traded_symbol))


def _raw_strength(candidate: _Candidate, config: PortfolioConfig) -> float:
    return 1.0 if config.weighting == "equal" else abs(candidate.position)


def _select_side(candidates: list[_Candidate], config: PortfolioConfig) -> list[_Candidate]:
    ordered = sorted(candidates, key=lambda item: (-_raw_strength(item, config), item.traded_symbol))
    return ordered[: config.max_positions_per_side] if config.max_positions_per_side is not None else ordered


def _capped_allocations(
    candidates: list[_Candidate],
    target_weight: float,
    config: PortfolioConfig,
) -> dict[str, float]:
    """Water-fill a side's absolute weights, leaving cash if caps bind."""

    if not candidates or target_weight <= 0:
        return {}
    remaining = target_weight
    active = list(candidates)
    allocations = {candidate.traded_symbol: 0.0 for candidate in candidates}
    while active and remaining > 1e-15:
        total_strength = sum(_raw_strength(candidate, config) for candidate in active)
        if total_strength <= 0:
            break
        capped: list[_Candidate] = []
        for candidate in active:
            proposed = remaining * _raw_strength(candidate, config) / total_strength
            capacity = config.max_abs_weight - allocations[candidate.traded_symbol]
            if proposed >= capacity - 1e-15:
                allocations[candidate.traded_symbol] += max(0.0, capacity)
                remaining -= max(0.0, capacity)
                capped.append(candidate)
        if not capped:
            for candidate in active:
                allocations[candidate.traded_symbol] += remaining * _raw_strength(candidate, config) / total_strength
            remaining = 0.0
        else:
            active = [candidate for candidate in active if candidate not in capped]
    return allocations


def _cohort_weights(candidates: list[_Candidate], horizon: int, config: PortfolioConfig) -> dict[str, float]:
    longs = _select_side([item for item in candidates if item.position > 0], config)
    shorts = _select_side([item for item in candidates if item.position < 0], config)
    sleeve_gross = config.gross_exposure / horizon
    if config.require_two_sided and (not longs or not shorts):
        return {}
    if config.dollar_neutral:
        if longs and shorts:
            long_target = short_target = sleeve_gross / 2
        elif longs:
            long_target, short_target = sleeve_gross, 0.0
        else:
            long_target, short_target = 0.0, sleeve_gross
    else:
        selected = longs + shorts
        total = sum(_raw_strength(item, config) for item in selected)
        long_strength = sum(_raw_strength(item, config) for item in longs)
        long_target = sleeve_gross * long_strength / total if total else 0.0
        short_target = sleeve_gross - long_target if total else 0.0
    weights = _capped_allocations(longs, long_target, config)
    weights.update({symbol: -weight for symbol, weight in _capped_allocations(shorts, short_target, config).items()})
    return weights


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _marked_open_value(lot: _Lot, session_date: str) -> float:
    """Value an existing lot at today's open, or carry its latest close."""

    previous_close: float | None = None
    for row in lot.candidate.path:
        if row.session_date == session_date:
            return lot.shares * row.open
        if row.session_date > session_date:
            break
        previous_close = row.close
    return lot.shares * previous_close if previous_close is not None else 0.0


def _cap_entry_weight(existing_weight: float, proposed_weight: float, max_abs_weight: float) -> float:
    """Clip a new order without reversing its intended direction."""

    if proposed_weight > 0:
        return min(proposed_weight, max(0.0, max_abs_weight - existing_weight))
    return max(proposed_weight, min(0.0, -max_abs_weight - existing_weight))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    left_sd, right_sd = stdev(left), stdev(right)
    if left_sd == 0 or right_sd == 0:
        return None
    value = float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])
    return value if math.isfinite(value) else None


def _annualized_sharpe(
    returns: Sequence[float],
    periods_per_year: int,
    annual_risk_free_rate: float = 0.0,
) -> float | None:
    if len(returns) < 2:
        return None
    daily_rf = (1 + annual_risk_free_rate) ** (1 / periods_per_year) - 1
    excess = [value - daily_rf for value in returns]
    sigma = stdev(excess)
    return math.sqrt(periods_per_year) * fmean(excess) / sigma if sigma else None


def _annualized_sortino(returns: Sequence[float], periods_per_year: int, annual_risk_free_rate: float) -> float | None:
    if len(returns) < 2:
        return None
    daily_rf = (1 + annual_risk_free_rate) ** (1 / periods_per_year) - 1
    excess = np.asarray(returns, dtype=float) - daily_rf
    downside_deviation = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    return math.sqrt(periods_per_year) * float(np.mean(excess)) / downside_deviation if downside_deviation else None


def _simulate_book(
    scorer_id: str,
    horizon: int,
    candidates: list[_Candidate],
    prices: Sequence[PriceRow],
    *,
    initial_nav: float,
    policy: DecisionPolicyConfig,
    config: PortfolioConfig,
) -> tuple[list[PortfolioTradeRow], list[DailyStockPnlRow], list[DailyPortfolioRow]]:
    entries: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        entries[candidate.entry_date].append(candidate)
    if not entries:
        return [], [], []

    first_date = min(entries)
    last_date = max(candidate.path[-1].session_date for candidate in candidates)
    calendar = sorted({row.session_date for row in prices if first_date <= row.session_date <= last_date})
    open_lots: list[_Lot] = []
    finished_lots: list[_Lot] = []
    stock_days: dict[tuple[str, str], _StockDay] = defaultdict(_StockDay)
    portfolio_days: list[DailyPortfolioRow] = []
    nav = initial_nav
    peak_nav = initial_nav

    for session_date in calendar:
        start_nav = nav
        day_entries = entries.get(session_date, [])
        weights = _cohort_weights(day_entries, horizon, config)
        opened = 0
        if start_nav > 0:
            existing_open_values: dict[str, float] = defaultdict(float)
            for lot in open_lots:
                existing_open_values[lot.candidate.traded_symbol] += _marked_open_value(lot, session_date)
            for candidate in day_entries:
                signed_weight = weights.get(candidate.traded_symbol)
                if signed_weight is None or signed_weight == 0:
                    continue
                signed_weight = _cap_entry_weight(
                    existing_open_values[candidate.traded_symbol] / start_nav,
                    signed_weight,
                    config.max_abs_weight,
                )
                if signed_weight == 0:
                    continue
                entry_price = candidate.path[0].open
                signed_notional = start_nav * signed_weight
                open_lots.append(
                    _Lot(
                        candidate=candidate,
                        horizon=horizon,
                        signed_target_weight=signed_weight,
                        entry_price=entry_price,
                        shares=signed_notional / entry_price,
                        signed_entry_notional_usd=signed_notional,
                    )
                )
                opened += 1
                existing_open_values[candidate.traded_symbol] += signed_notional

        exiting: list[_Lot] = []
        for lot in open_lots:
            path_index = next(
                (index for index, row in enumerate(lot.candidate.path) if row.session_date == session_date),
                None,
            )
            if path_index is None:
                continue
            row = lot.candidate.path[path_index]
            previous_price = lot.entry_price if path_index == 0 else lot.candidate.path[path_index - 1].close
            gross_pnl = lot.shares * (row.close - previous_price)
            close_value = lot.shares * row.close
            entry_notional = abs(lot.signed_entry_notional_usd) if path_index == 0 else 0.0
            exit_notional = abs(close_value) if path_index == horizon - 1 else 0.0
            transaction_cost = policy.transaction_cost_bps_per_side / 10_000 * (entry_notional + exit_notional)
            borrow_cost = (
                policy.short_borrow_bps_per_day / 10_000 * abs(close_value)
                if lot.shares < 0
                else 0.0
            )
            day = stock_days[(session_date, lot.candidate.traded_symbol)]
            day.gross_pnl += gross_pnl
            day.transaction_cost += transaction_cost
            day.borrow_cost += borrow_cost
            day.gross_exposure += abs(close_value)
            day.long_exposure += max(0.0, close_value)
            day.short_exposure += max(0.0, -close_value)
            day.net_exposure += close_value
            day.entry_notional += entry_notional
            day.exit_notional += exit_notional
            day.active_lots += 1
            lot.gross_pnl_usd += gross_pnl
            lot.transaction_cost_usd += transaction_cost
            lot.borrow_cost_usd += borrow_cost
            if path_index == horizon - 1:
                exiting.append(lot)

        session_stock_days = [value for (day_date, _), value in stock_days.items() if day_date == session_date]
        gross_pnl = sum(item.gross_pnl for item in session_stock_days)
        transaction_cost = sum(item.transaction_cost for item in session_stock_days)
        borrow_cost = sum(item.borrow_cost for item in session_stock_days)
        net_pnl = gross_pnl - transaction_cost - borrow_cost
        nav = start_nav + net_pnl
        peak_nav = max(peak_nav, nav)
        gross_exposure = sum(item.gross_exposure for item in session_stock_days)
        net_exposure = sum(item.net_exposure for item in session_stock_days)
        long_exposure = sum(item.long_exposure for item in session_stock_days)
        short_exposure = sum(item.short_exposure for item in session_stock_days)
        turnover_usd = sum(item.entry_notional + item.exit_notional for item in session_stock_days)
        portfolio_days.append(
            DailyPortfolioRow(
                scorer_id=scorer_id,
                horizon=horizon,
                date=session_date,
                start_nav_usd=start_nav,
                gross_pnl_usd=gross_pnl,
                transaction_cost_usd=transaction_cost,
                borrow_cost_usd=borrow_cost,
                net_pnl_usd=net_pnl,
                daily_return=_safe_ratio(net_pnl, start_nav),
                end_nav_usd=nav,
                cumulative_profit_usd=nav - initial_nav,
                cumulative_return=_safe_ratio(nav - initial_nav, initial_nav),
                drawdown=_safe_ratio(nav, peak_nav) - 1,
                gross_exposure_usd=gross_exposure,
                long_exposure_usd=long_exposure,
                short_exposure_usd=short_exposure,
                net_exposure_usd=net_exposure,
                turnover_usd=turnover_usd,
                turnover=_safe_ratio(turnover_usd, start_nav),
                entries=opened,
                exits=len(exiting),
                active_lots=sum(item.active_lots for item in session_stock_days),
            )
        )
        for lot in exiting:
            open_lots.remove(lot)
            finished_lots.append(lot)

    trades = [
        PortfolioTradeRow(
            scorer_id=scorer_id,
            horizon=horizon,
            source_symbols="|".join(lot.candidate.source_symbols),
            traded_symbol=lot.candidate.traded_symbol,
            news_dates="|".join(lot.candidate.news_dates),
            entry_date=lot.candidate.entry_date,
            exit_date=lot.candidate.path[-1].session_date,
            signed_target_weight=lot.signed_target_weight,
            entry_price=lot.entry_price,
            exit_price=lot.candidate.path[-1].close,
            shares=lot.shares,
            signed_entry_notional_usd=lot.signed_entry_notional_usd,
            gross_pnl_usd=lot.gross_pnl_usd,
            transaction_cost_usd=lot.transaction_cost_usd,
            borrow_cost_usd=lot.borrow_cost_usd,
            net_pnl_usd=lot.gross_pnl_usd - lot.transaction_cost_usd - lot.borrow_cost_usd,
        )
        for lot in finished_lots
    ]

    stock_rows: list[DailyStockPnlRow] = []
    cumulative_by_symbol: dict[str, float] = defaultdict(float)
    start_nav_by_date = {row.date: row.start_nav_usd for row in portfolio_days}
    traded_symbols = sorted({lot.candidate.traded_symbol for lot in finished_lots})
    for session_date in calendar:
        for symbol in traded_symbols:
            day = stock_days.get((session_date, symbol), _StockDay())
            cumulative_by_symbol[symbol] += day.net_pnl
            stock_rows.append(
                DailyStockPnlRow(
                    scorer_id=scorer_id,
                    horizon=horizon,
                    date=session_date,
                    symbol=symbol,
                    gross_pnl_usd=day.gross_pnl,
                    transaction_cost_usd=day.transaction_cost,
                    borrow_cost_usd=day.borrow_cost,
                    net_pnl_usd=day.net_pnl,
                    return_contribution=_safe_ratio(day.net_pnl, start_nav_by_date[session_date]),
                    cumulative_net_pnl_usd=cumulative_by_symbol[symbol],
                    gross_exposure_usd=day.gross_exposure,
                    net_exposure_usd=day.net_exposure,
                    entry_notional_usd=day.entry_notional,
                    exit_notional_usd=day.exit_notional,
                    active_lots=day.active_lots,
                )
            )
    return trades, stock_rows, portfolio_days


def _summarize_book(
    scorer_id: str,
    horizon: int,
    initial_nav: float,
    trades: Sequence[PortfolioTradeRow],
    days: Sequence[DailyPortfolioRow],
    config: PortfolioConfig,
) -> PortfolioSummary:
    returns = [row.daily_return for row in days]
    observations = len(days)
    final_nav = days[-1].end_nav_usd if days else initial_nav
    total_return = final_nav / initial_nav - 1
    annualized_return = (
        (final_nav / initial_nav) ** (config.periods_per_year / observations) - 1
        if observations and final_nav > 0
        else None
    )
    annualized_volatility = stdev(returns) * math.sqrt(config.periods_per_year) if observations >= 2 else None
    max_drawdown = abs(min((row.drawdown for row in days), default=0.0))
    positive = sum(row.net_pnl_usd for row in days if row.net_pnl_usd > 0)
    negative = abs(sum(row.net_pnl_usd for row in days if row.net_pnl_usd < 0))
    return PortfolioSummary(
        scorer_id=scorer_id,
        horizon=horizon,
        initial_nav_usd=initial_nav,
        final_nav_usd=final_nav,
        observations=observations,
        trade_count=len(trades),
        start_date=days[0].date if days else None,
        end_date=days[-1].date if days else None,
        total_profit_usd=final_nav - initial_nav,
        total_return=total_return,
        total_return_pct=total_return * 100,
        annualized_return=annualized_return,
        annualized_return_pct=annualized_return * 100 if annualized_return is not None else None,
        annualized_volatility=annualized_volatility,
        annualized_volatility_pct=annualized_volatility * 100 if annualized_volatility is not None else None,
        annualized_sharpe=_annualized_sharpe(returns, config.periods_per_year, config.annual_risk_free_rate),
        annualized_sortino=_annualized_sortino(returns, config.periods_per_year, config.annual_risk_free_rate),
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown * 100,
        calmar_ratio=annualized_return / max_drawdown if annualized_return is not None and max_drawdown else None,
        profitable_day_rate=sum(row.net_pnl_usd > 0 for row in days) / observations if observations else 0.0,
        profit_factor=positive / negative if negative else (None if positive == 0 else math.inf),
        total_transaction_cost_usd=sum(row.transaction_cost_usd for row in days),
        total_borrow_cost_usd=sum(row.borrow_cost_usd for row in days),
        average_daily_turnover=fmean(row.turnover for row in days) if days else 0.0,
        annualized_turnover=(fmean(row.turnover for row in days) * config.periods_per_year if days else 0.0),
    )


def _risk_outputs(
    scorer_id: str,
    horizon: int,
    trades: Sequence[PortfolioTradeRow],
    stock_rows: Sequence[DailyStockPnlRow],
    portfolio_days: Sequence[DailyPortfolioRow],
    config: PortfolioConfig,
) -> tuple[list[StockSummary], list[CorrelationRow], list[CovarianceRow]]:
    dates = [row.date for row in portfolio_days]
    symbols = sorted({row.symbol for row in stock_rows})
    by_key = {(row.date, row.symbol): row for row in stock_rows}
    series = {
        symbol: [
            by_key[(session_date, symbol)].return_contribution if (session_date, symbol) in by_key else 0.0
            for session_date in dates
        ]
        for symbol in symbols
    }
    portfolio_returns = [row.daily_return for row in portfolio_days]
    trade_counts: dict[str, int] = defaultdict(int)
    for trade in trades:
        trade_counts[trade.traded_symbol] += 1

    stock_summaries: list[StockSummary] = []
    for symbol in symbols:
        rows = [row for row in stock_rows if row.symbol == symbol]
        peak = 0.0
        max_loss = 0.0
        # Dollar contribution rows already form the correct additive path.
        for row in rows:
            cumulative = row.cumulative_net_pnl_usd
            peak = max(peak, cumulative)
            max_loss = max(max_loss, peak - cumulative)
        stock_summaries.append(
            StockSummary(
                scorer_id=scorer_id,
                horizon=horizon,
                symbol=symbol,
                observations=len(dates),
                active_days=sum(row.active_lots > 0 for row in rows),
                trade_count=trade_counts[symbol],
                gross_pnl_usd=sum(row.gross_pnl_usd for row in rows),
                transaction_cost_usd=sum(row.transaction_cost_usd for row in rows),
                borrow_cost_usd=sum(row.borrow_cost_usd for row in rows),
                net_pnl_usd=sum(row.net_pnl_usd for row in rows),
                annualized_contribution_sharpe=_annualized_sharpe(series[symbol], config.periods_per_year),
                correlation_to_portfolio=_correlation(series[symbol], portfolio_returns),
                max_peak_to_trough_loss_usd=max_loss,
            )
        )

    net_pnl_by_symbol = {row.symbol: row.net_pnl_usd for row in stock_summaries}
    correlations: list[CorrelationRow] = []
    for left_index, left in enumerate(symbols):
        for right in symbols[left_index + 1 :]:
            correlations.append(
                CorrelationRow(
                    scorer_id=scorer_id,
                    horizon=horizon,
                    symbol_a=left,
                    symbol_b=right,
                    observations=len(dates),
                    simultaneous_active_days=sum(
                        by_key[(session_date, left)].active_lots > 0
                        and by_key[(session_date, right)].active_lots > 0
                        for session_date in dates
                    ),
                    both_profitable=net_pnl_by_symbol[left] > 0 and net_pnl_by_symbol[right] > 0,
                    correlation=_correlation(series[left], series[right]),
                )
            )

    covariances: list[CovarianceRow] = []
    if symbols:
        matrix = np.asarray([[series[symbol][index] for symbol in symbols] for index in range(len(dates))], dtype=float)
        if len(dates) >= 2:
            sample = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
            if config.estimate_shrunk_covariance:
                shrunk = (
                    np.zeros((len(symbols), len(symbols)))
                    if np.all(np.var(matrix, axis=0) == 0)
                    else np.atleast_2d(LedoitWolf().fit(matrix).covariance_)
                )
            else:
                shrunk = None
        else:
            sample = None
            shrunk = None
        for left_index, left in enumerate(symbols):
            for right_index in range(left_index, len(symbols)):
                right = symbols[right_index]
                covariances.append(
                    CovarianceRow(
                        scorer_id=scorer_id,
                        horizon=horizon,
                        symbol_a=left,
                        symbol_b=right,
                        observations=len(dates),
                        sample_covariance=float(sample[left_index, right_index]) if sample is not None else None,
                        shrunk_covariance=float(shrunk[left_index, right_index]) if shrunk is not None else None,
                    )
                )
    return stock_summaries, correlations, covariances


def run_portfolio_backtest(
    signals: Sequence[DailySignal],
    prices: Sequence[PriceRow],
    policy: DecisionPolicyConfig,
    *,
    horizons: tuple[int, ...],
    initial_nav_usd: float,
    portfolio_config: PortfolioConfig | None = None,
    index_fallback: IndexFallback | None = None,
    timezone: str = "America/New_York",
    use_decision_policy: bool = True,
    decision_fn: DecisionFn | None = None,
) -> PortfolioBacktestResult:
    """Evaluate signals as independently funded scorer/horizon portfolios.

    A result is produced for every scorer present in the decisions and every
    requested horizon.  Raw-signal mode intentionally bypasses threshold and
    minimum-story gates but still returns the configured policy decisions in
    ``result.decisions`` for audit parity with the event-study evaluator.
    """

    if not math.isfinite(initial_nav_usd) or initial_nav_usd <= 0:
        raise ValueError("initial_nav_usd must be a finite positive number")
    if not horizons or any(not isinstance(horizon, int) or horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")
    config = portfolio_config or PortfolioConfig()
    policy_decisions = (decision_fn or make_trading_decisions)(list(signals), policy)
    basis = policy_decisions if use_decision_policy else _raw_signal_decisions(signals, policy)
    prices_by_symbol = _price_index(prices)

    all_trades: list[PortfolioTradeRow] = []
    all_stock_days: list[DailyStockPnlRow] = []
    all_portfolio_days: list[DailyPortfolioRow] = []
    summaries: list[PortfolioSummary] = []
    stock_summaries: list[StockSummary] = []
    correlations: list[CorrelationRow] = []
    covariances: list[CovarianceRow] = []

    scorer_ids = sorted({decision.scorer_id for decision in basis})
    for scorer_id in scorer_ids:
        scorer_decisions = [decision for decision in basis if decision.scorer_id == scorer_id]
        for horizon in sorted(horizons):
            candidates = _candidates(
                scorer_decisions,
                prices_by_symbol,
                horizon=horizon,
                index_fallback=index_fallback,
                timezone=timezone,
                duplicate_policy=config.duplicate_entry_policy,
            )
            trades, stock_days, portfolio_days = _simulate_book(
                scorer_id,
                horizon,
                candidates,
                prices,
                initial_nav=initial_nav_usd,
                policy=policy,
                config=config,
            )
            all_trades.extend(trades)
            all_stock_days.extend(stock_days)
            all_portfolio_days.extend(portfolio_days)
            summaries.append(
                _summarize_book(scorer_id, horizon, initial_nav_usd, trades, portfolio_days, config)
            )
            book_stock_summaries, book_correlations, book_covariances = _risk_outputs(
                scorer_id,
                horizon,
                trades,
                stock_days,
                portfolio_days,
                config,
            )
            stock_summaries.extend(book_stock_summaries)
            correlations.extend(book_correlations)
            covariances.extend(book_covariances)

    return PortfolioBacktestResult(
        decisions=policy_decisions,
        trades=all_trades,
        daily_stock_pnl=all_stock_days,
        daily_portfolio=all_portfolio_days,
        summaries=summaries,
        stock_summaries=stock_summaries,
        correlations=correlations,
        covariances=covariances,
    )
