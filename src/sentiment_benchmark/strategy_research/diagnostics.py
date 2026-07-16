"""Portfolio and cross-stock diagnostic primitives for research reports."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

import numpy as np

from .ledger import DailyLedgerRow
from .state import StateTransition
from .tuning import annualized_sharpe


@dataclass(frozen=True)
class PortfolioMetrics:
    observations: int
    cumulative_gross_return: float
    cumulative_net_return: float
    annualized_mean_return: float
    annualized_volatility: float | None
    after_cost_sharpe: float | None
    maximum_drawdown: float
    average_turnover: float
    annualized_turnover: float
    total_transaction_cost_usd: float
    active_portfolio_days: int
    profitable_active_day_rate: float
    average_gross_exposure: float
    average_net_exposure: float
    maximum_name_concentration: float
    supported_stocks: int


@dataclass(frozen=True)
class StockDiagnostic:
    symbol: str
    event_count: int
    active_days: int
    average_absolute_state: float | None
    average_state_persistence_sessions: float | None
    average_absolute_position: float
    gross_pnl_usd: float
    transaction_cost_usd: float
    net_pnl_usd: float
    turnover: float
    maximum_weight: float
    long_pnl_usd: float
    short_pnl_usd: float


@dataclass(frozen=True)
class ContributionConcentration:
    top_stock: str | None
    top_stock_share: float | None
    top_five_share: float | None
    portfolio_net_pnl_usd: float


def calculate_portfolio_metrics(
    rows: Sequence[DailyLedgerRow], *, periods_per_year: int = 252
) -> PortfolioMetrics:
    """Summarise a fixed ledger without selecting or dropping any stocks."""

    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    if not rows:
        return PortfolioMetrics(
            observations=0,
            cumulative_gross_return=0.0,
            cumulative_net_return=0.0,
            annualized_mean_return=0.0,
            annualized_volatility=None,
            after_cost_sharpe=None,
            maximum_drawdown=0.0,
            average_turnover=0.0,
            annualized_turnover=0.0,
            total_transaction_cost_usd=0.0,
            active_portfolio_days=0,
            profitable_active_day_rate=0.0,
            average_gross_exposure=0.0,
            average_net_exposure=0.0,
            maximum_name_concentration=0.0,
            supported_stocks=0,
        )
    gross_values = np.asarray([row.gross_return for row in rows], dtype=float)
    net_values = np.asarray([row.net_return for row in rows], dtype=float)
    if not np.all(np.isfinite(gross_values)) or not np.all(np.isfinite(net_values)):
        raise ValueError("ledger returns must be finite")
    cumulative_gross = float(np.prod(1 + gross_values) - 1)
    cumulative_net = float(np.prod(1 + net_values) - 1)
    annualized_mean = float(np.mean(net_values) * periods_per_year)
    annualized_volatility = (
        float(np.std(net_values, ddof=1) * math.sqrt(periods_per_year)) if len(net_values) >= 2 else None
    )
    nav_path = np.asarray([rows[0].start_nav_usd, *(row.end_nav_usd for row in rows)], dtype=float)
    running_peak = np.maximum.accumulate(nav_path)
    maximum_drawdown = float(np.min(nav_path / running_peak - 1))
    supported = {
        symbol
        for row in rows
        for symbol, weight in row.target_weights
        if weight != 0
    }
    maximum_name = max(
        (abs(weight) for row in rows for _, weight in row.target_weights),
        default=0.0,
    )
    active_rows = [row for row in rows if row.active_names > 0]
    return PortfolioMetrics(
        observations=len(rows),
        cumulative_gross_return=cumulative_gross,
        cumulative_net_return=cumulative_net,
        annualized_mean_return=annualized_mean,
        annualized_volatility=annualized_volatility,
        after_cost_sharpe=annualized_sharpe(tuple(float(value) for value in net_values), periods_per_year=periods_per_year),
        maximum_drawdown=maximum_drawdown,
        average_turnover=fmean(row.turnover for row in rows),
        annualized_turnover=fmean(row.turnover for row in rows) * periods_per_year,
        total_transaction_cost_usd=sum(
            order.transaction_cost_usd for row in rows for order in row.orders
        ),
        active_portfolio_days=len(active_rows),
        profitable_active_day_rate=(
            sum(row.net_return > 0 for row in active_rows) / len(active_rows) if active_rows else 0.0
        ),
        average_gross_exposure=fmean(row.gross_exposure for row in rows),
        average_net_exposure=fmean(row.net_exposure for row in rows),
        maximum_name_concentration=maximum_name,
        supported_stocks=len(supported),
    )


def calculate_stock_diagnostics(
    rows: Sequence[DailyLedgerRow], state_rows: Sequence[StateTransition] | None = None
) -> list[StockDiagnostic]:
    """Aggregate fixed-position contributions and actual order costs by stock."""

    positions: dict[str, list[float]] = defaultdict(list)
    gross_pnl: dict[str, float] = defaultdict(float)
    net_pnl: dict[str, float] = defaultdict(float)
    transaction_cost: dict[str, float] = defaultdict(float)
    turnover: dict[str, float] = defaultdict(float)
    long_pnl: dict[str, float] = defaultdict(float)
    short_pnl: dict[str, float] = defaultdict(float)
    symbols: set[str] = set()
    for row in rows:
        row_weights = dict(row.target_weights)
        symbols.update(row_weights)
        for contribution in row.contributions:
            symbol = contribution.symbol
            symbols.add(symbol)
            gross_pnl[symbol] += contribution.gross_pnl_usd
            net_pnl[symbol] += contribution.net_pnl_usd
            transaction_cost[symbol] += contribution.transaction_cost_usd
            order = next((item for item in row.orders if item.symbol == symbol), None)
            side_weight = contribution.target_weight or (order.current_weight if order is not None else 0.0)
            if side_weight > 0:
                long_pnl[symbol] += contribution.net_pnl_usd
            elif side_weight < 0:
                short_pnl[symbol] += contribution.net_pnl_usd
        for order in row.orders:
            symbols.add(order.symbol)
            turnover[order.symbol] += order.turnover

    for row in rows:
        row_weights = dict(row.target_weights)
        for symbol in symbols:
            positions[symbol].append(row_weights.get(symbol, 0.0))

    states_by_symbol: dict[str, list[StateTransition]] = defaultdict(list)
    for state_row in state_rows or ():
        symbols.add(state_row.symbol)
        states_by_symbol[state_row.symbol].append(state_row)

    def state_persistence(symbol: str) -> float | None:
        ordered_states = sorted(states_by_symbol[symbol], key=lambda row: row.session)
        run_lengths: list[int] = []
        current_run = 0
        for state_row in ordered_states:
            if state_row.final_state != 0:
                current_run += 1
            elif current_run:
                run_lengths.append(current_run)
                current_run = 0
        if current_run:
            run_lengths.append(current_run)
        return fmean(run_lengths) if run_lengths else None

    return [
        StockDiagnostic(
            symbol=symbol,
            event_count=sum(row.eligible_event_count for row in states_by_symbol[symbol]),
            active_days=sum(weight != 0 for weight in positions[symbol]),
            average_absolute_state=(
                fmean(abs(row.final_state) for row in states_by_symbol[symbol]) if states_by_symbol[symbol] else None
            ),
            average_state_persistence_sessions=state_persistence(symbol),
            average_absolute_position=(
                fmean(abs(weight) for weight in positions[symbol]) if positions[symbol] else 0.0
            ),
            gross_pnl_usd=gross_pnl[symbol],
            transaction_cost_usd=transaction_cost[symbol],
            net_pnl_usd=net_pnl[symbol],
            turnover=turnover[symbol],
            maximum_weight=max((abs(weight) for weight in positions[symbol]), default=0.0),
            long_pnl_usd=long_pnl[symbol],
            short_pnl_usd=short_pnl[symbol],
        )
        for symbol in sorted(symbols)
    ]


def contribution_concentration(stock_rows: Sequence[StockDiagnostic]) -> ContributionConcentration:
    """Report absolute PnL concentration without modifying the evaluated universe.

    The signed portfolio PnL is retained separately, while concentration shares
    use absolute stock-level net PnL.  Signed shares become misleading when
    winners and losers offset or the evaluated portfolio loses money.
    """

    ordered = sorted(stock_rows, key=lambda row: (-abs(row.net_pnl_usd), row.symbol))
    total = sum(row.net_pnl_usd for row in ordered)
    absolute_total = sum(abs(row.net_pnl_usd) for row in ordered)
    if not ordered:
        return ContributionConcentration(None, None, None, 0.0)
    if absolute_total == 0:
        top_share = None
        top_five_share = None
    else:
        top_share = abs(ordered[0].net_pnl_usd) / absolute_total
        top_five_share = sum(abs(row.net_pnl_usd) for row in ordered[:5]) / absolute_total
    return ContributionConcentration(
        top_stock=ordered[0].symbol,
        top_stock_share=top_share,
        top_five_share=top_five_share,
        portfolio_net_pnl_usd=total,
    )


def leave_one_stock_out_returns(rows: Sequence[DailyLedgerRow]) -> Mapping[str, Mapping[str, float]]:
    """Recombine fixed daily contributions while excluding each stock in turn.

    This is a diagnostic of the already-fixed strategy, not a re-optimised
    portfolio.  Each symbol's allocated trading costs are removed together with
    its gross contribution.
    """

    symbols = sorted({contribution.symbol for row in rows for contribution in row.contributions})
    result: dict[str, dict[str, float]] = {symbol: {} for symbol in symbols}
    for row in rows:
        by_symbol = {contribution.symbol: contribution.net_return_contribution for contribution in row.contributions}
        total = sum(by_symbol.values())
        for symbol in symbols:
            result[symbol][row.session] = total - by_symbol.get(symbol, 0.0)
    return result


def returns_by_time_block(rows: Sequence[DailyLedgerRow], *, prefix_length: int = 7) -> Mapping[str, float]:
    """Compound net returns by an ISO-date prefix such as month (7) or year (4)."""

    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.session[:prefix_length]].append(row.net_return)
    return {
        block: float(np.prod(1 + np.asarray(values, dtype=float)) - 1)
        for block, values in sorted(grouped.items())
    }
