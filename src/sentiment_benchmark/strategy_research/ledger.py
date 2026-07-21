"""One-position-per-symbol open-to-open portfolio ledger."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .market import OpenToOpenReturn
from .portfolio import TargetPortfolio


class LedgerError(ValueError):
    """Raised when target or return indexing cannot be accounted safely."""


@dataclass(frozen=True)
class OrderRecord:
    session: str
    symbol: str
    current_weight: float
    target_weight: float
    order_weight: float
    turnover: float
    transaction_cost_return: float
    transaction_cost_usd: float


@dataclass(frozen=True)
class PositionContribution:
    session: str
    next_session: str
    symbol: str
    target_weight: float
    asset_return: float
    gross_return_contribution: float
    transaction_cost_return: float
    net_return_contribution: float
    gross_pnl_usd: float
    transaction_cost_usd: float
    net_pnl_usd: float


@dataclass(frozen=True)
class DailyLedgerRow:
    session: str
    next_session: str | None
    start_nav_usd: float
    gross_return: float
    transaction_cost: float
    net_return: float
    end_nav_usd: float
    turnover: float
    gross_exposure: float
    long_exposure: float
    short_exposure: float
    net_exposure: float
    cash_weight: float
    active_names: int
    current_weights: tuple[tuple[str, float], ...]
    target_weights: tuple[tuple[str, float], ...]
    orders: tuple[OrderRecord, ...]
    contributions: tuple[PositionContribution, ...]
    final_liquidation: bool = False


def run_open_to_open_ledger(
    targets: Sequence[TargetPortfolio],
    returns: Sequence[OpenToOpenReturn],
    *,
    cost_rate_per_side: float = 0.001,
    initial_nav_usd: float = 1_000_000.0,
    accounting_reset_session: str | None = None,
    force_final_liquidation: bool = True,
) -> list[DailyLedgerRow]:
    """Account targets, drifted current weights, costs, and final liquidation.

    A target filled at open ``t`` earns the supplied ``t -> t+1`` return.  Before
    the next rebalance, its current weight is marked by that realised return and
    the portfolio's net NAV change.  Events never create separate lots: each
    target mapping contains at most one position per symbol.
    """

    if not math.isfinite(cost_rate_per_side) or cost_rate_per_side < 0:
        raise ValueError("cost_rate_per_side must be finite and non-negative")
    if not math.isfinite(initial_nav_usd) or initial_nav_usd <= 0:
        raise ValueError("initial_nav_usd must be finite and positive")
    target_rows = tuple(targets)
    target_sessions = tuple(target.session for target in target_rows)
    if len(target_sessions) != len(set(target_sessions)) or tuple(sorted(target_sessions)) != target_sessions:
        raise LedgerError("target sessions must be unique and chronological")
    if accounting_reset_session is not None and accounting_reset_session not in target_sessions:
        raise LedgerError(
            f"accounting reset session {accounting_reset_session} is absent from the target session spine"
        )

    return_lookup: dict[tuple[str, str], OpenToOpenReturn] = {}
    next_by_session: dict[str, str] = {}
    for row in returns:
        key = (row.session, row.symbol)
        if key in return_lookup:
            raise LedgerError(f"duplicate return for {row.symbol} on {row.session}")
        if not math.isfinite(row.value):
            raise LedgerError(f"non-finite return for {row.symbol} on {row.session}")
        existing_next = next_by_session.setdefault(row.session, row.next_session)
        if existing_next != row.next_session:
            raise LedgerError(f"inconsistent next session after {row.session}")
        return_lookup[key] = row

    ledger: list[DailyLedgerRow] = []
    current_weights: dict[str, float] = {}
    nav = initial_nav_usd
    final_next_session: str | None = None
    expected_session: str | None = None
    for target in target_rows:
        if expected_session is not None and target.session != expected_session:
            raise LedgerError(
                f"target session {target.session} does not follow prior return endpoint {expected_session}; "
                "supply a target for every decision session"
            )
        target_symbols = [position.symbol for position in target.positions]
        if len(target_symbols) != len(set(target_symbols)):
            raise LedgerError(f"target portfolio contains duplicate symbols on {target.session}")
        target_weights = {symbol: weight for symbol, weight in target.weights().items() if abs(weight) >= 1e-15}
        all_symbols = sorted(set(current_weights) | set(target_weights))
        if target.session == accounting_reset_session:
            # Positions have already drifted through the final development
            # interval. Reset only the reporting NAV at the first evaluation
            # open, preserving those point-in-time current weights.
            nav = initial_nav_usd
        start_nav = nav
        orders: list[OrderRecord] = []
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            desired = target_weights.get(symbol, 0.0)
            order = desired - current
            turnover = abs(order)
            cost_return = cost_rate_per_side * turnover
            orders.append(
                OrderRecord(
                    session=target.session,
                    symbol=symbol,
                    current_weight=current,
                    target_weight=desired,
                    order_weight=order,
                    turnover=turnover,
                    transaction_cost_return=cost_return,
                    transaction_cost_usd=start_nav * cost_return,
                )
            )
        total_turnover = sum(order.turnover for order in orders)
        total_cost = cost_rate_per_side * total_turnover

        next_session = next_by_session.get(target.session)
        if next_session is None:
            if target_weights:
                raise LedgerError(f"no executable return interval after {target.session}")
            raise LedgerError(f"cannot infer the next exchange session after {target.session}")
        final_next_session = next_session
        expected_session = next_session
        order_by_symbol = {order.symbol: order for order in orders}
        contributions: list[PositionContribution] = []
        gross_return = 0.0
        for symbol in sorted(set(target_weights) | set(order_by_symbol)):
            weight = target_weights.get(symbol, 0.0)
            if weight:
                return_row = return_lookup.get((target.session, symbol))
                if return_row is None:
                    raise LedgerError(f"missing return for active {symbol} position on {target.session}")
                asset_return = return_row.value
            else:
                asset_return = 0.0
            gross_contribution = weight * asset_return
            order_cost = order_by_symbol[symbol].transaction_cost_return
            net_contribution = gross_contribution - order_cost
            gross_return += gross_contribution
            contributions.append(
                PositionContribution(
                    session=target.session,
                    next_session=next_session,
                    symbol=symbol,
                    target_weight=weight,
                    asset_return=asset_return,
                    gross_return_contribution=gross_contribution,
                    transaction_cost_return=order_cost,
                    net_return_contribution=net_contribution,
                    gross_pnl_usd=start_nav * gross_contribution,
                    transaction_cost_usd=start_nav * order_cost,
                    net_pnl_usd=start_nav * net_contribution,
                )
            )
        net_return = gross_return - total_cost
        if 1 + net_return <= 0:
            raise LedgerError(f"portfolio NAV became non-positive on {target.session}")
        nav = start_nav * (1 + net_return)
        long_exposure = sum(weight for weight in target_weights.values() if weight > 0)
        short_exposure = sum(-weight for weight in target_weights.values() if weight < 0)
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        ledger.append(
            DailyLedgerRow(
                session=target.session,
                next_session=next_session,
                start_nav_usd=start_nav,
                gross_return=gross_return,
                transaction_cost=total_cost,
                net_return=net_return,
                end_nav_usd=nav,
                turnover=total_turnover,
                gross_exposure=gross_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                cash_weight=1 - net_exposure,
                active_names=sum(weight != 0 for weight in target_weights.values()),
                current_weights=tuple(sorted(current_weights.items())),
                target_weights=tuple(sorted(target_weights.items())),
                orders=tuple(orders),
                contributions=tuple(contributions),
            )
        )

        # Positions are sized against start-of-session NAV. Both realized asset
        # returns and paid trading costs change end NAV, so the next session's
        # pre-trade weights must be marked against the after-cost denominator.
        # Ignoring costs here understates later turnover and liquidation costs.
        denominator = 1 + net_return
        current_weights = {
            symbol: weight * (1 + return_lookup[(target.session, symbol)].value) / denominator
            for symbol, weight in target_weights.items()
            if weight
        }

    if force_final_liquidation and final_next_session is not None:
        start_nav = nav
        liquidation_orders = tuple(
            OrderRecord(
                session=final_next_session,
                symbol=symbol,
                current_weight=weight,
                target_weight=0.0,
                order_weight=-weight,
                turnover=abs(weight),
                transaction_cost_return=cost_rate_per_side * abs(weight),
                transaction_cost_usd=start_nav * cost_rate_per_side * abs(weight),
            )
            for symbol, weight in sorted(current_weights.items())
        )
        turnover = sum(order.turnover for order in liquidation_orders)
        cost = cost_rate_per_side * turnover
        nav = start_nav * (1 - cost)
        liquidation_contributions = tuple(
            PositionContribution(
                session=final_next_session,
                next_session=final_next_session,
                symbol=order.symbol,
                target_weight=0.0,
                asset_return=0.0,
                gross_return_contribution=0.0,
                transaction_cost_return=order.transaction_cost_return,
                net_return_contribution=-order.transaction_cost_return,
                gross_pnl_usd=0.0,
                transaction_cost_usd=order.transaction_cost_usd,
                net_pnl_usd=-order.transaction_cost_usd,
            )
            for order in liquidation_orders
        )
        ledger.append(
            DailyLedgerRow(
                session=final_next_session,
                next_session=None,
                start_nav_usd=start_nav,
                gross_return=0.0,
                transaction_cost=cost,
                net_return=-cost,
                end_nav_usd=nav,
                turnover=turnover,
                gross_exposure=0.0,
                long_exposure=0.0,
                short_exposure=0.0,
                net_exposure=0.0,
                cash_weight=1.0,
                active_names=0,
                current_weights=tuple(sorted(current_weights.items())),
                target_weights=(),
                orders=liquidation_orders,
                contributions=liquidation_contributions,
                final_liquidation=True,
            )
        )
    return ledger


def evaluation_rows(rows: Sequence[DailyLedgerRow], evaluation_start: str) -> list[DailyLedgerRow]:
    """Select only intervals starting inside the locked evaluation block."""

    return [row for row in rows if row.session >= evaluation_start]


def assert_no_boundary_crossing(rows: Sequence[DailyLedgerRow], boundary: str) -> None:
    """Reject any interval that begins before and ends after a split boundary."""

    for row in rows:
        if row.next_session is not None and row.session < boundary < row.next_session:
            raise LedgerError(f"return interval {row.session} -> {row.next_session} crosses boundary {boundary}")
