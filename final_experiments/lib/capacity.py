"""Point-in-time liquidity and square-root market-impact diagnostics.

This module is intentionally scoped to implementation analysis. It replays an
already specified sequence of target portfolios; it does not alter signals,
select holdings, or optimise a capacity threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentiment_benchmark.strategy_research.market import OpenToOpenReturn
from sentiment_benchmark.strategy_research.portfolio import TargetPortfolio


class CapacityError(ValueError):
    """Raised when a capacity replay cannot be accounted point in time."""


@dataclass(frozen=True)
class ImpactConfig:
    """Frozen execution assumptions for one capacity replay."""

    fixed_cost_rate_per_side: float = 0.001
    impact_coefficient: float = 1.0
    initial_nav_usd: float = 1_000_000.0
    participation_limit: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.fixed_cost_rate_per_side) or self.fixed_cost_rate_per_side < 0:
            raise ValueError("fixed_cost_rate_per_side must be finite and non-negative")
        if not math.isfinite(self.impact_coefficient) or self.impact_coefficient < 0:
            raise ValueError("impact_coefficient must be finite and non-negative")
        if not math.isfinite(self.initial_nav_usd) or self.initial_nav_usd <= 0:
            raise ValueError("initial_nav_usd must be finite and positive")
        if not math.isfinite(self.participation_limit) or not 0 < self.participation_limit <= 1:
            raise ValueError("participation_limit must be finite and in (0, 1]")


@dataclass(frozen=True)
class CapacitySimulation:
    """Daily ledger and non-zero order audit from one impact replay."""

    daily: pd.DataFrame
    orders: pd.DataFrame


def build_lagged_liquidity(
    prices: pd.DataFrame,
    *,
    window_sessions: int = 20,
    minimum_observations: int = 5,
) -> pd.DataFrame:
    """Build strictly lagged ADV and open-return volatility by symbol.

    Dollar volume is past close times reported volume. Both rolling estimates
    are shifted by one complete row, so neither current-session volume nor the
    return ending at the current open enters a decision-session estimate.
    """

    if window_sessions < 2:
        raise ValueError("window_sessions must be at least two")
    if not 2 <= minimum_observations <= window_sessions:
        raise ValueError("minimum_observations must be in [2, window_sessions]")
    required = {"session_date", "symbol", "open", "close", "volume"}
    if missing := required - set(prices.columns):
        raise ValueError(f"prices missing columns: {sorted(missing)}")

    frame = prices.loc[:, sorted(required)].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if frame[["session_date", "symbol"]].isna().any().any():
        raise ValueError("session_date and symbol must be non-missing")
    if frame.duplicated(["session_date", "symbol"]).any():
        raise ValueError("prices must be unique by session_date and symbol")
    for column in ("open", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must be finite and positive")

    frame = frame.sort_values(["symbol", "session_date"], kind="mergesort")
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    frame["open_to_open_return"] = frame.groupby("symbol", sort=False)["open"].pct_change(fill_method=None)
    frame["adv20_usd"] = frame.groupby("symbol", sort=False)["dollar_volume"].transform(
        lambda values: values.rolling(window_sessions, min_periods=minimum_observations).mean().shift(1)
    )
    frame["sigma20"] = frame.groupby("symbol", sort=False)["open_to_open_return"].transform(
        lambda values: values.rolling(window_sessions, min_periods=minimum_observations).std(ddof=1).shift(1)
    )
    return frame.loc[:, ["session_date", "symbol", "adv20_usd", "sigma20"]].reset_index(drop=True)


def _liquidity_lookup(liquidity: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    required = {"session_date", "symbol", "adv20_usd", "sigma20"}
    if missing := required - set(liquidity.columns):
        raise ValueError(f"liquidity missing columns: {sorted(missing)}")
    frame = liquidity.loc[:, ["session_date", "symbol", "adv20_usd", "sigma20"]].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.strftime("%Y-%m-%d")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if frame.duplicated(["session_date", "symbol"]).any():
        raise ValueError("liquidity must be unique by session_date and symbol")
    return {(row.session_date, row.symbol): (float(row.adv20_usd), float(row.sigma20)) for row in frame.itertuples(index=False)}


def simulate_square_root_impact(
    targets: Sequence[TargetPortfolio],
    returns: Sequence[OpenToOpenReturn],
    liquidity: pd.DataFrame,
    *,
    config: ImpactConfig | None = None,
    force_final_liquidation: bool = True,
) -> CapacitySimulation:
    """Replay fixed targets with drift, fixed costs, and square-root impact.

    For a non-zero order with notional ``q`` and lagged dollar volume ``ADV``,
    the added price-impact rate is ``Y * sigma * sqrt(q / ADV)``. The cost is
    charged on the full order notional. Impact changes NAV and therefore also
    changes the next session's drifted pre-trade weights.
    """

    cfg = config or ImpactConfig()
    target_rows = tuple(targets)
    if not target_rows:
        raise CapacityError("targets must contain at least one session")
    target_sessions = tuple(target.session for target in target_rows)
    if len(target_sessions) != len(set(target_sessions)) or tuple(sorted(target_sessions)) != target_sessions:
        raise CapacityError("target sessions must be unique and chronological")

    return_lookup: dict[tuple[str, str], OpenToOpenReturn] = {}
    next_by_session: dict[str, str] = {}
    for row in returns:
        key = (row.session, row.symbol)
        if key in return_lookup:
            raise CapacityError(f"duplicate return for {row.symbol} on {row.session}")
        if not math.isfinite(row.value):
            raise CapacityError(f"non-finite return for {row.symbol} on {row.session}")
        existing_next = next_by_session.setdefault(row.session, row.next_session)
        if existing_next != row.next_session:
            raise CapacityError(f"inconsistent next session after {row.session}")
        return_lookup[key] = row
    liquidity_lookup = _liquidity_lookup(liquidity)

    daily_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    current_weights: dict[str, float] = {}
    nav = cfg.initial_nav_usd
    expected_session: str | None = None
    final_next_session: str | None = None

    def account_order(
        *,
        session: str,
        symbol: str,
        current_weight: float,
        target_weight: float,
        start_nav: float,
        final_liquidation: bool,
    ) -> tuple[float, float, float]:
        order_weight = target_weight - current_weight
        turnover = abs(order_weight)
        if turnover < 1e-15:
            return 0.0, 0.0, 0.0
        key = (session, symbol)
        if key not in liquidity_lookup:
            raise CapacityError(f"missing lagged liquidity for {symbol} on {session}")
        adv_usd, sigma = liquidity_lookup[key]
        if not math.isfinite(adv_usd) or adv_usd <= 0:
            raise CapacityError(f"lagged ADV must be finite and positive for {symbol} on {session}")
        if not math.isfinite(sigma) or sigma < 0:
            raise CapacityError(f"lagged volatility must be finite and non-negative for {symbol} on {session}")

        trade_notional = start_nav * turnover
        participation = trade_notional / adv_usd
        impact_rate = cfg.impact_coefficient * sigma * math.sqrt(participation)
        fixed_cost_return = cfg.fixed_cost_rate_per_side * turnover
        impact_cost_return = impact_rate * turnover
        order_rows.append(
            {
                "session_date": pd.Timestamp(session),
                "symbol": symbol,
                "final_liquidation": final_liquidation,
                "start_nav": start_nav,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "order_weight": order_weight,
                "turnover": turnover,
                "trade_notional_usd": trade_notional,
                "adv20_usd": adv_usd,
                "sigma20": sigma,
                "participation": participation,
                "participation_breach": participation > cfg.participation_limit,
                "impact_rate": impact_rate,
                "fixed_cost_return": fixed_cost_return,
                "impact_cost_return": impact_cost_return,
                "fixed_cost_usd": start_nav * fixed_cost_return,
                "impact_cost_usd": start_nav * impact_cost_return,
            }
        )
        return turnover, fixed_cost_return, impact_cost_return

    for target in target_rows:
        if expected_session is not None and target.session != expected_session:
            raise CapacityError(f"target session {target.session} does not follow prior return endpoint {expected_session}")
        symbols = [position.symbol for position in target.positions]
        if len(symbols) != len(set(symbols)):
            raise CapacityError(f"target portfolio contains duplicate symbols on {target.session}")
        target_weights = {symbol: float(weight) for symbol, weight in target.weights().items() if abs(float(weight)) >= 1e-15}
        if not all(math.isfinite(weight) for weight in target_weights.values()):
            raise CapacityError(f"target portfolio contains non-finite weights on {target.session}")

        next_session = next_by_session.get(target.session)
        if next_session is None:
            raise CapacityError(f"cannot infer the next exchange session after {target.session}")
        final_next_session = next_session
        expected_session = next_session
        start_nav = nav
        total_turnover = 0.0
        fixed_cost = 0.0
        impact_cost = 0.0
        order_start = len(order_rows)
        for symbol in sorted(set(current_weights) | set(target_weights)):
            turnover, fixed, impact = account_order(
                session=target.session,
                symbol=symbol,
                current_weight=current_weights.get(symbol, 0.0),
                target_weight=target_weights.get(symbol, 0.0),
                start_nav=start_nav,
                final_liquidation=False,
            )
            total_turnover += turnover
            fixed_cost += fixed
            impact_cost += impact

        gross_return = 0.0
        for symbol, weight in target_weights.items():
            row = return_lookup.get((target.session, symbol))
            if row is None:
                raise CapacityError(f"missing return for active {symbol} on {target.session}")
            gross_return += weight * row.value
        net_return = gross_return - fixed_cost - impact_cost
        if 1.0 + net_return <= 0:
            raise CapacityError(f"portfolio NAV became non-positive on {target.session}")
        nav = start_nav * (1.0 + net_return)
        session_orders = order_rows[order_start:]
        daily_rows.append(
            {
                "session_date": pd.Timestamp(target.session),
                "return_end_date": pd.Timestamp(next_session),
                "start_nav": start_nav,
                "end_nav": nav,
                "gross_return": gross_return,
                "fixed_cost": fixed_cost,
                "impact_cost": impact_cost,
                "net_return": net_return,
                "turnover": total_turnover,
                "trade_notional_usd": sum(float(row["trade_notional_usd"]) for row in session_orders),
                "max_participation": max((float(row["participation"]) for row in session_orders), default=0.0),
                "participation_breaches": sum(bool(row["participation_breach"]) for row in session_orders),
                "gross_exposure": sum(abs(weight) for weight in target_weights.values()),
                "active_names": len(target_weights),
                "n_orders": len(session_orders),
                "final_liquidation": False,
            }
        )

        denominator = 1.0 + net_return
        current_weights = {
            symbol: weight * (1.0 + return_lookup[(target.session, symbol)].value) / denominator
            for symbol, weight in target_weights.items()
        }

    if force_final_liquidation and final_next_session is not None:
        start_nav = nav
        total_turnover = 0.0
        fixed_cost = 0.0
        impact_cost = 0.0
        order_start = len(order_rows)
        for symbol, weight in sorted(current_weights.items()):
            turnover, fixed, impact = account_order(
                session=final_next_session,
                symbol=symbol,
                current_weight=weight,
                target_weight=0.0,
                start_nav=start_nav,
                final_liquidation=True,
            )
            total_turnover += turnover
            fixed_cost += fixed
            impact_cost += impact
        net_return = -fixed_cost - impact_cost
        if 1.0 + net_return <= 0:
            raise CapacityError("portfolio NAV became non-positive during final liquidation")
        nav = start_nav * (1.0 + net_return)
        session_orders = order_rows[order_start:]
        daily_rows.append(
            {
                "session_date": pd.Timestamp(final_next_session),
                "return_end_date": pd.NaT,
                "start_nav": start_nav,
                "end_nav": nav,
                "gross_return": 0.0,
                "fixed_cost": fixed_cost,
                "impact_cost": impact_cost,
                "net_return": net_return,
                "turnover": total_turnover,
                "trade_notional_usd": sum(float(row["trade_notional_usd"]) for row in session_orders),
                "max_participation": max((float(row["participation"]) for row in session_orders), default=0.0),
                "participation_breaches": sum(bool(row["participation_breach"]) for row in session_orders),
                "gross_exposure": 0.0,
                "active_names": 0,
                "n_orders": len(session_orders),
                "final_liquidation": True,
            }
        )

    return CapacitySimulation(
        daily=pd.DataFrame.from_records(daily_rows),
        orders=pd.DataFrame.from_records(order_rows),
    )


def fold_terminal_liquidation(daily: pd.DataFrame) -> pd.DataFrame:
    """Fold one terminal liquidation row into the final return interval."""

    required = {
        "final_liquidation",
        "net_return",
        "fixed_cost",
        "impact_cost",
        "turnover",
        "trade_notional_usd",
        "max_participation",
        "participation_breaches",
        "n_orders",
        "end_nav",
    }
    if missing := required - set(daily.columns):
        raise ValueError(f"daily missing columns: {sorted(missing)}")
    liquidation = daily.loc[daily["final_liquidation"]]
    intervals = daily.loc[~daily["final_liquidation"]].copy().reset_index(drop=True)
    if len(liquidation) != 1 or intervals.empty:
        raise CapacityError("expected one terminal liquidation and at least one return interval")
    final = liquidation.iloc[0]
    last = intervals.index[-1]
    intervals.loc[last, "net_return"] = (1.0 + float(intervals.loc[last, "net_return"])) * (1.0 + float(final["net_return"])) - 1.0
    for column in (
        "fixed_cost",
        "impact_cost",
        "turnover",
        "trade_notional_usd",
        "participation_breaches",
        "n_orders",
    ):
        intervals.loc[last, column] += final[column]
    intervals.loc[last, "max_participation"] = max(float(intervals.loc[last, "max_participation"]), float(final["max_participation"]))
    intervals.loc[last, "end_nav"] = final["end_nav"]
    return intervals


def summarize_capacity(
    simulation: CapacitySimulation,
    *,
    participation_limit: float = 0.05,
    annualization_periods: int = 252,
) -> dict[str, float | int | bool]:
    """Summarise one predeclared AUM/impact replay without selecting it."""

    if not math.isfinite(participation_limit) or not 0 < participation_limit <= 1:
        raise ValueError("participation_limit must be finite and in (0, 1]")
    if annualization_periods < 1:
        raise ValueError("annualization_periods must be positive")
    daily = fold_terminal_liquidation(simulation.daily)
    orders = simulation.orders
    if daily.empty or orders.empty:
        raise CapacityError("capacity summary requires return intervals and non-zero orders")
    gross = daily["gross_return"].to_numpy(dtype=float)
    net = daily["net_return"].to_numpy(dtype=float)

    def sharpe(values: np.ndarray) -> float:
        standard_deviation = float(np.std(values, ddof=1))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0:
            return float("nan")
        return float(np.mean(values) / standard_deviation * math.sqrt(annualization_periods))

    equity = np.r_[1.0, np.cumprod(1.0 + net)]
    participation = orders["participation"].to_numpy(dtype=float)
    trade_notional = float(orders["trade_notional_usd"].sum())
    return {
        "n_sessions": len(daily),
        "active_sessions": int(daily["gross_exposure"].gt(0).sum()),
        "n_orders": len(orders),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "total_return_gross": float(np.prod(1.0 + gross) - 1.0),
        "total_return_net": float(equity[-1] - 1.0),
        "maximum_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1.0)),
        "annualized_volatility_net": float(np.std(net, ddof=1) * math.sqrt(annualization_periods)),
        "total_turnover": float(daily["turnover"].sum()),
        "total_trade_notional_usd": trade_notional,
        "total_fixed_cost_usd": float(orders["fixed_cost_usd"].sum()),
        "total_impact_cost_usd": float(orders["impact_cost_usd"].sum()),
        "mean_impact_bps_per_traded_dollar": float(10_000.0 * orders["impact_cost_usd"].sum() / trade_notional),
        "median_participation": float(np.median(participation)),
        "p95_participation": float(np.quantile(participation, 0.95)),
        "maximum_participation": float(np.max(participation)),
        "orders_over_participation_limit": int((participation > participation_limit).sum()),
        "fraction_orders_over_participation_limit": float(np.mean(participation > participation_limit)),
        "all_orders_within_participation_limit": bool(np.all(participation <= participation_limit)),
    }
