"""Deterministic sign-preserving target-weight projection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


class PortfolioProjectionError(ValueError):
    """Raised when an input or projected portfolio violates a hard constraint."""


@dataclass(frozen=True)
class PortfolioConstraints:
    gross_limit: float = 1.0
    net_limit: float = 0.20
    single_name_limit: float = 0.05
    volatility_floor: float = 0.005

    def __post_init__(self) -> None:
        if not math.isfinite(self.gross_limit) or self.gross_limit <= 0:
            raise ValueError("gross_limit must be finite and positive")
        if not math.isfinite(self.net_limit) or self.net_limit < 0:
            raise ValueError("net_limit must be finite and non-negative")
        if self.net_limit > self.gross_limit:
            raise ValueError("net_limit must not exceed gross_limit")
        if not math.isfinite(self.single_name_limit) or self.single_name_limit <= 0:
            raise ValueError("single_name_limit must be finite and positive")
        if self.single_name_limit > self.gross_limit:
            raise ValueError("single_name_limit must not exceed gross_limit")
        if not math.isfinite(self.volatility_floor) or self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be finite and positive")


@dataclass(frozen=True)
class PositionTarget:
    symbol: str
    action: float
    volatility: float | None
    raw_weight: float
    target_weight: float
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class TargetPortfolio:
    session: str
    positions: tuple[PositionTarget, ...]
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    cash_weight: float

    def weights(self) -> dict[str, float]:
        return {position.symbol: position.target_weight for position in self.positions}


def project_target_weights(
    session: str,
    actions: Mapping[str, float],
    volatilities: Mapping[str, float | None],
    constraints: PortfolioConstraints | None = None,
) -> TargetPortfolio:
    """Convert actions to a constrained portfolio in a fixed operation order.

    The order is inverse-volatility scaling, gross normalisation, name capping,
    then one-sided net correction.  The final step only reduces the overweight
    side, so projection can never invent an opposite-sign position merely to
    force neutrality.  Unused risk capacity remains cash.
    """

    limits = constraints or PortfolioConstraints()
    raw: dict[str, float] = {}
    exclusions: dict[str, str | None] = {}
    for symbol in sorted(actions):
        action = actions[symbol]
        if not math.isfinite(action) or not -1 <= action <= 1:
            raise PortfolioProjectionError(f"action for {symbol} must be finite and in [-1, 1]")
        if action == 0:
            raw[symbol] = 0.0
            exclusions[symbol] = None
            continue
        volatility = volatilities.get(symbol)
        if volatility is None:
            raw[symbol] = 0.0
            exclusions[symbol] = "missing_lagged_volatility"
            continue
        if not math.isfinite(volatility) or volatility < 0:
            raise PortfolioProjectionError(f"volatility for {symbol} must be finite and non-negative")
        raw[symbol] = action / max(volatility, limits.volatility_floor)
        exclusions[symbol] = None

    raw_gross = sum(abs(weight) for weight in raw.values())
    gross_multiplier = min(1.0, limits.gross_limit / raw_gross) if raw_gross else 0.0
    projected = {
        symbol: max(-limits.single_name_limit, min(limits.single_name_limit, weight * gross_multiplier))
        for symbol, weight in raw.items()
    }

    net = sum(projected.values())
    if net > limits.net_limit:
        positive_total = sum(weight for weight in projected.values() if weight > 0)
        required_reduction = net - limits.net_limit
        multiplier = max(0.0, 1 - required_reduction / positive_total) if positive_total else 0.0
        projected = {symbol: weight * multiplier if weight > 0 else weight for symbol, weight in projected.items()}
    elif net < -limits.net_limit:
        negative_total = sum(-weight for weight in projected.values() if weight < 0)
        required_reduction = -limits.net_limit - net
        multiplier = max(0.0, 1 - required_reduction / negative_total) if negative_total else 0.0
        projected = {symbol: weight * multiplier if weight < 0 else weight for symbol, weight in projected.items()}

    projected = {symbol: 0.0 if abs(weight) < 1e-15 else weight for symbol, weight in projected.items()}
    gross = sum(abs(weight) for weight in projected.values())
    net = sum(projected.values())
    long_exposure = sum(weight for weight in projected.values() if weight > 0)
    short_exposure = sum(-weight for weight in projected.values() if weight < 0)
    _assert_constraints(actions, projected, limits)

    positions = tuple(
        PositionTarget(
            symbol=symbol,
            action=actions[symbol],
            volatility=volatilities.get(symbol),
            raw_weight=raw[symbol],
            target_weight=projected[symbol],
            exclusion_reason=exclusions[symbol],
        )
        for symbol in sorted(actions)
    )
    return TargetPortfolio(
        session=session,
        positions=positions,
        gross_exposure=gross,
        net_exposure=net,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        cash_weight=1 - net,
    )


def _assert_constraints(
    actions: Mapping[str, float], weights: Mapping[str, float], limits: PortfolioConstraints, *, tolerance: float = 1e-12
) -> None:
    if sum(abs(weight) for weight in weights.values()) > limits.gross_limit + tolerance:
        raise PortfolioProjectionError("gross exposure limit violated")
    if abs(sum(weights.values())) > limits.net_limit + tolerance:
        raise PortfolioProjectionError("net exposure limit violated")
    if any(abs(weight) > limits.single_name_limit + tolerance for weight in weights.values()):
        raise PortfolioProjectionError("single-name exposure limit violated")
    for symbol, weight in weights.items():
        action = actions[symbol]
        if weight and action * weight <= 0:
            raise PortfolioProjectionError(f"projection changed the signal sign for {symbol}")
