"""Pure sentiment-state construction and development-only scaling.

The strategy pipeline deliberately keeps this module independent of news-source,
model-provider, and artifact code.  Orchestration converts scored news records to
``StateEvent`` objects, supplies the ordered exchange-session spine, and receives
immutable, serialisation-friendly transition and action rows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

import numpy as np

StateVariant = Literal["cash", "last_event_fixed_hold", "additive_decay", "decay_with_reset"]


class StateEngineError(ValueError):
    """Raised when state inputs violate a reproducibility invariant."""


@dataclass(frozen=True)
class StateConfig:
    """Parameters for one pooled state rule."""

    variant: StateVariant = "decay_with_reset"
    half_life_sessions: float = 3.0
    severity_power: float = 1.5
    reversal_reset: float = 0.75
    impulse_scale: float = 1.0
    state_cap: float = 3.0
    fixed_hold_sessions: int = 3

    def __post_init__(self) -> None:
        if self.variant not in ("cash", "last_event_fixed_hold", "additive_decay", "decay_with_reset"):
            raise ValueError(f"unsupported state variant: {self.variant}")
        if not math.isfinite(self.half_life_sessions) or self.half_life_sessions <= 0:
            raise ValueError("half_life_sessions must be finite and positive")
        if not math.isfinite(self.severity_power) or self.severity_power <= 0:
            raise ValueError("severity_power must be finite and positive")
        if not math.isfinite(self.reversal_reset) or not 0 <= self.reversal_reset <= 1:
            raise ValueError("reversal_reset must be in [0, 1]")
        if not math.isfinite(self.impulse_scale) or self.impulse_scale < 0:
            raise ValueError("impulse_scale must be finite and non-negative")
        if not math.isfinite(self.state_cap) or self.state_cap <= 0:
            raise ValueError("state_cap must be finite and positive")
        if self.fixed_hold_sessions < 1:
            raise ValueError("fixed_hold_sessions must be positive")


@dataclass(frozen=True)
class StateEvent:
    """One scored event assigned to an eligible execution session."""

    event_id: str
    symbol: str
    session: str
    available_at_utc: datetime
    score: float

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.available_at_utc.tzinfo is None or self.available_at_utc.utcoffset() is None:
            raise ValueError("available_at_utc must be timezone-aware")
        if not math.isfinite(self.score) or not -1 <= self.score <= 1:
            raise ValueError("score must be finite and in [-1, 1]")


class EventWeighter(Protocol):
    """Extension seam for a pre-declared event-weighting rule."""

    def weighted_impulse(self, event: StateEvent, base_impulse: float) -> float:
        """Return the impulse that the state engine should apply."""


@dataclass(frozen=True)
class IdentityEventWeighter:
    """The v1 rule: use the severity-transformed event score unchanged."""

    def weighted_impulse(self, event: StateEvent, base_impulse: float) -> float:
        del event
        return base_impulse


@dataclass(frozen=True)
class StateTransition:
    """Auditable state for one symbol at one exchange decision session."""

    session: str
    symbol: str
    previous_state: float
    elapsed_sessions: int
    state_after_decay: float
    eligible_event_count: int
    event_ids: tuple[str, ...]
    impulses: tuple[float, ...]
    reset_fractions: tuple[float, ...]
    reset_reasons: tuple[str, ...]
    state_after_events: float
    final_state: float
    run_id: str = ""


@dataclass(frozen=True)
class SymbolScale:
    symbol: str
    support: int
    stock_scale: float | None
    pooled_scale: float
    shrinkage_weight: float
    final_scale: float


@dataclass(frozen=True)
class ScaleEstimate:
    """Development-only scale fit; support means non-zero state sessions."""

    quantile: float
    shrinkage_k: float
    minimum_scale: float
    development_sessions: tuple[str, ...]
    pooled_scale: float
    symbols: tuple[SymbolScale, ...]

    def as_mapping(self) -> dict[str, float]:
        return {row.symbol: row.final_scale for row in self.symbols}


@dataclass(frozen=True)
class StateAction:
    session: str
    symbol: str
    state: float
    scale: float
    raw_action: float
    action: float


def severity_impulse(score: float, severity_power: float) -> float:
    """Transform a score in ``[-1, 1]`` while preserving its sign."""

    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError("score must be finite and in [-1, 1]")
    if not math.isfinite(severity_power) or severity_power <= 0:
        raise ValueError("severity_power must be finite and positive")
    return math.copysign(abs(score) ** severity_power, score) if score else 0.0


def build_state_table(
    sessions: Sequence[str],
    symbols: Sequence[str],
    events: Sequence[StateEvent],
    config: StateConfig,
    *,
    session_ordinals: Mapping[str, int] | None = None,
    event_weighter: EventWeighter | None = None,
    run_id: str = "",
) -> list[StateTransition]:
    """Build deterministic daily states over a declared trading-session spine.

    ``session_ordinals`` makes skipped exchange sessions explicit.  When omitted,
    the supplied session sequence is treated as a complete consecutive exchange
    calendar and enumerated from zero.  Events outside the declared spine fail
    closed instead of being silently dropped.
    """

    ordered_sessions = tuple(sessions)
    ordered_symbols = tuple(sorted(set(symbols)))
    if len(ordered_sessions) != len(set(ordered_sessions)):
        raise StateEngineError("sessions must be unique")
    if tuple(sorted(ordered_sessions)) != ordered_sessions:
        raise StateEngineError("sessions must be strictly chronological")
    if not ordered_symbols:
        return []

    ordinals = dict(session_ordinals or {session: index for index, session in enumerate(ordered_sessions)})
    if set(ordinals) != set(ordered_sessions):
        raise StateEngineError("session_ordinals must cover exactly the supplied sessions")
    ordinal_values = [ordinals[session] for session in ordered_sessions]
    if any(right <= left for left, right in zip(ordinal_values, ordinal_values[1:], strict=False)):
        raise StateEngineError("session ordinals must be strictly increasing")

    symbol_set = set(ordered_symbols)
    session_set = set(ordered_sessions)
    events_by_key: dict[tuple[str, str], list[StateEvent]] = defaultdict(list)
    seen_event_ids: set[str] = set()
    for event in events:
        if event.event_id in seen_event_ids:
            raise StateEngineError(f"duplicate state event_id: {event.event_id}")
        seen_event_ids.add(event.event_id)
        if event.symbol not in symbol_set:
            raise StateEngineError(f"event {event.event_id} has undeclared symbol {event.symbol}")
        if event.session not in session_set:
            raise StateEngineError(f"event {event.event_id} has undeclared session {event.session}")
        events_by_key[(event.symbol, event.session)].append(event)
    for keyed_events in events_by_key.values():
        keyed_events.sort(key=lambda event: (event.available_at_utc, event.event_id))

    weighter = event_weighter or IdentityEventWeighter()
    rows: list[StateTransition] = []
    for symbol in ordered_symbols:
        previous = 0.0
        previous_ordinal: int | None = None
        last_fixed_score = 0.0
        last_fixed_ordinal: int | None = None
        for session in ordered_sessions:
            ordinal = ordinals[session]
            elapsed = 0 if previous_ordinal is None else ordinal - previous_ordinal
            eligible = events_by_key.get((symbol, session), [])
            event_ids = tuple(event.event_id for event in eligible)

            if config.variant == "cash":
                after_decay = 0.0
                impulses = tuple(0.0 for _ in eligible)
                resets = tuple(0.0 for _ in eligible)
                reset_reasons = tuple("cash_variant" for _ in eligible)
                current = 0.0
            elif config.variant == "last_event_fixed_hold":
                after_decay = previous
                impulses = tuple(event.score for event in eligible)
                resets = tuple(0.0 for _ in eligible)
                reset_reasons = tuple("not_applicable_fixed_hold" for _ in eligible)
                if eligible:
                    last_fixed_score = eligible[-1].score
                    last_fixed_ordinal = ordinal
                if last_fixed_ordinal is not None and ordinal - last_fixed_ordinal < config.fixed_hold_sessions:
                    current = last_fixed_score
                else:
                    current = 0.0
            else:
                after_decay = previous * 2 ** (-elapsed / config.half_life_sessions)
                current = after_decay
                weighted_impulses: list[float] = []
                reset_fractions: list[float] = []
                event_reset_reasons: list[str] = []
                reversal_reset = 0.0 if config.variant == "additive_decay" else config.reversal_reset
                for event in eligible:
                    base_impulse = severity_impulse(event.score, config.severity_power)
                    impulse = weighter.weighted_impulse(event, base_impulse)
                    if not math.isfinite(impulse):
                        raise StateEngineError(f"event weighter returned non-finite impulse for {event.event_id}")
                    reset = reversal_reset * abs(impulse) if impulse * current < 0 else 0.0
                    if config.variant == "additive_decay":
                        reset_reason = "reset_disabled_additive_decay"
                    elif reset > 0:
                        reset_reason = "opposite_sign_reversal"
                    else:
                        reset_reason = "no_opposite_sign_reversal"
                    current = min(
                        config.state_cap,
                        max(-config.state_cap, (1 - reset) * current + config.impulse_scale * impulse),
                    )
                    weighted_impulses.append(impulse)
                    reset_fractions.append(reset)
                    event_reset_reasons.append(reset_reason)
                impulses = tuple(weighted_impulses)
                resets = tuple(reset_fractions)
                reset_reasons = tuple(event_reset_reasons)

            rows.append(
                StateTransition(
                    session=session,
                    symbol=symbol,
                    previous_state=previous,
                    elapsed_sessions=elapsed,
                    state_after_decay=after_decay,
                    eligible_event_count=len(eligible),
                    event_ids=event_ids,
                    impulses=impulses,
                    reset_fractions=resets,
                    reset_reasons=reset_reasons,
                    state_after_events=current,
                    final_state=current,
                    run_id=run_id,
                )
            )
            previous = current
            previous_ordinal = ordinal
    rows.sort(key=lambda row: (row.session, row.symbol))
    return rows


def estimate_state_scales(
    transitions: Sequence[StateTransition],
    development_sessions: Sequence[str],
    *,
    quantile: float = 0.75,
    shrinkage_k: float = 50.0,
    minimum_scale: float = 0.05,
) -> ScaleEstimate:
    """Fit robust stock scales from development non-zero state sessions only."""

    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    if not math.isfinite(shrinkage_k) or shrinkage_k < 0:
        raise ValueError("shrinkage_k must be finite and non-negative")
    if not math.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be finite and positive")
    development = tuple(development_sessions)
    if len(development) != len(set(development)):
        raise ValueError("development_sessions must be unique")
    development_set = set(development)
    symbols = sorted({row.symbol for row in transitions})
    by_symbol: dict[str, list[float]] = {symbol: [] for symbol in symbols}
    pooled_values: list[float] = []
    for row in transitions:
        if row.session not in development_set or row.final_state == 0:
            continue
        value = abs(row.final_state)
        by_symbol[row.symbol].append(value)
        pooled_values.append(value)

    pooled_raw = float(np.quantile(np.asarray(pooled_values, dtype=float), quantile)) if pooled_values else minimum_scale
    pooled_scale = max(minimum_scale, pooled_raw)
    estimates: list[SymbolScale] = []
    for symbol in symbols:
        values = by_symbol[symbol]
        support = len(values)
        if not values:
            stock_scale = None
            weight = 0.0
            final = pooled_scale
        else:
            stock_scale = max(minimum_scale, float(np.quantile(np.asarray(values, dtype=float), quantile)))
            weight = support / (support + shrinkage_k) if shrinkage_k else 1.0
            final = max(minimum_scale, weight * stock_scale + (1 - weight) * pooled_scale)
        estimates.append(
            SymbolScale(
                symbol=symbol,
                support=support,
                stock_scale=stock_scale,
                pooled_scale=pooled_scale,
                shrinkage_weight=weight,
                final_scale=final,
            )
        )
    return ScaleEstimate(
        quantile=quantile,
        shrinkage_k=shrinkage_k,
        minimum_scale=minimum_scale,
        development_sessions=development,
        pooled_scale=pooled_scale,
        symbols=tuple(estimates),
    )


def apply_state_scales(
    transitions: Sequence[StateTransition],
    scales: ScaleEstimate | Mapping[str, float],
    *,
    no_trade_band: float = 0.10,
) -> list[StateAction]:
    """Convert state to a bounded action using already-fitted scales."""

    if not math.isfinite(no_trade_band) or not 0 <= no_trade_band <= 1:
        raise ValueError("no_trade_band must be in [0, 1]")
    scale_map = scales.as_mapping() if isinstance(scales, ScaleEstimate) else dict(scales)
    actions: list[StateAction] = []
    for row in transitions:
        scale = scale_map.get(row.symbol)
        if scale is None or not math.isfinite(scale) or scale <= 0:
            raise StateEngineError(f"missing positive scale for {row.symbol}")
        raw_action = math.tanh(row.final_state / scale)
        action = 0.0 if abs(raw_action) < no_trade_band else raw_action
        actions.append(
            StateAction(
                session=row.session,
                symbol=row.symbol,
                state=row.final_state,
                scale=scale,
                raw_action=raw_action,
                action=action,
            )
        )
    return actions
