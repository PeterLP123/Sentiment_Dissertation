"""Validated adjusted-open returns and strictly lagged volatility features."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np


class PriceDataError(ValueError):
    """Raised when a price panel cannot support fail-closed execution."""


@dataclass(frozen=True)
class AdjustedOpen:
    """One split-adjusted execution price on an identified exchange session."""

    symbol: str
    session: str
    adjusted_open: float
    adjustment_supported: bool = True

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        try:
            date.fromisoformat(self.session)
        except ValueError as exc:
            raise ValueError(f"invalid ISO session: {self.session}") from exc
        if not math.isfinite(self.adjusted_open) or self.adjusted_open <= 0:
            raise ValueError("adjusted_open must be finite and positive")


@dataclass(frozen=True)
class OpenToOpenReturn:
    """Return earned by a weight filled at ``session`` until ``next_session``."""

    symbol: str
    session: str
    next_session: str
    value: float


@dataclass(frozen=True)
class LaggedVolatility:
    """Volatility available at the open of ``session``."""

    symbol: str
    session: str
    value: float | None
    observations: int
    eligible: bool


def validate_adjusted_opens(rows: Sequence[AdjustedOpen]) -> None:
    """Validate uniqueness, input chronology, and adjustment support."""

    seen: set[tuple[str, str]] = set()
    previous_session: dict[str, str] = {}
    for row in rows:
        key = (row.symbol, row.session)
        if key in seen:
            raise PriceDataError(f"duplicate adjusted-open row for {row.symbol} on {row.session}")
        seen.add(key)
        if not row.adjustment_supported:
            raise PriceDataError(f"unsupported adjusted-open treatment for {row.symbol} on {row.session}")
        previous = previous_session.get(row.symbol)
        if previous is not None and row.session <= previous:
            raise PriceDataError(f"non-monotonic sessions for {row.symbol}: {previous}, {row.session}")
        previous_session[row.symbol] = row.session


def load_adjusted_opens_csv(
    path: str | Path,
    *,
    symbol_column: str = "symbol",
    session_column: str = "session_date",
    price_column: str = "adjusted_open",
    adjustment_supported: bool = True,
) -> list[AdjustedOpen]:
    """Load a declared adjusted-open field without guessing field semantics.

    The caller must name the actual execution-price column.  A generic ``open``
    field is therefore usable only when a validated price manifest explicitly
    establishes that it is adjusted and the caller passes ``price_column='open'``.
    """

    price_path = Path(path)
    try:
        with price_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {symbol_column, session_column, price_column}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise PriceDataError(f"price panel is missing declared columns: {sorted(missing)}")
            loaded = [
                AdjustedOpen(
                    symbol=row[symbol_column].strip(),
                    session=row[session_column].strip(),
                    adjusted_open=float(row[price_column]),
                    adjustment_supported=adjustment_supported,
                )
                for row in reader
            ]
    except OSError as exc:
        raise PriceDataError(f"could not read price panel {price_path}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise PriceDataError(f"invalid adjusted-open value in {price_path}: {exc}") from exc
    validate_adjusted_opens(loaded)
    return loaded


def validate_exchange_sessions(rows: Sequence[AdjustedOpen], *, calendar_name: str = "XNYS") -> None:
    """Cross-check every declared price session against an exchange calendar."""

    if not rows:
        return
    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover - dependency is locked by the project
        raise PriceDataError("exchange-calendars is required for session validation") from exc
    calendar = xcals.get_calendar(calendar_name)
    first = min(row.session for row in rows)
    last = max(row.session for row in rows)
    valid_sessions = {str(value.date()) for value in calendar.sessions_in_range(first, last)}
    invalid = sorted({row.session for row in rows if row.session not in valid_sessions})
    if invalid:
        raise PriceDataError(f"price panel contains non-{calendar_name} sessions: {invalid[:5]}")


def calculate_open_to_open_returns(rows: Sequence[AdjustedOpen]) -> list[OpenToOpenReturn]:
    """Calculate adjacent-session adjusted-open returns after strict validation."""

    validate_adjusted_opens(rows)
    grouped: dict[str, list[AdjustedOpen]] = defaultdict(list)
    for row in rows:
        grouped[row.symbol].append(row)
    returns: list[OpenToOpenReturn] = []
    for symbol in sorted(grouped):
        symbol_rows = grouped[symbol]
        for current, following in zip(symbol_rows, symbol_rows[1:], strict=False):
            value = following.adjusted_open / current.adjusted_open - 1
            if not math.isfinite(value):
                raise PriceDataError(f"non-finite return for {symbol} on {current.session}")
            returns.append(
                OpenToOpenReturn(
                    symbol=symbol,
                    session=current.session,
                    next_session=following.session,
                    value=value,
                )
            )
    returns.sort(key=lambda row: (row.session, row.symbol))
    return returns


def calculate_lagged_volatility(
    returns: Sequence[OpenToOpenReturn],
    decision_sessions: Sequence[str],
    symbols: Sequence[str],
    *,
    window_sessions: int = 20,
    minimum_sessions: int = 15,
) -> list[LaggedVolatility]:
    """Calculate point-in-time volatility without using the current interval.

    A return row labelled ``session -> next_session`` becomes eligible only when
    ``next_session < decision_session``.  Thus a decision at open ``t`` never
    uses the return ending at that same open, matching the declared conservative
    one-session lag.
    """

    if window_sessions < 2:
        raise ValueError("window_sessions must be at least two")
    if minimum_sessions < 2 or minimum_sessions > window_sessions:
        raise ValueError("minimum_sessions must be between two and window_sessions")
    sessions = tuple(decision_sessions)
    if len(sessions) != len(set(sessions)) or tuple(sorted(sessions)) != sessions:
        raise ValueError("decision_sessions must be unique and chronological")
    ordered_symbols = tuple(sorted(set(symbols)))

    by_symbol: dict[str, list[OpenToOpenReturn]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in returns:
        key = (row.symbol, row.session)
        if key in seen:
            raise PriceDataError(f"duplicate return for {row.symbol} on {row.session}")
        seen.add(key)
        if row.next_session <= row.session:
            raise PriceDataError(f"invalid return interval for {row.symbol}: {row.session} -> {row.next_session}")
        if not math.isfinite(row.value):
            raise PriceDataError(f"non-finite return for {row.symbol} on {row.session}")
        by_symbol[row.symbol].append(row)
    for symbol_returns in by_symbol.values():
        symbol_returns.sort(key=lambda row: row.next_session)

    result: list[LaggedVolatility] = []
    for session in sessions:
        for symbol in ordered_symbols:
            available = [row.value for row in by_symbol.get(symbol, []) if row.next_session < session]
            window = available[-window_sessions:]
            observations = len(window)
            if observations < minimum_sessions:
                value = None
                eligible = False
            else:
                value = float(np.std(np.asarray(window, dtype=float), ddof=1))
                eligible = math.isfinite(value)
                if not eligible:
                    value = None
            result.append(
                LaggedVolatility(
                    symbol=symbol,
                    session=session,
                    value=value,
                    observations=observations,
                    eligible=eligible,
                )
            )
    return result


def volatility_mapping(rows: Sequence[LaggedVolatility]) -> dict[tuple[str, str], float | None]:
    """Return a checked ``(session, symbol)`` lookup for portfolio construction."""

    result: dict[tuple[str, str], float | None] = {}
    for row in rows:
        key = (row.session, row.symbol)
        if key in result:
            raise PriceDataError(f"duplicate volatility for {row.symbol} on {row.session}")
        result[key] = row.value if row.eligible else None
    return result


def price_coverage(
    returns: Sequence[OpenToOpenReturn], sessions: Sequence[str], symbols: Sequence[str]
) -> Mapping[str, int]:
    """Count executable intervals by symbol over a requested session block."""

    requested = set(sessions)
    counts = {symbol: 0 for symbol in sorted(set(symbols))}
    for row in returns:
        if row.symbol in counts and row.session in requested:
            counts[row.symbol] += 1
    return counts
