"""Price data providers for the trading backtest.

This module separates price *acquisition* (network and cache I/O) from the pure
backtest core so runs are deterministic and offline-replayable. ``PriceRow`` is
the canonical daily price record and lives here as its natural home;
``trading_strategy`` re-exports it for backwards compatibility.

The provider seam plugs into the existing ``price_loader`` injection point in
``run_trading_strategy``: a thin adapter in ``trading_strategy`` turns a
``TradingStrategyConfig`` into a ``PriceProvider`` and calls ``fetch``.
"""

from __future__ import annotations

import csv
import importlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable


class PriceProviderError(RuntimeError):
    """Raised when price data cannot be obtained or read from cache."""


@dataclass(frozen=True)
class PriceRow:
    """One symbol's daily OHLCV bar.

    Prices are adjusted close-equivalent when the provider is configured with
    ``adjusted=True`` (the default), matching the ``*_adjusted_*`` fields the
    backtest produces.
    """

    symbol: str
    session_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    repaired: bool


@runtime_checkable
class PriceProvider(Protocol):
    """Returns sorted daily ``PriceRow`` bars for the requested symbols/window."""

    def fetch(self, symbols: Sequence[str], start: str, end: str) -> list[PriceRow]: ...


def _sorted(rows: list[PriceRow]) -> list[PriceRow]:
    return sorted(rows, key=lambda row: (row.symbol, row.session_date))


@dataclass(frozen=True)
class YFinancePriceProvider:
    """Downloads daily bars from Yahoo Finance via ``yfinance``.

    ``adjusted=True`` uses ``auto_adjust`` so corporate actions are folded into
    the prices; ``repair=True`` enables yfinance's bad-tick repair.
    """

    cache_tag: ClassVar[str] = "yfinance"

    adjusted: bool = True
    repair: bool = True

    def fetch(self, symbols: Sequence[str], start: str, end: str) -> list[PriceRow]:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - exercised only without yfinance
            raise PriceProviderError("yfinance is required for trading price data") from exc
        rows: list[PriceRow] = []
        for symbol in symbols:
            frame = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=self.adjusted,
                actions=False,
                repair=self.repair,
                keepna=False,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
            if frame is None or frame.empty:
                raise PriceProviderError(f"no Yahoo Finance prices returned for {symbol}")
            for index, value in frame.iterrows():
                rows.append(
                    PriceRow(
                        symbol=symbol,
                        session_date=index.date().isoformat(),
                        open=float(value["Open"]),
                        high=float(value["High"]),
                        low=float(value["Low"]),
                        close=float(value["Close"]),
                        volume=float(value["Volume"]),
                        repaired=bool(value.get("Repaired?", False)),
                    )
                )
        return _sorted(rows)


# Daily-interval OHLCV fields for the LSEG historical-pricing access layer.
_LSEG_FIELDS = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"]
# Corrections plus capital-change (split) adjustments. NOTE: unlike yfinance's
# auto_adjust, LSEG does not back-adjust for dividends — prices follow the
# *price-return* convention. State this wherever cross-provider numbers meet.
_LSEG_ADJUSTMENTS = ["exchangeCorrection", "manualCorrection", "CCH"]


def _is_nan(value: Any) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


@dataclass(frozen=True)
class LsegPriceProvider:
    """Daily bars from LSEG Workspace via the ``lseg.data`` access layer.

    Requires a running Workspace session (same requirement as the news
    collection). Symbols are translated to RICs through ``ric_overrides``;
    unmapped symbols are passed through unchanged and left to LSEG's own
    resolution. ``adjusted=True`` applies exchange/manual corrections and
    capital-change (split) adjustments; dividends are *not* folded in, so
    returns are price returns rather than yfinance-style total returns.
    LSEG treats ``end`` as inclusive (yfinance excludes it); the backtest
    is insensitive to one extra trailing session.
    """

    cache_tag: ClassVar[str] = "lseg"

    adjusted: bool = True
    ric_overrides: Mapping[str, str] = field(default_factory=dict)

    def fetch(self, symbols: Sequence[str], start: str, end: str) -> list[PriceRow]:
        try:
            ld = importlib.import_module("lseg.data")
        except ImportError as exc:  # pragma: no cover - exercised only without lseg-data
            raise PriceProviderError("the lseg-data SDK is required for LSEG price data") from exc
        try:
            ld.open_session()
        except Exception as exc:
            raise PriceProviderError(f"cannot open LSEG session (is Workspace running?): {exc}") from exc
        try:
            rows: list[PriceRow] = []
            for symbol in symbols:
                ric = self.ric_overrides.get(symbol, symbol)
                try:
                    frame = ld.get_history(
                        universe=ric,
                        fields=_LSEG_FIELDS,
                        interval="daily",
                        start=start,
                        end=end,
                        adjustments=_LSEG_ADJUSTMENTS if self.adjusted else None,
                    )
                except Exception as exc:
                    raise PriceProviderError(f"LSEG history request failed for {symbol} ({ric}): {exc}") from exc
                if frame is None or getattr(frame, "empty", True):
                    raise PriceProviderError(f"no LSEG prices returned for {symbol} ({ric})")
                missing = [name for name in _LSEG_FIELDS[:4] if name not in frame.columns]
                if missing:
                    raise PriceProviderError(
                        f"LSEG history for {symbol} ({ric}) lacks fields {missing}; got {list(frame.columns)}"
                    )
                for index, value in frame.iterrows():
                    ohlc = [value["OPEN_PRC"], value["HIGH_1"], value["LOW_1"], value["TRDPRC_1"]]
                    if any(_is_nan(item) for item in ohlc):
                        continue  # non-trading or unpriced session
                    session_date = index.date().isoformat() if hasattr(index, "date") else str(index)[:10]
                    volume = value.get("ACVOL_UNS") if hasattr(value, "get") else None
                    rows.append(
                        PriceRow(
                            symbol=symbol,
                            session_date=session_date,
                            open=float(ohlc[0]),
                            high=float(ohlc[1]),
                            low=float(ohlc[2]),
                            close=float(ohlc[3]),
                            volume=0.0 if volume is None or _is_nan(volume) else float(volume),
                            repaired=False,
                        )
                    )
            return _sorted(rows)
        finally:
            try:
                ld.close_session()
            except Exception:  # pragma: no cover - session teardown is best-effort
                pass


_CACHE_FIELDS = [field.name for field in fields(PriceRow)]


def _cache_filename(tag: str, symbol: str, start: str, end: str) -> str:
    safe_symbol = re.sub(r"[^A-Za-z0-9._-]", "_", symbol)
    return f"{tag}__{safe_symbol}__{start}__{end}.csv"


def _read_cache(path: Path) -> list[PriceRow] | None:
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [
                PriceRow(
                    symbol=row["symbol"],
                    session_date=row["session_date"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    repaired=row["repaired"] == "True",
                )
                for row in reader
            ]
    except (OSError, KeyError, ValueError) as exc:
        raise PriceProviderError(f"corrupt price cache {path}: {exc}") from exc
    return rows


def _write_cache(path: Path, rows: list[PriceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CACHE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "symbol": row.symbol,
                    "session_date": row.session_date,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                    "repaired": row.repaired,
                }
            )


@dataclass(frozen=True)
class CachedPriceProvider:
    """Wraps a provider with a per-``(symbol, window)`` on-disk CSV cache.

    A cache hit performs no network call, so tuning sweeps that re-run the same
    window are deterministic and offline. Each symbol is cached independently so
    adding a company does not invalidate the others. Filenames are namespaced by
    the inner provider's ``cache_tag`` so switching provider with the same
    ``cache_dir`` cannot serve one source's bars as another's.
    """

    inner: PriceProvider
    cache_dir: Path

    def fetch(self, symbols: Sequence[str], start: str, end: str) -> list[PriceRow]:
        tag = getattr(self.inner, "cache_tag", type(self.inner).__name__.lower())
        rows: list[PriceRow] = []
        for symbol in symbols:
            path = self.cache_dir / _cache_filename(tag, symbol, start, end)
            cached = _read_cache(path)
            if cached is None:
                cached = self.inner.fetch([symbol], start, end)
                _write_cache(path, cached)
            rows.extend(cached)
        return _sorted(rows)


def make_price_provider(
    provider: str = "yfinance",
    *,
    cache_dir: str | Path | None = None,
    adjusted: bool = True,
    ric_overrides: Mapping[str, str] | None = None,
) -> PriceProvider:
    """Build a price provider from config values.

    ``provider`` selects the network backend; passing ``cache_dir`` wraps it in a
    :class:`CachedPriceProvider`. ``ric_overrides`` (symbol → RIC) applies to the
    LSEG provider only.
    """

    name = (provider or "yfinance").strip().lower()
    if name in {"yfinance", "yahoo"}:
        base: PriceProvider = YFinancePriceProvider(adjusted=adjusted)
    elif name in {"lseg", "workspace", "refinitiv"}:
        base = LsegPriceProvider(adjusted=adjusted, ric_overrides=dict(ric_overrides or {}))
    else:
        raise PriceProviderError(f"unknown price provider: {provider!r}")
    if cache_dir is not None:
        return CachedPriceProvider(inner=base, cache_dir=Path(cache_dir))
    return base
