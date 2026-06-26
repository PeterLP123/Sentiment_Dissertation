"""Tests for the price provider seam and on-disk cache."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from sentiment_benchmark.prices import (
    CachedPriceProvider,
    PriceProviderError,
    PriceRow,
    YFinancePriceProvider,
    make_price_provider,
)


class RecordingProvider:
    """Inner provider that returns deterministic rows and counts fetched symbols."""

    def __init__(self) -> None:
        self.fetched: list[str] = []

    def fetch(self, symbols: Sequence[str], start: str, end: str) -> list[PriceRow]:
        rows: list[PriceRow] = []
        for symbol in symbols:
            self.fetched.append(symbol)
            rows.append(PriceRow(symbol, "2026-06-08", 100.0, 105.0, 95.0, 101.5, 1000.0, True))
            rows.append(PriceRow(symbol, "2026-06-09", 101.0, 106.0, 96.0, 102.5, 1100.0, False))
        return rows


def test_cache_hit_performs_no_network_call(tmp_path: Path) -> None:
    inner = RecordingProvider()
    provider = CachedPriceProvider(inner=inner, cache_dir=tmp_path)

    first = provider.fetch(["AAPL"], "2026-06-08", "2026-06-15")
    assert inner.fetched == ["AAPL"]
    assert (tmp_path / "AAPL__2026-06-08__2026-06-15.csv").exists()

    second = provider.fetch(["AAPL"], "2026-06-08", "2026-06-15")
    # No new fetch: the second call is served entirely from cache.
    assert inner.fetched == ["AAPL"]
    assert second == first


def test_cache_round_trip_preserves_values_and_types(tmp_path: Path) -> None:
    inner = RecordingProvider()
    provider = CachedPriceProvider(inner=inner, cache_dir=tmp_path)
    provider.fetch(["AAPL"], "2026-06-08", "2026-06-15")

    reloaded = CachedPriceProvider(inner=RecordingProvider(), cache_dir=tmp_path).fetch(
        ["AAPL"], "2026-06-08", "2026-06-15"
    )
    row = reloaded[0]
    assert row.symbol == "AAPL"
    assert row.open == 100.0
    assert row.repaired is True
    assert reloaded[1].repaired is False


def test_cache_is_per_symbol(tmp_path: Path) -> None:
    inner = RecordingProvider()
    provider = CachedPriceProvider(inner=inner, cache_dir=tmp_path)
    provider.fetch(["AAPL", "MSFT"], "2026-06-08", "2026-06-15")
    assert sorted(inner.fetched) == ["AAPL", "MSFT"]

    # Re-requesting AAPL plus a new symbol only fetches the new one.
    provider.fetch(["AAPL", "NVDA"], "2026-06-08", "2026-06-15")
    assert inner.fetched.count("AAPL") == 1
    assert inner.fetched.count("NVDA") == 1


def test_symbol_with_caret_is_filename_safe(tmp_path: Path) -> None:
    inner = RecordingProvider()
    provider = CachedPriceProvider(inner=inner, cache_dir=tmp_path)
    provider.fetch(["^GSPC"], "2026-06-08", "2026-06-15")
    assert (tmp_path / "_GSPC__2026-06-08__2026-06-15.csv").exists()


def test_make_price_provider_selection(tmp_path: Path) -> None:
    assert isinstance(make_price_provider("yfinance"), YFinancePriceProvider)
    cached = make_price_provider("yahoo", cache_dir=tmp_path)
    assert isinstance(cached, CachedPriceProvider)
    with pytest.raises(PriceProviderError):
        make_price_provider("alpha_vantage")
