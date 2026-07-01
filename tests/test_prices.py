"""Tests for the price provider seam and on-disk cache."""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest

from sentiment_benchmark.prices import (
    CachedPriceProvider,
    LsegPriceProvider,
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
    assert (tmp_path / "recordingprovider__AAPL__2026-06-08__2026-06-15.csv").exists()

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
    assert (tmp_path / "recordingprovider___GSPC__2026-06-08__2026-06-15.csv").exists()


def test_cache_is_namespaced_by_provider(tmp_path: Path) -> None:
    """Switching provider with the same cache_dir must not serve stale bars."""
    CachedPriceProvider(inner=YFinancePriceProvider(), cache_dir=tmp_path)
    CachedPriceProvider(inner=LsegPriceProvider(), cache_dir=tmp_path)
    from sentiment_benchmark.prices import _cache_filename

    yf_name = _cache_filename(YFinancePriceProvider.cache_tag, "AAPL", "2026-06-08", "2026-06-15")
    lseg_name = _cache_filename(LsegPriceProvider.cache_tag, "AAPL", "2026-06-08", "2026-06-15")
    assert yf_name != lseg_name
    assert yf_name.startswith("yfinance__") and lseg_name.startswith("lseg__")


def test_make_price_provider_selection(tmp_path: Path) -> None:
    assert isinstance(make_price_provider("yfinance"), YFinancePriceProvider)
    cached = make_price_provider("yahoo", cache_dir=tmp_path)
    assert isinstance(cached, CachedPriceProvider)
    lseg = make_price_provider("lseg", ric_overrides={"AAPL": "AAPL.O"})
    assert isinstance(lseg, LsegPriceProvider)
    assert lseg.ric_overrides == {"AAPL": "AAPL.O"}
    with pytest.raises(PriceProviderError):
        make_price_provider("alpha_vantage")


# --- LSEG provider, exercised against a faked lseg.data module ------------- #
class _FakeFrame:
    """Duck-typed stand-in for the DataFrame lseg.data.get_history returns."""

    def __init__(self, rows: dict[str, dict[str, float | None]], columns: list[str]) -> None:
        self._rows = rows
        self.columns = columns
        self.empty = not rows

    def iterrows(self):
        for iso_date, values in self._rows.items():
            yield datetime.fromisoformat(iso_date), values


class _FakeLseg(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("lseg.data")
        self.session_calls: list[str] = []
        self.history_calls: list[dict] = []
        self.frames: dict[str, _FakeFrame] = {}

    def open_session(self):
        self.session_calls.append("open")

    def close_session(self):
        self.session_calls.append("close")

    def get_history(self, *, universe, fields, interval, start, end, adjustments):
        self.history_calls.append(
            {"universe": universe, "fields": fields, "interval": interval,
             "start": start, "end": end, "adjustments": adjustments}
        )
        return self.frames[universe]


@pytest.fixture
def fake_lseg(monkeypatch: pytest.MonkeyPatch) -> _FakeLseg:
    module = _FakeLseg()
    monkeypatch.setitem(sys.modules, "lseg.data", module)
    return module


_OHLCV = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"]


def test_lseg_provider_maps_rics_and_builds_rows(fake_lseg: _FakeLseg) -> None:
    fake_lseg.frames["AAPL.O"] = _FakeFrame(
        {
            "2026-06-08": {"OPEN_PRC": 100.0, "HIGH_1": 105.0, "LOW_1": 95.0, "TRDPRC_1": 101.5, "ACVOL_UNS": 1000.0},
            # A non-trading/unpriced session must be skipped, not crash.
            "2026-06-09": {"OPEN_PRC": None, "HIGH_1": None, "LOW_1": None, "TRDPRC_1": None, "ACVOL_UNS": None},
            # Missing volume degrades to 0.0 rather than failing.
            "2026-06-10": {"OPEN_PRC": 101.0, "HIGH_1": 106.0, "LOW_1": 96.0, "TRDPRC_1": 102.5, "ACVOL_UNS": None},
        },
        columns=_OHLCV,
    )
    provider = LsegPriceProvider(ric_overrides={"AAPL": "AAPL.O"})

    rows = provider.fetch(["AAPL"], "2026-06-01", "2026-06-15")

    assert [row.session_date for row in rows] == ["2026-06-08", "2026-06-10"]
    assert rows[0] == PriceRow("AAPL", "2026-06-08", 100.0, 105.0, 95.0, 101.5, 1000.0, False)
    assert rows[1].volume == 0.0
    call = fake_lseg.history_calls[0]
    assert call["universe"] == "AAPL.O"  # override applied; PriceRow keeps the plain symbol
    assert call["adjustments"] == ["exchangeCorrection", "manualCorrection", "CCH"]
    assert fake_lseg.session_calls == ["open", "close"]


def test_lseg_provider_unadjusted_passes_no_adjustments(fake_lseg: _FakeLseg) -> None:
    fake_lseg.frames["MSFT"] = _FakeFrame(
        {"2026-06-08": {"OPEN_PRC": 1.0, "HIGH_1": 1.0, "LOW_1": 1.0, "TRDPRC_1": 1.0, "ACVOL_UNS": 1.0}},
        columns=_OHLCV,
    )
    LsegPriceProvider(adjusted=False).fetch(["MSFT"], "2026-06-01", "2026-06-15")
    assert fake_lseg.history_calls[0]["adjustments"] is None
    assert fake_lseg.history_calls[0]["universe"] == "MSFT"  # unmapped symbols pass through


def test_lseg_provider_errors_close_the_session(fake_lseg: _FakeLseg) -> None:
    fake_lseg.frames["AAPL"] = _FakeFrame({}, columns=_OHLCV)  # empty → error
    with pytest.raises(PriceProviderError, match="no LSEG prices"):
        LsegPriceProvider().fetch(["AAPL"], "2026-06-01", "2026-06-15")
    assert fake_lseg.session_calls == ["open", "close"]


def test_lseg_provider_reports_missing_fields(fake_lseg: _FakeLseg) -> None:
    fake_lseg.frames["AAPL"] = _FakeFrame(
        {"2026-06-08": {"OPEN_PRC": 1.0}}, columns=["OPEN_PRC"]
    )
    with pytest.raises(PriceProviderError, match="lacks fields"):
        LsegPriceProvider().fetch(["AAPL"], "2026-06-01", "2026-06-15")
