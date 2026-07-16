"""Reproducible LSEG price export for headline-value experiments."""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .artifact_io import atomic_write_json, atomic_write_text, sha256_file
from .lseg_source import LsegCollectionConfig
from .prices import LsegPriceProvider, PriceProvider, PriceProviderError, PriceRow
from .runtime_metadata import collect_run_environment


class PriceExportError(RuntimeError):
    """Raised when a price export cannot be produced safely."""


@dataclass(frozen=True)
class PriceExportResult:
    output_path: Path
    manifest_path: Path
    row_count: int
    symbol_count: int


_FIELDS = tuple(PriceRow.__dataclass_fields__)


def _validate_dates(start: str, end: str) -> None:
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise PriceExportError("price dates must use YYYY-MM-DD") from exc
    if end_date <= start_date:
        raise PriceExportError("price end must be after start")


def _price_csv(rows: Sequence[PriceRow]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    return buffer.getvalue()


def export_lseg_prices(
    config: LsegCollectionConfig,
    *,
    start: str,
    end: str,
    output: str | Path,
    overwrite: bool = False,
    provider: PriceProvider | None = None,
    ric_overrides: Mapping[str, str] | None = None,
) -> PriceExportResult:
    """Fetch one deterministic OHLCV panel and write a hash manifest beside it."""

    _validate_dates(start, end)
    output_path = Path(output)
    manifest_path = output_path.with_suffix(".manifest.json")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise PriceExportError(f"refusing to overwrite existing price export: {output_path}")

    symbols = tuple(company.symbol for company in config.companies)
    resolved_rics = {company.symbol: company.ric for company in config.companies}
    for symbol, ric in (ric_overrides or {}).items():
        normalized_symbol = symbol.strip().upper()
        normalized_ric = ric.strip()
        if normalized_symbol not in resolved_rics:
            raise PriceExportError(f"price RIC override references unknown symbol: {normalized_symbol}")
        if not normalized_ric:
            raise PriceExportError(f"price RIC override for {normalized_symbol} is blank")
        resolved_rics[normalized_symbol] = normalized_ric
    active_provider = provider or LsegPriceProvider(ric_overrides=resolved_rics)
    try:
        rows = active_provider.fetch(symbols, start, end)
    except PriceProviderError as exc:
        raise PriceExportError(str(exc)) from exc

    counts = Counter(row.symbol for row in rows)
    missing = [symbol for symbol in symbols if counts[symbol] == 0]
    if missing:
        raise PriceExportError(f"LSEG price export has no rows for: {', '.join(missing)}")

    dates_by_symbol: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        dates_by_symbol[row.symbol].append(row.session_date)
    atomic_write_text(output_path, _price_csv(rows))
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "provider": "lseg",
            "return_convention": "split-adjusted price returns; dividends not back-adjusted",
            "collection_config_sha256": config.config_sha256,
            "requested_start": start,
            "requested_end": end,
            "symbols": list(symbols),
            "rics": resolved_rics,
            "ric_overrides": dict(sorted((ric_overrides or {}).items())),
            "counts": {
                "rows": len(rows),
                "symbols": len(symbols),
                "rows_by_symbol": dict(sorted(counts.items())),
            },
            "coverage": {
                symbol: {
                    "first_session": min(dates_by_symbol[symbol]),
                    "last_session": max(dates_by_symbol[symbol]),
                }
                for symbol in symbols
            },
            "file": {"path": output_path.name, "sha256": sha256_file(output_path)},
            "environment": collect_run_environment(),
        },
    )
    return PriceExportResult(output_path, manifest_path, len(rows), len(symbols))
