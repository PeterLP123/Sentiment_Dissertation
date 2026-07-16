"""Build a frozen strategy execution-price panel from a verified LSEG export."""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifact_io import atomic_write_json, atomic_write_text, read_json, sha256_file


class StrategyPriceArtifactError(RuntimeError):
    """Raised when an upstream price export cannot support strategy execution."""


@dataclass(frozen=True)
class StrategyPriceArtifactResult:
    panel_path: Path
    manifest_path: Path
    row_count: int
    symbol_count: int
    session_count: int


def _source_panel_hash(manifest: dict[str, Any], source_path: Path) -> str:
    entry = manifest.get("file")
    if not isinstance(entry, dict) or not entry.get("sha256"):
        raise StrategyPriceArtifactError("source LSEG manifest has no hashed price file")
    declared = str(entry.get("path") or "")
    if declared and Path(declared).name != source_path.name:
        raise StrategyPriceArtifactError(
            f"source LSEG manifest declares {declared!r}, not {source_path.name!r}"
        )
    return str(entry["sha256"])


def _load_source_rows(path: Path) -> list[tuple[str, str, float]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"symbol", "session_date", "open"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise StrategyPriceArtifactError(
                    f"source LSEG panel is missing required columns: {sorted(missing)}"
                )
            rows: list[tuple[str, str, float]] = []
            for line_number, row in enumerate(reader, start=2):
                symbol = row["symbol"].strip().upper()
                session = row["session_date"].strip()
                try:
                    adjusted_open = float(row["open"])
                except (TypeError, ValueError) as exc:
                    raise StrategyPriceArtifactError(
                        f"invalid LSEG open at {path}:{line_number}"
                    ) from exc
                if not symbol or not session or not math.isfinite(adjusted_open) or adjusted_open <= 0:
                    raise StrategyPriceArtifactError(
                        f"invalid LSEG execution-price row at {path}:{line_number}"
                    )
                rows.append((symbol, session, adjusted_open))
    except OSError as exc:
        raise StrategyPriceArtifactError(f"cannot read source LSEG panel {path}: {exc}") from exc
    if not rows:
        raise StrategyPriceArtifactError("source LSEG panel is empty")
    rows.sort(key=lambda value: (value[0], value[1]))
    keys = [(symbol, session) for symbol, session, _ in rows]
    if len(keys) != len(set(keys)):
        raise StrategyPriceArtifactError("source LSEG panel contains duplicate symbol-session rows")
    return rows


def _render_panel(rows: list[tuple[str, str, float]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("symbol", "session_date", "adjusted_open"))
    for symbol, session, value in rows:
        writer.writerow((symbol, session, repr(value)))
    return buffer.getvalue()


def build_strategy_price_artifact(
    source_panel: str | Path,
    source_manifest: str | Path,
    output_panel: str | Path,
    output_manifest: str | Path,
    *,
    expected_symbol_count: int = 33,
    calendar_name: str = "XNYS",
    timezone: str = "America/New_York",
) -> StrategyPriceArtifactResult:
    """Convert a verified split-adjusted LSEG OHLCV export into adjusted opens."""

    source_path = Path(source_panel)
    source_manifest_path = Path(source_manifest)
    panel_path = Path(output_panel)
    manifest_path = Path(output_manifest)
    if panel_path.exists() or manifest_path.exists():
        raise StrategyPriceArtifactError(
            f"refusing to overwrite frozen strategy price artifact: {panel_path}"
        )
    if expected_symbol_count < 1:
        raise StrategyPriceArtifactError("expected_symbol_count must be positive")
    manifest = read_json(source_manifest_path)
    if manifest.get("status") != "completed" or manifest.get("provider") != "lseg":
        raise StrategyPriceArtifactError("source price manifest must be a completed LSEG export")
    convention = str(manifest.get("return_convention") or "").lower()
    if "split-adjusted" not in convention or "dividend" not in convention or "not" not in convention:
        raise StrategyPriceArtifactError(
            "source LSEG manifest must establish split adjustment without dividend back-adjustment"
        )
    expected_hash = _source_panel_hash(manifest, source_path)
    actual_hash = sha256_file(source_path)
    if actual_hash != expected_hash:
        raise StrategyPriceArtifactError(
            f"source LSEG panel hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    rows = _load_source_rows(source_path)
    symbols = tuple(sorted({symbol for symbol, _, _ in rows}))
    if len(symbols) != expected_symbol_count:
        raise StrategyPriceArtifactError(
            f"strategy panel requires {expected_symbol_count} symbols; found {len(symbols)}"
        )
    declared_symbols = manifest.get("symbols")
    if not isinstance(declared_symbols, list):
        raise StrategyPriceArtifactError("source LSEG manifest has no symbols list")
    if tuple(sorted(str(value).strip().upper() for value in declared_symbols)) != symbols:
        raise StrategyPriceArtifactError("source LSEG manifest symbols do not match its price panel")

    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover - locked project dependency
        raise StrategyPriceArtifactError("exchange-calendars is required") from exc
    sessions = tuple(sorted({session for _, session, _ in rows}))
    calendar = xcals.get_calendar(calendar_name)
    expected_sessions = tuple(
        str(value.date()) for value in calendar.sessions_in_range(sessions[0], sessions[-1])
    )
    if sessions != expected_sessions:
        missing = sorted(set(expected_sessions) - set(sessions))
        extra = sorted(set(sessions) - set(expected_sessions))
        raise StrategyPriceArtifactError(
            f"source price session spine is incomplete; missing={missing[:5]}, extra={extra[:5]}"
        )
    observed = {(symbol, session) for symbol, session, _ in rows}
    missing_cells = [
        (symbol, session)
        for symbol in symbols
        for session in expected_sessions
        if (symbol, session) not in observed
    ]
    if missing_cells:
        preview = ", ".join(f"{symbol}@{session}" for symbol, session in missing_cells[:5])
        raise StrategyPriceArtifactError(
            f"source price panel is not a complete symbol-session grid: {preview}"
        )

    atomic_write_text(panel_path, _render_panel(rows))
    counts = Counter(symbol for symbol, _, _ in rows)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "source": "LSEG Workspace historical pricing",
            "calendar": calendar_name,
            "timezone": timezone,
            "execution_field": "adjusted_open",
            "return_convention": "open_to_open",
            "adjustment_supported": True,
            "split_adjusted": True,
            "dividend_adjusted": False,
            "adjustment_convention": "split-adjusted open price returns; dividends not back-adjusted",
            "adjustment_provenance": (
                "Verified LSEG export produced with exchangeCorrection, manualCorrection, and CCH "
                "capital-change adjustments; the LSEG provider does not back-adjust dividends."
            ),
            "symbols": list(symbols),
            "symbol_count": len(symbols),
            "session_count": len(expected_sessions),
            "row_count": len(rows),
            "first_session": expected_sessions[0],
            "last_session": expected_sessions[-1],
            "rows_by_symbol": dict(sorted(counts.items())),
            "rics": manifest.get("rics"),
            "requested_start": manifest.get("requested_start"),
            "requested_end": manifest.get("requested_end"),
            "source_export": {
                "panel_path": source_path.as_posix(),
                "panel_sha256": actual_hash,
                "manifest_path": source_manifest_path.as_posix(),
                "manifest_sha256": sha256_file(source_manifest_path),
            },
            "files": {
                "price_panel": {
                    "path": panel_path.name,
                    "sha256": sha256_file(panel_path),
                    "size_bytes": panel_path.stat().st_size,
                }
            },
        },
    )
    return StrategyPriceArtifactResult(
        panel_path=panel_path,
        manifest_path=manifest_path,
        row_count=len(rows),
        symbol_count=len(symbols),
        session_count=len(expected_sessions),
    )

