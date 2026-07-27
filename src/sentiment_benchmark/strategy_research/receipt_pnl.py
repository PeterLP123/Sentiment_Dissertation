"""Receipt-date P/L exports for portfolio-combination work.

The strategy ledgers store an open-to-open return interval against its entry
session. This module creates a reporting view in which gross trading P/L is
booked on the interval endpoint. The source ledger is never modified.
"""

from __future__ import annotations

import csv
import io
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from ..artifact_io import atomic_write_json, atomic_write_text, read_json, read_jsonl, sha256_file


class ReceiptPnlError(RuntimeError):
    """Raised when a ledger cannot be aligned without breaking its accounting."""


@dataclass(frozen=True)
class ReceiptPnlRow:
    date: str
    day: int
    scorer: str
    gross_pnl_received_usd: float
    cumulative_gross_pnl_usd: float
    active_trade_exits: int
    zero_filled: bool


@dataclass(frozen=True)
class ReceiptPnlPerformance:
    scorer: str
    observations: int
    nonzero_receipt_days: int
    zero_receipt_days: int
    total_gross_pnl_received_usd: float
    mean_daily_gross_pnl_received_usd: float
    daily_gross_pnl_received_std_usd: float
    receipt_pnl_annualized_sharpe: float | None
    start_date: str
    end_date: str


@dataclass(frozen=True)
class ReceiptPnlReportResult:
    output_dir: Path
    daily_long_path: Path
    portfolio_matrix_path: Path
    performance_path: Path
    figure_path: Path
    summary_path: Path
    manifest_path: Path
    rows: tuple[ReceiptPnlRow, ...]
    performance: tuple[ReceiptPnlPerformance, ...]


@dataclass
class _DailyComponents:
    gross_received: float = 0.0
    active_trade_exits: int = 0


_REQUIRED_FIELDS = {
    "scorer",
    "session",
    "next_session",
    "start_nav_usd",
    "gross_return",
    "transaction_cost",
    "net_return",
    "end_nav_usd",
    "active_names",
    "final_liquidation",
}


def _iso_date(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ReceiptPnlError(f"{field} must be an ISO date, got {value!r}") from exc
    return parsed.isoformat()


def _finite_float(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReceiptPnlError(f"ledger row requires numeric {field}") from exc
    if not math.isfinite(value):
        raise ReceiptPnlError(f"ledger row has non-finite {field}")
    return value


def _validate_source_rows(rows: Sequence[Mapping[str, Any]], scorers: Sequence[str]) -> None:
    if not rows:
        raise ReceiptPnlError("daily P/L input is empty")
    available = {str(row.get("scorer") or "").strip() for row in rows}
    missing_scorers = sorted(set(scorers) - available)
    if missing_scorers:
        raise ReceiptPnlError(f"requested scorers are absent from daily P/L: {missing_scorers}")
    for scorer in scorers:
        scorer_rows = [row for row in rows if str(row.get("scorer") or "").strip() == scorer]
        seen_sessions: set[str] = set()
        previous_session: str | None = None
        final_liquidations = 0
        for row_number, row in enumerate(scorer_rows, start=1):
            missing = sorted(_REQUIRED_FIELDS - set(row))
            if missing:
                raise ReceiptPnlError(f"{scorer} row {row_number} is missing fields: {missing}")
            session = _iso_date(row["session"], field="session")
            if session in seen_sessions:
                raise ReceiptPnlError(f"duplicate {scorer} ledger session: {session}")
            if previous_session is not None and session <= previous_session:
                raise ReceiptPnlError(f"{scorer} ledger sessions are not strictly chronological")
            seen_sessions.add(session)
            previous_session = session
            next_session_raw = row.get("next_session")
            if next_session_raw is not None:
                next_session = _iso_date(next_session_raw, field="next_session")
                if next_session <= session:
                    raise ReceiptPnlError(f"{scorer} next_session must follow session on {session}")
            start_nav = _finite_float(row, "start_nav_usd")
            end_nav = _finite_float(row, "end_nav_usd")
            gross_return = _finite_float(row, "gross_return")
            transaction_cost = _finite_float(row, "transaction_cost")
            net_return = _finite_float(row, "net_return")
            if start_nav <= 0 or end_nav <= 0:
                raise ReceiptPnlError(f"{scorer} NAV must be positive on {session}")
            if transaction_cost < 0:
                raise ReceiptPnlError(f"{scorer} transaction cost cannot be negative on {session}")
            expected_net_pnl = start_nav * (gross_return - transaction_cost)
            stored_net_pnl = start_nav * net_return
            if not math.isclose(expected_net_pnl, stored_net_pnl, rel_tol=1e-10, abs_tol=1e-7):
                raise ReceiptPnlError(f"{scorer} gross return, cost, and net return do not reconcile on {session}")
            if not math.isclose(end_nav - start_nav, stored_net_pnl, rel_tol=1e-10, abs_tol=1e-6):
                raise ReceiptPnlError(f"{scorer} NAV change and net P/L do not reconcile on {session}")
            if bool(row["final_liquidation"]):
                final_liquidations += 1
                if next_session_raw is not None:
                    raise ReceiptPnlError(f"{scorer} final-liquidation row must not have next_session")
        if final_liquidations > 1:
            raise ReceiptPnlError(f"{scorer} ledger contains multiple final-liquidation rows")


def align_pnl_to_receipt_dates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scorers: Sequence[str],
) -> tuple[ReceiptPnlRow, ...]:
    """Build one zero-filled trading-day P/L series per scorer.

    Gross P/L for ``session -> next_session`` is received on ``next_session``.
    The common date spine is the union of all source sessions and interval
    endpoints, which keeps every model aligned for later portfolio combination.
    """

    scorer_tuple = tuple(dict.fromkeys(str(value).strip() for value in scorers if str(value).strip()))
    if not scorer_tuple:
        raise ReceiptPnlError("at least one scorer is required")
    _validate_source_rows(rows, scorer_tuple)

    dates: set[str] = set()
    components: dict[tuple[str, str], _DailyComponents] = defaultdict(_DailyComponents)
    source_gross_pnl: dict[str, float] = defaultdict(float)
    for raw in rows:
        scorer = str(raw.get("scorer") or "").strip()
        if scorer not in scorer_tuple:
            continue
        session = _iso_date(raw["session"], field="session")
        next_session = None if raw.get("next_session") is None else _iso_date(raw["next_session"], field="next_session")
        dates.add(session)
        if next_session is not None:
            dates.add(next_session)
        start_nav = _finite_float(raw, "start_nav_usd")
        gross_pnl = start_nav * _finite_float(raw, "gross_return")
        source_gross_pnl[scorer] += gross_pnl

        receipt_date = next_session or session
        receipt = components[(scorer, receipt_date)]
        receipt.gross_received += gross_pnl
        if int(raw["active_names"]) > 0 and next_session is not None and not bool(raw["final_liquidation"]):
            receipt.active_trade_exits += 1

    ordered_dates = sorted(dates)
    result: list[ReceiptPnlRow] = []
    for scorer in scorer_tuple:
        cumulative = 0.0
        scorer_rows: list[ReceiptPnlRow] = []
        for day_number, session_date in enumerate(ordered_dates, start=1):
            value = components[(scorer, session_date)]
            cumulative += value.gross_received
            scorer_rows.append(
                ReceiptPnlRow(
                    date=session_date,
                    day=day_number,
                    scorer=scorer,
                    gross_pnl_received_usd=value.gross_received,
                    cumulative_gross_pnl_usd=cumulative,
                    active_trade_exits=value.active_trade_exits,
                    zero_filled=math.isclose(value.gross_received, 0.0, abs_tol=1e-12),
                )
            )
        if not math.isclose(cumulative, source_gross_pnl[scorer], rel_tol=1e-10, abs_tol=1e-6):
            raise ReceiptPnlError(f"receipt-aligned {scorer} gross P/L does not reconcile to the source ledger")
        result.extend(scorer_rows)
    return tuple(result)


def calculate_receipt_pnl_performance(rows: Sequence[ReceiptPnlRow]) -> tuple[ReceiptPnlPerformance, ...]:
    grouped: dict[str, list[ReceiptPnlRow]] = defaultdict(list)
    for row in rows:
        grouped[row.scorer].append(row)
    performance: list[ReceiptPnlPerformance] = []
    for scorer, scorer_rows in sorted(grouped.items()):
        ordered = sorted(scorer_rows, key=lambda row: row.date)
        gross_values = [row.gross_pnl_received_usd for row in ordered]
        if not gross_values:
            continue
        gross_std = stdev(gross_values) if len(gross_values) >= 2 else 0.0
        gross_sharpe = math.sqrt(252) * fmean(gross_values) / gross_std if gross_std > 0 else None
        nonzero = sum(not row.zero_filled for row in ordered)
        performance.append(
            ReceiptPnlPerformance(
                scorer=scorer,
                observations=len(ordered),
                nonzero_receipt_days=nonzero,
                zero_receipt_days=len(ordered) - nonzero,
                total_gross_pnl_received_usd=sum(row.gross_pnl_received_usd for row in ordered),
                mean_daily_gross_pnl_received_usd=fmean(gross_values),
                daily_gross_pnl_received_std_usd=gross_std,
                receipt_pnl_annualized_sharpe=gross_sharpe,
                start_date=ordered[0].date,
                end_date=ordered[-1].date,
            )
        )
    return tuple(performance)


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _float_text(value: float | None) -> str:
    return "" if value is None else f"{value:.12f}"


def _long_csv(rows: Sequence[ReceiptPnlRow]) -> str:
    fieldnames = list(ReceiptPnlRow.__dataclass_fields__)
    payloads: list[dict[str, Any]] = []
    for row in rows:
        payload = asdict(row)
        for field in (
            "gross_pnl_received_usd",
            "cumulative_gross_pnl_usd",
        ):
            payload[field] = _float_text(float(payload[field]))
        payloads.append(payload)
    return _csv_text(payloads, fieldnames)


def _matrix_csv(rows: Sequence[ReceiptPnlRow], scorers: Sequence[str]) -> str:
    by_key = {(row.date, row.scorer): row for row in rows}
    dates = sorted({row.date for row in rows})
    fields = ["date", *(f"{scorer}_pnl_usd" for scorer in scorers), "total_pnl_usd", "day"]
    fields.extend(f"cumulative_{scorer}_pnl_usd" for scorer in scorers)
    fields.append("cumulative_total_pnl_usd")
    cumulative_by_scorer = dict.fromkeys(scorers, 0.0)
    cumulative_total = 0.0
    payloads: list[dict[str, Any]] = []
    for day_number, session_date in enumerate(dates, start=1):
        daily_values = [by_key[(session_date, scorer)].gross_pnl_received_usd for scorer in scorers]
        daily_total = sum(daily_values)
        cumulative_total += daily_total
        payload: dict[str, Any] = {"date": session_date}
        for scorer, value in zip(scorers, daily_values, strict=True):
            cumulative_by_scorer[scorer] += value
            payload[f"{scorer}_pnl_usd"] = _float_text(value)
        payload["total_pnl_usd"] = _float_text(daily_total)
        payload["day"] = day_number
        payload.update(
            {
                f"cumulative_{scorer}_pnl_usd": _float_text(cumulative_by_scorer[scorer])
                for scorer in scorers
            }
        )
        payload["cumulative_total_pnl_usd"] = _float_text(cumulative_total)
        payloads.append(payload)
    return _csv_text(payloads, fields)


def _performance_csv(rows: Sequence[ReceiptPnlPerformance]) -> str:
    fields = list(ReceiptPnlPerformance.__dataclass_fields__)
    payloads: list[dict[str, Any]] = []
    for row in rows:
        payload = asdict(row)
        for field in (
            "total_gross_pnl_received_usd",
            "mean_daily_gross_pnl_received_usd",
            "daily_gross_pnl_received_std_usd",
            "receipt_pnl_annualized_sharpe",
        ):
            value = payload[field]
            payload[field] = _float_text(None if value is None else float(value))
        payloads.append(payload)
    return _csv_text(payloads, fields)


def _render_figure(rows: Sequence[ReceiptPnlRow], scorers: Sequence[str], path: Path, source_run_id: str) -> None:
    matplotlib_cache = Path(tempfile.gettempdir()) / "sentiment-dissertation-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache.as_posix())
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ReceiptPnlError("matplotlib is required to render the Week 7 cumulative P/L figure") from exc

    by_key = {(row.date, row.scorer): row for row in rows}
    dates = sorted({row.date for row in rows})
    parsed_dates = [date.fromisoformat(value) for value in dates]
    numeric_dates = mdates.date2num(parsed_dates)
    cumulative_total: list[float] = []
    running = 0.0
    for session_date in dates:
        running += sum(by_key[(session_date, scorer)].gross_pnl_received_usd for scorer in scorers)
        cumulative_total.append(running)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axis = plt.subplots(figsize=(11, 6.2))
    axis.plot(numeric_dates, cumulative_total, color="#2F6BBD", linewidth=2.2, marker="o", markersize=2.3, markevery=10)
    axis.axhline(0.0, color="#4A4A4A", linewidth=0.9)
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    fig.suptitle(
        "Cumulative P/L by receipt date",
        x=0.095,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#222222",
    )
    fig.text(
        0.095,
        0.935,
        "Gross open-to-open P/L booked on interval exit; inactive trading dates are zero",
        fontsize=9.5,
        color="#5F6670",
    )
    axis.set_xlabel("Receipt date")
    axis.set_ylabel("Cumulative P/L (USD)")
    axis.yaxis.set_major_formatter(lambda value, _position: f"${value:,.0f}")
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.annotate(
        f"${cumulative_total[-1]:,.0f}",
        xy=(numeric_dates[-1], cumulative_total[-1]),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        color="#244C83",
        fontweight="bold",
    )
    fig.text(0.01, 0.01, f"Source: {source_run_id}", fontsize=8, color="#6A7078")
    fig.tight_layout(rect=(0, 0.035, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def _summary(
    *,
    source_run_id: str,
    source_daily_path: Path,
    source_daily_sha256: str,
    scorers: Sequence[str],
    performance: Sequence[ReceiptPnlPerformance],
) -> str:
    lines = [
        "# Week 7 receipt-aligned P/L",
        "",
        f"- Source run: `{source_run_id}`",
        f"- Source daily ledger: `{source_daily_path.as_posix()}`",
        f"- Source SHA-256: `{source_daily_sha256}`",
        f"- Models: `{', '.join(scorers)}`",
        "- P/L: gross open-to-open P/L is booked on `next_session`, matching `Price(exit) - Price(entry)`.",
        "- Calendar: every trading date in the source session/endpoint union is present; inactive slots are zero.",
        "- Sharpe: annualised as `sqrt(252) * mean(daily P/L) / sample standard deviation`, with zero receipt slots included.",
        "",
        "## Results",
        "",
        "| Model | Dates | P/L receipt days | Zero slots | Gross P/L | Receipt P/L Sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in performance:
        receipt_sharpe = "n/a" if row.receipt_pnl_annualized_sharpe is None else f"{row.receipt_pnl_annualized_sharpe:.3f}"
        lines.append(
            f"| {row.scorer} | {row.observations} | {row.nonzero_receipt_days} | {row.zero_receipt_days} | "
            f"${row.total_gross_pnl_received_usd:,.2f} | {receipt_sharpe} |"
        )
    lines.extend(
        [
            "",
            "The cumulative figure is an arithmetic sum of gross receipt-date P/L in USD. It is not the source report's compounded return, "
            "and the source ledger's development/evaluation NAV reset remains unchanged.",
            "",
            "## Files",
            "",
            "- `daily_pnl_receipt_aligned.csv`: long, audit-friendly model/date table.",
            "- `portfolio_pnl_matrix.csv`: requested wide gross P/L matrix for later portfolio optimisation.",
            "- `performance.csv`: gross P/L total and receipt-date Sharpe.",
            "- `cumulative_pnl_by_receipt_date.png`: cumulative receipt-date gross P/L plot.",
            "- `manifest.json`: input/output identities and accounting contract.",
            "",
        ]
    )
    return "\n".join(lines)


def build_receipt_pnl_report(
    daily_pnl_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    scorers: Sequence[str],
) -> ReceiptPnlReportResult:
    """Verify a frozen strategy ledger and write a separate Week 7 report."""

    daily_path = Path(daily_pnl_path)
    manifest_path = Path(source_manifest_path)
    destination = Path(output_dir)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ReceiptPnlError("source strategy manifest is not completed")
    source_output = (manifest.get("outputs") or {}).get("daily_pnl")
    if not isinstance(source_output, dict) or not source_output.get("sha256"):
        raise ReceiptPnlError("source strategy manifest does not identify daily_pnl")
    source_sha = sha256_file(daily_path)
    if source_sha != source_output["sha256"]:
        raise ReceiptPnlError("source daily_pnl SHA-256 does not match its immutable manifest")
    source_run_id = str(manifest.get("run_id") or "").strip()
    if not source_run_id:
        raise ReceiptPnlError("source strategy manifest has no run_id")

    raw_rows = read_jsonl(daily_path)
    scorer_tuple = tuple(dict.fromkeys(str(value).strip() for value in scorers if str(value).strip()))
    aligned = align_pnl_to_receipt_dates(raw_rows, scorers=scorer_tuple)
    performance = calculate_receipt_pnl_performance(aligned)

    long_path = destination / "daily_pnl_receipt_aligned.csv"
    matrix_path = destination / "portfolio_pnl_matrix.csv"
    performance_path = destination / "performance.csv"
    figure_path = destination / "cumulative_pnl_by_receipt_date.png"
    summary_path = destination / "summary.md"
    output_manifest_path = destination / "manifest.json"
    atomic_write_text(long_path, _long_csv(aligned))
    atomic_write_text(matrix_path, _matrix_csv(aligned, scorer_tuple))
    atomic_write_text(performance_path, _performance_csv(performance))
    _render_figure(aligned, scorer_tuple, figure_path, source_run_id)
    atomic_write_text(
        summary_path,
        _summary(
            source_run_id=source_run_id,
            source_daily_path=daily_path,
            source_daily_sha256=source_sha,
            scorers=scorer_tuple,
            performance=performance,
        ),
    )
    output_files = {
        "daily_pnl_receipt_aligned": long_path,
        "portfolio_pnl_matrix": matrix_path,
        "performance": performance_path,
        "figure": figure_path,
        "summary": summary_path,
    }
    atomic_write_json(
        output_manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "source": {
                "run_id": source_run_id,
                "daily_pnl_path": daily_path.as_posix(),
                "daily_pnl_sha256": source_sha,
                "manifest_path": manifest_path.as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
            },
            "method": {
                "gross_pnl_date": "next_session",
                "date_spine": "union_of_source_sessions_and_interval_endpoints",
                "missing_slots": "zero",
                "pnl": "gross open-to-open P/L matching Price(exit) - Price(entry)",
                "cumulative_pnl": "arithmetic_sum_of_gross_receipt_pnl_usd",
                "sharpe": "sqrt(252) * mean(daily_gross_receipt_pnl_usd) / sample_std; zero receipt slots included",
            },
            "scorers": list(scorer_tuple),
            "row_counts": {
                "source": len(raw_rows),
                "receipt_aligned": len(aligned),
                "dates": len({row.date for row in aligned}),
            },
            "performance": [asdict(row) for row in performance],
            "outputs": {
                name: {"path": path.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for name, path in sorted(output_files.items())
            },
        },
    )
    return ReceiptPnlReportResult(
        output_dir=destination,
        daily_long_path=long_path,
        portfolio_matrix_path=matrix_path,
        performance_path=performance_path,
        figure_path=figure_path,
        summary_path=summary_path,
        manifest_path=output_manifest_path,
        rows=aligned,
        performance=performance,
    )
