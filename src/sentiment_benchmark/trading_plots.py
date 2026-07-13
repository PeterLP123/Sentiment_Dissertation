"""Trade- and parameter-level visualisation plus contamination sensitivity tables.

Two kinds of output, kept in one leaf module (imports only other leaves, so no
cycle with ``trading_strategy``):

* **Sensitivity summaries** (pure, testable): masked-vs-unmasked and pre-/post-
  knowledge-cutoff breakdowns of event returns. These are the dissertation
  payoff — layered *on top* of the frozen primary, never replacing it.
* **Figures** (guarded by matplotlib): equity curve and the parameter-sweep
  heatmap. If matplotlib is absent the plot functions return ``None``, mirroring
  the optional-dependency pattern in ``plotting``/``write_charts``.
"""

from __future__ import annotations

import csv
import importlib.util
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .backtest import EquityPoint, ReturnRow
from .model_roster import is_post_cutoff, knowledge_cutoff, roster_cutoff
from .portfolio import PortfolioBacktestResult
from .strategy_sweep import SweepResult

_MASKED_SUFFIX = "#masked"


def matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def _pyplot() -> Any | None:
    if not matplotlib_available():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _safe_name(scorer_id: str) -> str:
    return scorer_id.replace("/", "_").replace("#", "_").replace(" ", "_")


def _base_scorer(scorer_id: str) -> str:
    return scorer_id[: -len(_MASKED_SUFFIX)] if scorer_id.endswith(_MASKED_SUFFIX) else scorer_id


def _hit_rate(values: Sequence[float]) -> float:
    return sum(1 for value in values if value > 0) / len(values) if values else 0.0


@dataclass(frozen=True)
class SensitivityRow:
    scorer_id: str
    horizon: int
    group: str
    n: int
    mean_net_return_pct: float
    hit_rate: float


def _rows_from_groups(groups: dict[tuple[str, int, str], list[float]]) -> list[SensitivityRow]:
    return [
        SensitivityRow(
            scorer_id=scorer_id,
            horizon=horizon,
            group=group,
            n=len(values),
            mean_net_return_pct=fmean(values),
            hit_rate=_hit_rate(values),
        )
        for (scorer_id, horizon, group), values in sorted(groups.items())
    ]


def summarize_masking(returns: list[ReturnRow]) -> list[SensitivityRow]:
    """Mean net return / hit-rate split by masked vs unmasked, per base scorer and
    horizon. Empty unless a masked arm (``#masked`` scorer ids) is present."""
    if not any(row.scorer_id.endswith(_MASKED_SUFFIX) for row in returns):
        return []
    groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in returns:
        group = "masked" if row.scorer_id.endswith(_MASKED_SUFFIX) else "unmasked"
        groups[(_base_scorer(row.scorer_id), row.horizon, group)].append(row.net_strategy_return_pct)
    return _rows_from_groups(groups)


def summarize_cutoff(
    returns: list[ReturnRow],
    models: Sequence[str],
    overrides: Mapping[str, str] | None = None,
) -> list[SensitivityRow]:
    """Mean net return / hit-rate split by post- vs pre-knowledge-cutoff (and
    ``cutoff_na`` for baselines), per scorer and horizon. Consensus uses the
    most-conservative roster cutoff; computed from each return's news date."""
    consensus_cut = roster_cutoff(models, overrides)
    groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in returns:
        base = _base_scorer(row.scorer_id)
        if base.startswith("consensus"):
            cutoff = consensus_cut
        elif base.startswith("baseline/"):
            cutoff = None
        else:
            cutoff = knowledge_cutoff(base, overrides)
        try:
            event = date.fromisoformat(row.news_date)
        except ValueError:
            event = None
        post = is_post_cutoff(event, cutoff)
        group = "post_cutoff" if post is True else "pre_cutoff" if post is False else "cutoff_na"
        groups[(row.scorer_id, row.horizon, group)].append(row.net_strategy_return_pct)
    return _rows_from_groups(groups)


def _write_sensitivity_csv(rows: list[SensitivityRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scorer_id", "horizon", "group", "n", "mean_net_return_pct", "hit_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def write_sensitivity_csvs(
    results_dir: Path,
    returns: list[ReturnRow],
    models: Sequence[str],
    overrides: Mapping[str, str] | None = None,
) -> list[Path]:
    """Write sensitivity_cutoff.csv (always) and sensitivity_masking.csv (only if a
    masked arm exists). Returns the files written."""
    written: list[Path] = []
    cutoff_rows = summarize_cutoff(returns, models, overrides)
    if cutoff_rows:
        path = results_dir / "sensitivity_cutoff.csv"
        _write_sensitivity_csv(cutoff_rows, path)
        written.append(path)
    masking_rows = summarize_masking(returns)
    if masking_rows:
        path = results_dir / "sensitivity_masking.csv"
        _write_sensitivity_csv(masking_rows, path)
        written.append(path)
    return written


def sweep_heatmap_matrix(
    results: list[SweepResult],
    attr: str = "test_metric",
) -> tuple[list[float], list[int], list[list[float | None]]]:
    """Pivot sweep rows into (thresholds, horizons, matrix[threshold][horizon])."""
    thresholds = sorted({result.threshold for result in results})
    horizons = sorted({result.horizon for result in results})
    index = {(result.threshold, result.horizon): getattr(result, attr) for result in results}
    matrix = [[index.get((threshold, horizon)) for horizon in horizons] for threshold in thresholds]
    return thresholds, horizons, matrix


def plot_equity_curve(points: list[EquityPoint], path: Path, *, title: str = "Equity curve") -> Path | None:
    plt = _pyplot()
    if plt is None or not points:
        return None
    fig, axis = plt.subplots(figsize=(9, 4.5))
    positions = range(len(points))
    axis.plot(positions, [point.cumulative_pnl_usd for point in points], marker="o", color="#2563eb")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(list(positions))
    axis.set_xticklabels([point.date for point in points], rotation=45, ha="right", fontsize=7)
    axis.set(title=title, ylabel="Cumulative net P&L (USD)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_portfolio_charts(result: PortfolioBacktestResult, results_dir: Path) -> list[Path]:
    """Render funded equity, best-case stock contributions, and correlation.

    The equity figure includes every scorer/horizon account.  The two detailed
    figures use the finite-Sharpe leader (falling back to total return), making
    the selection visible without pretending that it is out-of-sample evidence.
    """

    plt = _pyplot()
    if plt is None or not result.daily_portfolio:
        return []
    results_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    books: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for row in result.daily_portfolio:
        books[(row.scorer_id, row.horizon)].append(row)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for (scorer_id, horizon), rows in sorted(books.items()):
        ordered = sorted(rows, key=lambda row: row.date)
        sessions = [date.fromisoformat(row.date) for row in ordered]
        axis.plot(
            sessions,
            [row.end_nav_usd for row in ordered],
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label=f"{scorer_id} H{horizon}",
        )
    axis.set(title="Funded portfolio equity curves", ylabel="End-of-day NAV (USD)", xlabel="Session")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=6, ncol=max(1, min(4, len(books) // 5 + 1)))
    all_sessions = sorted({date.fromisoformat(row.date) for row in result.daily_portfolio})
    if len(all_sessions) == 1:
        axis.set_xlim(all_sessions[0] - timedelta(days=1), all_sessions[0] + timedelta(days=1))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = results_dir / "portfolio_equity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    eligible = [row for row in result.summaries if row.trade_count > 0]
    if not eligible:
        return written
    best = max(
        eligible,
        key=lambda row: (
            row.annualized_sharpe is not None,
            row.annualized_sharpe if row.annualized_sharpe is not None else row.total_return,
        ),
    )
    best_stock_rows = [
        row
        for row in result.daily_stock_pnl
        if row.scorer_id == best.scorer_id and row.horizon == best.horizon
    ]
    symbols = sorted({row.symbol for row in best_stock_rows})
    if symbols:
        fig, axis = plt.subplots(figsize=(11, 5.5))
        for symbol in symbols:
            rows = sorted((row for row in best_stock_rows if row.symbol == symbol), key=lambda row: row.date)
            sessions = [date.fromisoformat(row.date) for row in rows]
            axis.plot(
                sessions,
                [row.cumulative_net_pnl_usd for row in rows],
                linewidth=1.0,
                marker="o",
                markersize=2.5,
                label=symbol,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            title=f"Stock cumulative P&L — {best.scorer_id} H{best.horizon}",
            ylabel="Cumulative net contribution (USD)",
            xlabel="Session",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6, ncol=max(1, min(5, len(symbols) // 7 + 1)))
        stock_sessions = sorted({date.fromisoformat(row.date) for row in best_stock_rows})
        if len(stock_sessions) == 1:
            axis.set_xlim(stock_sessions[0] - timedelta(days=1), stock_sessions[0] + timedelta(days=1))
        fig.autofmt_xdate()
        fig.tight_layout()
        path = results_dir / "stock_equity_curves.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

        index = {symbol: offset for offset, symbol in enumerate(symbols)}
        matrix = np.full((len(symbols), len(symbols)), np.nan, dtype=float)
        np.fill_diagonal(matrix, 1.0)
        for correlation_row in result.correlations:
            if (
                correlation_row.scorer_id != best.scorer_id
                or correlation_row.horizon != best.horizon
                or correlation_row.correlation is None
            ):
                continue
            if correlation_row.symbol_a in index and correlation_row.symbol_b in index:
                left, right = index[correlation_row.symbol_a], index[correlation_row.symbol_b]
                matrix[left, right] = matrix[right, left] = correlation_row.correlation
        fig, axis = plt.subplots(figsize=(max(6, len(symbols) * 0.45), max(5, len(symbols) * 0.4)))
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(range(len(symbols)), symbols, rotation=90, fontsize=7)
        axis.set_yticks(range(len(symbols)), symbols, fontsize=7)
        axis.set(title=f"Daily stock-contribution correlation — {best.scorer_id} H{best.horizon}")
        fig.colorbar(image, ax=axis, label="Correlation")
        fig.tight_layout()
        path = results_dir / "pnl_correlation_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def plot_sweep_heatmap(
    results: list[SweepResult],
    path: Path,
    *,
    attr: str = "test_metric",
    title: str = "Parameter sweep",
) -> Path | None:
    plt = _pyplot()
    if plt is None or not results:
        return None
    thresholds, horizons, matrix = sweep_heatmap_matrix(results, attr)
    data = [[math.nan if value is None else value for value in row] for row in matrix]
    fig, axis = plt.subplots(figsize=(1.4 * len(horizons) + 2, 0.6 * len(thresholds) + 2))
    image = axis.imshow(data, aspect="auto", cmap="RdYlGn")
    axis.set_xticks(range(len(horizons)), [str(horizon) for horizon in horizons])
    axis.set_yticks(range(len(thresholds)), [f"{threshold:g}" for threshold in thresholds])
    axis.set(title=f"{title} ({attr})", xlabel="Horizon", ylabel="Threshold")
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if not math.isnan(value):
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, label=attr)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
