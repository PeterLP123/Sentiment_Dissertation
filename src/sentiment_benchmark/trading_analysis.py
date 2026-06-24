from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .trading_effectiveness import evaluate_effectiveness, format_effectiveness_markdown


class TradingAnalysisError(RuntimeError):
    """Raised when trading-run analysis inputs or outputs are invalid."""


@dataclass(frozen=True)
class MeanInterval:
    mean: float
    lower: float
    upper: float
    n: int


@dataclass(frozen=True)
class TradingAnalysisResult:
    output_dir: Path
    summary_path: Path
    manifest_path: Path
    generated_files: tuple[Path, ...]


SCORER_DISPLAY = {
    "openai/gpt-4o-mini": "GPT-4o mini",
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "consensus/majority": "LLM consensus",
    "baseline/vader": "VADER",
    "baseline/finbert": "FinBERT",
}


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int = 42,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> MeanInterval:
    """Return a deterministic percentile bootstrap interval for an arithmetic mean."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("at least one finite value is required")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if array.size == 1:
        value = float(array[0])
        return MeanInterval(value, value, value, 1)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(means, [alpha, 1 - alpha])
    return MeanInterval(float(array.mean()), float(lower), float(upper), int(array.size))


def summarize_horizons(returns: pd.DataFrame, *, seed: int = 42, resamples: int = 10_000) -> pd.DataFrame:
    required = {"scorer_id", "horizon", "strategy_return_pct", "signal_value", "pnl_usd"}
    missing = required - set(returns.columns)
    if missing:
        raise TradingAnalysisError(f"returns data is missing columns: {', '.join(sorted(missing))}")
    rows: list[dict[str, Any]] = []
    for (scorer_id, horizon), group in returns.groupby(["scorer_id", "horizon"], sort=True):
        values = group["strategy_return_pct"].astype(float)
        net_column = "net_strategy_return_pct" if "net_strategy_return_pct" in group else "strategy_return_pct"
        net_values = group[net_column].astype(float)
        interval = bootstrap_mean_interval(values.tolist(), seed=seed, resamples=resamples)
        net_interval = bootstrap_mean_interval(net_values.tolist(), seed=seed, resamples=resamples)
        traded = group[group["signal_value"].astype(float) != 0]
        rows.append(
            {
                "scorer_id": str(scorer_id),
                "scorer": SCORER_DISPLAY.get(str(scorer_id), str(scorer_id)),
                "horizon": int(horizon),
                "n_events": len(group),
                "n_trades": len(traded),
                "n_neutral": len(group) - len(traded),
                "mean_return_pct": interval.mean,
                "ci95_lower_pct": interval.lower,
                "ci95_upper_pct": interval.upper,
                "median_return_pct": float(values.median()),
                "mean_traded_return_pct": float(traded["strategy_return_pct"].mean()) if len(traded) else np.nan,
                "trade_hit_rate": float((traded["strategy_return_pct"] > 0).mean()) if len(traded) else np.nan,
                "mean_pnl_usd": float(group["pnl_usd"].mean()),
                "net_mean_return_pct": net_interval.mean,
                "net_ci95_lower_pct": net_interval.lower,
                "net_ci95_upper_pct": net_interval.upper,
                "net_median_return_pct": float(net_values.median()),
                "net_mean_traded_return_pct": float(traded[net_column].mean()) if len(traded) else np.nan,
                "net_trade_hit_rate": float((traded[net_column] > 0).mean()) if len(traded) else np.nan,
                "mean_net_pnl_usd": float(
                    group["net_pnl_usd"].mean() if "net_pnl_usd" in group else group["pnl_usd"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_company_horizons(returns: pd.DataFrame, *, scorer_id: str = "consensus/majority") -> pd.DataFrame:
    selected = returns[returns["scorer_id"] == scorer_id]
    rows: list[dict[str, Any]] = []
    for (symbol, horizon), group in selected.groupby(["symbol", "horizon"], sort=True):
        traded = group[group["signal_value"].astype(float) != 0]
        rows.append(
            {
                "symbol": str(symbol),
                "horizon": int(horizon),
                "n_events": len(group),
                "n_trades": len(traded),
                "mean_return_pct": float(group["strategy_return_pct"].mean()),
                "median_return_pct": float(group["strategy_return_pct"].median()),
                "trade_hit_rate": float((traded["strategy_return_pct"] > 0).mean()) if len(traded) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_date_horizons(returns: pd.DataFrame, *, scorer_id: str = "consensus/majority") -> pd.DataFrame:
    selected = returns[returns["scorer_id"] == scorer_id]
    return (
        selected.groupby(["news_date", "horizon"], as_index=False)
        .agg(n_events=("strategy_return_pct", "size"), mean_return_pct=("strategy_return_pct", "mean"))
        .sort_values(["news_date", "horizon"])
    )


def summarize_signal_distribution(signals: pd.DataFrame) -> pd.DataFrame:
    data = signals.copy()
    data["signal"] = data["signal"].fillna("missing")
    rows = data.groupby(["scorer_id", "signal"], as_index=False).size().rename(columns={"size": "event_count"})
    rows["scorer"] = rows["scorer_id"].map(SCORER_DISPLAY).fillna(rows["scorer_id"])
    return rows[["scorer_id", "scorer", "signal", "event_count"]].sort_values(["scorer_id", "signal"])


def summarize_source_yield(articles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, group in articles.groupby("symbol", sort=True):
        accepted = group[group["screening_decision"] == "include"]
        provider_text = accepted["providers"].fillna("").astype(str)
        newsapi = provider_text.str.contains("newsapi", regex=False)
        tavily = provider_text.str.contains("tavily", regex=False)
        lseg = provider_text.str.contains("lseg", regex=False)
        rows.append(
            {
                "symbol": str(symbol),
                "discovered_articles": len(group),
                "accepted_articles": len(accepted),
                "excluded_articles": len(group) - len(accepted),
                "acceptance_rate": len(accepted) / len(group) if len(group) else np.nan,
                "accepted_newsapi_only": int((newsapi & ~tavily).sum()),
                "accepted_tavily_only": int((tavily & ~newsapi).sum()),
                "accepted_both": int((newsapi & tavily).sum()),
                "accepted_lseg": int(lseg.sum()),
            }
        )
    return pd.DataFrame(rows)


def _agreement_rate(left: pd.Series, right: pd.Series) -> tuple[int, float]:
    valid = left.notna() & right.notna()
    n = int(valid.sum())
    return n, float((left[valid] == right[valid]).mean()) if n else np.nan


def summarize_agreement(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    llm_scorers: Sequence[str],
    *,
    primary_scorer: str = "consensus/majority",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    article_pivot = scores.pivot_table(index="article_id", columns="scorer_id", values="label", aggfunc="first")
    pairwise_rows: list[dict[str, Any]] = []
    comparison_scorers = [*llm_scorers, "baseline/vader"]
    for scorer_a, scorer_b in combinations(comparison_scorers, 2):
        if scorer_a not in article_pivot or scorer_b not in article_pivot:
            continue
        n, rate = _agreement_rate(article_pivot[scorer_a], article_pivot[scorer_b])
        pairwise_rows.append(
            {
                "scorer_a": scorer_a,
                "scorer_a_display": SCORER_DISPLAY.get(scorer_a, scorer_a),
                "scorer_b": scorer_b,
                "scorer_b_display": SCORER_DISPLAY.get(scorer_b, scorer_b),
                "n_articles": n,
                "agreement_rate": rate,
            }
        )

    metrics: list[dict[str, Any]] = []
    complete_articles = article_pivot.dropna(subset=list(llm_scorers)) if set(llm_scorers) <= set(article_pivot) else pd.DataFrame()
    if not complete_articles.empty:
        unique_counts = complete_articles[list(llm_scorers)].nunique(axis=1)
        metrics.extend(
            [
                _metric("article_llm_unanimous", int((unique_counts == 1).sum()), len(unique_counts)),
                _metric("article_llm_majority_not_unanimous", int((unique_counts == 2).sum()), len(unique_counts)),
                _metric("article_llm_three_way_split", int((unique_counts == 3).sum()), len(unique_counts)),
            ]
        )

    event_pivot = signals.pivot_table(index=["symbol", "news_date"], columns="scorer_id", values="signal", aggfunc="first")
    complete_events = event_pivot.dropna(subset=list(llm_scorers)) if set(llm_scorers) <= set(event_pivot) else pd.DataFrame()
    if not complete_events.empty:
        unique_counts = complete_events[list(llm_scorers)].nunique(axis=1)
        metrics.append(_metric("event_llm_signal_unanimous", int((unique_counts == 1).sum()), len(unique_counts)))
    if {"consensus/majority", "baseline/vader"} <= set(event_pivot):
        valid = event_pivot[["consensus/majority", "baseline/vader"]].dropna()
        metrics.append(
            _metric(
                "event_consensus_vader_signal_agreement",
                int((valid["consensus/majority"] == valid["baseline/vader"]).sum()),
                len(valid),
            )
        )
    baseline_scorer = next(
        (scorer for scorer in ("baseline/vader", "baseline/finbert") if scorer in event_pivot),
        None,
    )
    if baseline_scorer is not None and primary_scorer in event_pivot:
        valid = event_pivot[[primary_scorer, baseline_scorer]].dropna()
        metrics.append(
            _metric(
                "event_primary_baseline_signal_agreement",
                int((valid[primary_scorer] == valid[baseline_scorer]).sum()),
                len(valid),
            )
        )
    return pd.DataFrame(metrics), pd.DataFrame(pairwise_rows)


def _metric(metric: str, numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else np.nan,
    }


def summarize_leave_one_company_out(returns: pd.DataFrame, *, scorer_id: str = "consensus/majority") -> pd.DataFrame:
    selected = returns[returns["scorer_id"] == scorer_id]
    symbols = sorted(selected["symbol"].unique())
    rows: list[dict[str, Any]] = []
    for horizon, horizon_rows in selected.groupby("horizon", sort=True):
        rows.append(
            {
                "horizon": int(horizon),
                "omitted_symbol": "(none)",
                "n_events": len(horizon_rows),
                "mean_return_pct": float(horizon_rows["strategy_return_pct"].mean()),
            }
        )
        for symbol in symbols:
            retained = horizon_rows[horizon_rows["symbol"] != symbol]
            rows.append(
                {
                    "horizon": int(horizon),
                    "omitted_symbol": symbol,
                    "n_events": len(retained),
                    "mean_return_pct": float(retained["strategy_return_pct"].mean()),
                }
            )
    return pd.DataFrame(rows)


def compare_runs(current_returns: pd.DataFrame, comparison_returns: pd.DataFrame) -> pd.DataFrame:
    current = (
        current_returns.groupby(["scorer_id", "horizon"], as_index=False)
        .agg(current_n_events=("strategy_return_pct", "size"), current_mean_return_pct=("strategy_return_pct", "mean"))
    )
    previous = (
        comparison_returns.groupby(["scorer_id", "horizon"], as_index=False)
        .agg(previous_n_events=("strategy_return_pct", "size"), previous_mean_return_pct=("strategy_return_pct", "mean"))
    )
    result = current.merge(previous, on=["scorer_id", "horizon"], how="inner")
    result["mean_return_change_pp"] = result["current_mean_return_pct"] - result["previous_mean_return_pct"]
    result["scorer"] = result["scorer_id"].map(SCORER_DISPLAY).fillna(result["scorer_id"])
    return result.sort_values(["scorer_id", "horizon"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_article_path(run_manifest: dict[str, Any]) -> Path:
    for value in run_manifest.get("files", {}):
        path = Path(value)
        if path.name == "articles.csv":
            return path
    raise TradingAnalysisError("run manifest does not reference an articles.csv input")


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _safe_rate(metrics: pd.DataFrame, metric: str) -> tuple[int, int, float]:
    rows = metrics[metrics["metric"] == metric]
    if rows.empty:
        return 0, 0, float("nan")
    row = rows.iloc[0]
    return int(row["numerator"]), int(row["denominator"]), float(row["rate"])


def _write_plots(
    output_dir: Path,
    horizon: pd.DataFrame,
    company: pd.DataFrame,
    signals: pd.DataFrame,
    source_yield: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:
        raise TradingAnalysisError("plot generation requires the figures extra: uv sync --extra figures") from exc

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    available_scorers = list(dict.fromkeys(horizon["scorer_id"].astype(str).tolist()))
    scorer_order = [scorer for scorer in SCORER_DISPLAY if scorer in available_scorers]
    scorer_order.extend(scorer for scorer in available_scorers if scorer not in scorer_order)
    colors = ["#4C78A8", "#F58518", "#B279A2", "#72B7B2", "#79706E", "#E45756", "#54A24B"]
    paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(11, 6.2))
    offsets = np.linspace(-0.28, 0.28, len(scorer_order))
    for index, (offset, scorer) in enumerate(zip(offsets, scorer_order, strict=True)):
        color = colors[index % len(colors)]
        rows = horizon[horizon["scorer_id"] == scorer].sort_values("horizon")
        yerr = np.vstack(
            [
                rows["mean_return_pct"] - rows["ci95_lower_pct"],
                rows["ci95_upper_pct"] - rows["mean_return_pct"],
            ]
        )
        axis.errorbar(
            rows["horizon"] + offset,
            rows["mean_return_pct"],
            yerr=yerr,
            fmt="o",
            capsize=2.5,
            markersize=4.5,
            linewidth=1,
            color=color,
            label=SCORER_DISPLAY.get(scorer, scorer),
        )
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set(
        title="Mean event return by exit horizon",
        xlabel="Trading sessions after entry",
        ylabel="Strategy return (%)",
        xticks=sorted(horizon["horizon"].unique()),
    )
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.text(
        0.01,
        0.01,
        "Bars are naive 95% event-bootstrap intervals (10,000 resamples); overlapping events are not independent.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = output_dir / "mean_returns_by_horizon.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    matrix = company.pivot(index="symbol", columns="horizon", values="mean_return_pct").sort_index()
    values = matrix.to_numpy(dtype=float)
    limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.01)
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    image = axis.imshow(values, cmap="PuOr", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    axis.set(
        title="Primary-scorer mean return by company and horizon",
        xlabel="Trading-session horizon",
        ylabel="Company",
        xticks=range(len(matrix.columns)),
        xticklabels=matrix.columns,
        yticks=range(len(matrix.index)),
        yticklabels=matrix.index,
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                axis.text(column_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Mean return (%)")
    fig.tight_layout()
    path = output_dir / "consensus_company_horizon_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    signal_order = ["negative", "neutral", "positive", "missing"]
    signal_colors = {"negative": "#6F4E7C", "neutral": "#BAB0AC", "positive": "#F28E2B", "missing": "#D6D6D6"}
    signal_matrix = signals.pivot(index="scorer", columns="signal", values="event_count").fillna(0)
    signal_matrix = signal_matrix.reindex(
        [SCORER_DISPLAY.get(s, s) for s in scorer_order if SCORER_DISPLAY.get(s, s) in signal_matrix.index]
    )
    fig, axis = plt.subplots(figsize=(9, 4.8))
    left = np.zeros(len(signal_matrix))
    for signal in signal_order:
        values_for_signal = signal_matrix[signal].to_numpy() if signal in signal_matrix else np.zeros(len(signal_matrix))
        axis.barh(signal_matrix.index, values_for_signal, left=left, color=signal_colors[signal], label=signal)
        left += values_for_signal
    axis.set(title="Daily trading signals by scorer", xlabel="Company-day events", ylabel="")
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = output_dir / "signal_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    symbols = source_yield["symbol"].tolist()
    bottom = np.zeros(len(symbols))
    source_columns = [
        ("accepted_tavily_only", "Tavily only", "#4C78A8"),
        ("accepted_newsapi_only", "NewsAPI only", "#F58518"),
        ("accepted_both", "Matched across providers", "#B279A2"),
        ("accepted_lseg", "LSEG Workspace", "#54A24B"),
    ]
    for column, label, color in source_columns:
        values_for_source = source_yield[column].to_numpy()
        axis.bar(symbols, values_for_source, bottom=bottom, label=label, color=color)
        bottom += values_for_source
    axis.set(title="Accepted article yield by company and provider", xlabel="Company", ylabel="Accepted articles")
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = output_dir / "source_yield_by_company.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    agreement_scorers = list(dict.fromkeys([*scorer_order[:3], "baseline/vader"]))
    labels = [SCORER_DISPLAY.get(scorer, scorer) for scorer in agreement_scorers]
    agreement = np.eye(len(agreement_scorers))
    for row in pairwise.itertuples():
        if row.scorer_a in agreement_scorers and row.scorer_b in agreement_scorers:
            index_a = agreement_scorers.index(row.scorer_a)
            index_b = agreement_scorers.index(row.scorer_b)
            agreement[index_a, index_b] = agreement[index_b, index_a] = row.agreement_rate
    fig, axis = plt.subplots(figsize=(7, 5.7))
    image = axis.imshow(agreement, cmap="Blues", vmin=0, vmax=1)
    axis.set(
        title="Article-level pairwise label agreement",
        xticks=range(len(labels)),
        xticklabels=labels,
        yticks=range(len(labels)),
        yticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=25)
    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            axis.text(column_index, row_index, f"{agreement[row_index, column_index]:.0%}", ha="center", va="center")
    fig.colorbar(image, ax=axis, shrink=0.8, label="Agreement rate")
    fig.tight_layout()
    path = output_dir / "pairwise_label_agreement.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def _write_summary(
    output_dir: Path,
    run_manifest: dict[str, Any],
    articles: pd.DataFrame,
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: pd.DataFrame,
    source_yield: pd.DataFrame,
    metrics: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    comparison: pd.DataFrame | None,
    primary_scorer: str,
    *,
    seed: int,
    resamples: int,
) -> Path:
    primary = horizon[horizon["scorer_id"] == primary_scorer].sort_values("horizon")
    if primary.empty:
        primary_scorer = str(horizon["scorer_id"].iloc[0])
        primary = horizon[horizon["scorer_id"] == primary_scorer].sort_values("horizon")
    primary_display = SCORER_DISPLAY.get(primary_scorer, primary_scorer)
    best = primary.loc[primary["net_mean_return_pct"].idxmax()]
    worst = primary.loc[primary["net_mean_return_pct"].idxmin()]
    planned_events = len(run_manifest["settings"]["symbols"]) * len(run_manifest["settings"]["dates"])
    primary_signals = signals[signals["scorer_id"].astype(str) == primary_scorer]
    if "valid_count" in primary_signals:
        primary_signals = primary_signals[primary_signals["valid_count"].fillna(0).astype(float) > 0]
    evaluable_events = int(primary_signals[["symbol", "news_date"]].drop_duplicates().shape[0])
    accepted = int((articles["screening_decision"] == "include").sum())
    llm_unanimous = _safe_rate(metrics, "article_llm_unanimous")
    event_unanimous = _safe_rate(metrics, "event_llm_signal_unanimous")
    baseline_agreement = _safe_rate(metrics, "event_primary_baseline_signal_agreement")
    primary_rows = [
        [
            str(int(row.horizon)),
            f"{row.mean_return_pct:.3f}%",
            f"{row.net_mean_return_pct:.3f}%",
            f"[{row.net_ci95_lower_pct:.3f}%, {row.net_ci95_upper_pct:.3f}%]",
            f"{row.net_trade_hit_rate:.1%}",
            f"{int(row.n_trades)}/{int(row.n_events)}",
        ]
        for row in primary.itertuples()
    ]
    final_horizon = int(primary["horizon"].max())
    final_rows = horizon[horizon["horizon"] == final_horizon].sort_values("mean_return_pct", ascending=False)
    scorer_rows = [
        [
            str(row.scorer),
            f"{row.mean_return_pct:.3f}%",
            f"[{row.ci95_lower_pct:.3f}%, {row.ci95_upper_pct:.3f}%]",
            f"{row.trade_hit_rate:.1%}",
        ]
        for row in final_rows.itertuples()
    ]
    coverage_rows = [
        [
            str(row.symbol),
            str(int(row.discovered_articles)),
            str(int(row.accepted_articles)),
            f"{row.acceptance_rate:.1%}",
        ]
        for row in source_yield.itertuples()
    ]
    loo_selected = leave_one_out[leave_one_out["omitted_symbol"] != "(none)"]
    loo_ranges = loo_selected.groupby("horizon")["mean_return_pct"].agg(["min", "max"]).reset_index()
    sign_stable = int(((loo_ranges["min"] > 0) | (loo_ranges["max"] < 0)).sum())
    comparison_note = ""
    if comparison is not None and not comparison.empty:
        comparison_primary = comparison[comparison["scorer_id"] == primary_scorer]
        delta = comparison_primary["mean_return_change_pp"].abs().mean()
        previous_n = int(comparison_primary["previous_n_events"].max())
        comparison_note = (
            f" Relative to the comparison run ({previous_n} events), the absolute change in the primary mean averaged "
            f"{delta:.3f} percentage points across horizons, showing that the small pilot was sensitive to panel composition."
        )

    zero_crossing = int(
        ((primary["net_ci95_lower_pct"] <= 0) & (primary["net_ci95_upper_pct"] >= 0)).sum()
    )
    horizon_count = int(primary["horizon"].nunique())
    missing_events = planned_events - evaluable_events
    title = f"{run_manifest.get('title') or run_manifest['run_id']}: Technical Results"
    text = f"""# {title}

Generated: {datetime.now(UTC).isoformat()}

## Technical summary

The run covered {len(run_manifest["settings"]["symbols"])} companies, {len(run_manifest["settings"]["dates"])} news dates,
and {horizon_count} trading-session exit horizons. It discovered {len(articles)} company-linked article revisions,
retained {accepted} after screening, and produced evaluable data for {evaluable_events} of {planned_events} planned
company-day events. The {missing_events} missing events were not imputed.

The primary strategy ({primary_display}) had a best net mean of {best.net_mean_return_pct:.3f}% at horizon {int(best.horizon)} and a
worst net mean of {worst.net_mean_return_pct:.3f}% at horizon {int(worst.horizon)}. Zero was inside
{zero_crossing}/{horizon_count} primary bootstrap intervals. These are descriptive event-level results, not evidence of a
reliable trading edge.{comparison_note}

## Key findings

1. **Uncertainty remains material.** The primary net means and intervals are shown below; confirmatory interpretation
   requires the frozen chronological holdout.
2. **Model agreement is incomplete.** The configured LLMs were unanimous on {llm_unanimous[0]}/{llm_unanimous[1]}
   articles ({llm_unanimous[2]:.1%}) and on {event_unanimous[0]}/{event_unanimous[1]} daily signals
   ({event_unanimous[2]:.1%}). The primary scorer and the first available baseline agreed on
   {baseline_agreement[0]}/{baseline_agreement[1]} evaluable daily signals ({baseline_agreement[2]:.1%}).
3. **Results were company-sensitive.** Leave-one-company-out means retained the full-panel sign at {sign_stable}/{horizon_count}
   horizons; the remaining horizons changed sign for at least one omission.
4. **Coverage remained uneven.** Accepted yield ranged from {int(source_yield.accepted_articles.min())} to
   {int(source_yield.accepted_articles.max())} articles per company, with {missing_events} planned event(s) unevaluable.

![Mean return estimates](mean_returns_by_horizon.png)

## Primary-scorer returns — {primary_display}

Intervals are percentile bootstrap intervals over company-day events using seed {seed} and {resamples:,} resamples.
The bootstrap treats events as independent even though dates and companies overlap, so it is a descriptive sensitivity
measure rather than a valid causal or portfolio-level confidence interval. Hit rate excludes neutral/no-trade events.

{_md_table(["Horizon", "Gross mean", "Net mean", "Net 95% interval", "Net hit rate", "Trades/events"], primary_rows)}

![Company-horizon heatmap](consensus_company_horizon_heatmap.png)

## Scorer comparison at horizon {final_horizon}

{_md_table(["Scorer", "Mean return", "Naive 95% interval", "Trade hit rate"], scorer_rows)}

![Signal distribution](signal_distribution.png)

![Pairwise label agreement](pairwise_label_agreement.png)

## Source coverage and screening

Articles were loaded from the source corpora recorded in the completed run manifest, deduplicated using their native
identities, assigned to exchange-local dates, and screened for target-company relevance. No source text was overwritten.

{_md_table(["Company", "Discovered", "Accepted", "Acceptance rate"], coverage_rows)}

![Source yield](source_yield_by_company.png)

## Method and metric definitions

- Companies: {", ".join(run_manifest["settings"]["symbols"])}.
- News dates: {", ".join(run_manifest["settings"]["dates"])}; timestamps assigned in
  `{run_manifest["settings"].get("timezone", "America/New_York")}`.
- Text scored: target-company context, headline, and the configured provider representation.
- Primary scorer: {primary_display}. Labels map to positive=1, neutral=0, negative=-1 and are averaged equally by
  company-day before the configured minimum-story and threshold policy is applied.
- Entry: first observed session open after the recorded decision availability timestamp; legacy web-only runs use the
  next session after the news date.
- Exit: adjusted close at sessions 1–7. Long return is `exit / entry - 1`; short return is its negative.
- P/L: gross and configured transaction-cost-adjusted net return on the run's fixed notional; short-borrow assumptions
  are disclosed in the run manifest.
- Aggregation: equal-weight arithmetic mean of event returns. Overlapping events are not combined into a funded portfolio.

## Robustness checks

- Naive event-bootstrap intervals quantify sampling sensitivity but do not resolve event dependence.
- Traded-only means and hit rates are reported separately from all-event means that include neutral zeros.
- Leave-one-company-out estimates expose concentration in individual symbols.
- Company and news-date CSV breakdowns expose cross-sectional and regime dependence.
- The completed run and this analysis are non-overwriting and independently manifested.

## Limitations

- The recorded date panel may not represent other market regimes; temporal generalization requires a frozen holdout.
- Evaluable company-day events number {evaluable_events}; this remains too small for confirmatory inference.
- News providers differ in indexing, timestamp precision, entitlements, and source mix. Sentiment is not the same as
  market-impact prediction.
- Manual relevance review improves precision but introduces researcher judgment; all decisions are auditable.
- Multiple horizons, scorers, and companies create substantial multiplicity. Adjusted p-values are exploratory screening
  diagnostics and no causal claims are made.
- Net returns include configured transaction and borrow costs, but omit market impact, liquidity constraints, and funding interactions.

## Recommended next steps

1. Expand to at least several non-adjacent weeks and freeze the inclusion rules before collection.
2. Pre-register one primary scorer and one primary horizon; treat the rest as sensitivity analyses.
3. Add market- and sector-adjusted abnormal returns, clustered uncertainty by date, and realistic trading costs.
4. Double-screen a sample of articles and report inter-rater agreement for relevance decisions.
5. Reserve a later time block as a fully untouched holdout before interpreting any apparent edge.

## Reproducibility artifacts

CSV tables, plots, input hashes, package versions, model identifiers, bootstrap settings, and source mappings are stored
beside this report. See `analysis_manifest.json` and `source_map.md`.
"""
    path = output_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def analyze_trading_run(
    run_dir: Path,
    output_dir: Path,
    *,
    comparison_run_dir: Path | None = None,
    seed: int = 42,
    resamples: int = 10_000,
) -> TradingAnalysisResult:
    """Create non-overwriting robustness tables, plots, and a technical report for a completed trading run."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise TradingAnalysisError(f"analysis output directory already exists: {output_dir}")
    manifest_path = run_dir / "run_manifest.json"
    required_files = [manifest_path, run_dir / "returns.csv", run_dir / "daily_signals.csv", run_dir / "sentiment_scores.csv"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise TradingAnalysisError(f"completed run inputs are missing: {', '.join(missing)}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "completed":
        raise TradingAnalysisError("trading run manifest is not completed")
    articles_path = _input_article_path(run_manifest)
    if not articles_path.exists():
        raise TradingAnalysisError(f"articles input does not exist: {articles_path}")

    articles = pd.read_csv(articles_path)
    returns = pd.read_csv(run_dir / "returns.csv")
    signals = pd.read_csv(run_dir / "daily_signals.csv")
    scores = pd.read_csv(run_dir / "sentiment_scores.csv")
    llm_scorers = tuple(str(value) for value in run_manifest["settings"]["models"])
    configured_primary = str(run_manifest["settings"].get("primary_model") or "consensus/majority")
    available_return_scorers = set(returns["scorer_id"].astype(str))
    primary_scorer = configured_primary if configured_primary in available_return_scorers else str(returns["scorer_id"].iloc[0])

    horizon = summarize_horizons(returns, seed=seed, resamples=resamples)
    company = summarize_company_horizons(returns, scorer_id=primary_scorer)
    date = summarize_date_horizons(returns, scorer_id=primary_scorer)
    signal_distribution = summarize_signal_distribution(signals)
    source_yield = summarize_source_yield(articles)
    agreement_metrics, pairwise = summarize_agreement(scores, signals, llm_scorers, primary_scorer=primary_scorer)
    leave_one_out = summarize_leave_one_company_out(returns, scorer_id=primary_scorer)
    effectiveness = evaluate_effectiveness(returns, scorer_display=SCORER_DISPLAY)
    comparison: pd.DataFrame | None = None
    if comparison_run_dir is not None:
        comparison_path = Path(comparison_run_dir) / "returns.csv"
        if not comparison_path.exists():
            raise TradingAnalysisError(f"comparison returns input does not exist: {comparison_path}")
        comparison = compare_runs(returns, pd.read_csv(comparison_path))

    output_dir.mkdir(parents=True, exist_ok=False)
    generated = [
        _write_csv(horizon, output_dir / "horizon_summary.csv"),
        _write_csv(company, output_dir / "company_horizon_summary.csv"),
        _write_csv(date, output_dir / "date_horizon_summary.csv"),
        _write_csv(signal_distribution, output_dir / "signal_distribution.csv"),
        _write_csv(source_yield, output_dir / "source_yield.csv"),
        _write_csv(agreement_metrics, output_dir / "agreement_metrics.csv"),
        _write_csv(pairwise, output_dir / "pairwise_label_agreement.csv"),
        _write_csv(leave_one_out, output_dir / "leave_one_company_out.csv"),
        _write_csv(effectiveness.significance, output_dir / "effectiveness_significance.csv"),
        _write_csv(effectiveness.benchmarks, output_dir / "effectiveness_benchmarks.csv"),
        _write_csv(effectiveness.economics, output_dir / "effectiveness_economics.csv"),
    ]
    if comparison is not None:
        generated.append(_write_csv(comparison, output_dir / "pilot_comparison.csv"))
    generated.extend(_write_plots(output_dir, horizon, company, signal_distribution, source_yield, pairwise))
    summary_path = _write_summary(
        output_dir,
        run_manifest,
        articles,
        returns,
        signals,
        horizon,
        source_yield,
        agreement_metrics,
        leave_one_out,
        comparison,
        primary_scorer,
        seed=seed,
        resamples=resamples,
    )
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + format_effectiveness_markdown(effectiveness, primary_scorer=primary_scorer) + "\n")
    generated.append(summary_path)
    source_map = output_dir / "source_map.md"
    source_map.write_text(
        "# Source map\n\n"
        f"- `{articles_path}`: article discovery, providers, screening decisions, and accepted yield.\n"
        f"- `{run_dir / 'sentiment_scores.csv'}`: article-level labels and scorer metadata.\n"
        f"- `{run_dir / 'daily_signals.csv'}`: company-day means and trading signals.\n"
        f"- `{run_dir / 'returns.csv'}`: prices, horizons, strategy returns, and hypothetical P/L.\n"
        f"- `{manifest_path}`: run configuration, source corpora, environment, and input/output hashes.\n",
        encoding="utf-8",
    )
    generated.append(source_map)
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    analysis_manifest = {
        "schema_version": 1,
        "analysis_id": output_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "source_run_id": run_manifest["run_id"],
        "source_run_dir": str(run_dir),
        "comparison_run_dir": str(comparison_run_dir) if comparison_run_dir else None,
        "exploratory": True,
        "preregistered": False,
        "causal_claim": False,
        "bootstrap": {"method": "percentile_event_resampling", "seed": seed, "resamples": resamples, "confidence": 0.95},
        "inputs": {str(path): _sha256(path) for path in [manifest_path, articles_path, *required_files[1:]]},
        "files": {str(path): _sha256(path) for path in generated},
        "notes": [
            "Bootstrap intervals treat overlapping events as independent and are descriptive only.",
            "Neutral signals contribute zero to all-event mean returns; traded-only metrics are reported separately.",
            "Effectiveness p-values assume event independence and are screening diagnostics; Benjamini-Hochberg "
            "controls the false discovery rate across the scorer-horizon mean-return family only.",
        ],
    }
    analysis_manifest_path.write_text(json.dumps(analysis_manifest, indent=2, sort_keys=True), encoding="utf-8")
    generated.append(analysis_manifest_path)
    return TradingAnalysisResult(output_dir, summary_path, analysis_manifest_path, tuple(generated))
