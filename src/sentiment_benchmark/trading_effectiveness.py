"""Formal effectiveness tests for a completed news-sentiment trading run.

Three complementary families, all computed per (scorer, horizon) cell:

1. Significance & direction — whether mean event returns differ from zero
   (t-test, Wilcoxon signed-rank, sign test) and whether the traded hit rate
   beats a coin flip (binomial), with Benjamini-Hochberg control of the false
   discovery rate across the scorer-horizon mean-return family.
2. Benchmark comparison — strategy returns against a passive buy-and-hold of the
   same names, with a paired test of the per-event difference.
3. Risk-adjusted economics — per-event return/risk ratio, win/loss profile,
   profit factor, naive max drawdown, and realised P/L.

Every test treats overlapping company-day events as independent, so the
p-values are optimistic screening diagnostics rather than confirmatory inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED_COLUMNS = {
    "scorer_id",
    "horizon",
    "strategy_return_pct",
    "signal_value",
    "market_return",
    "pnl_usd",
    "entry_date",
    "symbol",
}


class TradingEffectivenessError(RuntimeError):
    """Raised when effectiveness inputs are missing required columns."""


@dataclass(frozen=True)
class EffectivenessResult:
    significance: pd.DataFrame
    benchmarks: pd.DataFrame
    economics: pd.DataFrame


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Return BH false-discovery-rate adjusted q-values, ignoring non-finite p-values."""
    values = np.asarray(list(pvalues), dtype=float)
    q_values = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return q_values.tolist()
    finite_p = values[finite]
    n = finite_p.size
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(n)
    restored[order] = adjusted
    q_values[finite] = restored
    return q_values.tolist()


def _ttest_vs_zero(values: np.ndarray) -> float:
    if values.size < 2 or float(values.std(ddof=1)) == 0.0:
        return float("nan")
    return float(stats.ttest_1samp(values, 0.0).pvalue)


def _wilcoxon_vs_zero(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if nonzero.size < 1:
        return float("nan")
    try:
        return float(stats.wilcoxon(nonzero).pvalue)
    except ValueError:
        return float("nan")


def _sign_test(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return float("nan")
    positives = int((nonzero > 0).sum())
    return float(stats.binomtest(positives, int(nonzero.size), 0.5).pvalue)


def _display(scorer_id: str, scorer_display: dict[str, str] | None) -> str:
    return (scorer_display or {}).get(scorer_id, scorer_id)


def _require(returns: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(returns.columns)
    if missing:
        raise TradingEffectivenessError(f"returns data is missing columns: {', '.join(sorted(missing))}")


def _net_return_column(returns: pd.DataFrame) -> str:
    return "net_strategy_return_pct" if "net_strategy_return_pct" in returns else "strategy_return_pct"


def _net_pnl_column(returns: pd.DataFrame) -> str:
    return "net_pnl_usd" if "net_pnl_usd" in returns else "pnl_usd"


def significance_tests(returns: pd.DataFrame, *, scorer_display: dict[str, str] | None = None) -> pd.DataFrame:
    """Per scorer-horizon: mean-return tests vs zero and a directional hit-rate test."""
    _require(returns)
    rows: list[dict[str, object]] = []
    net_column = _net_return_column(returns)
    for (scorer_id, horizon), group in returns.groupby(["scorer_id", "horizon"], sort=True):
        values = group["strategy_return_pct"].to_numpy(dtype=float)
        traded = group[group["signal_value"].astype(float) != 0]["strategy_return_pct"].to_numpy(dtype=float)
        net_values = group[net_column].to_numpy(dtype=float)
        net_traded = group[group["signal_value"].astype(float) != 0][net_column].to_numpy(dtype=float)
        wins = int((traded > 0).sum())
        n_trades = int(traded.size)
        rows.append(
            {
                "scorer_id": str(scorer_id),
                "scorer": _display(str(scorer_id), scorer_display),
                "horizon": int(horizon),
                "n_events": int(values.size),
                "mean_return_pct": float(values.mean()),
                "t_test_p": _ttest_vs_zero(values),
                "wilcoxon_p": _wilcoxon_vs_zero(values),
                "sign_test_p": _sign_test(values),
                "n_trades": n_trades,
                "trade_hit_rate": wins / n_trades if n_trades else float("nan"),
                "hit_rate_p": float(stats.binomtest(wins, n_trades, 0.5).pvalue) if n_trades else float("nan"),
                "net_mean_return_pct": float(net_values.mean()),
                "net_t_test_p": _ttest_vs_zero(net_values),
                "net_wilcoxon_p": _wilcoxon_vs_zero(net_values),
                "net_sign_test_p": _sign_test(net_values),
                "net_trade_hit_rate": float((net_traded > 0).mean()) if net_traded.size else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["t_test_q_bh"] = benjamini_hochberg(frame["t_test_p"].tolist())
        frame["significant_bh_5pct"] = frame["t_test_q_bh"] < 0.05
        frame["net_t_test_q_bh"] = benjamini_hochberg(frame["net_t_test_p"].tolist())
        frame["net_significant_bh_5pct"] = frame["net_t_test_q_bh"] < 0.05
    return frame


def benchmark_comparison(returns: pd.DataFrame, *, scorer_display: dict[str, str] | None = None) -> pd.DataFrame:
    """Per scorer-horizon: strategy returns against a passive buy-and-hold of the same names."""
    _require(returns)
    data = returns.copy()
    data["market_return_pct"] = data["market_return"].astype(float) * 100.0
    net_column = _net_return_column(data)
    rows: list[dict[str, object]] = []
    for (scorer_id, horizon), group in data.groupby(["scorer_id", "horizon"], sort=True):
        strategy = group["strategy_return_pct"].to_numpy(dtype=float)
        net_strategy = group[net_column].to_numpy(dtype=float)
        market = group["market_return_pct"].to_numpy(dtype=float)
        difference = strategy - market
        rows.append(
            {
                "scorer_id": str(scorer_id),
                "scorer": _display(str(scorer_id), scorer_display),
                "horizon": int(horizon),
                "n_events": int(strategy.size),
                "strategy_mean_pct": float(strategy.mean()),
                "buy_and_hold_mean_pct": float(market.mean()),
                "excess_vs_buy_and_hold_pp": float(strategy.mean() - market.mean()),
                "paired_t_p": _ttest_vs_zero(difference),
                "paired_wilcoxon_p": _wilcoxon_vs_zero(difference),
                "net_strategy_mean_pct": float(net_strategy.mean()),
                "net_excess_vs_buy_and_hold_pp": float(net_strategy.mean() - market.mean()),
                "net_paired_t_p": _ttest_vs_zero(net_strategy - market),
                "net_paired_wilcoxon_p": _wilcoxon_vs_zero(net_strategy - market),
            }
        )
    return pd.DataFrame(rows)


def risk_economics(returns: pd.DataFrame, *, scorer_display: dict[str, str] | None = None) -> pd.DataFrame:
    """Per scorer-horizon: return/risk ratio, win/loss profile, profit factor, and drawdown."""
    _require(returns)
    rows: list[dict[str, object]] = []
    net_column = _net_return_column(returns)
    net_pnl_column = _net_pnl_column(returns)
    for (scorer_id, horizon), group in returns.groupby(["scorer_id", "horizon"], sort=True):
        ordered = group.sort_values(["entry_date", "symbol"])
        all_values = ordered["strategy_return_pct"].to_numpy(dtype=float)
        net_values = ordered[net_column].to_numpy(dtype=float)
        traded = ordered[ordered["signal_value"].astype(float) != 0]["strategy_return_pct"].to_numpy(dtype=float)
        wins = traded[traded > 0]
        losses = traded[traded < 0]
        volatility = float(all_values.std(ddof=1)) if all_values.size > 1 else float("nan")
        mean = float(all_values.mean())
        net_mean = float(net_values.mean())
        equity = np.cumsum(all_values)
        drawdown = equity - np.maximum.accumulate(equity)
        if losses.size and losses.sum() != 0:
            profit_factor = float(wins.sum() / abs(losses.sum()))
        else:
            profit_factor = float("inf") if wins.size else float("nan")
        rows.append(
            {
                "scorer_id": str(scorer_id),
                "scorer": _display(str(scorer_id), scorer_display),
                "horizon": int(horizon),
                "n_events": int(all_values.size),
                "n_trades": int(traded.size),
                "mean_return_pct": mean,
                "volatility_pct": volatility,
                "return_risk_ratio": mean / volatility if volatility and not np.isnan(volatility) else float("nan"),
                "win_rate": float((traded > 0).mean()) if traded.size else float("nan"),
                "avg_win_pct": float(wins.mean()) if wins.size else float("nan"),
                "avg_loss_pct": float(losses.mean()) if losses.size else float("nan"),
                "profit_factor": profit_factor,
                "total_pnl_usd": float(ordered["pnl_usd"].astype(float).sum()),
                "max_drawdown_pct": float(drawdown.min()) if drawdown.size else float("nan"),
                "net_mean_return_pct": net_mean,
                "net_volatility_pct": float(net_values.std(ddof=1)) if net_values.size > 1 else float("nan"),
                "total_net_pnl_usd": float(ordered[net_pnl_column].astype(float).sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_effectiveness(returns: pd.DataFrame, *, scorer_display: dict[str, str] | None = None) -> EffectivenessResult:
    """Compute all three effectiveness families for a completed run's returns table."""
    return EffectivenessResult(
        significance=significance_tests(returns, scorer_display=scorer_display),
        benchmarks=benchmark_comparison(returns, scorer_display=scorer_display),
        economics=risk_economics(returns, scorer_display=scorer_display),
    )


def _fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.3f}%"


def _fmt_pp(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:+.3f} pp"


def _fmt_ratio(value: float) -> str:
    if pd.isna(value):
        return "—"
    if np.isinf(value):
        return "∞"
    return f"{value:.2f}"


def _fmt_p(value: float) -> str:
    if pd.isna(value):
        return "—"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _fmt_rate(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.0%}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_effectiveness_markdown(
    result: EffectivenessResult,
    *,
    primary_scorer: str = "consensus/majority",
) -> str:
    """Build a self-contained 'Trading effectiveness' report section with a data-driven verdict."""
    significance = result.significance
    benchmarks = result.benchmarks
    economics = result.economics
    if significance.empty:
        return "## Trading effectiveness\n\nNo return rows were available for effectiveness testing.\n"

    total_cells = len(significance)
    n_significant = int(significance["net_significant_bh_5pct"].sum())
    n_events = int(significance["n_events"].max())
    beat_benchmark = benchmarks[
        (benchmarks["net_excess_vs_buy_and_hold_pp"] > 0) & (benchmarks["net_paired_t_p"] < 0.05)
    ]
    n_beat = int(len(beat_benchmark))

    ranked = economics.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_risk_ratio"])
    if not ranked.empty:
        best = ranked.loc[ranked["return_risk_ratio"].idxmax()]
        best_line = (
            f"The strongest risk-adjusted cell was {best['scorer']} at horizon {int(best['horizon'])} "
            f"(per-event return/risk {best['return_risk_ratio']:.2f}, mean {best['mean_return_pct']:.3f}%)."
        )
    else:
        best_line = "No cell had a finite return/risk ratio."

    if n_significant == 0 and n_beat == 0:
        verdict = (
            "No scorer-horizon mean return was distinguishable from zero after correction, and none beat "
            "buy-and-hold at conventional significance — consistent with no exploitable edge in this sample."
        )
    elif n_significant > 0:
        verdict = (
            f"{n_significant} of {total_cells} cells survived correction; treat these as hypotheses to confirm "
            "on an untouched holdout rather than as established edges, given the event dependence and small sample."
        )
    else:
        verdict = (
            f"No cell survived the zero-return correction, but {n_beat} beat buy-and-hold (paired); treat as "
            "exploratory and confirm out of sample."
        )

    if primary_scorer not in set(significance["scorer_id"]):
        primary_scorer = str(significance["scorer_id"].iloc[0])
    primary_display = str(significance.loc[significance["scorer_id"] == primary_scorer, "scorer"].iloc[0])

    sig_rows = [
        [
            str(int(row.horizon)),
            _fmt_pct(row.mean_return_pct),
            _fmt_pct(row.net_mean_return_pct),
            _fmt_p(row.net_t_test_p),
            _fmt_p(row.net_wilcoxon_p),
            _fmt_p(row.net_sign_test_p),
            _fmt_p(row.net_t_test_q_bh),
            "yes" if bool(row.net_significant_bh_5pct) else "no",
        ]
        for row in significance[significance["scorer_id"] == primary_scorer].sort_values("horizon").itertuples()
    ]
    bench_rows = [
        [
            str(int(row.horizon)),
            _fmt_pct(row.strategy_mean_pct),
            _fmt_pct(row.net_strategy_mean_pct),
            _fmt_pct(row.buy_and_hold_mean_pct),
            _fmt_pp(row.net_excess_vs_buy_and_hold_pp),
            _fmt_p(row.net_paired_t_p),
            _fmt_p(row.net_paired_wilcoxon_p),
        ]
        for row in benchmarks[benchmarks["scorer_id"] == primary_scorer].sort_values("horizon").itertuples()
    ]
    econ_rows = [
        [
            str(int(row.horizon)),
            _fmt_pct(row.mean_return_pct),
            _fmt_pct(row.net_mean_return_pct),
            _fmt_pct(row.volatility_pct),
            _fmt_ratio(row.return_risk_ratio),
            _fmt_rate(row.win_rate),
            _fmt_ratio(row.profit_factor),
            _fmt_pct(row.max_drawdown_pct),
        ]
        for row in economics[economics["scorer_id"] == primary_scorer].sort_values("horizon").itertuples()
    ]

    return f"""## Trading effectiveness

Formal tests of whether the strategy works, computed per scorer and exit horizon. Every test treats the
{n_events} company-day events as independent even though dates and companies overlap, so the p-values are
optimistic screening diagnostics, not confirmatory inference. Benjamini-Hochberg controls the false discovery
rate across the {total_cells} scorer-horizon mean-return hypotheses.

**Headline.** {n_significant} of {total_cells} scorer-horizon mean returns differed from zero at q<0.05 after
correction, and {n_beat} beat buy-and-hold at p<0.05 (paired). {best_line} {verdict}

### Significance and direction — {primary_display}

Mean-return tests are vs zero. Gross is shown for transparency; inference uses net returns when cost-adjusted columns exist.

{_table(
    ["Horizon", "Gross mean", "Net mean", "Net t p", "Net Wilcoxon p", "Net sign p", "Net BH q", "Sig. q<0.05"],
    sig_rows,
)}

### Strategy vs buy-and-hold — {primary_display}

Buy-and-hold is an always-long position in the same names over the same windows; the paired test is on the
per-event strategy-minus-benchmark difference.

{_table(
    ["Horizon", "Gross mean", "Net mean", "Buy-and-hold mean", "Net excess", "Net paired t p", "Net Wilcoxon p"],
    bench_rows,
)}

### Risk-adjusted economics — {primary_display}

Return/risk is the per-event mean over standard deviation across exported return rows. Policy-enabled runs omit holds
from this table and retain them in the decision/coverage outputs. Win rate, average win/loss, and profit factor use
traded events only. Max drawdown is the trough of a
naive equal-weight, date-ordered cumulative-return curve and is illustrative given overlapping events.

{_table(
    ["Horizon", "Gross mean", "Net mean", "Gross volatility", "Gross return/risk", "Win rate", "Profit factor", "Gross max drawdown"],
    econ_rows,
)}

Per-scorer effectiveness tables for every scorer are saved beside this report as
`effectiveness_significance.csv`, `effectiveness_benchmarks.csv`, and `effectiveness_economics.csv`.
"""
