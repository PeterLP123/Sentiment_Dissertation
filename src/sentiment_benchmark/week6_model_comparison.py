"""Development-tuned, funded Week 6 comparison across sentiment scorers.

The single-scorer :mod:`week6_pnl` workflow remains the accounting authority.
This module coordinates identical scorer runs, converts development-only score
quantiles into scorer-specific absolute gates, and freezes a strictly negative-
correlation portfolio before evaluation.
"""

from __future__ import annotations

import json
import math
import shlex
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .strategy_sweep import chronological_split_date
from .week6_pnl import (
    AlignmentCounts,
    Week6PnlConfig,
    Week6PnlError,
    _git_metadata,
    _sha256,
    _write_csv,
    build_manual_timing_check,
    build_portfolio_daily,
    build_portfolio_performance,
    build_stock_performance,
    build_strategy_grid,
    development_correlations,
    load_and_validate_inputs,
    select_stable_strategy,
)


@dataclass(frozen=True)
class Week6ModelComparisonConfig:
    """Frozen assumptions for one multi-scorer comparison."""

    run_id: str
    scorer_ids: tuple[str, ...]
    starting_capital: float = 100_000.0
    transaction_cost_bps_per_side: float = 10.0
    development_fraction: float = 0.7
    threshold_quantiles: tuple[float, ...] = (0.0, 0.5, 0.7)
    holding_periods: tuple[int, ...] = (1, 3, 5, 7)
    annualization_periods: int = 252
    annual_risk_free_rate: float = 0.0
    exchange_timezone: str = "America/New_York"
    min_joint_active_dates: int = 10
    min_stock_active_days: int = 10
    negative_portfolio_stock_count: int | None = None
    maximum_stock_weight: float = 0.6


@dataclass(frozen=True)
class Week6ModelComparisonResult:
    """Paths and key frozen choices from a completed comparison."""

    output_dir: Path
    split_date: str
    selected_rules: tuple[tuple[str, float, float, int], ...]


def _validate_config(config: Week6ModelComparisonConfig) -> None:
    if len(config.scorer_ids) < 2 or len(set(config.scorer_ids)) != len(config.scorer_ids):
        raise Week6PnlError("model comparison requires at least two unique scorer ids")
    if not 0 < config.development_fraction < 1:
        raise Week6PnlError("development_fraction must be between zero and one")
    if not config.threshold_quantiles or any(not 0 <= value < 1 for value in config.threshold_quantiles):
        raise Week6PnlError("threshold quantiles must lie in [0, 1)")
    if any(value < 1 for value in config.holding_periods):
        raise Week6PnlError("holding periods must be positive")
    if config.min_joint_active_dates < 1 or config.min_stock_active_days < 1:
        raise Week6PnlError("activity support thresholds must be positive")
    if not 0 < config.maximum_stock_weight <= 1:
        raise Week6PnlError("maximum stock weight must lie in (0, 1]")


def common_comparison_split(
    signals_path: str | Path,
    scorer_ids: tuple[str, ...],
    development_fraction: float,
) -> str:
    """Return one split date after enforcing identical scorer company-day keys."""

    frame = pd.read_csv(signals_path, usecols=["symbol", "news_date", "scorer_id"])
    frame = frame.loc[frame["scorer_id"].astype(str).isin(scorer_ids)].copy()
    available = set(frame["scorer_id"].astype(str))
    if missing := set(scorer_ids) - available:
        raise Week6PnlError(f"comparison scorers are absent: {sorted(missing)}")
    expected: set[tuple[str, str]] | None = None
    for scorer_id in scorer_ids:
        keys = set(
            frame.loc[frame["scorer_id"].astype(str).eq(scorer_id), ["symbol", "news_date"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if expected is None:
            expected = keys
        elif keys != expected:
            raise Week6PnlError("all comparison scorers must cover identical company-day keys")
    assert expected is not None
    return chronological_split_date(sorted({news_date for _, news_date in expected}), development_fraction)


def development_quantile_thresholds(
    signals: pd.DataFrame,
    *,
    split_date: str,
    quantiles: tuple[float, ...],
) -> pd.DataFrame:
    """Map prespecified score quantiles to absolute gates using development only."""

    development = pd.to_numeric(
        signals.loc[signals["news_date"].astype(str) < split_date, "mean_score"],
        errors="coerce",
    ).dropna()
    values = development.abs().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise Week6PnlError("scorer has no finite development scores for threshold calibration")
    rows: list[dict[str, float]] = []
    for quantile in sorted(set(quantiles)):
        threshold = 0.0 if quantile == 0.0 else float(np.quantile(values, quantile, method="linear"))
        rows.append({"threshold_quantile": float(quantile), "absolute_threshold": threshold})
    return pd.DataFrame(rows)


def profitable_negative_pairs(
    correlations: pd.DataFrame,
    stock_daily: pd.DataFrame,
    *,
    min_active_days: int,
) -> pd.DataFrame:
    """Annotate correlation evidence and gate pairs on development profitability."""

    output_columns = [
        *correlations.columns,
        "symbol_a_development_net_profit",
        "symbol_b_development_net_profit",
        "symbol_a_development_active_days",
        "symbol_b_development_active_days",
        "both_profitable",
        "eligible_negative_pair",
    ]
    if correlations.empty:
        return pd.DataFrame(columns=list(dict.fromkeys(output_columns)))
    development = stock_daily.loc[stock_daily["split"].eq("development")]
    summary = development.groupby("symbol", sort=True).agg(
        development_net_profit=("net_pnl", "sum"),
        development_active_days=("active_trade", "sum"),
    )
    rows: list[dict[str, Any]] = []
    for row in correlations.to_dict("records"):
        symbol_a = str(row["symbol_a"])
        symbol_b = str(row["symbol_b"])
        profit_a = float(summary.loc[symbol_a, "development_net_profit"])
        profit_b = float(summary.loc[symbol_b, "development_net_profit"])
        active_a = int(summary.loc[symbol_a, "development_active_days"])
        active_b = int(summary.loc[symbol_b, "development_active_days"])
        correlation = row["pairwise_correlation"]
        eligible = bool(
            row["support_status"] == "supported"
            and correlation is not None
            and math.isfinite(float(correlation))
            and float(correlation) < 0
            and profit_a > 0
            and profit_b > 0
            and active_a >= min_active_days
            and active_b >= min_active_days
        )
        rows.append(
            {
                **row,
                "symbol_a_development_net_profit": profit_a,
                "symbol_b_development_net_profit": profit_b,
                "symbol_a_development_active_days": active_a,
                "symbol_b_development_active_days": active_b,
                "both_profitable": profit_a > 0 and profit_b > 0,
                "eligible_negative_pair": eligible,
            }
        )
    return pd.DataFrame(rows)


def select_negative_correlation_clique(
    pair_evidence: pd.DataFrame,
    *,
    requested_count: int | None,
) -> tuple[str, ...]:
    """Select a development-only clique whose every supported pair is negative."""

    eligible = pair_evidence.loc[pair_evidence["eligible_negative_pair"].astype(bool)].copy()
    if eligible.empty:
        return ()
    eligible = eligible.sort_values(
        ["pairwise_correlation", "jointly_active_dates", "symbol_a", "symbol_b"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    first = eligible.iloc[0]
    selected = [str(first["symbol_a"]), str(first["symbol_b"])]
    symbols = sorted(set(eligible["symbol_a"].astype(str)) | set(eligible["symbol_b"].astype(str)))
    target = requested_count if requested_count is not None else math.ceil(math.sqrt(len(symbols)))
    target = max(2, min(int(target), len(symbols)))
    lookup = {
        tuple(sorted((str(row["symbol_a"]), str(row["symbol_b"])))): float(row["pairwise_correlation"])
        for row in eligible.to_dict("records")
    }
    while len(selected) < target:
        candidates: list[tuple[float, float, str]] = []
        for symbol in symbols:
            if symbol in selected:
                continue
            values = [lookup.get(tuple(sorted((symbol, other)))) for other in selected]
            if all(value is not None and value < 0 for value in values):
                finite = [float(value) for value in values if value is not None]
                candidates.append((float(np.mean(finite)), max(finite), symbol))
        if not candidates:
            break
        selected.append(min(candidates)[2])
    return tuple(selected)


def optimize_development_weights(
    stock_daily: pd.DataFrame,
    symbols: tuple[str, ...],
    *,
    maximum_weight: float,
) -> pd.DataFrame:
    """Fit capped long-only minimum-variance weights on development returns."""

    if len(symbols) < 2:
        return pd.DataFrame(columns=["symbol", "weight", "weighting_method"])
    effective_cap = max(maximum_weight, 1.0 / len(symbols))
    development = stock_daily.loc[
        stock_daily["split"].eq("development") & stock_daily["symbol"].isin(symbols)
    ]
    matrix = (
        development.pivot(index="date", columns="symbol", values="net_return")
        .reindex(columns=list(symbols))
        .fillna(0.0)
    )
    if len(matrix) < 2:
        weights = np.repeat(1.0 / len(symbols), len(symbols))
        method = "equal_weight_fallback_insufficient_observations"
    else:
        covariance = LedoitWolf().fit(matrix.to_numpy(dtype=float)).covariance_
        initial = np.repeat(1.0 / len(symbols), len(symbols))
        result = minimize(
            lambda values: float(values @ covariance @ values),
            initial,
            method="SLSQP",
            bounds=[(0.0, effective_cap)] * len(symbols),
            constraints={"type": "eq", "fun": lambda values: float(values.sum() - 1.0)},
            options={"ftol": 1e-15, "maxiter": 1_000},
        )
        if not result.success or not np.isfinite(result.x).all():
            weights = initial
            method = "equal_weight_fallback_optimizer_failure"
        else:
            weights = np.asarray(result.x, dtype=float)
            weights /= weights.sum()
            method = "development_ledoit_wolf_capped_minimum_variance"
    return pd.DataFrame(
        {
            "symbol": list(symbols),
            "weight": weights,
            "weighting_method": method,
            "maximum_weight": effective_cap,
        }
    )


def build_weighted_portfolio_daily(
    stock_daily: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    portfolio_name: str,
    starting_capital: float,
) -> pd.DataFrame:
    """Apply frozen stock weights to both development and evaluation daily returns."""

    if weights.empty:
        return pd.DataFrame()
    mapping = dict(zip(weights["symbol"].astype(str), weights["weight"].astype(float), strict=True))
    subset = stock_daily.loc[stock_daily["symbol"].isin(mapping)].copy()
    subset["weight"] = subset["symbol"].map(mapping)
    for column in ("gross_return", "net_return", "turnover"):
        subset[f"weighted_{column}"] = subset[column].astype(float) * subset["weight"]
    daily = subset.groupby(["date", "split"], sort=True).agg(
        gross_return=("weighted_gross_return", "sum"),
        net_return=("weighted_net_return", "sum"),
        turnover=("weighted_turnover", "sum"),
        active_days=("active_trade", "sum"),
    ).reset_index()
    daily["fixed_notional_gross_pnl"] = starting_capital * daily["gross_return"]
    daily["fixed_notional_net_pnl"] = starting_capital * daily["net_return"]
    gross_value = starting_capital
    net_value = starting_capital
    running_peak = starting_capital
    cumulative_fixed_gross = 0.0
    cumulative_fixed_net = 0.0
    rows: list[dict[str, Any]] = []
    for item in daily.to_dict("records"):
        start_net = net_value
        gross_wealth_pnl = gross_value * float(item["gross_return"])
        gross_pnl = start_net * float(item["gross_return"])
        net_pnl = start_net * float(item["net_return"])
        cost_rate = max(0.0, float(item["gross_return"] - item["net_return"]))
        transaction_cost = start_net * cost_rate
        gross_value += gross_wealth_pnl
        net_value += net_pnl
        running_peak = max(running_peak, net_value)
        cumulative_fixed_gross += float(item["fixed_notional_gross_pnl"])
        cumulative_fixed_net += float(item["fixed_notional_net_pnl"])
        rows.append(
            {
                "date": item["date"],
                "split": item["split"],
                "portfolio": portfolio_name,
                "selected_stock_count": len(mapping),
                "weight_per_stock": float("nan"),
                "starting_capital": starting_capital,
                "start_portfolio_value": start_net,
                "daily_gross_return": item["gross_return"],
                "daily_net_return": item["net_return"],
                "daily_gross_pnl": gross_pnl,
                "daily_net_pnl": net_pnl,
                "transaction_cost": transaction_cost,
                "gross_wealth_pnl": gross_wealth_pnl,
                "fixed_notional_gross_pnl": item["fixed_notional_gross_pnl"],
                "fixed_notional_net_pnl": item["fixed_notional_net_pnl"],
                "fixed_notional_transaction_cost": starting_capital * cost_rate,
                "cumulative_fixed_notional_gross_pnl": cumulative_fixed_gross,
                "cumulative_fixed_notional_net_pnl": cumulative_fixed_net,
                "cumulative_gross_pnl": gross_value - starting_capital,
                "cumulative_net_pnl": net_value - starting_capital,
                "gross_portfolio_value": gross_value,
                "portfolio_value": net_value,
                "running_peak": running_peak,
                "drawdown": net_value / running_peak - 1.0,
                "turnover": item["turnover"],
                "transaction_cost_rate": cost_rate,
                "active_stock_count": int(item["active_days"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_comparison_outputs(
    portfolio_daily: pd.DataFrame,
    stock_daily: pd.DataFrame,
    correlations: pd.DataFrame,
    performance: pd.DataFrame,
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise Week6PnlError("matplotlib is required for comparison figures") from exc

    evaluation = portfolio_daily.loc[portfolio_daily["split"].eq("evaluation")].copy()
    evaluation["evaluation_equity"] = evaluation.groupby(["scorer_id", "portfolio"])["daily_net_return"].transform(
        lambda values: (1.0 + values).cumprod() * float(portfolio_daily["starting_capital"].iloc[0])
    )
    fig, axis = plt.subplots(figsize=(11, 6))
    for (scorer, portfolio), group in evaluation.groupby(["scorer_id", "portfolio"], sort=True):
        style = "-" if portfolio == "all_stock_equal_weight" else "--"
        axis.plot(pd.to_datetime(group["date"]), group["evaluation_equity"], label=f"{scorer} · {portfolio}", linestyle=style)
    axis.set(title="Frozen model strategies: evaluation equity", xlabel="Date", ylabel="Portfolio value")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "best_strategies_equity.png", dpi=180)
    plt.close(fig)

    ranked = performance.loc[
        performance["period"].eq("evaluation")
        & performance["return_variant"].eq("net")
        & performance["annualized_sharpe"].notna()
    ].sort_values("annualized_sharpe", ascending=False)
    if ranked.empty:
        return
    winner = ranked.iloc[0]
    scorer = str(winner["scorer_id"])
    selected_stock = stock_daily.loc[stock_daily["scorer_id"].eq(scorer) & stock_daily["split"].eq("evaluation")].copy()
    selected_stock["cumulative_net_pnl"] = selected_stock.groupby("symbol")["net_pnl"].cumsum()
    fig, axis = plt.subplots(figsize=(11, 6))
    for symbol, group in selected_stock.groupby("symbol", sort=True):
        axis.plot(pd.to_datetime(group["date"]), group["cumulative_net_pnl"], label=symbol, linewidth=1.0, alpha=0.8)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(title=f"Evaluation stock equity curves · {scorer}", xlabel="Date", ylabel="Cumulative net P&L")
    axis.grid(alpha=0.2)
    axis.legend(ncol=4, fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "best_strategy_stock_equity.png", dpi=180)
    plt.close(fig)

    chosen = correlations.loc[correlations["scorer_id"].eq(scorer)]
    if chosen.empty:
        return
    symbols = sorted(set(chosen["symbol_a"]) | set(chosen["symbol_b"]))
    matrix = pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
    for row in chosen.to_dict("records"):
        value = row["pairwise_correlation"]
        if value is not None and math.isfinite(float(value)):
            matrix.loc[row["symbol_a"], row["symbol_b"]] = float(value)
            matrix.loc[row["symbol_b"], row["symbol_a"]] = float(value)
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(symbols)), labels=symbols, rotation=90, fontsize=7)
    axis.set_yticks(range(len(symbols)), labels=symbols, fontsize=7)
    axis.set_title(f"Development stock P&L correlations · {scorer}")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "best_strategy_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def _write_best_strategies_report(
    path: Path,
    *,
    performance: pd.DataFrame,
    rules: pd.DataFrame,
    pairs: pd.DataFrame,
    weights: pd.DataFrame,
    split_date: str,
) -> None:
    evaluation = performance.loc[
        performance["period"].eq("evaluation") & performance["return_variant"].eq("net")
    ].copy()
    evaluation = evaluation.sort_values(
        ["annualized_sharpe", "actual_sample_percentage_return"], ascending=[False, False], na_position="last"
    )
    rule_map = rules.set_index("scorer_id").to_dict("index")
    rows = []
    for row in evaluation.to_dict("records"):
        rule = rule_map[str(row["scorer_id"])]
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["scorer_id"]),
                    str(row["portfolio"]),
                    f"{float(rule['selected_threshold_quantile']):.0%}",
                    f"{float(rule['selected_absolute_threshold']):.4f}",
                    str(int(rule["selected_holding_period"])),
                    f"£{float(row['sample_total_profit']):,.2f}",
                    f"{float(row['actual_sample_percentage_return']):.3f}%",
                    f"{float(row['annualized_percentage_return']):.3f}%",
                    "n/a" if pd.isna(row["annualized_sharpe"]) else f"{float(row['annualized_sharpe']):.3f}",
                    f"{float(row['maximum_drawdown']):.3f}%",
                    str(int(row["active_days"])),
                ]
            )
            + " |"
        )
    eligible_pairs = pairs.loc[pairs["eligible_negative_pair"].astype(bool)] if not pairs.empty else pairs
    pair_lines = []
    for row in eligible_pairs.sort_values("pairwise_correlation").head(20).to_dict("records"):
        pair_lines.append(
            f"| {row['scorer_id']} | {row['symbol_a']} | {row['symbol_b']} | "
            f"{float(row['pairwise_correlation']):.3f} | {int(row['jointly_active_dates'])} | "
            f"£{float(row['symbol_a_development_net_profit']):,.2f} | £{float(row['symbol_b_development_net_profit']):,.2f} |"
        )
    selected_lines = []
    for scorer, group in weights.groupby("scorer_id", sort=True):
        allocation = ", ".join(f"{row.symbol} {row.weight:.1%}" for row in group.itertuples())
        selected_lines.append(f"- `{scorer}`: {allocation} ({group['weighting_method'].iloc[0]}).")
    best = evaluation.iloc[0] if not evaluation.empty else None
    headline = (
        f"The highest evaluation Sharpe is **{float(best['annualized_sharpe']):.3f}** for "
        f"`{best['scorer_id']}` / `{best['portfolio']}`, with "
        f"£{float(best['sample_total_profit']):,.2f} net profit."
        if best is not None and pd.notna(best["annualized_sharpe"])
        else "No strategy has a finite evaluation Sharpe."
    )
    pair_section = (
        "\n".join(
            [
                "| Scorer | Stock A | Stock B | Correlation | Joint active days | A development profit | B development profit |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
                *pair_lines,
            ]
        )
        if pair_lines
        else "No pair passed all profitability, negative-correlation, and activity-support gates."
    )
    weight_section = (
        "\n".join(selected_lines)
        if selected_lines
        else "No scorer had a supported profitable negative-correlation pair, so no combined portfolio was forced."
    )
    performance_header = (
        "| Scorer | Portfolio | Gate quantile | Absolute gate | Hold | Net profit | Sample return | "
        "Annual return | Sharpe | Max drawdown | Active days |"
    )
    text = f"""# Week 6 Best Strategies

## Headline result

{headline}

All thresholds, holding periods, eligible negative-correlation stocks, and portfolio
weights were selected using dates before `{split_date}`. Ranking by evaluation Sharpe
is descriptive and must not be used to retune this run.

## Frozen evaluation performance

{performance_header}
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Profitable supported negative-correlation pairs

{pair_section}

## Frozen negative-correlation portfolios

{weight_section}

## Interpretation guardrails

- Daily funded returns drive Sharpe, annual return, and drawdown.
- Per-stock Sharpe should be read beside active-day counts; sparse winners are unstable.
- Score quantiles are calibrated separately because scorer magnitudes are not directly comparable.
- This six-month corpus and its scorer outputs were already explored; the evaluation split is
  leak-controlled within this comparison, but it is not a pristine confirmatory holdout.
- Results are exploratory, cost-aware price-return simulations, not causal evidence or investment advice.
"""
    path.write_text(text, encoding="utf-8")


def _write_best_strategies_html(
    path: Path,
    *,
    performance: pd.DataFrame,
    rules: pd.DataFrame,
    pairs: pd.DataFrame,
    weights: pd.DataFrame,
    split_date: str,
) -> None:
    """Write a local presentation page with aggregate-only comparison evidence."""

    evaluation = performance.loc[
        performance["period"].eq("evaluation") & performance["return_variant"].eq("net")
    ].copy()
    evaluation = evaluation.sort_values("annualized_sharpe", ascending=False, na_position="last").reset_index(drop=True)
    rule_columns = rules[
        [
            "scorer_id",
            "selected_threshold_quantile",
            "selected_absolute_threshold",
            "selected_holding_period",
            "negative_portfolio_status",
        ]
    ].copy()
    rule_columns["selected_threshold_quantile"] = rule_columns["selected_threshold_quantile"].map(
        lambda value: f"{float(value):.0%}"
    )
    rule_columns["selected_absolute_threshold"] = rule_columns["selected_absolute_threshold"].map(
        lambda value: f"{float(value):.4f}"
    )
    display = evaluation[
        [
            "scorer_id",
            "portfolio",
            "sample_total_profit",
            "actual_sample_percentage_return",
            "annualized_percentage_return",
            "annualized_sharpe",
            "maximum_drawdown",
            "active_days",
        ]
    ].copy()
    display.insert(0, "rank", np.arange(1, len(display) + 1))
    display["sample_total_profit"] = display["sample_total_profit"].map(lambda value: f"£{float(value):,.2f}")
    for column in ("actual_sample_percentage_return", "annualized_percentage_return", "maximum_drawdown"):
        display[column] = display[column].map(lambda value: f"{float(value):.3f}%")
    display["annualized_sharpe"] = display["annualized_sharpe"].map(
        lambda value: "n/a" if pd.isna(value) else f"{float(value):.3f}"
    )
    eligible = pairs.loc[pairs["eligible_negative_pair"].astype(bool)].copy() if not pairs.empty else pairs
    pair_display = eligible[
        ["scorer_id", "symbol_a", "symbol_b", "pairwise_correlation", "jointly_active_dates"]
    ].sort_values("pairwise_correlation").head(20)
    if not pair_display.empty:
        pair_display["pairwise_correlation"] = pair_display["pairwise_correlation"].map(lambda value: f"{float(value):.3f}")
    weight_display = weights[["scorer_id", "symbol", "weight", "weighting_method"]].copy()
    if not weight_display.empty:
        weight_display["weight"] = weight_display["weight"].map(lambda value: f"{float(value):.1%}")
    best = evaluation.iloc[0] if not evaluation.empty else None
    headline = (
        f"Highest evaluation Sharpe: {float(best['annualized_sharpe']):.3f} · "
        f"{best['scorer_id']} · {best['portfolio']}"
        if best is not None and pd.notna(best["annualized_sharpe"])
        else "No finite evaluation Sharpe"
    )
    table_options = {"index": False, "classes": "data", "border": 0, "justify": "left"}
    pair_html = pair_display.to_html(**table_options) if not pair_display.empty else "<p>No eligible negative pair.</p>"
    weight_html = weight_display.to_html(**table_options) if not weight_display.empty else "<p>No combined portfolio was forced.</p>"
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Week 6 Best Strategies</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1180px; padding: 32px 24px 64px; line-height: 1.45; }}
h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 38px; }}
.kicker {{ color: #68707a; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; }}
.headline {{ background: color-mix(in srgb, #2f80ed 12%, transparent); border-left: 4px solid #2f80ed;
  border-radius: 8px; font-size: 20px; margin: 22px 0; padding: 18px 20px; }}
.note {{ color: #68707a; max-width: 900px; }}
.data {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
.data th, .data td {{ border-bottom: 1px solid color-mix(in srgb, currentColor 14%, transparent); padding: 9px 8px; }}
.data th {{ position: sticky; top: 0; text-align: left; }}
.data tbody tr:first-child {{ background: color-mix(in srgb, #2f80ed 10%, transparent); font-weight: 650; }}
.figure-grid {{ display: grid; gap: 22px; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }}
figure {{ margin: 0; }} img {{ background: white; border-radius: 10px; max-width: 100%; }}
figcaption {{ color: #68707a; font-size: 13px; margin-top: 6px; }}
@media (max-width: 620px) {{ body {{ padding: 20px 12px; }} .figure-grid {{ grid-template-columns: 1fr; }}
  .data {{ display: block; overflow-x: auto; }} }}
</style>
</head>
<body>
<div class="kicker">Sentiment dissertation · funded Week 6 comparison</div>
<h1>Best Strategies</h1>
<p class="note">Frozen evaluation begins {split_date}. Strategy gates, holding periods, stock eligibility,
and portfolio weights use development data only. Evaluation ranking is descriptive and is not a retuning rule.</p>
<div class="headline">{headline}</div>
<h2>Evaluation ranking</h2>
{display.to_html(**table_options)}
<h2>Frozen model rules</h2>
{rule_columns.to_html(**table_options)}
<h2>Equity curves</h2>
<div class="figure-grid">
<figure><img src="best_strategies_equity.png" alt="Evaluation equity curves">
<figcaption>All frozen scorer and portfolio cases.</figcaption></figure>
<figure><img src="best_strategy_stock_equity.png" alt="Stock equity curves">
<figcaption>Per-stock net P&amp;L for the leading scorer.</figcaption></figure>
</div>
<h2>Profitable negative-correlation evidence</h2>
<p class="note">Only supported pairs with two profitable stocks and adequate development activity appear.</p>
{pair_html}
<h2>Frozen combined-portfolio weights</h2>
{weight_html}
<h2>Correlation audit</h2>
<figure><img src="best_strategy_correlation_heatmap.png" alt="Development P&amp;L correlation heatmap">
<figcaption>Development correlations for the leading scorer; unsupported sparse pairs remain descriptive.</figcaption></figure>
<h2>Limits</h2>
<p class="note">Daily funded returns drive Sharpe, annual return, and drawdown. Sparse stock results are unstable.
This six-month corpus and its scorer outputs were already explored, so the evaluation split is
leak-controlled within this comparison but is not a pristine confirmatory holdout. This is exploratory,
cost-aware price-return evidence, not causal evidence or investment advice.</p>
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def run_week6_model_comparison(
    signals_path: str | Path,
    prices_path: str | Path,
    output_root: str | Path,
    config: Week6ModelComparisonConfig,
    *,
    command: str | None = None,
    repo_root: str | Path | None = None,
    generated_at: str | None = None,
) -> Week6ModelComparisonResult:
    """Run an immutable development-tuned comparison and one untouched evaluation."""

    _validate_config(config)
    signal_file = Path(signals_path).expanduser().resolve()
    price_file = Path(prices_path).expanduser().resolve()
    output_base = Path(output_root).expanduser().resolve()
    final_dir = output_base / config.run_id
    temporary_dir = output_base / f".{config.run_id}.tmp"
    if final_dir.exists() or temporary_dir.exists():
        raise Week6PnlError(f"refusing to overwrite existing model comparison: {final_dir}")
    output_base.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        split_date = common_comparison_split(signal_file, config.scorer_ids, config.development_fraction)
        all_stock_frames: list[pd.DataFrame] = []
        portfolio_frames: list[pd.DataFrame] = []
        stock_performance_frames: list[pd.DataFrame] = []
        portfolio_performance_frames: list[pd.DataFrame] = []
        grid_frames: list[pd.DataFrame] = []
        correlation_frames: list[pd.DataFrame] = []
        pair_frames: list[pd.DataFrame] = []
        weight_frames: list[pd.DataFrame] = []
        manual_frames: list[pd.DataFrame] = []
        rule_rows: list[dict[str, Any]] = []
        alignment_rows: list[dict[str, Any]] = []

        for scorer_id in config.scorer_ids:
            signals, prices = load_and_validate_inputs(signal_file, price_file, scorer_id)
            calibration = development_quantile_thresholds(
                signals,
                split_date=split_date,
                quantiles=config.threshold_quantiles,
            )
            threshold_to_quantile = (
                calibration.sort_values("threshold_quantile")
                .drop_duplicates("absolute_threshold", keep="first")
                .set_index("absolute_threshold")["threshold_quantile"]
                .to_dict()
            )
            thresholds = tuple(sorted(float(value) for value in threshold_to_quantile))
            scorer_config = Week6PnlConfig(
                run_id=config.run_id,
                scorer_id=scorer_id,
                starting_capital=config.starting_capital,
                transaction_cost_bps_per_side=config.transaction_cost_bps_per_side,
                development_fraction=config.development_fraction,
                thresholds=thresholds,
                holding_periods=config.holding_periods,
                annualization_periods=config.annualization_periods,
                annual_risk_free_rate=config.annual_risk_free_rate,
                exchange_timezone=config.exchange_timezone,
                min_joint_active_dates=config.min_joint_active_dates,
            )
            grid, candidate_frames, candidate_counts = build_strategy_grid(
                signals,
                prices,
                config=scorer_config,
                split_date=split_date,
            )
            threshold, holding, grid = select_stable_strategy(grid)
            threshold_quantile = float(threshold_to_quantile[threshold])
            grid["scorer_id"] = scorer_id
            grid["threshold_quantile"] = grid["threshold"].map(threshold_to_quantile)
            grid_frames.append(grid)
            stock_daily = candidate_frames[(threshold, holding)].copy()
            alignment: AlignmentCounts = candidate_counts[(threshold, holding)]
            stock_daily["scorer_id"] = scorer_id
            all_stock_frames.append(stock_daily)
            correlations = development_correlations(
                stock_daily,
                min_joint_active_dates=config.min_joint_active_dates,
            )
            correlations["scorer_id"] = scorer_id
            correlation_frames.append(correlations)
            pairs = profitable_negative_pairs(
                correlations,
                stock_daily,
                min_active_days=config.min_stock_active_days,
            )
            pairs["scorer_id"] = scorer_id
            pair_frames.append(pairs)
            selected_symbols = select_negative_correlation_clique(
                pairs,
                requested_count=config.negative_portfolio_stock_count,
            )
            weights = optimize_development_weights(
                stock_daily,
                selected_symbols,
                maximum_weight=config.maximum_stock_weight,
            )
            if not weights.empty:
                weights["scorer_id"] = scorer_id
                weight_frames.append(weights)
            all_symbols = sorted(stock_daily["symbol"].unique())
            portfolio_daily = build_portfolio_daily(
                stock_daily,
                {"all_stock_equal_weight": all_symbols},
                starting_capital=config.starting_capital,
            )
            weighted = build_weighted_portfolio_daily(
                stock_daily,
                weights,
                portfolio_name="development_negative_correlation",
                starting_capital=config.starting_capital,
            )
            if not weighted.empty:
                portfolio_daily = pd.concat([portfolio_daily, weighted], ignore_index=True)
            portfolio_daily["scorer_id"] = scorer_id
            portfolio_frames.append(portfolio_daily)
            stock_performance = build_stock_performance(stock_daily, scorer_config)
            stock_performance["scorer_id"] = scorer_id
            stock_performance_frames.append(stock_performance)
            portfolio_performance = build_portfolio_performance(portfolio_daily, scorer_config)
            portfolio_performance["scorer_id"] = scorer_id
            portfolio_performance_frames.append(portfolio_performance)
            manual = build_manual_timing_check(
                stock_daily,
                cost_rate=config.transaction_cost_bps_per_side / 10_000.0,
            )
            manual["scorer_id"] = scorer_id
            manual_frames.append(manual)
            rule_rows.append(
                {
                    "scorer_id": scorer_id,
                    "selected_threshold_quantile": threshold_quantile,
                    "selected_absolute_threshold": threshold,
                    "selected_holding_period": holding,
                    "selection_data": "development_only",
                    "negative_portfolio_status": "selected" if selected_symbols else "no_eligible_pair",
                    "negative_portfolio_symbols": "|".join(selected_symbols),
                }
            )
            alignment_rows.append({"scorer_id": scorer_id, **asdict(alignment)})

        stock_daily_all = pd.concat(all_stock_frames, ignore_index=True)
        portfolio_daily_all = pd.concat(portfolio_frames, ignore_index=True)
        stock_performance_all = pd.concat(stock_performance_frames, ignore_index=True)
        portfolio_performance_all = pd.concat(portfolio_performance_frames, ignore_index=True)
        grid_all = pd.concat(grid_frames, ignore_index=True)
        correlations_all = pd.concat(correlation_frames, ignore_index=True)
        pairs_all = pd.concat(pair_frames, ignore_index=True)
        weights_all = (
            pd.concat(weight_frames, ignore_index=True)
            if weight_frames
            else pd.DataFrame(columns=["symbol", "weight", "weighting_method", "maximum_weight", "scorer_id"])
        )
        manual_all = pd.concat(manual_frames, ignore_index=True)
        rules = pd.DataFrame(rule_rows)
        alignments = pd.DataFrame(alignment_rows)

        outputs = {
            "model_stock_daily_pnl.csv": stock_daily_all,
            "model_portfolio_daily_pnl.csv": portfolio_daily_all,
            "model_stock_performance.csv": stock_performance_all,
            "model_portfolio_performance.csv": portfolio_performance_all,
            "model_strategy_grid.csv": grid_all,
            "model_pnl_correlations.csv": correlations_all,
            "profitable_negative_pairs.csv": pairs_all,
            "selected_negative_portfolio_weights.csv": weights_all,
            "selected_model_rules.csv": rules,
            "alignment_counts.csv": alignments,
            "manual_timing_checks.csv": manual_all,
        }
        for name, frame in outputs.items():
            _write_csv(frame, temporary_dir / name)
        _plot_comparison_outputs(
            portfolio_daily_all,
            stock_daily_all,
            correlations_all,
            portfolio_performance_all,
            temporary_dir,
        )
        _write_best_strategies_report(
            temporary_dir / "best_strategies.md",
            performance=portfolio_performance_all,
            rules=rules,
            pairs=pairs_all,
            weights=weights_all,
            split_date=split_date,
        )
        _write_best_strategies_html(
            temporary_dir / "best_strategies.html",
            performance=portfolio_performance_all,
            rules=rules,
            pairs=pairs_all,
            weights=weights_all,
            split_date=split_date,
        )
        root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
        output_hashes = {
            path.name: _sha256(path)
            for path in sorted(temporary_dir.iterdir(), key=lambda value: value.name)
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "schema_version": 1,
            "run_id": config.run_id,
            "generated_at": generated_at or datetime.now(UTC).isoformat(),
            "git": _git_metadata(root),
            "command": command or shlex.join(["sentiment-bench", "compare-week6-models"]),
            "inputs": {
                "daily_signals": {"path": str(signal_file), "sha256": _sha256(signal_file)},
                "prices": {"path": str(price_file), "sha256": _sha256(price_file)},
            },
            "configuration": asdict(config),
            "development_evaluation": {"split_date": split_date},
            "selection_contract": {
                "strategy": "threshold quantile and holding selected on development stability only",
                "negative_pairs": "supported, both development-profitable, negative correlation, minimum activity",
                "weights": "development Ledoit-Wolf capped long-only minimum variance",
                "evaluation": "frozen rules and weights evaluated once",
            },
            "selected_rules": rule_rows,
            "alignment_counts": alignment_rows,
            "output_hashes": output_hashes,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_dir.rename(final_dir)
        selected = tuple(
            (
                str(row["scorer_id"]),
                float(row["selected_threshold_quantile"]),
                float(row["selected_absolute_threshold"]),
                int(row["selected_holding_period"]),
            )
            for row in rule_rows
        )
        return Week6ModelComparisonResult(final_dir, split_date, selected)
    except Exception:
        if temporary_dir.exists():
            for path in temporary_dir.iterdir():
                if path.is_file():
                    path.unlink()
            temporary_dir.rmdir()
        raise
