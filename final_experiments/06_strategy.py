# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 06 — The signals as a trading strategy
#
# Notebook 03 asked whether the firm-day signals rank next-session returns.
# This one asks what happens to an account that trades them: equity curve,
# drawdown, per-year behaviour, and the cost at which the whole thing dies.
#
# Two levers that 03 never touched, and that decide whether its economic null is
# real or an artefact of how it traded:
#
# - **Holding horizon.** 03 held for one session and paid ~0.7× turnover a day.
#   A horizon-h strategy runs h overlapping tranches, one opened per session,
#   each 1/h of gross — so it trades roughly a fifth as much at h=5.
# - **Breadth.** 03 traded every name with news (`threshold=0`). Concentrating
#   into the tails of each day's cross-section trades less and, if the signal is
#   monotone, trades better.
#
# ## Declarations — fixed before anything below is computed
#
# - **Universe/spine:** FNSPID news-bearing firm-days, 570 priced symbols.
# - **Return:** exact next-exchange-session dense market-adjusted open-to-open,
#   from the price archive, so
#   positions can be held through sessions with no news. Same convention as
#   `ar_open_h1`, extended off the news panel.
# - **Positions:** within-session centred rank of the oriented signal, weights
#   proportional, gross exposure 1, dollar-neutral by construction.
# - **Grid:** 9 signals × horizons {1, 2, 3, 5, 10} × breadth {0.0, 0.5, 0.8}.
# - **Costs:** 10 bps per side throughout. Turnover is half-L1, so costs are
#   charged on both sides as `2 × turnover × per-side rate`; a cost curve follows.
# - **Selection rule (declared now, applied once):** highest **development** net
#   Sharpe after costs, among cells with ≥250 sessions and ≥5 average positions;
#   ties break on lower turnover. Written to `frozen_spec.json` **before** the
#   evaluation block is read.
# - **Evaluation:** one run of the frozen spec. No re-selection, no second look,
#   whatever it returns is the answer.
# - **Inference:** circular block bootstrap on daily net return, block 10, 999
#   reps, seed 20260731.
#
# The evaluation block was already opened by `04_surprise`, so it is not a
# pristine holdout. It is a chronological evaluation block and the strategy
# result inherits that caveat.

# %%
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / "pyproject.toml").exists():
    raise RuntimeError("run from repo root or final_experiments/")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.aggregators import AGGREGATOR_NAMES  # noqa: E402
from final_experiments.lib.evaluate import DEFAULT_SIGNAL_ORIENT  # noqa: E402
from final_experiments.lib.panel import (  # noqa: E402
    DEFAULT_PRICE_ZIP,
    FROZEN_DEV_END,
    FROZEN_EVAL_START,
    load_adjusted_open_prices,
)
from final_experiments.lib.plots import diverging_cmap  # noqa: E402
from final_experiments.lib.strategy import (  # noqa: E402
    DEFAULT_BREADTHS,
    DEFAULT_HORIZONS,
    StrategyConfig,
    build_daily_ar_panel,
    cost_curve,
    run_backtest,
    run_frozen_spec,
    select_frozen_spec,
    sweep_development,
    yearly_table,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover

    def display(obj: object) -> None:
        print(obj)


plt.rcParams.update(
    {
        "figure.figsize": (9.5, 4.2),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    }
)

OUTPUT_DIR = REPO_ROOT / "final_experiments" / "outputs" / "06_strategy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGG_PATH = REPO_ROOT / "final_experiments" / "outputs" / "03_aggregation" / "firm_day_aggregators.parquet"
SPEC_PATH = OUTPUT_DIR / "frozen_spec.json"

COST_BPS = 10.0
MIN_NAMES = 5
MIN_SESSIONS = 250
MIN_POSITIONS = 5.0

print(f"horizons={DEFAULT_HORIZONS}  breadths={DEFAULT_BREADTHS}  cost={COST_BPS} bps/side")
print(f"split: development ≤ {FROZEN_DEV_END}; evaluation ≥ {FROZEN_EVAL_START}")

# %% [markdown]
# ## 0. Dense return panel
#
# Positions survive sessions on which a name has no news, so returns cannot come
# from the news panel. `missing_weight_share` reports any held weight whose
# return is unavailable rather than silently treating it as a flat day.

# %%
agg = pd.read_parquet(AGG_PATH)
agg["session_date"] = pd.to_datetime(agg["session_date"]).dt.normalize()
symbols = set(agg["symbol"].astype(str))
prices, missing_prices = load_adjusted_open_prices(DEFAULT_PRICE_ZIP, symbols, market_symbol="SPY")
daily_ar = build_daily_ar_panel(prices, market_symbol="SPY")

print(f"signal rows      : {len(agg):,}")
print(f"dense AR rows    : {len(daily_ar):,}  symbols={daily_ar['symbol'].nunique():,}")
print(f"AR sessions      : {daily_ar['session_date'].nunique():,}")
print(f"missing prices   : {missing_prices}")

# %% [markdown]
# ## 1. Development sweep
#
# 9 signals × 5 horizons × 3 breadth bands = 135 cells, development rows only.
# The evaluation block is untouched here.

# %%
sweep = sweep_development(
    agg,
    daily_ar,
    signals=AGGREGATOR_NAMES,
    horizons=DEFAULT_HORIZONS,
    breadths=DEFAULT_BREADTHS,
    orients=DEFAULT_SIGNAL_ORIENT,
    cost_bps_per_side=COST_BPS,
    min_names=MIN_NAMES,
)
sweep.to_csv(OUTPUT_DIR / "development_sweep.csv", index=False)
print(f"cells: {len(sweep)}")
display(
    sweep.sort_values("sharpe_net", ascending=False)
    .head(15)[
        [
            "signal", "horizon", "breadth", "sharpe_net", "sharpe_gross",
            "mean_net", "annualized_turnover", "breakeven_bps_per_side",
            "max_drawdown", "mean_positions", "n_sessions",
        ]
    ]
    .round(4)
)

# %% [markdown]
# ### Where the levers actually bite
#
# Net Sharpe as a surface over horizon and breadth. If the h=1 economic null in
# 03 was a construction artefact, it shows up here as a gradient.

# %%
top_signals = (
    sweep.groupby("signal")["sharpe_net"].max().sort_values(ascending=False).head(6).index.tolist()
)
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
vmax = float(np.nanmax(np.abs(sweep["sharpe_net"])))
# Diverging scale with a NEUTRAL GREY midpoint. A yellow or green centre (as in
# RdYlGn) puts a colour at zero, which makes "no effect" read as a value.
cmap = diverging_cmap()
for ax, signal in zip(axes.ravel(), top_signals, strict=False):
    pivot = (
        sweep.loc[sweep["signal"] == signal]
        .pivot(index="breadth", columns="horizon", values="sharpe_net")
        .sort_index()
    )
    im = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{b:g}" for b in pivot.index])
    ax.set_title(signal, fontsize=10)
    ax.set_xlabel("horizon (sessions)")
    ax.set_ylabel("breadth")
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.to_numpy()[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="development net Sharpe")
fig.suptitle("Development net Sharpe by holding horizon and breadth (10 bps/side)", y=0.98)
fig.savefig(OUTPUT_DIR / "development_sharpe_surface.png", dpi=140, bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for horizon in DEFAULT_HORIZONS:
    sub = sweep.loc[(sweep["horizon"] == horizon) & (sweep["breadth"] == 0.0)]
    axes[0].scatter(sub["annualized_turnover"], sub["breakeven_bps_per_side"], label=f"h={horizon}", s=45)
axes[0].axhline(COST_BPS, color="#8c3d3d", ls="--", lw=1.2, label=f"{COST_BPS:g} bps charged")
axes[0].set_xlabel("annualised turnover (× per year)")
axes[0].set_ylabel("break-even cost (bps/side)")
axes[0].set_title("Longer holds trade less — does break-even rise?")
axes[0].legend(frameon=False, fontsize=8)

best_by_cell = sweep.groupby(["horizon", "breadth"])["breakeven_bps_per_side"].max().reset_index()
for breadth in DEFAULT_BREADTHS:
    sub = best_by_cell.loc[best_by_cell["breadth"] == breadth]
    axes[1].plot(sub["horizon"], sub["breakeven_bps_per_side"], marker="o", label=f"breadth {breadth:g}")
axes[1].axhline(COST_BPS, color="#8c3d3d", ls="--", lw=1.2, label=f"{COST_BPS:g} bps charged")
axes[1].set_xlabel("horizon (sessions)")
axes[1].set_ylabel("best break-even across signals (bps/side)")
axes[1].set_title("Best achievable break-even per cell")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "development_breakeven_vs_horizon.png", dpi=140)
plt.show()

viable = sweep.loc[sweep["breakeven_bps_per_side"].astype(float) > COST_BPS]
print(
    f"Development cells clearing {COST_BPS:g} bps/side: {len(viable)} of {len(sweep)}"
    + (f"\n  {viable['signal'].value_counts().to_dict()}" if len(viable) else "")
)

# %% [markdown]
# ## 2. Apply the declared selection rule, and freeze it
#
# Highest development net Sharpe after costs, ≥250 sessions, ≥5 average
# positions, ties to lower turnover. Written to disk **before** §3 runs.

# %%
spec = select_frozen_spec(
    sweep, criterion="sharpe_net", min_sessions=MIN_SESSIONS, min_positions=MIN_POSITIONS
)
spec.update(
    {
        "orient": float(DEFAULT_SIGNAL_ORIENT.get(spec["signal"], 1.0)),
        "cost_bps_per_side": COST_BPS,
        "min_names": MIN_NAMES,
        "position_rule": "within-session centred rank, gross 1, dollar-neutral",
        "return_convention": "dense market-adjusted open-to-open",
        "split_boundary_rule": "formation and every held return remain inside the named split",
        "grid": {"horizons": list(DEFAULT_HORIZONS), "breadths": list(DEFAULT_BREADTHS)},
        "frozen_before_evaluation_read": True,
    }
)
SPEC_PATH.write_text(json.dumps(spec, indent=2, default=str) + "\n")
print(json.dumps(spec, indent=2, default=str))
print(f"\nfrozen → {SPEC_PATH}")

# %% [markdown]
# ## 3. The single evaluation run
#
# One call, the frozen spec, 2020-01-01 onward. Whatever this returns is the
# answer; there is no second draw.

# %%
spec_on_disk = json.loads(SPEC_PATH.read_text())
evaluation = run_frozen_spec(
    agg,
    daily_ar,
    spec_on_disk,
    orients=DEFAULT_SIGNAL_ORIENT,
    cost_bps_per_side=COST_BPS,
    min_names=MIN_NAMES,
    split="evaluation",
)
development = run_backtest(
    agg,
    daily_ar,
    StrategyConfig(
        signal=spec_on_disk["signal"],
        horizon=int(spec_on_disk["horizon"]),
        breadth=float(spec_on_disk["breadth"]),
        orient=float(DEFAULT_SIGNAL_ORIENT.get(spec_on_disk["signal"], 1.0)),
        cost_bps_per_side=COST_BPS,
        min_names=MIN_NAMES,
    ),
    split="development",
)

compare = pd.DataFrame([development.summary, evaluation.summary], index=["development", "evaluation"])
show = [
    "n_sessions", "years", "mean_net", "cagr_net", "ann_vol", "sharpe_gross", "sharpe_net",
    "max_drawdown", "hit_rate", "annualized_turnover", "breakeven_bps_per_side",
    "mean_positions", "mean_net_exposure", "max_missing_weight_share",
    "bootstrap_net_ci_low", "bootstrap_net_ci_high",
]
display(compare[show].T.round(5))
compare.to_csv(OUTPUT_DIR / "development_vs_evaluation.csv")
evaluation.daily.to_csv(OUTPUT_DIR / "evaluation_daily.csv", index=False)
development.daily.to_csv(OUTPUT_DIR / "development_daily.csv", index=False)

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={"height_ratios": [2, 1]})
for label, result, colour in (
    ("development", development, "#2f5d8c"),
    ("evaluation", evaluation, "#8a5a2b"),
):
    daily = result.daily
    if daily.empty:
        continue
    equity = np.cumprod(1.0 + daily["net_return"].to_numpy(dtype=float))
    axes[0].plot(daily["session_date"], equity, color=colour, lw=1.4, label=f"{label} (net of {COST_BPS:g} bps)")
    gross_equity = np.cumprod(1.0 + daily["gross_return"].to_numpy(dtype=float))
    axes[0].plot(daily["session_date"], gross_equity, color=colour, lw=1.0, ls=":", alpha=0.7, label=f"{label} gross")
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    axes[1].fill_between(daily["session_date"], drawdown, 0.0, color=colour, alpha=0.4, label=label)

axes[0].axhline(1.0, color="#666", lw=1)
axes[0].axvline(pd.Timestamp(FROZEN_EVAL_START), color="#333", ls="--", lw=1)
axes[0].set_ylabel("equity (start = 1.0)")
axes[0].set_title(
    f"{spec_on_disk['signal']} — horizon {spec_on_disk['horizon']}, breadth {spec_on_disk['breadth']:g}"
)
axes[0].legend(frameon=False, fontsize=8, ncol=2)
axes[1].axvline(pd.Timestamp(FROZEN_EVAL_START), color="#333", ls="--", lw=1)
axes[1].set_ylabel("drawdown")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "equity_and_drawdown.png", dpi=140)
plt.show()

# %% [markdown]
# ### Per year
#
# A strategy that only works in one regime is not a strategy. 2020 in particular
# is a volatility outlier and should be readable on its own.

# %%
years = pd.concat(
    [
        yearly_table(development.daily).assign(split="development"),
        yearly_table(evaluation.daily).assign(split="evaluation"),
    ],
    ignore_index=True,
)
display(years.round(4))
years.to_csv(OUTPUT_DIR / "yearly_performance.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4.2))
colours = ["#2f5d8c" if s == "development" else "#8a5a2b" for s in years["split"]]
ax.bar(years["year"], years["net_return"], color=colours)
ax.axhline(0.0, color="#666", lw=1)
ax.set_ylabel(f"net return (after {COST_BPS:g} bps/side)")
ax.set_title("Per-year net return — blue = development (in-sample), brown = evaluation")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "yearly_net_return.png", dpi=140)
plt.show()

# %% [markdown]
# ## 4. What cost kills it
#
# Break-even is one number; the whole curve is more honest.

# %%
frozen_config = StrategyConfig(
    signal=spec_on_disk["signal"],
    horizon=int(spec_on_disk["horizon"]),
    breadth=float(spec_on_disk["breadth"]),
    orient=float(DEFAULT_SIGNAL_ORIENT.get(spec_on_disk["signal"], 1.0)),
    cost_bps_per_side=COST_BPS,
    min_names=MIN_NAMES,
)
curves = pd.concat(
    [
        cost_curve(agg, daily_ar, frozen_config, split="development").assign(split="development"),
        cost_curve(agg, daily_ar, frozen_config, split="evaluation").assign(split="evaluation"),
    ],
    ignore_index=True,
)
display(curves.round(4))
curves.to_csv(OUTPUT_DIR / "cost_curve.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 4.4))
for split, colour in (("development", "#2f5d8c"), ("evaluation", "#8a5a2b")):
    sub = curves.loc[curves["split"] == split]
    ax.plot(sub["cost_bps_per_side"], sub["sharpe_net"], marker="o", color=colour, label=split)
ax.axhline(0.0, color="#666", lw=1)
ax.axvline(COST_BPS, color="#8c3d3d", ls="--", lw=1.2, label=f"{COST_BPS:g} bps charged")
ax.set_xlabel("cost charged (bps per side)")
ax.set_ylabel("net Sharpe")
ax.set_title("Net Sharpe against trading cost")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cost_curve.png", dpi=140)
plt.show()

# %% [markdown]
# ## 5. Verdict

# %%
ev = evaluation.summary
dv = development.summary
survives = bool(ev["bootstrap_net_ci_low"] > 0)
print(f"Frozen spec: {spec_on_disk['signal']}  horizon={spec_on_disk['horizon']}  breadth={spec_on_disk['breadth']:g}")
print(f"  development net Sharpe {dv['sharpe_net']:+.3f} | evaluation net Sharpe {ev['sharpe_net']:+.3f}")
print(f"  evaluation mean daily net {ev['mean_net']:+.6f}  95% CI [{ev['bootstrap_net_ci_low']:+.6f}, {ev['bootstrap_net_ci_high']:+.6f}]")
print(f"  evaluation max drawdown {ev['max_drawdown']:.1%}   hit rate {ev['hit_rate']:.1%}")
print(f"  break-even {ev['breakeven_bps_per_side']:.2f} bps/side vs {COST_BPS:g} charged")
print()
print(
    "VERDICT: evaluation net return is positive with a bootstrap CI excluding zero."
    if survives
    else "VERDICT: the strategy does not make money net of costs on the evaluation block."
)

manifest = {
    "notebook": "06_strategy",
    "built_for": "Trading-strategy test of the W3 firm-day signals",
    "frozen_spec": spec_on_disk,
    "selection_split": "development",
    "evaluation_runs": 1,
    "development": dv,
    "evaluation": ev,
    "cost_bps_per_side": COST_BPS,
    "grid_cells": int(len(sweep)),
    "development_cells_clearing_cost": int(len(viable)),
    "evaluation_net_ci_excludes_zero": survives,
    "caveat": (
        "Evaluation block was already opened by 04_surprise; it is a chronological "
        "evaluation block, not a pristine holdout."
    ),
    "analysis_status": "exploratory_repaired_2026-07-31",
    "supersedes": "outputs/superseded/20260731_pre_rigour_repair/06_strategy",
}
(OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

# %% [markdown]
# ### Carry-forward
#
# 1. The frozen spec and its selection rule are on disk; any re-run must reuse
#    `frozen_spec.json` rather than re-selecting.
# 2. Report the development→evaluation Sharpe drop honestly — it is the size of
#    the selection effect on a 135-cell grid.
# 3. The cost curve, not the single break-even number, is the figure to promote.
