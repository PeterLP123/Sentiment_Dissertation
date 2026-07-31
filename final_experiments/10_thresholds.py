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
# # 10 — Learned trade/no-trade thresholds
#
# Learn a gate for the W3 survivor (`negative_share`), not a return forecast.
# The direction remains the within-session rank of economically oriented
# negative share. The target is whether that direction earns a positive h1
# abnormal return.
#
# **Chronology.** Fit through 2017; choose the fixed band and probability cutoff
# on 2018–2019; disclose 2020–2023 once per frozen arm. Costs are 10 bps/side;
# evaluation net-mean intervals use a 20-session circular block bootstrap
# (999 replications, seed 20260731).

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.thresholds import (  # noqa: E402
    FEATURES,
    MIN_ACTIVE_SESSION_SHARE,
    MIN_MEAN_ACTIVE,
    SEED,
    choose_fixed_band,
    choose_probability_cutoff,
    gate_daily_portfolio,
    make_models,
    portfolio_summary,
    prepare_gate_frame,
    split_gate_frame,
)

OUTPUT = REPO_ROOT / "final_experiments" / "outputs" / "10_thresholds"
OUTPUT.mkdir(parents=True, exist_ok=True)
AGG_PATH = REPO_ROOT / "final_experiments" / "outputs" / "03_aggregation" / "firm_day_aggregators.parquet"
SURPRISE_PATH = REPO_ROOT / "final_experiments" / "outputs" / "04_surprise" / "firm_day_surprise.parquet"

print("fit: <=2017-12-31; validation: 2018-01-01..2019-12-31; evaluation: >=2020-01-01")
print("models: fixed band, logistic, gradient boosted, small MLP, shuffled-label MLP")
print("seed:", SEED)
print("cost: 10 bps/side; inference: 20-session block bootstrap")
print(f"cutoff eligibility: mean active >= {MIN_MEAN_ACTIVE:g}, active-session share >= {MIN_ACTIVE_SESSION_SHARE:.0%}")

# %%
agg = pd.read_parquet(AGG_PATH)
surprise = pd.read_parquet(SURPRISE_PATH)
frame = prepare_gate_frame(agg, surprise)
train, validation, evaluation = split_gate_frame(frame)
print(f"train={len(train):,}; validation={len(validation):,}; evaluation={len(evaluation):,}")
print("features:", FEATURES)

# %% [markdown]
# ## Fixed-band baseline first

# %%
fixed_band, fixed_sweep = choose_fixed_band(validation)
fixed_sweep.to_csv(OUTPUT / "fixed_band_validation_sweep.csv", index=False)
display(fixed_sweep.round(6))
print("frozen fixed band:", fixed_band)

# %% [markdown]
# ## Fit the three declared learned gates

# %%
x_train = train[list(FEATURES)]
y_train = train["target_correct_direction"]
x_validation = validation[list(FEATURES)]
x_evaluation = evaluation[list(FEATURES)]

models = make_models(SEED)
frozen = []
cutoff_sweeps = []
probabilities = {}
for name, model in models.items():
    print("fitting", name)
    model.fit(x_train, y_train)
    p_validation = model.predict_proba(x_validation)[:, 1]
    cutoff, sweep = choose_probability_cutoff(validation, p_validation)
    sweep.insert(0, "model", name)
    cutoff_sweeps.append(sweep)
    probabilities[(name, "validation")] = p_validation
    probabilities[(name, "evaluation")] = model.predict_proba(x_evaluation)[:, 1]
    frozen.append(
        {
            "model": name,
            "probability_cutoff": cutoff,
            "validation_auc": roc_auc_score(validation["target_correct_direction"], p_validation),
        }
    )

cutoff_table = pd.concat(cutoff_sweeps, ignore_index=True)
cutoff_table.to_csv(OUTPUT / "learned_gate_validation_sweeps.csv", index=False)
frozen_table = pd.DataFrame(frozen)
frozen_table.to_csv(OUTPUT / "frozen_gate_specs.csv", index=False)
display(frozen_table.round(6))

# %% [markdown]
# ## Label-shuffle capacity control

# %%
rng = np.random.default_rng(SEED)
shuffled_y = rng.permutation(y_train.to_numpy())
shuffled = make_models(SEED)["mlp"]
shuffled.fit(x_train, shuffled_y)
p_shuffle_validation = shuffled.predict_proba(x_validation)[:, 1]
shuffle_cutoff, shuffle_sweep = choose_probability_cutoff(validation, p_shuffle_validation)
shuffle_sweep.insert(0, "model", "mlp_label_shuffle")
shuffle_sweep.to_csv(OUTPUT / "label_shuffle_validation_sweep.csv", index=False)
p_shuffle_evaluation = shuffled.predict_proba(x_evaluation)[:, 1]
print("shuffled-label MLP cutoff:", shuffle_cutoff)

# %% [markdown]
# ## Frozen evaluation comparison

# %%
rows = []
fixed_daily = gate_daily_portfolio(
    evaluation,
    evaluation["abs_oriented_rank"] > fixed_band,
)
rows.append({"arm": "fixed_band", "threshold": fixed_band, **portfolio_summary(fixed_daily)})

for spec in frozen:
    name = spec["model"]
    cutoff = float(spec["probability_cutoff"])
    daily = gate_daily_portfolio(evaluation, probabilities[(name, "evaluation")] >= cutoff)
    rows.append(
        {
            "arm": name,
            "threshold": cutoff,
            "evaluation_auc": roc_auc_score(evaluation["target_correct_direction"], probabilities[(name, "evaluation")]),
            **portfolio_summary(daily),
        }
    )

shuffle_daily = gate_daily_portfolio(evaluation, p_shuffle_evaluation >= shuffle_cutoff)
rows.append(
    {
        "arm": "mlp_label_shuffle",
        "threshold": shuffle_cutoff,
        "evaluation_auc": roc_auc_score(evaluation["target_correct_direction"], p_shuffle_evaluation),
        **portfolio_summary(shuffle_daily),
    }
)
evaluation_table = pd.DataFrame(rows)
evaluation_table.to_csv(OUTPUT / "evaluation_gate_comparison.csv", index=False)
display(evaluation_table.round(6))

# %% [markdown]
# ## Learned threshold surfaces and granularity feasibility
#
# Mean validation trade probability is tabulated by story-count and dispersion
# quintile. This is the interpretable threshold surface; it is not selected on
# evaluation. FNSPID has no sector map, so the required sector-conditioned arm
# cannot be fit honestly. The plan orders sector before stock, so per-stock
# thresholds are also not promoted past feasibility.

# %%
surface_base = validation[["article_count", "dispersion"]].copy()
surface_base["story_count_bin"] = pd.cut(
    surface_base["article_count"],
    bins=[0, 1, 5, 20, 75, np.inf],
    labels=["1", "2-5", "6-20", "21-75", "76+"],
)
surface_base["dispersion_quintile"] = pd.qcut(surface_base["dispersion"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
surface_rows = []
for spec in frozen:
    name = spec["model"]
    local = surface_base.copy()
    local["trade_probability"] = probabilities[(name, "validation")]
    grouped = (
        local.groupby(["story_count_bin", "dispersion_quintile"], observed=True)
        .agg(mean_trade_probability=("trade_probability", "mean"), n=("trade_probability", "size"))
        .reset_index()
    )
    grouped.insert(0, "model", name)
    grouped["frozen_probability_cutoff"] = spec["probability_cutoff"]
    surface_rows.append(grouped)
surface = pd.concat(surface_rows, ignore_index=True)
surface.to_csv(OUTPUT / "learned_threshold_surfaces.csv", index=False)

feasibility = pd.DataFrame(
    [
        {
            "variant": "pooled",
            "status": "COMPLETED",
            "reason": "wide FNSPID panel supports pooled development fitting",
        },
        {
            "variant": "per_sector",
            "status": "BLOCKED_MISSING_INPUT",
            "reason": "no explicit point-in-time sector map exists for the 570-symbol FNSPID panel",
        },
        {
            "variant": "per_stock",
            "status": "NOT_RUN_BY_PREDECLARED_ORDER",
            "reason": "protocol requires sector-conditioned before stock-specific thresholds",
        },
    ]
)
feasibility.to_csv(OUTPUT / "threshold_granularity_feasibility.csv", index=False)
display(feasibility)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
plot = evaluation_table.set_index("arm")
ax.bar(plot.index, plot["net_sharpe"], color="#2f5d8c")
ax.axhline(0, color="#666", lw=1)
ax.set_ylabel("Evaluation net Sharpe (10 bps/side)")
ax.set_title("Frozen learned gates vs fixed band")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(OUTPUT / "evaluation_gate_comparison.png", dpi=150)
plt.show()

manifest = {
    "target": "whether oriented negative-share rank earns positive ar_open_h1",
    "fit": "<=2017-12-31",
    "validation": "2018-01-01..2019-12-31",
    "evaluation": ">=2020-01-01 chronological evaluation block",
    "models": ["fixed_band", "logistic", "gradient_boosted", "mlp", "mlp_label_shuffle"],
    "features": list(FEATURES),
    "cost_bps_per_side": 10.0,
    "cutoff_selection_constraints": {
        "minimum_mean_active_names": MIN_MEAN_ACTIVE,
        "minimum_active_session_share": MIN_ACTIVE_SESSION_SHARE,
    },
    "bootstrap": {"block_length": 20, "replications": 999, "seed": SEED},
    "sector_variant": "BLOCKED_MISSING_FNSPID_SECTOR_MAP",
    "stock_variant": "NOT_RUN_BECAUSE_SECTOR_FIRST_RULE_IS_BINDING",
}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
