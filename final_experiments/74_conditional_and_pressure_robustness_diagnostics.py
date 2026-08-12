# %% [markdown]
# # Post-hoc robustness diagnostics for Notebooks 71 and 73
#
# **Objective.** Bound the two newest results before they enter the write-up:
# the FNSPID conditional negative-share coefficient (Notebook 71) and the
# backward LSEG negative-pressure downside result (Notebook 73).
#
# **Status.** These are labelled post-hoc diagnostics on already-opened data.
# Nothing here is a frozen gate, nothing can promote or reverse a frozen
# result, and no seed, threshold, or estimand from the original notebooks is
# changed. The questions are the ones an examiner asks: is the coefficient
# stable over time, does it exist on genuinely multi-story days, is the
# downside result one episode in disguise, and does the inference survive
# reasonable dependence assumptions?

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name == "final_experiments":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_experiments.lib.conditional_aggregation import (  # noqa: E402
    conditional_negative_share_test,
    hac_mean_coefficient,
)
from final_experiments.lib.plots import (  # noqa: E402
    CATEGORICAL,
    INK,
    apply_house_style,
    zero_line,
)
from final_experiments.lib.volatility_target import (  # noqa: E402
    paired_circular_block_mean,
)

apply_house_style()
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")

OUTPUT = ROOT / "final_experiments/outputs/74_conditional_and_pressure_robustness_diagnostics"
OUTPUT.mkdir(parents=True, exist_ok=True)

NB71_OUTPUT = ROOT / "final_experiments/outputs/71_conditional_negative_share_transfer"
NB73_OUTPUT = ROOT / "final_experiments/outputs/73_lseg_negative_pressure_har_backward_transfer"
FNSPID_PATH = ROOT / "final_experiments/outputs/03_aggregation/firm_day_aggregators.parquet"

# Frozen Notebook 71 headline numbers, reproduced before any diagnostic runs.
FROZEN_71 = {"estimate": -0.008310, "ci_low": -0.014100, "ci_high": -0.002520}
# Frozen Notebook 73 primary means (overlay minus HAR net return; HAR minus
# overlay downside-squared return).
FROZEN_73 = {"return_mean": 0.000014, "downside_mean": 0.000005}

DIAGNOSTIC_SEED = 20260840  # new seed family; original frozen seeds untouched

# %% [markdown]
# ## Part A — stability of the conditional negative-share coefficient
#
# The frozen claim rests on one number: the mean daily negative-share rank
# coefficient over 2,264 development sessions. These diagnostics ask whether
# that mean is a broad property of the period or an artefact of a few years,
# a lag choice, or single-story firm-days where "the tail of the day's
# distribution" is just a recode of the only story.

# %%
daily71 = pd.read_csv(
    NB71_OUTPUT / "fnspid_finbert_development_daily_coefficients.csv",
    parse_dates=["session_date"],
)
reproduced = hac_mean_coefficient(daily71, coefficient_col="beta_negative_share", hac_lags=5)
assert abs(reproduced["estimate"] - FROZEN_71["estimate"]) < 5e-6, reproduced
assert abs(reproduced["ci_low"] - FROZEN_71["ci_low"]) < 5e-5, reproduced
display(
    Markdown(
        f"Reproduced the frozen headline coefficient from the saved daily series: "
        f"{reproduced['estimate']:.6f} [{reproduced['ci_low']:.6f}, {reproduced['ci_high']:.6f}], "
        f"p = {reproduced['p_two_sided']:.4f} on {reproduced['n_clusters']} sessions."
    )
)

# %% [markdown]
# ### A1. Year-by-year coefficient

# %%
yearly_rows = []
for year, block in daily71.groupby(daily71["session_date"].dt.year):
    result = hac_mean_coefficient(block, coefficient_col="beta_negative_share", hac_lags=5)
    yearly_rows.append(
        {
            "year": int(year),
            "n_sessions": result["n_clusters"],
            "estimate": result["estimate"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "p_two_sided": result["p_two_sided"],
        }
    )
yearly = pd.DataFrame(yearly_rows)
yearly["negative"] = yearly["estimate"] < 0
yearly.to_csv(OUTPUT / "yearly_negative_share_coefficients.csv", index=False)
display(yearly)
display(
    Markdown(
        f"{int(yearly['negative'].sum())} of {len(yearly)} years have a negative point estimate; "
        f"the share of individual development sessions with a negative daily coefficient is "
        f"{float((daily71['beta_negative_share'] < 0).mean()):.3f}."
    )
)

# %%
fig, ax = plt.subplots(figsize=(8.0, 4.2))
colour = CATEGORICAL[0]
ax.errorbar(
    yearly["year"],
    yearly["estimate"],
    yerr=[yearly["estimate"] - yearly["ci_low"], yearly["ci_high"] - yearly["estimate"]],
    fmt="o",
    color=colour,
    ecolor=colour,
    elinewidth=1.6,
    capsize=3,
    markersize=6,
)
zero_line(ax)
ax.axhline(FROZEN_71["estimate"], color=INK["reference"], lw=1.0, ls="--", alpha=0.7)
ax.text(
    yearly["year"].iloc[-1] + 0.15,
    FROZEN_71["estimate"],
    "full-period\nestimate",
    color=INK["secondary"],
    fontsize=9,
    va="center",
)
ax.set_xlabel("Development year")
ax.set_ylabel("Mean daily negative-share rank coefficient")
ax.set_title("Conditional negative-share coefficient by year, HAC(5) 95% intervals")
fig.tight_layout()
fig.savefig(OUTPUT / "yearly_negative_share_coefficients.png", dpi=200)
plt.show()

# %% [markdown]
# ### A2. HAC lag sensitivity
#
# The frozen inference used lag 5. Longer kernels widen the interval if the
# daily coefficient series is more persistent than assumed.

# %%
lag_rows = []
for lags in (5, 10, 21):
    result = hac_mean_coefficient(daily71, coefficient_col="beta_negative_share", hac_lags=lags)
    lag_rows.append(
        {
            "hac_lags": lags,
            "estimate": result["estimate"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "p_two_sided": result["p_two_sided"],
        }
    )
lag_sensitivity = pd.DataFrame(lag_rows)
lag_sensitivity.to_csv(OUTPUT / "hac_lag_sensitivity.csv", index=False)
display(lag_sensitivity)

# %% [markdown]
# ### A3. Multi-story mechanism check
#
# On a firm-day with one story, negative-story share is a binary recode of
# that story's label and "beyond the mean" has no distributional content.
# The frozen estimand includes those days by design. This diagnostic reruns
# the identical regression on firm-days with at least two, then at least
# three, same-day stories. If the coefficient vanishes there, the claim is
# about single-story days and the write-up must say so.

# %%
panel = pd.read_parquet(FNSPID_PATH)
panel["session_date"] = pd.to_datetime(panel["session_date"])
development = panel.loc[panel["split"] == "development"].copy()
single_share = float((development["n"] == 1).mean())
display(
    Markdown(
        f"Development firm-days: {len(development):,}; share with exactly one story: "
        f"{single_share:.3f}."
    )
)

strata_rows = []
for label, minimum_stories in (("all firm-days (frozen)", 1), ("n >= 2", 2), ("n >= 3", 3)):
    subset = development.loc[development["n"] >= minimum_stories]
    summary, daily_strata, audit = conditional_negative_share_test(
        subset,
        date_col="session_date",
        outcome_col="ar_open_h1",
        mean_col="mean_continuous",
        negative_share_col="negative_share",
        count_col="n",
        min_names=10,
        hac_lags=5,
    )
    strata_rows.append(
        {
            "stratum": label,
            "firm_days": int(len(subset)),
            "sessions_used": audit["dates_used"],
            "estimate": summary["estimate"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "p_two_sided": summary["p_two_sided"],
        }
    )
count_strata = pd.DataFrame(strata_rows)
count_strata.to_csv(OUTPUT / "story_count_strata.csv", index=False)
display(count_strata)

# %% [markdown]
# ## Part B — concentration and dependence checks for the backward pressure result
#
# Notebook 73's surviving numbers are a downside-squared reduction
# (q = 0.0336) and a circular-timing pass (q = 0.0482) from 53 risk-off
# sessions across 18 entries in 2024–2025. With that few episodes the first
# question is concentration: is this one drawdown episode wearing a
# significance costume?

# %%
state = pd.read_csv(NB73_OUTPUT / "backward_hysteresis_state.csv", parse_dates=["session_date"])
har = pd.read_parquet(NB73_OUTPUT / "frozen_har_daily.parquet")
overlay = pd.read_parquet(NB73_OUTPUT / "har_negative_pressure_overlay_daily.parquet")
for frame in (har, overlay):
    frame["session_date"] = pd.to_datetime(frame["session_date"])

merged = har[["session_date", "net_return"]].merge(
    overlay[["session_date", "net_return"]],
    on="session_date",
    suffixes=("_har", "_overlay"),
    validate="one_to_one",
)
merged = merged.sort_values("session_date").reset_index(drop=True)
merged["d_return"] = merged["net_return_overlay"] - merged["net_return_har"]
merged["d_downside"] = (
    np.minimum(merged["net_return_har"], 0.0) ** 2
    - np.minimum(merged["net_return_overlay"], 0.0) ** 2
)
assert abs(merged["d_return"].mean() - FROZEN_73["return_mean"]) < 5e-6
assert abs(merged["d_downside"].mean() - FROZEN_73["downside_mean"]) < 5e-6
display(
    Markdown(
        f"Reproduced the frozen paired means: return difference "
        f"{merged['d_return'].mean() * 1e4:.3f} bps/session, downside reduction "
        f"{merged['d_downside'].mean() * 1e8:.3f} bps²/session over {len(merged)} sessions."
    )
)

# %% [markdown]
# ### B1. Risk-off episodes and their downside contributions
#
# An episode is a contiguous risk-off run from the frozen hysteresis state.
# The session immediately after an exit is attached to its episode so the
# re-entry turnover cost is attributed to the episode that caused it.

# %%
state = state.sort_values("session_date").reset_index(drop=True)
episode_id = np.full(len(state), -1, dtype=int)
current = 0
for index, row in state.iterrows():
    if bool(row["risk_off_entry"]):
        current += 1
    if bool(row["risk_off"]):
        episode_id[index] = current
    elif index > 0 and episode_id[index - 1] > 0 and not bool(row["risk_off"]):
        # first risk-on session after an exit carries the re-entry cost
        if bool(state.loc[index - 1, "risk_off"]):
            episode_id[index] = episode_id[index - 1]
state["episode"] = episode_id
n_episodes = int(state.loc[state["episode"] > 0, "episode"].nunique())

merged = merged.merge(state[["session_date", "episode"]], on="session_date", validate="one_to_one")
unattributed = merged.loc[merged["episode"] < 0, ["d_return", "d_downside"]].abs().sum()
display(
    Markdown(
        f"{n_episodes} episodes reconstructed (Notebook 73 reports 18 entries). "
        f"Absolute difference mass outside any episode window: return "
        f"{unattributed['d_return']:.2e}, downside {unattributed['d_downside']:.2e} "
        f"(should be ~0: arms coincide when risk-on)."
    )
)

episodes = (
    merged.loc[merged["episode"] > 0]
    .groupby("episode")
    .agg(
        start=("session_date", "min"),
        end=("session_date", "max"),
        n_sessions=("session_date", "size"),
        return_diff_sum=("d_return", "sum"),
        downside_reduction_sum=("d_downside", "sum"),
    )
    .reset_index()
)
total_downside = float(merged["d_downside"].sum())
episodes["downside_share"] = episodes["downside_reduction_sum"] / total_downside
episodes = episodes.sort_values("start").reset_index(drop=True)
episodes.to_csv(OUTPUT / "episode_attribution.csv", index=False)
display(episodes)
top = episodes.loc[episodes["downside_reduction_sum"].idxmax()]
display(
    Markdown(
        f"Largest episode: {top['start'].date()} to {top['end'].date()} "
        f"({int(top['n_sessions'])} sessions) contributes "
        f"{top['downside_share']:.1%} of the total downside-squared reduction."
    )
)

# %%
fig, ax = plt.subplots(figsize=(8.0, 4.2))
positions = np.arange(len(episodes))
ax.bar(
    positions,
    episodes["downside_share"],
    color=CATEGORICAL[0],
    width=0.7,
)
zero_line(ax)
ax.set_xticks(positions)
ax.set_xticklabels(
    [f"{row.start:%b %y}" for row in episodes.itertuples()],
    rotation=45,
    ha="right",
    fontsize=8,
)
ax.set_ylabel("Share of total downside-squared reduction")
ax.set_xlabel("Risk-off episode (by entry month)")
ax.set_title("Episode concentration of the backward downside result")
fig.tight_layout()
fig.savefig(OUTPUT / "episode_downside_concentration.png", dpi=200)
plt.show()

# %% [markdown]
# ### B2. Leave-one-episode-out downside inference
#
# For each episode, the paired difference series is set to zero on that
# episode's sessions — as if the state had never fired there — and the
# 20-session paired circular block bootstrap is re-run on the downside
# estimand with a fresh labelled seed. Dropping an episode changes the
# estimand, so this is a concentration diagnostic, not a re-test of the
# frozen two-test family.

# %%
loo_rows = []
for k, episode in enumerate(sorted(episodes["episode"].tolist())):
    mask = merged["episode"] == episode
    modified = merged["d_downside"].where(~mask, 0.0).to_numpy(dtype=float)
    result = paired_circular_block_mean(
        modified,
        block_length=20,
        replications=4999,
        seed=DIAGNOSTIC_SEED + k,
    )
    loo_rows.append(
        {
            "dropped_episode": int(episode),
            "start": episodes.loc[episodes["episode"] == episode, "start"].iloc[0],
            "downside_mean": result["mean"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "p_two_sided": result["p_two_sided"],
        }
    )
leave_one_out = pd.DataFrame(loo_rows)
leave_one_out["ci_excludes_zero"] = leave_one_out["ci_low"] > 0
leave_one_out.to_csv(OUTPUT / "leave_one_episode_out_downside.csv", index=False)
display(leave_one_out)
worst = leave_one_out.loc[leave_one_out["p_two_sided"].idxmax()]
display(
    Markdown(
        f"Weakest leave-one-out result: dropping the episode entered "
        f"{worst['start']:%Y-%m-%d} gives mean {worst['downside_mean']:.2e}, "
        f"interval [{worst['ci_low']:.2e}, {worst['ci_high']:.2e}], "
        f"p = {worst['p_two_sided']:.4f}. "
        f"{int(leave_one_out['ci_excludes_zero'].sum())} of {len(leave_one_out)} "
        f"drops keep the interval above zero."
    )
)

# %% [markdown]
# ### B3. Block-length sensitivity
#
# The frozen inference fixed 20-session blocks. Shorter blocks assume less
# dependence; longer blocks are more conservative when volatility episodes
# span a month or more.

# %%
block_rows = []
for j, block_length in enumerate((10, 20, 40)):
    for name, series, _frozen_mean in (
        ("overlay_minus_har_net_return", merged["d_return"], FROZEN_73["return_mean"]),
        ("har_minus_overlay_downside_squared", merged["d_downside"], FROZEN_73["downside_mean"]),
    ):
        result = paired_circular_block_mean(
            series.to_numpy(dtype=float),
            block_length=block_length,
            replications=4999,
            seed=DIAGNOSTIC_SEED + 100 + j,
        )
        block_rows.append(
            {
                "estimand": name,
                "block_length": block_length,
                "is_frozen_choice": block_length == 20,
                "mean": result["mean"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "p_two_sided": result["p_two_sided"],
            }
        )
block_sensitivity = pd.DataFrame(block_rows)
block_sensitivity.to_csv(OUTPUT / "block_length_sensitivity.csv", index=False)
display(block_sensitivity)

# %% [markdown]
# ## Interpretation boundary
#
# These diagnostics are post-hoc and labelled as such. They cannot promote
# the backward downside result past its failed full gate, cannot rescue the
# LSEG conditional transfer, and cannot tighten the frozen intervals. Their
# only legitimate use is in the write-up's robustness language: either the
# claims survive these checks and the existing bounded phrasing stands, or
# they do not and the phrasing must narrow further (for example, "the
# downside reduction is concentrated in one 2025 episode" or "the
# conditional coefficient is a single-story-day effect"). No threshold,
# seed, or estimand from Notebooks 71–73 changes on the basis of anything
# here.
