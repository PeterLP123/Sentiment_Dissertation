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
# # 09 — Earnings-date effects
#
# **Windows.** Pre = sessions −5…−1, event = 0, post = +1…+5, outside
# otherwise. Distances use the exchange-session ordinal and the frozen LSEG
# BMO/AMC/unknown/during-market mapping.
#
# **Primary exploratory estimand.** Difference between each inside-window slope
# and the outside-window slope for oriented `negative_share` on development
# rows. One pooled regression, date-clustered errors, BH across three
# interactions × h1. This is not a PEAD test.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.aggregators import AGGREGATOR_NAMES, development_ic_table  # noqa: E402
from final_experiments.lib.earnings import (  # noqa: E402
    INTERACTION_FAMILY,
    WINDOW,
    attach_earnings_distance,
    earnings_interaction_table,
    map_earnings_sessions,
    read_quarterly_calendar,
)

OUTPUT = REPO_ROOT / "final_experiments" / "outputs" / "09_earnings"
OUTPUT.mkdir(parents=True, exist_ok=True)
AGG_PATH = REPO_ROOT / "final_experiments" / "outputs" / "03_aggregation" / "firm_day_aggregators.parquet"
TYPE_PATH = REPO_ROOT / "final_experiments" / "outputs" / "08_story_type" / "firm_day_story_types.parquet"
CALENDAR_PATH = REPO_ROOT / "final_experiments" / "data" / "earnings" / "fnspid_earnings_calendar_2011_2023_quarterly.csv"

print(f"windows: pre=-{WINDOW}..-1, event=0, post=1..{WINDOW}")
print("split: development through 2019-12-31")
print("cluster: session_date")
print("multiplicity:", INTERACTION_FAMILY)

# %% [markdown]
# ## Map calendar and attach nearest-event distance

# %%
agg = pd.read_parquet(AGG_PATH)
calendar = read_quarterly_calendar(CALENDAR_PATH)
mapped = map_earnings_sessions(calendar, agg["session_date"])
earnings = attach_earnings_distance(agg, mapped)
earnings.to_parquet(OUTPUT / "firm_day_earnings_windows.parquet", index=False)
window_mix = (
    earnings.groupby(["split", "earnings_window"], observed=True)
    .agg(firm_days=("symbol", "size"), symbols=("symbol", "nunique"), sessions=("session_date", "nunique"))
    .reset_index()
)
window_mix.to_csv(OUTPUT / "window_coverage.csv", index=False)
display(window_mix)

# %% [markdown]
# ## Validate earnings-guidance clustering
#
# This is a validation of the *machine taxonomy against the calendar*, not a
# validation of the taxonomy against human labels.

# %%
types = pd.read_parquet(TYPE_PATH)
typed = earnings[["symbol", "session_date", "sessions_to_earnings"]].merge(
    types[["symbol", "session_date", "earnings_guidance_count", "typed_story_count"]],
    on=["symbol", "session_date"],
    how="inner",
)
guidance = typed.loc[typed["earnings_guidance_count"] > 0].copy()
validation = pd.DataFrame(
    [
        {
            "machine_earnings_guidance_firm_days": len(guidance),
            "within_event_session_share": float(guidance["sessions_to_earnings"].eq(0).mean()),
            "within_plus_minus_1_share": float(guidance["sessions_to_earnings"].abs().le(1).mean()),
            "within_plus_minus_5_share": float(guidance["sessions_to_earnings"].abs().le(5).mean()),
            "calendar_covered_share": float(guidance["sessions_to_earnings"].notna().mean()),
            "taxonomy_status": "PROVISIONAL_PENDING_HUMAN_AUDIT",
        }
    ]
)
validation.to_csv(OUTPUT / "earnings_guidance_calendar_validation.csv", index=False)
display(validation.T)

# %% [markdown]
# ## Pooled window interactions and timing sensitivity

# %%
dev = earnings.loc[earnings["split"] == "development"].copy()
interactions = earnings_interaction_table(dev)
interactions["sample"] = "all_frozen_timing_rules"

# During-market rows can contaminate an open-to-open event-session outcome;
# unknown-time rows use the conservative next-session convention. Remove both
# only when the nearest event is inside the declared window.
uncertain_inside = dev["inside_earnings_window"] & dev["nearest_earnings_timing"].isin(["during_market", "unknown"])
sensitivity = earnings_interaction_table(dev.loc[~uncertain_inside])
sensitivity["sample"] = "drop_inside_during_market_and_unknown"
interaction_all = pd.concat([interactions, sensitivity], ignore_index=True)
interaction_all.to_csv(OUTPUT / "development_earnings_interactions.csv", index=False)
display(interaction_all.round(6))

plot = interactions.set_index("window").loc[["pre", "event", "post"]]
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.errorbar(
    plot.index,
    plot["slope_delta_vs_outside"],
    yerr=1.96 * plot["se"],
    fmt="o",
    color="#2f5d8c",
    capsize=4,
)
ax.axhline(0, color="#666", lw=1)
ax.set_ylabel("Slope difference vs outside (95% clustered CI)")
ax.set_title("Earnings-window interaction — development only")
fig.tight_layout()
fig.savefig(OUTPUT / "development_earnings_interactions.png", dpi=150)
plt.show()

# %% [markdown]
# ## Exclusion robustness for the complete W3 family
#
# Re-run the nine-rule development family after dropping every ±5-session
# earnings-window firm-day. BH is recomputed within the same nine-rule family.

# %%
outside = earnings.loc[~earnings["inside_earnings_window"]].copy()
outside_ic = development_ic_table(outside, aggregators=AGGREGATOR_NAMES, by_n_bin=False)
outside_ic.to_csv(OUTPUT / "development_ic_excluding_earnings_windows.csv", index=False)
display(outside_ic.sort_values("p").round(6))

manifest = {
    "estimand": "inside-window slope differences vs outside for oriented negative_share",
    "split": "development <= 2019-12-31",
    "windows": {"pre": [-5, -1], "event": [0, 0], "post": [1, 5]},
    "cluster": "session_date",
    "multiplicity_family": INTERACTION_FAMILY,
    "calendar_rows": len(calendar),
    "mapped_calendar_rows": len(mapped),
    "bh_surviving_interactions": int(interactions["bh_reject_q05"].sum()),
    "exclusion_family": "nine aggregators x h1; BH-FDR q=0.05",
    "pead_boundary": "No post-earnings drift claim: outcome is next-session h1 and windows are conditioning variables.",
    "limitations": [
        "earnings-guidance taxonomy remains provisional pending human audit",
        "recoverable calendar mapping and issuer-title coverage holes remain documented",
        "during-market event-session returns can include pre-announcement trading",
    ],
}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
