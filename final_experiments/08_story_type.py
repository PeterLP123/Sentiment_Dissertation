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
# # 08 — Story class and event-type conditioning
#
# **Estimand.** Type-specific slope of the development-only, economically
# oriented `negative_share` signal against exact next-session `ar_open_h1`.
#
# **Design.** One pooled model with 12 type intercepts and 12 type slopes;
# session-date clustered standard errors; BH-FDR over 12 slopes × h1. The event
# taxonomy and breaking/repetition/routine candidate classes are deterministic
# regex/novelty outputs. They are explicitly **provisional** because the seeded
# 180-story human audit is not labelled. No evaluation rows select a type.

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

from final_experiments.lib.event_types import (  # noqa: E402
    AUDIT_N,
    AUDIT_SEED,
    TYPE_FAMILY,
    build_event_type_artifacts,
    pooled_type_slopes,
)
from final_experiments.lib.panel import DEFAULT_EVENTS_DB  # noqa: E402

OUTPUT = REPO_ROOT / "final_experiments" / "outputs" / "08_story_type"
OUTPUT.mkdir(parents=True, exist_ok=True)
AGG_PATH = REPO_ROOT / "final_experiments" / "outputs" / "03_aggregation" / "firm_day_aggregators.parquet"
NOVELTY_PATH = REPO_ROOT / "final_experiments" / "outputs" / "02_filters_and_distribution" / "story_novelty.parquet"

print("estimand: type-specific oriented-negative-share slope vs ar_open_h1")
print("split: development through 2019-12-31")
print("cluster: session_date")
print("multiplicity:", TYPE_FAMILY)
print(f"audit: seed={AUDIT_SEED}, target n={AUDIT_N}, human labels required")

# %% [markdown]
# ## Build provisional story classes and dominant firm-day types

# %%
firm_day_types, class_mix, audit = build_event_type_artifacts(
    DEFAULT_EVENTS_DB,
    NOVELTY_PATH,
)
firm_day_types.to_parquet(OUTPUT / "firm_day_story_types.parquet", index=False)
class_mix.to_csv(OUTPUT / "story_class_event_type_mix.csv", index=False)
audit.to_csv(OUTPUT / "story_type_audit_sample.csv", index=False)
print(f"typed firm-days={len(firm_day_types):,}; audit rows={len(audit):,}")
print("audit human labels populated:", int(audit["human_event_type"].astype(bool).sum()))
display(class_mix.sort_values("stories", ascending=False).head(20))

# %% [markdown]
# ## Pooled type interaction
#
# `other` is not selected as a reference: the full-rank parameterisation gives
# every type its own intercept and slope inside one pooled regression.

# %%
agg = pd.read_parquet(AGG_PATH)
analysis = agg.merge(firm_day_types, on=["symbol", "session_date"], how="inner")
dev = analysis.loc[analysis["split"] == "development"].copy()
slopes = pooled_type_slopes(dev)
slopes.to_csv(OUTPUT / "development_type_interactions.csv", index=False)
display(slopes.sort_values("p").round(6))

ordered = slopes.sort_values("slope")
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.errorbar(
    ordered["slope"],
    range(len(ordered)),
    xerr=1.96 * ordered["se"],
    fmt="o",
    color="#2f5d8c",
    capsize=3,
)
ax.axvline(0, color="#666", lw=1)
ax.set_yticks(range(len(ordered)))
ax.set_yticklabels(ordered["event_type"])
ax.set_xlabel("Slope of oriented negative share vs ar_open_h1 (95% clustered CI)")
ax.set_title("Provisional FNSPID event-type interactions — development only")
fig.tight_layout()
fig.savefig(OUTPUT / "development_type_interactions.png", dpi=150)
plt.show()

# %% [markdown]
# ## Conditional type weighting gate
#
# A type-weighted aggregator is allowed only if at least one of the 12 slopes
# survives the declared correction. The status below is generated, not inferred
# from visual inspection.

# %%
n_survive = int(slopes["bh_reject_q05"].sum())
conditional = pd.DataFrame(
    [
        {
            "eligible": n_survive > 0,
            "bh_surviving_type_slopes": n_survive,
            "action": (
                "type-weighted aggregator may be specified in a new frozen arm"
                if n_survive > 0
                else "retain unweighted default; do not create a type-weighted aggregator"
            ),
            "audit_status": "BLOCKED_PENDING_HUMAN_LABELS",
        }
    ]
)
conditional.to_csv(OUTPUT / "type_weighting_gate.csv", index=False)
display(conditional)

manifest = {
    "status": "PROVISIONAL_MACHINE_CLASSIFICATION",
    "estimand": "type-specific slope of oriented negative_share vs ar_open_h1",
    "split": "development <= 2019-12-31",
    "cluster": "session_date",
    "multiplicity_family": TYPE_FAMILY,
    "audit_seed": AUDIT_SEED,
    "audit_n": len(audit),
    "human_labels_present": 0,
    "bh_surviving_type_slopes": n_survive,
    "limitations": [
        "12-type regex taxonomy has not passed the seeded FNSPID human audit",
        "breaking and routine are candidate labels, not validated classes",
        "market recap remains an orthogonal flag and is not treated as routine reporting",
        "publisher and story-family fields are absent on FNSPID checkpoint grain",
    ],
}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
