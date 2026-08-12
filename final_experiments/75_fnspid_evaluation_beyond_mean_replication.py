# %% [markdown]
# # FNSPID evaluation-block replication of the beyond-the-mean estimands
#
# **Objective.** Open the 2020–2023 evaluation block exactly once for the
# RQ-A statistical estimands: the nine-rule daily IC family (Family A) and
# the conditional negative-share coefficient with its pre-declared
# multi-story stratum (Family B).
#
# **Status.** PREPARED, NOT EXECUTED. The design is frozen in
# `frozen_specs/fnspid_evaluation_beyond_mean_replication_v1_20260812.json`.
# Running this notebook is the one-shot opening of the evaluation block for
# these estimands — do it deliberately, after confirming the Gate F1 RQ-A
# framing, and never rerun it with modified tests.

# %%
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name == "final_experiments":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_experiments.lib.aggregators import (  # noqa: E402
    AGGREGATOR_NAMES,
    cross_sectional_daily_ic,
)
from final_experiments.lib.conditional_aggregation import (  # noqa: E402
    conditional_negative_share_test,
)
from final_experiments.lib.plots import apply_house_style  # noqa: E402

apply_house_style()
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")

SPEC_PATH = (
    ROOT
    / "final_experiments/frozen_specs/fnspid_evaluation_beyond_mean_replication_v1_20260812.json"
)
OUTPUT = ROOT / "final_experiments/outputs/75_fnspid_evaluation_beyond_mean_replication"
FNSPID_PATH = ROOT / "final_experiments/outputs/03_aggregation/firm_day_aggregators.parquet"

# %% [markdown]
# ## Input gate: verify the frozen specification and panel identity

# %%
spec = json.loads(SPEC_PATH.read_text())
digest = hashlib.sha256(FNSPID_PATH.read_bytes()).hexdigest()
frozen_digest = spec["inputs"]["fnspid_firm_day_aggregators"]["sha256"]
assert digest == frozen_digest, f"panel hash changed: {digest} != {frozen_digest}"

panel = pd.read_parquet(FNSPID_PATH)
panel["session_date"] = pd.to_datetime(panel["session_date"])
evaluation = panel.loc[panel["split"] == "evaluation"].copy()
display(
    Markdown(
        f"Evaluation slice: {len(evaluation):,} firm-days, "
        f"{evaluation['session_date'].nunique():,} sessions, "
        f"{evaluation['session_date'].min().date()} to "
        f"{evaluation['session_date'].max().date()}."
    )
)
assert len(evaluation) == 203393, "evaluation row count differs from the frozen split"

OUTPUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Family A — nine-rule daily IC on evaluation sessions
#
# Identical estimand and parameters as the Notebook 03 development
# tournament: min 3 names per session, HAC lag 5, BH across the nine
# two-sided p-values at q = 0.05.

# %%
family_a_rows = []
for aggregator in AGGREGATOR_NAMES:
    result = cross_sectional_daily_ic(
        evaluation[aggregator],
        evaluation["ar_open_h1"],
        evaluation["session_date"],
        min_names=3,
        hac_lags=5,
    )
    family_a_rows.append({"aggregator": aggregator, **result})
family_a = pd.DataFrame(family_a_rows)
family_a = family_a.sort_values("p").reset_index(drop=True)
m = len(family_a)
family_a["bh_rank"] = range(1, m + 1)
family_a["bh_critical"] = family_a["bh_rank"] / m * 0.05
passing = family_a.loc[family_a["p"] <= family_a["bh_critical"]]
cutoff = int(passing["bh_rank"].max()) if len(passing) else 0
family_a["bh_reject_q05"] = family_a["bh_rank"] <= cutoff
family_a.to_csv(OUTPUT / "evaluation_nine_rule_ic.csv", index=False)
display(family_a)

# %% [markdown]
# ## Family B — conditional negative-share coefficient on evaluation sessions
#
# Primary: the exact Notebook 71 estimand. Secondary: the multi-story
# (n >= 2) stratum declared after the Notebook 74 development diagnostic.
# BH across the two two-sided p-values at q = 0.05.

# %%
family_b_rows = []
for name, subset in (
    ("all_firm_days", evaluation),
    ("multi_story_n_ge_2", evaluation.loc[evaluation["n"] >= 2]),
):
    summary, daily, audit = conditional_negative_share_test(
        subset,
        date_col="session_date",
        outcome_col="ar_open_h1",
        mean_col="mean_continuous",
        negative_share_col="negative_share",
        count_col="n",
        min_names=10,
        hac_lags=5,
    )
    daily.to_csv(OUTPUT / f"evaluation_daily_coefficients_{name}.csv", index=False)
    family_b_rows.append(
        {
            "member": name,
            "firm_days": int(len(subset)),
            "sessions_used": audit["dates_used"],
            "estimate": summary["estimate"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "p_two_sided": summary["p_two_sided"],
        }
    )
family_b = pd.DataFrame(family_b_rows)
family_b = family_b.sort_values("p_two_sided").reset_index(drop=True)
raw_q = (family_b["p_two_sided"] * 2 / (family_b.index + 1)).clip(upper=1.0)
family_b["q_bh_two_test"] = raw_q[::-1].cummin()[::-1]
family_b["direction_negative"] = family_b["estimate"] < 0
family_b.to_csv(OUTPUT / "evaluation_conditional_coefficients.csv", index=False)
display(family_b)

# %% [markdown]
# ## Interpretation boundary
#
# The recorded readings from the frozen specification apply verbatim:
#
# - Primary passes, multi-story stratum fails → the beyond-the-mean claim
#   narrows to firm-days with few stories, where negative share approximates
#   a hard negative label.
# - Both pass → the distributional reading extends to multi-story days.
# - Primary fails → the development coefficient did not replicate across
#   eras, and that is the reported finding.
#
# The 2020–2023 block is a chronological evaluation block, not a pristine
# holdout. No retuning, re-stratification, or re-ranking follows from any
# outcome, and no trading arm is computed under this specification.
