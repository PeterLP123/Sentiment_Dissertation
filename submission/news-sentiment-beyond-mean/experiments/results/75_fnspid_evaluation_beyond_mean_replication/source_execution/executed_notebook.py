# %% [markdown]
# # FNSPID evaluation-block replication of the beyond-the-mean estimands
#
# **Objective.** Open the 2020–2023 evaluation block exactly once for the
# RQ-A statistical estimands: the nine-rule daily IC family (Family A) and
# the conditional negative-share coefficient with its pre-declared
# multi-story stratum (Family B).
#
# **Execution contract.** The design was frozen before outcome opening in
# `frozen_specs/fnspid_evaluation_beyond_mean_replication_v1_20260812.json`.
# A successful one-shot opening is recorded by the output manifest; the code
# refuses every pre-existing output path and must never be rerun with modified
# tests.

# %%
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

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
    hac_regime_mean_contrast,
)
from final_experiments.lib.plots import apply_house_style  # noqa: E402

apply_house_style()
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")

SPEC_PATH = ROOT / "final_experiments/frozen_specs/fnspid_evaluation_beyond_mean_replication_v1_20260812.json"
AMENDMENT_PATH = ROOT / "final_experiments/frozen_specs/fnspid_evaluation_beyond_mean_preexecution_amendment_v1_20260814.json"
SOURCE_PATH = ROOT / "final_experiments/75_fnspid_evaluation_beyond_mean_replication.py"
HELPER_PATH = ROOT / "final_experiments/lib/conditional_aggregation.py"
OUTPUT = ROOT / "final_experiments/outputs/75_fnspid_evaluation_beyond_mean_replication"
FNSPID_PATH = ROOT / "final_experiments/outputs/03_aggregation/firm_day_aggregators.parquet"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# %% [markdown]
# ## Preflight gate: verify the freeze, amendment, source and untouched output

# %%
spec = json.loads(SPEC_PATH.read_text())
amendment = json.loads(AMENDMENT_PATH.read_text())
assert amendment["status"] == "frozen_pre_outcome_amendment; not yet executed"
assert amendment["parent_spec_sha256"] == sha256_file(SPEC_PATH)
assert amendment["implementation_source_sha256"] == sha256_file(SOURCE_PATH)
assert amendment["conditional_helper_sha256"] == sha256_file(HELPER_PATH)
assert list(AGGREGATOR_NAMES) == spec["family_a_nine_rule_ic"]["aggregators"]
if OUTPUT.exists():
    raise FileExistsError(f"one-shot output path already exists; refusing to overwrite or resume: {OUTPUT}")

digest = sha256_file(FNSPID_PATH)
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
assert evaluation["session_date"].nunique() == 998, "evaluation session count changed"
assert evaluation["session_date"].min() == pd.Timestamp("2020-01-02")
assert evaluation["session_date"].max() == pd.Timestamp("2023-12-18")

OUTPUT.mkdir(parents=True, exist_ok=False)

# %% [markdown]
# ## Outcome-blind power context
#
# This conservative prospective calculation was added before opening the
# evaluation estimand. It scales the frozen development HAC standard error by
# the square root of the development/evaluation session-count ratio and uses
# the first BH-2 hurdle (two-sided alpha 0.025). It is interpretive context,
# not a success gate and not a substitute for the realised evaluation interval.

# %%
development = panel.loc[panel["split"] == "development"].copy()
development_summary, development_daily, development_audit = conditional_negative_share_test(
    development,
    date_col="session_date",
    outcome_col="ar_open_h1",
    mean_col="mean_continuous",
    negative_share_col="negative_share",
    count_col="n",
    min_names=10,
    hac_lags=5,
)
assert development_audit["dates_used"] == 2264
assert math.isclose(
    float(development_summary["estimate"]),
    -0.008310052990470732,
    rel_tol=0.0,
    abs_tol=1e-15,
)
assert math.isclose(
    float(development_summary["se"]),
    0.0029540906894468953,
    rel_tol=0.0,
    abs_tol=1e-15,
)

normal = NormalDist()
conservative_alpha = 0.05 / 2
scaled_evaluation_se = float(development_summary["se"]) * math.sqrt(development_audit["dates_used"] / evaluation["session_date"].nunique())
stable_effect_z = abs(float(development_summary["estimate"])) / scaled_evaluation_se
critical_z = normal.inv_cdf(1 - conservative_alpha / 2)
approximate_power = normal.cdf(-critical_z + stable_effect_z) + (1 - normal.cdf(critical_z + stable_effect_z))
mde80 = (critical_z + normal.inv_cdf(0.80)) * scaled_evaluation_se
power_context = pd.DataFrame(
    [
        {
            "role": "prospective_interpretive_context_not_a_gate",
            "method": "development_hac_se_scaled_by_sqrt_session_ratio",
            "development_estimate": float(development_summary["estimate"]),
            "development_hac_se": float(development_summary["se"]),
            "development_sessions": development_audit["dates_used"],
            "evaluation_sessions_frozen": int(evaluation["session_date"].nunique()),
            "conservative_two_sided_alpha": conservative_alpha,
            "scaled_evaluation_se": scaled_evaluation_se,
            "stable_effect_z": stable_effect_z,
            "approximate_power_for_stable_development_effect": approximate_power,
            "approximate_mde80": mde80,
            "limitations": (
                "normal approximation using a scaled development HAC standard error; realised evaluation dependence and variance may differ"
            ),
        }
    ]
)
power_context.to_csv(OUTPUT / "prospective_power_context.csv", index=False)
development_daily.to_csv(OUTPUT / "development_daily_coefficients_all_firm_days.csv", index=False)
display(power_context)

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
evaluation_daily_by_member = {}
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
    evaluation_daily_by_member[name] = daily.copy()
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
# ## Predeclared temporal-stability contrast
#
# The amendment adds one separate secondary diagnostic: the evaluation-minus-
# development difference in the all-firm-day daily coefficient mean, with
# HAC(5) uncertainty. It does not change either primary family, its correction,
# or its pass/fail rule, and it cannot rescue a failed evaluation result.

# %%
development_daily_for_contrast = development_daily.assign(regime="development")
evaluation_daily_for_contrast = evaluation_daily_by_member["all_firm_days"].assign(regime="evaluation")
stability_daily = pd.concat(
    [development_daily_for_contrast, evaluation_daily_for_contrast],
    ignore_index=True,
).sort_values("session_date", kind="mergesort")
stability_contrast = hac_regime_mean_contrast(
    stability_daily,
    coefficient_col="beta_negative_share",
    date_col="session_date",
    regime_col="regime",
    reference="development",
    comparison="evaluation",
    hac_lags=5,
)
stability_contrast_table = pd.DataFrame([stability_contrast])
stability_contrast_table.to_csv(
    OUTPUT / "evaluation_minus_development_stability_contrast.csv",
    index=False,
)
display(stability_contrast_table)

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

# %%
family_a_negative = family_a.loc[family_a["aggregator"] == "negative_share"].iloc[0]
family_b_indexed = family_b.set_index("member")
all_result = family_b_indexed.loc["all_firm_days"]
multi_result = family_b_indexed.loc["multi_story_n_ge_2"]
family_a_pass = bool(family_a_negative["ic"] < 0 and family_a_negative["bh_reject_q05"])
all_result_pass = bool(all_result["estimate"] < 0 and all_result["q_bh_two_test"] <= 0.05)
multi_result_pass = bool(multi_result["estimate"] < 0 and multi_result["q_bh_two_test"] <= 0.05)
if all_result_pass and multi_result_pass:
    recorded_interpretation = (
        "The all-firm-day and multi-story conditional coefficients both pass; the distributional reading extends to multi-story firm-days."
    )
elif all_result_pass:
    recorded_interpretation = (
        "The all-firm-day coefficient passes but the multi-story coefficient does not; "
        "the claim narrows to a hard-negative threshold/nonlinearity on few-story days."
    )
else:
    recorded_interpretation = (
        "The all-firm-day conditional coefficient does not pass on the evaluation block; "
        "the development coefficient did not replicate across eras under the frozen gate."
    )

result_summary = {
    "schema_version": 1,
    "status": "completed_one_shot_evaluation_opening",
    "executed_at_utc": datetime.now(UTC).isoformat(),
    "family_a_negative_share_pass": family_a_pass,
    "family_b_all_firm_days_pass": all_result_pass,
    "family_b_multi_story_pass": multi_result_pass,
    "recorded_interpretation": recorded_interpretation,
    "stability_contrast_role": "separate predeclared secondary diagnostic; not a rescue gate",
    "stability_contrast": stability_contrast,
    "power_context": power_context.iloc[0].to_dict(),
}
(OUTPUT / "result_summary.json").write_text(json.dumps(result_summary, indent=2, sort_keys=True, default=str) + "\n")
display(Markdown(f"**Recorded interpretation:** {recorded_interpretation}"))

# %% [markdown]
# ## Run provenance
#
# The manifest records the immutable input, parent freeze, pre-outcome
# amendment, source hash, current repository state and every generated result
# file. The original frozen specification is not rewritten after seeing the
# outcome.

# %%
git_head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
git_status = subprocess.run(
    ["git", "status", "--short"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
generated_files = sorted(path for path in OUTPUT.iterdir() if path.is_file())
run_manifest = {
    "schema_version": 1,
    "status": "complete",
    "experiment": "fnspid_evaluation_beyond_mean_replication_v1",
    "executed_at_utc": result_summary["executed_at_utc"],
    "git_head_at_execution": git_head,
    "working_tree_dirty_at_execution": bool(git_status),
    "working_tree_status_at_execution": git_status,
    "python": sys.version,
    "pandas": pd.__version__,
    "inputs": {
        str(FNSPID_PATH.relative_to(ROOT)): digest,
        str(SPEC_PATH.relative_to(ROOT)): sha256_file(SPEC_PATH),
        str(AMENDMENT_PATH.relative_to(ROOT)): sha256_file(AMENDMENT_PATH),
        str(SOURCE_PATH.relative_to(ROOT)): sha256_file(SOURCE_PATH),
        str(HELPER_PATH.relative_to(ROOT)): sha256_file(HELPER_PATH),
    },
    "outputs": {str(path.relative_to(OUTPUT)): sha256_file(path) for path in generated_files},
}
(OUTPUT / "manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
display(run_manifest)
