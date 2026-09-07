"""Generate licence-safe dissertation figures, tables and derived metrics.

The script reads only the committed aggregate evidence snapshot. It does not
open licensed story text, refit a model or reopen the one-shot testing study.
Outputs go to ``experiments/generated/dissertation`` for review before selected
files are promoted into the manuscript.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from experiments.lib.plots import display_label

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_OUTPUT = ROOT / "experiments" / "generated" / "dissertation"
LOCKED_FIGURE_PACKAGES = ("matplotlib", "numpy", "pandas")

# Report-native editorial-atlas system. Figures are authored at the manuscript
# text width rather than as presentation slides that LaTeX has to shrink. A
# midnight blue carries primary evidence; copper is reserved for comparators,
# costs and negative economic deltas. Shape, fill, line style and panel position
# repeat every colour distinction so the figures remain readable in greyscale.
FIGURE_WIDTH = 6.30
COLOUR = {
    "navy": "#183F5B",
    "navy_mid": "#3F718E",
    "navy_light": "#DCEAF1",
    "rust": "#C15D35",
    "rust_light": "#F2DDD4",
    "ink": "#182026",
    "secondary": "#59636C",
    "muted": "#9AA1A5",
    "grid": "#DDE1E3",
    "paper": "#FFFFFF",
    "soft": "#F5F6F4",
    "summary": "#E7E5E1",
}

# Shared by the prompt-gate figure and the appendix selection table so the two
# can never drift apart.
PROMPT_LABELS = {
    "reaction_h1_direct_v1": "Direct one-session reaction",
    "reaction_h5_delayed_v1": "Delayed five-session reaction",
    "cashflow_revision_h5_v1": "Cash-flow revision",
    "surprise_catalyst_h5_v1": "Surprise catalyst",
    "redteam_consensus_h5_v1": "Red-team consensus",
    "structured_first_reaction_v1": "Structured reaction score",
}


def require_locked_figure_environment() -> None:
    """Fail before export when the active figure toolchain differs from ``uv.lock``."""
    lock_path = ROOT / "uv.lock"
    if not lock_path.exists():
        raise RuntimeError("uv.lock is required; run `uv sync --extra dev --locked`")

    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    locked_versions = {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] in LOCKED_FIGURE_PACKAGES
    }
    problems: list[str] = []
    if sys.version_info[:2] != (3, 12):
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is active; Python 3.12 is required"
        )
    for package_name in LOCKED_FIGURE_PACKAGES:
        expected = locked_versions.get(package_name)
        actual = distribution_version(package_name)
        if expected is None:
            problems.append(f"{package_name} is absent from uv.lock")
        elif actual != expected:
            problems.append(f"{package_name} {actual} is active; uv.lock requires {expected}")
    if problems:
        detail = "\n- ".join(problems)
        raise RuntimeError(
            "Figure generation requires the locked project environment:\n"
            f"- {detail}\n"
            "Run `uv sync --extra dev --locked`, activate `.venv`, and retry."
        )


def read_csv(relative: str) -> pd.DataFrame:
    path = RESULTS / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = frame.loc[frame[column] == value]
    if len(rows) != 1:
        raise ValueError(f"expected one {column}={value!r} row, found {len(rows)}")
    return rows.iloc[0]


def derive_metrics() -> tuple[pd.DataFrame, dict[str, float]]:
    aggregation = read_csv("03_aggregation/development_ic_primary_ranked.csv")
    controls = read_csv(
        "79_fnspid_development_reversal_confound/conditional_price_control_summary.csv"
    )
    transfer = read_csv("71_conditional_negative_share_transfer/conditional_coefficients.csv")
    evaluation = read_csv(
        "75_fnspid_evaluation_beyond_mean_replication/"
        "evaluation_conditional_coefficients.csv"
    )
    temporal_contrast = read_csv(
        "75_fnspid_evaluation_beyond_mean_replication/"
        "evaluation_minus_development_stability_contrast.csv"
    ).iloc[0]
    mde = read_csv("77_null_minimum_detectable_effects/minimum_detectable_effects.csv")
    firm_summary = read_csv("82_lseg33_firm_level_loss_control/evaluation_summary.csv")
    # The saved column is named ``gbp_per_million`` because the frozen spec used a
    # sterling notional. No currency conversion was ever applied: the value is a
    # return difference times a notional of one million. The underlying cash flows
    # are USD-denominated US equity returns, so it is reported here in USD.
    firm_decomposition = read_csv("82_lseg33_firm_level_loss_control/economic_decomposition.csv")
    prompts = read_csv("83_lseg_prompt_economic_value/selection_2024.csv")
    reaction = read_csv("84_lseg_structured_reaction_score/selection_2024.csv")
    multiplicity = read_csv("78_null_results_and_provenance_tables/multiplicity_summary.csv").iloc[0]
    strata = read_csv(
        "74_conditional_and_pressure_robustness_diagnostics/story_count_strata.csv"
    )
    episodes = read_csv(
        "74_conditional_and_pressure_robustness_diagnostics/episode_attribution.csv"
    )
    outcome_legs = read_csv(
        "86_fnspid_outcome_leg_decomposition/coefficient_summary.csv"
    )
    rank_translation = read_csv(
        "86_fnspid_outcome_leg_decomposition/rank_effect_translation.csv"
    )
    power_reconciliation = read_csv(
        "86_fnspid_outcome_leg_decomposition/power_reconciliation.csv"
    )
    fnspid_drawdown = read_csv(
        "86_fnspid_outcome_leg_decomposition/drawdown_reconciliation.csv"
    )
    prior_open = read_csv(
        "87_fnspid_external_review_robustness_pack/prior_open_timing_probe.csv"
    ).iloc[0]
    beta_contrasts = read_csv(
        "87_fnspid_external_review_robustness_pack/beta_robustness_contrasts.csv"
    )
    crash_exclusion = read_csv(
        "87_fnspid_external_review_robustness_pack/fnspid_crash_exclusion.csv"
    ).iloc[0]
    leave_one_out = read_csv(
        "87_fnspid_external_review_robustness_pack/fnspid_leave_one_episode_out.csv"
    )

    full = require_row(controls, "specification", "full_price_path_controls")
    best_break_even = float(aggregation["breakeven_bps_per_side"].max())
    charged_cost = 10.0

    fully_invested = require_row(firm_summary, "arm", "equal_weight_long")
    overlay = require_row(firm_summary, "arm", "selected_firm_news_overlay")
    matched = require_row(
        firm_summary, "arm", "development_symbol_matched_constant"
    )
    ending_wealth = float(
        require_row(
            firm_decomposition,
            "component",
            "compounded_ending_wealth_difference",
        )["gbp_per_million"]
    )
    arithmetic = float(
        require_row(
            firm_decomposition,
            "component",
            "net_arithmetic_difference",
        )["gbp_per_million"]
    )

    lseg_mde = mde.loc[mde["family"] == "LSEG conditional transfer"].copy()
    if set(lseg_mde["test"]) != {"FinBERT backward", "FinBERT recent"}:
        raise ValueError("unexpected LSEG MDE rows")

    component_sum = float(firm_decomposition.iloc[:3]["gbp_per_million"].sum())
    backward_mde_multiple = float(
        require_row(lseg_mde, "test", "FinBERT backward")[
            "nominal_mde_multiple_of_reference"
        ]
    )
    recent_mde_multiple = float(
        require_row(lseg_mde, "test", "FinBERT recent")[
            "nominal_mde_multiple_of_reference"
        ]
    )
    observed_rank = require_row(
        rank_translation, "translation", "pooled_observed_p90_minus_p10"
    )
    session_rank = require_row(
        rank_translation, "translation", "mean_session_specific_p90_minus_p10"
    )
    prospective_power = require_row(
        power_reconciliation, "role", "prospective_before_evaluation"
    )
    realised_power = require_row(
        power_reconciliation, "role", "realised_precision_reconciliation"
    )
    overall_drawdown = require_row(fnspid_drawdown, "period", "2020_2023")
    development_legs = outcome_legs.loc[outcome_legs["regime"] == "development"]
    intraday_leg = require_row(
        development_legs, "outcome_leg", "assigned_session_intraday"
    )
    post_close_leg = require_row(
        development_legs, "outcome_leg", "post_close_overnight"
    )
    beta_adjusted_contrast = require_row(
        beta_contrasts, "specification", "trailing_beta_adjusted"
    )
    beta_control_contrast = require_row(
        beta_contrasts,
        "specification",
        "trailing_beta_adjusted_plus_beta_control",
    )

    values = {
        "rank_effect_p10_to_p90_percentile_points": float(
            observed_rank["fitted_percentile_points"]
        ),
        "rank_effect_mean_session_percentile_points": float(
            session_rank["fitted_percentile_points"]
        ),
        "observed_negative_share_rank_spread": float(observed_rank["rank_change"]),
        "evaluation_conditional_estimate": float(
            require_row(evaluation, "member", "all_firm_days")["estimate"]
        ),
        "evaluation_conditional_q": float(
            require_row(evaluation, "member", "all_firm_days")["q_bh_two_test"]
        ),
        "evaluation_multi_story_estimate": float(
            require_row(evaluation, "member", "multi_story_n_ge_2")["estimate"]
        ),
        "evaluation_minus_development_contrast": float(temporal_contrast["estimate"]),
        "evaluation_minus_development_p": float(temporal_contrast["p_two_sided"]),
        "evaluation_stable_effect_power": float(prospective_power["power"]),
        "evaluation_mde80": float(prospective_power["mde80"]),
        "evaluation_realised_effect_power": float(realised_power["power"]),
        "evaluation_realised_mde80": float(realised_power["mde80"]),
        "evaluation_realised_se_ratio": float(realised_power["se_ratio_to_projection"]),
        "development_intraday_coefficient": float(intraday_leg["estimate"]),
        "development_intraday_q": float(intraday_leg["q_bh_three_leg"]),
        "development_post_close_coefficient": float(post_close_leg["estimate"]),
        "development_post_close_q": float(post_close_leg["q_bh_three_leg"]),
        "development_prior_open_coefficient": float(prior_open["estimate"]),
        "development_prior_open_p": float(prior_open["p_two_sided"]),
        "beta_adjusted_era_contrast": float(beta_adjusted_contrast["estimate"]),
        "beta_adjusted_era_contrast_q": float(
            beta_adjusted_contrast["q_bh_two_beta_specs"]
        ),
        "beta_control_era_contrast": float(beta_control_contrast["estimate"]),
        "beta_control_era_contrast_q": float(
            beta_control_contrast["q_bh_two_beta_specs"]
        ),
        "fnspid_control_max_drawdown": float(overall_drawdown["control_max_drawdown"]),
        "fnspid_overlay_max_drawdown": float(overall_drawdown["overlay_max_drawdown"]),
        "fnspid_drawdown_relative_improvement": float(
            overall_drawdown["drawdown_relative_improvement"]
        ),
        "fnspid_episode_leave_one_out_min_ci_low": float(leave_one_out["ci_low"].min()),
        "fnspid_crash_exclusion_ci_low": float(crash_exclusion["ci_low"]),
        "fnspid_crash_exclusion_p": float(crash_exclusion["p_two_sided"]),
        "maximum_break_even_bps_per_side": best_break_even,
        "charged_cost_multiple_of_best_break_even": charged_cost / best_break_even,
        "lseg_backward_mde_multiple": backward_mde_multiple,
        "lseg_recent_mde_multiple": recent_mde_multiple,
        "lseg_backward_information_multiple_for_fnspid_sized_mde": (
            backward_mde_multiple**2
        ),
        "lseg_recent_information_multiple_for_fnspid_sized_mde": (
            recent_mde_multiple**2
        ),
        "lseg_firm_overlay_drawdown_improvement_vs_fully_invested_pp": (
            abs(float(fully_invested["max_drawdown"]))
            - abs(float(overlay["max_drawdown"]))
        )
        * 100,
        "lseg_firm_overlay_drawdown_disadvantage_vs_matched_pp": (
            abs(float(overlay["max_drawdown"])) - abs(float(matched["max_drawdown"]))
        )
        * 100,
        "lseg_firm_overlay_downside_deviation_improvement_vs_fully_invested_pp": (
            float(fully_invested["annual_downside_deviation"])
            - float(overlay["annual_downside_deviation"])
        )
        * 100,
        "lseg_firm_overlay_downside_deviation_disadvantage_vs_matched_pp": (
            float(overlay["annual_downside_deviation"])
            - float(matched["annual_downside_deviation"])
        )
        * 100,
        "lseg_firm_overlay_turnover_multiple_vs_matched": float(overlay["annual_turnover"])
        / float(matched["annual_turnover"]),
        "lseg_firm_overlay_ending_wealth_difference_usd_per_million": ending_wealth,
        "lseg_firm_overlay_compounding_residual_usd_per_million": ending_wealth - arithmetic,
        "lseg_firm_overlay_arithmetic_reconciliation_usd_per_million": (
            arithmetic - component_sum
        ),
        "best_prompt_variant_net_bps_session": float(prompts["mean_net_bps_session"].max()),
        "structured_reaction_net_bps_session": float(reaction.iloc[0]["mean_net_bps_session"]),
        "multiplicity_survivor_share": float(multiplicity["bh_survivors"])
        / float(multiplicity["tests_or_reported_decisions"]),
        "multiplicity_all_null_family_share": float(multiplicity["all_null_families"])
        / float(multiplicity["families"]),
        "conditional_estimate_min_two_stories": float(
            require_row(strata, "stratum", "n >= 2")["estimate"]
        ),
        "conditional_estimate_min_three_stories": float(
            require_row(strata, "stratum", "n >= 3")["estimate"]
        ),
        "conditional_retained_share_min_three_stories": (
            float(require_row(strata, "stratum", "n >= 3")["estimate"])
            / float(require_row(strata, "stratum", "all firm-days (frozen)")["estimate"])
        ),
        "lseg_backward_top_two_episode_downside_share": float(
            episodes["downside_share"].nlargest(2).sum()
        ),
    }

    # Independent consistency checks for the highest-impact arithmetic.
    if not np.isclose(component_sum, arithmetic, atol=1.0):
        raise ValueError("LSEG economic components do not reconcile within USD 1 per million")
    if not np.isclose(best_break_even, 0.6248161477733559, atol=1e-9):
        raise ValueError("best break-even cost has drifted")
    if not np.isclose(float(full["estimate"]), -0.009140, atol=1e-6):
        raise ValueError("full-control coefficient has drifted")
    if len(transfer) != 5:
        raise ValueError("conditional transfer row count has drifted")
    if not np.isclose(values["evaluation_conditional_estimate"], 0.00583332, atol=1e-8):
        raise ValueError("evaluation conditional coefficient has drifted")
    if not np.isclose(
        values["evaluation_minus_development_contrast"], 0.01414337, atol=1e-8
    ):
        raise ValueError("evaluation-minus-development contrast has drifted")
    if not np.isclose(values["conditional_estimate_min_three_stories"], -0.001828, atol=1e-6):
        raise ValueError("multi-story attenuation diagnostic has drifted")
    if not np.isclose(
        values["rank_effect_p10_to_p90_percentile_points"], -0.5123741194, atol=1e-9
    ):
        raise ValueError("observed tied-rank translation has drifted")
    if not np.isclose(values["development_intraday_coefficient"], -0.010291, atol=1e-6):
        raise ValueError("development intraday coefficient has drifted")
    if values["fnspid_overlay_max_drawdown"] >= values["fnspid_control_max_drawdown"]:
        raise ValueError("expected adverse FNSPID overlay maximum drawdown")

    records = [
        {
            "metric": "observed 10th-to-90th negative-share rank translation",
            "value": values["rank_effect_p10_to_p90_percentile_points"],
            "unit": "next-return-rank percentile points",
            "formula": "observed pooled tied-rank spread x full-control coefficient x 100",
            "source": "Notebook 86 correction using Notebook 79 model rows",
        },
        {
            "metric": "FNSPID testing-period conditional coefficient",
            "value": values["evaluation_conditional_estimate"],
            "unit": "rank coefficient",
            "formula": "mean daily all-firm-day coefficient",
            "source": "Notebook 75 frozen testing-period opening",
        },
        {
            "metric": "testing-minus-training coefficient contrast",
            "value": values["evaluation_minus_development_contrast"],
            "unit": "rank coefficient difference",
            "formula": "testing-period mean beta - training-period mean beta",
            "source": "Notebook 75 predeclared HAC(5) stability contrast",
        },
        {
            "metric": "prospective power for stable training-period effect",
            "value": values["evaluation_stable_effect_power"],
            "unit": "approximate power",
            "formula": "scaled training-period HAC SE at corrected two-test threshold",
            "source": "Notebook 75 pre-outcome amendment",
        },
        {
            "metric": "realised-precision power for stable training-period effect",
            "value": values["evaluation_realised_effect_power"],
            "unit": "ex-post approximate power",
            "formula": "observed testing-period HAC SE at corrected two-test threshold",
            "source": "Notebook 86 power reconciliation",
        },
        {
            "metric": "training-period assigned-session intraday coefficient",
            "value": values["development_intraday_coefficient"],
            "unit": "rank coefficient",
            "formula": "mean daily coefficient on open-to-close abnormal return rank",
            "source": "Notebook 86 outcome-leg decomposition",
        },
        {
            "metric": "training-period previous-open timing-probe coefficient",
            "value": values["development_prior_open_coefficient"],
            "unit": "rank coefficient",
            "formula": "mean daily coefficient on previous-open-to-assigned-open abnormal return rank",
            "source": "Notebook 87 additional timing and sensitivity checks",
        },
        {
            "metric": "FNSPID overlay maximum drawdown",
            "value": values["fnspid_overlay_max_drawdown"],
            "unit": "return",
            "formula": "minimum cumulative wealth drawdown in 2020-2023",
            "source": "Notebook 86 reconciliation of Notebook 24 paths",
        },
        {
            "metric": "charged cost relative to best break-even",
            "value": values["charged_cost_multiple_of_best_break_even"],
            "unit": "multiple",
            "formula": "10 bps / maximum aggregation break-even bps",
            "source": "Notebook 03 aggregation family",
        },
        {
            "metric": "LSEG earlier-period nominal MDE relative to FNSPID estimate",
            "value": values["lseg_backward_mde_multiple"],
            "unit": "multiple",
            "formula": "nominal 80% MDE / abs(FNSPID coefficient)",
            "source": "Notebook 77 MDE table",
        },
        {
            "metric": "LSEG later-period nominal MDE relative to FNSPID estimate",
            "value": values["lseg_recent_mde_multiple"],
            "unit": "multiple",
            "formula": "nominal 80% MDE / abs(FNSPID coefficient)",
            "source": "Notebook 77 MDE table",
        },
        {
            "metric": "LSEG earlier-period information multiple for an FNSPID-sized MDE",
            "value": values[
                "lseg_backward_information_multiple_for_fnspid_sized_mde"
            ],
            "unit": "approximate information multiple",
            "formula": "square of earlier-period nominal MDE multiple",
            "source": "Notebook 77 MDE table; square-root precision scaling",
        },
        {
            "metric": "LSEG later-period information multiple for an FNSPID-sized MDE",
            "value": values[
                "lseg_recent_information_multiple_for_fnspid_sized_mde"
            ],
            "unit": "approximate information multiple",
            "formula": "square of later-period nominal MDE multiple",
            "source": "Notebook 77 MDE table; square-root precision scaling",
        },
        {
            "metric": "firm-level rule drawdown improvement versus fully invested",
            "value": values[
                "lseg_firm_overlay_drawdown_improvement_vs_fully_invested_pp"
            ],
            "unit": "percentage points",
            "formula": "abs(full drawdown) - abs(rule drawdown)",
            "source": "Notebook 82 testing-period summary",
        },
        {
            "metric": "firm-level rule drawdown disadvantage versus matched constant",
            "value": values[
                "lseg_firm_overlay_drawdown_disadvantage_vs_matched_pp"
            ],
            "unit": "percentage points",
            "formula": "abs(rule drawdown) - abs(matched drawdown)",
            "source": "Notebook 82 testing-period summary",
        },
        {
            "metric": "firm-level rule annual turnover relative to matched constant",
            "value": values["lseg_firm_overlay_turnover_multiple_vs_matched"],
            "unit": "multiple",
            "formula": "rule annual turnover / matched annual turnover",
            "source": "Notebook 82 testing-period summary",
        },
        {
            "metric": "firm-level rule ending wealth difference",
            "value": values[
                "lseg_firm_overlay_ending_wealth_difference_usd_per_million"
            ],
            "unit": "USD per USD 1m",
            "formula": "saved compounded ending wealth difference",
            "source": "Notebook 82 economic decomposition",
        },
        {
            "metric": "conditional coefficient on firm-days with at least three stories",
            "value": values["conditional_estimate_min_three_stories"],
            "unit": "rank coefficient",
            "formula": "baseline specification restricted to n >= 3",
            "source": "Notebook 74 story-count strata",
        },
        {
            "metric": "share of the full-sample coefficient retained at n >= 3",
            "value": values["conditional_retained_share_min_three_stories"],
            "unit": "share",
            "formula": "n >= 3 estimate / all-firm-day estimate",
            "source": "Notebook 74 story-count strata",
        },
        {
            "metric": "share of LSEG earlier-period downside reduction from the two largest episodes",
            "value": values["lseg_backward_top_two_episode_downside_share"],
            "unit": "share",
            "formula": "sum of the two largest episode downside shares",
            "source": "Notebook 74 episode attribution",
        },
    ]
    return pd.DataFrame.from_records(records), values


def dissertation_style() -> None:
    """Set the compact editorial-atlas style used by every report figure."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "figure.facecolor": COLOUR["paper"],
            "axes.facecolor": COLOUR["paper"],
            "savefig.facecolor": COLOUR["paper"],
            "font.family": "sans-serif",
            # DejaVu Sans ships with the Matplotlib version pinned in uv.lock.
            # Use only its concrete normal and bold faces: relative 500/600
            # weights trigger version-dependent fallback and alter figure bytes.
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.9,
            "legend.fontsize": 7.5,
            "axes.labelcolor": COLOUR["secondary"],
            "axes.titlecolor": COLOUR["ink"],
            "text.color": COLOUR["ink"],
            "xtick.color": COLOUR["secondary"],
            "ytick.color": COLOUR["secondary"],
            "axes.edgecolor": COLOUR["muted"],
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": COLOUR["grid"],
            "grid.linewidth": 0.50,
            "grid.alpha": 1.0,
            "lines.linewidth": 1.35,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def bare_axes(ax: Any, *, categorical_y: bool = True) -> None:
    """Keep only the value-axis scaffolding needed for a comparison."""
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    if categorical_y:
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLOUR["muted"])


def finish(fig: Any, output: Path) -> None:
    """Export fixed-canvas PNG and vector companions for LaTeX."""
    fig.savefig(output, facecolor=COLOUR["paper"])
    fig.savefig(
        output.with_suffix(".pdf"),
        facecolor=COLOUR["paper"],
        metadata={
            "Creator": "news-sentiment-beyond-mean figure generator",
            # Suppress the wall-clock default so equivalent runs are byte-stable.
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def draw_interval_rows(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    positions: list[float] | np.ndarray | None = None,
    colour: str = COLOUR["navy"],
    marker: str = "o",
    value_fmt: str | None = None,
    filled: bool = True,
) -> None:
    """Draw interval rows with consistent marks and optional value labels."""
    y = np.asarray(
        positions if positions is not None else np.arange(len(rows))[::-1],
        dtype=float,
    )
    for pos, row in zip(y, rows, strict=True):
        row_colour = str(row.get("colour", colour))
        row_marker = str(row.get("marker", marker))
        row_filled = bool(row.get("filled", filled))
        ax.errorbar(
            row["estimate"],
            pos,
            xerr=[[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]],
            fmt=row_marker,
            color=row_colour,
            markerfacecolor=row_colour if row_filled else COLOUR["paper"],
            markeredgecolor=row_colour,
            markeredgewidth=1.0,
            markersize=float(row.get("markersize", 4.6)),
            capsize=2.4,
            elinewidth=float(row.get("linewidth", 1.35)),
            zorder=3,
        )
        if value_fmt is not None:
            ax.text(
                1.02,
                pos,
                value_fmt.format(row["estimate"]),
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=7.5,
                color=COLOUR["ink"] if row.get("focal") else COLOUR["secondary"],
                fontweight="bold" if row.get("focal") else "normal",
                clip_on=False,
            )


def aggregation_family_figure(output: Path) -> None:
    """Ranked confidence rail with a quiet focal band for the sole survivor."""
    frame = read_csv("03_aggregation/development_ic_primary_ranked.csv").copy()
    frame["se"] = frame["ic"].abs() / frame["t"].abs()
    frame = pd.concat(
        [
            frame.loc[frame["bh_reject_q05"]],
            frame.loc[~frame["bh_reject_q05"]].sort_values("ic", ascending=False),
        ],
        ignore_index=True,
    )

    rows = [
        {
            "label": display_label(r["aggregator"]),
            "estimate": float(r["ic"]),
            "low": float(r["ic"]) - 1.959964 * float(r["se"]),
            "high": float(r["ic"]) + 1.959964 * float(r["se"]),
            "focal": bool(r["bh_reject_q05"]),
            "colour": COLOUR["navy"] if bool(r["bh_reject_q05"]) else COLOUR["muted"],
            "filled": bool(r["bh_reject_q05"]),
            "markersize": 5.4 if bool(r["bh_reject_q05"]) else 4.2,
            "linewidth": 1.7 if bool(r["bh_reject_q05"]) else 1.15,
        }
        for _, r in frame.iterrows()
    ]

    dissertation_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.05))
    y = np.arange(len(rows))[::-1]
    ax.axhspan(y[0] - 0.44, y[0] + 0.44, color=COLOUR["navy_light"], zorder=0)
    draw_interval_rows(ax, rows, positions=y, value_fmt="{:+.5f}")
    span = max(r["high"] for r in rows) - min(r["low"] for r in rows)
    ax.set_xlim(
        min(r["low"] for r in rows) - span * 0.07,
        max(r["high"] for r in rows) + span * 0.07,
    )
    ax.axvline(0, color=COLOUR["secondary"], lw=0.8, zorder=1)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    bare_axes(ax)
    ax.text(
        1.02,
        0.985,
        "ESTIMATE",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLOUR["secondary"],
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        1.02,
        y[0] - 0.25,
        "BH survivor",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        fontsize=6.8,
        color=COLOUR["ink"],
        fontweight="bold",
        clip_on=False,
    )
    ax.set_xlabel("Mean daily cross-sectional Spearman IC")
    fig.subplots_adjust(left=0.29, right=0.82, bottom=0.16, top=0.98)
    finish(fig, output)


def estimate_stability_figure(output: Path) -> None:
    """Paired specification paths for controls and story-count attenuation."""
    controls = read_csv(
        "79_fnspid_development_reversal_confound/conditional_price_control_summary.csv"
    )
    strata = read_csv(
        "74_conditional_and_pressure_robustness_diagnostics/story_count_strata.csv"
    )

    control_labels = {
        "published_baseline_full_sample": "Baseline",
        "baseline_price_complete_sample": "Price-complete sample",
        "plus_lagged_return_1": "+ 1-session return",
        "plus_lagged_returns_1_and_5": "+ 1- and 5-session returns",
        "full_price_path_controls": "+ 20-session volatility",
    }
    control_rows = [
        {
            "label": control_labels[r["specification"]],
            "estimate": float(r["estimate"]),
            "low": float(r["ci_low"]),
            "high": float(r["ci_high"]),
            "focal": r["specification"] == "full_price_path_controls",
            "colour": COLOUR["navy"],
            "marker": "o",
            "filled": r["specification"] == "full_price_path_controls",
        }
        for _, r in controls.iterrows()
    ]
    stratum_labels = {
        "all firm-days (frozen)": "All firm-days",
        "n >= 2": "At least two stories",
        "n >= 3": "At least three stories",
    }
    stratum_rows = [
        {
            "label": stratum_labels[r["stratum"]],
            "estimate": float(r["estimate"]),
            "low": float(r["ci_low"]),
            "high": float(r["ci_high"]),
            "focal": r["stratum"] == "n >= 3",
            "colour": COLOUR["rust"],
            "marker": "s",
            "filled": r["stratum"] == "n >= 3",
        }
        for _, r in strata.iterrows()
    ]

    all_rows = control_rows + stratum_rows
    low = min(r["low"] for r in all_rows)
    high = max(r["high"] for r in all_rows)
    span = high - low
    limits = (low - span * 0.08, high + span * 0.08)
    annotation_band_top = low - span * 0.012
    annotation_y = (limits[0] + annotation_band_top) / 2
    panel_data = [
        (
            control_rows,
            "(a) Recent-price controls",
            ["Baseline", "Complete\nsample", "+1", "+1 & +5", "+Volatility"],
            COLOUR["navy"],
            "o",
            "-",
        ),
        (
            stratum_rows,
            "(b) Same-day story count",
            ["All", "$n\\geq2$", "$n\\geq3$"],
            COLOUR["rust"],
            "s",
            (0, (3, 2)),
        ),
    ]

    dissertation_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, 3.20),
        sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.12},
    )
    for ax, (rows, title, labels, colour, marker, line_style) in zip(
        axes, panel_data, strict=True
    ):
        x = np.arange(len(rows))
        estimates = np.array([r["estimate"] for r in rows])
        low_err = estimates - np.array([r["low"] for r in rows])
        high_err = np.array([r["high"] for r in rows]) - estimates
        ax.axhspan(
            limits[0],
            annotation_band_top,
            color=COLOUR["soft"],
            zorder=0,
        )
        ax.axhline(0, color=COLOUR["secondary"], lw=0.8, zorder=1)
        ax.plot(
            x,
            estimates,
            color=colour,
            lw=1.25,
            ls=line_style,
            zorder=2,
        )
        ax.errorbar(
            x,
            estimates,
            yerr=[low_err, high_err],
            fmt=marker,
            color=colour,
            markerfacecolor=colour if marker == "o" else COLOUR["paper"],
            markeredgecolor=colour,
            markeredgewidth=1.05,
            markersize=5.0,
            capsize=2.4,
            elinewidth=1.25,
            zorder=3,
        )
        for xpos, estimate in zip(x, estimates, strict=True):
            ax.text(
                xpos,
                annotation_y,
                f"{estimate:+.4f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=COLOUR["secondary"],
                fontweight="bold",
            )
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.45, len(rows) - 0.55)
        ax.set_ylim(*limits)
        ax.set_title(title, loc="left", pad=7)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0, pad=5, labelsize=7.0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Negative-share coefficient (rank units)")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.22, top=0.90)
    finish(fig, output)


def double_sort_figure(output: Path) -> None:
    """Ordered quintile profiles with distinct pooled-estimate cards."""
    quintiles = read_csv(
        "76_fnspid_development_negative_share_double_sort/quintile_spread_summary.csv"
    )
    pooled = read_csv(
        "76_fnspid_development_negative_share_double_sort/pooled_spread_summary.csv"
    )

    samples = [
        ("all_firm_days", "(a) All firm-days", COLOUR["navy"], "o"),
        ("multi_story_n_ge_2", "(b) At least two stories", COLOUR["rust"], "s"),
    ]
    tick_labels = ["Most\nnegative", "Q2", "Q3", "Q4", "Most\npositive"]

    dissertation_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, 3.15),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    limits = (-5.0, 6.7)
    x = np.arange(1, 6)
    for ax, (sample, title, colour, marker) in zip(axes, samples, strict=True):
        block = quintiles.loc[quintiles["sample"] == sample].sort_values("mean_quintile")
        pooled_row = require_row(pooled, "sample", sample)
        estimates = block["mean_bps"].to_numpy(dtype=float)
        lows = block["ci_low_bps"].to_numpy(dtype=float)
        highs = block["ci_high_bps"].to_numpy(dtype=float)
        ax.axhspan(limits[0], 0, color=COLOUR["navy_light"], alpha=0.28, zorder=0)
        ax.axhline(0, color=COLOUR["secondary"], lw=0.8, zorder=1)
        ax.plot(
            x,
            estimates,
            color=colour,
            lw=1.15,
            ls="-" if marker == "o" else (0, (3, 2)),
            alpha=0.86,
            zorder=2,
        )
        ax.errorbar(
            x,
            estimates,
            yerr=[estimates - lows, highs - estimates],
            fmt=marker,
            color=colour,
            markerfacecolor=colour if marker == "o" else COLOUR["paper"],
            markeredgecolor=colour,
            markeredgewidth=1.05,
            markersize=4.8,
            capsize=2.3,
            elinewidth=1.15,
            zorder=3,
        )
        pooled_estimate = float(pooled_row["mean_bps"])
        pooled_low = float(pooled_row["ci_low_bps"])
        pooled_high = float(pooled_row["ci_high_bps"])
        ax.text(
            0.96,
            0.96,
            "POOLED HIGH–LOW",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
            color=COLOUR["secondary"],
            fontweight="bold",
            bbox={"facecolor": COLOUR["paper"], "edgecolor": "none", "pad": 1.2},
        )
        ax.text(
            0.96,
            0.875,
            f"{pooled_estimate:+.2f} bps  [{pooled_low:+.2f}, {pooled_high:+.2f}]",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
            color=COLOUR["ink"],
            fontweight="bold",
            bbox={"facecolor": COLOUR["paper"], "edgecolor": "none", "pad": 1.2},
        )
        ax.set_xlim(0.45, 5.55)
        ax.set_ylim(*limits)
        ax.set_xticks(x, tick_labels)
        ax.set_title(title, loc="left", pad=5)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", length=0, labelsize=6.6, pad=5)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Spread (bps per session)")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.supxlabel("Mean-sentiment quintile", y=0.02, fontsize=8.5, color=COLOUR["secondary"])
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.91)
    finish(fig, output)


def timing_placebo_figure(output: Path) -> None:
    """Percentile tracks with separate exact-value and BH-decision columns."""
    frame = read_csv("45_sentiment_downside_timing_placebo/timing_results.csv").copy()

    dissertation_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.25))
    y = np.arange(len(frame))[::-1]
    ax.axvspan(95, 100, color=COLOUR["rust_light"], alpha=0.55, zorder=0)
    ax.axvline(95, color=COLOUR["rust"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    for pos, (_, r) in zip(y, frame.iterrows(), strict=True):
        percentile = float(r["downside_percentile_nonzero_shifts"]) * 100
        passes = bool(r["downside_timing_gate"])
        colour = COLOUR["navy"] if passes else COLOUR["navy_mid"]
        ax.hlines(pos, 0, 100, color=COLOUR["grid"], lw=4.2, zorder=1)
        ax.hlines(
            pos,
            0,
            percentile,
            color=colour,
            lw=4.2 if passes else 2.4,
            zorder=2,
        )
        ax.scatter(
            percentile,
            pos,
            s=38 if passes else 30,
            facecolor=colour if passes else COLOUR["paper"],
            edgecolor=colour,
            linewidth=1.1,
            zorder=3,
        )
        ax.text(
            1.03,
            pos,
            f"{percentile:.1f}%",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=7.3,
            color=COLOUR["ink"] if passes else COLOUR["secondary"],
            fontweight="bold" if passes else "normal",
            clip_on=False,
        )
        ax.text(
            1.23,
            pos,
            f"$q={float(r['downside_q_bh']):.3f}$  " + ("UNUSUAL" if passes else "not unusual"),
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=6.8,
            color=COLOUR["navy"] if passes else COLOUR["secondary"],
            fontweight="bold" if passes else "normal",
            clip_on=False,
        )

    ax.set_yticks(y, [str(r) for r in frame["regime"]])
    bare_axes(ax)
    ax.set_xlim(0, 101)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.1, len(frame) - 0.9)
    ax.set_xlabel("Same-session risk schedule percentile among circular shifts")
    ax.text(
        0.90,
        1.03,
        "UPPER 5% ZONE",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=COLOUR["secondary"],
        fontweight="bold",
    )
    ax.text(
        1.23,
        1.03,
        "BH ALIGNMENT CHECK",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=COLOUR["secondary"],
        fontweight="bold",
        clip_on=False,
    )
    fig.subplots_adjust(left=0.25, right=0.70, bottom=0.25, top=0.86)
    finish(fig, output)


def risk_regime_figure(output: Path) -> None:
    """Chronological diverging columns for the unstable overlay return."""
    hysteresis = read_csv("24_fnspid_har_sentiment_hysteresis/temporal_results.csv")
    walkforward = read_csv("25_fnspid_har_sentiment_walkforward/temporal_results.csv")

    def delta(frame: pd.DataFrame, period: str, overlay: str, base: str) -> float:
        rows = frame.loc[frame["period"] == period]
        top = require_row(rows, "arm", overlay)["mean_net"]
        bottom = require_row(rows, "arm", base)["mean_net"]
        return (float(top) - float(bottom)) * 1e4

    periods = [
        ("2013–2015", delta(walkforward, "2013_2015", "walkforward_har_sentiment", "walkforward_har")),
        ("2016–2019", delta(walkforward, "2016_2019", "walkforward_har_sentiment", "walkforward_har")),
        ("2020", delta(hysteresis, "2020", "har_sentiment", "control_har_target")),
        ("2021–2023", delta(hysteresis, "2021_2023", "har_sentiment", "control_har_target")),
    ]

    dissertation_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.65))
    x = np.arange(len(periods))
    values = np.array([v for _, v in periods])
    for pos, value in zip(x, values, strict=True):
        positive = value >= 0
        colour = COLOUR["navy"] if positive else COLOUR["rust_light"]
        edge = COLOUR["navy"] if positive else COLOUR["rust"]
        ax.bar(
            pos,
            value,
            width=0.58,
            color=colour,
            edgecolor=edge,
            linewidth=1.0,
            hatch=None if positive else "///",
            zorder=3,
        )
        ax.text(
            pos,
            value + 0.10 if positive else value - 0.10,
            f"{value:+.2f} bps",
            ha="center",
            va="bottom" if positive else "top",
            fontsize=7.8,
            color=COLOUR["ink"],
            fontweight="bold",
        )
    ax.axhline(0, color=COLOUR["secondary"], lw=0.8, zorder=2)
    ax.set_xticks(x, [label for label, _ in periods])
    ax.set_xlim(-0.55, len(periods) - 0.45)
    ax.set_ylim(-0.95, 3.20)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Overlay minus base\n(bps per session)")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.20, top=0.97)
    finish(fig, output)


def coefficient_drift_figure(output: Path) -> None:
    """Calendar-year mean of the daily baseline coefficient across both eras.

    Descriptive only. Annual standard errors are plain rather than HAC because
    the coefficient series is close to white noise once regime means are removed
    (first-order autocorrelation 0.0054), and each annual interval is wide enough
    that the figure dates no break.
    """
    base = "75_fnspid_evaluation_beyond_mean_replication/"
    development = read_csv(base + "development_daily_coefficients_all_firm_days.csv")
    evaluation = read_csv(base + "evaluation_daily_coefficients_all_firm_days.csv")
    development["era"] = "development"
    evaluation["era"] = "evaluation"
    daily = pd.concat([development, evaluation], ignore_index=True)
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    daily["year"] = daily["session_date"].dt.year
    if len(daily) != 3262:
        raise ValueError("stacked coefficient series should hold 3,262 sessions")

    grouped = daily.groupby("year", sort=True)["beta_negative_share"]
    years = np.array(sorted(daily["year"].unique()))
    means = grouped.mean().to_numpy()
    errors = (grouped.std(ddof=1) / np.sqrt(grouped.size())).to_numpy() * 1.96
    era_of_year = daily.groupby("year")["era"].first().to_numpy()
    is_eval = era_of_year == "evaluation"

    dissertation_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.95))
    ax.axhline(0, color=COLOUR["ink"], lw=0.9, zorder=2)
    ax.axvspan(2019.5, years.max() + 0.5, color=COLOUR["soft"], zorder=0)
    ax.text(
        2019.62,
        0.0335,
        "frozen testing period",
        fontsize=7.2,
        color=COLOUR["secondary"],
        va="top",
    )
    for year, mean, err, evaluation_year in zip(
        years, means, errors, is_eval, strict=True
    ):
        colour = COLOUR["rust"] if evaluation_year else COLOUR["navy"]
        ax.errorbar(
            year,
            mean,
            yerr=err,
            fmt="D" if evaluation_year else "o",
            markersize=4.6,
            color=colour,
            ecolor=colour,
            elinewidth=1.1,
            capsize=2.6,
            markerfacecolor=COLOUR["paper"] if evaluation_year else colour,
            markeredgewidth=1.2,
            zorder=4,
        )
    for label, value, colour, style in (
        ("training-period mean", float(development["beta_negative_share"].mean()), COLOUR["navy"], "--"),
        ("testing-period mean", float(evaluation["beta_negative_share"].mean()), COLOUR["rust"], ":"),
    ):
        ax.axhline(value, color=colour, lw=1.0, ls=style, zorder=3, label=label)
    ax.set_xticks(years, [str(year) for year in years], rotation=45)
    ax.set_xlim(years.min() - 0.6, years.max() + 0.6)
    ax.set_ylabel("Mean daily coefficient")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=7.6, handlelength=2.4)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.22, top=0.97)
    finish(fig, output)


def prompt_gate_figure(output: Path) -> None:
    """Ordered two-stage loss bars separating gross loss from cost drag."""
    prompts = read_csv("83_lseg_prompt_economic_value/selection_2024.csv")
    reaction = read_csv("84_lseg_structured_reaction_score/selection_2024.csv")

    labels = PROMPT_LABELS
    records = [
        {
            "label": labels[r["prompt_id"]],
            "gross": float(r["mean_gross_bps_session"]),
            "net": float(r["mean_net_bps_session"]),
        }
        for _, r in pd.concat([prompts, reaction], ignore_index=True).iterrows()
    ]
    records.sort(key=lambda r: r["net"], reverse=True)

    dissertation_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.05))
    y = np.arange(len(records))[::-1]
    for pos, row in zip(y, records, strict=True):
        gross_width = abs(row["gross"])
        cost_width = row["gross"] - row["net"]
        ax.barh(
            pos,
            gross_width,
            left=row["gross"],
            height=0.50,
            color=COLOUR["summary"],
            edgecolor=COLOUR["secondary"],
            linewidth=0.7,
            zorder=2,
        )
        ax.barh(
            pos,
            cost_width,
            left=row["net"],
            height=0.50,
            color=COLOUR["rust_light"],
            edgecolor=COLOUR["rust"],
            linewidth=0.8,
            hatch="///",
            zorder=3,
        )
        ax.scatter(row["net"], pos, s=22, color=COLOUR["rust"], zorder=4)
        ax.text(
            row["net"],
            pos + 0.36,
            f"{row['net']:.2f} net",
            va="bottom",
            ha="center",
            fontsize=6.9,
            color=COLOUR["ink"],
            fontweight="bold",
        )
    ax.axvline(0, color=COLOUR["secondary"], lw=0.8, zorder=1)
    ax.set_yticks(y, [r["label"] for r in records])
    bare_axes(ax)
    ax.set_xlabel("Mean 2024 portfolio return (bps per session)")
    ax.set_xlim(-22.2, 1.8)
    ax.set_ylim(-0.55, len(records) - 0.05)
    legend_handles = [
        Patch(
            facecolor=COLOUR["summary"],
            edgecolor=COLOUR["secondary"],
            linewidth=0.7,
            label="Gross loss",
        ),
        Patch(
            facecolor=COLOUR["rust_light"],
            edgecolor=COLOUR["rust"],
            linewidth=0.8,
            hatch="///",
            label="Additional cost drag",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.0,
        frameon=False,
    )
    ax.text(
        0,
        len(records) - 0.12,
        "pre-specified criteria",
        ha="right",
        va="top",
        fontsize=6.9,
        color=COLOUR["secondary"],
    )
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.18, top=0.96)
    finish(fig, output)


def coefficient_figure(output: Path) -> None:
    """Vertical interval atlas across FNSPID eras and LSEG blocks."""
    controls = read_csv(
        "79_fnspid_development_reversal_confound/conditional_price_control_summary.csv"
    )
    transfer = read_csv("71_conditional_negative_share_transfer/conditional_coefficients.csv")
    mde = read_csv("77_null_minimum_detectable_effects/minimum_detectable_effects.csv")
    evaluation = read_csv(
        "75_fnspid_evaluation_beyond_mean_replication/"
        "evaluation_conditional_coefficients.csv"
    )
    evaluation_power = read_csv(
        "75_fnspid_evaluation_beyond_mean_replication/prospective_power_context.csv"
    ).iloc[0]

    baseline = require_row(controls, "specification", "published_baseline_full_sample")
    full = require_row(controls, "specification", "full_price_path_controls")
    lseg = transfer.loc[
        (transfer["source"] == "LSEG") & (transfer["scorer"] == "finbert")
    ].copy()
    mde_rows = mde.loc[mde["family"] == "LSEG conditional transfer"]
    multiples = {
        "backward": float(
            require_row(mde_rows, "test", "FinBERT backward")[
                "nominal_mde_multiple_of_reference"
            ]
        ),
        "recent": float(
            require_row(mde_rows, "test", "FinBERT recent")[
                "nominal_mde_multiple_of_reference"
            ]
        ),
    }

    rows = [
        {
            "label": "FNSPID training-period baseline",
            "row": baseline,
            "colour": COLOUR["navy"],
            "marker": "o",
            "filled": True,
            "power": "—",
        },
        {
            "label": "FNSPID training period + controls",
            "row": full,
            "colour": COLOUR["navy"],
            "marker": "o",
            "filled": True,
            "power": "—",
        },
        {
            "label": "FNSPID testing period",
            "row": require_row(evaluation, "member", "all_firm_days"),
            "colour": COLOUR["navy"],
            "marker": "D",
            "filled": False,
            "power": (
                f"{float(evaluation_power['approximate_power_for_stable_development_effect']) * 100:.0f}% power"
            ),
        },
        {
            "label": "LSEG earlier period",
            "row": require_row(lseg, "block", "backward"),
            "colour": COLOUR["rust"],
            "marker": "s",
            "filled": False,
            "power": f"MDE {multiples['backward']:.1f}×",
        },
        {
            "label": "LSEG later period",
            "row": require_row(lseg, "block", "recent"),
            "colour": COLOUR["rust"],
            "marker": "s",
            "filled": False,
            "power": f"MDE {multiples['recent']:.1f}×",
        },
    ]

    dissertation_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.05))
    x = np.arange(len(rows))
    estimate_label_y = -0.109
    context_label_y = 0.080
    for pos, item in zip(x, rows, strict=True):
        r = item["row"]
        colour = str(item["colour"])
        ax.errorbar(
            pos,
            float(r["estimate"]),
            yerr=[
                [float(r["estimate"]) - float(r["ci_low"])],
                [float(r["ci_high"]) - float(r["estimate"])],
            ],
            fmt=str(item["marker"]),
            color=colour,
            markerfacecolor=colour if bool(item["filled"]) else COLOUR["paper"],
            markeredgecolor=colour,
            markeredgewidth=1.0,
            markersize=4.8,
            capsize=2.5,
            lw=1.4,
            zorder=3,
        )
        ax.text(
            pos,
            estimate_label_y,
            f"{float(r['estimate']):+.4f}",
            va="center",
            ha="center",
            fontsize=6.6,
            color=COLOUR["secondary"],
            fontweight="bold",
        )
        if item["power"] != "—":
            ax.text(
                pos,
                context_label_y,
                str(item["power"]),
                va="center",
                ha="center",
                fontsize=6.8,
                color=colour,
                fontweight="bold",
            )

    ax.axhline(0, color=COLOUR["secondary"], lw=0.8, zorder=1)
    ax.axvline(2.5, color=COLOUR["grid"], lw=0.8, zorder=1)
    ax.axvspan(-0.45, 2.45, color=COLOUR["navy_light"], alpha=0.18, zorder=0)
    ax.axvspan(2.55, 4.45, color=COLOUR["rust_light"], alpha=0.16, zorder=0)
    ax.set_xticks(
        x,
        [
            "Training-period\nbaseline",
            "Training period\n+ controls",
            "Frozen\ntesting period",
            "LSEG\nearlier period",
            "LSEG\nlater period",
        ],
    )
    ax.set_xlim(-0.45, len(rows) - 0.55)
    ax.set_ylim(-0.118, 0.092)
    ax.set_yticks([-0.10, -0.05, 0.00, 0.05])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=5, labelsize=7.0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=False)
    ax.text(
        1.0,
        1.025,
        "FNSPID CHRONOLOGY",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=COLOUR["navy"],
        fontweight="bold",
    )
    ax.text(
        3.5,
        1.025,
        "LSEG PORTABILITY",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=COLOUR["rust"],
        fontweight="bold",
    )
    ax.set_ylabel("Negative-share coefficient (rank units)")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.22, top=0.94)
    finish(fig, output)


def lseg_value_figure(output: Path) -> None:
    """Matched-risk comparison plus a true cumulative value waterfall."""
    summary = read_csv("82_lseg33_firm_level_loss_control/evaluation_summary.csv")
    decomposition = read_csv("82_lseg33_firm_level_loss_control/economic_decomposition.csv")

    selected_arms = [
        ("equal_weight_long", "Fully invested", COLOUR["muted"], "o", True),
        ("selected_firm_news_overlay", "News rule", COLOUR["navy"], "s", True),
        (
            "development_symbol_matched_constant",
            "Matched constant",
            COLOUR["rust"],
            "^",
            False,
        ),
    ]
    metrics = [
        ("max_drawdown", "Maximum drawdown", True),
        ("annual_downside_deviation", "Downside deviation", False),
        ("expected_shortfall_5_loss", "Expected shortfall (5%)", False),
    ]

    dissertation_style()
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, 3.35),
        gridspec_kw={"width_ratios": [1.08, 1.42], "wspace": 0.28},
    )

    metric_styles = [
        (COLOUR["navy"], "o", "-", "Drawdown"),
        (COLOUR["rust"], "s", (0, (3, 2)), "Downside dev."),
        (COLOUR["secondary"], "^", (0, (1, 1)), "ES (5%)"),
    ]
    arm_x = np.arange(len(selected_arms))
    for (column, _metric_label, absolute), (colour, marker, line_style, short_label) in zip(
        metrics, metric_styles, strict=True
    ):
        values = []
        for arm, *_ in selected_arms:
            row = require_row(summary, "arm", arm)
            raw_value = float(row[column])
            values.append((abs(raw_value) if absolute else raw_value) * 100)
        ax1.plot(
            arm_x,
            values,
            color=colour,
            lw=1.25,
            ls=line_style,
            zorder=2,
        )
        ax1.scatter(
            arm_x,
            values,
            marker=marker,
            s=30,
            facecolor=colour if marker == "o" else COLOUR["paper"],
            edgecolor=colour,
            linewidth=1.0,
            zorder=3,
        )
        ax1.text(
            arm_x[-1] + 0.10,
            values[-1],
            f"{short_label}  {values[-1]:.2f}%",
            ha="left",
            va="center",
            fontsize=6.2,
            color=colour,
            fontweight="bold",
        )
    ax1.set_xticks(
        arm_x,
        ["Fully\ninvested", "News\nrule", "Matched\nconstant"],
    )
    ax1.set_xlim(-0.18, 3.12)
    ax1.set_ylim(0, 8.25)
    ax1.set_ylabel("Loss magnitude (%)")
    ax1.tick_params(axis="x", length=0, labelsize=6.4, pad=5)
    ax1.tick_params(axis="y", length=0)
    ax1.grid(axis="x", visible=False)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.set_title("(a) Loss profile", loc="left", pad=6)

    base_components = {
        row["component"]: float(row["gbp_per_million"])
        for _, row in decomposition.iterrows()
    }
    ending = base_components["compounded_ending_wealth_difference"]
    drivers = [
        ("Losses\navoided", base_components["losses_avoided_when_fixed_control_lost"] / 1000),
        ("Upside\nforgone", base_components["return_difference_when_fixed_control_gained"] / 1000),
        ("Trading\ncost", base_components["incremental_trading_cost"] / 1000),
    ]
    component_sum = sum(value for _, value in drivers)
    drivers.append(("Compounding\nresidual", ending / 1000 - component_sum))

    running = 0.0
    width = 0.62
    for idx, (_label, value) in enumerate(drivers):
        next_value = running + value
        bottom = min(running, next_value)
        height = abs(value)
        positive = value >= 0
        colour = COLOUR["navy"] if positive else COLOUR["rust_light"]
        edge = COLOUR["navy"] if positive else COLOUR["rust"]
        ax2.bar(
            idx,
            height,
            bottom=bottom,
            width=width,
            color=colour,
            edgecolor=edge,
            linewidth=1.0,
            hatch=None if positive else "///",
            zorder=3,
        )
        if idx < len(drivers) - 1:
            ax2.plot(
                [idx + width / 2, idx + 1 - width / 2],
                [next_value, next_value],
                color=COLOUR["muted"],
                lw=0.7,
                zorder=2,
            )
        ax2.text(
            idx,
            next_value + (1.5 if positive else -1.5),
            f"{value:+.1f}",
            ha="center",
            va="bottom" if positive else "top",
            fontsize=7.1,
            color=COLOUR["ink"],
            fontweight="bold",
        )
        running = next_value

    if not np.isclose(running * 1000, ending, atol=1.0):
        raise ValueError("waterfall components do not reconcile to ending wealth")
    final_idx = len(drivers)
    ax2.bar(
        final_idx,
        abs(ending / 1000),
        bottom=min(0, ending / 1000),
        width=width,
        color=COLOUR["summary"],
        edgecolor=COLOUR["ink"],
        linewidth=1.0,
        zorder=3,
    )
    ax2.text(
        final_idx,
        ending / 1000 - 1.5,
        f"{ending / 1000:+.1f}",
        ha="center",
        va="top",
        fontsize=7.4,
        color=COLOUR["ink"],
        fontweight="bold",
    )
    ax2.axhline(0, color=COLOUR["ink"], lw=0.8, zorder=1)
    ax2.set_xticks(
        np.arange(final_idx + 1),
        [label for label, _ in drivers] + ["Ending\nwealth"],
    )
    ax2.set_ylim(-38, 31)
    ax2.set_title("(b) Cumulative value bridge", loc="left", pad=6)
    ax2.text(
        1.0,
        0.985,
        "USD thousands per $1m",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=COLOUR["secondary"],
    )
    ax2.grid(axis="x", visible=False)
    ax2.tick_params(axis="x", length=0, labelsize=6.1, pad=4)
    ax2.set_axisbelow(True)

    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.25, top=0.88)
    finish(fig, output)


def break_even_figure(output: Path) -> None:
    """Ranked log-scale bullet chart with an explicit best-to-cost shortfall."""
    frame = read_csv("03_aggregation/development_ic_primary_ranked.csv").copy()
    frame["label"] = frame["aggregator"].map(display_label)
    frame = frame.sort_values("breakeven_bps_per_side", ascending=False)
    charged = 10.0
    best = float(frame["breakeven_bps_per_side"].max())

    dissertation_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.05))
    y = np.arange(len(frame))[::-1]

    ax.axvspan(best, charged, color=COLOUR["rust_light"], alpha=0.48, lw=0, zorder=0)

    for pos, (_, row) in zip(y, frame.iterrows(), strict=True):
        focal = row["aggregator"] == "negative_share"
        colour = COLOUR["navy"] if focal else COLOUR["muted"]
        value = float(row["breakeven_bps_per_side"])
        ax.hlines(
            pos,
            0.1,
            charged,
            color=COLOUR["grid"],
            lw=4.8,
            zorder=1,
        )
        ax.hlines(
            pos,
            0.1,
            value,
            color=colour,
            lw=4.8 if focal else 2.7,
            alpha=1.0 if focal else 0.86,
            zorder=2,
        )
        ax.scatter(
            value,
            pos,
            s=34 if focal else 24,
            color=colour if focal else COLOUR["paper"],
            edgecolor=colour,
            linewidth=1.0,
            zorder=3,
        )
        ax.text(
            value * 1.12,
            pos,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.2,
            color=COLOUR["ink"] if focal else COLOUR["secondary"],
            fontweight="bold" if focal else "normal",
        )

    ax.axvline(charged, color=COLOUR["rust"], lw=1.0, ls=(0, (4, 2)), zorder=4)
    fig.text(
        0.29,
        0.965,
        f"BEST  {best:.3f} bps     ASSUMED  {charged:.1f} bps     SHORTFALL  {charged / best:.0f}×",
        ha="left",
        va="top",
        fontsize=7.1,
        color=COLOUR["ink"],
        fontweight="bold",
    )
    ax.text(
        charged,
        len(frame) - 0.55,
        "10 bps cost",
        ha="right",
        va="bottom",
        fontsize=6.7,
        color=COLOUR["rust"],
        fontweight="bold",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.09, 13)
    ax.set_ylim(-0.55, len(frame) - 0.45)
    ax.set_yticks(y, frame["label"])
    bare_axes(ax)
    ax.set_xlabel("Break-even transaction cost per side (bps, log scale)")
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.18, top=0.89)
    finish(fig, output)


def write_aggregation_table(output: Path) -> None:
    """Exact nine-rule family results behind ``fig_aggregation_ic``."""
    frame = read_csv("03_aggregation/development_ic_primary_ranked.csv").copy()
    if len(frame) != 9:
        raise ValueError("aggregation family should contain nine rules")
    frame["label"] = frame["aggregator"].map(display_label)
    frame = frame.sort_values("p")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption[Nine firm-day aggregation rules]{Nine aggregation rules in the FNSPID training period."
        r" $\overline{IC}$ is the mean daily Spearman rank correlation with the assigned-session"
        r" open-to-next-open abnormal return, and $t$ uses Newey--West HAC(5). BH applies"
        r" Benjamini--Hochberg correction across the nine rules. Break-even is the per-side"
        r" cost that makes the corresponding rank portfolio's mean net return zero. The"
        r" rules use 512,145--512,149 firm-days across 2,264 sessions.}",
        r"\label{tab:aggregation-family}",
        r"\small",
        r"\begin{tabular}{@{}l d{-1.5} d{-1.2} d{1.4} c d{1.3}@{}}",
        r"\toprule",
        r"Aggregation rule & {$\overline{IC}$} & {HAC $t$} & {$p$} & {BH}"
        r" & {Break-even (bps)} \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        survivor = r"\checkmark" if bool(row["bh_reject_q05"]) else r"\textendash"
        lines.append(
            f"{row['label']} & {row['ic']:.5f} & {row['t']:.2f}"
            f" & {row['p']:.4f} & {survivor}"
            f" & {row['breakeven_bps_per_side']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_robustness_table(output: Path) -> None:
    """Story-count strata and inference-choice sensitivity for the main estimate."""
    strata = read_csv(
        "74_conditional_and_pressure_robustness_diagnostics/story_count_strata.csv"
    )
    hac = read_csv(
        "74_conditional_and_pressure_robustness_diagnostics/hac_lag_sensitivity.csv"
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption[Baseline sensitivity to lags and story count]{How the baseline negative-share"
        r" coefficient changes with same-day story count (upper panel) and Newey--West lag"
        r" length (lower panel). All rows use average-tie percentile ranks shifted by one"
        r" half within session and the assigned-session open-to-next-open abnormal return."
        r" Row counts are shown before missing values are dropped; four full-panel rows"
        r" lack the return.}",
        r"\label{tab:robustness-diagnostics}",
        r"\small",
        r"\begin{tabular}{@{}l d{-1.5} c d{1.4} d{6.0}@{}}",
        r"\toprule",
        r"Variation & {$\hat\beta_{neg}$} & {95\% CI} & {$p$} & {Panel rows} \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Same-day story count}} \\",
    ]
    stratum_labels = {
        "all firm-days (frozen)": "All firm-days (baseline)",
        "n >= 2": r"At least two stories ($n_{it}\geq2$)",
        "n >= 3": r"At least three stories ($n_{it}\geq3$)",
    }
    for _, row in strata.iterrows():
        label = stratum_labels.get(row["stratum"])
        if label is None:
            raise ValueError(f"unmapped story-count stratum {row['stratum']!r}")
        lines.append(
            f"\\quad {label} & {row['estimate']:.5f}"
            f" & $[{row['ci_low']:.5f},\\,{row['ci_high']:.5f}]$"
            f" & {row['p_two_sided']:.4f} & {int(row['firm_days'])} \\\\"
        )
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{5}{@{}l}{\emph{Newey--West lag length}} \\")
    firm_days = int(strata.iloc[0]["firm_days"])
    for _, row in hac.iterrows():
        baseline = " (baseline)" if int(row["hac_lags"]) == 5 else ""
        lines.append(
            f"\\quad {int(row['hac_lags'])} lags{baseline} & {row['estimate']:.5f}"
            f" & $[{row['ci_low']:.5f},\\,{row['ci_high']:.5f}]$"
            f" & {row['p_two_sided']:.4f} & {firm_days} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_sensitivity_table(output: Path) -> None:
    """Post-hoc close-to-close, FF3 and soft-mass coefficients."""
    frame = read_csv(
        "85_fnspid_development_outcome_and_label_sensitivities/sensitivity_coefficients.csv"
    )
    labels = {
        "headline_open_hard_share": "Assigned-window hard share (baseline)",
        "close_to_close_hard_share": "Close-to-close hard share",
        "ff3_residual_hard_share": "FF3 residual hard share",
        "open_soft_negative_mass": "Assigned-window soft negative mass",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption[Measurement sensitivities]{Measurement checks added after the main FNSPID"
        r" training-period result. The baseline row repeats the earlier next-open hard-share"
        r" estimate and is not a fourth test. BH is Benjamini--Hochberg at"
        r" $q=0.05$ across the three new coefficients. Close-to-close starts at"
        r" the assigned session close; FF3 residuals use trailing 252-session"
        r" betas that exclude the outcome session.}",
        r"\label{tab:outcome-label-sensitivities}",
        r"\scriptsize",
        r"\begin{tabular}{@{}l d{-1.5} c d{1.4} c c@{}}",
        r"\toprule",
        r"Test & {Estimate} & {95\% CI} & {$p$} & {$q$} & {BH} \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        label = labels.get(row["test"])
        if label is None:
            raise ValueError(f"unmapped sensitivity test {row['test']!r}")
        q_val = row["q_bh_three_test"]
        q_tex = "--" if pd.isna(q_val) else f"{float(q_val):.3f}"
        survivor = r"\checkmark" if bool(row["bh_reject_q05"]) else "--"
        lines.append(
            f"{label} & {row['estimate']:.5f}"
            f" & $[{row['ci_low']:.5f},\\,{row['ci_high']:.5f}]$"
            f" & {row['p_two_sided']:.4f} & {q_tex} & {survivor} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_outcome_leg_table(output: Path) -> None:
    """Write the post-hoc timing-decomposition table."""
    coefficients = read_csv(
        "86_fnspid_outcome_leg_decomposition/coefficient_summary.csv"
    )
    contrasts = read_csv(
        "86_fnspid_outcome_leg_decomposition/era_contrasts.csv"
    )
    prior_open = read_csv(
        "87_fnspid_external_review_robustness_pack/prior_open_timing_probe.csv"
    ).iloc[0]
    labels = {
        "assigned_open_to_next_open": "Assigned open to next open (reference)",
        "assigned_session_intraday": "Assigned open to close",
        "post_close_overnight": "Assigned close to next open",
        "pre_open_overnight": "Previous close to assigned open",
    }
    order = [
        "assigned_open_to_next_open",
        "assigned_session_intraday",
        "post_close_overnight",
        "pre_open_overnight",
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption[Outcome-leg decomposition]{Post-hoc FNSPID return-window decomposition."
        r" The assigned-open-to-next-open row repeats the baseline and is excluded from the"
        r" new multiplicity adjustment. Within each period, BH correction covers the three"
        r" decomposed return legs. The three between-period differences form a separate"
        r" family. The previous-open analysis is a separate post-hoc timing diagnostic."
        r" All intervals use HAC(5).}",
        r"\label{tab:outcome-leg-decomposition}",
        r"\scriptsize",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.5cm} d{-1.5} c d{1.4} c@{}}",
        r"\toprule",
        r"Outcome or contrast & {Estimate} & {95\% CI} & {$p$} & {BH $q$} \\",
        r"\midrule",
    ]
    for regime, heading in (
        ("development", "Training period"),
        ("evaluation", "Testing period (additional timing diagnostic)"),
    ):
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{heading}}}}} \\")
        frame = coefficients.loc[coefficients["regime"] == regime]
        for outcome in order:
            row = require_row(frame, "outcome_leg", outcome)
            q_value = row["q_bh_three_leg"]
            q_tex = "--" if pd.isna(q_value) else f"{float(q_value):.4f}"
            lines.append(
                f"\\quad {labels[outcome]} & {row['estimate']:.5f}"
                f" & $[{row['ci_low']:.5f},\\,{row['ci_high']:.5f}]$"
                f" & {row['p_two_sided']:.4f} & {q_tex} \\\\"
            )
        if regime == "development":
            lines.append(
                f"\\quad Previous open to assigned open (separate) & {prior_open['estimate']:.5f}"
                f" & $[{prior_open['ci_low']:.5f},\\,{prior_open['ci_high']:.5f}]$"
                f" & \\multicolumn{{1}}{{c}}{{$7.9\\times10^{{-9}}$}} & -- \\\\"
            )
        lines.append(r"\addlinespace")

    lines.append(r"\multicolumn{5}{@{}l}{\emph{Testing minus training}} \\")
    for outcome in order:
        row = require_row(contrasts, "outcome_leg", outcome)
        q_value = row["q_bh_three_leg"]
        q_tex = "--" if pd.isna(q_value) else f"{float(q_value):.4f}"
        lines.append(
            f"\\quad {labels[outcome]} & {row['estimate']:.5f}"
            f" & $[{row['ci_low']:.5f},\\,{row['ci_high']:.5f}]$"
            f" & {row['p_two_sided']:.4f} & {q_tex} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_hypothesis_table(output: Path) -> None:
    """Compact verdict map aligned to the hypotheses in Chapter 1."""
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption[Hypothesis and transfer verdicts]{Final hypothesis and cross-source verdicts."
        r" ``Consistent'' means the selected training-period estimate has the direction stated"
        r" in H1a, but it is exploratory rather than a confirmation.}",
        r"\label{tab:hypothesis-verdicts}",
        r"\small",
        r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{3.3cm} >{\raggedright\arraybackslash}p{8.0cm}@{}}",
        r"\toprule",
        r"Claim & Verdict & Reason \\",
        r"\midrule",
        r"H1a & Consistent, not confirmed & Negative selected training-period coefficient; timing is intraday and assignment remains ambiguous. \\",
        r"H1b & Not supported & Neither testing-period conditional coefficient is negative, and no testing-period rule survives correction. \\",
        r"H2 & Direction only & High-minus-low spread is negative, descriptive and imprecise. \\",
        r"H3 & Fails & Turnover makes the daily implementation negative after the declared cost. \\",
        r"H4 & Fails & No LSEG risk rule meets all matched-control economic criteria. \\",
        r"Cross-source result & Unresolved & Both LSEG FinBERT intervals cross zero and their MDEs are many times the FNSPID coefficient. \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_prompt_selection_table(output: Path) -> None:
    """Per-prompt 2024 selection results for Notebooks 83 and 84."""
    variants = read_csv("83_lseg_prompt_economic_value/selection_2024.csv").copy()
    structured = read_csv(
        "84_lseg_structured_reaction_score/selection_2024.csv"
    ).copy()
    if len(variants) != 5:
        raise ValueError("prompt variant family should contain five prompts")
    if len(structured) != 1:
        raise ValueError("structured reaction score should contain one prompt")

    structured["horizon"] = 1
    structured["eligible"] = structured["selection_pass"]
    frame = pd.concat([variants, structured], ignore_index=True)
    if bool(frame["eligible"].any()):
        raise ValueError("no prompt should meet the frozen 2024 selection criteria")
    frame["label"] = frame["prompt_id"].map(PROMPT_LABELS)
    if frame["label"].isna().any():
        raise ValueError("unmapped prompt identifier in the selection tables")

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption[Prompt training-period results]{Prompt results from the fixed 2024 LSEG training period. Means"
        r" are basis points per formation session; net figures charge 10 basis"
        r" points per side. Break-even is the per-side cost that would set the"
        r" mean net return to zero, so a negative value means no non-negative cost level makes"
        r" the prompt profitable. The Quarters column counts the 2024 quarters with a"
        r" positive net mean, out of four. The test required a positive"
        r" net mean, a positive net Sharpe, a break-even of at least 10 basis"
        r" points, three positive quarters and three quarters beating the"
        r" same-date Gemma comparator. No prompt met all five pre-specified"
        r" conditions, so the final analysis formed no 2025 prompt portfolio.}",
        r"\label{tab:prompt-selection}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l c d{-1.3} d{-2.3} d{-1.3} d{-2.3} c c@{}}",
        r"\toprule",
        r"Prompt & {Horizon} & {Gross} & {Net} & {Net Sharpe}"
        r" & {Break-even} & {Quarters} & {Passes} \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{row['label']} & {int(row['horizon'])}"
            f" & {row['mean_gross_bps_session']:.3f}"
            f" & {row['mean_net_bps_session']:.3f}"
            f" & {row['net_sharpe']:.3f}"
            f" & {row['breakeven_bps_per_side']:.3f}"
            f" & {int(row['positive_quarters'])}"
            r" & \textendash \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_headline_table(output: Path) -> None:
    aggregation = read_csv("03_aggregation/development_ic_primary_ranked.csv")
    controls = read_csv(
        "79_fnspid_development_reversal_confound/conditional_price_control_summary.csv"
    )
    double_sort = read_csv(
        "76_fnspid_development_negative_share_double_sort/pooled_spread_summary.csv"
    )
    transfer = read_csv("71_conditional_negative_share_transfer/conditional_coefficients.csv")
    evaluation = read_csv(
        "75_fnspid_evaluation_beyond_mean_replication/"
        "evaluation_conditional_coefficients.csv"
    )
    temporal_contrast = read_csv(
        "75_fnspid_evaluation_beyond_mean_replication/"
        "evaluation_minus_development_stability_contrast.csv"
    ).iloc[0]

    negative_share = require_row(aggregation, "aggregator", "negative_share")
    baseline = require_row(controls, "specification", "published_baseline_full_sample")
    full = require_row(controls, "specification", "full_price_path_controls")
    spread = require_row(double_sort, "sample", "all_firm_days")
    lseg = transfer.loc[
        (transfer["source"] == "LSEG") & (transfer["scorer"] == "finbert")
    ]
    backward = require_row(lseg, "block", "backward")
    recent = require_row(lseg, "block", "recent")
    evaluation_all = require_row(evaluation, "member", "all_firm_days")
    evaluation_multi = require_row(evaluation, "member", "multi_story_n_ge_2")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption[Headline negative-share estimates]{Main negative-story-share estimates. FNSPID uses assigned-session open-to-next-open abnormal returns; LSEG uses raw open-to-open returns under its recorded availability rule. The two FNSPID testing-period conditional rows are corrected together. The difference between periods is a separate check fixed in advance. Where shown, intervals are 95\% HAC intervals.}",
        r"\label{tab:headline-estimates}",
        r"\small",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.9cm}"
        r" d{-1.5} c l d{4.0}@{}}",
        r"\toprule",
        r"Test & {Estimate} & {95\% interval} & {$p$ or $q$} & {Sessions} \\",
        r"\midrule",
        f"Aggregation-family IC & {negative_share['ic']:.5f} & {{\\textendash}} & $p={negative_share['p']:.4f}$ & {int(negative_share['n_clusters'])} \\\\",
        f"Conditional baseline & {baseline['estimate']:.5f} & $[{baseline['ci_low']:.5f},\\,{baseline['ci_high']:.5f}]$ & $p={baseline['p_two_sided']:.4f}$ & {int(baseline['n_clusters'])} \\\\",
        f"Full price-path controls & {full['estimate']:.5f} & $[{full['ci_low']:.5f},\\,{full['ci_high']:.5f}]$ & $p={full['p_two_sided']:.4f}$ & {int(full['n_clusters'])} \\\\",
        f"Testing period: all firm-days & {evaluation_all['estimate']:.5f} & $[{evaluation_all['ci_low']:.5f},\\,{evaluation_all['ci_high']:.5f}]$ & $q={evaluation_all['q_bh_two_test']:.3f}$ & {int(evaluation_all['sessions_used'])} \\\\",
        f"Testing period: at least two stories & {evaluation_multi['estimate']:.5f} & $[{evaluation_multi['ci_low']:.5f},\\,{evaluation_multi['ci_high']:.5f}]$ & $q={evaluation_multi['q_bh_two_test']:.3f}$ & {int(evaluation_multi['sessions_used'])} \\\\",
        f"Testing minus training & {temporal_contrast['estimate']:.5f} & $[{temporal_contrast['ci_low']:.5f},\\,{temporal_contrast['ci_high']:.5f}]$ & $p={temporal_contrast['p_two_sided']:.4f}$ & {int(temporal_contrast['n_clusters'])} \\\\",
        f"Model-free high--low (bps) & {spread['mean_bps']:.3f} & $[{spread['ci_low_bps']:.3f},\\,{spread['ci_high_bps']:.3f}]$ & $p={spread['p_two_sided_descriptive']:.3f}$ & {int(spread['n_sessions'])} \\\\",
        f"LSEG earlier-period coefficient & {backward['estimate']:.5f} & $[{backward['ci_low']:.5f},\\,{backward['ci_high']:.5f}]$ & $q={backward['q_bh_lseg_finbert_two_block']:.3f}$ & {int(backward['n_clusters'])} \\\\",
        f"LSEG later-period coefficient & {recent['estimate']:.5f} & $[{recent['ci_low']:.5f},\\,{recent['ci_high']:.5f}]$ & $q={recent['q_bh_lseg_finbert_two_block']:.3f}$ & {int(recent['n_clusters'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(output: Path, metrics: dict[str, float]) -> None:
    text = f"""# Derived-artifact validation report

Overall assessment: **ready to use with the stated caveats**.

- Notebook 75 was opened exactly once in the source repository and promoted
  without re-execution. Its all-firm-day testing-period coefficient is
  **{metrics['evaluation_conditional_estimate']:+.5f}** and the predeclared
  testing-minus-training contrast is
  **{metrics['evaluation_minus_development_contrast']:+.5f}**.
- The prospective approximation gave
  **{metrics['evaluation_stable_effect_power'] * 100:.1f}%** power to recover
  the training-period baseline effect at the corrected two-test threshold. Realised precision
  implies **{metrics['evaluation_realised_effect_power'] * 100:.1f}%** ex-post
  power at the same threshold. The failed testing-period family is therefore not proof
  of a zero effect; the separate contrast is consistent with instability in the
  measured baseline pipeline association without identifying its cause.

## Inputs and grain

- All inputs are committed aggregate CSV files under `experiments/results/`.
- FNSPID and LSEG rows are kept separate. No row-level story text is read.
- Coefficients share shifted average-tie percentile-rank units, but FNSPID uses abnormal returns and
  LSEG uses raw returns. The cross-source figure states this difference.
- The FNSPID same-session news risk rule uses current-session news; only its
  normalisation baseline is lagged. Its circular shifts are schedule-alignment
  diagnostics, not evidence that the rule was observable before the return.
- The LSEG economic comparison uses one 167-session block and the
  symbol-matched constant selected in the training period as comparator.

## Calculation spot checks

- Full-control translation over the observed pooled tied-rank spread
  ({metrics['observed_negative_share_rank_spread']:.5f}):
  **{metrics['rank_effect_p10_to_p90_percentile_points']:.4f} percentile points**.
- Training-period timing coefficients: intraday
  **{metrics['development_intraday_coefficient']:+.5f}** and post-close
  **{metrics['development_post_close_coefficient']:+.5f}**. The separate
  previous-open probe is **{metrics['development_prior_open_coefficient']:+.5f}**.
- FNSPID 2020--2023 maximum drawdown: control
  **{metrics['fnspid_control_max_drawdown'] * 100:.2f}%**, overlay
  **{metrics['fnspid_overlay_max_drawdown'] * 100:.2f}%**.
- Charged cost / best break-even: **{metrics['charged_cost_multiple_of_best_break_even']:.4f}x**.
- LSEG MDE multiples: **{metrics['lseg_backward_mde_multiple']:.4f}x** in the earlier period and **{metrics['lseg_recent_mde_multiple']:.4f}x** in the later period.
- Under square-root precision scaling, those ratios imply approximately
  **{metrics['lseg_backward_information_multiple_for_fnspid_sized_mde']:.1f}x** and
  **{metrics['lseg_recent_information_multiple_for_fnspid_sized_mde']:.1f}x** as much
  time-series information for an FNSPID-sized MDE. This is an approximation,
  not a forecast of the exact sessions required.
- LSEG rule drawdown improvement versus fully invested: **{metrics['lseg_firm_overlay_drawdown_improvement_vs_fully_invested_pp']:.4f} percentage points**.
- LSEG rule drawdown disadvantage versus matched constant: **{metrics['lseg_firm_overlay_drawdown_disadvantage_vs_matched_pp']:.4f} percentage points**.
- LSEG rule ending-wealth difference: **USD {metrics['lseg_firm_overlay_ending_wealth_difference_usd_per_million']:.2f} per USD 1m**.
- Conditional coefficient by same-day story count:
  **{metrics['conditional_estimate_min_two_stories']:.5f}** at n >= 2 and
  **{metrics['conditional_estimate_min_three_stories']:.5f}** at n >= 3, which retains
  **{metrics['conditional_retained_share_min_three_stories'] * 100:.1f}%**
  of the full-sample estimate.
- The two largest LSEG earlier-period risk-off episodes supply
  **{metrics['lseg_backward_top_two_episode_downside_share'] * 100:.1f}%**
  of the block's squared-downside reduction.
- The first three economic components reconcile to the saved arithmetic
  difference within USD 1 per USD 1m (difference:
  **USD {metrics['lseg_firm_overlay_arithmetic_reconciliation_usd_per_million']:.2f}**).
  The figure uses an exact residual from their sum to compounded ending wealth.

## Required caveats

- The tied-rank translation is a linear interpretation of a rank coefficient,
  not a basis-point return forecast. The superseded hypothetical 0.8 move is not
  attainable on the observed regressor.
- MDE ratios describe power; they do not prove the LSEG coefficient is zero.
- The firm-level LSEG comparison is retrospective and selected on training-period
  data. It does not meet all registered economic-value criteria.
- The plots and tables summarise existing evidence and add arithmetic translations. They
  are not new model searches or confirmatory tests.
- Notebook 85 close-to-close, FF3 residual and soft-mass coefficients are a
  three-member post-hoc family frozen after the headline result. They are not
  confirmation of H1.
- Notebooks 86 and 87 are post-review diagnostics. The intraday and prior-open
  results require withdrawal of predictive-window specificity because story
  arrival cannot be reconstructed at row level.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    require_locked_figure_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    derived, metrics = derive_metrics()
    derived.to_csv(output / "derived_metrics.csv", index=False)
    (output / "headline_numbers.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    aggregation_family_figure(output / "fig_aggregation_family.png")
    estimate_stability_figure(output / "fig_estimate_stability.png")
    double_sort_figure(output / "fig_double_sort.png")
    break_even_figure(output / "fig_break_even_cost_gap.png")
    risk_regime_figure(output / "fig_risk_regime_stability.png")
    timing_placebo_figure(output / "fig_timing_placebo.png")
    coefficient_figure(output / "fig_cross_source_coefficient_power.png")
    lseg_value_figure(output / "fig_lseg_matched_control_value.png")
    prompt_gate_figure(output / "fig_prompt_gate.png")
    coefficient_drift_figure(output / "fig_coefficient_drift.png")
    write_headline_table(output / "tab_headline_estimates.tex")
    write_aggregation_table(output / "tab_aggregation_family.tex")
    write_robustness_table(output / "tab_robustness_diagnostics.tex")
    write_sensitivity_table(output / "tab_outcome_label_sensitivities.tex")
    write_outcome_leg_table(output / "tab_outcome_leg_decomposition.tex")
    write_hypothesis_table(output / "tab_hypothesis_verdicts.tex")
    write_prompt_selection_table(output / "tab_prompt_selection.tex")
    write_validation_report(output / "validation_report.md", metrics)
    print(f"Generated dissertation artifacts in {output}")


if __name__ == "__main__":
    main()
