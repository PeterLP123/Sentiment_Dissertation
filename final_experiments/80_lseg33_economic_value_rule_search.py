# %% [markdown]
# # 80 — LSEG-33 economic-value rule search
#
# This exploratory search is intentionally separated from the frozen unchanged-rule
# transfers in Notebooks 72 and 73. It selects one causal negative-news risk rule
# using only the earlier 456-session LSEG-33 block, then applies that rule unchanged
# to the later 167-session block. All 6,672 candidates are retained.

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from itertools import product
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "sentiment_dissertation_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name == "final_experiments":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_experiments.lib.plots import CATEGORICAL, apply_house_style  # noqa: E402

apply_house_style()
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")

SEED = 20260812
COST_BPS = 2.0
BLOCK_LENGTH = 20
BOOTSTRAP_REPLICATIONS = 9_999
RISK_AVERSION = 5.0
BOUNDARY = pd.Timestamp("2025-10-27")
N_FIRMS = 33

SPEC_PATH = (
    ROOT
    / "final_experiments/frozen_specs/lseg33_economic_value_rule_search_v1_20260812.json"
)
OUTPUT = ROOT / "final_experiments/outputs/80_lseg33_economic_value_rule_search"
FIGURE_OUTPUT = ROOT / "dissertation/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def price_path(
    forward_return: np.ndarray,
    exposure: np.ndarray,
    *,
    cost_bps_per_side: float,
) -> dict[str, np.ndarray]:
    returns = np.asarray(forward_return, dtype=float)
    weights = np.asarray(exposure, dtype=float)
    if len(returns) != len(weights):
        raise ValueError("return and exposure lengths differ")
    if np.any(~np.isfinite(returns)) or np.any(~np.isfinite(weights)):
        raise ValueError("return and exposure arrays must be finite")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("exposure must remain in [0, 1]")
    previous = np.r_[0.0, weights[:-1]]
    turnover = 0.5 * np.abs(weights - previous)
    turnover[-1] += 0.5 * abs(weights[-1])
    gross = weights * returns
    cost = 2.0 * turnover * cost_bps_per_side / 10_000.0
    return {
        "gross_return": gross,
        "turnover": turnover,
        "cost": cost,
        "net_return": gross - cost,
        "exposure": weights,
    }


def downside_utility(values: np.ndarray, gamma: float = RISK_AVERSION) -> np.ndarray:
    returns = np.asarray(values, dtype=float)
    return returns - gamma * np.minimum(returns, 0.0) ** 2


def expected_shortfall_loss(values: np.ndarray, alpha: float = 0.05) -> float:
    returns = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(math.ceil(alpha * len(returns))))
    return float(-returns[:count].mean())


def max_drawdown(values: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.asarray(values, dtype=float))
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def path_summary(path: dict[str, np.ndarray]) -> dict[str, float | int]:
    net = path["net_return"]
    mean = float(net.mean())
    volatility = float(net.std(ddof=1))
    downside_deviation = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)))
    equity = float(np.prod(1.0 + net))
    return {
        "n_sessions": len(net),
        "mean_net_bps_session": mean * 10_000,
        "total_return_net": equity - 1.0,
        "sharpe_net": math.sqrt(252) * mean / volatility if volatility > 0 else np.nan,
        "sortino_net": (
            math.sqrt(252) * mean / downside_deviation
            if downside_deviation > 0
            else np.nan
        ),
        "annual_volatility": volatility * math.sqrt(252),
        "annual_downside_deviation": downside_deviation * math.sqrt(252),
        "expected_shortfall_5_loss": expected_shortfall_loss(net),
        "max_drawdown": max_drawdown(net),
        "annual_downside_utility": float(downside_utility(net).mean() * 252),
        "mean_exposure": float(path["exposure"].mean()),
        "annual_turnover": float(path["turnover"].mean() * 252),
    }


def modifier_from_z(
    z_values: np.ndarray,
    *,
    mode: str,
    entry_z: float,
    risk_multiplier: float,
    exit_gap: float | None = None,
    full_cut_distance: float | None = None,
) -> np.ndarray:
    z = np.asarray(z_values, dtype=float)
    modifier = np.ones(len(z), dtype=float)
    finite = np.isfinite(z)
    if mode == "one_day":
        modifier[finite & (z >= entry_z)] = risk_multiplier
    elif mode == "hysteresis":
        if exit_gap is None:
            raise ValueError("hysteresis requires exit_gap")
        state = False
        exit_z = entry_z - exit_gap
        for index, value in enumerate(z):
            if np.isfinite(value):
                if not state and value >= entry_z:
                    state = True
                elif state and value <= exit_z:
                    state = False
            modifier[index] = risk_multiplier if state else 1.0
    elif mode == "smooth":
        if full_cut_distance is None:
            raise ValueError("smooth requires full_cut_distance")
        intensity = np.clip((z - entry_z) / full_cut_distance, 0.0, 1.0)
        modifier[finite] = 1.0 - (1.0 - risk_multiplier) * intensity[finite]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return modifier


def bootstrap_metric(
    challenger: np.ndarray,
    comparator: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    seed: int,
) -> dict[str, float | int]:
    left = np.asarray(challenger, dtype=float)
    right = np.asarray(comparator, dtype=float)
    if len(left) != len(right):
        raise ValueError("paired bootstrap arrays differ in length")
    n = len(left)
    point = float(metric(left, right))
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / BLOCK_LENGTH))
    samples = np.empty(BOOTSTRAP_REPLICATIONS, dtype=float)
    for replication in range(BOOTSTRAP_REPLICATIONS):
        starts = rng.integers(0, n, size=blocks)
        indices = np.concatenate(
            [np.arange(start, start + BLOCK_LENGTH) % n for start in starts]
        )[:n]
        samples[replication] = metric(left[indices], right[indices])
    centred = samples - point
    return {
        "n": n,
        "estimate": point,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_two_sided": float(
            (1 + np.sum(np.abs(centred) >= abs(point)))
            / (BOOTSTRAP_REPLICATIONS + 1)
        ),
        "block_length": BLOCK_LENGTH,
        "replications": BOOTSTRAP_REPLICATIONS,
        "seed": seed,
    }


print(f"repo:   {ROOT}")
print(f"spec:   {SPEC_PATH.relative_to(ROOT)}")
print(f"output: {OUTPUT.relative_to(ROOT)}")

# %% [markdown]
# ## Input identities and chronological boundary

# %%
spec = json.loads(SPEC_PATH.read_text())
if spec["status"] != "frozen_before_notebook_80_search_result":
    raise ValueError("unexpected search-spec status")

input_paths = {
    key: ROOT / value["path"]
    for key, value in spec["inputs"].items()
    if isinstance(value, dict) and "path" in value
}
input_audit_rows = []
for key, path in input_paths.items():
    actual = sha256(path)
    expected = spec["inputs"][key]["sha256"]
    if actual != expected:
        raise ValueError(f"input hash mismatch for {key}: {path}")
    input_audit_rows.append(
        {
            "input": key,
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    )
input_audit = pd.DataFrame(input_audit_rows)
display(input_audit)

backward = pd.read_parquet(input_paths["backward_finbert_firm_open"])
recent = pd.read_parquet(input_paths["recent_finbert_firm_open"])
development_har = pd.read_parquet(input_paths["development_har_daily"])
evaluation_har = pd.read_parquet(input_paths["evaluation_har_daily"])
for frame in (backward, recent, development_har, evaluation_har):
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()

backward = backward.loc[backward["session_date"].lt(BOUNDARY)].copy()
recent = recent.loc[recent["session_date"].ge(BOUNDARY)].copy()
development_har = development_har.sort_values("session_date", kind="mergesort").reset_index(drop=True)
evaluation_har = evaluation_har.sort_values("session_date", kind="mergesort").reset_index(drop=True)
if len(development_har) != 456 or len(evaluation_har) != 167:
    raise ValueError("unexpected chronological sample size")
if development_har["session_date"].max() >= BOUNDARY:
    raise ValueError("development HAR crosses the boundary")
if evaluation_har["session_date"].min() < BOUNDARY:
    raise ValueError("evaluation HAR crosses the boundary")

all_sessions = pd.DatetimeIndex(
    pd.concat(
        [development_har["session_date"], evaluation_har["session_date"]],
        ignore_index=True,
    )
)
if not all_sessions.is_monotonic_increasing or all_sessions.has_duplicates:
    raise ValueError("combined session calendar must be ordered and unique")

# %% [markdown]
# ## Eight daily pressure measures and causal normalisation

# %%
firm_open = pd.concat([backward, recent], ignore_index=True)
firm_open = firm_open.loc[firm_open["session_date"].isin(all_sessions)].copy()
if firm_open.duplicated(["session_date", "symbol"]).any():
    raise ValueError("duplicate firm-session rows")

counts = firm_open.groupby("session_date")["symbol"].nunique().reindex(all_sessions, fill_value=0)
complete_sessions = counts.index[counts.eq(N_FIRMS)]
complete = firm_open.loc[firm_open["session_date"].isin(complete_sessions)].copy()
complete["negative_headline_count"] = (
    complete["negative_share"].astype(float)
    * complete["unique_headline_count"].astype(float)
)

daily = complete.groupby("session_date", sort=True).agg(
    negative_headlines=("negative_headline_count", "sum"),
    headline_count=("unique_headline_count", "sum"),
    equal_firm_negative_share=("negative_share", "mean"),
    negative_equal_firm_mean_continuous=("mean_continuous", lambda values: -float(values.mean())),
)
daily["headline_weighted_negative_share"] = (
    daily["negative_headlines"] / daily["headline_count"]
)
daily["upper_quartile_firm_negative_share"] = complete.groupby(
    "session_date", sort=True
)["negative_share"].apply(lambda values: float(values.nlargest(9).mean()))
for threshold, suffix in ((0.20, "0.20"), (0.25, "0.25"), (0.33, "0.33"), (0.50, "0.50")):
    daily[f"negative_firm_breadth_ge_{suffix}"] = complete.groupby(
        "session_date", sort=True
    )["negative_share"].apply(lambda values, cut=threshold: float(values.ge(cut).mean()))

measure_names = spec["pressure_family"]["daily_measures"]
pressure_measures = daily.reindex(all_sessions)[measure_names]
if pressure_measures.notna().all(axis=1).sum() != len(complete_sessions):
    raise ValueError("pressure completeness does not match the 33-firm census")

z_series: dict[tuple[str, str, int], pd.Series] = {}
for measure, normaliser, lookback in product(
    measure_names,
    ("rolling", "ewm"),
    (63, 126, 252),
):
    values = pressure_measures[measure].astype(float)
    prior = values.shift(1)
    min_periods = int(math.ceil(lookback / 2))
    if normaliser == "rolling":
        centre = prior.rolling(lookback, min_periods=min_periods).mean()
        scale = prior.rolling(lookback, min_periods=min_periods).std(ddof=1)
    else:
        centre = prior.ewm(
            span=lookback,
            min_periods=min_periods,
            adjust=False,
        ).mean()
        scale = prior.ewm(
            span=lookback,
            min_periods=min_periods,
            adjust=False,
        ).std(bias=False)
    z_series[(measure, normaliser, lookback)] = ((values - centre) / scale).where(
        scale.gt(0)
    )

pressure_audit = pd.DataFrame(
    [
        {
            "sessions": len(all_sessions),
            "complete_33_firm_sessions": len(complete_sessions),
            "incomplete_sessions": int(counts.ne(N_FIRMS).sum()),
            "development_sessions": len(development_har),
            "evaluation_sessions": len(evaluation_har),
            "pressure_measures": len(measure_names),
            "causal_z_series": len(z_series),
        }
    ]
)
display(pressure_audit)

# %% [markdown]
# ## Complete 6,672-candidate development search
#
# Candidate generation and ranking below receive development returns only. The
# evaluation return array is not created until after a single candidate is selected.

# %%
development_dates = development_har["session_date"]
development_mask = all_sessions < BOUNDARY
development_returns = development_har["forward_return"].to_numpy(dtype=float)
development_base_exposure = development_har["exposure"].to_numpy(dtype=float)
fold_indices = np.array_split(np.arange(len(development_har)), 3)
if any(len(indices) != 152 for indices in fold_indices):
    raise ValueError("development folds are not equal 152-session blocks")

mapping_specs: list[dict[str, float | str | None]] = []
for entry_z, risk_multiplier in product(
    (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    (0.0, 0.25, 0.5, 0.75),
):
    mapping_specs.append(
        {
            "mode": "one_day",
            "entry_z": entry_z,
            "risk_multiplier": risk_multiplier,
            "exit_gap": None,
            "full_cut_distance": None,
        }
    )
for entry_z, exit_gap, risk_multiplier in product(
    (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    (0.5, 1.0, 1.5),
    (0.0, 0.25, 0.5, 0.75),
):
    mapping_specs.append(
        {
            "mode": "hysteresis",
            "entry_z": entry_z,
            "risk_multiplier": risk_multiplier,
            "exit_gap": exit_gap,
            "full_cut_distance": None,
        }
    )
for entry_z, full_cut_distance, risk_multiplier in product(
    (0.0, 0.5, 1.0),
    (0.75, 1.25, 2.0),
    (0.25, 0.5, 0.75),
):
    mapping_specs.append(
        {
            "mode": "smooth",
            "entry_z": entry_z,
            "risk_multiplier": risk_multiplier,
            "exit_gap": None,
            "full_cut_distance": full_cut_distance,
        }
    )
if len(mapping_specs) != 139:
    raise RuntimeError("mapping grid does not contain 139 specifications")

candidate_rows = []
candidate_modifiers: dict[str, np.ndarray] = {}
candidate_number = 0
for (measure, normaliser, lookback), full_z in z_series.items():
    development_z = full_z.loc[development_mask].to_numpy(dtype=float)
    for mapping in mapping_specs:
        candidate_number += 1
        candidate_id = f"candidate_{candidate_number:04d}"
        modifier = modifier_from_z(development_z, **mapping)
        candidate_modifiers[candidate_id] = modifier
        fold_rows = []
        for fold_number, indices in enumerate(fold_indices, start=1):
            fold_returns = development_returns[indices]
            fold_base = development_base_exposure[indices]
            fold_modifier = modifier[indices]
            overlay = price_path(
                fold_returns,
                fold_base * fold_modifier,
                cost_bps_per_side=COST_BPS,
            )
            har = price_path(
                fold_returns,
                fold_base,
                cost_bps_per_side=COST_BPS,
            )
            constant = price_path(
                fold_returns,
                fold_base * float(fold_modifier.mean()),
                cost_bps_per_side=COST_BPS,
            )
            overlay_utility = float(downside_utility(overlay["net_return"]).mean())
            constant_utility = float(downside_utility(constant["net_return"]).mean())
            har_utility = float(downside_utility(har["net_return"]).mean())
            fold_rows.append(
                {
                    "fold": fold_number,
                    "utility_delta_vs_constant": (overlay_utility - constant_utility) * 252,
                    "utility_delta_vs_har": (overlay_utility - har_utility) * 252,
                    "net_delta_vs_har": float(
                        overlay["net_return"].mean() - har["net_return"].mean()
                    ),
                    "downside_reduction_vs_constant": float(
                        np.mean(np.minimum(constant["net_return"], 0.0) ** 2)
                        - np.mean(np.minimum(overlay["net_return"], 0.0) ** 2)
                    ),
                }
            )
        fold = pd.DataFrame(fold_rows).set_index("fold")
        full_path = price_path(
            development_returns,
            development_base_exposure * modifier,
            cost_bps_per_side=COST_BPS,
        )
        mean_modifier = float(modifier.mean())
        eligible = bool(
            0.50 <= mean_modifier <= 0.95
            and fold["net_delta_vs_har"].ge(-0.0001).all()
            and fold["downside_reduction_vs_constant"].gt(0).sum() >= 2
            and fold["utility_delta_vs_constant"].gt(0).sum() >= 2
        )
        row: dict[str, float | int | str | bool | None] = {
            "candidate_id": candidate_id,
            "measure": measure,
            "normaliser": normaliser,
            "lookback": lookback,
            **mapping,
            "mean_modifier": mean_modifier,
            "reduced_exposure_share": float(np.mean(modifier < 1.0 - 1e-12)),
            "annual_turnover": float(full_path["turnover"].mean() * 252),
            "eligible": eligible,
            "selection_score": float(fold["utility_delta_vs_constant"].min()),
            "mean_fold_utility_delta_vs_constant": float(
                fold["utility_delta_vs_constant"].mean()
            ),
            "minimum_fold_utility_delta_vs_har": float(
                fold["utility_delta_vs_har"].min()
            ),
        }
        for fold_number in (1, 2, 3):
            for column in (
                "utility_delta_vs_constant",
                "utility_delta_vs_har",
                "net_delta_vs_har",
                "downside_reduction_vs_constant",
            ):
                row[f"fold_{fold_number}_{column}"] = float(fold.loc[fold_number, column])
        candidate_rows.append(row)

candidate_search = pd.DataFrame(candidate_rows)
if len(candidate_search) != spec["mapping_family"]["candidate_count"]:
    raise RuntimeError("candidate count differs from frozen grid")
eligible_candidates = candidate_search.loc[candidate_search["eligible"]].copy()
if eligible_candidates.empty:
    raise RuntimeError("no candidate satisfies the frozen eligibility rules")
selected_row = (
    eligible_candidates.sort_values(
        [
            "selection_score",
            "mean_fold_utility_delta_vs_constant",
            "minimum_fold_utility_delta_vs_har",
            "annual_turnover",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    .iloc[0]
)
selected_id = str(selected_row["candidate_id"])
selected_development_modifier = candidate_modifiers[selected_id]
selected_development_multiplier = float(selected_development_modifier.mean())

display(
    candidate_search.sort_values("selection_score", ascending=False)
    .head(15)[
        [
            "candidate_id",
            "measure",
            "normaliser",
            "lookback",
            "mode",
            "entry_z",
            "exit_gap",
            "full_cut_distance",
            "risk_multiplier",
            "mean_modifier",
            "selection_score",
            "eligible",
        ]
    ]
)
display(selected_row.to_frame("selected"))

# %% [markdown]
# ## One unchanged application to the later 167-session block

# %%
selected_key = (
    str(selected_row["measure"]),
    str(selected_row["normaliser"]),
    int(selected_row["lookback"]),
)
selected_mapping = {
    "mode": str(selected_row["mode"]),
    "entry_z": float(selected_row["entry_z"]),
    "risk_multiplier": float(selected_row["risk_multiplier"]),
    "exit_gap": (
        None if pd.isna(selected_row["exit_gap"]) else float(selected_row["exit_gap"])
    ),
    "full_cut_distance": (
        None
        if pd.isna(selected_row["full_cut_distance"])
        else float(selected_row["full_cut_distance"])
    ),
}
selected_full_modifier = modifier_from_z(
    z_series[selected_key].to_numpy(dtype=float),
    **selected_mapping,
)
if not np.allclose(
    selected_full_modifier[development_mask],
    selected_development_modifier,
):
    raise RuntimeError("selected modifier does not reproduce on development")

# Evaluation return arrays are deliberately created only after selection.
evaluation_returns = evaluation_har["forward_return"].to_numpy(dtype=float)
evaluation_base_exposure = evaluation_har["exposure"].to_numpy(dtype=float)
evaluation_modifier = selected_full_modifier[~development_mask]
if len(evaluation_modifier) != len(evaluation_har):
    raise RuntimeError("selected modifier does not align to evaluation")

evaluation_exposures = {
    "frozen_har": evaluation_base_exposure,
    "selected_news_overlay": evaluation_base_exposure * evaluation_modifier,
    "development_matched_constant": (
        evaluation_base_exposure * selected_development_multiplier
    ),
}
evaluation_paths = {
    arm: price_path(
        evaluation_returns,
        exposure,
        cost_bps_per_side=COST_BPS,
    )
    for arm, exposure in evaluation_exposures.items()
}
evaluation_summary = pd.DataFrame(
    [
        {"arm": arm, **path_summary(path)}
        for arm, path in evaluation_paths.items()
    ]
)
display(evaluation_summary)

metric_functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "challenger minus comparator mean net return": (
        lambda challenger, comparator: float(np.mean(challenger - comparator))
    ),
    "comparator minus challenger downside squared return": (
        lambda challenger, comparator: float(
            np.mean(np.minimum(comparator, 0.0) ** 2)
            - np.mean(np.minimum(challenger, 0.0) ** 2)
        )
    ),
    "comparator minus challenger expected shortfall 5 loss": (
        lambda challenger, comparator: float(
            expected_shortfall_loss(comparator)
            - expected_shortfall_loss(challenger)
        )
    ),
    "challenger minus comparator downside utility gamma 5": (
        lambda challenger, comparator: float(
            np.mean(downside_utility(challenger) - downside_utility(comparator))
        )
    ),
}
comparison_arms = {
    "selected_overlay_vs_har": "frozen_har",
    "selected_overlay_vs_development_matched_constant": "development_matched_constant",
}
inference_rows = []
selected_net = evaluation_paths["selected_news_overlay"]["net_return"]
seed_offset = 0
for comparison, comparator_arm in comparison_arms.items():
    comparator_net = evaluation_paths[comparator_arm]["net_return"]
    for estimand, metric in metric_functions.items():
        result = bootstrap_metric(
            selected_net,
            comparator_net,
            metric,
            seed=SEED + 100 + seed_offset,
        )
        inference_rows.append(
            {
                "comparison": comparison,
                "comparator_arm": comparator_arm,
                "estimand": estimand,
                **result,
            }
        )
        seed_offset += 1
evaluation_inference = pd.DataFrame(inference_rows)
evaluation_inference["q_bh_eight_tests"] = bh_qvalues(
    evaluation_inference["p_two_sided"].to_numpy(dtype=float)
)
evaluation_inference["bh_reject_q05"] = evaluation_inference[
    "q_bh_eight_tests"
].lt(0.05)
display(evaluation_inference)

# %%
cost_rows = []
for cost_bps in (0.0, 2.0, 5.0, 10.0):
    paths = {
        arm: price_path(
            evaluation_returns,
            exposure,
            cost_bps_per_side=cost_bps,
        )
        for arm, exposure in evaluation_exposures.items()
    }
    for arm, path in paths.items():
        summary = path_summary(path)
        cost_rows.append(
            {
                "cost_bps_per_side": cost_bps,
                "arm": arm,
                "mean_net_bps_session": summary["mean_net_bps_session"],
                "total_return_net": summary["total_return_net"],
                "sharpe_net": summary["sharpe_net"],
                "annual_downside_utility": summary["annual_downside_utility"],
            }
        )
cost_sensitivity = pd.DataFrame(cost_rows)

utility_sensitivity_rows = []
for gamma in (2.0, 5.0, 10.0):
    for comparator_arm in ("frozen_har", "development_matched_constant"):
        difference = float(
            np.mean(
                downside_utility(selected_net, gamma)
                - downside_utility(
                    evaluation_paths[comparator_arm]["net_return"],
                    gamma,
                )
            )
            * 252
        )
        utility_sensitivity_rows.append(
            {
                "gamma": gamma,
                "comparison": f"selected_news_overlay_vs_{comparator_arm}",
                "annual_utility_difference": difference,
            }
        )
utility_sensitivity = pd.DataFrame(utility_sensitivity_rows)

base_path = evaluation_paths["frozen_har"]
overlay_path = evaluation_paths["selected_news_overlay"]
underlying_down = evaluation_returns < 0
gross_difference = overlay_path["gross_return"] - base_path["gross_return"]
economic_decomposition = pd.DataFrame(
    [
        {
            "component": "losses_avoided_on_down_days",
            "gbp_per_million": float(gross_difference[underlying_down].sum() * 1_000_000),
        },
        {
            "component": "upside_given_up_on_up_days",
            "gbp_per_million": float(gross_difference[~underlying_down].sum() * 1_000_000),
        },
        {
            "component": "incremental_trading_cost",
            "gbp_per_million": float(
                -(overlay_path["cost"] - base_path["cost"]).sum() * 1_000_000
            ),
        },
        {
            "component": "net_arithmetic_difference",
            "gbp_per_million": float(
                (overlay_path["net_return"] - base_path["net_return"]).sum()
                * 1_000_000
            ),
        },
        {
            "component": "compounded_ending_wealth_difference",
            "gbp_per_million": float(
                (
                    np.prod(1.0 + overlay_path["net_return"])
                    - np.prod(1.0 + base_path["net_return"])
                )
                * 1_000_000
            ),
        },
    ]
)
display(cost_sensitivity)
display(utility_sensitivity)
display(economic_decomposition)

# %% [markdown]
# ## Evaluation gate

# %%
def inference_row(comparison: str, contains: str) -> pd.Series:
    return evaluation_inference.loc[
        evaluation_inference["comparison"].eq(comparison)
        & evaluation_inference["estimand"].str.contains(contains, regex=False)
    ].iloc[0]


gate_rows = []
for comparison in comparison_arms:
    return_row = inference_row(comparison, "mean net return")
    downside_row = inference_row(comparison, "downside squared")
    es_row = inference_row(comparison, "expected shortfall")
    utility_row = inference_row(comparison, "downside utility")
    gate_rows.append(
        {
            "comparison": comparison,
            "utility_positive_and_bh": bool(
                utility_row["estimate"] > 0
                and utility_row["q_bh_eight_tests"] < 0.05
            ),
            "tail_metric_positive_and_bh": bool(
                (
                    downside_row["estimate"] > 0
                    and downside_row["q_bh_eight_tests"] < 0.05
                )
                or (
                    es_row["estimate"] > 0
                    and es_row["q_bh_eight_tests"] < 0.05
                )
            ),
            "return_lower_bound_within_minus_1bp": bool(
                return_row["ci_low"] >= -0.0001
            ),
        }
    )
evaluation_gates = pd.DataFrame(gate_rows)
economic_value_gate = bool(
    evaluation_gates[
        [
            "utility_positive_and_bh",
            "tail_metric_positive_and_bh",
            "return_lower_bound_within_minus_1bp",
        ]
    ]
    .all(axis=None)
)
evaluation_gates["complete_comparison_gate"] = evaluation_gates[
    [
        "utility_positive_and_bh",
        "tail_metric_positive_and_bh",
        "return_lower_bound_within_minus_1bp",
    ]
].all(axis=1)
display(evaluation_gates)

# %% [markdown]
# ## Licence-safe figures

# %%
dates = evaluation_har["session_date"]
selected_z = z_series[selected_key].loc[~development_mask].to_numpy(dtype=float)
fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True)
axes[0].plot(dates, selected_z, color=CATEGORICAL[0], linewidth=1.4)
axes[0].set_ylabel("Selected pressure z")
axes[0].set_title("Selected pressure state on the later LSEG-33 window")
axes[1].plot(dates, evaluation_modifier, color=CATEGORICAL[1], linewidth=1.5)
axes[1].axhline(
    selected_development_multiplier,
    color=CATEGORICAL[2],
    linestyle="--",
    linewidth=1.2,
    label="Development-matched constant",
)
axes[1].set_ylabel("News multiplier")
axes[1].legend(loc="lower left")
for arm, label, colour in (
    ("frozen_har", "HAR", CATEGORICAL[0]),
    ("selected_news_overlay", "Selected news overlay", CATEGORICAL[1]),
    ("development_matched_constant", "Development-matched constant", CATEGORICAL[2]),
):
    axes[2].plot(
        dates,
        np.cumprod(1.0 + evaluation_paths[arm]["net_return"]),
        label=label,
        color=colour,
        linewidth=1.6,
    )
axes[2].set_ylabel("Growth of £1")
axes[2].legend(loc="upper left")
fig.suptitle(
    "Development-selected LSEG-33 risk rule: later-window application",
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(
    FIGURE_OUTPUT / "fig_lseg33_tuned_rule_evaluation.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)

metric_plot = evaluation_summary.set_index("arm")
fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
for axis, column, title, scale in (
    (axes[0], "total_return_net", "Total net return", 100),
    (axes[1], "annual_downside_deviation", "Annual downside deviation", 100),
    (axes[2], "expected_shortfall_5_loss", "Worst 5% average loss", 100),
):
    values = metric_plot[column] * scale
    axis.bar(
        np.arange(len(values)),
        values,
        color=[CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2]],
    )
    axis.set_xticks(
        np.arange(len(values)),
        ["HAR", "News\noverlay", "Constant\nexposure"],
    )
    axis.set_title(title)
    axis.set_ylabel("Percent")
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.2f}%", ha="center", va="bottom")
fig.suptitle(
    "Economic metrics on the later 167-session LSEG-33 window",
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(
    FIGURE_OUTPUT / "fig_lseg33_tuned_rule_economic_metrics.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)

# %% [markdown]
# ## Save aggregate results and manifest

# %%
input_audit.to_csv(OUTPUT / "input_audit.csv", index=False)
pressure_audit.to_csv(OUTPUT / "pressure_audit.csv", index=False)
pressure_measures.reset_index(names="session_date").to_parquet(
    OUTPUT / "daily_pressure_measures.parquet",
    index=False,
)
candidate_search.to_parquet(OUTPUT / "candidate_search.parquet", index=False)
selected_row.to_frame().T.to_csv(OUTPUT / "selected_candidate.csv", index=False)
pd.DataFrame(
    {
        "session_date": evaluation_har["session_date"],
        "selected_pressure_z": selected_z,
        "selected_modifier": evaluation_modifier,
        "frozen_har_exposure": evaluation_base_exposure,
        "selected_overlay_exposure": evaluation_exposures["selected_news_overlay"],
        "development_matched_constant_exposure": evaluation_exposures[
            "development_matched_constant"
        ],
    }
).to_csv(OUTPUT / "evaluation_state.csv", index=False)
evaluation_summary.to_csv(OUTPUT / "evaluation_summary.csv", index=False)
evaluation_inference.to_csv(OUTPUT / "evaluation_inference.csv", index=False)
evaluation_gates.to_csv(OUTPUT / "evaluation_gates.csv", index=False)
cost_sensitivity.to_csv(OUTPUT / "cost_sensitivity.csv", index=False)
utility_sensitivity.to_csv(OUTPUT / "utility_sensitivity.csv", index=False)
economic_decomposition.to_csv(OUTPUT / "economic_decomposition.csv", index=False)
for arm, path in evaluation_paths.items():
    pd.DataFrame(
        {
            "session_date": evaluation_har["session_date"],
            **path,
        }
    ).to_parquet(OUTPUT / f"{arm}_evaluation_daily.parquet", index=False)

git_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
manifest = {
    "notebook": "80_lseg33_economic_value_rule_search",
    "status": "complete_exploratory_development_tuning_and_chronological_test",
    "git_commit_at_execution": git_commit,
    "spec": str(SPEC_PATH.relative_to(ROOT)),
    "spec_sha256": sha256(SPEC_PATH),
    "candidate_count": len(candidate_search),
    "eligible_candidate_count": len(eligible_candidates),
    "selected_candidate": selected_row.to_dict(),
    "selected_development_multiplier": selected_development_multiplier,
    "evaluation_summary": json.loads(
        evaluation_summary.to_json(orient="records")
    ),
    "evaluation_inference": json.loads(
        evaluation_inference.to_json(orient="records")
    ),
    "evaluation_gates": json.loads(evaluation_gates.to_json(orient="records")),
    "economic_value_gate": economic_value_gate,
    "claim_boundaries": spec["claim_boundaries"],
}
(OUTPUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=True) + "\n"
)

selected_summary = evaluation_summary.set_index("arm").loc[
    "selected_news_overlay"
]
har_summary = evaluation_summary.set_index("arm").loc["frozen_har"]
constant_summary = evaluation_summary.set_index("arm").loc[
    "development_matched_constant"
]
display(
    Markdown(
        f"""### Decision

The development-tuned rule **{'passes' if economic_value_gate else 'does not pass'}**
the frozen later-window economic-value gate.

- Grid: **{len(candidate_search):,}** candidates; **{len(eligible_candidates):,}**
  met the development stability constraints.
- Selected: **{selected_id}** — {selected_row['measure']},
  {selected_row['normaliser']} {int(selected_row['lookback'])}-session normalisation,
  {selected_row['mode']} mapping.
- Later-window reduced-exposure share:
  **{np.mean(evaluation_modifier < 1 - 1e-12):.1%}**.
- Net total return — HAR **{har_summary['total_return_net']:+.2%}**;
  selected overlay **{selected_summary['total_return_net']:+.2%}**;
  development-matched constant **{constant_summary['total_return_net']:+.2%}**.
- Annual downside deviation — HAR **{har_summary['annual_downside_deviation']:.2%}**;
  selected overlay **{selected_summary['annual_downside_deviation']:.2%}**;
  constant **{constant_summary['annual_downside_deviation']:.2%}**.
- Worst-5% average loss — HAR **{har_summary['expected_shortfall_5_loss']:.2%}**;
  selected overlay **{selected_summary['expected_shortfall_5_loss']:.2%}**;
  constant **{constant_summary['expected_shortfall_5_loss']:.2%}**.
- Complete economic-value gate: **{economic_value_gate}**.

This is an exploratory post-hoc search. A positive result is evidence for a
candidate mechanism, not confirmation or a deployable rule.
"""
    )
)
