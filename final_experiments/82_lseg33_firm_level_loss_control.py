# %% [markdown]
# # 82 — LSEG-33 firm-level loss control
#
# The market-wide pressure overlay in Notebook 80 asks one news statistic to time
# the whole 33-stock basket. This experiment instead asks the narrower economic
# question: can Reuters sentiment reduce the weight of the affected company and
# thereby avoid losses beyond what is achieved by simply holding less equity?
#
# The complete candidate family and chronological boundary are frozen in
# `lseg33_firm_level_loss_control_search_v1_20260812.json`. Candidate selection
# uses the first 445 complete 33-company sessions only. The later 167 sessions
# are opened once, after one rule has been selected.

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
pd.set_option("display.max_columns", 120)
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
    / "final_experiments/frozen_specs/"
    "lseg33_firm_level_loss_control_search_v1_20260812.json"
)
OUTPUT = ROOT / "final_experiments/outputs/82_lseg33_firm_level_loss_control"
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


def portfolio_path(
    asset_returns: np.ndarray,
    target_weights: np.ndarray,
    *,
    cost_bps_per_side: float,
) -> dict[str, np.ndarray]:
    returns = np.asarray(asset_returns, dtype=float)
    targets = np.asarray(target_weights, dtype=float)
    if returns.shape != targets.shape or returns.ndim != 2:
        raise ValueError("returns and targets must be matching two-dimensional arrays")
    if not np.isfinite(returns).all() or not np.isfinite(targets).all():
        raise ValueError("returns and targets must be finite")
    if np.any(targets < 0.0) or np.any(targets.sum(axis=1) > 1.0 + 1e-12):
        raise ValueError("targets must be long-only with gross exposure no greater than one")

    current = np.zeros(targets.shape[1], dtype=float)
    gross_rows: list[float] = []
    net_rows: list[float] = []
    turnover_rows: list[float] = []
    cost_rows: list[float] = []
    for target, period_return in zip(targets, returns, strict=True):
        turnover = float(np.abs(target - current).sum())
        cost = turnover * cost_bps_per_side / 10_000.0
        gross = float(np.dot(target, period_return))
        net = gross - cost
        gross_rows.append(gross)
        net_rows.append(net)
        turnover_rows.append(turnover)
        cost_rows.append(cost)
        current = target * (1.0 + period_return) / (1.0 + net)

    liquidation_turnover = float(np.abs(current).sum())
    liquidation_cost = liquidation_turnover * cost_bps_per_side / 10_000.0
    turnover_rows[-1] += liquidation_turnover
    cost_rows[-1] += liquidation_cost
    net_rows[-1] = (1.0 + net_rows[-1]) * (1.0 - liquidation_cost) - 1.0
    return {
        "gross_return": np.asarray(gross_rows),
        "net_return": np.asarray(net_rows),
        "turnover": np.asarray(turnover_rows),
        "cost": np.asarray(cost_rows),
        "gross_exposure": targets.sum(axis=1),
    }


def path_summary(path: dict[str, np.ndarray]) -> dict[str, float | int]:
    net = path["net_return"]
    mean = float(net.mean())
    volatility = float(net.std(ddof=1))
    downside_deviation = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)))
    return {
        "n_sessions": len(net),
        "mean_net_bps_session": mean * 10_000,
        "total_return_net": float(np.prod(1.0 + net) - 1.0),
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
        "mean_exposure": float(path["gross_exposure"].mean()),
        "annual_turnover": float(path["turnover"].mean() * 252),
    }


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


def conditional_negative_share_residual(frame: pd.DataFrame) -> pd.Series:
    residual = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("session_date", sort=False):
        y = group["negative_share"].to_numpy(dtype=float)
        x = np.column_stack(
            [
                np.ones(len(group)),
                group["mean_continuous"].to_numpy(dtype=float),
                np.log1p(group["unique_headline_count"].to_numpy(dtype=float)),
            ]
        )
        fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        residual.loc[group.index] = y - fitted
    return residual


def percentile_matrix(values: np.ndarray) -> np.ndarray:
    return (
        pd.DataFrame(values)
        .rank(axis=1, method="average", pct=True, na_option="keep")
        .to_numpy(dtype=float)
    )


def firm_prior_z(values: np.ndarray, lookback: int) -> np.ndarray:
    frame = pd.DataFrame(values)
    prior = frame.shift(1)
    minimum = int(math.ceil(lookback / 2))
    centre = prior.rolling(lookback, min_periods=minimum).mean()
    scale = prior.rolling(lookback, min_periods=minimum).std(ddof=1)
    return ((frame - centre) / scale.where(scale.gt(0))).to_numpy(dtype=float)


def risk_modifier(
    risk_percentile: np.ndarray,
    market_gate: np.ndarray,
    *,
    threshold: float,
    risk_multiplier: float,
    mapping: str,
) -> np.ndarray:
    percentile = np.asarray(risk_percentile, dtype=float)
    modifier = np.ones_like(percentile)
    finite = np.isfinite(percentile) & market_gate[:, None]
    if mapping == "step":
        modifier[finite & (percentile >= threshold)] = risk_multiplier
    elif mapping == "smooth":
        intensity = np.clip((percentile - threshold) / (1.0 - threshold), 0.0, 1.0)
        modifier[finite] = 1.0 - (1.0 - risk_multiplier) * intensity[finite]
    else:
        raise ValueError(f"unknown mapping: {mapping}")
    return modifier


print(f"repo:   {ROOT}")
print(f"spec:   {SPEC_PATH.relative_to(ROOT)}")
print(f"output: {OUTPUT.relative_to(ROOT)}")

# %% [markdown]
# ## Input identities and the complete-company clock

# %%
spec = json.loads(SPEC_PATH.read_text())
if spec["status"] != "frozen_before_notebook_82_search_result":
    raise ValueError("unexpected search-spec status")

input_paths = {key: ROOT / value["path"] for key, value in spec["inputs"].items()}
input_audit_rows = []
for key, path in input_paths.items():
    actual = sha256(path)
    if actual != spec["inputs"][key]["sha256"]:
        raise ValueError(f"input hash mismatch: {key}")
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

panels: dict[str, pd.DataFrame] = {}
for scorer, key in (("gemma4_26b", "gemma_firm_open"), ("finbert", "finbert_firm_open")):
    frame = pd.read_parquet(input_paths[key]).copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame.sort_values(["session_date", "symbol"], kind="mergesort", inplace=True)
    if frame.duplicated(["session_date", "symbol"]).any():
        raise ValueError(f"duplicate firm-open row in {scorer}")
    panels[scorer] = frame

symbols = sorted(set(panels["gemma4_26b"]["symbol"]))
if len(symbols) != N_FIRMS or set(symbols) != set(panels["finbert"]["symbol"]):
    raise ValueError("the two scorers do not share the same 33-company cohort")
complete_by_scorer = {
    scorer: set(
        frame.groupby("session_date")["symbol"].nunique().loc[lambda value: value.eq(N_FIRMS)].index
    )
    for scorer, frame in panels.items()
}
complete_dates = pd.DatetimeIndex(
    sorted(complete_by_scorer["gemma4_26b"] & complete_by_scorer["finbert"])
)
development_dates = complete_dates[complete_dates < BOUNDARY]
evaluation_dates = complete_dates[complete_dates >= BOUNDARY]
if len(development_dates) != 445 or len(evaluation_dates) != 167:
    raise ValueError(
        f"unexpected complete-session split: {len(development_dates)} / {len(evaluation_dates)}"
    )


def complete_frame(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame.loc[frame["session_date"].isin(complete_dates)].copy()
    return complete.set_index(["session_date", "symbol"]).loc[
        pd.MultiIndex.from_product([complete_dates, symbols])
    ].reset_index()


panels = {scorer: complete_frame(frame) for scorer, frame in panels.items()}
gemma_returns = panels["gemma4_26b"]["raw_open_h1"].to_numpy(dtype=float).reshape(-1, N_FIRMS)
finbert_returns = panels["finbert"]["raw_open_h1"].to_numpy(dtype=float).reshape(-1, N_FIRMS)
if not np.allclose(gemma_returns, finbert_returns, atol=1e-14):
    raise RuntimeError("Gemma and FinBERT panels do not reproduce identical return matrices")
return_matrix = gemma_returns
development_mask = complete_dates < BOUNDARY
evaluation_mask = ~development_mask
fold_indices = np.array_split(np.arange(len(development_dates)), 3)

population_audit = pd.DataFrame(
    [
        {
            "companies": len(symbols),
            "complete_sessions": len(complete_dates),
            "development_sessions": len(development_dates),
            "evaluation_sessions": len(evaluation_dates),
            "development_fold_sizes": "/".join(str(len(values)) for values in fold_indices),
            "first_session": complete_dates.min(),
            "last_session": complete_dates.max(),
        }
    ]
)
display(population_audit)

# %% [markdown]
# ## Return-free company-risk features

# %%
risk_matrices: dict[tuple[str, str, str], np.ndarray] = {}
market_gates: dict[tuple[str, str], np.ndarray] = {}
feature_audit_rows = []
for scorer, frame in panels.items():
    frame = frame.copy()
    frame["conditional_negative_share_residual"] = conditional_negative_share_residual(frame)
    frame["negative_mean_continuous"] = -frame["mean_continuous"]
    frame["negative_trimmed_mean"] = -frame["trimmed_mean"]
    frame["negative_median_continuous"] = -frame["median_continuous"]
    frame["negative_decayed_state"] = -frame["decayed_state"]
    frame["attention_weighted_negative_share"] = frame["negative_share"] * np.log1p(
        frame["unique_headline_count"]
    )
    features = spec["candidate_grid"]["risk_features"]
    for feature in features:
        raw = frame[feature].to_numpy(dtype=float).reshape(-1, N_FIRMS)
        transformations = {
            "current_cross_section": raw,
            "firm_rolling_63": firm_prior_z(raw, 63),
            "firm_rolling_126": firm_prior_z(raw, 126),
        }
        for normaliser, transformed in transformations.items():
            percentiles = percentile_matrix(transformed)
            risk_matrices[(scorer, feature, normaliser)] = percentiles
            feature_audit_rows.append(
                {
                    "scorer": scorer,
                    "feature": feature,
                    "normaliser": normaliser,
                    "finite_share": float(np.isfinite(percentiles).mean()),
                }
            )

    aggregate = (
        frame["negative_share"].to_numpy(dtype=float).reshape(-1, N_FIRMS).mean(axis=1)
    )
    aggregate_series = pd.Series(aggregate, index=complete_dates)
    prior = aggregate_series.shift(1)
    rolling = prior.rolling(126, min_periods=63)
    market_gates[(scorer, "always")] = np.ones(len(complete_dates), dtype=bool)
    market_gates[(scorer, "aggregate_negative_share_above_prior_126_median")] = (
        aggregate_series.gt(rolling.quantile(0.50)).fillna(False).to_numpy(dtype=bool)
    )
    market_gates[(scorer, "aggregate_negative_share_above_prior_126_q75")] = (
        aggregate_series.gt(rolling.quantile(0.75)).fillna(False).to_numpy(dtype=bool)
    )

feature_audit = pd.DataFrame(feature_audit_rows)
display(feature_audit.groupby(["normaliser"])["finite_share"].agg(["min", "max"]))

# %% [markdown]
# ## Complete 4,608-candidate development search
#
# No evaluation return is used below. Each candidate is judged against fixed
# stock-by-stock weights carrying the same development-average exposure.

# %%
development_returns = return_matrix[development_mask]
base_development_weights = np.full_like(development_returns, 1.0 / N_FIRMS)
candidate_rows = []
candidate_modifiers: dict[str, np.ndarray] = {}
candidate_number = 0

for scorer, feature, normaliser, threshold, multiplier, mapping, gate_name in product(
    spec["candidate_grid"]["scorers"],
    spec["candidate_grid"]["risk_features"],
    spec["candidate_grid"]["normalisers"],
    spec["candidate_grid"]["cross_sectional_risk_percentile_thresholds"],
    spec["candidate_grid"]["risk_multipliers"],
    spec["candidate_grid"]["mappings"],
    spec["candidate_grid"]["market_gates"],
):
    candidate_number += 1
    candidate_id = f"candidate_{candidate_number:04d}"
    full_modifier = risk_modifier(
        risk_matrices[(scorer, feature, normaliser)],
        market_gates[(scorer, gate_name)],
        threshold=float(threshold),
        risk_multiplier=float(multiplier),
        mapping=mapping,
    )
    modifier = full_modifier[development_mask]
    candidate_modifiers[candidate_id] = full_modifier
    overlay_weights = base_development_weights * modifier
    symbol_mean_modifier = modifier.mean(axis=0)
    symbol_constant_weights = np.tile(symbol_mean_modifier / N_FIRMS, (len(modifier), 1))
    fold_rows = []
    for fold_number, indices in enumerate(fold_indices, start=1):
        overlay = portfolio_path(
            development_returns[indices],
            overlay_weights[indices],
            cost_bps_per_side=COST_BPS,
        )
        constant = portfolio_path(
            development_returns[indices],
            symbol_constant_weights[indices],
            cost_bps_per_side=COST_BPS,
        )
        overlay_net = overlay["net_return"]
        constant_net = constant["net_return"]
        fold_rows.append(
            {
                "fold": fold_number,
                "utility_delta_vs_symbol_constant": float(
                    np.mean(downside_utility(overlay_net) - downside_utility(constant_net))
                    * 252
                ),
                "es_reduction_vs_symbol_constant": float(
                    expected_shortfall_loss(constant_net)
                    - expected_shortfall_loss(overlay_net)
                ),
                "net_delta_vs_symbol_constant": float(
                    np.mean(overlay_net - constant_net)
                ),
            }
        )
    fold = pd.DataFrame(fold_rows).set_index("fold")
    full_path = portfolio_path(
        development_returns,
        overlay_weights,
        cost_bps_per_side=COST_BPS,
    )
    eligible = bool(
        fold["utility_delta_vs_symbol_constant"].gt(0).sum() >= 2
        and fold["es_reduction_vs_symbol_constant"].gt(0).sum() >= 2
    )
    row: dict[str, float | int | str | bool] = {
        "candidate_id": candidate_id,
        "scorer": scorer,
        "feature": feature,
        "normaliser": normaliser,
        "threshold": float(threshold),
        "risk_multiplier": float(multiplier),
        "mapping": mapping,
        "market_gate": gate_name,
        "mean_exposure": float(overlay_weights.sum(axis=1).mean()),
        "reduced_company_share": float(np.mean(modifier < 1.0 - 1e-12)),
        "annual_turnover": float(full_path["turnover"].mean() * 252),
        "eligible": eligible,
        "selection_score": float(fold["utility_delta_vs_symbol_constant"].min()),
        "mean_fold_utility_delta_vs_symbol_constant": float(
            fold["utility_delta_vs_symbol_constant"].mean()
        ),
        "mean_fold_es_reduction_vs_symbol_constant": float(
            fold["es_reduction_vs_symbol_constant"].mean()
        ),
    }
    for fold_number in (1, 2, 3):
        for column in fold.columns:
            row[f"fold_{fold_number}_{column}"] = float(fold.loc[fold_number, column])
    candidate_rows.append(row)

candidate_search = pd.DataFrame(candidate_rows)
if len(candidate_search) != spec["candidate_grid"]["expected_candidates"]:
    raise RuntimeError("candidate count differs from frozen grid")
eligible_candidates = candidate_search.loc[candidate_search["eligible"]].copy()
if eligible_candidates.empty:
    raise RuntimeError("no candidate satisfies the frozen eligibility rule")
selected = (
    eligible_candidates.sort_values(
        [
            "selection_score",
            "mean_fold_utility_delta_vs_symbol_constant",
            "mean_fold_es_reduction_vs_symbol_constant",
            "annual_turnover",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    .iloc[0]
)
selected_id = str(selected["candidate_id"])
selected_modifier = candidate_modifiers[selected_id]
selected_development_modifier = selected_modifier[development_mask]
symbol_mean_modifier = selected_development_modifier.mean(axis=0)
aggregate_mean_modifier = float(selected_development_modifier.mean())

display(
    candidate_search.sort_values("selection_score", ascending=False)
    .head(15)[
        [
            "candidate_id",
            "scorer",
            "feature",
            "normaliser",
            "threshold",
            "risk_multiplier",
            "mapping",
            "market_gate",
            "mean_exposure",
            "selection_score",
            "eligible",
        ]
    ]
)
display(selected.to_frame("selected"))

# %% [markdown]
# ## One unchanged forward evaluation on the later 167 sessions

# %%
evaluation_returns = return_matrix[evaluation_mask]
base_weights = np.full_like(evaluation_returns, 1.0 / N_FIRMS)
overlay_weights = base_weights * selected_modifier[evaluation_mask]
symbol_constant_weights = np.tile(
    symbol_mean_modifier / N_FIRMS,
    (len(evaluation_dates), 1),
)
aggregate_constant_weights = np.full_like(
    evaluation_returns,
    aggregate_mean_modifier / N_FIRMS,
)
weight_paths = {
    "equal_weight_long": base_weights,
    "selected_firm_news_overlay": overlay_weights,
    "development_symbol_matched_constant": symbol_constant_weights,
    "development_aggregate_matched_constant": aggregate_constant_weights,
}
paths = {
    arm: portfolio_path(
        evaluation_returns,
        weights,
        cost_bps_per_side=COST_BPS,
    )
    for arm, weights in weight_paths.items()
}
evaluation_summary = pd.DataFrame(
    [{"arm": arm, **path_summary(path)} for arm, path in paths.items()]
)
display(evaluation_summary)

# %% [markdown]
# ## Corrected inference against all three exposure controls

# %%
metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "challenger minus comparator mean net return": lambda challenger, comparator: float(
        np.mean(challenger - comparator)
    ),
    "comparator minus challenger downside squared return": lambda challenger, comparator: float(
        np.mean(np.minimum(comparator, 0.0) ** 2 - np.minimum(challenger, 0.0) ** 2)
    ),
    "comparator minus challenger expected shortfall loss": lambda challenger, comparator: float(
        expected_shortfall_loss(comparator) - expected_shortfall_loss(challenger)
    ),
    "challenger minus comparator downside utility gamma 5": lambda challenger, comparator: float(
        np.mean(downside_utility(challenger) - downside_utility(comparator))
    ),
}
comparators = {
    "selected_overlay_vs_equal_weight_long": "equal_weight_long",
    "selected_overlay_vs_symbol_matched_constant": "development_symbol_matched_constant",
    "selected_overlay_vs_aggregate_matched_constant": "development_aggregate_matched_constant",
}
inference_rows = []
challenger = paths["selected_firm_news_overlay"]["net_return"]
test_number = 0
for comparison, comparator_arm in comparators.items():
    comparator = paths[comparator_arm]["net_return"]
    for estimand, metric in metrics.items():
        test_number += 1
        inference_rows.append(
            {
                "comparison": comparison,
                "comparator_arm": comparator_arm,
                "estimand": estimand,
                **bootstrap_metric(
                    challenger,
                    comparator,
                    metric,
                    seed=SEED + 100 + test_number,
                ),
            }
        )
evaluation_inference = pd.DataFrame(inference_rows)
evaluation_inference["q_bh_twelve_tests"] = bh_qvalues(
    evaluation_inference["p_two_sided"].to_numpy()
)
evaluation_inference["bh_reject_q05"] = evaluation_inference[
    "q_bh_twelve_tests"
].lt(0.05)
display(evaluation_inference)

# %% [markdown]
# ## Economic decomposition, severe-loss states and stability

# %%
symbol_control = paths["development_symbol_matched_constant"]
control_down = symbol_control["gross_return"] < 0
economic_decomposition = pd.DataFrame(
    [
        {
            "component": "losses_avoided_when_fixed_control_lost",
            "gbp_per_million": float(
                np.sum(
                    paths["selected_firm_news_overlay"]["gross_return"][control_down]
                    - symbol_control["gross_return"][control_down]
                )
                * 1_000_000
            ),
        },
        {
            "component": "return_difference_when_fixed_control_gained",
            "gbp_per_million": float(
                np.sum(
                    paths["selected_firm_news_overlay"]["gross_return"][~control_down]
                    - symbol_control["gross_return"][~control_down]
                )
                * 1_000_000
            ),
        },
        {
            "component": "incremental_trading_cost",
            "gbp_per_million": float(
                -np.sum(
                    paths["selected_firm_news_overlay"]["cost"]
                    - symbol_control["cost"]
                )
                * 1_000_000
            ),
        },
        {
            "component": "net_arithmetic_difference",
            "gbp_per_million": float(
                np.sum(
                    paths["selected_firm_news_overlay"]["net_return"]
                    - symbol_control["net_return"]
                )
                * 1_000_000
            ),
        },
        {
            "component": "compounded_ending_wealth_difference",
            "gbp_per_million": float(
                (
                    np.prod(1.0 + paths["selected_firm_news_overlay"]["net_return"])
                    - np.prod(1.0 + symbol_control["net_return"])
                )
                * 1_000_000
            ),
        },
    ]
)

quantile_labels = ["worst 10% control days", "middle 80%", "best 10% control days"]
control_net = symbol_control["net_return"]
lower, upper = np.quantile(control_net, [0.10, 0.90])
state_masks = [control_net <= lower, (control_net > lower) & (control_net < upper), control_net >= upper]
return_state_rows = []
for label, mask in zip(quantile_labels, state_masks, strict=True):
    return_state_rows.append(
        {
            "state": label,
            "sessions": int(mask.sum()),
            "control_return_bps_session": float(control_net[mask].mean() * 10_000),
            "overlay_minus_control_bps_session": float(
                np.mean(challenger[mask] - control_net[mask]) * 10_000
            ),
            "overlay_minus_control_gbp_per_million": float(
                np.sum(challenger[mask] - control_net[mask]) * 1_000_000
            ),
        }
    )
return_states = pd.DataFrame(return_state_rows)

half_indices = np.array_split(np.arange(len(evaluation_dates)), 2)
stability_rows = []
for half_number, indices in enumerate(half_indices, start=1):
    stability_rows.append(
        {
            "evaluation_half": half_number,
            "start": evaluation_dates[indices].min(),
            "end": evaluation_dates[indices].max(),
            "overlay_minus_symbol_constant_bps_session": float(
                np.mean(challenger[indices] - control_net[indices]) * 10_000
            ),
            "compounded_wealth_difference": float(
                np.prod(1.0 + challenger[indices])
                - np.prod(1.0 + control_net[indices])
            ),
        }
    )
evaluation_halves = pd.DataFrame(stability_rows)

leave_one_out_rows = []
for company_number, symbol in enumerate(symbols):
    keep = np.arange(N_FIRMS) != company_number
    loo_returns = evaluation_returns[:, keep]
    loo_modifier = selected_modifier[evaluation_mask][:, keep]
    loo_base = np.full_like(loo_returns, 1.0 / (N_FIRMS - 1))
    loo_overlay = portfolio_path(
        loo_returns,
        loo_base * loo_modifier,
        cost_bps_per_side=COST_BPS,
    )
    loo_symbol_mean = symbol_mean_modifier[keep]
    loo_constant = portfolio_path(
        loo_returns,
        np.tile(loo_symbol_mean / (N_FIRMS - 1), (len(evaluation_dates), 1)),
        cost_bps_per_side=COST_BPS,
    )
    leave_one_out_rows.append(
        {
            "excluded_symbol": symbol,
            "overlay_minus_symbol_constant_bps_session": float(
                np.mean(loo_overlay["net_return"] - loo_constant["net_return"]) * 10_000
            ),
            "compounded_ending_wealth_difference": float(
                np.prod(1.0 + loo_overlay["net_return"])
                - np.prod(1.0 + loo_constant["net_return"])
            ),
        }
    )
leave_one_company_out = pd.DataFrame(leave_one_out_rows)

cost_rows = []
for cost in spec["portfolio"]["cost_sensitivity_bps_per_side"]:
    for arm, weights in weight_paths.items():
        summary = path_summary(
            portfolio_path(
                evaluation_returns,
                weights,
                cost_bps_per_side=float(cost),
            )
        )
        cost_rows.append(
            {
                "cost_bps_per_side": float(cost),
                "arm": arm,
                "mean_net_bps_session": summary["mean_net_bps_session"],
                "total_return_net": summary["total_return_net"],
                "sharpe_net": summary["sharpe_net"],
                "annual_downside_utility": summary["annual_downside_utility"],
            }
        )
cost_sensitivity = pd.DataFrame(cost_rows)

display(economic_decomposition)
display(return_states)
display(evaluation_halves)
display(leave_one_company_out.describe())

# %% [markdown]
# ## Gate audit

# %%


def inference_row(comparison: str, pattern: str) -> pd.Series:
    rows = evaluation_inference.loc[
        evaluation_inference["comparison"].eq(comparison)
        & evaluation_inference["estimand"].str.contains(pattern, regex=False)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"could not resolve inference row: {comparison} / {pattern}")
    return rows.iloc[0]


comparison = "selected_overlay_vs_symbol_matched_constant"
return_row = inference_row(comparison, "mean net return")
downside_row = inference_row(comparison, "downside squared")
es_row = inference_row(comparison, "expected shortfall")
utility_row = inference_row(comparison, "downside utility")
wealth_difference = float(
    economic_decomposition.loc[
        economic_decomposition["component"].eq("compounded_ending_wealth_difference"),
        "gbp_per_million",
    ].iloc[0]
)
evaluation_gates = pd.DataFrame(
    [
        {
            "comparison": comparison,
            "corrected_tail_or_utility_evidence": bool(
                (
                    downside_row["estimate"] > 0
                    and downside_row["q_bh_twelve_tests"] < 0.05
                )
                or (es_row["estimate"] > 0 and es_row["q_bh_twelve_tests"] < 0.05)
                or (
                    utility_row["estimate"] > 0
                    and utility_row["q_bh_twelve_tests"] < 0.05
                )
            ),
            "return_lower_bound_within_minus_1bp": bool(return_row["ci_low"] > -0.0001),
            "positive_compounded_wealth_difference": bool(wealth_difference > 0),
            "positive_both_evaluation_halves": bool(
                evaluation_halves["compounded_wealth_difference"].gt(0).all()
            ),
            "positive_all_leave_one_company_out": bool(
                leave_one_company_out["compounded_ending_wealth_difference"].gt(0).all()
            ),
        }
    ]
)
evaluation_gates["complete_economic_value_gate"] = evaluation_gates[
    [
        "corrected_tail_or_utility_evidence",
        "return_lower_bound_within_minus_1bp",
        "positive_compounded_wealth_difference",
        "positive_both_evaluation_halves",
        "positive_all_leave_one_company_out",
    ]
].all(axis=1)
display(evaluation_gates)

# %% [markdown]
# ## Dissertation figures

# %%
colors = {
    "equal_weight_long": CATEGORICAL[0],
    "selected_firm_news_overlay": CATEGORICAL[2],
    "development_symbol_matched_constant": CATEGORICAL[1],
    "development_aggregate_matched_constant": CATEGORICAL[3],
}
labels = {
    "equal_weight_long": "Equal-weight long",
    "selected_firm_news_overlay": "Firm-level Reuters control",
    "development_symbol_matched_constant": "Same stock-by-stock exposure",
    "development_aggregate_matched_constant": "Same total exposure",
}

fig, axes = plt.subplots(3, 1, figsize=(11.4, 10.2), sharex=True)
for arm in (
    "equal_weight_long",
    "selected_firm_news_overlay",
    "development_symbol_matched_constant",
):
    axes[0].plot(
        evaluation_dates,
        np.cumprod(1.0 + paths[arm]["net_return"]),
        label=labels[arm],
        color=colors[arm],
        lw=2.0,
    )
axes[0].axhline(1.0, color="#999999", lw=0.8)
axes[0].set_ylabel("£1 initial wealth")
axes[0].set_title("Does company-specific Reuters risk control preserve wealth?")
axes[0].legend(frameon=False, ncol=3, fontsize=9)

cumulative_difference = np.cumsum(challenger - control_net) * 10_000
axes[1].plot(evaluation_dates, cumulative_difference, color=CATEGORICAL[2], lw=2.0)
axes[1].axhline(0.0, color="#666666", lw=0.8)
axes[1].set_ylabel("Cumulative difference (bps)")
axes[1].set_title("Reuters timing value versus the same stock-by-stock exposure")

axes[2].plot(
    evaluation_dates,
    paths["selected_firm_news_overlay"]["gross_exposure"],
    color=CATEGORICAL[2],
    lw=1.5,
    label="Firm-level Reuters control",
)
axes[2].axhline(
    float(symbol_constant_weights.sum(axis=1).mean()),
    color=CATEGORICAL[1],
    lw=1.5,
    ls="--",
    label="Fixed matched exposure",
)
axes[2].set_ylabel("Gross equity exposure")
axes[2].set_xlabel("Signal date")
axes[2].set_ylim(0, 1.03)
axes[2].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(
    FIGURE_OUTPUT / "fig_lseg33_firm_loss_control_evaluation.png",
    dpi=190,
    bbox_inches="tight",
)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
waterfall = economic_decomposition.iloc[:3]
axes[0].bar(
    ["Losses\navoided", "Return on\ngain days", "Extra\ncost"],
    waterfall["gbp_per_million"],
    color=[CATEGORICAL[2], CATEGORICAL[1], CATEGORICAL[3]],
)
axes[0].axhline(0, color="#555555", lw=0.8)
axes[0].set_ylabel("£ per £1m over 167 sessions")
axes[0].set_title("Where the economic difference comes from")

axes[1].bar(
    return_states["state"],
    return_states["overlay_minus_control_gbp_per_million"],
    color=[CATEGORICAL[2], CATEGORICAL[0], CATEGORICAL[1]],
)
axes[1].axhline(0, color="#555555", lw=0.8)
axes[1].tick_params(axis="x", rotation=18)
axes[1].set_ylabel("£ per £1m")
axes[1].set_title("Value by fixed-control return state")

summary_index = evaluation_summary.set_index("arm")
metric_names = ["annual_downside_deviation", "expected_shortfall_5_loss"]
x = np.arange(len(metric_names))
width = 0.26
for offset, arm in zip(
    (-width, 0.0, width),
    (
        "equal_weight_long",
        "selected_firm_news_overlay",
        "development_symbol_matched_constant",
    ),
    strict=True,
):
    axes[2].bar(
        x + offset,
        [summary_index.loc[arm, metric] * 100 for metric in metric_names],
        width,
        label=labels[arm],
        color=colors[arm],
    )
axes[2].set_xticks(x, ["Downside\ndeviation", "Worst 5%\naverage loss"])
axes[2].set_ylabel("Percent")
axes[2].set_title("Risk must beat matched exposure")
axes[2].legend(frameon=False, fontsize=7.8)
fig.tight_layout()
fig.savefig(
    FIGURE_OUTPUT / "fig_lseg33_firm_loss_saving.png",
    dpi=190,
    bbox_inches="tight",
)
plt.close(fig)

# %% [markdown]
# ## Save aggregate, licence-safe evidence

# %%
candidate_search.to_parquet(OUTPUT / "candidate_search.parquet", index=False)
selected.to_frame().T.to_csv(OUTPUT / "selected_candidate.csv", index=False)
input_audit.to_csv(OUTPUT / "input_audit.csv", index=False)
population_audit.to_csv(OUTPUT / "population_audit.csv", index=False)
feature_audit.to_csv(OUTPUT / "feature_audit.csv", index=False)
evaluation_summary.to_csv(OUTPUT / "evaluation_summary.csv", index=False)
evaluation_inference.to_csv(OUTPUT / "evaluation_inference.csv", index=False)
economic_decomposition.to_csv(OUTPUT / "economic_decomposition.csv", index=False)
return_states.to_csv(OUTPUT / "return_states.csv", index=False)
evaluation_halves.to_csv(OUTPUT / "evaluation_halves.csv", index=False)
leave_one_company_out.to_csv(OUTPUT / "leave_one_company_out.csv", index=False)
cost_sensitivity.to_csv(OUTPUT / "cost_sensitivity.csv", index=False)
evaluation_gates.to_csv(OUTPUT / "evaluation_gates.csv", index=False)
pd.DataFrame(
    {
        "session_date": evaluation_dates,
        "equal_weight_long": paths["equal_weight_long"]["net_return"],
        "selected_firm_news_overlay": challenger,
        "development_symbol_matched_constant": control_net,
        "development_aggregate_matched_constant": paths[
            "development_aggregate_matched_constant"
        ]["net_return"],
        "selected_gross_exposure": paths["selected_firm_news_overlay"][
            "gross_exposure"
        ],
    }
).to_parquet(OUTPUT / "evaluation_daily_paths.parquet", index=False)

try:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
except (OSError, subprocess.CalledProcessError):
    git_commit = "unavailable"

manifest = {
    "notebook": "82_lseg33_firm_level_loss_control",
    "spec_path": str(SPEC_PATH.relative_to(ROOT)),
    "spec_sha256": sha256(SPEC_PATH),
    "git_commit": git_commit,
    "inputs": input_audit.to_dict(orient="records"),
    "population": population_audit.iloc[0].astype(str).to_dict(),
    "candidate_count": len(candidate_search),
    "eligible_candidate_count": len(eligible_candidates),
    "selected_candidate": selected.to_dict(),
    "primary_cost_bps_per_side": COST_BPS,
    "bootstrap": {
        "block_length": BLOCK_LENGTH,
        "replications": BOOTSTRAP_REPLICATIONS,
        "family_size": len(evaluation_inference),
        "seed_base": SEED,
    },
    "claim_status": "post_hoc_exploratory_chronological_forward_test",
    "licensed_text_written": False,
}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

gate_pass = bool(evaluation_gates.loc[0, "complete_economic_value_gate"])
display(
    Markdown(
        f"""
## Decision

The firm-level LSEG economic-value gate **{'passes' if gate_pass else 'does not pass'}**.

- Selected rule: **{selected['scorer']} / {selected['feature']} /
  {selected['normaliser']}**, {selected['mapping']} reduction above the
  cross-sectional **{selected['threshold']:.0%}** risk percentile, multiplier
  **{selected['risk_multiplier']:.2f}**, market gate **{selected['market_gate']}**.
- Later-block compounded wealth difference versus the development
  stock-by-stock matched control: **£{wealth_difference:,.0f} per £1m**.
- This remains a post-hoc LSEG analysis. A pass supports economic usefulness in
  this fixed 33-company set; it does not establish a causal or population-wide rule.
"""
    )
)
