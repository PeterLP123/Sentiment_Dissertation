from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .artifact_io import atomic_write_json, sha256_file
from .trading_effectiveness import benjamini_hochberg


class L3AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class GStudyResult:
    components: dict[str, float]
    variance_shares: dict[str, float]
    g_coefficient: float
    dependability_coefficient: float
    converged: bool


@dataclass(frozen=True)
class L3Result:
    output_dir: Path
    components_path: Path
    item_reliability_path: Path
    daily_reliability_path: Path
    rules_path: Path
    hypotheses_path: Path
    combined_hypotheses_path: Path | None = None


def load_measurements(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise L3AnalysisError("score matrix is empty")
    frame = frame[(frame["status"] == "success") & frame["label_probabilities"].map(lambda value: isinstance(value, dict))].copy()
    frame["score"] = frame["label_probabilities"].map(
        lambda value: float(value.get("positive", 0)) - float(value.get("negative", 0))
    )
    frame["symbol"] = frame["metadata"].map(lambda value: value.get("symbol"))
    frame["news_date"] = frame["metadata"].map(lambda value: value.get("news_date"))
    frame["chronological_split"] = frame["metadata"].map(lambda value: value.get("chronological_split"))
    frame["pb_agreement_tier"] = frame["metadata"].map(lambda value: value.get("pb_agreement_tier"))
    return frame


def fit_crossed_gstudy(measurements: pd.DataFrame, *, n_models: int = 5, n_prompts: int = 3, n_samples: int = 5) -> GStudyResult:
    required = {"item_id", "model_id", "prompt_id", "score"}
    missing = required - set(measurements.columns)
    if missing:
        raise L3AnalysisError(f"G-study data is missing columns: {', '.join(sorted(missing))}")
    data = measurements.dropna(subset=list(required)).copy()
    data["all"] = "all"
    data["item_model"] = data["item_id"].astype(str) + "|" + data["model_id"].astype(str)
    data["item_prompt"] = data["item_id"].astype(str) + "|" + data["prompt_id"].astype(str)
    data["model_prompt"] = data["model_id"].astype(str) + "|" + data["prompt_id"].astype(str)
    variance_formulas = {
        "item": "0 + C(item_id)",
        "model": "0 + C(model_id)",
        "prompt": "0 + C(prompt_id)",
        "item_model": "0 + C(item_model)",
        "item_prompt": "0 + C(item_prompt)",
        "model_prompt": "0 + C(model_prompt)",
    }
    model = smf.mixedlm("score ~ 1", data, groups=data["all"], vc_formula=variance_formulas, re_formula="0")
    fitted = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
    if not fitted.converged:
        fitted = model.fit(reml=True, method="powell", maxiter=1000, disp=False)
    components = {
        name: max(0.0, float(value))
        for name, value in zip(fitted.model.exog_vc.names, fitted.vcomp, strict=True)
    }
    components["sample_residual"] = max(0.0, float(fitted.scale))
    total = sum(components.values())
    shares = {key: value / total if total else 0.0 for key, value in components.items()}
    item = components.get("item", 0.0)
    relative_error = (
        components.get("item_model", 0.0) / n_models
        + components.get("item_prompt", 0.0) / n_prompts
        + components["sample_residual"] / (n_models * n_prompts * n_samples)
    )
    absolute_error = (
        relative_error
        + components.get("model", 0.0) / n_models
        + components.get("prompt", 0.0) / n_prompts
        + components.get("model_prompt", 0.0) / (n_models * n_prompts)
    )
    g_coefficient = item / (item + relative_error) if item + relative_error else 0.0
    dependability = item / (item + absolute_error) if item + absolute_error else 0.0
    return GStudyResult(components, shares, g_coefficient, dependability, bool(fitted.converged))


def item_reliability(measurements: pd.DataFrame, item_variance: float, primary_prompt: str) -> pd.DataFrame:
    primary = measurements[measurements["prompt_id"] == primary_prompt]
    rows: list[dict[str, Any]] = []
    for item_id, group in primary.groupby("item_id", sort=True):
        values = group["score"].to_numpy(float)
        error_variance = float(values.var(ddof=1) / len(values)) if len(values) > 1 else 0.0
        reliability = item_variance / (item_variance + error_variance) if item_variance + error_variance else 0.0
        rows.append({
            "item_id": item_id,
            "symbol": group.iloc[0].get("symbol"),
            "news_date": group.iloc[0].get("news_date"),
            "chronological_split": group.iloc[0].get("chronological_split"),
            "mean_score": float(values.mean()),
            "measurement_error_variance": error_variance,
            "reliability": min(1.0, max(0.0, reliability)),
        })
    return pd.DataFrame(rows)


def aggregate_daily_reliability(items: pd.DataFrame, item_variance: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (symbol, news_date), group in items.groupby(["symbol", "news_date"], sort=True):
        count = len(group)
        true_variance = item_variance / count
        error_variance = float(group["measurement_error_variance"].sum() / (count**2))
        reliability = true_variance / (true_variance + error_variance) if true_variance + error_variance else 0.0
        rows.append({
            "symbol": symbol,
            "news_date": news_date,
            "chronological_split": group.iloc[0]["chronological_split"],
            "item_count": count,
            "mean_score": float(group["mean_score"].mean()),
            "measurement_error_variance": error_variance,
            "reliability": min(1.0, max(0.0, reliability)),
        })
    return pd.DataFrame(rows)


def _return_risk(values: pd.Series) -> float:
    deviation = float(values.std(ddof=1))
    return float(values.mean() / deviation) if deviation else -math.inf


def compare_reliability_rules(daily: pd.DataFrame, event_returns: pd.DataFrame, cost_per_side: float = 0.001) -> pd.DataFrame:
    realised = (
        event_returns[event_returns["horizon"] == 5]
        .groupby(["symbol", "news_date"], as_index=False)["car"]
        .mean()
    )
    data = daily.merge(realised, on=["symbol", "news_date"], how="inner")
    development = data[data["chronological_split"] == "development"].copy()
    holdout = data[data["chronological_split"] == "holdout"].copy()
    if development.empty or holdout.empty:
        raise L3AnalysisError("reliability rules require non-empty development and holdout rows")
    candidates = np.arange(0.0, 0.81, 0.1)
    best_threshold = 0.0
    best_ratio = -math.inf
    for threshold in candidates:
        signal = np.where(development["mean_score"].abs() >= threshold, np.sign(development["mean_score"]), 0)
        net = signal * development["car"] - (signal != 0) * cost_per_side * 2
        ratio = _return_risk(pd.Series(net))
        if ratio > best_ratio:
            best_threshold, best_ratio = float(threshold), ratio
    development_snr = (
        development["reliability"] * development["mean_score"]
        / np.sqrt(development["measurement_error_variance"].clip(lower=1e-12))
    )
    snr_cap = float(development_snr.abs().quantile(0.95)) or 1.0
    fixed = np.where(holdout["mean_score"].abs() >= best_threshold, np.sign(holdout["mean_score"]), 0)
    shrunk_score = holdout["reliability"] * holdout["mean_score"]
    shrunk = np.where(shrunk_score.abs() >= best_threshold, np.sign(shrunk_score), 0)
    snr = np.asarray(np.clip(
        shrunk_score / np.sqrt(holdout["measurement_error_variance"].clip(lower=1e-12)) / snr_cap,
        -1,
        1,
    ))
    rows = []
    for rule, positions in (("fixed_threshold", fixed), ("reliability_shrinkage", shrunk), ("signal_to_noise", snr)):
        gross = positions * holdout["car"].to_numpy(float)
        net = gross - (np.abs(positions) > 0) * cost_per_side * 2
        for index, value in enumerate(net):
            rows.append({
                "rule": rule,
                "symbol": holdout.iloc[index]["symbol"],
                "news_date": holdout.iloc[index]["news_date"],
                "position": float(positions[index]),
                "gross_return": float(gross[index]),
                "net_return": float(value),
                "threshold": best_threshold,
                "snr_cap": snr_cap,
            })
    return pd.DataFrame(rows)


def _date_block_pvalue(rules: pd.DataFrame, challenger: str, seed: int = 42, resamples: int = 10000) -> float:
    pivot = rules.pivot_table(index="news_date", columns="rule", values="net_return", aggfunc="mean").dropna()
    differences = pivot[challenger] - pivot["fixed_threshold"]
    if differences.empty:
        return float("nan")
    generator = np.random.default_rng(seed)
    values = differences.to_numpy()
    means = [float((values * generator.choice([-1, 1], len(values))).mean()) for _ in range(resamples)]
    observed = float(differences.mean())
    return float((sum(abs(value) >= abs(observed) for value in means) + 1) / (resamples + 1))


def h3_hypotheses(
    measurements: pd.DataFrame,
    benchmark_measurements: pd.DataFrame | None,
    rules: pd.DataFrame,
    components: GStudyResult,
) -> pd.DataFrame:
    model_groups = [group["score"].to_numpy(float) for _, group in measurements.groupby("model_id")]
    prompt_groups = [group["score"].to_numpy(float) for _, group in measurements.groupby("prompt_id")]
    model_p = float(stats.f_oneway(*model_groups).pvalue)
    prompt_p = float(stats.f_oneway(*prompt_groups).pvalue)
    h3a_p = float(stats.combine_pvalues([model_p, prompt_p], method="fisher").pvalue)
    h3b_stat, h3b_p = float("nan"), float("nan")
    if benchmark_measurements is not None:
        item_variance = benchmark_measurements.groupby("item_id")["score"].var().reset_index(name="variance")
        tiers = benchmark_measurements.groupby("item_id")["pb_agreement_tier"].first().reset_index()
        tier_data = item_variance.merge(tiers, on="item_id").dropna()
        if not tier_data.empty:
            h3b_stat, h3b_p = stats.spearmanr(tier_data["pb_agreement_tier"].astype(float), tier_data["variance"])
    challenger = (
        rules.groupby("rule")["net_return"].mean().drop("fixed_threshold").idxmax()
    )
    h3c_p = _date_block_pvalue(rules, challenger)
    rows = [
        {"hypothesis": "H3a", "statistic": 1 - components.variance_shares.get("item", 0), "p_value": h3a_p},
        {"hypothesis": "H3b", "statistic": float(h3b_stat), "p_value": float(h3b_p)},
        {"hypothesis": "H3c", "statistic": float(rules.groupby("rule")["net_return"].mean()[challenger]), "p_value": h3c_p},
    ]
    result = pd.DataFrame(rows)
    result["q_bh"] = benjamini_hochberg(result["p_value"].tolist())
    return result


def analyze_l3(
    scores_path: str | Path,
    event_returns_path: str | Path,
    output_dir: str | Path,
    benchmark_scores_path: str | Path | None = None,
    l2_hypotheses_path: str | Path | None = None,
) -> L3Result:
    destination = Path(output_dir)
    if destination.exists():
        raise L3AnalysisError(f"refusing to overwrite L3 output: {destination}")
    measurements = load_measurements(scores_path)
    facet = measurements[measurements["prompt_id"].str.startswith("target_company_soft_label_")]
    components = fit_crossed_gstudy(facet)
    items = item_reliability(measurements, components.components.get("item", 0), "target_company_soft_label_base")
    daily = aggregate_daily_reliability(items, components.components.get("item", 0))
    event_returns = pd.read_csv(event_returns_path)
    rules = compare_reliability_rules(daily, event_returns)
    benchmark = load_measurements(benchmark_scores_path) if benchmark_scores_path else None
    hypotheses = h3_hypotheses(facet, benchmark, rules, components)
    destination.mkdir(parents=True)
    components_path = destination / "variance_components.json"
    atomic_write_json(components_path, {
        "components": components.components,
        "variance_shares": components.variance_shares,
        "g_coefficient": components.g_coefficient,
        "dependability_coefficient": components.dependability_coefficient,
        "converged": components.converged,
    })
    item_path = destination / "item_reliability.csv"
    daily_path = destination / "daily_reliability.csv"
    rules_path = destination / "rule_returns.csv"
    hypothesis_path = destination / "hypotheses.csv"
    combined_path: Path | None = None
    items.to_csv(item_path, index=False, lineterminator="\n")
    daily.to_csv(daily_path, index=False, lineterminator="\n")
    rules.to_csv(rules_path, index=False, lineterminator="\n")
    hypotheses.to_csv(hypothesis_path, index=False, lineterminator="\n")
    if l2_hypotheses_path is not None:
        l2 = pd.read_csv(l2_hypotheses_path)
        combined = pd.concat([l2.drop(columns=["q_bh"], errors="ignore"), hypotheses.drop(columns=["q_bh"])], ignore_index=True)
        combined["q_bh"] = benjamini_hochberg(combined["p_value"].tolist())
        combined_path = destination / "combined_h2_h3_hypotheses.csv"
        combined.to_csv(combined_path, index=False, lineterminator="\n")
    atomic_write_json(destination / "manifest.json", {
        "schema_version": 1,
        "inputs": {str(scores_path): sha256_file(scores_path), str(event_returns_path): sha256_file(event_returns_path)},
        "transaction_cost_bps_per_side": 10,
        "files": {
            path.name: sha256_file(path)
            for path in (components_path, item_path, daily_path, rules_path, hypothesis_path, combined_path)
            if path is not None
        },
    })
    return L3Result(destination, components_path, item_path, daily_path, rules_path, hypothesis_path, combined_path)
