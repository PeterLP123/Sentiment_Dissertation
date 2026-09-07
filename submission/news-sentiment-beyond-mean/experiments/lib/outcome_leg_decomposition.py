"""Run the frozen Notebook 86 FNSPID outcome-leg decomposition.

The runner reads the hash-bound aggregate firm-day panel and price archive,
writes licence-safe aggregate outputs, and never executes Notebook 75 or reads
the story-scoring checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from experiments.lib.aggregators import benjamini_hochberg, benjamini_hochberg_q_values
from experiments.lib.conditional_aggregation import (
    conditional_negative_share_test,
    hac_mean_coefficient,
    hac_regime_mean_contrast,
)
from experiments.lib.panel import (
    DEFAULT_PRICE_ZIP,
    FROZEN_DEV_END,
    FROZEN_EVAL_START,
    load_adjusted_open_prices,
    session_open_close_return_legs,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC_NAME = "fnspid_outcome_leg_decomposition_v1_20260816.json"
NB75 = ROOT / "experiments/notebooks/75_fnspid_evaluation_beyond_mean_replication.ipynb"
SOURCE_ARCHIVE = Path("/Users/peterprendergast/Documents/Sentiment_Dissertation")
SOURCE_PRICES = Path(
    "/Users/peterprendergast/Documents/Sentiment_Dissertation_data/FNSPID/"
    "bf9189c41527198897d1af3e17b1a0095279fc45/raw/Stock_price/full_history.zip"
)
AGGREGATORS_HASH = "1ca1c38109bc06af232119adeff17de66a567174d457038bc923a930a2da61f1"
PRICE_HASH = "03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91"
HAC_LAGS = 5
MIN_NAMES = 10
NEW_LEGS = [
    "assigned_session_intraday",
    "post_close_overnight",
    "pre_open_overnight",
]
OUTCOMES = {
    "assigned_open_to_next_open": "ar_assigned_open_to_next_open",
    "assigned_session_intraday": "ar_assigned_session_intraday",
    "post_close_overnight": "ar_post_close_overnight",
    "pre_open_overnight": "ar_pre_open_overnight",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("none of these paths exist: " + ", ".join(str(p) for p in candidates))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def resolve_inputs(spec: dict[str, Any]) -> dict[str, Path]:
    aggregators = first_existing(
        [
            ROOT / "experiments/generated/03_aggregation/firm_day_aggregators.parquet",
            SOURCE_ARCHIVE / "final_experiments/outputs/03_aggregation/firm_day_aggregators.parquet",
        ]
    )
    prices = first_existing(
        [
            DEFAULT_PRICE_ZIP,
            Path(spec["inputs"]["fnspid_price_archive"]["path"]),
            SOURCE_PRICES,
        ]
    )
    return {
        "aggregators": aggregators,
        "prices": prices,
        "notebook_75": NB75,
        "notebook_75_results": ROOT / spec["inputs"]["notebook_75_results"]["directory"],
        "notebook_24_results": ROOT / spec["inputs"]["notebook_24_results"]["directory"],
        "notebook_79_results": ROOT / "experiments/results/79_fnspid_development_reversal_confound",
        "notebook_74_results": ROOT
        / "experiments/results/74_conditional_and_pressure_robustness_diagnostics",
    }


def _attach_leg_returns(panel: pd.DataFrame, leg_returns: pd.DataFrame) -> pd.DataFrame:
    keep = ["symbol", "session_date", *OUTCOMES.values()]
    out = panel.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.normalize()
    return out.merge(
        leg_returns.loc[leg_returns["symbol"].ne("SPY"), keep],
        on=["symbol", "session_date"],
        how="left",
        validate="1:1",
    )


def _coefficient_outputs(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    daily_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for regime in ("development", "evaluation"):
        regime_frame = panel.loc[panel["split"].eq(regime)].copy()
        for outcome_name, outcome_col in OUTCOMES.items():
            summary, daily, audit = conditional_negative_share_test(
                regime_frame,
                date_col="session_date",
                outcome_col=outcome_col,
                mean_col="mean_continuous",
                negative_share_col="negative_share",
                count_col="n",
                min_names=MIN_NAMES,
                hac_lags=HAC_LAGS,
            )
            summary_rows.append(
                {
                    "regime": regime,
                    "outcome_leg": outcome_name,
                    "role": (
                        "published_full_window_audit"
                        if outcome_name == "assigned_open_to_next_open"
                        else "post_review_outcome_leg"
                    ),
                    **summary,
                    **audit,
                }
            )
            daily_lookup[(regime, outcome_name)] = daily
            daily_rows.append(
                daily[
                    [
                        "session_date",
                        "n_names",
                        "beta_mean_continuous",
                        "beta_negative_share",
                        "beta_log_story_count",
                    ]
                ].assign(regime=regime, outcome_leg=outcome_name)
            )

    summary_table = pd.DataFrame(summary_rows)
    summary_table["q_bh_three_leg"] = pd.Series(pd.NA, index=summary_table.index, dtype="Float64")
    summary_table["bh_reject_q05"] = False
    summary_table["multiplicity_family"] = "published_baseline_excluded"
    for regime in ("development", "evaluation"):
        mask = summary_table["regime"].eq(regime) & summary_table["outcome_leg"].isin(NEW_LEGS)
        p_values = summary_table.loc[mask, "p_two_sided"].astype(float).tolist()
        summary_table.loc[mask, "q_bh_three_leg"] = benjamini_hochberg_q_values(p_values)
        summary_table.loc[mask, "bh_reject_q05"] = benjamini_hochberg(p_values, q=0.05)
        summary_table.loc[mask, "multiplicity_family"] = f"{regime}_three_new_legs"

    for regime in ("development", "evaluation"):
        baseline = float(
            summary_table.loc[
                summary_table["regime"].eq(regime)
                & summary_table["outcome_leg"].eq("assigned_open_to_next_open"),
                "estimate",
            ].iloc[0]
        )
        mask = summary_table["regime"].eq(regime)
        summary_table.loc[mask, "absolute_point_estimate_ratio_to_full"] = (
            summary_table.loc[mask, "estimate"].abs() / abs(baseline)
            if baseline != 0
            else float("nan")
        )

    contrast_rows: list[dict[str, Any]] = []
    for outcome_name in OUTCOMES:
        combined = pd.concat(
            [
                daily_lookup[(regime, outcome_name)].assign(regime=regime)
                for regime in ("development", "evaluation")
            ],
            ignore_index=True,
        )
        contrast_rows.append(
            {
                "outcome_leg": outcome_name,
                "role": (
                    "published_full_window_contrast_audit"
                    if outcome_name == "assigned_open_to_next_open"
                    else "post_review_leg_contrast"
                ),
                **hac_regime_mean_contrast(
                    combined,
                    coefficient_col="beta_negative_share",
                    date_col="session_date",
                    regime_col="regime",
                    reference="development",
                    comparison="evaluation",
                    hac_lags=HAC_LAGS,
                ),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["q_bh_three_leg"] = pd.Series(pd.NA, index=contrasts.index, dtype="Float64")
    contrasts["bh_reject_q05"] = False
    contrasts["multiplicity_family"] = "published_baseline_excluded"
    new_mask = contrasts["outcome_leg"].isin(NEW_LEGS)
    contrast_p = contrasts.loc[new_mask, "p_two_sided"].astype(float).tolist()
    contrasts.loc[new_mask, "q_bh_three_leg"] = benjamini_hochberg_q_values(contrast_p)
    contrasts.loc[new_mask, "bh_reject_q05"] = benjamini_hochberg(contrast_p, q=0.05)
    contrasts.loc[new_mask, "multiplicity_family"] = "three_new_leg_era_contrasts"

    development = summary_table.loc[summary_table["regime"].eq("development")].set_index(
        "outcome_leg"
    )
    intraday_negative = bool(
        development.loc["assigned_session_intraday", "estimate"] < 0
        and development.loc["assigned_session_intraday", "ci_high"] < 0
    )
    post_close_negative = bool(
        development.loc["post_close_overnight", "estimate"] < 0
        and development.loc["post_close_overnight", "ci_high"] < 0
    )
    pre_open_negative = bool(
        development.loc["pre_open_overnight", "estimate"] < 0
        and development.loc["pre_open_overnight", "ci_high"] < 0
    )
    if post_close_negative and not intraday_negative:
        decision = "post_close_only_or_dominant"
    elif intraday_negative and not post_close_negative:
        decision = "intraday_only_or_dominant"
    elif intraday_negative and post_close_negative:
        decision = "both_detected"
    else:
        decision = "neither_detected"
    gate = {
        "decision": decision,
        "assigned_session_intraday_interval_negative": intraday_negative,
        "post_close_overnight_interval_negative": post_close_negative,
        "pre_open_overnight_interval_negative": pre_open_negative,
        "assigned_session_intraday_bh_reject": bool(
            development.loc["assigned_session_intraday", "bh_reject_q05"]
        ),
        "post_close_overnight_bh_reject": bool(
            development.loc["post_close_overnight", "bh_reject_q05"]
        ),
        "pre_open_overnight_bh_reject": bool(
            development.loc["pre_open_overnight", "bh_reject_q05"]
        ),
    }
    daily_table = pd.concat(daily_rows, ignore_index=True)[
        [
            "regime",
            "outcome_leg",
            "session_date",
            "n_names",
            "beta_mean_continuous",
            "beta_negative_share",
            "beta_log_story_count",
        ]
    ]
    return summary_table, contrasts, daily_table, gate


def _story_count_strata(panel: pd.DataFrame, notebook_74_results: Path) -> pd.DataFrame:
    development = panel.loc[panel["split"].eq("development")].copy()
    rows: list[dict[str, Any]] = []
    for label, mask, count_col in (
        ("n = 1", development["n"].eq(1), None),
        ("n >= 2", development["n"].ge(2), "n"),
        ("n >= 3", development["n"].ge(3), "n"),
    ):
        frame = development.loc[mask]
        summary, _, audit = conditional_negative_share_test(
            frame,
            date_col="session_date",
            outcome_col="ar_assigned_open_to_next_open",
            count_col=count_col,
            min_names=MIN_NAMES,
            hac_lags=HAC_LAGS,
        )
        rows.append(
            {
                "stratum": label,
                "firm_days": int(len(frame)),
                "story_count_control_included": count_col is not None,
                **summary,
                **audit,
                "comparability_note": (
                    "descriptive only; ranks are recomputed within each stratum and the n=1 model "
                    "omits the constant count control"
                ),
            }
        )
    table = pd.DataFrame(rows)
    committed = pd.read_csv(notebook_74_results / "story_count_strata.csv")
    for label in ("n >= 2", "n >= 3"):
        reproduced = float(table.loc[table["stratum"].eq(label), "estimate"].iloc[0])
        expected = float(committed.loc[committed["stratum"].eq(label), "estimate"].iloc[0])
        if not np.isclose(reproduced, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Notebook 74 {label} coefficient did not reproduce")
    return table


def _identity_audit(
    panel: pd.DataFrame,
    leg_returns: pd.DataFrame,
    *,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, frame in (
        ("stock", leg_returns.loc[leg_returns["symbol"].ne("SPY")]),
        ("market", leg_returns.loc[leg_returns["symbol"].eq("SPY")]),
    ):
        complete = frame.dropna(
            subset=[
                "ret_assigned_session_intraday",
                "ret_post_close_overnight",
                "ret_assigned_open_to_next_open",
            ]
        )
        error = complete["ret_assigned_open_to_next_open"] - (
            (1.0 + complete["ret_assigned_session_intraday"])
            * (1.0 + complete["ret_post_close_overnight"])
            - 1.0
        )
        max_error = float(error.abs().max()) if len(error) else float("nan")
        rows.append(
            {
                "check": f"{label}_gross_return_identity",
                "n_complete": int(len(error)),
                "max_absolute_error": max_error,
                "tolerance": tolerance,
                "passes": bool(math.isfinite(max_error) and max_error <= tolerance),
            }
        )

    comparison = panel[["ar_open_h1", "ar_assigned_open_to_next_open"]].dropna()
    baseline_error = comparison["ar_open_h1"] - comparison["ar_assigned_open_to_next_open"]
    max_baseline_error = (
        float(baseline_error.abs().max()) if len(baseline_error) else float("nan")
    )
    rows.append(
        {
            "check": "recomputed_abnormal_full_window_matches_committed_panel",
            "n_complete": int(len(baseline_error)),
            "max_absolute_error": max_baseline_error,
            "tolerance": tolerance,
            "passes": bool(
                math.isfinite(max_baseline_error) and max_baseline_error <= tolerance
            ),
        }
    )
    audit = pd.DataFrame(rows)
    if not audit["passes"].all():
        raise RuntimeError("outcome-leg identity audit failed")
    return audit


def _price_complete_rank_translation(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    notebook_79_results: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    development = panel.loc[panel["split"].eq("development")].copy()
    prices = prices.sort_values(["symbol", "session_date"], kind="mergesort").copy()
    price_groups = prices.groupby("symbol", sort=False)
    prices["lagged_return_1"] = price_groups["adjusted_open"].pct_change(fill_method=None)
    prices["lagged_return_5"] = prices["adjusted_open"].div(
        price_groups["adjusted_open"].shift(5)
    ).sub(1.0)
    prices["sigma20"] = prices.groupby("symbol", sort=False)["lagged_return_1"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=1)
    )
    price_features = prices.loc[
        prices["symbol"].ne("SPY"),
        ["symbol", "session_date", "lagged_return_1", "lagged_return_5", "sigma20"],
    ]
    analysis = development.merge(
        price_features,
        on=["symbol", "session_date"],
        how="left",
        validate="1:1",
    )
    required = [
        "ar_open_h1",
        "mean_continuous",
        "negative_share",
        "n",
        "lagged_return_1",
        "lagged_return_5",
        "sigma20",
    ]
    complete = analysis.loc[analysis[required].notna().all(axis=1)].copy()

    committed = pd.read_csv(notebook_79_results / "conditional_price_control_summary.csv")
    full_model = committed.loc[committed["specification"].eq("full_price_path_controls")].iloc[0]
    expected_rows = int(full_model["input_rows"])
    if len(complete) != expected_rows:
        raise RuntimeError(
            f"Notebook 79 price-complete rows did not reproduce: {len(complete)} != {expected_rows}"
        )

    complete["rank_negative_share"] = (
        complete.groupby("session_date", sort=False)["negative_share"].rank(
            method="average", pct=True
        )
        - 0.5
    )
    pooled_low = float(complete["rank_negative_share"].quantile(0.10))
    pooled_high = float(complete["rank_negative_share"].quantile(0.90))
    pooled_spread = pooled_high - pooled_low
    daily_quantiles = complete.groupby("session_date")["rank_negative_share"].quantile([0.1, 0.9])
    daily_quantile_table = daily_quantiles.unstack()
    daily_spreads = daily_quantile_table[0.9] - daily_quantile_table[0.1]
    coefficient = float(full_model["estimate"])

    rows = [
        {
            "translation": "hypothetical_fixed_0_8",
            "rank_change": 0.8,
            "coefficient": coefficient,
            "fitted_rank_change": 0.8 * coefficient,
            "fitted_percentile_points": 100.0 * 0.8 * coefficient,
            "valid_for_observed_tied_regressor": False,
            "note": "superseded hypothetical translation; exceeds the observed pooled 10th-to-90th spread",
        },
        {
            "translation": "pooled_observed_p90_minus_p10",
            "rank_change": pooled_spread,
            "coefficient": coefficient,
            "fitted_rank_change": pooled_spread * coefficient,
            "fitted_percentile_points": 100.0 * pooled_spread * coefficient,
            "valid_for_observed_tied_regressor": True,
            "note": "primary corrected translation on exact Notebook 79 price-complete rows",
        },
        {
            "translation": "mean_session_specific_p90_minus_p10",
            "rank_change": float(daily_spreads.mean()),
            "coefficient": coefficient,
            "fitted_rank_change": float(daily_spreads.mean()) * coefficient,
            "fitted_percentile_points": 100.0 * float(daily_spreads.mean()) * coefficient,
            "valid_for_observed_tied_regressor": True,
            "note": "sensitivity averaging each session's observed rank spread",
        },
    ]
    audit = {
        "price_complete_rows": int(len(complete)),
        "price_complete_sessions": int(complete["session_date"].nunique()),
        "pooled_rank_p10": pooled_low,
        "pooled_rank_p90": pooled_high,
    }
    return pd.DataFrame(rows), audit


def _power_reconciliation(notebook_75_results: Path) -> pd.DataFrame:
    prospective = pd.read_csv(notebook_75_results / "prospective_power_context.csv").iloc[0]
    evaluation_daily = pd.read_csv(
        notebook_75_results / "evaluation_daily_coefficients_all_firm_days.csv"
    )
    realised_inference = hac_mean_coefficient(
        evaluation_daily,
        coefficient_col="beta_negative_share",
        hac_lags=HAC_LAGS,
    )
    alpha = float(prospective["conservative_two_sided_alpha"])
    effect = float(prospective["development_estimate"])
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    target = float(norm.ppf(0.80))

    def power_for(se: float) -> float:
        noncentrality = abs(effect) / se
        return float(norm.sf(critical - noncentrality) + norm.cdf(-critical - noncentrality))

    projected_se = float(prospective["scaled_evaluation_se"])
    realised_se = float(realised_inference["se"])
    rows = [
        {
            "role": "prospective_before_evaluation",
            "method": "development HAC SE scaled by square-root session ratio",
            "two_sided_alpha": alpha,
            "effect": effect,
            "standard_error": projected_se,
            "effect_z": abs(effect) / projected_se,
            "power": power_for(projected_se),
            "mde80": (critical + target) * projected_se,
            "se_ratio_to_projection": 1.0,
        },
        {
            "role": "realised_precision_reconciliation",
            "method": "observed evaluation HAC(5) standard error; post-outcome precision diagnostic",
            "two_sided_alpha": alpha,
            "effect": effect,
            "standard_error": realised_se,
            "effect_z": abs(effect) / realised_se,
            "power": power_for(realised_se),
            "mde80": (critical + target) * realised_se,
            "se_ratio_to_projection": realised_se / projected_se,
        },
    ]
    table = pd.DataFrame(rows)
    if not np.isclose(
        float(table.loc[0, "power"]),
        float(prospective["approximate_power_for_stable_development_effect"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("prospective power calculation did not reproduce")
    return table


def _drawdown_reconciliation(notebook_24_results: Path) -> pd.DataFrame:
    overall = pd.read_csv(notebook_24_results / "results.csv")
    temporal = pd.read_csv(notebook_24_results / "temporal_results.csv")
    rows: list[dict[str, Any]] = []
    for period, frame in [("2020_2023", overall), *list(temporal.groupby("period", sort=False))]:
        control = frame.loc[frame["arm"].eq("control_har_target")].iloc[0]
        overlay = frame.loc[frame["arm"].eq("har_sentiment")].iloc[0]
        control_drawdown = float(control["max_drawdown"])
        overlay_drawdown = float(overlay["max_drawdown"])
        rows.append(
            {
                "period": str(period),
                "control_arm": "control_har_target",
                "overlay_arm": "har_sentiment",
                "control_max_drawdown": control_drawdown,
                "overlay_max_drawdown": overlay_drawdown,
                "drawdown_relative_improvement": (
                    (abs(control_drawdown) - abs(overlay_drawdown)) / abs(control_drawdown)
                ),
                "overlay_has_smaller_drawdown": abs(overlay_drawdown) < abs(control_drawdown),
            }
        )
    return pd.DataFrame(rows)


def _coverage_audit(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    missing_symbols: list[str],
    rank_audit: dict[str, int | float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"item": "panel_rows", "value": int(len(panel))},
        {"item": "panel_symbols", "value": int(panel["symbol"].nunique())},
        {"item": "price_rows", "value": int(len(prices))},
        {"item": "missing_price_symbols", "value": int(len(missing_symbols))},
    ]
    for regime in ("development", "evaluation"):
        frame = panel.loc[panel["split"].eq(regime)]
        rows.extend(
            [
                {"item": f"{regime}_rows", "value": int(len(frame))},
                {"item": f"{regime}_sessions", "value": int(frame["session_date"].nunique())},
                {"item": f"{regime}_singleton_share", "value": float(frame["n"].eq(1).mean())},
            ]
        )
        for name, outcome in OUTCOMES.items():
            rows.append(
                {
                    "item": f"{regime}_{name}_complete_rows",
                    "value": int(frame[outcome].notna().sum()),
                }
            )
    rows.extend({"item": name, "value": value} for name, value in rank_audit.items())
    return pd.DataFrame(rows)


def run(*, output_dir: Path | None = None) -> dict[str, Path]:
    spec_path = ROOT / "experiments/specs" / SPEC_NAME
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["status"] != "frozen_before_outcome_leg_results_after_external_review":
        raise ValueError("unexpected frozen specification status")
    paths = resolve_inputs(spec)
    output = output_dir or (ROOT / "experiments/generated/86_fnspid_outcome_leg_decomposition")
    output.mkdir(parents=True, exist_ok=True)

    notebook_75_hash_before = sha256(paths["notebook_75"])
    aggregators_hash = sha256(paths["aggregators"])
    prices_hash = sha256(paths["prices"])
    if aggregators_hash != AGGREGATORS_HASH:
        raise RuntimeError(f"aggregators parquet hash mismatch: {aggregators_hash}")
    if prices_hash != PRICE_HASH:
        raise RuntimeError(f"price zip hash mismatch: {prices_hash}")

    panel = pd.read_parquet(
        paths["aggregators"],
        columns=[
            "symbol",
            "session_date",
            "split",
            "ar_open_h1",
            "n",
            "mean_continuous",
            "negative_share",
        ],
    )
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.normalize()
    panel = panel.loc[
        (
            panel["split"].eq("development")
            & panel["session_date"].le(pd.Timestamp(FROZEN_DEV_END))
        )
        | (
            panel["split"].eq("evaluation")
            & panel["session_date"].ge(pd.Timestamp(FROZEN_EVAL_START))
        )
    ].copy()
    prices, missing_symbols = load_adjusted_open_prices(
        paths["prices"],
        set(panel["symbol"].unique()) | {"SPY"},
        start="2010-01-01",
        end="2023-12-31",
    )
    if missing_symbols:
        raise RuntimeError(f"price archive is missing {len(missing_symbols)} panel symbols")
    leg_returns = session_open_close_return_legs(prices)
    analysis = _attach_leg_returns(panel, leg_returns)
    identity = _identity_audit(
        analysis,
        leg_returns,
        tolerance=float(spec["identity_checks"]["absolute_tolerance"]),
    )

    coefficients, contrasts, daily, interpretation_gate = _coefficient_outputs(analysis)
    strata = _story_count_strata(analysis, paths["notebook_74_results"])
    rank_translation, rank_audit = _price_complete_rank_translation(
        analysis, prices, paths["notebook_79_results"]
    )
    power = _power_reconciliation(paths["notebook_75_results"])
    drawdown = _drawdown_reconciliation(paths["notebook_24_results"])
    coverage = _coverage_audit(analysis, prices, missing_symbols, rank_audit)

    published_development = pd.read_csv(
        paths["notebook_75_results"] / "development_daily_coefficients_all_firm_days.csv"
    )
    published_evaluation = pd.read_csv(
        paths["notebook_75_results"] / "evaluation_daily_coefficients_all_firm_days.csv"
    )
    for regime, published in (
        ("development", published_development),
        ("evaluation", published_evaluation),
    ):
        reproduced = coefficients.loc[
            coefficients["regime"].eq(regime)
            & coefficients["outcome_leg"].eq("assigned_open_to_next_open")
        ].iloc[0]
        expected = hac_mean_coefficient(
            published,
            coefficient_col="beta_negative_share",
            hac_lags=HAC_LAGS,
        )
        if not np.isclose(
            float(reproduced["estimate"]),
            float(expected["estimate"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"{regime} full-window coefficient did not reproduce")

    tables = {
        "coefficient_summary": coefficients,
        "era_contrasts": contrasts,
        "daily_coefficients": daily,
        "story_count_strata": strata,
        "timing_identity_audit": identity,
        "rank_effect_translation": rank_translation,
        "power_reconciliation": power,
        "drawdown_reconciliation": drawdown,
        "coverage_audit": coverage,
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path

    notebook_75_hash_after = sha256(paths["notebook_75"])
    if notebook_75_hash_after != notebook_75_hash_before:
        raise RuntimeError("Notebook 75 changed during Notebook 86 execution")
    executed_at = datetime.now(UTC).isoformat()
    output_records = [
        {
            "name": name,
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in written.items()
    ]
    manifest = {
        "schema_version": 1,
        "notebook": "86_fnspid_outcome_leg_decomposition",
        "status": "completed_post_review_diagnostic_not_confirmation",
        "executed_at_utc": executed_at,
        "git_commit_at_execution": git_commit(),
        "spec": str(spec_path.relative_to(ROOT)),
        "spec_sha256": sha256(spec_path),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "inputs": {
            "aggregators": {"path": str(paths["aggregators"]), "sha256": aggregators_hash},
            "prices": {"path": str(paths["prices"]), "sha256": prices_hash},
            "notebook_75": {
                "path": str(paths["notebook_75"].relative_to(ROOT)),
                "sha256_before": notebook_75_hash_before,
                "sha256_after": notebook_75_hash_after,
                "executed": False,
            },
        },
        "coverage": coverage.to_dict(orient="records"),
        "interpretation_gate": interpretation_gate,
        "claim_boundaries": spec["claim_boundaries"],
        "outputs": output_records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="write outputs here instead of the default generated directory",
    )
    args = parser.parse_args(argv)
    written = run(output_dir=args.output_dir)
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
