"""Run the frozen Notebook 87 FNSPID external-review robustness pack."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

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
from experiments.lib.volatility_target import paired_circular_block_mean

ROOT = Path(__file__).resolve().parents[2]
SPEC_NAME = "fnspid_external_review_robustness_pack_v1_20260816.json"
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
BASE_REGRESSORS = ["mean_continuous", "negative_share", "log1p_n"]


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
    return {
        "aggregators": first_existing(
            [
                ROOT / "experiments/generated/03_aggregation/firm_day_aggregators.parquet",
                SOURCE_ARCHIVE
                / "final_experiments/outputs/03_aggregation/firm_day_aggregators.parquet",
            ]
        ),
        "prices": first_existing(
            [
                DEFAULT_PRICE_ZIP,
                Path(spec["inputs"]["fnspid_price_archive"]["path"]),
                SOURCE_PRICES,
            ]
        ),
        "development_daily": ROOT / spec["inputs"]["development_daily_coefficients"]["path"],
        "evaluation_daily": ROOT / spec["inputs"]["evaluation_daily_coefficients"]["path"],
        "risk_control": Path(spec["inputs"]["fnspid_har_control_daily"]["path"]),
        "risk_overlay": Path(spec["inputs"]["fnspid_har_overlay_daily"]["path"]),
        "notebook_86_coefficients": ROOT
        / spec["inputs"]["notebook_86_coefficients"]["path"],
        "notebook_86_strata": ROOT / spec["inputs"]["notebook_86_strata"]["path"],
        "notebook_75": NB75,
    }


def trailing_open_beta_features(leg_returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate trailing open-return beta on the SPY session calendar."""

    required = {
        "ar_previous_open_to_assigned_open",
        "symbol",
        "session_date",
        "ret_previous_open_to_assigned_open",
        "spy_ret_previous_open_to_assigned_open",
    }
    if missing := required - set(leg_returns.columns):
        raise ValueError(f"leg returns missing columns: {sorted(missing)}")
    use = leg_returns.loc[:, sorted(required)].copy()
    use = use.sort_values(["symbol", "session_date"], kind="mergesort")
    pieces: list[pd.DataFrame] = []
    for _, group in use.groupby("symbol", sort=False):
        group = group.copy()
        stock = group["ret_previous_open_to_assigned_open"]
        market = group["spy_ret_previous_open_to_assigned_open"]
        covariance = stock.rolling(252, min_periods=126).cov(market)
        variance = market.rolling(252, min_periods=126).var(ddof=1)
        group["trailing_open_beta"] = covariance / variance.replace(0.0, np.nan)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def daily_rank_coefficients(
    frame: pd.DataFrame,
    *,
    outcome_col: str,
    regressors: list[str],
    min_names: int = MIN_NAMES,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Estimate daily centred-rank regressions for a declared regressor list."""

    required = ["session_date", outcome_col, *regressors]
    if missing := set(required) - set(frame.columns):
        raise ValueError(f"frame missing regression columns: {sorted(missing)}")
    use = frame.loc[:, required].copy()
    use["session_date"] = pd.to_datetime(use["session_date"]).dt.normalize()
    audit = {
        "dates_total": int(use["session_date"].nunique()),
        "dates_used": 0,
        "dates_below_min_names": 0,
        "dates_constant_variable": 0,
        "dates_rank_deficient": 0,
    }
    rows: list[dict[str, Any]] = []
    complete_columns = [outcome_col, *regressors]
    for session, day in use.groupby("session_date", sort=True):
        day = day.dropna(subset=complete_columns)
        if len(day) < min_names:
            audit["dates_below_min_names"] += 1
            continue
        if any(day[column].nunique() < 2 for column in complete_columns):
            audit["dates_constant_variable"] += 1
            continue
        y = day[outcome_col].rank(method="average", pct=True).sub(0.5).to_numpy(dtype=float)
        ranked = np.column_stack(
            [
                day[column].rank(method="average", pct=True).sub(0.5).to_numpy(dtype=float)
                for column in regressors
            ]
        )
        design = np.column_stack([np.ones(len(day), dtype=float), ranked])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            audit["dates_rank_deficient"] += 1
            continue
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        row: dict[str, Any] = {
            "session_date": pd.Timestamp(session),
            "n_names": int(len(day)),
            "intercept": float(beta[0]),
        }
        row.update(
            {
                f"beta_{name}": float(value)
                for name, value in zip(regressors, beta[1:], strict=True)
            }
        )
        rows.append(row)
    daily = pd.DataFrame(rows).sort_values("session_date", kind="mergesort").reset_index(drop=True)
    audit["dates_used"] = int(len(daily))
    excluded = sum(
        audit[key]
        for key in ("dates_below_min_names", "dates_constant_variable", "dates_rank_deficient")
    )
    if excluded + audit["dates_used"] != audit["dates_total"]:
        raise RuntimeError("daily regression audit does not reconcile")
    return daily, audit


def episode_ids(active: pd.Series) -> pd.Series:
    """Number contiguous active runs from one, leaving inactive rows at zero."""

    flags = active.fillna(False).astype(bool).to_numpy()
    starts = flags & ~np.r_[False, flags[:-1]]
    identifiers = np.cumsum(starts)
    identifiers[~flags] = 0
    return pd.Series(identifiers, index=active.index, dtype=int)


def recover_class_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover hard-label class counts from aggregate sufficient statistics."""

    required = {"n", "negative_share", "mean_hard_label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"panel missing class-count columns: {sorted(missing)}")
    n = pd.to_numeric(frame["n"], errors="raise").astype(int)
    negative_raw = n * pd.to_numeric(frame["negative_share"], errors="raise")
    label_sum_raw = n * pd.to_numeric(frame["mean_hard_label"], errors="raise")
    negative = negative_raw.round().astype(int)
    label_sum = label_sum_raw.round().astype(int)
    if (negative_raw - negative).abs().max() > 1e-8:
        raise RuntimeError("negative counts do not reconcile to integers")
    if (label_sum_raw - label_sum).abs().max() > 1e-8:
        raise RuntimeError("hard-label sums do not reconcile to integers")
    positive = label_sum + negative
    neutral = n - negative - positive
    counts = pd.DataFrame({"negative": negative, "neutral": neutral, "positive": positive})
    if counts.lt(0).any().any() or not counts.sum(axis=1).eq(n).all():
        raise RuntimeError("recovered hard-label class counts are invalid")
    return counts


def _prior_open_probe(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = panel.loc[panel["split"].eq("development")]
    summary, daily, audit = conditional_negative_share_test(
        development,
        date_col="session_date",
        outcome_col="ar_previous_open_to_assigned_open",
        mean_col="mean_continuous",
        negative_share_col="negative_share",
        count_col="n",
        min_names=MIN_NAMES,
        hac_lags=HAC_LAGS,
    )
    table = pd.DataFrame(
        [
            {
                "test": "previous_open_to_assigned_open",
                "role": "single_post_review_timing_probe",
                **summary,
                **audit,
                "q_single_test": summary["p_two_sided"],
            }
        ]
    )
    return table, daily


def _beta_robustness(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    specifications = [
        (
            "unit_beta_same_sample",
            "ar_open_h1",
            BASE_REGRESSORS,
            "same_sample_audit_excluded",
        ),
        (
            "trailing_beta_adjusted",
            "ar_trailing_beta_open_h1",
            BASE_REGRESSORS,
            "post_review_beta_family",
        ),
        (
            "trailing_beta_adjusted_plus_beta_control",
            "ar_trailing_beta_open_h1",
            [*BASE_REGRESSORS, "trailing_open_beta"],
            "post_review_beta_family",
        ),
    ]
    summary_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    daily_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    beta_rows = 0
    for regime in ("development", "evaluation"):
        regime_frame = panel.loc[panel["split"].eq(regime)].copy()
        regime_frame = regime_frame.dropna(
            subset=[
                "ar_open_h1",
                "ar_trailing_beta_open_h1",
                "trailing_open_beta",
                *BASE_REGRESSORS,
            ]
        )
        beta_rows += len(regime_frame)
        for name, outcome, regressors, family in specifications:
            daily, audit = daily_rank_coefficients(
                regime_frame,
                outcome_col=outcome,
                regressors=regressors,
            )
            inference = hac_mean_coefficient(
                daily,
                coefficient_col="beta_negative_share",
                hac_lags=HAC_LAGS,
            )
            summary_rows.append(
                {
                    "regime": regime,
                    "specification": name,
                    "outcome": outcome,
                    "regressors": ", ".join(regressors),
                    "input_rows": int(len(regime_frame)),
                    "multiplicity_family": family,
                    **inference,
                    **audit,
                }
            )
            daily_lookup[(regime, name)] = daily
            daily_rows.append(
                daily[["session_date", "n_names", "beta_negative_share"]].assign(
                    regime=regime,
                    specification=name,
                )
            )

    summary = pd.DataFrame(summary_rows)
    summary["q_bh_two_beta_specs"] = pd.Series(pd.NA, index=summary.index, dtype="Float64")
    summary["bh_reject_q05"] = False
    for regime in ("development", "evaluation"):
        mask = summary["regime"].eq(regime) & summary["multiplicity_family"].eq(
            "post_review_beta_family"
        )
        p_values = summary.loc[mask, "p_two_sided"].astype(float).tolist()
        summary.loc[mask, "q_bh_two_beta_specs"] = benjamini_hochberg_q_values(p_values)
        summary.loc[mask, "bh_reject_q05"] = benjamini_hochberg(p_values, q=0.05)

    contrast_rows: list[dict[str, Any]] = []
    for name, outcome, regressors, family in specifications:
        combined = pd.concat(
            [
                daily_lookup[(regime, name)].assign(regime=regime)
                for regime in ("development", "evaluation")
            ],
            ignore_index=True,
        )
        contrast_rows.append(
            {
                "specification": name,
                "outcome": outcome,
                "regressors": ", ".join(regressors),
                "multiplicity_family": family,
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
    contrasts["q_bh_two_beta_specs"] = pd.Series(
        pd.NA, index=contrasts.index, dtype="Float64"
    )
    contrasts["bh_reject_q05"] = False
    mask = contrasts["multiplicity_family"].eq("post_review_beta_family")
    p_values = contrasts.loc[mask, "p_two_sided"].astype(float).tolist()
    contrasts.loc[mask, "q_bh_two_beta_specs"] = benjamini_hochberg_q_values(p_values)
    contrasts.loc[mask, "bh_reject_q05"] = benjamini_hochberg(p_values, q=0.05)
    daily_table = pd.concat(daily_rows, ignore_index=True)[
        ["regime", "specification", "session_date", "n_names", "beta_negative_share"]
    ]
    return summary, contrasts, daily_table, {"beta_complete_rows": int(beta_rows)}


def _contrast_diagnostics(
    development_path: Path,
    evaluation_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = pd.read_csv(development_path).assign(regime="development")
    evaluation = pd.read_csv(evaluation_path).assign(regime="evaluation")
    combined = pd.concat([development, evaluation], ignore_index=True)
    combined["session_date"] = pd.to_datetime(combined["session_date"])
    combined = combined.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    sensitivity = pd.DataFrame(
        [
            hac_regime_mean_contrast(
                combined,
                coefficient_col="beta_negative_share",
                date_col="session_date",
                regime_col="regime",
                reference="development",
                comparison="evaluation",
                hac_lags=lag,
            )
            for lag in (0, 5, 21, 42)
        ]
    )
    residuals = combined["beta_negative_share"] - combined.groupby("regime")[
        "beta_negative_share"
    ].transform("mean")
    acf_one = float(acf(residuals.to_numpy(dtype=float), nlags=1, fft=True)[1])
    ljung = acorr_ljungbox(residuals, lags=[5, 10, 21], return_df=True)
    rows = [{"diagnostic": "acf", "lag": 1, "statistic": acf_one, "p_value": np.nan}]
    rows.extend(
        {
            "diagnostic": "ljung_box",
            "lag": int(lag),
            "statistic": float(row["lb_stat"]),
            "p_value": float(row["lb_pvalue"]),
        }
        for lag, row in ljung.iterrows()
    )
    return sensitivity, pd.DataFrame(rows)


def _episode_robustness(
    control_path: Path,
    overlay_path: Path,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    control = pd.read_parquet(control_path).sort_values("session_date").reset_index(drop=True)
    overlay = pd.read_parquet(overlay_path).sort_values("session_date").reset_index(drop=True)
    if not control["session_date"].equals(overlay["session_date"]):
        raise RuntimeError("risk-arm dates do not align")
    merged = pd.DataFrame(
        {
            "session_date": pd.to_datetime(control["session_date"]),
            "d_return": overlay["net_return"] - control["net_return"],
            "d_downside": control["downside_sq"] - overlay["downside_sq"],
            "exposure_difference": overlay["exposure"] - control["exposure"],
        }
    )
    active = merged["exposure_difference"].abs().gt(1e-15) | merged["d_return"].abs().gt(1e-15)
    merged["episode"] = episode_ids(active)
    if (merged.loc[merged["episode"].eq(0), ["d_return", "d_downside"]].abs().sum() > 1e-12).any():
        raise RuntimeError("risk-arm differences remain outside reconstructed episodes")

    attribution = (
        merged.loc[merged["episode"].gt(0)]
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
    attribution["downside_share"] = attribution["downside_reduction_sum"] / total_downside

    seeds = spec["diagnostics"]["fnspid_episode_robustness"]["seeds"]
    leave_rows: list[dict[str, Any]] = []
    for offset, episode in enumerate(attribution["episode"].astype(int).tolist()):
        modified = merged["d_downside"].where(merged["episode"].ne(episode), 0.0)
        inference = paired_circular_block_mean(
            modified,
            block_length=20,
            replications=4999,
            seed=int(seeds["leave_one_out_start"]) + offset,
        )
        episode_row = attribution.loc[attribution["episode"].eq(episode)].iloc[0]
        leave_rows.append(
            {
                "dropped_episode": episode,
                "start": episode_row["start"],
                "end": episode_row["end"],
                **inference,
                "ci_excludes_zero": bool(inference["ci_low"] > 0),
            }
        )
    leave_one_out = pd.DataFrame(leave_rows)

    crash_mask = merged["session_date"].between("2020-02-01", "2020-04-30")
    crash_modified = merged["d_downside"].where(~crash_mask, 0.0)
    crash_inference = paired_circular_block_mean(
        crash_modified,
        block_length=20,
        replications=4999,
        seed=int(seeds["crash_exclusion"]),
    )
    crash = pd.DataFrame(
        [
            {
                "excluded_start": "2020-02-01",
                "excluded_end": "2020-04-30",
                "excluded_sessions": int(crash_mask.sum()),
                **crash_inference,
                "ci_excludes_zero": bool(crash_inference["ci_low"] > 0),
            }
        ]
    )
    gate = {
        "episodes": int(len(attribution)),
        "leave_one_out_all_intervals_above_zero": bool(leave_one_out["ci_excludes_zero"].all()),
        "crash_excluded_interval_above_zero": bool(crash.loc[0, "ci_excludes_zero"]),
        "passes": bool(
            leave_one_out["ci_excludes_zero"].all() and crash.loc[0, "ci_excludes_zero"]
        ),
    }
    return attribution, leave_one_out, crash, gate


def _class_base_rates(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    regimes = [
        ("development", panel.loc[panel["split"].eq("development")]),
        ("evaluation", panel.loc[panel["split"].eq("evaluation")]),
        ("overall", panel),
    ]
    for regime, frame in regimes:
        counts = recover_class_counts(frame)
        total = int(counts.to_numpy().sum())
        for label in ("negative", "neutral", "positive"):
            count = int(counts[label].sum())
            class_rows.append(
                {
                    "regime": regime,
                    "class": label,
                    "story_count": count,
                    "story_share": count / total,
                    "total_stories": total,
                }
            )
        tie_rows.append(
            {
                "regime": regime,
                "firm_days": int(len(frame)),
                "negative_share_equals_zero": int(frame["negative_share"].eq(0).sum()),
                "negative_share_equals_zero_fraction": float(frame["negative_share"].eq(0).mean()),
                "negative_share_equals_one": int(frame["negative_share"].eq(1).sum()),
                "negative_share_equals_one_fraction": float(frame["negative_share"].eq(1).mean()),
                "singleton_fraction": float(frame["n"].eq(1).mean()),
            }
        )
    return pd.DataFrame(class_rows), pd.DataFrame(tie_rows)


def _stratum_rank_spreads(panel: pd.DataFrame, notebook_86_strata: Path) -> pd.DataFrame:
    development = panel.loc[panel["split"].eq("development")].copy()
    committed = pd.read_csv(notebook_86_strata)
    rows: list[dict[str, Any]] = []
    for label, mask, count_col in (
        ("n = 1", development["n"].eq(1), None),
        ("n >= 2", development["n"].ge(2), "n"),
        ("n >= 3", development["n"].ge(3), "n"),
    ):
        frame = development.loc[mask]
        summary, daily, _ = conditional_negative_share_test(
            frame,
            date_col="session_date",
            outcome_col="ar_open_h1",
            count_col=count_col,
            min_names=MIN_NAMES,
            hac_lags=HAC_LAGS,
        )
        expected = float(committed.loc[committed["stratum"].eq(label), "estimate"].iloc[0])
        if not np.isclose(float(summary["estimate"]), expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Notebook 86 {label} coefficient did not reproduce")
        required = ["session_date", "ar_open_h1", "mean_continuous", "negative_share"]
        if count_col is not None:
            required.append(count_col)
        use = frame.loc[frame["session_date"].isin(daily["session_date"]), required].dropna()
        use = use.copy()
        use["rank_negative_share"] = (
            use.groupby("session_date")["negative_share"].rank(method="average", pct=True) - 0.5
        )
        pooled_low = float(use["rank_negative_share"].quantile(0.1))
        pooled_high = float(use["rank_negative_share"].quantile(0.9))
        pooled_spread = pooled_high - pooled_low
        daily_quantiles = use.groupby("session_date")["rank_negative_share"].quantile([0.1, 0.9])
        daily_table = daily_quantiles.unstack()
        mean_daily_spread = float((daily_table[0.9] - daily_table[0.1]).mean())
        rows.append(
            {
                "stratum": label,
                "model_rows": int(len(use)),
                "sessions": int(len(daily)),
                "coefficient": float(summary["estimate"]),
                "pooled_rank_p10": pooled_low,
                "pooled_rank_p90": pooled_high,
                "pooled_rank_spread": pooled_spread,
                "pooled_fitted_percentile_points": 100.0
                * pooled_spread
                * float(summary["estimate"]),
                "mean_session_rank_spread": mean_daily_spread,
                "mean_session_fitted_percentile_points": 100.0
                * mean_daily_spread
                * float(summary["estimate"]),
                "count_control_included": count_col is not None,
            }
        )
    return pd.DataFrame(rows)


def _coverage_audit(panel: pd.DataFrame, prices: pd.DataFrame, beta_audit: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"item": "panel_rows", "value": int(len(panel))},
        {"item": "panel_symbols", "value": int(panel["symbol"].nunique())},
        {"item": "price_rows", "value": int(len(prices))},
        {"item": "rows_with_trailing_beta", "value": int(panel["trailing_open_beta"].notna().sum())},
        {
            "item": "rows_with_prior_open_outcome",
            "value": int(panel["ar_previous_open_to_assigned_open"].notna().sum()),
        },
    ]
    rows.extend({"item": name, "value": value} for name, value in beta_audit.items())
    return pd.DataFrame(rows)


def run(*, output_dir: Path | None = None) -> dict[str, Path]:
    spec_path = ROOT / "experiments/specs" / SPEC_NAME
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["status"] != "frozen_before_external_review_robustness_pack_results":
        raise ValueError("unexpected frozen specification status")
    paths = resolve_inputs(spec)
    output = output_dir or (ROOT / "experiments/generated/87_fnspid_external_review_robustness_pack")
    output.mkdir(parents=True, exist_ok=True)

    expected_hashes = {
        "aggregators": AGGREGATORS_HASH,
        "prices": PRICE_HASH,
        "development_daily": spec["inputs"]["development_daily_coefficients"]["sha256"],
        "evaluation_daily": spec["inputs"]["evaluation_daily_coefficients"]["sha256"],
        "risk_control": spec["inputs"]["fnspid_har_control_daily"]["sha256"],
        "risk_overlay": spec["inputs"]["fnspid_har_overlay_daily"]["sha256"],
        "notebook_86_coefficients": spec["inputs"]["notebook_86_coefficients"]["sha256"],
        "notebook_86_strata": spec["inputs"]["notebook_86_strata"]["sha256"],
    }
    observed_hashes = {name: sha256(paths[name]) for name in expected_hashes}
    for name, expected in expected_hashes.items():
        if observed_hashes[name] != expected:
            raise RuntimeError(f"{name} hash mismatch: {observed_hashes[name]}")
    notebook_75_hash_before = sha256(paths["notebook_75"])

    panel = pd.read_parquet(
        paths["aggregators"],
        columns=[
            "symbol",
            "session_date",
            "split",
            "ar_open_h1",
            "ret_open_h1",
            "spy_ret_open_h1",
            "n",
            "mean_continuous",
            "negative_share",
            "mean_hard_label",
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
    panel["log1p_n"] = np.log1p(panel["n"].astype(float))

    prices, missing_symbols = load_adjusted_open_prices(
        paths["prices"],
        set(panel["symbol"].unique()) | {"SPY"},
        start="2010-01-01",
        end="2023-12-31",
    )
    if missing_symbols:
        raise RuntimeError(f"price archive is missing {len(missing_symbols)} panel symbols")
    leg_returns = session_open_close_return_legs(prices)
    beta_features = trailing_open_beta_features(leg_returns)
    features = beta_features.loc[
        beta_features["symbol"].ne("SPY"),
        [
            "symbol",
            "session_date",
            "ar_previous_open_to_assigned_open",
            "trailing_open_beta",
        ],
    ]
    panel = panel.merge(features, on=["symbol", "session_date"], how="left", validate="1:1")
    panel["ar_trailing_beta_open_h1"] = (
        panel["ret_open_h1"] - panel["trailing_open_beta"] * panel["spy_ret_open_h1"]
    )

    prior_summary, prior_daily = _prior_open_probe(panel)
    beta_summary, beta_contrasts, beta_daily, beta_audit = _beta_robustness(panel)
    hac_sensitivity, dependence = _contrast_diagnostics(
        paths["development_daily"], paths["evaluation_daily"]
    )
    episode_attribution, leave_one_out, crash_exclusion, episode_gate = _episode_robustness(
        paths["risk_control"], paths["risk_overlay"], spec
    )
    class_rates, tie_mass = _class_base_rates(panel)
    stratum_spreads = _stratum_rank_spreads(panel, paths["notebook_86_strata"])
    coverage = _coverage_audit(panel, prices, beta_audit)

    tables = {
        "prior_open_timing_probe": prior_summary,
        "prior_open_daily_coefficients": prior_daily,
        "beta_robustness_coefficients": beta_summary,
        "beta_robustness_contrasts": beta_contrasts,
        "beta_robustness_daily_coefficients": beta_daily,
        "contrast_hac_sensitivity": hac_sensitivity,
        "coefficient_dependence_diagnostics": dependence,
        "fnspid_episode_attribution": episode_attribution,
        "fnspid_leave_one_episode_out": leave_one_out,
        "fnspid_crash_exclusion": crash_exclusion,
        "finbert_class_base_rates": class_rates,
        "negative_share_tie_mass": tie_mass,
        "story_count_stratum_rank_spreads": stratum_spreads,
        "coverage_audit": coverage,
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path

    notebook_75_hash_after = sha256(paths["notebook_75"])
    if notebook_75_hash_after != notebook_75_hash_before:
        raise RuntimeError("Notebook 75 changed during Notebook 87 execution")
    outputs = [
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
        "notebook": "87_fnspid_external_review_robustness_pack",
        "status": "completed_post_review_robustness_not_confirmation",
        "executed_at_utc": datetime.now(UTC).isoformat(),
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
            name: {"path": str(paths[name]), "sha256": value}
            for name, value in observed_hashes.items()
        },
        "notebook_75": {
            "path": str(paths["notebook_75"].relative_to(ROOT)),
            "sha256_before": notebook_75_hash_before,
            "sha256_after": notebook_75_hash_after,
            "executed": False,
        },
        "episode_gate": episode_gate,
        "claim_boundaries": spec["claim_boundaries"],
        "outputs": outputs,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    written = run(output_dir=args.output_dir)
    print(json.dumps({name: str(path) for name, path in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
