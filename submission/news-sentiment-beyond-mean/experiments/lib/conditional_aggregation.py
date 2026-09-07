"""Daily cross-sectional tests of distribution information beyond the mean.

The core estimand is a Fama--MacBeth-style time-series mean of daily rank-
regression coefficients.  Each session ranks the next-open outcome, mean
sentiment, negative-story share, and log story count independently.  The
negative-share coefficient therefore asks a direct conditional question: does
the negative tail explain return ranks after the usual mean and news volume are
already in the same daily cross-section?
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def centred_percentile_rank(values: pd.Series) -> pd.Series:
    """Average-tie percentile ranks centred at zero."""

    return values.rank(method="average", pct=True) - 0.5


def daily_conditional_rank_coefficients(
    frame: pd.DataFrame,
    *,
    date_col: str,
    outcome_col: str,
    mean_col: str,
    negative_share_col: str,
    count_col: str | None,
    min_names: int = 10,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Estimate one conditional rank regression per session.

    Regressors are the centred within-session ranks of mean sentiment and
    negative-story share. ``log1p(count)`` is also included unless
    ``count_col`` is ``None``. A session is excluded when it has too few
    complete names, a constant ranked variable, or a rank-deficient design.
    Exclusion counts are returned for the notebook audit.
    """

    if min_names < 4:
        raise ValueError("min_names must be at least four")
    required = {date_col, outcome_col, mean_col, negative_share_col}
    if count_col is not None:
        required.add(count_col)
    if missing := required - set(frame.columns):
        raise ValueError(f"frame missing columns: {sorted(missing)}")

    use = frame.loc[:, list(required)].copy()
    use[date_col] = pd.to_datetime(use[date_col]).dt.normalize()
    if count_col is not None:
        use[count_col] = pd.to_numeric(use[count_col], errors="coerce")
        if use[count_col].dropna().lt(0).any():
            raise ValueError("story counts must be non-negative")
        use["_log_count"] = np.log1p(use[count_col].astype(float))

    audit = {
        "dates_total": int(use[date_col].nunique()),
        "dates_used": 0,
        "dates_below_min_names": 0,
        "dates_constant_variable": 0,
        "dates_rank_deficient": 0,
    }
    rows: list[dict[str, Any]] = []
    model_columns = [mean_col, negative_share_col]
    if count_col is not None:
        model_columns.append("_log_count")
    complete_columns = [outcome_col, *model_columns]
    for session, day in use.groupby(date_col, sort=True):
        day = day.dropna(subset=complete_columns).copy()
        if len(day) < min_names:
            audit["dates_below_min_names"] += 1
            continue
        if any(day[column].nunique() < 2 for column in complete_columns):
            audit["dates_constant_variable"] += 1
            continue

        y = centred_percentile_rank(day[outcome_col]).to_numpy(dtype=float)
        ranked = np.column_stack(
            [centred_percentile_rank(day[column]).to_numpy(dtype=float) for column in model_columns]
        )
        design = np.column_stack([np.ones(len(day), dtype=float), ranked])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            audit["dates_rank_deficient"] += 1
            continue
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        residual = y - fitted
        total_sum_squares = float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {
                "session_date": pd.Timestamp(session),
                "n_names": int(len(day)),
                "intercept": float(beta[0]),
                "beta_mean_continuous": float(beta[1]),
                "beta_negative_share": float(beta[2]),
                "beta_log_story_count": float(beta[3]) if count_col is not None else float("nan"),
                "r_squared": (
                    float(1.0 - np.sum(residual**2) / total_sum_squares)
                    if total_sum_squares > 0
                    else float("nan")
                ),
                "condition_number": float(np.linalg.cond(design)),
            }
        )

    daily_columns = [
        "session_date",
        "n_names",
        "intercept",
        "beta_mean_continuous",
        "beta_negative_share",
        "beta_log_story_count",
        "r_squared",
        "condition_number",
    ]
    daily = pd.DataFrame(rows, columns=daily_columns)
    daily = daily.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    audit["dates_used"] = int(len(daily))
    excluded = sum(
        audit[key]
        for key in audit
        if key.startswith("dates_") and key not in {"dates_total", "dates_used"}
    )
    if excluded + audit["dates_used"] != audit["dates_total"]:
        raise RuntimeError("daily regression audit does not reconcile")
    return daily, audit


def hac_mean_coefficient(
    daily: pd.DataFrame,
    *,
    coefficient_col: str = "beta_negative_share",
    hac_lags: int = 5,
) -> dict[str, float | int | str]:
    """Estimate the time-series mean of a daily coefficient with HAC inference."""

    import statsmodels.api as sm

    if coefficient_col not in daily.columns:
        raise ValueError(f"daily coefficients missing {coefficient_col}")
    values = pd.to_numeric(daily[coefficient_col], errors="coerce")
    values = values[np.isfinite(values.to_numpy(dtype=float))]
    base: dict[str, float | int | str] = {
        "coefficient": coefficient_col,
        "n_clusters": int(len(values)),
        "hac_lags": int(hac_lags),
        "inference": "mean_daily_cross_sectional_rank_coefficient_hac",
    }
    if len(values) < 2:
        return {
            **base,
            "estimate": float("nan"),
            "se": float("nan"),
            "t": float("nan"),
            "p_two_sided": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    model = sm.OLS(values.to_numpy(dtype=float), np.ones((len(values), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": min(hac_lags, len(values) - 1)},
    )
    estimate = float(model.params[0])
    se = float(model.bse[0])
    critical = 1.959963984540054
    return {
        **base,
        "estimate": estimate,
        "se": se,
        "t": float(model.tvalues[0]),
        "p_two_sided": float(model.pvalues[0]),
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
    }


def hac_regime_mean_contrast(
    daily: pd.DataFrame,
    *,
    coefficient_col: str = "beta_negative_share",
    date_col: str = "session_date",
    regime_col: str = "regime",
    reference: str = "development",
    comparison: str = "evaluation",
    hac_lags: int = 5,
) -> dict[str, float | int | str]:
    """Estimate a predeclared difference in daily-coefficient means.

    The binary-regime slope is ``comparison - reference``. HAC inference is
    applied after sorting the combined daily coefficient series by session.
    This tests the change directly instead of treating significance in one
    regime and non-significance in another as evidence of a difference.
    """

    import statsmodels.api as sm

    required = {coefficient_col, date_col, regime_col}
    if missing := required - set(daily.columns):
        raise ValueError(f"daily coefficients missing columns: {sorted(missing)}")
    if reference == comparison:
        raise ValueError("reference and comparison regimes must differ")

    use = daily.loc[:, [date_col, regime_col, coefficient_col]].copy()
    use[date_col] = pd.to_datetime(use[date_col], errors="coerce").dt.normalize()
    use[coefficient_col] = pd.to_numeric(use[coefficient_col], errors="coerce")
    use = use.loc[use[regime_col].isin([reference, comparison])]
    finite = np.isfinite(use[coefficient_col].to_numpy(dtype=float))
    use = use.loc[use[date_col].notna() & finite].sort_values(
        date_col, kind="mergesort"
    )
    if use[date_col].duplicated().any():
        raise ValueError("daily coefficient series must contain at most one row per session")

    counts = use[regime_col].value_counts()
    for regime in (reference, comparison):
        if int(counts.get(regime, 0)) < 2:
            raise ValueError(f"regime {regime!r} needs at least two daily coefficients")

    indicator = (use[regime_col] == comparison).astype(float).to_numpy()
    design = np.column_stack([np.ones(len(use), dtype=float), indicator])
    model = sm.OLS(use[coefficient_col].to_numpy(dtype=float), design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": min(hac_lags, len(use) - 1)},
    )
    estimate = float(model.params[1])
    se = float(model.bse[1])
    critical = 1.959963984540054
    reference_values = use.loc[use[regime_col] == reference, coefficient_col]
    comparison_values = use.loc[use[regime_col] == comparison, coefficient_col]
    return {
        "contrast": f"{comparison}_minus_{reference}",
        "coefficient": coefficient_col,
        "reference": reference,
        "comparison": comparison,
        "reference_estimate": float(reference_values.mean()),
        "comparison_estimate": float(comparison_values.mean()),
        "estimate": estimate,
        "se": se,
        "t": float(model.tvalues[1]),
        "p_two_sided": float(model.pvalues[1]),
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "n_reference": int(len(reference_values)),
        "n_comparison": int(len(comparison_values)),
        "n_clusters": int(len(use)),
        "hac_lags": int(hac_lags),
        "inference": "daily_cross_sectional_rank_coefficient_regime_contrast_hac",
    }


def conditional_negative_share_test(
    frame: pd.DataFrame,
    *,
    date_col: str,
    outcome_col: str,
    mean_col: str = "mean_continuous",
    negative_share_col: str = "negative_share",
    count_col: str | None = "n",
    min_names: int = 10,
    hac_lags: int = 5,
) -> tuple[dict[str, float | int | str], pd.DataFrame, dict[str, int]]:
    """Run the frozen negative-share conditional test and return all audit layers."""

    daily, audit = daily_conditional_rank_coefficients(
        frame,
        date_col=date_col,
        outcome_col=outcome_col,
        mean_col=mean_col,
        negative_share_col=negative_share_col,
        count_col=count_col,
        min_names=min_names,
    )
    summary = hac_mean_coefficient(
        daily,
        coefficient_col="beta_negative_share",
        hac_lags=hac_lags,
    )
    summary.update(
        {
            "n_rows_complete": int(
                frame[
                    [
                        date_col,
                        outcome_col,
                        mean_col,
                        negative_share_col,
                        *([count_col] if count_col is not None else []),
                    ]
                ]
                .dropna()
                .shape[0]
            ),
            "minimum_names": int(min_names),
            "expected_direction": "negative",
        }
    )
    if math.isfinite(float(summary["estimate"])):
        summary["direction_matches"] = bool(float(summary["estimate"]) < 0)
    else:
        summary["direction_matches"] = False
    return summary, daily, audit


__all__ = [
    "centred_percentile_rank",
    "conditional_negative_share_test",
    "daily_conditional_rank_coefficients",
    "hac_mean_coefficient",
    "hac_regime_mean_contrast",
]
