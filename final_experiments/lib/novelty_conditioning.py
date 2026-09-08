"""Split firm-day sentiment into new versus already-known stories.

Workstream 2 asks whether a firm-day's stories carry information because they
are *new*, or whether much of the flow restates what the same firm already
published.  ``lib/novelty.py`` builds the per-story repetition screen; this
module turns that screen into firm-day aggregates that can enter the same daily
cross-sectional rank regression used for the frozen beyond-the-mean tests.

Two things are deliberately kept here rather than in the notebook because a
silent error would corrupt every downstream coefficient:

* the new/already-known split and its firm-day aggregation, which must reproduce
  the published all-story ``mean_continuous`` / ``negative_share`` exactly when
  the two legs are recombined; and
* the generic daily rank regression, which must drop a session rather than
  return a coefficient from a rank-deficient or near-constant design.

The screen itself is uncalibrated: the blinded 180-story audit in
``outputs/02_filters_and_distribution/`` has no human labels, so "already known"
means "near-verbatim repeat of a strictly earlier same-firm headline within the
trailing window", not "the market already knew".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.aggregators import HARD_LABEL
from final_experiments.lib.conditional_aggregation import centred_percentile_rank
from final_experiments.lib.distribution import polarity_label

# Operative screen. 0.80 leaves too few mixed firm-days to identify a contrast;
# 0.30 starts absorbing generic boilerplate overlap. Frozen return-blind at 0.50.
PRIMARY_JACCARD_THRESHOLD = 0.50
SENSITIVITY_JACCARD_THRESHOLDS: tuple[float, ...] = (0.40, 0.60, 0.80)

_SPLIT_COLUMNS = (
    "n",
    "n_new",
    "n_known",
    "known_share",
    "mean_new",
    "mean_known",
    "negshare_new",
    "negshare_known",
    "mean_all",
    "negshare_all",
)


def flag_already_known(
    stories: pd.DataFrame,
    *,
    jaccard_threshold: float = PRIMARY_JACCARD_THRESHOLD,
) -> pd.Series:
    """Mark stories that repeat a strictly earlier same-firm headline.

    An exact normalized-hash repeat always counts. Otherwise the story counts
    when its best trailing-window Jaccard overlap reaches ``jaccard_threshold``.
    Both inputs come from ``novelty.annotate_novelty`` and are already restricted
    to strictly earlier firm dates, so no same-day information leaks in.
    """

    if not 0.0 < jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in (0, 1]")
    required = {"exact_repeat_in_window", "max_jaccard_in_window"}
    if missing := required - set(stories.columns):
        raise ValueError(f"stories missing columns: {sorted(missing)}")

    exact = stories["exact_repeat_in_window"].astype(bool).to_numpy()
    overlap = pd.to_numeric(stories["max_jaccard_in_window"], errors="coerce").to_numpy(dtype=float)
    known = exact | (np.nan_to_num(overlap, nan=0.0) >= float(jaccard_threshold))
    return pd.Series(known, index=stories.index, name="is_already_known")


def firm_day_split_aggregates(
    stories: pd.DataFrame,
    *,
    jaccard_threshold: float = PRIMARY_JACCARD_THRESHOLD,
) -> pd.DataFrame:
    """Aggregate scored stories to firm-days, separately for each novelty leg.

    Required columns: ``symbol``, ``session_date``, ``finbert_score``, the three
    FinBERT class probabilities (or a precomputed ``hard_label``), and the two
    repetition-screen fields consumed by :func:`flag_already_known`.

    ``mean_*`` is the mean continuous FinBERT score and ``negshare_*`` the share
    of argmax-negative stories, matching ``lib/aggregators``. A leg with no
    stories yields ``NaN`` for its mean and share and ``0`` for its count.
    """

    required = {"symbol", "session_date", "finbert_score"}
    if missing := required - set(stories.columns):
        raise ValueError(f"stories missing columns: {sorted(missing)}")

    frame = stories.copy()
    if "hard_label" not in frame.columns:
        frame["hard_label"] = polarity_label(frame).map(HARD_LABEL).astype(float)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["finbert_score"] = pd.to_numeric(frame["finbert_score"], errors="coerce").astype(float)
    frame["is_already_known"] = flag_already_known(frame, jaccard_threshold=jaccard_threshold)
    frame["is_negative"] = (frame["hard_label"].to_numpy(dtype=float) < 0).astype(float)

    keys = ["symbol", "session_date"]
    legs: dict[bool, pd.DataFrame] = {}
    for known in (False, True):
        part = frame.loc[frame["is_already_known"].to_numpy() == known]
        legs[known] = (
            part.groupby(keys, sort=False)
            .agg(count=("finbert_score", "size"), mean=("finbert_score", "mean"), negshare=("is_negative", "mean"))
            .reset_index()
        )

    out = (
        frame.groupby(keys, sort=False)
        .agg(n=("finbert_score", "size"), mean_all=("finbert_score", "mean"), negshare_all=("is_negative", "mean"))
        .reset_index()
    )
    for known, suffix in ((False, "new"), (True, "known")):
        leg = legs[known].rename(
            columns={"count": f"n_{suffix}", "mean": f"mean_{suffix}", "negshare": f"negshare_{suffix}"}
        )
        out = out.merge(leg, on=keys, how="left")
        out[f"n_{suffix}"] = out[f"n_{suffix}"].fillna(0).astype(int)

    out["known_share"] = out["n_known"].to_numpy(dtype=float) / out["n"].to_numpy(dtype=float)
    if not out["n"].eq(out["n_new"] + out["n_known"]).all():
        raise RuntimeError("novelty legs do not reconcile to the firm-day story count")
    out["is_mixed_day"] = (out["n_new"] > 0) & (out["n_known"] > 0)
    ordered = [*keys, *_SPLIT_COLUMNS, "is_mixed_day"]
    return out.loc[:, ordered].sort_values(keys, kind="mergesort").reset_index(drop=True)


def recombination_error(split: pd.DataFrame) -> pd.DataFrame:
    """Largest absolute error from rebuilding all-story aggregates from the legs.

    The count-weighted average of the two legs must return the all-story mean and
    negative share.  A non-trivial error means the split lost or double-counted
    stories, which would invalidate every contrast estimated from it.
    """

    weight_new = split["n_new"].to_numpy(dtype=float) / split["n"].to_numpy(dtype=float)
    weight_known = split["n_known"].to_numpy(dtype=float) / split["n"].to_numpy(dtype=float)
    rows = []
    for statistic in ("mean", "negshare"):
        new = np.nan_to_num(split[f"{statistic}_new"].to_numpy(dtype=float), nan=0.0)
        known = np.nan_to_num(split[f"{statistic}_known"].to_numpy(dtype=float), nan=0.0)
        rebuilt = weight_new * new + weight_known * known
        error = np.abs(rebuilt - split[f"{statistic}_all"].to_numpy(dtype=float))
        rows.append({"statistic": f"{statistic}_all", "max_abs_error": float(np.nanmax(error))})
    return pd.DataFrame(rows)


def daily_rank_regression(
    frame: pd.DataFrame,
    *,
    date_col: str,
    outcome_col: str,
    regressor_cols: list[str],
    min_names: int = 10,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One within-session rank regression per date, with an exclusion audit.

    Every variable is converted to an average-tie centred percentile rank inside
    the session, matching the frozen conditional test. Sessions are dropped when
    they have too few complete names, a constant ranked variable, or a
    rank-deficient design; the counts must reconcile to the total.
    """

    if min_names < 4:
        raise ValueError("min_names must be at least four")
    if len(set(regressor_cols)) != len(regressor_cols):
        raise ValueError("regressor_cols must be unique")
    required = {date_col, outcome_col, *regressor_cols}
    if missing := required - set(frame.columns):
        raise ValueError(f"frame missing columns: {sorted(missing)}")

    use = frame.loc[:, sorted(required)].copy()
    use[date_col] = pd.to_datetime(use[date_col]).dt.normalize()
    complete = [outcome_col, *regressor_cols]

    audit = {
        "dates_total": int(use[date_col].nunique()),
        "dates_used": 0,
        "dates_below_min_names": 0,
        "dates_constant_variable": 0,
        "dates_rank_deficient": 0,
    }
    rows: list[dict[str, Any]] = []
    for session, day in use.groupby(date_col, sort=True):
        day = day.dropna(subset=complete)
        if len(day) < min_names:
            audit["dates_below_min_names"] += 1
            continue
        if any(day[column].nunique() < 2 for column in complete):
            audit["dates_constant_variable"] += 1
            continue
        y = centred_percentile_rank(day[outcome_col]).to_numpy(dtype=float)
        ranked = np.column_stack(
            [centred_percentile_rank(day[column]).to_numpy(dtype=float) for column in regressor_cols]
        )
        design = np.column_stack([np.ones(len(day), dtype=float), ranked])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            audit["dates_rank_deficient"] += 1
            continue
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        total = float(np.sum((y - y.mean()) ** 2))
        record: dict[str, Any] = {
            "session_date": pd.Timestamp(session),
            "n_names": int(len(day)),
            "intercept": float(beta[0]),
            "r_squared": float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan"),
            "condition_number": float(np.linalg.cond(design)),
        }
        for position, column in enumerate(regressor_cols, start=1):
            record[f"beta_{column}"] = float(beta[position])
        rows.append(record)

    columns = [
        "session_date",
        "n_names",
        "intercept",
        "r_squared",
        "condition_number",
        *(f"beta_{column}" for column in regressor_cols),
    ]
    daily = pd.DataFrame(rows, columns=columns).sort_values("session_date", kind="mergesort").reset_index(drop=True)
    audit["dates_used"] = int(len(daily))
    excluded = sum(
        value for key, value in audit.items() if key.startswith("dates_") and key not in {"dates_total", "dates_used"}
    )
    if excluded + audit["dates_used"] != audit["dates_total"]:
        raise RuntimeError("daily regression audit does not reconcile")
    return daily, audit


def residualise_within_session(
    frame: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    control_cols: list[str],
    min_names: int = 10,
) -> pd.Series:
    """Within-session rank residual of ``target_col`` after the control ranks.

    A conditional regression coefficient is not tradable as a raw sort: it
    describes the part of the signal orthogonal to the other regressors in the
    same cross-section.  This returns exactly that part, so a portfolio can be
    built from the quantity the coefficient actually refers to.  Sessions that
    are too small or rank-deficient return ``NaN`` rather than an unregularised
    residual.
    """

    required = {date_col, target_col, *control_cols}
    if missing := required - set(frame.columns):
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    if target_col in control_cols:
        raise ValueError("target_col must not appear in control_cols")

    out = pd.Series(np.nan, index=frame.index, dtype=float, name=f"{target_col}_residual")
    complete = [target_col, *control_cols]
    for _, day in frame.groupby(pd.to_datetime(frame[date_col]).dt.normalize(), sort=False):
        day = day.dropna(subset=complete)
        if len(day) < min_names or any(day[column].nunique() < 2 for column in complete):
            continue
        y = centred_percentile_rank(day[target_col]).to_numpy(dtype=float)
        design = np.column_stack(
            [
                np.ones(len(day), dtype=float),
                *(centred_percentile_rank(day[column]).to_numpy(dtype=float) for column in control_cols),
            ]
        )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        out.loc[day.index] = y - design @ beta
    return out


def contrast_series(daily: pd.DataFrame, *, minuend: str, subtrahend: str, name: str) -> pd.DataFrame:
    """Attach a paired within-session coefficient difference.

    Both coefficients come from the same daily regression, so the difference is
    paired by construction and inherits that session's sample.
    """

    for column in (minuend, subtrahend):
        if column not in daily.columns:
            raise ValueError(f"daily coefficients missing {column}")
    out = daily.copy()
    out[name] = out[minuend].to_numpy(dtype=float) - out[subtrahend].to_numpy(dtype=float)
    return out


__all__ = [
    "PRIMARY_JACCARD_THRESHOLD",
    "SENSITIVITY_JACCARD_THRESHOLDS",
    "contrast_series",
    "daily_rank_regression",
    "firm_day_split_aggregates",
    "flag_already_known",
    "recombination_error",
    "residualise_within_session",
]
