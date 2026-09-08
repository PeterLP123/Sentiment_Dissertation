from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from final_experiments.lib.novelty_conditioning import (
    contrast_series,
    daily_rank_regression,
    firm_day_split_aggregates,
    flag_already_known,
    recombination_error,
    residualise_within_session,
)


def _stories() -> pd.DataFrame:
    # Two firm-days. AA is mixed (two new, one already-known); BB is all new.
    return pd.DataFrame(
        [
            # symbol, date, score, p_neg, p_neu, p_pos, exact, jaccard
            ("AA", "2015-03-02", 0.8, 0.05, 0.15, 0.80, False, 0.10),
            ("AA", "2015-03-02", -0.6, 0.70, 0.20, 0.10, False, 0.20),
            ("AA", "2015-03-02", -0.4, 0.60, 0.30, 0.10, True, 1.00),
            ("BB", "2015-03-02", 0.2, 0.20, 0.50, 0.30, False, 0.45),
            ("BB", "2015-03-02", 0.4, 0.10, 0.40, 0.50, False, 0.30),
        ],
        columns=[
            "symbol",
            "session_date",
            "finbert_score",
            "p_negative",
            "p_neutral",
            "p_positive",
            "exact_repeat_in_window",
            "max_jaccard_in_window",
        ],
    )


def test_exact_repeat_is_known_regardless_of_overlap_threshold() -> None:
    stories = _stories()
    stories.loc[2, "max_jaccard_in_window"] = 0.0
    known = flag_already_known(stories, jaccard_threshold=0.99)
    assert known.tolist() == [False, False, True, False, False]


def test_threshold_moves_the_boundary_but_not_the_split_total() -> None:
    stories = _stories()
    loose = firm_day_split_aggregates(stories, jaccard_threshold=0.40)
    strict = firm_day_split_aggregates(stories, jaccard_threshold=0.50)
    # BB's 0.45-overlap story crosses back to "new" when the screen tightens.
    assert loose.loc[loose["symbol"] == "BB", "n_known"].item() == 1
    assert strict.loc[strict["symbol"] == "BB", "n_known"].item() == 0
    for frame in (loose, strict):
        assert frame["n"].equals(frame["n_new"] + frame["n_known"])


def test_legs_recombine_to_the_published_all_story_aggregates() -> None:
    split = firm_day_split_aggregates(_stories(), jaccard_threshold=0.50)
    errors = recombination_error(split)
    assert errors["max_abs_error"].max() < 1e-12

    aa = split.loc[split["symbol"] == "AA"].iloc[0]
    assert aa["n_new"] == 2 and aa["n_known"] == 1
    assert aa["mean_new"] == pytest.approx(0.1)
    assert aa["mean_known"] == pytest.approx(-0.4)
    # Argmax polarity: one positive and one negative among the new stories.
    assert aa["negshare_new"] == pytest.approx(0.5)
    assert aa["negshare_known"] == pytest.approx(1.0)
    assert bool(aa["is_mixed_day"]) is True

    bb = split.loc[split["symbol"] == "BB"].iloc[0]
    assert bb["n_known"] == 0
    assert np.isnan(bb["mean_known"])
    assert bool(bb["is_mixed_day"]) is False


def test_empty_leg_never_enters_a_regression() -> None:
    # A day with no already-known story has NaN on that leg, so dropna removes it
    # rather than letting a zero stand in for "no repeated news".
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2015-03-02"] * 12),
            "outcome": np.linspace(-1, 1, 12),
            "mean_new": np.linspace(0, 1, 12),
            "mean_known": [np.nan] * 6 + list(np.linspace(0, 1, 6)),
        }
    )
    daily, audit = daily_rank_regression(
        frame,
        date_col="session_date",
        outcome_col="outcome",
        regressor_cols=["mean_new", "mean_known"],
        min_names=10,
    )
    assert daily.empty
    assert audit["dates_below_min_names"] == 1
    assert audit["dates_used"] == 0


def test_paired_contrast_recovers_a_known_difference() -> None:
    rows = []
    for offset, session in enumerate(pd.date_range("2015-01-05", periods=20, freq="B")):
        for name in range(24):
            new = ((name * 5 + offset) % 24) / 23
            known = ((name * 11 + offset) % 24) / 23
            # New news is priced; already-known news is not.
            rows.append(
                {
                    "session_date": session,
                    "outcome": 1.0 * new + 0.0 * known,
                    "mean_new": new,
                    "mean_known": known,
                }
            )
    daily, audit = daily_rank_regression(
        pd.DataFrame(rows),
        date_col="session_date",
        outcome_col="outcome",
        regressor_cols=["mean_new", "mean_known"],
        min_names=10,
    )
    assert audit["dates_used"] == 20
    paired = contrast_series(daily, minuend="beta_mean_new", subtrahend="beta_mean_known", name="gap")
    assert paired["gap"].mean() > 0.5
    assert (paired["gap"] > 0).all()


def test_residual_removes_the_control_and_keeps_the_orthogonal_part() -> None:
    rows = []
    for offset, session in enumerate(pd.date_range("2015-01-05", periods=8, freq="B")):
        for name in range(20):
            control = ((name * 3 + offset) % 20) / 19
            extra = ((name * 7 + offset) % 20) / 19
            rows.append(
                {
                    "session_date": session,
                    # target is the control plus an orthogonal-ish component
                    "target": control + 0.5 * extra,
                    "control": control,
                }
            )
    frame = pd.DataFrame(rows)
    residual = residualise_within_session(
        frame,
        date_col="session_date",
        target_col="target",
        control_cols=["control"],
        min_names=10,
    )
    assert residual.notna().all()
    # A residual is orthogonal to its control and mean-zero inside every session.
    for _, day in frame.assign(residual=residual).groupby("session_date"):
        assert day["residual"].mean() == pytest.approx(0.0, abs=1e-12)
        assert np.corrcoef(day["residual"], day["control"])[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert residual.std() > 0


def test_residual_is_nan_when_the_session_is_too_small() -> None:
    frame = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2015-01-05"] * 6),
            "target": np.linspace(0, 1, 6),
            "control": np.linspace(1, 0, 6),
        }
    )
    residual = residualise_within_session(
        frame,
        date_col="session_date",
        target_col="target",
        control_cols=["control"],
        min_names=10,
    )
    assert residual.isna().all()


def test_rank_deficient_session_is_dropped_not_fitted() -> None:
    rows = []
    for name in range(12):
        value = name / 11
        rows.append(
            {
                "session_date": pd.Timestamp("2015-03-02"),
                "outcome": value,
                "a": value,
                "b": value,  # perfectly collinear after ranking
            }
        )
    daily, audit = daily_rank_regression(
        pd.DataFrame(rows),
        date_col="session_date",
        outcome_col="outcome",
        regressor_cols=["a", "b"],
        min_names=10,
    )
    assert daily.empty
    assert audit["dates_rank_deficient"] == 1
