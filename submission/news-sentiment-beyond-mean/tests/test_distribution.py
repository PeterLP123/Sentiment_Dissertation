"""Focused tests for within-firm-day distribution helpers."""

from __future__ import annotations

import pandas as pd

from experiments.lib.distribution import (
    assign_n_bin,
    firm_day_moments,
    mass_by_n_bin,
    polarity_label,
)


def test_assign_n_bin_edges() -> None:
    n = pd.Series([1, 2, 5, 6, 20, 21, 75, 76, 100])
    bins = assign_n_bin(n)
    assert list(bins) == ["1", "2-5", "2-5", "6-20", "6-20", "21-75", "21-75", "76+", "76+"]


def test_polarity_argmax() -> None:
    frame = pd.DataFrame(
        {
            "p_negative": [0.8, 0.1, 0.1],
            "p_neutral": [0.1, 0.8, 0.1],
            "p_positive": [0.1, 0.1, 0.8],
        }
    )
    assert list(polarity_label(frame)) == ["negative", "neutral", "positive"]


def test_firm_day_moments_mean_median_gap_and_shares() -> None:
    stories = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "session_date": pd.to_datetime(["2020-01-06"] * 3),
            "finbert_score": [-0.9, 0.0, 0.3],
            "is_recap": [0, 1, 0],
            "p_negative": [0.9, 0.1, 0.1],
            "p_neutral": [0.05, 0.8, 0.1],
            "p_positive": [0.05, 0.1, 0.8],
        }
    )
    stories["polarity"] = polarity_label(stories)
    moments = firm_day_moments(stories)
    assert len(moments) == 1
    row = moments.iloc[0]
    assert row["n"] == 3
    assert abs(row["mean"] - (-0.2)) < 1e-9
    assert abs(row["median"] - 0.0) < 1e-9
    assert abs(row["mean_minus_median"] - (-0.2)) < 1e-9
    assert abs(row["share_negative"] - 1 / 3) < 1e-9
    assert abs(row["share_neutral"] - 1 / 3) < 1e-9
    assert abs(row["share_positive"] - 1 / 3) < 1e-9
    assert row["n_bin"] == "2-5"


def test_mass_by_n_bin_shares_sum_to_one() -> None:
    moments = pd.DataFrame(
        {
            "n": [1, 1, 10, 100],
            "n_bin": assign_n_bin(pd.Series([1, 1, 10, 100])),
        }
    )
    mass = mass_by_n_bin(moments)
    assert abs(mass["firm_day_share"].sum() - 1.0) < 1e-12
    assert abs(mass["article_share"].sum() - 1.0) < 1e-12
    assert mass.loc["1", "firm_days"] == 2
    assert mass.loc["76+", "articles"] == 100.0
