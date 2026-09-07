"""Focused tests for firm-day aggregation rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.lib.aggregators import (
    attention_log_n,
    benjamini_hochberg,
    benjamini_hochberg_q_values,
    cross_sectional_daily_ic,
    decayed_state_series,
    dispersion,
    firm_day_soft_negative_mass,
    mean_continuous,
    mean_hard_label,
    median_continuous,
    negative_share,
    soft_negative_mass,
    strongest_event,
    trimmed_mean,
)


def _stories(scores: list[float], *, recap: list[bool] | None = None) -> pd.DataFrame:
    n = len(scores)
    probs = []
    for s in scores:
        # Reconstruct a simple probability triple with score = p_pos - p_neg.
        p_pos = (1.0 + s) / 2.0 * 0.8 + 0.1
        p_neg = (1.0 - s) / 2.0 * 0.8 + 0.1
        p_neu = max(1.0 - p_pos - p_neg, 0.01)
        total = p_pos + p_neg + p_neu
        probs.append((p_neg / total, p_neu / total, p_pos / total))
    frame = pd.DataFrame(
        {
            "finbert_score": scores,
            "p_negative": [p[0] for p in probs],
            "p_neutral": [p[1] for p in probs],
            "p_positive": [p[2] for p in probs],
            "is_recap": recap if recap is not None else [False] * n,
            "event_index": list(range(n)),
        }
    )
    return frame


def test_mean_hard_and_continuous() -> None:
    stories = _stories([-0.9, 0.0, 0.8])
    assert abs(mean_continuous(stories) - float(np.mean([-0.9, 0.0, 0.8]))) < 1e-12
    assert abs(median_continuous(stories) - 0.0) < 1e-12


def test_negative_share_and_dispersion() -> None:
    stories = _stories([-0.9, -0.5, 0.8])
    # All three have distinct argmax from constructed probs.
    share = negative_share(stories)
    assert 0.0 <= share <= 1.0
    assert dispersion(stories) > 0
    assert dispersion(_stories([0.5])) == 0.0


def test_trimmed_mean_small_n_equals_mean() -> None:
    stories = _stories([0.1, 0.2, 0.3])
    assert abs(trimmed_mean(stories) - mean_continuous(stories)) < 1e-12


def test_trimmed_mean_drops_tails() -> None:
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    stories = _stories(scores)
    # 10% of 10 → drop 1 each side → mean of middle 8
    expected = float(np.mean(scores[1:-1]))
    assert abs(trimmed_mean(stories) - expected) < 1e-12


def test_strongest_prefers_non_recap_and_flags_conflict() -> None:
    ok = _stories([0.2, 0.9], recap=[True, False])
    assert abs(strongest_event(ok) - 0.9) < 1e-12
    conflict = _stories([0.9, -0.9], recap=[False, False])
    assert np.isnan(strongest_event(conflict))


def test_attention_log_n() -> None:
    stories = _stories([0.5, 0.5])
    assert abs(attention_log_n(stories) - 0.5 * np.log1p(2)) < 1e-12


def test_decayed_state_decays_and_resets_on_reversal() -> None:
    firm_day = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "session_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "mean_continuous": [0.8, 0.8, -0.8],
        }
    )
    ordinals = {
        pd.Timestamp("2020-01-02"): 0,
        pd.Timestamp("2020-01-03"): 1,
        pd.Timestamp("2020-01-06"): 4,
    }
    state = decayed_state_series(firm_day, session_ordinals=ordinals)
    assert state.iloc[0] > 0
    assert state.iloc[1] > state.iloc[0]  # same-sign add after 1-session decay
    assert state.iloc[2] < state.iloc[1]  # reversal reset + negative impulse


def test_mean_hard_label_incumbent() -> None:
    stories = _stories([-0.9, 0.1, 0.8])
    # Constructed probs should yield neg / ~neu-or-pos / pos
    val = mean_hard_label(stories)
    assert -1.0 <= val <= 1.0


def test_soft_negative_mass_is_mean_probability_not_hard_share() -> None:
    stories = pd.DataFrame(
        {
            "p_negative": [0.60, 0.20, 0.10],
            "p_neutral": [0.30, 0.70, 0.20],
            "p_positive": [0.10, 0.10, 0.70],
            "finbert_score": [-0.50, -0.10, 0.60],
        }
    )
    assert abs(soft_negative_mass(stories) - 0.30) < 1e-12
    assert abs(negative_share(stories) - (1.0 / 3.0)) < 1e-12


def test_firm_day_soft_negative_mass_groups_stories() -> None:
    stories = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "session_date": ["2020-01-02", "2020-01-02", "2020-01-02"],
            "p_negative": [0.2, 0.4, 0.9],
        }
    )
    out = firm_day_soft_negative_mass(stories).set_index("symbol")
    assert abs(float(out.loc["AAA", "soft_negative_mass"]) - 0.3) < 1e-12
    assert abs(float(out.loc["BBB", "soft_negative_mass"]) - 0.9) < 1e-12


def test_benjamini_hochberg_rejects_strong_signals() -> None:
    flags = benjamini_hochberg([0.001, 0.02, 0.5, 0.8], q=0.05)
    assert flags[0] is True
    assert flags[-1] is False
    q_values = benjamini_hochberg_q_values([0.001, 0.02, 0.5, 0.8])
    assert q_values[0] < 0.05
    assert q_values[-1] > 0.05


def test_ic_is_cross_sectional_within_each_day_not_globally_pooled() -> None:
    # Across dates both x and y jump by ~100, which makes a pooled rank
    # correlation look positive. Within every date their ordering is exactly
    # opposite, so the research estimand must be -1.
    x = pd.Series([1, 2, 3, 101, 102, 103], dtype=float)
    y = pd.Series([3, 2, 1, 103, 102, 101], dtype=float)
    dates = pd.Series(pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3))
    stats = cross_sectional_daily_ic(x, y, dates, min_names=3, hac_lags=1)
    assert abs(float(stats["ic"]) + 1.0) < 1e-12
    assert stats["n_clusters"] == 2
    assert stats["inference"] == "mean_daily_cross_sectional_spearman_hac"
