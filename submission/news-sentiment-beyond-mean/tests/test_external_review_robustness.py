from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.lib.external_review_robustness import (
    episode_ids,
    recover_class_counts,
    trailing_open_beta_features,
)


def test_trailing_open_beta_recovers_linear_market_loading() -> None:
    dates = pd.bdate_range("2020-01-02", periods=150)
    market = 0.01 * np.sin(np.arange(len(dates)) / 7.0) + 0.002 * np.cos(
        np.arange(len(dates)) / 3.0
    )
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "session_date": dates,
            "ret_previous_open_to_assigned_open": 2.0 * market,
            "spy_ret_previous_open_to_assigned_open": market,
            "ar_previous_open_to_assigned_open": market,
        }
    )

    result = trailing_open_beta_features(frame)

    assert result["trailing_open_beta"].notna().sum() == 25
    assert result.iloc[-1]["trailing_open_beta"] == pytest.approx(2.0)


def test_episode_ids_number_contiguous_active_runs() -> None:
    active = pd.Series([False, True, True, False, True, False, True, True])
    assert episode_ids(active).tolist() == [0, 1, 1, 0, 2, 0, 3, 3]


def test_recover_class_counts_reconciles_hard_label_sufficient_statistics() -> None:
    frame = pd.DataFrame(
        {
            "n": [4, 3],
            "negative_share": [0.25, 0.0],
            "mean_hard_label": [0.25, 0.0],
        }
    )

    counts = recover_class_counts(frame)

    assert counts.to_dict(orient="records") == [
        {"negative": 1, "neutral": 1, "positive": 2},
        {"negative": 0, "neutral": 3, "positive": 0},
    ]
