from __future__ import annotations

import pandas as pd
import pytest

from final_experiments.lib.sparse_events import (
    annotate_strict_prior_novelty,
    build_sparse_daily_portfolio,
    build_sparse_state_panels,
    classify_proxy_cash_flow_distance,
    select_strongest_sparse_events,
)


def test_novelty_is_strictly_prior_and_expires_after_window() -> None:
    stories = pd.DataFrame(
        {
            "headline_sha256": ["a", "b", "c", "d"],
            "symbol": ["AAA"] * 4,
            "first_timestamp": pd.to_datetime(
                [
                    "2026-01-01T12:00:00Z",
                    "2026-01-01T12:00:00Z",
                    "2026-01-02T12:00:00Z",
                    "2026-02-02T12:00:00Z",
                ],
                utc=True,
            ),
            "headline": [
                "AAA launches new cloud platform",
                "AAA launches new cloud platform today",
                "AAA launches new cloud platform",
                "AAA launches new cloud platform",
            ],
        }
    )

    got = annotate_strict_prior_novelty(stories).set_index("headline_sha256")

    assert bool(got.loc["a", "is_novel_material_event"])
    assert bool(got.loc["b", "is_novel_material_event"])
    assert bool(got.loc["c", "near_duplicate_in_window"])
    assert bool(got.loc["d", "is_novel_material_event"])
    assert "headline" not in got


def test_cash_flow_distance_uses_frozen_priority_and_keeps_unknown_missing() -> None:
    headlines = pd.Series(
        [
            "AAA considers deal after reporting profit",
            "AAA awaits phase 2 trial results",
            "AAA reports higher quarterly revenue",
            "AAA signs supply contract",
            "AAA updates corporate website",
        ]
    )

    got = classify_proxy_cash_flow_distance(headlines)

    assert got.tolist()[:4] == [3, 2, 0, 1]
    assert pd.isna(got.iloc[4])


def test_strongest_selection_is_deterministic_and_skips_direction_conflict() -> None:
    mapped = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "entry_session": pd.to_datetime(["2026-01-05"] * 4),
            "headline_sha256": ["a", "b", "c", "d"],
            "score": [0.4, 0.8, 0.9, -0.9],
            "proxy_cash_flow_distance": pd.Series([0, 3, 1, 1], dtype="Int64"),
            "event_type": ["earnings_guidance"] * 4,
        }
    )

    selected = select_strongest_sparse_events(mapped)

    assert selected["symbol"].tolist() == ["AAA"]
    assert selected["material_signal"].iloc[0] == pytest.approx(0.8)
    assert selected["distance_signal"].iloc[0] == pytest.approx(0.2)


def test_state_horizon_uses_exchange_sessions_and_terminal_return_is_missing() -> None:
    sessions = pd.to_datetime(
        ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    )
    prices = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "session_date": sessions,
            "adjusted_open": [100.0, 110.0, 121.0, 133.1],
        }
    )
    selected = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "entry_session": [sessions[0]],
            "material_signal": [0.8],
            "distance_signal": [0.4],
        }
    )

    panel = build_sparse_state_panels(selected, prices, horizons=(1, 3))[3]

    assert panel["material_signal"].notna().tolist() == [True, True, True, False]
    assert panel.loc[0, "raw_open_h1"] == pytest.approx(0.1)
    assert panel.loc[2, "raw_open_h1"] == pytest.approx(0.1)
    assert pd.isna(panel.loc[3, "raw_open_h1"])


def test_new_unknown_distance_event_clears_older_distance_state() -> None:
    sessions = pd.to_datetime(
        ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    )
    prices = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "session_date": sessions,
            "adjusted_open": [100.0, 101.0, 102.0, 103.0],
        }
    )
    selected = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "entry_session": sessions[:2],
            "material_signal": [0.8, -0.6],
            "distance_signal": [0.4, float("nan")],
        }
    )

    panel = build_sparse_state_panels(selected, prices, horizons=(3,))[3]

    assert panel["material_signal"].tolist()[:3] == pytest.approx([0.8, -0.6, -0.6])
    assert panel.loc[0, "distance_signal"] == pytest.approx(0.4)
    assert panel.loc[1:, "distance_signal"].isna().all()


def test_sparse_portfolio_closes_on_first_inactive_session() -> None:
    sessions = pd.to_datetime(["2026-01-05", "2026-01-06"])
    panel = pd.DataFrame(
        {
            "session_date": sessions.repeat(10),
            "symbol": [f"S{i:02d}" for i in range(10)] * 2,
            "material_signal": list(range(10)) + [float("nan")] * 10,
            "raw_open_h1": [0.001] * 20,
        }
    )

    daily = build_sparse_daily_portfolio(
        panel, "material_signal", cost_bps_per_side=10.0, min_names=10
    )

    assert daily["n_names"].tolist() == [10, 0]
    assert daily["turnover"].tolist() == pytest.approx([0.5, 0.5])
    assert daily["cost"].tolist() == pytest.approx([0.001, 0.001])
