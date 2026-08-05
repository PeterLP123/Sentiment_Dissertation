"""Focused accounting tests for the fixed expanded-LSEG sector portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from final_experiments.lib.sector_portfolios import (
    SECTOR_MEMBERS,
    SectorPortfolioConfig,
    build_cost_curve,
    build_sector_strategy_family,
    circular_block_mean_test,
    validate_sector_map,
)


def _panel(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["session_date", "symbol", "strongest_event", "raw_open_h1"],
    ).assign(
        session_date=lambda frame: pd.to_datetime(frame["session_date"]),
        split="exploratory_full_period",
    )


def test_frozen_sector_map_is_eleven_disjoint_groups_of_four() -> None:
    audit = validate_sector_map()

    assert audit == {"sectors": 11, "symbols": 44, "members_per_sector": [4]}
    assert len({symbol for members in SECTOR_MEMBERS.values() for symbol in members}) == 44


def test_sector_rank_is_sector_neutral_with_equal_sector_budgets() -> None:
    frame = _panel(
        [
            ("2026-01-05", "AAPL", 1.0, 0.04),
            ("2026-01-05", "MSFT", -1.0, -0.04),
            ("2026-01-05", "JPM", 0.8, 0.02),
            ("2026-01-05", "BAC", -0.8, -0.02),
        ]
    )
    cfg = SectorPortfolioConfig(min_global_names=2, min_active_sectors=2)

    daily = build_sector_strategy_family(frame, config=cfg)["sector_rank"]

    assert daily.loc[0, "gross_return"] == pytest.approx(0.03)
    assert daily.loc[0, "net_exposure"] == pytest.approx(0.0, abs=1e-15)
    assert daily.loc[0, "gross_exposure"] == pytest.approx(1.0)
    assert daily.loc[0, "max_abs_weight"] == pytest.approx(0.25)
    assert daily.loc[0, "n_long"] == 2
    assert daily.loc[0, "n_short"] == 2
    # Opening plus the explicit final close are charged on this one-row run.
    assert daily.loc[0, "turnover"] == pytest.approx(1.0)
    assert daily.loc[0, "cost"] == pytest.approx(0.002)


def test_hysteresis_retains_extremes_inside_their_half_and_reduces_turnover() -> None:
    frame = _panel(
        [
            ("2026-01-05", "AAPL", 4.0, 0.0),
            ("2026-01-05", "MSFT", 3.0, 0.0),
            ("2026-01-05", "NVDA", 2.0, 0.0),
            ("2026-01-05", "AVGO", 1.0, 0.0),
            ("2026-01-06", "AAPL", 3.0, 0.0),
            ("2026-01-06", "MSFT", 4.0, 0.0),
            ("2026-01-06", "NVDA", 1.0, 0.0),
            ("2026-01-06", "AVGO", 2.0, 0.0),
        ]
    )
    cfg = SectorPortfolioConfig(min_global_names=2, min_active_sectors=1)

    outputs = build_sector_strategy_family(frame, config=cfg)
    extreme = outputs["sector_extremes"]
    hysteresis = outputs["sector_hysteresis"]

    # Day two switches both names under pure extremes, but retains both under
    # the top-half / bottom-half buffer. Final liquidation is included.
    assert extreme.loc[1, "turnover"] == pytest.approx(1.5)
    assert hysteresis.loc[1, "turnover"] == pytest.approx(0.5)
    assert hysteresis["net_exposure"].abs().max() < 1e-15
    assert hysteresis["max_abs_weight"].max() == pytest.approx(0.5)


def test_global_rank_matches_existing_daily_builder_accounting() -> None:
    frame = _panel(
        [
            ("2026-01-05", "AAPL", 1.0, 0.02),
            ("2026-01-05", "MSFT", -1.0, -0.02),
            ("2026-01-06", "AAPL", -1.0, 0.01),
            ("2026-01-06", "MSFT", 1.0, -0.01),
        ]
    )
    cfg = SectorPortfolioConfig(min_global_names=2, min_active_sectors=1)

    daily = build_sector_strategy_family(frame, config=cfg)["global_rank"]

    assert daily["gross_return"].tolist() == pytest.approx([0.02, -0.01])
    assert daily["turnover"].tolist() == pytest.approx([0.5, 1.5])
    assert daily["cost"].tolist() == pytest.approx([0.001, 0.003])


def test_sector_breadth_gate_closes_the_book_and_resets_hysteresis() -> None:
    frame = _panel(
        [
            ("2026-01-05", "AAPL", 1.0, 0.0),
            ("2026-01-05", "MSFT", -1.0, 0.0),
            ("2026-01-05", "JPM", 1.0, 0.0),
            ("2026-01-05", "BAC", -1.0, 0.0),
            ("2026-01-06", "AAPL", 1.0, 0.0),
            ("2026-01-06", "MSFT", -1.0, 0.0),
        ]
    )
    cfg = SectorPortfolioConfig(min_global_names=2, min_active_sectors=2)

    daily = build_sector_strategy_family(frame, config=cfg)["sector_hysteresis"]

    assert daily["gross_exposure"].tolist() == pytest.approx([1.0, 0.0])
    assert daily["turnover"].tolist() == pytest.approx([0.5, 0.5])
    assert daily["cost"].tolist() == pytest.approx([0.001, 0.001])


def test_cost_curve_reprices_fixed_turnover_without_changing_gross_returns() -> None:
    frame = _panel(
        [
            ("2026-01-05", "AAPL", 1.0, 0.02),
            ("2026-01-05", "MSFT", -1.0, -0.02),
        ]
    )
    cfg = SectorPortfolioConfig(min_global_names=2, min_active_sectors=1)
    outputs = build_sector_strategy_family(frame, config=cfg)

    curve = build_cost_curve(outputs, cost_bps_per_side=(0.0, 10.0))
    global_rows = curve.loc[curve["strategy"].eq("global_rank")].set_index(
        "cost_bps_per_side"
    )

    assert global_rows.loc[0.0, "mean_net"] == pytest.approx(0.02)
    assert global_rows.loc[10.0, "mean_net"] == pytest.approx(0.018)


def test_circular_block_mean_test_is_deterministic_and_bounded() -> None:
    values = np.array([0.01, -0.005, 0.012, 0.003, -0.002, 0.006])

    first = circular_block_mean_test(values, block_length=2, replications=199, seed=7)
    second = circular_block_mean_test(values, block_length=2, replications=199, seed=7)

    assert first == second
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    assert 0.0 < first["p_two_sided"] <= 1.0
