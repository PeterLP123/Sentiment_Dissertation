from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from final_experiments.lib.fixed_share import (
    build_sparse_extreme_targets,
    equal_dollar_base_shares,
    evaluate_fixed_share_targets,
    simulate_fixed_share_targets,
    summarize_fixed_share_path,
)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 2.0]],
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        columns=["A", "B"],
    )


def _flags(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative = pd.DataFrame(False, index=index, columns=["A", "B"])
    positive = negative.copy()
    negative.loc[index[0], "A"] = True
    positive.loc[index[0], "B"] = True
    return negative, positive


def test_equal_dollar_shares_reserve_initial_cost() -> None:
    prices = _prices()
    shares = equal_dollar_base_shares(prices, prices.index[0], entry_cost_bps=10.0)
    notional = float((shares * prices.iloc[0]).sum())
    assert notional * 1.001 == pytest.approx(1.0)


def test_sparse_rotation_conserves_released_notional() -> None:
    prices = _prices()
    sessions = prices.index[:2]
    negative, positive = _flags(sessions)
    targets = build_sparse_extreme_targets(
        prices,
        sessions,
        negative,
        positive,
        risk_fraction=0.25,
        entry_cost_bps=0.0,
        reallocate_to_positive=True,
    )
    assert targets.loc[sessions[0], "A"] == pytest.approx(0.125)
    assert targets.loc[sessions[0], "B"] == pytest.approx(0.875)
    assert float((targets.loc[sessions[0]] * prices.loc[sessions[0]]).sum()) == pytest.approx(1.0)


def test_sparse_brake_leaves_released_notional_in_cash() -> None:
    prices = _prices()
    sessions = prices.index[:2]
    negative, positive = _flags(sessions)
    targets = build_sparse_extreme_targets(
        prices,
        sessions,
        negative,
        positive,
        risk_fraction=0.25,
        entry_cost_bps=0.0,
        reallocate_to_positive=False,
    )
    assert targets.loc[sessions[0], "A"] == pytest.approx(0.125)
    assert targets.loc[sessions[0], "B"] == pytest.approx(0.5)
    assert float((targets.loc[sessions[0]] * prices.loc[sessions[0]]).sum()) == pytest.approx(0.625)


def test_fixed_share_simulator_includes_price_move_cost_and_liquidation() -> None:
    prices = _prices().iloc[:2]
    sessions = prices.index[:1]
    flags = pd.DataFrame(False, index=sessions, columns=prices.columns)
    targets = build_sparse_extreme_targets(
        prices,
        sessions,
        flags,
        flags,
        risk_fraction=0.25,
        entry_cost_bps=0.0,
        reallocate_to_positive=False,
    )
    gross = simulate_fixed_share_targets(prices, targets, cost_bps_per_side=0.0)
    net = simulate_fixed_share_targets(prices, targets, cost_bps_per_side=10.0)
    assert gross.loc[0, "net_return"] == pytest.approx(0.5)
    assert net.loc[0, "net_return"] < gross.loc[0, "net_return"]
    assert net.loc[0, "turnover"] > 1.0


def test_evaluation_and_summary_keep_gross_and_net_separate() -> None:
    prices = _prices()
    sessions = prices.index[:2]
    flags = pd.DataFrame(False, index=sessions, columns=prices.columns)
    targets = build_sparse_extreme_targets(
        prices,
        sessions,
        flags,
        flags,
        risk_fraction=0.25,
        entry_cost_bps=10.0,
        reallocate_to_positive=False,
    )
    daily = evaluate_fixed_share_targets(prices, targets, cost_bps_per_side=10.0)
    summary = summarize_fixed_share_path(daily)
    assert np.all(daily["gross_return"] >= daily["net_return"])
    assert summary["total_return_gross"] > summary["total_return_net"]
    assert summary["n_sessions"] == 2
