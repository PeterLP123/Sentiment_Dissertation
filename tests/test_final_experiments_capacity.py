from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from final_experiments.lib.capacity import (
    CapacityError,
    ImpactConfig,
    build_lagged_liquidity,
    fold_terminal_liquidation,
    minimize_square_root_impact_weights,
    simulate_square_root_impact,
    square_root_impact_proxy,
)
from sentiment_benchmark.strategy_research.ledger import run_open_to_open_ledger
from sentiment_benchmark.strategy_research.market import OpenToOpenReturn
from sentiment_benchmark.strategy_research.portfolio import PositionTarget, TargetPortfolio


def _target(session: str, weights: dict[str, float]) -> TargetPortfolio:
    positions = tuple(
        PositionTarget(
            symbol=symbol,
            action=float(np.sign(weight)),
            volatility=None,
            raw_weight=weight,
            target_weight=weight,
        )
        for symbol, weight in sorted(weights.items())
    )
    long_exposure = sum(weight for weight in weights.values() if weight > 0)
    short_exposure = sum(-weight for weight in weights.values() if weight < 0)
    return TargetPortfolio(
        session=session,
        positions=positions,
        gross_exposure=long_exposure + short_exposure,
        net_exposure=long_exposure - short_exposure,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        cash_weight=1.0 - long_exposure + short_exposure,
    )


def _returns() -> tuple[OpenToOpenReturn, ...]:
    return (
        OpenToOpenReturn("AAA", "2026-01-02", "2026-01-05", 0.02),
        OpenToOpenReturn("BBB", "2026-01-02", "2026-01-05", -0.01),
        OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", -0.01),
        OpenToOpenReturn("BBB", "2026-01-05", "2026-01-06", 0.02),
    )


def _liquidity() -> pd.DataFrame:
    return pd.DataFrame(
        [(session, symbol, 100_000_000.0, 0.02) for session in ("2026-01-02", "2026-01-05", "2026-01-06") for symbol in ("AAA", "BBB")],
        columns=["session_date", "symbol", "adv20_usd", "sigma20"],
    )


def test_lagged_liquidity_excludes_current_session() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "session_date": dates,
            "symbol": "AAA",
            "open": [100.0, 110.0, 99.0, 99.0, 108.9],
            "close": 1.0,
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    result = build_lagged_liquidity(prices, window_sessions=2, minimum_observations=2).set_index("session_date")

    assert result.loc[dates[2], "adv20_usd"] == pytest.approx(15.0)
    assert np.isnan(result.loc[dates[2], "sigma20"])
    assert result.loc[dates[3], "adv20_usd"] == pytest.approx(25.0)
    assert result.loc[dates[3], "sigma20"] == pytest.approx(np.sqrt(0.02))


def test_zero_impact_reproduces_core_drift_ledger() -> None:
    targets = (
        _target("2026-01-02", {"AAA": 0.5, "BBB": -0.5}),
        _target("2026-01-05", {"AAA": -0.5, "BBB": 0.5}),
    )
    expected = run_open_to_open_ledger(
        targets,
        _returns(),
        cost_rate_per_side=0.001,
        initial_nav_usd=1_000_000.0,
        force_final_liquidation=True,
    )
    actual = simulate_square_root_impact(
        targets,
        _returns(),
        _liquidity(),
        config=ImpactConfig(impact_coefficient=0.0),
    ).daily

    for index, row in enumerate(expected):
        assert actual.loc[index, "gross_return"] == pytest.approx(row.gross_return)
        assert actual.loc[index, "fixed_cost"] == pytest.approx(row.transaction_cost)
        assert actual.loc[index, "impact_cost"] == 0.0
        assert actual.loc[index, "net_return"] == pytest.approx(row.net_return)
        assert actual.loc[index, "turnover"] == pytest.approx(row.turnover)
        assert actual.loc[index, "start_nav"] == pytest.approx(row.start_nav_usd)
        assert actual.loc[index, "end_nav"] == pytest.approx(row.end_nav_usd)


def test_larger_aum_increases_participation_and_impact_drag() -> None:
    targets = (
        _target("2026-01-02", {"AAA": 0.5, "BBB": -0.5}),
        _target("2026-01-05", {"AAA": -0.5, "BBB": 0.5}),
    )
    smaller = simulate_square_root_impact(
        targets,
        _returns(),
        _liquidity(),
        config=ImpactConfig(initial_nav_usd=1_000_000.0, impact_coefficient=1.0),
    )
    larger = simulate_square_root_impact(
        targets,
        _returns(),
        _liquidity(),
        config=ImpactConfig(initial_nav_usd=10_000_000.0, impact_coefficient=1.0),
    )

    assert larger.orders["participation"].max() > smaller.orders["participation"].max()
    assert larger.orders["impact_rate"].mean() > smaller.orders["impact_rate"].mean()
    assert fold_terminal_liquidation(larger.daily)["net_return"].sum() < fold_terminal_liquidation(smaller.daily)["net_return"].sum()


def test_nonzero_order_fails_closed_without_lagged_liquidity() -> None:
    target = (_target("2026-01-02", {"AAA": 1.0}),)
    returns = (OpenToOpenReturn("AAA", "2026-01-02", "2026-01-05", 0.0),)
    liquidity = pd.DataFrame(
        [("2026-01-05", "AAA", 100_000_000.0, 0.02)],
        columns=["session_date", "symbol", "adv20_usd", "sigma20"],
    )

    with pytest.raises(CapacityError, match="missing lagged liquidity"):
        simulate_square_root_impact(target, returns, liquidity)


def test_impact_minimizer_is_equal_when_inputs_are_equal() -> None:
    weights = minimize_square_root_impact_weights(
        {"AAA": 100.0, "BBB": 100.0},
        {"AAA": 0.02, "BBB": 0.02},
        total_weight=0.5,
    )

    assert weights == pytest.approx({"AAA": 0.25, "BBB": 0.25})


def test_impact_minimizer_water_fills_after_name_cap() -> None:
    weights = minimize_square_root_impact_weights(
        {"AAA": 100.0, "BBB": 1.0, "CCC": 1.0},
        {"AAA": 0.01, "BBB": 0.01, "CCC": 0.01},
        total_weight=0.5,
    )

    assert weights == pytest.approx({"AAA": 0.25, "BBB": 0.125, "CCC": 0.125})


def test_impact_minimizer_reduces_convex_proxy() -> None:
    adv = {"AAA": 500.0, "BBB": 100.0, "CCC": 50.0}
    volatility = {"AAA": 0.01, "BBB": 0.02, "CCC": 0.04}
    optimized = minimize_square_root_impact_weights(
        adv,
        volatility,
        total_weight=0.45,
    )
    equal = {symbol: 0.15 for symbol in adv}

    assert sum(optimized.values()) == pytest.approx(0.45)
    assert max(optimized.values()) <= 0.25
    assert square_root_impact_proxy(optimized, adv, volatility) < square_root_impact_proxy(equal, adv, volatility)


def test_impact_minimizer_fails_when_names_cannot_support_leg() -> None:
    with pytest.raises(CapacityError, match="cannot support"):
        minimize_square_root_impact_weights(
            {"AAA": 100.0},
            {"AAA": 0.02},
            total_weight=0.5,
        )
