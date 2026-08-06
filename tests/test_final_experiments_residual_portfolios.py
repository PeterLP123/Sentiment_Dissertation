from __future__ import annotations

import numpy as np
import pytest

from final_experiments.lib.residual_portfolios import (
    build_exposure_residual_targets,
)
from sentiment_benchmark.strategy_research.portfolio import (
    PositionTarget,
    TargetPortfolio,
)

SYMBOLS = ("AAA", "AAB", "BBA", "BBB")
SECTORS = {"AAA": "a", "AAB": "a", "BBA": "b", "BBB": "b"}


def _target(session: str, weights: tuple[float, ...]) -> TargetPortfolio:
    positions = tuple(
        PositionTarget(
            symbol=symbol,
            action=float(np.sign(weight)),
            volatility=None,
            raw_weight=weight,
            target_weight=weight,
            exclusion_reason=None,
        )
        for symbol, weight in zip(SYMBOLS, weights, strict=True)
    )
    return TargetPortfolio(
        session=session,
        positions=positions,
        gross_exposure=float(sum(weights)),
        net_exposure=float(sum(weights)),
        long_exposure=float(sum(weights)),
        short_exposure=0.0,
        cash_weight=1.0 - float(sum(weights)),
    )


def _weights(target: TargetPortfolio) -> np.ndarray:
    return np.asarray([position.target_weight for position in target.positions])


def test_market_residual_matches_selected_minus_equal_weight_control() -> None:
    selected = (_target("2026-01-02", (0.25, 0.0, 0.125, 0.0)),)
    targets, audit = build_exposure_residual_targets(selected, mode="market")
    result = _weights(targets[0])
    control = np.full(4, 0.375 / 4)
    returns = np.asarray([0.02, -0.01, 0.03, 0.01])

    np.testing.assert_allclose(result, np.asarray([0.25, 0.0, 0.125, 0.0]) - control)
    assert np.isclose(result @ returns, np.asarray([0.25, 0.0, 0.125, 0.0]) @ returns - control @ returns)
    assert np.isclose(result.sum(), 0.0)
    assert audit.loc[0, "exposure_match_error"] == pytest.approx(0.0)


def test_sector_residual_is_neutral_inside_each_sector() -> None:
    selected = (_target("2026-01-02", (0.25, 0.0, 0.125, 0.0)),)
    targets, audit = build_exposure_residual_targets(selected, mode="sector", sector_by_symbol=SECTORS)
    result = _weights(targets[0])
    expected = np.asarray([0.125, -0.125, 0.0625, -0.0625])
    returns = np.asarray([0.02, -0.01, 0.03, 0.01])
    control = np.asarray([0.125, 0.125, 0.0625, 0.0625])

    np.testing.assert_allclose(result, expected)
    assert np.isclose(result[:2].sum(), 0.0)
    assert np.isclose(result[2:].sum(), 0.0)
    assert np.isclose(result @ returns, np.asarray([0.25, 0.0, 0.125, 0.0]) @ returns - control @ returns)
    assert audit.loc[0, "max_abs_sector_exposure"] == pytest.approx(0.0)


def test_inactive_target_remains_cash() -> None:
    selected = (_target("2026-01-02", (0.0, 0.0, 0.0, 0.0)),)
    targets, audit = build_exposure_residual_targets(selected, mode="market")

    np.testing.assert_array_equal(_weights(targets[0]), np.zeros(4))
    assert targets[0].gross_exposure == 0.0
    assert audit.loc[0, "active_names"] == 0


def test_invalid_mode_short_weight_and_sector_mapping_fail_closed() -> None:
    valid = (_target("2026-01-02", (0.25, 0.0, 0.125, 0.0)),)
    with pytest.raises(ValueError, match="mode"):
        build_exposure_residual_targets(valid, mode="other")
    with pytest.raises(ValueError, match="sector mapping missing"):
        build_exposure_residual_targets(valid, mode="sector", sector_by_symbol={"AAA": "a"})

    short = (_target("2026-01-02", (0.25, -0.01, 0.125, 0.0)),)
    with pytest.raises(ValueError, match="cannot contain short"):
        build_exposure_residual_targets(short, mode="market")
