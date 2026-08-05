from __future__ import annotations

import pandas as pd
import pytest

from final_experiments.lib.sparse_spread import (
    build_exact_extrema_targets,
    sector_neutralize_targets,
    sector_project_targets,
)
from sentiment_benchmark.strategy_research.portfolio import PositionTarget, TargetPortfolio


def _panel(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["session_date", "symbol", "strongest_event"])


def _target(weights: dict[str, float]) -> TargetPortfolio:
    long_exposure = sum(weight for weight in weights.values() if weight > 0)
    short_exposure = sum(-weight for weight in weights.values() if weight < 0)
    return TargetPortfolio(
        session="2026-01-02",
        positions=tuple(
            PositionTarget(
                symbol=symbol,
                action=float((weight > 0) - (weight < 0)),
                volatility=None,
                raw_weight=weight,
                target_weight=weight,
            )
            for symbol, weight in weights.items()
        ),
        gross_exposure=long_exposure + short_exposure,
        net_exposure=long_exposure - short_exposure,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        cash_weight=1.0 - long_exposure + short_exposure,
    )


def test_exact_extrema_targets_are_equal_leg_and_cash_when_one_sided() -> None:
    panel = _panel(
        [
            ("2026-01-02", "AAA", 1.0),
            ("2026-01-02", "BBB", -1.0),
            ("2026-01-02", "CCC", 0.2),
            ("2026-01-05", "AAA", 1.0),
            ("2026-01-05", "BBB", 0.0),
            ("2026-01-05", "CCC", float("nan")),
        ]
    )

    targets, audit = build_exact_extrema_targets(panel)

    assert targets[0].weights() == pytest.approx({"AAA": 0.5, "BBB": -0.5, "CCC": 0.0})
    assert targets[0].gross_exposure == pytest.approx(1.0)
    assert targets[0].net_exposure == pytest.approx(0.0)
    assert targets[1].weights() == pytest.approx({"AAA": 0.0, "BBB": 0.0, "CCC": 0.0})
    assert audit["eligible"].tolist() == [True, False]
    assert audit["max_abs_name_weight"].tolist() == pytest.approx([0.5, 0.0])


def test_exact_extrema_targets_split_each_leg_equally() -> None:
    panel = _panel(
        [
            ("2026-01-02", "AAA", 1.0),
            ("2026-01-02", "BBB", 1.0),
            ("2026-01-02", "CCC", -1.0),
            ("2026-01-02", "DDD", -1.0),
        ]
    )

    targets, audit = build_exact_extrema_targets(panel)

    assert targets[0].weights() == pytest.approx(
        {"AAA": 0.25, "BBB": 0.25, "CCC": -0.25, "DDD": -0.25}
    )
    assert audit.loc[0, "long_hhi"] == pytest.approx(0.5)
    assert audit.loc[0, "short_hhi"] == pytest.approx(0.5)


def test_two_name_minimum_fails_closed() -> None:
    panel = _panel(
        [
            ("2026-01-02", "AAA", 1.0),
            ("2026-01-02", "BBB", -1.0),
        ]
    )

    targets, audit = build_exact_extrema_targets(panel, minimum_names_per_side=2)

    assert targets[0].gross_exposure == 0.0
    assert not bool(audit.loc[0, "eligible"])


def test_single_name_cap_scales_both_legs_to_sparse_side_capacity() -> None:
    panel = _panel(
        [
            ("2026-01-02", "AAA", 1.0),
            ("2026-01-02", "BBB", -1.0),
            ("2026-01-02", "CCC", -1.0),
        ]
    )

    targets, audit = build_exact_extrema_targets(panel, single_name_cap=0.25)

    assert targets[0].weights() == pytest.approx(
        {"AAA": 0.25, "BBB": -0.125, "CCC": -0.125}
    )
    assert targets[0].gross_exposure == pytest.approx(0.5)
    assert targets[0].net_exposure == pytest.approx(0.0)
    assert audit.loc[0, "leg_exposure"] == pytest.approx(0.25)
    assert audit.loc[0, "unused_gross_capacity"] == pytest.approx(0.5)
    assert audit.loc[0, "max_abs_name_weight"] == pytest.approx(0.25)


def test_single_name_cap_preserves_full_gross_when_both_legs_have_capacity() -> None:
    panel = _panel(
        [
            ("2026-01-02", "AAA", 1.0),
            ("2026-01-02", "BBB", 1.0),
            ("2026-01-02", "CCC", -1.0),
            ("2026-01-02", "DDD", -1.0),
        ]
    )

    targets, audit = build_exact_extrema_targets(panel, single_name_cap=0.25)

    assert targets[0].gross_exposure == pytest.approx(1.0)
    assert audit.loc[0, "unused_gross_capacity"] == pytest.approx(0.0)
    assert audit.loc[0, "max_abs_name_weight"] == pytest.approx(0.25)


@pytest.mark.parametrize("cap", [0.0, -0.1, 0.51, float("nan")])
def test_single_name_cap_is_bounded(cap: float) -> None:
    panel = _panel(
        [
            ("2026-01-02", "AAA", 1.0),
            ("2026-01-02", "BBB", -1.0),
        ]
    )
    with pytest.raises(ValueError, match="single_name_cap"):
        build_exact_extrema_targets(panel, single_name_cap=cap)


def test_sector_neutralization_adds_equal_weight_hedges() -> None:
    target = _target(
        {
            "AAA": 0.25,
            "AAB": 0.0,
            "AAC": 0.0,
            "AAD": 0.0,
            "BBA": -0.25,
            "BBB": 0.0,
            "BBC": 0.0,
            "BBD": 0.0,
        }
    )
    sectors = {
        "AAA": "a",
        "AAB": "a",
        "AAC": "a",
        "AAD": "a",
        "BBA": "b",
        "BBB": "b",
        "BBC": "b",
        "BBD": "b",
    }

    neutral, audit = sector_neutralize_targets((target,), sector_by_symbol=sectors)

    assert neutral[0].weights() == pytest.approx(
        {
            "AAA": 0.1875,
            "AAB": -0.0625,
            "AAC": -0.0625,
            "AAD": -0.0625,
            "BBA": -0.1875,
            "BBB": 0.0625,
            "BBC": 0.0625,
            "BBD": 0.0625,
        }
    )
    assert neutral[0].gross_exposure == pytest.approx(0.75)
    assert neutral[0].net_exposure == pytest.approx(0.0)
    assert audit.loc[0, "base_max_abs_sector_exposure"] == pytest.approx(0.25)
    assert audit.loc[0, "neutral_max_abs_sector_exposure"] == pytest.approx(0.0)
    assert audit.loc[0, "hedge_names"] == 6


def test_sector_neutralization_scales_uniformly_to_both_limits() -> None:
    target = _target(
        {
            "AAA": 0.5,
            "AAB": 0.0,
            "AAC": 0.0,
            "AAD": 0.0,
            "BBA": -0.5,
            "BBB": 0.0,
            "BBC": 0.0,
            "BBD": 0.0,
        }
    )
    sectors = {symbol: symbol[0] for symbol in target.weights()}

    neutral, audit = sector_neutralize_targets((target,), sector_by_symbol=sectors)

    assert neutral[0].weights()["AAA"] == pytest.approx(0.25)
    assert neutral[0].weights()["AAB"] == pytest.approx(-1.0 / 12.0)
    assert neutral[0].weights()["BBA"] == pytest.approx(-0.25)
    assert neutral[0].weights()["BBB"] == pytest.approx(1.0 / 12.0)
    assert neutral[0].gross_exposure == pytest.approx(1.0)
    assert audit.loc[0, "scale_multiplier"] == pytest.approx(2.0 / 3.0)
    assert audit.loc[0, "neutral_max_name_weight"] == pytest.approx(0.25)


def test_sector_neutralization_preserves_balanced_sector_and_flat_session() -> None:
    active = _target({"AAA": 0.25, "AAB": -0.25})
    flat = TargetPortfolio(
        session="2026-01-05",
        positions=tuple(
            PositionTarget(symbol=symbol, action=0.0, volatility=None, raw_weight=0.0, target_weight=0.0)
            for symbol in ("AAA", "AAB")
        ),
        gross_exposure=0.0,
        net_exposure=0.0,
        long_exposure=0.0,
        short_exposure=0.0,
        cash_weight=1.0,
    )

    neutral, audit = sector_neutralize_targets(
        (active, flat), sector_by_symbol={"AAA": "a", "AAB": "a"}
    )

    assert neutral[0].weights() == pytest.approx(active.weights())
    assert neutral[1].gross_exposure == pytest.approx(0.0)
    assert audit["hedge_names"].tolist() == [0, 0]


def test_sector_neutralization_fails_closed_on_bad_mapping_or_cap() -> None:
    target = _target({"AAA": 0.25, "AAB": -0.25})

    with pytest.raises(ValueError, match="missing symbols"):
        sector_neutralize_targets((target,), sector_by_symbol={"AAA": "a"})
    with pytest.raises(ValueError, match="at least two symbols"):
        sector_neutralize_targets(
            (target,), sector_by_symbol={"AAA": "a", "AAB": "b"}
        )
    with pytest.raises(ValueError, match="single_name_cap"):
        sector_neutralize_targets(
            (target,), sector_by_symbol={"AAA": "a", "AAB": "a"}, single_name_cap=0.0
        )


def test_sector_projection_spreads_sector_exposure_across_constituents() -> None:
    target = _target(
        {
            "AAA": 0.25,
            "AAB": 0.0,
            "AAC": 0.0,
            "AAD": 0.0,
            "BBA": -0.25,
            "BBB": 0.0,
            "BBC": 0.0,
            "BBD": 0.0,
        }
    )
    sectors = {symbol: symbol[0] for symbol in target.weights()}

    projected, audit = sector_project_targets((target,), sector_by_symbol=sectors)

    assert projected[0].weights() == pytest.approx(
        {
            "AAA": 0.0625,
            "AAB": 0.0625,
            "AAC": 0.0625,
            "AAD": 0.0625,
            "BBA": -0.0625,
            "BBB": -0.0625,
            "BBC": -0.0625,
            "BBD": -0.0625,
        }
    )
    assert projected[0].gross_exposure == pytest.approx(0.5)
    assert projected[0].net_exposure == pytest.approx(0.0)
    assert audit.loc[0, "active_sectors"] == 2
    assert audit.loc[0, "non_signal_peer_names"] == 6


def test_sector_projection_removes_within_sector_opposition() -> None:
    target = _target({"AAA": 0.25, "AAB": -0.25})

    projected, audit = sector_project_targets(
        (target,), sector_by_symbol={"AAA": "a", "AAB": "a"}
    )

    assert projected[0].gross_exposure == pytest.approx(0.0)
    assert audit.loc[0, "projected_names"] == 0
    assert audit.loc[0, "active_sectors"] == 0


def test_sector_projection_scales_down_but_never_up() -> None:
    target = _target(
        {
            "AAA": 1.0,
            "AAB": 0.0,
            "AAC": 0.0,
            "AAD": 0.0,
            "BBA": -1.0,
            "BBB": 0.0,
            "BBC": 0.0,
            "BBD": 0.0,
        }
    )
    sectors = {symbol: symbol[0] for symbol in target.weights()}

    projected, audit = sector_project_targets((target,), sector_by_symbol=sectors)

    assert projected[0].gross_exposure == pytest.approx(1.0)
    assert max(map(abs, projected[0].weights().values())) == pytest.approx(0.125)
    assert audit.loc[0, "scale_multiplier"] == pytest.approx(0.5)


def test_sector_projection_fails_closed_on_bad_input() -> None:
    target = _target({"AAA": 0.25, "AAB": -0.25})
    non_neutral = _target({"AAA": 0.25, "AAB": 0.0})

    with pytest.raises(ValueError, match="missing symbols"):
        sector_project_targets((target,), sector_by_symbol={"AAA": "a"})
    with pytest.raises(ValueError, match="at least two symbols"):
        sector_project_targets((target,), sector_by_symbol={"AAA": "a", "AAB": "b"})
    with pytest.raises(ValueError, match="dollar-neutral"):
        sector_project_targets(
            (non_neutral,), sector_by_symbol={"AAA": "a", "AAB": "a"}
        )


def test_exact_extrema_targets_reject_unbalanced_or_duplicate_panels() -> None:
    unbalanced = _panel(
        [
            ("2026-01-02", "AAA", 1.0),
            ("2026-01-02", "BBB", -1.0),
            ("2026-01-05", "AAA", 1.0),
        ]
    )
    with pytest.raises(ValueError, match="same symbol population"):
        build_exact_extrema_targets(unbalanced)

    duplicate = pd.concat([unbalanced.iloc[:2], unbalanced.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        build_exact_extrema_targets(duplicate)
