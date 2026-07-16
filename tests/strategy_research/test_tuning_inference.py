from __future__ import annotations

from dataclasses import replace

import pytest

from sentiment_benchmark.strategy_research.inference import paired_block_bootstrap
from sentiment_benchmark.strategy_research.tuning import (
    CandidateGuardrails,
    FoldMetric,
    TuningError,
    annualized_sharpe,
    build_candidate_grid,
    build_chronological_split,
    build_expanding_folds,
    default_candidate_grid,
    evaluate_candidate,
    select_candidate,
)


def sessions(count: int) -> list[str]:
    return [f"S{index:03d}" for index in range(count)]


def fold_metric(index: int, sharpe: float | None, *, turnover: float = 0.1) -> FoldMetric:
    return FoldMetric(
        fold_index=index,
        sharpe=sharpe,
        average_turnover=turnover,
        active_days=20,
        supported_stocks=33,
    )


def test_exact_default_folds_are_expanding_non_overlapping_and_use_all_history() -> None:
    folds = build_expanding_folds(sessions(130))
    assert len(folds) == 3
    assert [len(fold.validation_sessions) for fold in folds] == [20, 20, 20]
    assert [len(fold.training_sessions) for fold in folds] == [70, 90, 110]
    assert folds[-1].validation_sessions[-1] == "S129"
    for fold in folds:
        assert set(fold.training_sessions).isdisjoint(fold.validation_sessions)
        assert fold.training_sessions[-1] < fold.validation_sessions[0]


def test_shorter_history_uses_largest_equal_defensible_folds() -> None:
    folds = build_expanding_folds(sessions(89))
    assert len(folds) == 2
    assert [len(fold.validation_sessions) for fold in folds] == [14, 14]
    assert len(folds[0].training_sessions) == 61


def test_formal_tuning_requires_two_minimum_validation_folds() -> None:
    with pytest.raises(TuningError, match="cannot support two"):
        build_expanding_folds(sessions(79))


def test_explicit_master_boundary_is_applied_without_moving_it() -> None:
    split = build_chronological_split(sessions(10), "S007")
    assert split.development_sessions == tuple(sessions(7))
    assert split.evaluation_sessions == ("S007", "S008", "S009")


def test_declared_grid_has_45_stably_ordered_candidates() -> None:
    grid = default_candidate_grid()
    assert len(grid) == 45
    assert len({candidate.identity for candidate in grid}) == 45
    assert grid[0].identity[:3] == (1.0, 0.5, 0.0)
    assert grid[-1].identity[:3] == (10.0, 0.9, 0.2)


def test_custom_smoke_grid_is_a_stable_cartesian_product() -> None:
    grid = build_candidate_grid(
        half_life_sessions=(2, 3),
        state_scale_quantiles=(0.75,),
        no_trade_bands=(0.0, 0.1),
    )
    assert [candidate.identity[:3] for candidate in grid] == [
        (2.0, 0.75, 0.0),
        (2.0, 0.75, 0.1),
        (3.0, 0.75, 0.0),
        (3.0, 0.75, 0.1),
    ]


def test_objective_is_median_minus_half_iqr() -> None:
    candidate = default_candidate_grid()[0]
    result = evaluate_candidate(candidate, [fold_metric(0, 1), fold_metric(1, 2), fold_metric(2, 3)])
    assert result.median_sharpe == 2
    assert result.sharpe_iqr == 1
    assert result.objective == 1.5
    assert result.valid


def test_undefined_sharpe_and_guardrail_failures_are_retained_as_invalid() -> None:
    candidate = default_candidate_grid()[0]
    invalid = FoldMetric(
        fold_index=0,
        sharpe=None,
        average_turnover=2,
        active_days=1,
        supported_stocks=2,
        exposure_violations=1,
        valid_price_coverage=False,
    )
    result = evaluate_candidate(
        candidate,
        [invalid],
        guardrails=CandidateGuardrails(maximum_average_turnover=1),
    )
    assert not result.valid
    assert result.objective is None
    assert len(result.rejection_reasons) == 6


def test_selection_tie_breaks_dispersion_then_turnover_then_complexity_then_identity() -> None:
    first, second = default_candidate_grid()[:2]
    base = evaluate_candidate(first, [fold_metric(0, 1), fold_metric(1, 1), fold_metric(2, 1)])
    higher_turnover = replace(base, candidate=second, average_turnover=0.2)
    assert select_candidate([higher_turnover, base]) == base

    simpler = replace(base, candidate=replace(second, complexity_rank=0))
    complex_result = replace(base, candidate=replace(first, complexity_rank=1))
    assert select_candidate([complex_result, simpler]) == simpler


def test_all_candidates_invalid_stops_selection() -> None:
    invalid = evaluate_candidate(default_candidate_grid()[0], [fold_metric(0, None)])
    with pytest.raises(TuningError, match="all strategy candidates"):
        select_candidate([invalid])


def test_annualized_sharpe_handles_zero_volatility_safely() -> None:
    assert annualized_sharpe([0.01, 0.01]) is None
    assert annualized_sharpe([0.01, -0.01]) == pytest.approx(0)


def test_paired_block_bootstrap_is_date_aligned_and_deterministic() -> None:
    strategy = {f"2026-01-{day:02d}": 0.01 + day / 100_000 for day in range(1, 21)}
    comparator = {date: 0.0 for date in strategy}
    first = paired_block_bootstrap(strategy, comparator, block_length=5, replications=200, seed=20260715)
    second = paired_block_bootstrap(strategy, comparator, block_length=5, replications=200, seed=20260715)
    assert first == second
    assert first.mean_direction == "positive"
    assert first.confidence_direction == "positive"
    assert len(first.effective_dates) == 20


def test_paired_bootstrap_rejects_unmatched_dates() -> None:
    with pytest.raises(ValueError, match="identical dates"):
        paired_block_bootstrap({"2026-01-01": 0.1}, {"2026-01-02": 0.1}, block_length=1)
