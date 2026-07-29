"""Focused synthetic tests for the Sharpe-driven signal-combination path.

Every fixture in this module is built inline from numpy, so nothing here reads
the licensed 33-stock panel or writes immutable study directories.
``build_signal_panel`` and ``run_signal_portfolio`` are therefore exercised by
the frozen run receipts, while the pure panel pivot and selection algebra are
covered here with synthetic data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sentiment_benchmark import signal_portfolio as signal_portfolio_module
from sentiment_benchmark.signal_portfolio import (
    MAX_EXHAUSTIVE_SIGNALS,
    SignalPortfolioConfig,
    SignalPortfolioError,
    best_subset_of_size,
    bootstrap_subset_sharpe,
    breakeven_cost_bps_per_side,
    build_signal_panel,
    certify_add_drop_stability,
    enumerate_subset_sharpes,
    equal_weight_returns,
    panel_return_matrix,
    run_signal_portfolio,
    selection_null_distribution,
    sharpe_standard_error,
    solve_two_signal_weights,
    stepwise_search,
)
from sentiment_benchmark.week6_pnl import annualized_sharpe

PERIODS_PER_YEAR = 252
# A coarse grid keeps the 100k-point default out of the test suite while still
# resolving the interior optimum to far better than the 1e-3 tolerance asserted.
GRID_LIMIT = 2.0
GRID_POINTS = 4001


def _pair_frame(
    seed: int,
    means: tuple[float, float],
    volatilities: tuple[float, float],
    correlation: float,
    *,
    rows: int = 90,
) -> pd.DataFrame:
    """Two daily return series whose sample mean and sample volatility are exact by construction.

    Each column is standardised before rescaling, so ``mean`` and ``volatilities``
    are the realised sample moments rather than population targets. Correlation
    is invariant to that per-column affine map, so it lands near ``correlation``
    without being pinned to it.
    """

    factor = np.linalg.cholesky(np.array([[1.0, correlation], [correlation, 1.0]]))
    shaped = np.random.default_rng(seed).standard_normal((rows, 2)) @ factor.T
    standardised = (shaped - shaped.mean(axis=0)) / shaped.std(axis=0, ddof=1)
    values = standardised * np.asarray(volatilities, dtype=float) + np.asarray(means, dtype=float)
    return pd.DataFrame(values, columns=["alpha", "beta"], index=pd.RangeIndex(rows))


def _triple_frame(seed: int, rows: int) -> pd.DataFrame:
    """Three loosely correlated daily return series with one deliberately negative drift."""

    drift = np.array([0.0015, 0.0008, -0.0004])
    values = np.random.default_rng(seed).standard_normal((rows, 3)) * 0.01 + drift
    return pd.DataFrame(values, columns=["s1", "s2", "s3"], index=pd.RangeIndex(rows))


def _tangency_weight_a(development: pd.DataFrame, pair: tuple[str, str]) -> tuple[float, float]:
    """Independent reference for ``Sigma^-1 mu / (1' Sigma^-1 mu)``, solved rather than inverted."""

    sample = development[list(pair)].to_numpy(dtype=float)
    direction = np.linalg.solve(np.cov(sample, rowvar=False, ddof=1), sample.mean(axis=0))
    denominator = float(direction.sum())
    return float(direction[0] / denominator), denominator


def _two_asset_closed_form(development: pd.DataFrame, pair: tuple[str, str]) -> float:
    """The scalar two-asset tangency weight, written out from the sample moments.

    ``w_a = (mu_a s_b^2 - mu_b rho s_a s_b) / (mu_a s_b^2 + mu_b s_a^2 - rho s_a s_b (mu_a + mu_b))``.
    """

    sample = development[list(pair)].to_numpy(dtype=float)
    mean = sample.mean(axis=0)
    covariance = np.cov(sample, rowvar=False, ddof=1)
    deviation_a, deviation_b = float(np.sqrt(covariance[0, 0])), float(np.sqrt(covariance[1, 1]))
    correlation = float(covariance[0, 1] / (deviation_a * deviation_b))
    numerator = mean[0] * deviation_b**2 - mean[1] * correlation * deviation_a * deviation_b
    denominator = mean[0] * deviation_b**2 + mean[1] * deviation_a**2 - correlation * deviation_a * deviation_b * (mean[0] + mean[1])
    return float(numerator / denominator)


# Three names, hand-computed so that ("a", "b") is add/drop stable, ("a",) is
# destabilised by a single addition, and ("c",) is a vacuously removal-stable
# singleton. Every subset the certifier can reach is present.
STABILITY_OBJECTIVE: dict[tuple[str, ...], float] = {
    ("a",): 1.0,
    ("b",): 0.9,
    ("c",): 1.5,
    ("a", "b"): 1.2,
    ("a", "c"): 0.6,
    ("b", "c"): 0.7,
    ("a", "b", "c"): 1.0,
}

# Four names, all Sharpe values distinct, unique global optimum ("a", "b") = 1.30.
SWEEP_OBJECTIVE: dict[tuple[str, ...], float] = {
    ("a",): 1.00,
    ("b",): 0.90,
    ("c",): 0.40,
    ("d",): 0.50,
    ("a", "b"): 1.30,
    ("a", "c"): 0.70,
    ("a", "d"): 0.80,
    ("b", "c"): 0.60,
    ("b", "d"): 0.65,
    ("c", "d"): 0.45,
    ("a", "b", "c"): 1.10,
    ("a", "b", "d"): 1.20,
    ("a", "c", "d"): 0.75,
    ("b", "c", "d"): 0.70,
    ("a", "b", "c", "d"): 1.05,
}
SWEEP_NAMES = ("a", "b", "c", "d")

# ("a", "b") and ("a", "b", "c") are tied at 1.30. A hill climb that accepted
# ties would oscillate between them; strict improvement must refuse the step.
PLATEAU_OBJECTIVE: dict[tuple[str, ...], float] = {
    ("a",): 1.00,
    ("b",): 0.90,
    ("c",): 0.40,
    ("a", "b"): 1.30,
    ("a", "c"): 0.70,
    ("b", "c"): 0.60,
    ("a", "b", "c"): 1.30,
}


def _config(**overrides: Any) -> SignalPortfolioConfig:
    payload: dict[str, Any] = {"run_id": "unit_test", "scorer_ids": ("headline/a", "headline/b")}
    payload.update(overrides)
    return SignalPortfolioConfig(**payload)


# --------------------------------------------------------------------------- #
# Part 4: Lagrange weights for two signals
# --------------------------------------------------------------------------- #


def test_two_signal_weights_match_analytic_tangency_and_sum_to_the_budget() -> None:
    development = _pair_frame(11, (0.0020, 0.0012), (0.010, 0.014), 0.25)
    evaluation = _pair_frame(12, (0.0015, 0.0010), (0.011, 0.013), 0.30)
    solution = solve_two_signal_weights(
        development,
        evaluation,
        "alpha",
        "beta",
        periods_per_year=PERIODS_PER_YEAR,
        annual_risk_free_rate=0.0,
        grid_limit=GRID_LIMIT,
        grid_points=GRID_POINTS,
    )
    expected_weight_a, denominator = _tangency_weight_a(development, ("alpha", "beta"))

    assert solution.budget_denominator == pytest.approx(denominator)
    assert solution.budget_denominator > 0
    assert solution.weight_a == pytest.approx(expected_weight_a)
    assert solution.weight_a_closed_form == pytest.approx(solution.weight_a)
    assert solution.closed_form_agrees
    # A third, independent route to the same weight: the two-asset closed form
    # written out in terms of the sample moments. This keeps the closed-form
    # cross-check from resting on the module's own agreement flag.
    assert solution.weight_a_closed_form == pytest.approx(_two_asset_closed_form(development, ("alpha", "beta")))
    assert solution.weight_a + solution.weight_b == pytest.approx(1.0)
    # Sharpe is scale invariant, so the sum-to-one budget is never binding.
    assert solution.lagrange_multiplier == 0.0
    # Sample moments are exact by construction in _pair_frame.
    assert (solution.mean_a, solution.mean_b) == (pytest.approx(0.0020), pytest.approx(0.0012))
    assert (solution.volatility_a, solution.volatility_b) == (pytest.approx(0.010), pytest.approx(0.014))
    assert solution.correlation == pytest.approx(float(np.corrcoef(development.to_numpy(dtype=float), rowvar=False)[0, 1]))


def test_positive_budget_denominator_makes_the_stationary_point_the_constrained_maximum() -> None:
    development = _pair_frame(11, (0.0020, 0.0012), (0.010, 0.014), 0.25)
    evaluation = _pair_frame(12, (0.0015, 0.0010), (0.011, 0.013), 0.30)
    solution = solve_two_signal_weights(
        development,
        evaluation,
        "alpha",
        "beta",
        periods_per_year=PERIODS_PER_YEAR,
        annual_risk_free_rate=0.0,
        grid_limit=GRID_LIMIT,
        grid_points=GRID_POINTS,
    )

    assert solution.budget_denominator > 0
    assert solution.stationary_point_is_maximum
    assert solution.optimal_sharpe is not None
    assert solution.equal_weight_sharpe is not None
    assert solution.best_single_sharpe is not None
    # Optimising over the budget line can only weakly beat two feasible points on it.
    assert solution.optimal_sharpe >= solution.equal_weight_sharpe
    assert solution.optimal_sharpe >= solution.best_single_sharpe
    # The brute-force grid is the independent confirmation that this really is the maximiser.
    assert solution.grid_best_sharpe == pytest.approx(solution.optimal_sharpe, abs=1e-3)
    assert solution.grid_best_weight_a == pytest.approx(solution.weight_a, abs=2.0 * GRID_LIMIT / (GRID_POINTS - 1))
    # sqrt(mu' Sigma^-1 mu), annualised, is the tangency Sharpe when the denominator is positive.
    sample = development.to_numpy(dtype=float)
    mean = sample.mean(axis=0)
    quadratic = float(mean @ np.linalg.solve(np.cov(sample, rowvar=False, ddof=1), mean))
    assert solution.optimal_sharpe == pytest.approx(math.sqrt(PERIODS_PER_YEAR) * math.sqrt(quadratic))


def test_frozen_development_weights_are_reported_on_evaluation_without_reoptimising() -> None:
    development = _pair_frame(11, (0.0020, 0.0012), (0.010, 0.014), 0.25)
    evaluation = _pair_frame(12, (0.0015, 0.0010), (0.011, 0.013), 0.30)
    solution = solve_two_signal_weights(
        development,
        evaluation,
        "alpha",
        "beta",
        periods_per_year=PERIODS_PER_YEAR,
        annual_risk_free_rate=0.0,
        grid_limit=GRID_LIMIT,
        grid_points=GRID_POINTS,
    )
    weights = np.array([solution.weight_a, solution.weight_b], dtype=float)
    expected = annualized_sharpe(
        evaluation[["alpha", "beta"]].to_numpy(dtype=float) @ weights,
        periods_per_year=PERIODS_PER_YEAR,
        annual_risk_free_rate=0.0,
    )

    assert solution.evaluation_sharpe_at_frozen_weights == pytest.approx(expected)
    # A feasible interior stationary maximum is also the long-only KKT solution.
    assert 0.0 <= solution.weight_a <= 1.0
    assert solution.long_only_weight_a == pytest.approx(solution.weight_a)
    assert solution.long_only_sharpe == pytest.approx(solution.optimal_sharpe)


def test_negative_budget_denominator_flags_a_minimum_and_the_grid_beats_the_stationary_point() -> None:
    """With equal volatilities the denominator's sign is the sign of ``mu_a + mu_b``.

    A clearly negative-drift second signal therefore drives ``1' Sigma^-1 mu``
    below zero, where the normalised stationary point is the constrained
    *minimum*. A naive solver would report it as the optimum.
    """

    grid_limit = 5.0
    development = _pair_frame(21, (0.0020, -0.0060), (0.010, 0.010), 0.30)
    evaluation = _pair_frame(22, (0.0018, -0.0055), (0.010, 0.010), 0.30)
    solution = solve_two_signal_weights(
        development,
        evaluation,
        "alpha",
        "beta",
        periods_per_year=PERIODS_PER_YEAR,
        annual_risk_free_rate=0.0,
        grid_limit=grid_limit,
        grid_points=2001,
    )
    _, denominator = _tangency_weight_a(development, ("alpha", "beta"))

    assert denominator < 0
    assert solution.budget_denominator == pytest.approx(denominator)
    assert solution.stationary_point_is_maximum is False
    assert solution.optimal_sharpe is not None
    # The supremum lies strictly beyond the stationary point, which is a minimum.
    assert solution.grid_best_sharpe > solution.optimal_sharpe
    # No interior maximum exists: the grid optimum sits on the leverage boundary.
    assert abs(solution.grid_best_weight_a) == pytest.approx(grid_limit)
    # The algebra itself is still correct; only its interpretation changes.
    assert solution.closed_form_agrees
    assert solution.weight_a + solution.weight_b == pytest.approx(1.0)
    assert solution.lagrange_multiplier == 0.0
    # The stationary point is the constrained minimum, so the inequalities of the
    # positive-denominator case are reversed: it is the worst point on the line.
    assert solution.equal_weight_sharpe is not None
    assert solution.best_single_sharpe is not None
    assert solution.optimal_sharpe <= solution.equal_weight_sharpe
    assert solution.optimal_sharpe <= solution.best_single_sharpe
    # Long-only cannot use the infeasible stationary point, so it holds alpha alone.
    assert solution.long_only_weight_a == pytest.approx(1.0)
    assert solution.long_only_sharpe == pytest.approx(solution.best_single_sharpe)


def test_long_only_collapses_to_a_corner_when_a_feasible_stationary_point_is_a_minimum() -> None:
    """Two negative-drift signals put the stationary weight inside ``[0, 1]`` while ``d < 0``.

    Feasibility alone is therefore not enough to justify holding the tangency
    portfolio: the second-order condition has to be checked as well. This is the
    configuration where the real run's long-only KKT solution collapsed to
    ``[1, 0]``.
    """

    development = _pair_frame(51, (-0.0020, -0.0025), (0.010, 0.010), 0.5)
    evaluation = _pair_frame(52, (-0.0018, -0.0022), (0.010, 0.010), 0.5)
    solution = solve_two_signal_weights(
        development,
        evaluation,
        "alpha",
        "beta",
        periods_per_year=PERIODS_PER_YEAR,
        annual_risk_free_rate=0.0,
        grid_limit=5.0,
        grid_points=2001,
    )

    assert solution.budget_denominator < 0
    assert solution.stationary_point_is_maximum is False
    # The stationary weight is feasible for a long-only book, and still must not be held.
    assert 0.0 < solution.weight_a < 1.0
    assert solution.optimal_sharpe is not None
    assert solution.best_single_sharpe is not None
    assert solution.optimal_sharpe < solution.best_single_sharpe
    assert solution.long_only_weight_a in {0.0, 1.0}
    assert solution.long_only_sharpe == pytest.approx(solution.best_single_sharpe)
    assert solution.grid_best_sharpe > solution.optimal_sharpe


@pytest.mark.parametrize(
    ("signal_a", "signal_b"),
    [("alpha", "alpha"), ("alpha", "gamma"), ("gamma", "beta")],
)
def test_two_signal_weights_reject_degenerate_signal_pairs(signal_a: str, signal_b: str) -> None:
    development = _pair_frame(11, (0.0020, 0.0012), (0.010, 0.014), 0.25)

    with pytest.raises(SignalPortfolioError):
        solve_two_signal_weights(
            development,
            development,
            signal_a,
            signal_b,
            periods_per_year=PERIODS_PER_YEAR,
            annual_risk_free_rate=0.0,
            grid_limit=GRID_LIMIT,
            grid_points=101,
        )


def test_two_signal_weights_refuse_perfectly_collinear_series() -> None:
    collinear = pd.DataFrame({"alpha": [0.01, -0.02, 0.03, 0.005, -0.01], "beta": [0.02, -0.04, 0.06, 0.010, -0.02]})

    with pytest.raises(SignalPortfolioError, match="singular"):
        solve_two_signal_weights(
            collinear,
            collinear,
            "alpha",
            "beta",
            periods_per_year=PERIODS_PER_YEAR,
            annual_risk_free_rate=0.0,
            grid_limit=GRID_LIMIT,
            grid_points=101,
        )


# --------------------------------------------------------------------------- #
# Part 3: add/drop stability and stepwise search
# --------------------------------------------------------------------------- #


def test_subset_is_add_drop_stable_only_when_every_move_strictly_lowers_sharpe() -> None:
    certificate = certify_add_drop_stability(STABILITY_OBJECTIVE, ("a", "b"), ("a", "b", "c"))

    assert certificate.is_add_drop_stable is True
    assert certificate.sharpe == pytest.approx(1.2)
    assert certificate.size == 2
    assert certificate.best_addition == "c"
    assert certificate.best_addition_sharpe == pytest.approx(1.0)
    # best_removal names the drop that leaves the highest Sharpe behind, still below 1.2.
    assert certificate.best_removal == "b"
    assert certificate.best_removal_sharpe == pytest.approx(1.0)


def test_one_improving_addition_destabilises_a_subset_and_is_named() -> None:
    certificate = certify_add_drop_stability(STABILITY_OBJECTIVE, ("a",), ("a", "b", "c"))

    assert certificate.is_add_drop_stable is False
    assert certificate.best_addition == "b"
    assert certificate.best_addition_sharpe == pytest.approx(1.2)
    assert certificate.best_addition_sharpe > certificate.sharpe


def test_singleton_is_vacuously_removal_stable_so_stability_turns_only_on_additions() -> None:
    stable_singleton = certify_add_drop_stability(STABILITY_OBJECTIVE, ("c",), ("a", "b", "c"))
    unstable_singleton = certify_add_drop_stability(STABILITY_OBJECTIVE, ("a",), ("a", "b", "c"))

    for certificate in (stable_singleton, unstable_singleton):
        assert certificate.size == 1
        assert certificate.best_removal is None
        assert certificate.best_removal_sharpe is None
    assert stable_singleton.is_add_drop_stable is True
    assert unstable_singleton.is_add_drop_stable is False


def test_stability_requires_a_strictly_lower_sharpe_so_a_tied_move_is_not_stable() -> None:
    """The stopping rule is strict: a move that merely matches the incumbent breaks stability.

    ``PLATEAU_OBJECTIVE`` ties ("a", "b") with ("a", "b", "c") at 1.30, so the
    addition tie and the removal tie are both exercised. A non-strict rule would
    certify both subsets as stable and report two "optimal" subsets of different
    size with the same Sharpe.
    """

    tied_addition = certify_add_drop_stability(PLATEAU_OBJECTIVE, ("a", "b"), ("a", "b", "c"))
    tied_removal = certify_add_drop_stability(PLATEAU_OBJECTIVE, ("a", "b", "c"), ("a", "b", "c"))

    assert tied_addition.is_add_drop_stable is False
    assert tied_addition.best_addition == "c"
    assert tied_addition.best_addition_sharpe == pytest.approx(tied_addition.sharpe)

    assert tied_removal.is_add_drop_stable is False
    assert tied_removal.best_addition is None
    assert tied_removal.best_removal == "c"
    assert tied_removal.best_removal_sharpe == pytest.approx(tied_removal.sharpe)


@pytest.mark.parametrize("start", sorted(SWEEP_OBJECTIVE, key=lambda subset: (len(subset), subset)))
def test_stepwise_search_climbs_strictly_and_halts_on_an_add_drop_stable_subset(start: tuple[str, ...]) -> None:
    path = stepwise_search(SWEEP_OBJECTIVE, SWEEP_NAMES, start=start)
    values = [SWEEP_OBJECTIVE[node] for node in path]

    assert path[0] == start
    assert len(set(path)) == len(path)
    assert all(later > earlier for earlier, later in zip(values[:-1], values[1:], strict=True))
    assert certify_add_drop_stability(SWEEP_OBJECTIVE, path[-1], SWEEP_NAMES).is_add_drop_stable is True


def test_stepwise_search_reaches_the_global_optimum_by_dropping_a_signal_it_had_added() -> None:
    path = stepwise_search(SWEEP_OBJECTIVE, SWEEP_NAMES, start=("c",))

    assert path == [("c",), ("a", "c"), ("a", "b", "c"), ("a", "b")]
    assert path[-1] == best_subset_of_size(SWEEP_OBJECTIVE, 2)[0]
    assert max(SWEEP_OBJECTIVE, key=lambda subset: SWEEP_OBJECTIVE[subset]) == path[-1]


def test_stepwise_search_never_revisits_a_subset_across_a_sharpe_plateau() -> None:
    path = stepwise_search(PLATEAU_OBJECTIVE, ("a", "b", "c"), start=("c",))

    assert path == [("c",), ("a", "c"), ("a", "b", "c")]
    assert len(set(path)) == len(path)
    values = [PLATEAU_OBJECTIVE[node] for node in path]
    assert all(later > earlier for earlier, later in zip(values[:-1], values[1:], strict=True))
    # ("a", "b") ties the terminal subset at 1.30, so a non-strict rule would step
    # onto it and then straight back, never terminating.
    assert ("a", "b") not in path


def test_stepwise_search_refuses_an_unscoreable_start() -> None:
    with pytest.raises(SignalPortfolioError, match="not scoreable"):
        stepwise_search(SWEEP_OBJECTIVE, SWEEP_NAMES, start=("a", "z"))


# --------------------------------------------------------------------------- #
# Parts 1 and 2: best subset of a fixed size
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("size", "expected_subset", "expected_sharpe"),
    [(1, ("a",), 1.00), (2, ("a", "b"), 1.30), (3, ("a", "b", "d"), 1.20), (4, ("a", "b", "c", "d"), 1.05)],
)
def test_best_subset_of_size_returns_the_development_argmax(size: int, expected_subset: tuple[str, ...], expected_sharpe: float) -> None:
    subset, sharpe = best_subset_of_size(SWEEP_OBJECTIVE, size)

    assert subset == expected_subset
    assert sharpe == pytest.approx(expected_sharpe)


def test_best_subset_of_size_breaks_ties_deterministically_by_name() -> None:
    subset, sharpe = best_subset_of_size({("a", "b"): 1.0, ("a", "c"): 1.0}, 2)

    assert subset == ("a", "b")
    assert sharpe == pytest.approx(1.0)


@pytest.mark.parametrize("size", [0, 5, 9])
def test_best_subset_of_size_refuses_a_size_with_no_scoreable_subset(size: int) -> None:
    with pytest.raises(SignalPortfolioError, match=f"no subset of size {size}"):
        best_subset_of_size(SWEEP_OBJECTIVE, size)


# --------------------------------------------------------------------------- #
# Exhaustive enumeration and the shared Sharpe convention
# --------------------------------------------------------------------------- #


def test_enumeration_covers_every_non_empty_subset_and_reuses_the_week6_sharpe_convention() -> None:
    development = _triple_frame(31, 60)
    evaluation = _triple_frame(32, 30)
    objective, rows = enumerate_subset_sharpes(development, evaluation, periods_per_year=260, annual_risk_free_rate=0.02)

    assert len(objective) == 2**3 - 1 == 7
    assert len(rows) == len(objective)
    assert sorted(objective) == sorted(row.signals for row in rows)
    assert all(row.size == len(row.signals) for row in rows)
    assert sorted(len(subset) for subset in objective) == [1, 1, 1, 2, 2, 2, 3]

    single = next(row for row in rows if row.signals == ("s2",))
    # A singleton's equal-weighted average is the column itself, so the Sharpe must
    # be byte-identical to week6_pnl.annualized_sharpe under the same conventions.
    assert single.development_sharpe == annualized_sharpe(development["s2"], periods_per_year=260, annual_risk_free_rate=0.02)
    assert single.evaluation_sharpe == annualized_sharpe(evaluation["s2"], periods_per_year=260, annual_risk_free_rate=0.02)
    assert single.development_sharpe_standard_error == pytest.approx(sharpe_standard_error(development["s2"], periods_per_year=260))
    assert single.development_mean_daily_return == pytest.approx(float(development["s2"].mean()))
    assert single.development_daily_volatility == pytest.approx(float(development["s2"].std(ddof=1)))

    pair = next(row for row in rows if row.signals == ("s1", "s2"))
    assert pair.development_sharpe == pytest.approx(
        annualized_sharpe(development[["s1", "s2"]].mean(axis=1), periods_per_year=260, annual_risk_free_rate=0.02)
    )


def test_enumeration_refuses_a_panel_wider_than_the_exhaustive_limit() -> None:
    width = MAX_EXHAUSTIVE_SIGNALS + 1
    wide = pd.DataFrame(np.zeros((4, width)), columns=[f"signal_{index}" for index in range(width)])

    with pytest.raises(SignalPortfolioError, match="exhaustive enumeration refuses"):
        enumerate_subset_sharpes(wide, wide, periods_per_year=PERIODS_PER_YEAR, annual_risk_free_rate=0.0)


def test_enumeration_refuses_when_no_subset_has_defined_sharpe() -> None:
    constant = pd.DataFrame({"s1": [0.01] * 5, "s2": [0.02] * 5})

    with pytest.raises(SignalPortfolioError, match="no signal subset produced a defined Sharpe"):
        enumerate_subset_sharpes(
            constant,
            constant,
            periods_per_year=PERIODS_PER_YEAR,
            annual_risk_free_rate=0.0,
        )


def test_equal_weight_returns_is_the_plain_row_mean_of_the_chosen_columns() -> None:
    matrix = pd.DataFrame(
        {
            "s1": [0.010, -0.020, 0.030],
            "s2": [0.020, 0.000, -0.010],
            "s3": [-0.030, 0.040, 0.050],
        },
        index=["2026-01-02", "2026-01-05", "2026-01-06"],
    )

    combined = equal_weight_returns(matrix, ["s1", "s3"])

    pd.testing.assert_series_equal(combined, matrix[["s1", "s3"]].mean(axis=1))
    assert combined.tolist() == pytest.approx([-0.010, 0.010, 0.040])
    assert combined.index.tolist() == matrix.index.tolist()
    # A singleton subset must pass its column through untouched.
    assert equal_weight_returns(matrix, ["s2"]).tolist() == pytest.approx(matrix["s2"].tolist())


@pytest.mark.parametrize("subset", [[], ["s1", "missing"], ["missing"]])
def test_equal_weight_returns_refuses_empty_and_unknown_subsets(subset: list[str]) -> None:
    matrix = pd.DataFrame({"s1": [0.01, 0.02], "s2": [0.03, 0.04]})

    with pytest.raises(SignalPortfolioError):
        equal_weight_returns(matrix, subset)


def test_panel_return_matrix_pivots_returns_and_preserves_date_splits() -> None:
    panel = pd.DataFrame(
        [
            {"date": "2026-01-02", "split": "development", "signal": "s1", "daily_gross_return": 0.01},
            {"date": "2026-01-02", "split": "development", "signal": "s2", "daily_gross_return": 0.02},
            {"date": "2026-01-05", "split": "evaluation", "signal": "s1", "daily_gross_return": -0.01},
            {"date": "2026-01-05", "split": "evaluation", "signal": "s2", "daily_gross_return": 0.03},
        ]
    )

    matrix, splits = panel_return_matrix(panel, return_variant="gross")

    expected = pd.DataFrame(
        {"s1": [0.01, -0.01], "s2": [0.02, 0.03]},
        index=pd.Index(["2026-01-02", "2026-01-05"], name="date"),
    )
    expected.columns.name = "signal"
    pd.testing.assert_frame_equal(matrix, expected)
    pd.testing.assert_series_equal(
        splits,
        pd.Series(["development", "evaluation"], index=expected.index, name="split"),
    )


def test_panel_return_matrix_refuses_ragged_signal_dates() -> None:
    panel = pd.DataFrame(
        [
            {"date": "2026-01-02", "split": "development", "signal": "s1", "daily_net_return": 0.01},
            {"date": "2026-01-02", "split": "development", "signal": "s2", "daily_net_return": 0.02},
            {"date": "2026-01-05", "split": "evaluation", "signal": "s1", "daily_net_return": -0.01},
        ]
    )

    with pytest.raises(SignalPortfolioError, match="contains gaps"):
        panel_return_matrix(panel, return_variant="net")


def test_panel_return_matrix_rejects_an_unknown_return_variant() -> None:
    with pytest.raises(SignalPortfolioError, match="return_variant"):
        panel_return_matrix(pd.DataFrame(), return_variant="excess")


def _stub_signal_panel_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ragged: bool = False,
) -> None:
    from sentiment_benchmark import week6_model_comparison

    monkeypatch.setattr(
        week6_model_comparison,
        "common_comparison_split",
        lambda *_args, **_kwargs: "2026-01-05",
    )

    def fake_load(_signals_path: Path, _prices_path: Path, scorer_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        signals = pd.DataFrame({"symbol": ["A"]})
        signals.attrs["scorer_id"] = scorer_id
        return signals, pd.DataFrame()

    def fake_stock_daily(signals: pd.DataFrame, _prices: pd.DataFrame, **_kwargs: Any) -> tuple[pd.DataFrame, dict]:
        return pd.DataFrame({"symbol": ["A"], "scorer_id": [signals.attrs["scorer_id"]]}), {}

    def fake_portfolio_daily(stock_daily: pd.DataFrame, _portfolios: dict, *, starting_capital: float) -> pd.DataFrame:
        del starting_capital
        scorer_id = str(stock_daily["scorer_id"].iloc[0])
        dates = ["2026-01-02"] if ragged and scorer_id.endswith("/b") else ["2026-01-02", "2026-01-05"]
        return pd.DataFrame(
            {
                "date": dates,
                "split": ["development"] * len(dates),
                "daily_gross_return": [0.0, 0.01][: len(dates)],
                "daily_net_return": [0.0, 0.009][: len(dates)],
                "turnover": [0.0, 0.1][: len(dates)],
                "active_stock_count": [0, 1][: len(dates)],
            }
        )

    monkeypatch.setattr(signal_portfolio_module, "load_and_validate_inputs", fake_load)
    monkeypatch.setattr(signal_portfolio_module, "build_stock_daily_pnl", fake_stock_daily)
    monkeypatch.setattr(signal_portfolio_module, "build_portfolio_daily", fake_portfolio_daily)


def test_build_signal_panel_drops_only_sessions_inactive_for_every_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_signal_panel_dependencies(monkeypatch)

    panel = build_signal_panel("signals.csv", "prices.csv", _config())

    assert panel["date"].unique().tolist() == ["2026-01-05"]
    assert panel.attrs["dropped_inactive_sessions"] == ["2026-01-02"]
    assert panel.attrs["split_date"] == "2026-01-05"


def test_build_signal_panel_can_retain_structural_zero_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_signal_panel_dependencies(monkeypatch)

    panel = build_signal_panel("signals.csv", "prices.csv", _config(drop_inactive_sessions=False))

    assert panel["date"].unique().tolist() == ["2026-01-02", "2026-01-05"]
    assert panel.attrs["dropped_inactive_sessions"] == []


def test_build_signal_panel_refuses_ragged_scorer_date_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_signal_panel_dependencies(monkeypatch, ragged=True)

    with pytest.raises(SignalPortfolioError, match="ragged date grids"):
        build_signal_panel("signals.csv", "prices.csv", _config())


def test_run_signal_portfolio_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    config = _config(run_id="already_exists")
    (tmp_path / config.run_id).mkdir()

    with pytest.raises(SignalPortfolioError, match="already exists and is immutable"):
        run_signal_portfolio(
            "missing-signals.csv",
            "missing-prices.csv",
            tmp_path,
            config,
            command="unit test",
            repo_root=tmp_path,
        )


# --------------------------------------------------------------------------- #
# Honesty diagnostics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("periods_per_year", [12, 252, 260])
def test_sharpe_standard_error_matches_the_lo_2002_formula(periods_per_year: int) -> None:
    values = np.array([0.010, -0.020, 0.030, 0.005, -0.015, 0.020])
    period_sharpe = float(values.mean()) / float(values.std(ddof=1))
    expected = math.sqrt(periods_per_year) * math.sqrt((1.0 + 0.5 * period_sharpe**2) / len(values))

    assert sharpe_standard_error(values, periods_per_year=periods_per_year) == pytest.approx(expected)
    assert sharpe_standard_error(pd.Series(values), periods_per_year=periods_per_year) == pytest.approx(expected)


@pytest.mark.parametrize(
    "returns",
    [
        [0.01, -0.01],
        [0.01],
        [],
        [0.01, float("nan"), 0.02],
        [0.01, 0.01, 0.01, 0.01, 0.01],
    ],
)
def test_sharpe_standard_error_is_undefined_for_short_or_constant_series(returns: list[float]) -> None:
    assert sharpe_standard_error(returns) is None


def test_selection_null_distribution_is_reproducible_from_its_seed() -> None:
    development = _triple_frame(41, 50)
    scalar_keys = ("replications", "subsets_searched", "mean_max_sharpe", "median_max_sharpe", "percentile_95_max_sharpe")
    kwargs: dict[str, Any] = {"periods_per_year": PERIODS_PER_YEAR, "replications": 25, "block_length": 5}

    first = selection_null_distribution(development, seed=7, **kwargs)
    repeat = selection_null_distribution(development, seed=7, **kwargs)
    other = selection_null_distribution(development, seed=8, **kwargs)

    assert {key: first[key] for key in scalar_keys} == {key: repeat[key] for key in scalar_keys}
    np.testing.assert_array_equal(first["distribution"], repeat["distribution"])
    assert first["mean_max_sharpe"] != other["mean_max_sharpe"]
    assert first["subsets_searched"] == 2**3 - 1
    assert first["replications"] == 25
    assert first["seed"] == 7


def test_selection_null_refuses_a_block_longer_than_the_development_sample() -> None:
    with pytest.raises(SignalPortfolioError, match="too short"):
        selection_null_distribution(
            _triple_frame(42, 5),
            periods_per_year=PERIODS_PER_YEAR,
            replications=5,
            block_length=5,
            seed=1,
        )


def test_bootstrap_subset_sharpe_is_reproducible_from_its_seed() -> None:
    returns = _triple_frame(41, 50)["s1"]
    kwargs: dict[str, Any] = {"periods_per_year": PERIODS_PER_YEAR, "replications": 25, "block_length": 5}

    first = bootstrap_subset_sharpe(returns, seed=7, **kwargs)
    repeat = bootstrap_subset_sharpe(returns, seed=7, **kwargs)
    other = bootstrap_subset_sharpe(returns, seed=8, **kwargs)

    assert first == repeat
    assert first["mean"] != other["mean"]
    assert first["confidence_interval_low"] <= first["mean"] <= first["confidence_interval_high"]
    assert 0.0 <= first["share_at_or_below_zero"] <= 1.0


def test_bootstrap_refuses_a_series_shorter_than_the_block_length() -> None:
    with pytest.raises(SignalPortfolioError, match="too short"):
        bootstrap_subset_sharpe(pd.Series([0.01, -0.02, 0.03]), periods_per_year=PERIODS_PER_YEAR, replications=5, block_length=5, seed=1)


def test_bootstrap_refuses_constant_returns_with_no_defined_replication() -> None:
    with pytest.raises(SignalPortfolioError, match="no defined Sharpe"):
        bootstrap_subset_sharpe(
            pd.Series([0.0] * 10),
            periods_per_year=PERIODS_PER_YEAR,
            replications=5,
            block_length=3,
            seed=1,
        )


# --------------------------------------------------------------------------- #
# Breakeven cost
# --------------------------------------------------------------------------- #


def _cost_panel(turnovers: tuple[float, float, float, float]) -> pd.DataFrame:
    """Two signals over two development sessions, plus evaluation rows that must be ignored."""

    rows = [
        {"signal": "s1", "split": "development", "date": "2026-01-02", "daily_gross_return": 0.004, "turnover": turnovers[0]},
        {"signal": "s1", "split": "development", "date": "2026-01-05", "daily_gross_return": 0.002, "turnover": turnovers[1]},
        {"signal": "s2", "split": "development", "date": "2026-01-02", "daily_gross_return": 0.008, "turnover": turnovers[2]},
        {"signal": "s2", "split": "development", "date": "2026-01-05", "daily_gross_return": 0.004, "turnover": turnovers[3]},
        {"signal": "s1", "split": "evaluation", "date": "2026-01-06", "daily_gross_return": 0.900, "turnover": 0.9},
        {"signal": "s2", "split": "evaluation", "date": "2026-01-06", "daily_gross_return": 0.900, "turnover": 0.9},
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        # 10_000 * mean(gross) / mean(turnover), means taken over sessions after
        # averaging across the held signals within each session.
        (["s1"], 10_000.0 * 0.003 / 0.40),
        (["s2"], 10_000.0 * 0.006 / 0.40),
        (["s1", "s2"], 10_000.0 * 0.0045 / 0.40),
    ],
)
def test_breakeven_cost_is_gross_over_turnover_on_the_development_block(subset: list[str], expected: float) -> None:
    panel = _cost_panel((0.50, 0.30, 0.20, 0.60))

    assert breakeven_cost_bps_per_side(panel, subset) == pytest.approx(expected)


def test_breakeven_cost_ignores_the_evaluation_block() -> None:
    panel = _cost_panel((0.50, 0.30, 0.20, 0.60))
    development_only = panel.loc[panel["split"].eq("development")].reset_index(drop=True)

    assert breakeven_cost_bps_per_side(panel, ["s1", "s2"]) == pytest.approx(breakeven_cost_bps_per_side(development_only, ["s1", "s2"]))


@pytest.mark.parametrize(
    ("panel_kwargs", "subset", "split"),
    [
        ({"turnovers": (0.0, 0.0, 0.0, 0.0)}, ["s1", "s2"], "development"),
        ({"turnovers": (0.50, 0.30, 0.20, 0.60)}, ["s1"], "holdout"),
        ({"turnovers": (0.50, 0.30, 0.20, 0.60)}, ["absent"], "development"),
    ],
)
def test_breakeven_cost_is_undefined_without_turnover_or_matching_rows(
    panel_kwargs: dict[str, tuple[float, float, float, float]],
    subset: list[str],
    split: str,
) -> None:
    panel = _cost_panel(**panel_kwargs)

    assert breakeven_cost_bps_per_side(panel, subset, split=split) is None


# --------------------------------------------------------------------------- #
# Configuration fail-closed behaviour
# --------------------------------------------------------------------------- #


def test_default_config_discloses_its_exploratory_post_hoc_provenance() -> None:
    config = _config()

    assert config.exploratory is True
    assert config.specified_after_week6_results is True
    assert config.return_variant == "gross"
    assert config.holding_period == 1
    assert 0.0 < config.development_fraction < 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_id": "  "}, "run_id"),
        ({"scorer_ids": ("headline/a", "headline/a")}, "unique"),
        ({"scorer_ids": ("headline/a",)}, "at least two scorers"),
        ({"scorer_ids": ()}, "at least two scorers"),
        ({"return_variant": "excess"}, "return_variant"),
        ({"threshold": -0.1}, "threshold"),
        ({"holding_period": 0}, "holding_period"),
        ({"starting_capital": 0.0}, "starting_capital"),
        ({"starting_capital": -1.0}, "starting_capital"),
        ({"transaction_cost_bps_per_side": -1.0}, "transaction cost"),
        ({"development_fraction": 0.0}, "development_fraction"),
        ({"development_fraction": 1.0}, "development_fraction"),
        ({"annualization_periods": 0}, "annualization_periods"),
        ({"bootstrap_replications": 0}, "bootstrap_replications"),
        ({"bootstrap_block_length": 0}, "bootstrap_block_length"),
        # The study may not hide that it is a post-hoc in-sample Sharpe search.
        ({"exploratory": False}, "exploratory"),
        ({"specified_after_week6_results": False}, "specified after the Week 6 results"),
    ],
)
def test_config_rejects_invalid_assumptions(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(SignalPortfolioError, match=message):
        _config(**overrides)
