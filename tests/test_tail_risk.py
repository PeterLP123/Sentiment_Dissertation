"""Synthetic unit tests for the FNSPID tail-risk primitives.

Every fixture here is generated; none of it is empirical evidence.  The tests
defend the invariants the notebook relies on: leak-safe timing, a
point-in-time variance recursion, a VaR/ES parameterisation that cannot violate
``es < var < 0``, an FZ0 loss minimised at the true tail functionals, and a
block bootstrap that keeps whole date cross-sections together.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from sentiment_benchmark.tail_risk import (
    GjrParams,
    _fz0_value_and_gradient,
    _positive_exponential,
    apply_news_scale_adjustment,
    block_bootstrap_indices,
    build_next_session_targets,
    christoffersen_tests,
    date_block_bootstrap_mean,
    es_identification_residual,
    fit_joint_var_es,
    fit_news_scale_adjustment,
    fz0_loss,
    gjr_conditional_variances,
    gjr_conditional_variances_piecewise,
    kupiec_test,
    map_dates_to_reaction_sessions,
    pinball_loss,
    prepare_clean_output_directory,
    repair_adjusted_close,
    resolve_tail_risk_variant,
    select_expanding_refit_window,
    tail_forecasts,
    verify_input_file,
    verify_output_manifest,
    verify_strictly_after,
)


def test_variant_resolver_covers_the_frozen_two_by_two_design() -> None:
    assert resolve_tail_risk_variant(" V1 ")[:3] == (
        "v1",
        "none",
        "frozen_development",
    )
    assert resolve_tail_risk_variant("price_only")[:3] == (
        "price_only",
        "min_abs_return",
        "frozen_development",
    )
    assert resolve_tail_risk_variant("refit_only")[:3] == (
        "refit_only",
        "none",
        "annual_expanding",
    )
    assert resolve_tail_risk_variant("v2")[:3] == (
        "v2",
        "min_abs_return",
        "annual_expanding",
    )
    with pytest.raises(ValueError, match="variant must be one of"):
        resolve_tail_risk_variant("unknown")


def test_output_directory_must_be_new_or_empty(tmp_path) -> None:
    output = tmp_path / "run"
    assert prepare_clean_output_directory(output) == output
    (output / "sentinel.txt").write_text("prior run", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        prepare_clean_output_directory(output)


def test_input_identity_verification_rejects_size_and_hash_mismatches(tmp_path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"frozen input")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        verify_input_file(
            source,
            expected_sha256=digest,
            expected_size=len(b"frozen input"),
        )
        == digest
    )
    with pytest.raises(ValueError, match="bytes"):
        verify_input_file(source, expected_sha256=digest, expected_size=1)
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_input_file(source, expected_sha256=digest)


def test_output_manifest_verifier_rejects_tampering_and_unexpected_files(
    tmp_path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    artifact = output / "table.csv"
    artifact.write_bytes(b"a,b\n1,2\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    records = {
        "table.csv": {
            "sha256": digest,
            "size_bytes": artifact.stat().st_size,
        }
    }
    (output / "manifest.json").write_text("excluded", encoding="utf-8")
    assert verify_output_manifest(output, records) == 1

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output manifest mismatch"):
        verify_output_manifest(output, records)

    artifact.write_bytes(b"a,b\n1,2\n")
    (output / "stale.csv").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected=.*stale.csv"):
        verify_output_manifest(output, records)


def test_expanding_refit_window_excludes_predevelopment_and_current_year() -> None:
    index = pd.to_datetime(["2010-12-31", "2011-01-03", "2016-12-30", "2017-01-03", "2018-01-02"])
    returns = pd.Series(np.arange(index.size, dtype=float), index=index)
    window = select_expanding_refit_window(
        returns,
        development_start="2011-01-01",
        evaluation_year=2017,
    )
    assert window.index.tolist() == pd.to_datetime(["2011-01-03", "2016-12-30"]).tolist()
    assert window.index.max() < pd.Timestamp("2017-01-01")


SESSIONS = pd.to_datetime(
    [
        "2020-01-02",  # Thursday
        "2020-01-03",  # Friday
        "2020-01-06",  # Monday
        "2020-01-07",
        "2020-01-08",
    ]
)


def test_news_maps_to_the_first_session_strictly_after_the_calendar_date() -> None:
    calendar_dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"])
    mapped = map_dates_to_reaction_sessions(calendar_dates, SESSIONS)
    assert list(pd.DatetimeIndex(mapped).strftime("%Y-%m-%d")) == [
        "2020-01-03",  # same-session news must roll forward, never react on its own date
        "2020-01-06",  # Friday news reacts on Monday
        "2020-01-06",  # Saturday collapses onto the same Monday session
        "2020-01-06",  # Sunday collapses onto the same Monday session
        "2020-01-07",
    ]
    assert verify_strictly_after(calendar_dates, mapped) == 0


def test_dates_without_a_later_session_map_to_nat_rather_than_the_last_session() -> None:
    mapped = map_dates_to_reaction_sessions(pd.to_datetime(["2020-01-08", "2020-02-01"]), SESSIONS)
    assert pd.isna(pd.DatetimeIndex(mapped)).all()


def test_map_rejects_unsorted_sessions() -> None:
    with pytest.raises(ValueError):
        map_dates_to_reaction_sessions(["2020-01-02"], pd.to_datetime(["2020-01-06", "2020-01-02"]))


def test_target_return_is_the_next_session_close_to_close_move() -> None:
    prices = pd.DataFrame({"session_date": SESSIONS, "adjusted_close": [100.0, 101.0, 99.0, 99.5, 103.0]})
    frame = build_next_session_targets(prices)
    assert frame["target_date"].tolist()[:-1] == list(SESSIONS[1:])
    assert pd.isna(frame["target_date"].iloc[-1])
    assert frame["target_return"].iloc[0] == pytest.approx(math.log(101.0 / 100.0))
    assert frame["target_return"].iloc[2] == pytest.approx(math.log(99.5 / 99.0))
    assert pd.isna(frame["target_return"].iloc[-1])
    # The realised session return and the previous row's target are the same move.
    assert frame["log_return"].iloc[1] == pytest.approx(frame["target_return"].iloc[0])


def test_target_construction_rejects_duplicate_or_non_positive_prices() -> None:
    duplicated = pd.DataFrame({"session_date": pd.to_datetime(["2020-01-02", "2020-01-02"]), "adjusted_close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        build_next_session_targets(duplicated)
    negative = pd.DataFrame({"session_date": SESSIONS[:2], "adjusted_close": [1.0, 0.0]})
    with pytest.raises(ValueError, match="positive"):
        build_next_session_targets(negative)


def test_gjr_recursion_matches_the_hand_computed_path_and_is_point_in_time() -> None:
    params = GjrParams(mu=0.001, omega=2e-6, alpha=0.05, gamma=0.08, beta=0.90)
    returns = np.array([0.01, -0.02, 0.005])
    sigma2 = gjr_conditional_variances(returns, params, initial_variance=4e-4)

    expected = [4e-4]
    for value in returns:
        shock = value - params.mu
        expected.append(
            params.omega + params.alpha * shock**2 + (params.gamma * shock**2 if shock < 0 else 0.0) + params.beta * expected[-1]
        )
    assert sigma2 == pytest.approx(np.array(expected))
    assert sigma2.size == returns.size + 1

    # Changing return t must leave every variance at index <= t untouched.
    perturbed = returns.copy()
    perturbed[1] = -0.30
    shifted = gjr_conditional_variances(perturbed, params, initial_variance=4e-4)
    assert shifted[:2] == pytest.approx(sigma2[:2])
    assert shifted[2] > sigma2[2]


def test_gjr_leverage_term_only_fires_on_negative_shocks() -> None:
    params = GjrParams(mu=0.0, omega=1e-6, alpha=0.05, gamma=0.10, beta=0.85)
    up = gjr_conditional_variances(np.array([0.03]), params, initial_variance=1e-4)
    down = gjr_conditional_variances(np.array([-0.03]), params, initial_variance=1e-4)
    assert down[-1] > up[-1]
    assert down[-1] - up[-1] == pytest.approx(params.gamma * 0.03**2)


def test_gjr_persistence_and_unconditional_variance() -> None:
    params = GjrParams(mu=0.0, omega=1e-6, alpha=0.04, gamma=0.06, beta=0.90)
    assert params.persistence == pytest.approx(0.97)
    assert params.unconditional_variance == pytest.approx(1e-6 / 0.03)


def test_gjr_rejects_non_finite_returns_and_bad_initialisation() -> None:
    params = GjrParams(mu=0.0, omega=1e-6, alpha=0.05, gamma=0.0, beta=0.9)
    with pytest.raises(ValueError):
        gjr_conditional_variances(np.array([0.01, np.nan]), params, initial_variance=1e-4)
    with pytest.raises(ValueError):
        gjr_conditional_variances(np.array([0.01]), params, initial_variance=0.0)


def test_piecewise_recursion_reduces_to_the_fixed_parameter_recursion() -> None:
    params = GjrParams(mu=0.001, omega=2e-6, alpha=0.05, gamma=0.08, beta=0.90)
    returns = np.array([0.01, -0.02, 0.005, -0.011, 0.004])
    fixed = gjr_conditional_variances(returns, params, initial_variance=4e-4)
    piecewise = gjr_conditional_variances_piecewise(returns, [params] * returns.size, initial_variance=4e-4)
    assert piecewise == pytest.approx(fixed)


def test_piecewise_recursion_switches_parameters_without_restarting_the_state() -> None:
    calm = GjrParams(mu=0.0, omega=1e-6, alpha=0.02, gamma=0.02, beta=0.95)
    jumpy = GjrParams(mu=0.0, omega=8e-6, alpha=0.15, gamma=0.15, beta=0.70)
    returns = np.array([0.01, -0.02, 0.005, -0.03])
    schedule = [calm, calm, jumpy, jumpy]
    sigma2 = gjr_conditional_variances_piecewise(returns, schedule, initial_variance=4e-4)

    # Steps governed by `calm` must match a pure-calm run exactly...
    pure_calm = gjr_conditional_variances(returns, calm, initial_variance=4e-4)
    assert sigma2[:3] == pytest.approx(pure_calm[:3])
    # ...and the switch must carry the variance state rather than reset it.
    shock = returns[2] - jumpy.mu
    expected = jumpy.omega + jumpy.alpha * shock**2 + jumpy.beta * sigma2[2]
    assert sigma2[3] == pytest.approx(expected)

    with pytest.raises(ValueError, match="one parameter set per return"):
        gjr_conditional_variances_piecewise(returns, schedule[:2], initial_variance=4e-4)


def test_price_repair_leaves_a_clean_series_untouched() -> None:
    dates = pd.bdate_range("2020-01-01", periods=12)
    raw = np.linspace(100.0, 111.0, 12)
    adjusted = raw * 0.9  # a constant factor is a perfectly consistent adjustment
    repaired, replaced = repair_adjusted_close(dates, adjusted, raw, gap_threshold=0.05)
    assert not replaced.any()
    assert repaired == pytest.approx(adjusted)


def test_price_repair_keeps_the_adjusted_return_through_a_genuine_split() -> None:
    # A 2-for-1 split: the raw close halves, the adjusted close is continuous.
    dates = pd.bdate_range("2020-01-01", periods=6)
    raw = np.array([100.0, 101.0, 102.0, 51.5, 52.0, 52.5])
    adjusted = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    repaired, replaced = repair_adjusted_close(dates, adjusted, raw, gap_threshold=0.05)
    assert not replaced.any(), "the adjusted series is the correct one across a split"
    assert repaired == pytest.approx(adjusted)


def test_price_repair_replaces_an_inconsistent_adjustment_step() -> None:
    # The Merck 2020-07-06 signature: raw moves a little, adjusted lurches.
    dates = pd.bdate_range("2020-06-29", periods=6)
    raw = np.array([77.0, 78.1, 78.8, 75.9, 75.1, 74.4])
    adjusted = raw.copy()
    adjusted[3:] *= 0.896  # the factor steps mid-series for no economic reason
    repaired, replaced = repair_adjusted_close(dates, adjusted, raw, gap_threshold=0.05)

    assert replaced.tolist() == [False, False, False, True, False, False]
    repaired_returns = np.diff(np.log(repaired))
    raw_returns = np.diff(np.log(raw))
    # The artifact is gone: the repaired path now tracks the real price moves.
    assert repaired_returns == pytest.approx(raw_returns)
    assert abs(repaired_returns[2]) < 0.05
    # Anchoring is preserved, so levels stay comparable with the original.
    assert repaired[0] == pytest.approx(adjusted[0])


def test_price_repair_validates_its_inputs() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(ValueError, match="same length"):
        repair_adjusted_close(dates, [1.0, 2.0], [1.0, 2.0, 3.0], gap_threshold=0.05)
    with pytest.raises(ValueError, match="sorted"):
        repair_adjusted_close(dates[::-1], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], gap_threshold=0.05)
    with pytest.raises(ValueError, match="positive"):
        repair_adjusted_close(dates, [1.0, 0.0, 3.0], [1.0, 2.0, 3.0], gap_threshold=0.05)


def test_tail_forecasts_always_satisfy_es_below_var_below_zero() -> None:
    generator = np.random.default_rng(11)
    # Deliberately extreme linear indices: some overflow the safe band.
    design = np.column_stack([np.ones(500), generator.normal(size=(500, 3)) * 6.0])
    beta_var = np.array([0.7, 2.0, -2.0, 0.5])
    beta_gap = np.array([-0.9, -1.5, 1.5, 0.25])
    var, es, extended = tail_forecasts(design, beta_var, beta_gap)
    assert np.all(var < 0.0)
    assert np.all(es < var), "the relative gap floor must survive float64 absorption"
    assert np.isfinite(var).all() and np.isfinite(es).all()
    assert extended > 0, "this fixture is meant to exercise the numerical extension"

    # Ordinary linear indices must not touch the rails at all.
    modest = np.column_stack([np.ones(500), generator.normal(size=(500, 3))])
    var, es, extended = tail_forecasts(modest, np.array([0.7, 0.3, -0.2, 0.1]), np.array([-0.9, 0.2, 0.1, -0.1]))
    assert extended == 0
    assert np.all(es < var) and np.all(var < 0.0)


def test_guarded_exponential_stays_positive_and_keeps_a_live_derivative() -> None:
    index = np.array([-2000.0, -700.0, -10.0, 0.0, 10.0, 40.0, 60.0, 5_000.0])
    value, derivative, outside = _positive_exponential(index)
    assert np.all(value > 0.0), "a zero gap would break the strict es < var ordering"
    assert np.isfinite(value).all(), "the ceiling must prevent overflow"
    assert np.all(derivative > 0.0), "a zero derivative would trap a gradient optimiser"
    assert np.all(np.diff(value) >= 0.0), "the map must stay monotone"
    # Strictly increasing everywhere except at the underflow floor, which clamps.
    assert np.all(np.diff(value[1:]) > 0.0)
    assert outside.tolist() == [True, False, False, False, False, False, True, True]
    # Inside the band it is exactly exp.
    assert value[2:6] == pytest.approx(np.exp(index[2:6]))


def test_fit_joint_var_es_recovers_the_expected_shortfall_level_not_just_the_quantile() -> None:
    # Regression guard for two real defects found by the first executed run:
    #  1. a narrow linear-index clip whose gradient was zeroed outside the band,
    #     which froze the optimiser;
    #  2. a warm start that smoothed the indicator alone, so sigma(u) * (v - y)
    #     went negative for y > v and the objective preferred es -> var.
    # Both left the VaR slope looking sensible while ES collapsed onto VaR,
    # which is exactly why this test pins the ES level and the gap.
    alpha = 0.025
    generator = np.random.default_rng(808)
    size = 60_000
    sample = generator.standard_normal(size)
    design = np.ones((size, 1))

    fit = fit_joint_var_es(design, sample, alpha=alpha, name="unconditional", columns=("const",))
    var, es = fit.forecast(design)

    # FZ0 is minimised at the EMPIRICAL alpha-quantile and the empirical mean
    # below it, so those are the estimator's targets.
    empirical_var = float(np.quantile(sample, alpha))
    empirical_es = float(sample[sample <= empirical_var].mean())
    assert float(var[0]) == pytest.approx(empirical_var, abs=5e-3)
    assert float(es[0]) == pytest.approx(empirical_es, abs=5e-3)

    from scipy import stats as scipy_stats

    population_var = float(scipy_stats.norm.ppf(alpha))
    population_es = float(-scipy_stats.norm.pdf(scipy_stats.norm.ppf(alpha)) / alpha)
    assert float(var[0]) == pytest.approx(population_var, abs=0.08)
    assert float(es[0]) == pytest.approx(population_es, abs=0.10)

    # The gap must be a real Expected Shortfall gap, not a numerical floor.
    assert float(var[0] - es[0]) > 0.3
    assert fit.clipped_rows == 0
    assert fit.converged


def test_fz0_matches_its_closed_form_and_rejects_invalid_orderings() -> None:
    realised = np.array([-3.0, -1.0])
    var = np.array([-2.0, -2.0])
    es = np.array([-2.5, -2.5])
    alpha = 0.025
    expected_hit = -(1.0 / (alpha * -2.5)) * (-2.0 - -3.0) + (-2.0 / -2.5) + math.log(2.5) - 1.0
    expected_miss = (-2.0 / -2.5) + math.log(2.5) - 1.0
    assert fz0_loss(realised, var, es, alpha) == pytest.approx([expected_hit, expected_miss])
    with pytest.raises(ValueError, match="es < var < 0"):
        fz0_loss(realised, var, np.array([-1.0, -1.0]), alpha)
    with pytest.raises(ValueError, match="es < var < 0"):
        fz0_loss(realised, np.array([0.5, 0.5]), es, alpha)


def test_analytic_fz0_gradient_matches_central_differences() -> None:
    alpha = 0.025
    generator = np.random.default_rng(404)
    size = 2_000
    design = np.column_stack([np.ones(size), generator.standard_normal(size), generator.standard_normal(size)])
    realised = generator.standard_normal(size)
    parameters = np.array([0.75, 0.12, -0.08, -0.55, 0.05, 0.09])

    for smoothing in (0.5, 0.0):
        value, gradient = _fz0_value_and_gradient(parameters, design, realised, alpha, smoothing)
        assert math.isfinite(value)
        step = 1e-6
        numerical = np.empty_like(parameters)
        for index in range(parameters.size):
            up = parameters.copy()
            down = parameters.copy()
            up[index] += step
            down[index] -= step
            numerical[index] = (
                _fz0_value_and_gradient(up, design, realised, alpha, smoothing)[0]
                - _fz0_value_and_gradient(down, design, realised, alpha, smoothing)[0]
            ) / (2.0 * step)
        # The unsmoothed objective is only differentiable almost everywhere, so
        # a loose tolerance is used there; the smoothed stage must be tight.
        tolerance = 1e-7 if smoothing > 0.0 else 5e-3
        assert gradient == pytest.approx(numerical, abs=tolerance)


def test_smoothed_fz0_gradient_remains_exact_for_extreme_tail_observations() -> None:
    design = np.ones((3, 1))
    realised = np.array([-100.0, -80.0, -60.0])
    parameters = np.array([0.0, 0.0])
    smoothing = 0.5
    _, gradient = _fz0_value_and_gradient(
        parameters,
        design,
        realised,
        0.025,
        smoothing,
    )
    step = 1e-6
    numerical = np.empty_like(parameters)
    for index in range(parameters.size):
        up = parameters.copy()
        down = parameters.copy()
        up[index] += step
        down[index] -= step
        numerical[index] = (
            _fz0_value_and_gradient(up, design, realised, 0.025, smoothing)[0]
            - _fz0_value_and_gradient(
                down,
                design,
                realised,
                0.025,
                smoothing,
            )[0]
        ) / (2.0 * step)
    assert gradient == pytest.approx(numerical, rel=1e-7, abs=1e-7)


def test_fz0_is_minimised_at_the_true_tail_functionals() -> None:
    alpha = 0.025
    generator = np.random.default_rng(7)
    sample = generator.standard_normal(400_000)
    from scipy import stats as scipy_stats

    true_var = float(scipy_stats.norm.ppf(alpha))
    true_es = float(-scipy_stats.norm.pdf(scipy_stats.norm.ppf(alpha)) / alpha)

    def mean_loss(var_value: float, es_value: float) -> float:
        return float(
            fz0_loss(
                sample,
                np.full(sample.size, var_value),
                np.full(sample.size, es_value),
                alpha,
            ).mean()
        )

    optimal = mean_loss(true_var, true_es)
    for var_shift, es_shift in ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.3), (0.0, -0.3), (0.2, 0.2)):
        assert mean_loss(true_var + var_shift, true_es + es_shift) > optimal


def test_fit_joint_var_es_recovers_a_conditional_tail_and_ranks_below_a_misspecified_fit() -> None:
    alpha = 0.025
    generator = np.random.default_rng(2026)
    size = 40_000
    signal = generator.standard_normal(size)
    scale = np.exp(0.25 * signal)
    realised = scale * generator.standard_normal(size)
    design = np.column_stack([np.ones(size), signal])

    fit = fit_joint_var_es(design, realised, alpha=alpha, name="conditional", columns=("const", "signal"))
    assert fit.converged
    # The VaR magnitude must grow with the scale driver.
    assert fit.beta_var[1] == pytest.approx(0.25, abs=0.06)
    var, es = fit.forecast(design)
    assert np.all(es < var) and np.all(var < 0.0)
    assert abs(float((realised <= var).mean()) - alpha) < 0.004

    unconditional = fit_joint_var_es(design[:, :1], realised, alpha=alpha, name="unconditional", columns=("const",))
    assert fit.objective < unconditional.objective
    assert {row["term"] for row in fit.coefficient_rows()} == {"const", "signal"}
    assert {row["block"] for row in fit.coefficient_rows()} == {"var", "es_gap"}


def test_pinball_loss_is_minimised_at_the_true_quantile() -> None:
    alpha = 0.025
    generator = np.random.default_rng(5)
    sample = generator.standard_normal(200_000)
    from scipy import stats as scipy_stats

    truth = float(scipy_stats.norm.ppf(alpha))
    best = float(pinball_loss(sample, np.full(sample.size, truth), alpha).mean())
    for shift in (-0.2, 0.2):
        assert float(pinball_loss(sample, np.full(sample.size, truth + shift), alpha).mean()) > best


def test_es_identification_residual_has_zero_mean_for_correct_forecasts() -> None:
    alpha = 0.025
    generator = np.random.default_rng(13)
    sample = generator.standard_normal(400_000)
    from scipy import stats as scipy_stats

    var = np.full(sample.size, float(scipy_stats.norm.ppf(alpha)))
    es = np.full(sample.size, float(-scipy_stats.norm.pdf(scipy_stats.norm.ppf(alpha)) / alpha))
    assert abs(float(es_identification_residual(sample, var, es, alpha).mean())) < 0.02
    too_shallow = es * 0.7
    assert float(es_identification_residual(sample, var, too_shallow, alpha).mean()) > 0.1


def test_block_bootstrap_keeps_contiguous_blocks_and_is_seed_deterministic() -> None:
    draws = block_bootstrap_indices(50, block_length=10, replications=25, seed=99)
    assert draws.shape == (25, 50)
    assert draws.min() >= 0 and draws.max() < 50
    for row in draws:
        for start in range(0, 50, 10):
            block = row[start : start + 10]
            assert np.array_equal(block, np.arange(block[0], block[0] + 10))
    repeat = block_bootstrap_indices(50, block_length=10, replications=25, seed=99)
    assert np.array_equal(draws, repeat)
    assert not np.array_equal(draws, block_bootstrap_indices(50, block_length=10, replications=25, seed=100))


def test_block_bootstrap_handles_a_block_longer_than_the_sample() -> None:
    draws = block_bootstrap_indices(4, block_length=20, replications=3, seed=1)
    assert draws.shape == (3, 4)
    assert np.array_equal(draws, np.tile(np.arange(4), (3, 1)))


def test_date_block_bootstrap_point_estimate_is_the_weighted_mean() -> None:
    values = np.array([-1.0, 0.5, -0.25, 2.0])
    weights = np.array([10.0, 1.0, 4.0, 1.0])
    summary = date_block_bootstrap_mean(values, weights, block_length=2, replications=500, seed=3, confidence=0.95)
    assert summary["point_estimate"] == pytest.approx(float(np.sum(values * weights) / weights.sum()))
    assert summary["ci_low"] <= summary["bootstrap_mean"] <= summary["ci_high"]
    assert 0.0 <= summary["share_below_zero"] <= 1.0
    assert summary["n_dates"] == 4


def test_date_equal_and_observation_equal_bootstraps_use_distinct_weights() -> None:
    date_means = np.array([-1.0, 1.0])
    firm_counts = np.array([100.0, 1.0])
    observation_equal = date_block_bootstrap_mean(
        date_means,
        firm_counts,
        block_length=1,
        replications=50,
        seed=9,
    )
    date_equal = date_block_bootstrap_mean(
        date_means,
        None,
        block_length=1,
        replications=50,
        seed=9,
    )
    assert observation_equal["point_estimate"] == pytest.approx(-99.0 / 101.0)
    assert date_equal["point_estimate"] == pytest.approx(0.0)


def test_date_block_bootstrap_interval_excludes_zero_for_a_clear_negative_effect() -> None:
    generator = np.random.default_rng(21)
    values = generator.normal(loc=-0.05, scale=0.01, size=600)
    summary = date_block_bootstrap_mean(values, block_length=20, replications=800, seed=42)
    assert summary["ci_high"] < 0.0
    assert summary["share_below_zero"] == 1.0


def test_kupiec_and_christoffersen_flag_miscalibrated_and_clustered_hits() -> None:
    alpha = 0.025
    generator = np.random.default_rng(31)
    calibrated = (generator.random(4000) < alpha).astype(int)
    assert kupiec_test(calibrated, alpha)["p_value"] > 0.05

    over_violating = (generator.random(4000) < 0.10).astype(int)
    assert kupiec_test(over_violating, alpha)["p_value"] < 1e-6

    clustered = np.zeros(4000, dtype=int)
    clustered[100:140] = 1
    clustered[2000:2060] = 1
    assert christoffersen_tests(clustered, alpha)["independence_p_value"] < 1e-6
    assert christoffersen_tests(calibrated, alpha)["conditional_coverage_p_value"] > 0.05


def test_christoffersen_handles_isolated_hits_with_no_one_to_one_transitions() -> None:
    hits = np.zeros(1_000, dtype=int)
    hits[np.arange(50, 1_000, 100)] = 1
    result = christoffersen_tests(hits, 0.025)
    assert math.isfinite(result["independence_statistic"])
    assert math.isfinite(result["independence_p_value"])
    assert math.isfinite(result["conditional_coverage_statistic"])


def test_news_scale_adjustment_recovers_a_planted_volatility_multiplier() -> None:
    generator = np.random.default_rng(17)
    size = 60_000
    news = (generator.random(size) < 0.35).astype(float)
    base_sigma = np.full(size, 0.02)
    true_theta = np.array([0.6])
    true_sigma = apply_news_scale_adjustment(base_sigma, news[:, None], true_theta)
    shocks = true_sigma * generator.standard_normal(size)

    theta, diagnostics = fit_news_scale_adjustment(news[:, None], shocks, base_sigma)
    assert theta[0] == pytest.approx(true_theta[0], abs=0.08)
    assert diagnostics["qlike_final"] <= diagnostics["qlike_start"]
    fitted = apply_news_scale_adjustment(base_sigma, news[:, None], theta)
    assert np.all(fitted > 0.0)
    assert fitted[news == 1].mean() > fitted[news == 0].mean()
