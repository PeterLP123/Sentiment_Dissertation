from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.lib.volatility_target import (
    CONTROL_FEATURES,
    SENTIMENT_FEATURE,
    LogVarianceModel,
    VolatilityTargetConfig,
    backtest_single_asset_exposure,
    build_lagged_price_trend,
    build_volatility_frame,
    build_volatility_target_exposure,
    compose_sentiment_risk_exposure,
    fit_log_variance_model,
    multiply_exposures,
    predict_variance,
    qlike_loss,
    walk_forward_variance_forecast,
)


def _sessions(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_volatility_frame_uses_only_prior_returns_for_features() -> None:
    sessions = _sessions(30)
    returns = pd.DataFrame(
        {"session_date": sessions, "gross_return": np.arange(1, 31, dtype=float) / 100.0}
    )
    pressure = pd.DataFrame(
        {"session_date": sessions, "negative_pressure_z": np.linspace(-1.0, 4.0, 30)}
    )
    cfg = VolatilityTargetConfig(forecast_horizon=3, weekly_window=2, monthly_window=3)

    frame = build_volatility_frame(returns, pressure, config=cfg)

    row = frame.iloc[3]
    assert row["log_var_daily"] == pytest.approx(np.log(0.03**2 + cfg.variance_floor))
    assert row["log_var_weekly"] == pytest.approx(
        np.log(np.mean([0.02**2, 0.03**2]) + cfg.variance_floor)
    )
    assert row["log_var_monthly"] == pytest.approx(
        np.log(np.mean([0.01**2, 0.02**2, 0.03**2]) + cfg.variance_floor)
    )
    assert row["realized_variance_forward"] == pytest.approx(
        np.mean([0.04**2, 0.05**2, 0.06**2])
    )
    assert row["target_end_date"] == sessions[5]


def test_sentiment_feature_is_positive_and_clipped() -> None:
    sessions = _sessions(25)
    returns = pd.DataFrame({"session_date": sessions, "gross_return": 0.01})
    pressure = pd.DataFrame(
        {
            "session_date": sessions,
            "negative_pressure_z": np.r_[-2.0, 0.5, 5.0, np.repeat(1.0, 22)],
        }
    )
    frame = build_volatility_frame(returns, pressure)
    assert frame.loc[:2, SENTIMENT_FEATURE].tolist() == [0.0, 0.5, 3.0]


def test_model_fit_and_prediction_are_positive_and_development_only() -> None:
    n = 80
    sessions = _sessions(n)
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "session_date": sessions,
            "log_var_daily": rng.normal(-9.0, 0.2, n),
            "log_var_weekly": rng.normal(-9.0, 0.1, n),
            "log_var_monthly": rng.normal(-9.0, 0.1, n),
            SENTIMENT_FEATURE: rng.uniform(0.0, 2.0, n),
        }
    )
    log_y = -1.0 + 0.3 * frame["log_var_monthly"] + 0.4 * frame[SENTIMENT_FEATURE]
    frame["log_realized_variance_forward"] = log_y
    frame["realized_variance_forward"] = np.exp(log_y)
    train = frame["session_date"].le(sessions[59])

    model = fit_log_variance_model(frame, [*CONTROL_FEATURES, SENTIMENT_FEATURE], train)
    prediction = predict_variance(frame, model)

    assert model.n_train == 60
    assert model.coefficient_map()[SENTIMENT_FEATURE] == pytest.approx(0.4, abs=1e-10)
    assert prediction.notna().all()
    assert prediction.gt(0).all()


def test_volatility_target_is_bounded_and_never_levered() -> None:
    frame = pd.DataFrame({"session_date": _sessions(3)})
    cfg = VolatilityTargetConfig(minimum_exposure=0.25, maximum_exposure=1.0)
    forecast = pd.Series([1e-8, (0.10 / np.sqrt(252)) ** 2, 1.0])

    exposure = build_volatility_target_exposure(frame, forecast, config=cfg, arm="test")

    assert exposure["exposure"].tolist() == pytest.approx([1.0, 1.0, 0.25])
    assert exposure["exposure"].between(0.0, 1.0).all()


def test_single_asset_costs_include_entry_transition_and_liquidation() -> None:
    sessions = _sessions(3)
    returns = pd.DataFrame({"session_date": sessions, "forward_return": [0.0, 0.0, 0.0]})
    exposure = pd.DataFrame({"session_date": sessions, "exposure": [1.0, 0.5, 0.5]})

    daily = backtest_single_asset_exposure(returns, exposure, cost_bps_per_side=10.0)

    assert daily["turnover"].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert daily["cost"].sum() == pytest.approx(0.002)
    assert daily["net_return"].sum() == pytest.approx(-0.002)


def test_lagged_price_trend_does_not_use_current_or_previous_return() -> None:
    sessions = _sessions(9)
    returns = pd.DataFrame(
        {"session_date": sessions, "gross_return": np.repeat(0.01, len(sessions))}
    )
    altered = returns.copy()
    altered.loc[5, "gross_return"] = -0.50

    original_trend = build_lagged_price_trend(returns, lookback=3)
    altered_trend = build_lagged_price_trend(altered, lookback=3)

    pd.testing.assert_series_equal(
        original_trend.loc[:6, "exposure"],
        altered_trend.loc[:6, "exposure"],
    )
    assert original_trend.loc[7, "formation_price"] != pytest.approx(
        altered_trend.loc[7, "formation_price"]
    )


def test_multiply_exposures_is_aligned_and_never_levered() -> None:
    sessions = _sessions(3)
    base = pd.DataFrame({"session_date": sessions, "exposure": [1.0, 0.6, 0.2]})
    modifier = pd.DataFrame(
        {"session_date": sessions, "exposure": [0.25, 1.0, 0.0]}
    )

    combined = multiply_exposures(base, modifier, arm="combined")

    assert combined["arm"].eq("combined").all()
    assert combined["exposure"].tolist() == pytest.approx([0.25, 0.6, 0.0])
    assert combined["exposure"].between(0.0, 1.0).all()


def test_sentiment_composition_distinguishes_multiplier_from_absolute_cap() -> None:
    sessions = _sessions(3)
    base = pd.DataFrame({"session_date": sessions, "exposure": [1.0, 0.8, 0.25]})
    modifier = pd.DataFrame(
        {"session_date": sessions, "exposure": [1.0, 0.25, 0.25]}
    )

    product = compose_sentiment_risk_exposure(
        base,
        modifier,
        mode="multiplier",
        arm="product",
    )
    cap = compose_sentiment_risk_exposure(
        base,
        modifier,
        mode="absolute_cap",
        arm="cap",
    )

    assert product["exposure"].tolist() == pytest.approx([1.0, 0.2, 0.0625])
    assert cap["exposure"].tolist() == pytest.approx([1.0, 0.25, 0.25])
    assert cap["composition_mode"].eq("absolute_cap").all()


def test_qlike_prefers_a_correct_variance_forecast() -> None:
    actual = np.array([0.01, 0.04])
    correct = qlike_loss(actual, actual)
    wrong = qlike_loss(actual, np.array([0.04, 0.01]))
    assert correct.mean() < wrong.mean()


def test_predict_variance_keeps_incomplete_rows_missing() -> None:
    model = LogVarianceModel(
        feature_names=("feature",),
        intercept=-8.0,
        coefficients=(0.5,),
        qlike_scale=1.0,
        n_train=10,
    )
    frame = pd.DataFrame({"feature": [0.0, np.nan]})
    prediction = predict_variance(frame, model)
    assert prediction.iloc[0] == pytest.approx(np.exp(-8.0))
    assert np.isnan(prediction.iloc[1])


def test_walk_forward_forecast_uses_only_targets_ending_before_refit() -> None:
    n = 60
    sessions = _sessions(n)
    feature = np.linspace(-9.5, -8.5, n)
    frame = pd.DataFrame(
        {
            "session_date": sessions,
            "target_end_date": pd.Series(sessions).shift(-2),
            "feature": feature,
            "log_realized_variance_forward": -2.0 + 0.5 * feature,
        }
    )
    frame["realized_variance_forward"] = np.exp(
        frame["log_realized_variance_forward"]
    )

    forecasts, audit = walk_forward_variance_forecast(
        frame,
        ["feature"],
        initial_training_rows=20,
        refit_every=10,
    )

    assert forecasts.iloc[0]["session_date"] == sessions[20]
    assert len(forecasts) == 40
    assert forecasts["forecast_variance"].gt(0).all()
    assert (
        audit["maximum_training_target_end"] < audit["refit_date"]
    ).all()
    assert audit["n_train"].is_monotonic_increasing
