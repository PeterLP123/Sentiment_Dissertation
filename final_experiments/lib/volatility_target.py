"""Sentiment-augmented market-volatility forecasts and exposure sizing.

Notebook 21 asks a deliberately narrower question than the earlier directional
strategies: does aggregate negative-news pressure improve a conventional
HAR-style forecast of near-term SPY variance, and is that incremental forecast
useful when sizing an otherwise independent long-SPY exposure?

The implementation keeps the research invariants visible:

* return features end at the current open and never use the forward return;
* model coefficients and variance calibration are fitted on development only;
* sentiment is one predeclared clipped positive-pressure feature;
* both policies use the same target, bounds, returns, and cost accounting; and
* the portfolio is long-or-cash and never levered above one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR = 252
CONTROL_FEATURES = ("log_var_daily", "log_var_weekly", "log_var_monthly")
SENTIMENT_FEATURE = "positive_negative_pressure_z"
RiskCompositionMode = Literal["multiplier", "absolute_cap"]


@dataclass(frozen=True)
class VolatilityTargetConfig:
    """Frozen design inputs for the Notebook 21 experiment."""

    forecast_horizon: int = 5
    weekly_window: int = 5
    monthly_window: int = 22
    variance_floor: float = 1e-8
    sentiment_clip: float = 3.0
    target_annual_volatility: float = 0.10
    minimum_exposure: float = 0.25
    maximum_exposure: float = 1.00
    cost_bps_per_side: float = 2.0
    bootstrap_block_length: int = 20
    bootstrap_replications: int = 4_999
    random_seed: int = 20260805

    def __post_init__(self) -> None:
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive")
        if not 1 < self.weekly_window < self.monthly_window:
            raise ValueError("windows must satisfy 1 < weekly < monthly")
        if self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        if self.sentiment_clip <= 0:
            raise ValueError("sentiment_clip must be positive")
        if self.target_annual_volatility <= 0:
            raise ValueError("target_annual_volatility must be positive")
        if not 0 <= self.minimum_exposure <= self.maximum_exposure <= 1:
            raise ValueError("exposure bounds must satisfy 0 <= min <= max <= 1")
        if self.cost_bps_per_side < 0:
            raise ValueError("cost_bps_per_side must be non-negative")
        if self.bootstrap_block_length < 1 or self.bootstrap_replications < 1:
            raise ValueError("bootstrap settings must be positive")


@dataclass(frozen=True)
class LogVarianceModel:
    """Development-fitted log-variance model with QLIKE calibration."""

    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    qlike_scale: float
    n_train: int

    def coefficient_map(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.coefficients, strict=True))


def build_volatility_frame(
    returns: pd.DataFrame,
    pressure: pd.DataFrame,
    *,
    config: VolatilityTargetConfig | None = None,
    return_col: str = "gross_return",
) -> pd.DataFrame:
    """Build causal HAR features and a forward realised-variance target.

    Row ``t`` is a decision at the adjusted open on ``session_date``.  The
    daily/weekly/monthly variance features use returns ending at that open
    (``return_col.shift(1)`` and earlier).  The outcome is the mean squared
    return from ``t`` through ``t + forecast_horizon - 1`` and is used only for
    fitting/scoring, never to construct row-t exposure.
    """

    cfg = config or VolatilityTargetConfig()
    need_returns = {"session_date", return_col}
    need_pressure = {"session_date", "negative_pressure_z"}
    if missing := need_returns - set(returns.columns):
        raise ValueError(f"returns missing columns: {sorted(missing)}")
    if missing := need_pressure - set(pressure.columns):
        raise ValueError(f"pressure missing columns: {sorted(missing)}")

    ret = returns.loc[:, ["session_date", return_col]].copy()
    news = pressure.loc[:, ["session_date", "negative_pressure_z"]].copy()
    for frame in (ret, news):
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
        if frame["session_date"].duplicated().any():
            raise ValueError("inputs must be unique by session_date")
    out = (
        ret.merge(news, on="session_date", how="left", validate="1:1")
        .sort_values("session_date", kind="mergesort")
        .reset_index(drop=True)
        .rename(columns={return_col: "forward_return"})
    )

    squared = out["forward_return"].astype(float).pow(2)
    known_squared = squared.shift(1)
    out["log_var_daily"] = np.log(known_squared + cfg.variance_floor)
    out["log_var_weekly"] = np.log(
        known_squared.rolling(cfg.weekly_window, min_periods=cfg.weekly_window).mean()
        + cfg.variance_floor
    )
    out["log_var_monthly"] = np.log(
        known_squared.rolling(cfg.monthly_window, min_periods=cfg.monthly_window).mean()
        + cfg.variance_floor
    )
    out[SENTIMENT_FEATURE] = (
        out["negative_pressure_z"].clip(lower=0.0, upper=cfg.sentiment_clip)
    )

    forward_squares = pd.concat(
        [squared.shift(-step) for step in range(cfg.forecast_horizon)],
        axis=1,
    )
    complete_target = forward_squares.notna().all(axis=1)
    out["realized_variance_forward"] = forward_squares.mean(axis=1).where(complete_target)
    out["log_realized_variance_forward"] = np.log(
        out["realized_variance_forward"] + cfg.variance_floor
    )
    out["target_end_date"] = out["session_date"].shift(-(cfg.forecast_horizon - 1))
    out.loc[~complete_target, "target_end_date"] = pd.NaT
    return out


def _valid_model_rows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    row_mask: pd.Series | np.ndarray,
) -> pd.Series:
    columns = [*feature_names, "log_realized_variance_forward", "realized_variance_forward"]
    if missing := set(columns) - set(frame.columns):
        raise ValueError(f"model frame missing columns: {sorted(missing)}")
    mask = pd.Series(np.asarray(row_mask, dtype=bool), index=frame.index)
    finite = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
    return mask & finite


def fit_log_variance_model(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    row_mask: pd.Series | np.ndarray,
) -> LogVarianceModel:
    """Fit OLS in log variance and a positive development-only QLIKE scale."""

    names = tuple(feature_names)
    if not names:
        raise ValueError("at least one feature is required")
    valid = _valid_model_rows(frame, names, row_mask)
    if int(valid.sum()) <= len(names) + 1:
        raise ValueError("insufficient complete training rows")
    x = frame.loc[valid, list(names)].to_numpy(dtype=float)
    y_log = frame.loc[valid, "log_realized_variance_forward"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y_log, rcond=None)
    raw_variance = np.exp(design @ beta)
    actual_variance = frame.loc[valid, "realized_variance_forward"].to_numpy(dtype=float)
    scale = float(np.mean(actual_variance / np.maximum(raw_variance, 1e-16)))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("variance calibration is not finite and positive")
    return LogVarianceModel(
        feature_names=names,
        intercept=float(beta[0]),
        coefficients=tuple(float(value) for value in beta[1:]),
        qlike_scale=scale,
        n_train=int(valid.sum()),
    )


def predict_variance(frame: pd.DataFrame, model: LogVarianceModel) -> pd.Series:
    """Apply a frozen model; incomplete feature rows remain missing."""

    if missing := set(model.feature_names) - set(frame.columns):
        raise ValueError(f"prediction frame missing columns: {sorted(missing)}")
    x = frame.loc[:, list(model.feature_names)].to_numpy(dtype=float)
    finite = np.isfinite(x).all(axis=1)
    prediction = np.full(len(frame), np.nan, dtype=float)
    if finite.any():
        linear = model.intercept + x[finite] @ np.asarray(model.coefficients, dtype=float)
        prediction[finite] = np.exp(linear) * model.qlike_scale
    return pd.Series(prediction, index=frame.index, name="forecast_variance")


def walk_forward_variance_forecast(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    initial_training_rows: int = 504,
    refit_every: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce expanding, prior-target-only variance forecasts.

    The first decision follows ``initial_training_rows`` chronological rows.
    At each refit date, a target is eligible only when its complete
    ``target_end_date`` is strictly earlier than the decision date.  The fitted
    model is then held fixed for ``refit_every`` decisions.  This makes the
    training boundary auditable while avoiding a misleading full-sample
    in-sample development path.
    """

    names = tuple(feature_names)
    if not names:
        raise ValueError("at least one feature is required")
    if initial_training_rows < 2:
        raise ValueError("initial_training_rows must be at least two")
    if refit_every < 1:
        raise ValueError("refit_every must be positive")
    required = {
        "session_date",
        "target_end_date",
        "log_realized_variance_forward",
        "realized_variance_forward",
        *names,
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"model frame missing columns: {sorted(missing)}")

    ordered = frame.copy()
    ordered["session_date"] = pd.to_datetime(ordered["session_date"]).dt.normalize()
    ordered["target_end_date"] = pd.to_datetime(ordered["target_end_date"]).dt.normalize()
    if ordered["session_date"].duplicated().any():
        raise ValueError("model frame must be unique by session_date")
    ordered = ordered.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    if initial_training_rows >= len(ordered):
        raise ValueError("initial_training_rows must leave at least one forecast row")

    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for fit_number, start in enumerate(
        range(initial_training_rows, len(ordered), refit_every),
        start=1,
    ):
        stop = min(start + refit_every, len(ordered))
        refit_date = ordered.loc[start, "session_date"]
        train_mask = ordered["target_end_date"].notna() & ordered["target_end_date"].lt(
            refit_date
        )
        model = fit_log_variance_model(ordered, names, train_mask)
        block = ordered.iloc[start:stop]
        predictions = predict_variance(block, model)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "session_date": block["session_date"].to_numpy(),
                    "forecast_variance": predictions.to_numpy(),
                    "fit_number": fit_number,
                    "refit_date": refit_date,
                }
            )
        )
        audit_rows.append(
            {
                "fit_number": fit_number,
                "refit_date": refit_date,
                "forecast_start": block["session_date"].min(),
                "forecast_end": block["session_date"].max(),
                "n_forecast_rows": len(block),
                "n_train": model.n_train,
                "maximum_training_target_end": ordered.loc[
                    train_mask, "target_end_date"
                ].max(),
                "qlike_scale": model.qlike_scale,
                **model.coefficient_map(),
            }
        )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(audit_rows)


def qlike_loss(actual_variance: np.ndarray, forecast_variance: np.ndarray) -> np.ndarray:
    """Gaussian QLIKE variance loss (constants omitted)."""

    actual = np.asarray(actual_variance, dtype=float)
    forecast = np.asarray(forecast_variance, dtype=float)
    if np.any(~np.isfinite(actual)) or np.any(~np.isfinite(forecast)):
        raise ValueError("QLIKE inputs must be finite")
    if np.any(actual < 0) or np.any(forecast <= 0):
        raise ValueError("QLIKE requires actual >= 0 and forecast > 0")
    return actual / forecast + np.log(forecast)


def build_volatility_target_exposure(
    frame: pd.DataFrame,
    forecast_variance: pd.Series | np.ndarray,
    *,
    config: VolatilityTargetConfig | None = None,
    arm: str,
) -> pd.DataFrame:
    """Translate a variance forecast into an unlevered long-or-cash exposure."""

    cfg = config or VolatilityTargetConfig()
    forecast = np.asarray(forecast_variance, dtype=float)
    if len(forecast) != len(frame):
        raise ValueError("forecast length must match frame")
    annual_vol = np.sqrt(forecast * TRADING_SESSIONS_PER_YEAR)
    raw = cfg.target_annual_volatility / annual_vol
    exposure = np.clip(raw, cfg.minimum_exposure, cfg.maximum_exposure)
    exposure[~np.isfinite(annual_vol) | (annual_vol <= 0)] = np.nan
    return pd.DataFrame(
        {
            "session_date": pd.to_datetime(frame["session_date"]).dt.normalize(),
            "arm": arm,
            "forecast_variance": forecast,
            "forecast_annual_volatility": annual_vol,
            "exposure": exposure,
        }
    )


def backtest_single_asset_exposure(
    returns: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    cost_bps_per_side: float,
    return_col: str = "forward_return",
) -> pd.DataFrame:
    """Backtest a long/cash exposure with entry, transition and liquidation costs."""

    if cost_bps_per_side < 0:
        raise ValueError("cost_bps_per_side must be non-negative")
    need_returns = {"session_date", return_col}
    need_exposure = {"session_date", "exposure"}
    if missing := need_returns - set(returns.columns):
        raise ValueError(f"returns missing columns: {sorted(missing)}")
    if missing := need_exposure - set(exposure.columns):
        raise ValueError(f"exposure missing columns: {sorted(missing)}")

    ret = returns.loc[:, ["session_date", return_col]].copy()
    exp = exposure.copy()
    for part in (ret, exp):
        part["session_date"] = pd.to_datetime(part["session_date"]).dt.normalize()
        if part["session_date"].duplicated().any():
            raise ValueError("inputs must be unique by session_date")
    daily = (
        ret.merge(exp, on="session_date", how="inner", validate="1:1")
        .dropna(subset=[return_col, "exposure"])
        .sort_values("session_date", kind="mergesort")
        .reset_index(drop=True)
    )
    if daily.empty:
        return daily
    if not daily["exposure"].between(0.0, 1.0).all():
        raise ValueError("exposure must remain in [0, 1]")

    weights = daily["exposure"].to_numpy(dtype=float)
    previous = np.r_[0.0, weights[:-1]]
    turnover = 0.5 * np.abs(weights - previous)
    turnover[-1] += 0.5 * abs(weights[-1])
    gross_return = weights * daily[return_col].to_numpy(dtype=float)
    cost = 2.0 * turnover * (cost_bps_per_side / 10_000.0)
    daily["gross_exposure"] = weights
    daily["net_exposure"] = weights
    daily["n_positions"] = (weights > 0).astype(int)
    daily["gross_return"] = gross_return
    daily["turnover"] = turnover
    daily["cost"] = cost
    daily["net_return"] = gross_return - cost
    daily["missing_weight_share"] = 0.0
    return daily


def build_lagged_price_trend(
    returns: pd.DataFrame,
    *,
    lookback: int = 200,
    return_col: str = "gross_return",
) -> pd.DataFrame:
    """Build a one-session-lagged moving-average trend exposure.

    ``return_col`` on row ``t`` is the return from the row-t open to the next
    open.  The reconstructed relative price on row ``t`` is therefore known at
    that open.  The trend decision deliberately uses the relative price and
    moving average through row ``t - 1`` so signal formation and execution do
    not share the same price observation.  Incomplete warm-up rows remain
    missing instead of being treated as risk-on.
    """

    if lookback < 2:
        raise ValueError("lookback must be at least two sessions")
    need = {"session_date", return_col}
    if missing := need - set(returns.columns):
        raise ValueError(f"returns missing columns: {sorted(missing)}")
    out = returns.loc[:, ["session_date", return_col]].copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.normalize()
    if out["session_date"].duplicated().any():
        raise ValueError("returns must be unique by session_date")
    out = out.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    gross_growth = 1.0 + out[return_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(gross_growth)) or np.any(gross_growth <= 0):
        raise ValueError("returns must be finite and greater than -100%")

    relative_price = pd.Series(gross_growth).cumprod().shift(1, fill_value=1.0)
    formation_price = relative_price.shift(1)
    formation_average = formation_price.rolling(
        lookback,
        min_periods=lookback,
    ).mean()
    complete = formation_price.notna() & formation_average.notna()
    trend_on = formation_price.ge(formation_average).astype(float).where(complete)
    return pd.DataFrame(
        {
            "session_date": out["session_date"],
            "relative_price": relative_price,
            "formation_price": formation_price,
            "formation_average": formation_average,
            "trend_on": trend_on,
            "exposure": trend_on,
        }
    )


def multiply_exposures(
    base: pd.DataFrame,
    modifier: pd.DataFrame,
    *,
    arm: str,
) -> pd.DataFrame:
    """Multiply two complete, unlevered exposure paths on matching sessions."""

    required = {"session_date", "exposure"}
    if missing := required - set(base.columns):
        raise ValueError(f"base exposure missing columns: {sorted(missing)}")
    if missing := required - set(modifier.columns):
        raise ValueError(f"modifier exposure missing columns: {sorted(missing)}")
    left = base.loc[:, ["session_date", "exposure"]].rename(
        columns={"exposure": "base_exposure"}
    )
    right = modifier.loc[:, ["session_date", "exposure"]].rename(
        columns={"exposure": "modifier_exposure"}
    )
    for frame in (left, right):
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
        if frame["session_date"].duplicated().any():
            raise ValueError("exposure inputs must be unique by session_date")
    out = left.merge(right, on="session_date", how="left", validate="1:1")
    if out["modifier_exposure"].isna().any():
        raise ValueError("modifier exposure does not cover every base session")
    if not left["base_exposure"].between(0.0, 1.0).all():
        raise ValueError("base exposure must remain in [0, 1]")
    if not right["modifier_exposure"].between(0.0, 1.0).all():
        raise ValueError("modifier exposure must remain in [0, 1]")
    out["arm"] = arm
    out["exposure"] = out["base_exposure"] * out["modifier_exposure"]
    return out


def compose_sentiment_risk_exposure(
    base: pd.DataFrame,
    modifier: pd.DataFrame,
    *,
    mode: RiskCompositionMode,
    arm: str,
) -> pd.DataFrame:
    """Combine a base exposure with a frozen sentiment risk state.

    ``multiplier`` reproduces Notebook 24: the risk-off value scales whatever
    exposure the base requested. ``absolute_cap`` instead interprets the same
    risk-off value as a maximum permitted exposure. The latter preserves the
    25% floor/cap semantics of the two component policies rather than allowing
    their product to fall as low as 6.25%.
    """

    if mode not in {"multiplier", "absolute_cap"}:
        raise ValueError("mode must be 'multiplier' or 'absolute_cap'")
    out = multiply_exposures(base, modifier, arm=arm)
    if mode == "absolute_cap":
        out["exposure"] = np.minimum(
            out["base_exposure"].to_numpy(dtype=float),
            out["modifier_exposure"].to_numpy(dtype=float),
        )
    out["composition_mode"] = mode
    return out


def paired_circular_block_mean(
    values: np.ndarray | pd.Series,
    *,
    block_length: int,
    replications: int,
    seed: int,
) -> dict[str, float | int]:
    """Circular block-bootstrap interval and centred two-sided p-value."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n < 2:
        return {
            "n": n,
            "mean": float(np.mean(array)) if n else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_two_sided": float("nan"),
        }
    if not 1 <= block_length < n:
        raise ValueError("block_length must be in [1, n)")
    if replications < 1:
        raise ValueError("replications must be positive")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_length))
    means = np.empty(replications, dtype=float)
    for replication in range(replications):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate(
            [np.arange(start, start + block_length) % n for start in starts]
        )[:n]
        means[replication] = float(array[indices].mean())
    point = float(array.mean())
    centred = means - point
    return {
        "n": n,
        "mean": point,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_two_sided": float(
            (1 + np.sum(np.abs(centred) >= abs(point))) / (replications + 1)
        ),
        "block_length": block_length,
        "replications": replications,
        "seed": seed,
    }


def model_table(models: dict[str, LogVarianceModel]) -> pd.DataFrame:
    """Return a compact coefficient/calibration table for notebook display."""

    rows: list[dict[str, Any]] = []
    for arm, model in models.items():
        row: dict[str, Any] = {
            "arm": arm,
            "n_train": model.n_train,
            "intercept": model.intercept,
            "qlike_scale": model.qlike_scale,
        }
        row.update(model.coefficient_map())
        rows.append(row)
    return pd.DataFrame(rows)
