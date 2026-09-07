"""Reusable primitives for the FNSPID sentiment-conditioned tail-risk experiment.

Only the mechanical, unit-testable pieces of the workflow live here: leak-safe
session timing, the GJR-GARCH(1,1) variance recursion, the joint VaR/ES
parameterisation and its Fissler-Ziegel FZ0 loss, the date-block bootstrap, and
the standard backtest statistics.  Every empirical decision, table, and figure
stays visible in ``notebooks/fnsipid_tail_risk_core.py``.

Sign conventions used throughout:

* returns are close-to-close log returns of adjusted prices;
* ``z`` is a standardised return, so the lower tail is negative;
* a VaR forecast ``v`` and an Expected Shortfall forecast ``e`` at level
  ``alpha`` satisfy ``e < v < 0`` in standardised units.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

__all__ = [
    "GjrParams",
    "JointTailFit",
    "apply_news_scale_adjustment",
    "block_bootstrap_indices",
    "build_next_session_targets",
    "christoffersen_tests",
    "date_block_bootstrap_mean",
    "es_identification_residual",
    "fit_joint_var_es",
    "fit_news_scale_adjustment",
    "fz0_loss",
    "gjr_conditional_variances",
    "gjr_conditional_variances_piecewise",
    "kupiec_test",
    "map_dates_to_reaction_sessions",
    "pinball_loss",
    "qlike_loss",
    "repair_adjusted_close",
    "tail_forecasts",
    "prepare_clean_output_directory",
    "resolve_tail_risk_variant",
    "select_expanding_refit_window",
    "verify_input_file",
    "verify_output_manifest",
    "verify_strictly_after",
]


def prepare_clean_output_directory(path: str | Path) -> Path:
    """Create an output directory or accept it only when it is empty."""

    output = Path(path)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise FileExistsError(f"refusing to reuse non-empty output directory: {output}")
    else:
        output.mkdir(parents=True)
    return output


TAIL_RISK_VARIANTS: dict[str, tuple[str, str, str]] = {
    "v1": ("none", "frozen_development", "the pre-registered specification"),
    "price_only": (
        "min_abs_return",
        "frozen_development",
        "v1, changing only the adjusted-price repair",
    ),
    "refit_only": (
        "none",
        "annual_expanding",
        "v1, changing only the volatility-filter refit policy",
    ),
    "v2": (
        "min_abs_return",
        "annual_expanding",
        "v1, changing both the adjusted-price repair and the volatility-filter refit policy",
    ),
}


def resolve_tail_risk_variant(value: str) -> tuple[str, str, str, str]:
    """Resolve one frozen cell of the 2 x 2 defect-attribution design."""

    variant = value.strip().lower()
    try:
        price_repair, volatility_refit, relative_to = TAIL_RISK_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f"variant must be one of {sorted(TAIL_RISK_VARIANTS)}") from error
    return variant, price_repair, volatility_refit, relative_to


def verify_input_file(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
) -> str:
    """Fail closed unless a local input matches its frozen identity."""

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    observed_size = candidate.stat().st_size
    if expected_size is not None and observed_size != expected_size:
        raise ValueError(f"{candidate} has {observed_size} bytes; expected {expected_size}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{candidate} SHA-256 mismatch")
    return observed_sha256


def verify_output_manifest(
    root: str | Path,
    outputs: Mapping[str, Mapping[str, Any]],
    *,
    excluded_names: Sequence[str] = (
        "manifest.json",
        "fnsipid_tail_risk_core.executed.ipynb",
    ),
) -> int:
    """Fail unless a final output bundle exactly matches its manifest records."""

    directory = Path(root)
    excluded = set(excluded_names)
    actual = {str(path.relative_to(directory)): path for path in directory.rglob("*") if path.is_file() and path.name not in excluded}
    expected = set(outputs)
    missing = sorted(expected - actual.keys())
    unexpected = sorted(actual.keys() - expected)
    mismatched: list[str] = []
    for relative in sorted(expected & actual.keys()):
        path = actual[relative]
        record = outputs[relative]
        try:
            verify_input_file(
                path,
                expected_sha256=str(record["sha256"]),
                expected_size=int(record["size_bytes"]),
            )
        except (FileNotFoundError, ValueError) as error:
            mismatched.append(f"{relative}: {error}")
    if missing or unexpected or mismatched:
        raise ValueError(f"output manifest mismatch: missing={missing}, unexpected={unexpected}, mismatched={mismatched}")
    return len(actual)


def select_expanding_refit_window(
    returns: pd.Series,
    *,
    development_start: str | pd.Timestamp,
    evaluation_year: int,
) -> pd.Series:
    """Select returns known strictly before one evaluation year."""

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    cutoff = pd.Timestamp(year=int(evaluation_year), month=1, day=1)
    start = pd.Timestamp(development_start)
    if cutoff <= start:
        raise ValueError("evaluation_year must follow development_start")
    return returns[(returns.index >= start) & (returns.index < cutoff)]


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def map_dates_to_reaction_sessions(
    calendar_dates: Sequence[Any],
    sessions: Sequence[Any],
) -> np.ndarray:
    """Map each calendar date to the first session strictly after it.

    ``sessions`` must be sorted, unique trading sessions.  Dates with no
    session strictly after them map to ``NaT`` so the caller can drop them
    explicitly instead of silently reusing the final session.
    """

    session_index = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize()
    if not session_index.is_monotonic_increasing or session_index.has_duplicates:
        raise ValueError("sessions must be sorted and unique")
    wanted = pd.DatetimeIndex(pd.to_datetime(list(calendar_dates))).normalize()
    positions = session_index.searchsorted(wanted, side="right")
    mapped = np.full(len(wanted), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    inside = positions < len(session_index)
    mapped[inside] = session_index.to_numpy()[positions[inside]]
    return mapped


def verify_strictly_after(
    calendar_dates: Sequence[Any],
    mapped_sessions: Sequence[Any],
) -> int:
    """Return the number of mappings that are not strictly forward in time."""

    origin = pd.DatetimeIndex(pd.to_datetime(list(calendar_dates))).normalize()
    mapped = pd.DatetimeIndex(pd.to_datetime(list(mapped_sessions))).normalize()
    if len(origin) != len(mapped):
        raise ValueError("calendar_dates and mapped_sessions must be the same length")
    observed = mapped.notna()
    return int((mapped[observed] <= origin[observed]).sum())


def build_next_session_targets(prices: pd.DataFrame) -> pd.DataFrame:
    """Attach next-session close-to-close targets to one firm's price frame.

    ``prices`` must hold ``session_date`` and strictly positive
    ``adjusted_close`` columns.  The returned frame keeps one row per session
    and adds:

    ``log_return``
        return realised *during* the row's session, known at its close;
    ``target_date`` / ``target_return``
        the next available session and the close-to-close log return from this
        session's close to that session's close.

    The final session has no target and is returned with ``NaT``/``NaN`` so the
    caller decides how to drop it.
    """

    required = {"session_date", "adjusted_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"price frame is missing columns: {sorted(missing)}")
    frame = prices.loc[:, ["session_date", "adjusted_close"]].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    if frame["session_date"].duplicated().any():
        raise ValueError("price frame contains duplicate session dates")
    frame = frame.sort_values("session_date", kind="stable").reset_index(drop=True)
    if not (frame["adjusted_close"] > 0).all():
        raise ValueError("adjusted_close must be strictly positive")

    log_price = np.log(frame["adjusted_close"].to_numpy(dtype=float))
    frame["log_return"] = np.concatenate(([np.nan], np.diff(log_price)))
    frame["target_return"] = np.concatenate((np.diff(log_price), [np.nan]))
    frame["target_date"] = frame["session_date"].shift(-1)
    return frame


# --------------------------------------------------------------------------
# GJR-GARCH(1,1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GjrParams:
    """Frozen GJR-GARCH(1,1) parameters in the units of the fitted returns."""

    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float = math.inf

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def unconditional_variance(self) -> float:
        slack = 1.0 - self.persistence
        return self.omega / slack if slack > 0 else math.nan


def gjr_conditional_variances(
    returns: Sequence[float] | np.ndarray,
    params: GjrParams,
    *,
    initial_variance: float,
) -> np.ndarray:
    """Run the one-step-ahead GJR recursion over an ordered return series.

    Returns an array of length ``len(returns) + 1``.  Element ``t`` for
    ``t < len(returns)`` is the variance of ``returns[t]`` conditional on
    information through ``t - 1``; the final element is the one-step-ahead
    forecast for the first unobserved return.  Nothing at index ``t`` uses
    ``returns[t]`` or later, which is what makes the forecasts point-in-time.
    """

    series = np.asarray(returns, dtype=float)
    if series.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.isfinite(series).all():
        raise ValueError("returns must be finite; drop or impute before recursing")
    if not initial_variance > 0:
        raise ValueError("initial_variance must be strictly positive")

    sigma2 = np.empty(series.size + 1, dtype=float)
    sigma2[0] = initial_variance
    current = initial_variance
    for index in range(series.size):
        shock = series[index] - params.mu
        squared = shock * shock
        current = params.omega + params.alpha * squared + (params.gamma * squared if shock < 0.0 else 0.0) + params.beta * current
        if not current > 0:
            raise ValueError("GJR recursion produced a non-positive variance")
        sigma2[index + 1] = current
    return sigma2


def gjr_conditional_variances_piecewise(
    returns: Sequence[float] | np.ndarray,
    params_per_step: Sequence[GjrParams],
    *,
    initial_variance: float,
) -> np.ndarray:
    """GJR recursion whose parameters may change between steps.

    ``params_per_step[t]`` governs the update that produces the variance of
    ``returns[t + 1]`` from ``returns[t]``.  This supports a filter that is
    periodically re-estimated on an expanding window: the variance state is
    carried across a parameter switch rather than restarted, and no step ever
    consults a return at or beyond the one it is forecasting.

    Passing the same parameters at every step reproduces
    :func:`gjr_conditional_variances` exactly.
    """

    series = np.asarray(returns, dtype=float)
    schedule = list(params_per_step)
    if series.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if len(schedule) != series.size:
        raise ValueError("params_per_step must supply one parameter set per return")
    if not np.isfinite(series).all():
        raise ValueError("returns must be finite; drop or impute before recursing")
    if not initial_variance > 0:
        raise ValueError("initial_variance must be strictly positive")

    sigma2 = np.empty(series.size + 1, dtype=float)
    sigma2[0] = initial_variance
    current = initial_variance
    for index in range(series.size):
        params = schedule[index]
        shock = series[index] - params.mu
        squared = shock * shock
        current = params.omega + params.alpha * squared + (params.gamma * squared if shock < 0.0 else 0.0) + params.beta * current
        if not current > 0:
            raise ValueError("GJR recursion produced a non-positive variance")
        sigma2[index + 1] = current
    return sigma2


def repair_adjusted_close(
    session_dates: Sequence[Any],
    adjusted_close: Sequence[float] | np.ndarray,
    raw_close: Sequence[float] | np.ndarray,
    *,
    gap_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild an adjusted-close series where its adjustment factor misbehaves.

    In a correctly adjusted series the adjusted-close and raw-close log returns
    differ only on ex-dividend and split dates.  A large gap on an ordinary day
    means the cumulative adjustment factor has stepped inconsistently, and the
    adjusted "return" is an artifact rather than a price move.

    On a flagged day the smaller-magnitude of the two returns is taken.  That is
    correct in both directions:

    * on a genuine split the raw return is the artifact (a 2-for-1 shows as
      -69% raw, ~0% adjusted) and the adjusted return is kept;
    * on an adjustment defect the adjusted return is the artifact (Merck on
      2020-07-06: -3.7% raw, -14.7% adjusted) and the raw return is kept.

    Returns the repaired price series, anchored at the first adjusted close, and
    a boolean mask of the days whose return was replaced.  Days that are not
    flagged are reproduced to floating-point exactness.
    """

    dates = pd.DatetimeIndex(pd.to_datetime(list(session_dates)))
    adjusted = np.asarray(adjusted_close, dtype=float)
    raw = np.asarray(raw_close, dtype=float)
    if not (dates.size == adjusted.size == raw.size):
        raise ValueError("session_dates, adjusted_close and raw_close must be the same length")
    if not dates.is_monotonic_increasing:
        raise ValueError("session_dates must be sorted")
    if adjusted.size == 0:
        return adjusted.copy(), np.zeros(0, dtype=bool)
    if np.any(adjusted <= 0.0) or np.any(raw <= 0.0):
        raise ValueError("prices must be strictly positive")
    if not gap_threshold > 0.0:
        raise ValueError("gap_threshold must be positive")

    adjusted_returns = np.diff(np.log(adjusted))
    raw_returns = np.diff(np.log(raw))
    flagged = np.abs(adjusted_returns - raw_returns) > gap_threshold
    prefer_raw = flagged & (np.abs(raw_returns) < np.abs(adjusted_returns))
    repaired_returns = np.where(prefer_raw, raw_returns, adjusted_returns)

    repaired = np.empty_like(adjusted)
    repaired[0] = adjusted[0]
    repaired[1:] = adjusted[0] * np.exp(np.cumsum(repaired_returns))
    replaced = np.concatenate(([False], prefer_raw))
    return repaired, replaced


# --------------------------------------------------------------------------
# Joint VaR / ES
# --------------------------------------------------------------------------

# Numerical guards on the linear indices.  These are overflow/underflow rails,
# NOT a modelling band: a narrow clip creates a flat region whose gradient is
# zero, and a gradient optimiser that wanders into it freezes there.  The
# ceiling therefore uses a tangent (linear) extension, which keeps the map
# strictly increasing with a strictly positive derivative everywhere, and the
# floor sits far below any plausible optimum so ``exp`` never underflows to
# exactly zero (which would break ``es < var``).
LINEAR_INDEX_FLOOR = -700.0
LINEAR_INDEX_CEILING = 40.0
# Minimum ES gap as a fraction of the VaR magnitude.  Four orders of magnitude
# above float64 epsilon, and eight below any economically meaningful gap, so it
# only ever prevents ``es`` from being absorbed into ``var`` by rounding.
ES_GAP_RELATIVE_FLOOR = 1e-8
_CEILING_VALUE = math.exp(LINEAR_INDEX_CEILING)


def _positive_exponential(index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly positive, strictly increasing, C1 surrogate for ``exp``.

    Returns the value, its derivative, and a mask marking entries that fell
    outside the safe band and were therefore extrapolated rather than
    exponentiated.
    """

    above = index > LINEAR_INDEX_CEILING
    value = np.exp(np.clip(index, LINEAR_INDEX_FLOOR, LINEAR_INDEX_CEILING))
    derivative = value
    if above.any():
        extended = _CEILING_VALUE * (1.0 + (index - LINEAR_INDEX_CEILING))
        value = np.where(above, extended, value)
        derivative = np.where(above, _CEILING_VALUE, derivative)
    return value, derivative, above | (index < LINEAR_INDEX_FLOOR)


def tail_forecasts(
    design: np.ndarray,
    beta_var: np.ndarray,
    beta_gap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Map linear indices to a valid ``(VaR, ES)`` pair with ``es < var < 0``.

    ``var = -exp(X @ beta_var)`` and ``es = var - exp(X @ beta_gap)``.  Both
    exponentials are strictly positive, so the ordering constraint holds by
    construction rather than by penalty.  The third return value counts how
    many rows needed the numerical extension.
    """

    var_magnitude, _, outside_var = _positive_exponential(design @ beta_var)
    gap, _, outside_gap = _positive_exponential(design @ beta_gap)
    var = -var_magnitude
    es = var - (gap + ES_GAP_RELATIVE_FLOOR * var_magnitude)
    return var, es, int(np.count_nonzero(outside_var | outside_gap))


def fz0_loss(
    realised: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fissler-Ziegel ``FZ0`` loss for lower-tail VaR and ES forecasts.

    ``L = -(1 / (alpha * e)) * (v - y)^+ + v / e + log(-e) - 1``

    where ``(v - y)^+ = I(y <= v) * (v - y)``.  This is the scoring function;
    the optimiser's smoothed warm-start variant lives in
    ``_fz0_value_and_gradient`` and is never used to score a forecast.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    realised = np.asarray(realised, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    es_forecast = np.asarray(es_forecast, dtype=float)
    if np.any(es_forecast >= var_forecast) or np.any(var_forecast >= 0.0):
        raise ValueError("FZ0 requires es < var < 0")
    hinge = np.maximum(var_forecast - realised, 0.0)
    return -(1.0 / (alpha * es_forecast)) * hinge + var_forecast / es_forecast + np.log(-es_forecast) - 1.0


def pinball_loss(realised: np.ndarray, var_forecast: np.ndarray, alpha: float) -> np.ndarray:
    """Quantile (pinball) loss at level ``alpha`` for the lower tail."""

    realised = np.asarray(realised, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    excess = realised - var_forecast
    return np.where(excess < 0.0, (alpha - 1.0) * excess, alpha * excess)


def es_identification_residual(
    realised: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Lower-tail ES identification residual with conditional mean zero."""

    realised = np.asarray(realised, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    es_forecast = np.asarray(es_forecast, dtype=float)
    hit = (realised <= var_forecast).astype(float)
    return es_forecast - var_forecast + ((var_forecast - realised) / alpha) * hit


@dataclass
class JointTailFit:
    """Outcome of one joint VaR/ES estimation."""

    name: str
    columns: tuple[str, ...]
    beta_var: np.ndarray
    beta_gap: np.ndarray
    objective: float
    converged: bool
    message: str
    iterations: int
    starts_attempted: int
    starts_converged: int
    start_objective_spread: float
    clipped_rows: int
    warnings: tuple[str, ...] = field(default=())

    def forecast(self, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        var, es, _ = tail_forecasts(design, self.beta_var, self.beta_gap)
        return var, es

    def coefficient_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for block, values in (("var", self.beta_var), ("es_gap", self.beta_gap)):
            for column, value in zip(self.columns, values, strict=True):
                rows.append({"model": self.name, "block": block, "term": column, "coefficient": float(value)})
        return rows


def _fz0_value_and_gradient(
    parameters: np.ndarray,
    design: np.ndarray,
    realised: np.ndarray,
    alpha: float,
    smoothing: float,
) -> tuple[float, np.ndarray]:
    """Mean FZ0 loss and its analytic gradient in the linear-index parameters.

    With ``v = -m(a)``, ``e = v - g(b)``, ``a = X beta_var``, ``b = X beta_gap``
    and ``m``/``g`` the guarded exponential, the chain rule gives
    ``dL/da = (dL/dv + dL/de) * (-m'(a))`` and ``dL/db = dL/de * (-g'(b))``.

    Smoothing replaces the hinge ``(v - y)^+`` with the softplus
    ``h * log(1 + exp((v - y) / h))``, whose derivative in ``v`` is the logistic
    ``sigma((v - y) / h)``.  Smoothing the *indicator alone* instead would let
    ``sigma(u) * (v - y)`` go negative for ``y > v``; the warm start would then
    prefer ``e -> v``, collapsing Expected Shortfall onto VaR.  The softplus
    hinge is non-negative for every observation and converges to the exact FZ0
    hinge as ``h -> 0``.
    """

    width = design.shape[1]
    var_magnitude, var_slope, _ = _positive_exponential(design @ parameters[:width])
    gap, gap_slope, _ = _positive_exponential(design @ parameters[width:])
    var = -var_magnitude
    # A relative floor keeps es strictly below var in float64 even when the gap
    # is many orders of magnitude smaller than the VaR level.
    gap = gap + ES_GAP_RELATIVE_FLOOR * var_magnitude
    es = var - gap

    excess = var - realised
    if smoothing > 0.0:
        scaled = excess / smoothing
        hinge = smoothing * np.logaddexp(0.0, scaled)
        hit = special.expit(scaled)
    else:
        hinge = np.maximum(excess, 0.0)
        hit = (realised <= var).astype(float)

    losses = -(1.0 / (alpha * es)) * hinge + var / es + np.log(-es) - 1.0
    value = float(np.mean(losses))
    if not math.isfinite(value):
        return 1e12, np.zeros_like(parameters)

    d_var = -(1.0 / (alpha * es)) * hit + 1.0 / es
    d_es = hinge / (alpha * es * es) - var / (es * es) + 1.0 / es
    # d(es)/d(a) picks up the relative gap floor as well as v itself.
    d_index_var = -var_slope * (d_var + (1.0 + ES_GAP_RELATIVE_FLOOR) * d_es)
    d_index_gap = -gap_slope * d_es

    rows = float(realised.size)
    gradient = np.concatenate((design.T @ d_index_var, design.T @ d_index_gap)) / rows
    if not np.isfinite(gradient).all():
        return 1e12, np.zeros_like(parameters)
    return value, gradient


def _fz0_objective(
    parameters: np.ndarray,
    design: np.ndarray,
    realised: np.ndarray,
    alpha: float,
    smoothing: float,
) -> float:
    return _fz0_value_and_gradient(parameters, design, realised, alpha, smoothing)[0]


def _initial_parameters(design: np.ndarray, realised: np.ndarray, alpha: float) -> np.ndarray:
    width = design.shape[1]
    quantile = float(np.quantile(realised, alpha))
    quantile = min(quantile, -1e-4)
    tail = realised[realised <= quantile]
    shortfall = float(tail.mean()) if tail.size else quantile * 1.25
    gap = max(quantile - shortfall, 1e-4)
    start = np.zeros(2 * width, dtype=float)
    start[0] = math.log(-quantile)
    start[width] = math.log(gap)
    return start


def fit_joint_var_es(
    design: np.ndarray,
    realised: np.ndarray,
    *,
    alpha: float,
    name: str,
    columns: Sequence[str],
    smoothing_schedule: Sequence[float] = (0.5, 0.1, 0.0),
    start_offsets: Sequence[float] = (0.0, 0.15, -0.15, 0.35),
    max_iterations: int = 4000,
    objective_tolerance: float = 1e-4,
) -> JointTailFit:
    """Estimate a joint VaR/ES model by minimising mean FZ0 loss.

    The optimisation sequence is frozen and never inspects evaluation data: a
    smoothed-indicator warm start, a weaker smoothing pass, then a final
    unsmoothed bounded pass, each solved with L-BFGS-B on the analytic
    gradient, repeated from several deterministic starting points.  A fit is
    retained only when the best solution is finite, satisfies ``es < var < 0``,
    and at least one other start lands within ``objective_tolerance`` of it.
    """

    design = np.asarray(design, dtype=float)
    realised = np.asarray(realised, dtype=float)
    columns = tuple(columns)
    if design.ndim != 2 or design.shape[0] != realised.size:
        raise ValueError("design and realised are not conformable")
    if design.shape[1] != len(columns):
        raise ValueError("columns must name every design column")
    if not np.isfinite(design).all() or not np.isfinite(realised).all():
        raise ValueError("design and realised must be finite")

    base = _initial_parameters(design, realised, alpha)
    # Wide box bounds keep L-BFGS-B from wandering into the extension region;
    # they are far outside any plausible optimum in standardised return units.
    bounds = [(-60.0, 60.0)] * base.size
    solutions: list[tuple[float, np.ndarray, int, str]] = []
    for offset in start_offsets:
        parameters = base.copy()
        parameters[0] += offset
        parameters[design.shape[1]] += offset
        iterations = 0
        message = ""
        ok = True
        for smoothing in smoothing_schedule:
            result = optimize.minimize(
                _fz0_value_and_gradient,
                parameters,
                args=(design, realised, alpha, smoothing),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": max_iterations, "maxfun": 4 * max_iterations, "ftol": 1e-14, "gtol": 1e-9},
            )
            iterations += int(result.nit)
            message = str(result.message)
            if not np.isfinite(result.fun):
                ok = False
                break
            parameters = np.asarray(result.x, dtype=float)
        if not ok:
            continue
        objective = _fz0_objective(parameters, design, realised, alpha, 0.0)
        if math.isfinite(objective) and objective < 1e11:
            solutions.append((objective, parameters, iterations, message))

    if not solutions:
        raise RuntimeError(f"joint VaR/ES fit for {name} did not converge from any start")

    solutions.sort(key=lambda item: item[0])
    best_objective, best_parameters, best_iterations, best_message = solutions[0]
    spread = float(solutions[-1][0] - best_objective) if len(solutions) > 1 else math.inf
    agreeing = sum(1 for objective, *_ in solutions if objective - best_objective <= objective_tolerance)

    width = design.shape[1]
    beta_var = best_parameters[:width]
    beta_gap = best_parameters[width:]
    var, es, clipped = tail_forecasts(design, beta_var, beta_gap)
    notes: list[str] = []
    if clipped:
        notes.append(f"{clipped} in-sample linear indices needed the numerical extension")
    if agreeing < 2:
        notes.append("only one start reached the retained optimum")
    if not np.all(es < var) or not np.all(var < 0.0):
        raise RuntimeError(f"joint VaR/ES fit for {name} violated es < var < 0")

    return JointTailFit(
        name=name,
        columns=columns,
        beta_var=beta_var,
        beta_gap=beta_gap,
        objective=float(best_objective),
        converged=agreeing >= 2,
        message=best_message,
        iterations=int(best_iterations),
        starts_attempted=len(start_offsets),
        starts_converged=len(solutions),
        start_objective_spread=spread,
        clipped_rows=clipped,
        warnings=tuple(notes),
    )


# --------------------------------------------------------------------------
# News-conditioned volatility scale
# --------------------------------------------------------------------------


def qlike_loss(squared_returns: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """QLIKE volatility loss ``r^2 / sigma^2 + log(sigma^2)``."""

    squared_returns = np.asarray(squared_returns, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if np.any(variances <= 0.0):
        raise ValueError("variances must be strictly positive")
    return squared_returns / variances + np.log(variances)


def fit_news_scale_adjustment(
    news_design: np.ndarray,
    shocks: np.ndarray,
    base_sigma: np.ndarray,
    *,
    max_iterations: int = 4000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit ``sigma_tilde = base_sigma * exp(0.5 * X_news @ theta)`` under QLIKE.

    ``shocks`` are demeaned returns in the same units as ``base_sigma``.  The
    adjustment is multiplicative and estimated on development rows only.
    """

    news_design = np.asarray(news_design, dtype=float)
    shocks = np.asarray(shocks, dtype=float)
    base_sigma = np.asarray(base_sigma, dtype=float)
    if np.any(base_sigma <= 0.0):
        raise ValueError("base_sigma must be strictly positive")
    squared = shocks * shocks
    base_variance = base_sigma * base_sigma

    def objective(theta: np.ndarray) -> float:
        adjustment = np.exp(np.clip(news_design @ theta, -4.0, 4.0))
        value = float(np.mean(qlike_loss(squared, base_variance * adjustment)))
        return value if math.isfinite(value) else 1e12

    start = np.zeros(news_design.shape[1], dtype=float)
    result = optimize.minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": max_iterations, "xatol": 1e-7, "fatol": 1e-11, "adaptive": True},
    )
    theta = np.asarray(result.x, dtype=float)
    return theta, {
        "converged": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "qlike_start": objective(start),
        "qlike_final": float(result.fun),
    }


def apply_news_scale_adjustment(
    base_sigma: np.ndarray,
    news_design: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Apply a frozen multiplicative news-conditioned scale adjustment."""

    base_sigma = np.asarray(base_sigma, dtype=float)
    adjustment = np.exp(0.5 * np.clip(np.asarray(news_design, dtype=float) @ np.asarray(theta, dtype=float), -4.0, 4.0))
    return base_sigma * adjustment


# --------------------------------------------------------------------------
# Dependence-aware uncertainty
# --------------------------------------------------------------------------


def block_bootstrap_indices(
    n_dates: int,
    *,
    block_length: int,
    replications: int,
    seed: int,
) -> np.ndarray:
    """Moving-block bootstrap index matrix over an ordered date axis.

    Blocks start at any position in ``0..n_dates - block_length`` and wrap is
    not used; each replication concatenates ``ceil(n_dates / block_length)``
    blocks and is trimmed to ``n_dates`` positions.  Returns an integer array of
    shape ``(replications, n_dates)`` whose entries index the date axis.
    """

    if n_dates <= 0:
        raise ValueError("n_dates must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    span = min(block_length, n_dates)
    blocks = math.ceil(n_dates / span)
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, n_dates - span + 1, size=(replications, blocks))
    offsets = np.arange(span)
    draws = (starts[:, :, None] + offsets[None, None, :]).reshape(replications, blocks * span)
    return draws[:, :n_dates]


def date_block_bootstrap_mean(
    date_values: np.ndarray,
    date_weights: np.ndarray | None = None,
    *,
    block_length: int,
    replications: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap a weighted mean of per-date statistics with block resampling.

    ``date_values`` are already aggregated across the full cross-section of
    firms on each target date, so resampling a date keeps every firm on that
    date together.  ``date_weights`` default to the firm counts supplied by the
    caller (observation-equal aggregation) or to ones (date-equal aggregation).
    """

    values = np.asarray(date_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("date_values must be one-dimensional")
    weights = np.ones_like(values) if date_weights is None else np.asarray(date_weights, dtype=float)
    if weights.shape != values.shape:
        raise ValueError("date_weights must match date_values")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("date_values and date_weights must be finite")

    point = float(np.sum(values * weights) / np.sum(weights))
    draws = block_bootstrap_indices(
        values.size,
        block_length=block_length,
        replications=replications,
        seed=seed,
    )
    sampled_values = values[draws]
    sampled_weights = weights[draws]
    means = np.sum(sampled_values * sampled_weights, axis=1) / np.sum(sampled_weights, axis=1)
    lower_tail = (1.0 - confidence) / 2.0
    return {
        "point_estimate": point,
        "bootstrap_mean": float(means.mean()),
        "bootstrap_std": float(means.std(ddof=1)) if means.size > 1 else math.nan,
        "ci_low": float(np.quantile(means, lower_tail)),
        "ci_high": float(np.quantile(means, 1.0 - lower_tail)),
        "share_below_zero": float(np.mean(means < 0.0)),
        "replications": int(replications),
        "block_length": int(block_length),
        "n_dates": int(values.size),
        "seed": int(seed),
    }


# --------------------------------------------------------------------------
# VaR backtests
# --------------------------------------------------------------------------


def kupiec_test(hits: np.ndarray, alpha: float) -> dict[str, float]:
    """Kupiec unconditional-coverage likelihood-ratio test."""

    hits = np.asarray(hits, dtype=float)
    total = hits.size
    failures = float(hits.sum())
    if total == 0:
        return {"statistic": math.nan, "p_value": math.nan, "observations": 0, "failures": 0.0}
    rate = failures / total
    if failures == 0 or failures == total:
        statistic = math.nan
    else:
        log_null = failures * math.log(alpha) + (total - failures) * math.log1p(-alpha)
        log_alt = failures * math.log(rate) + (total - failures) * math.log1p(-rate)
        statistic = -2.0 * (log_null - log_alt)
    p_value = float(stats.chi2.sf(statistic, 1)) if math.isfinite(statistic) else math.nan
    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "observations": int(total),
        "failures": failures,
        "hit_rate": rate,
    }


def christoffersen_tests(hits: np.ndarray, alpha: float) -> dict[str, float | dict[str, int]]:
    """Christoffersen independence and conditional-coverage tests."""

    hits = np.asarray(hits, dtype=int)
    if hits.size < 2:
        return {
            "independence_statistic": math.nan,
            "independence_p_value": math.nan,
            "conditional_coverage_statistic": math.nan,
            "conditional_coverage_p_value": math.nan,
        }
    previous, current = hits[:-1], hits[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    def _safe_log(value: float, weight: int) -> float:
        return weight * math.log(value) if weight and value > 0.0 else 0.0

    total = n00 + n01 + n10 + n11
    pooled = (n01 + n11) / total if total else math.nan
    pi0 = n01 / (n00 + n01) if (n00 + n01) else math.nan
    pi1 = n11 / (n10 + n11) if (n10 + n11) else math.nan
    if not all(math.isfinite(value) for value in (pooled, pi0, pi1)):
        independence = math.nan
    else:
        log_null = _safe_log(1.0 - pooled, n00 + n10) + _safe_log(pooled, n01 + n11)
        log_alt = _safe_log(1.0 - pi0, n00) + _safe_log(pi0, n01) + _safe_log(1.0 - pi1, n10) + _safe_log(pi1, n11)
        independence = -2.0 * (log_null - log_alt)
    unconditional = kupiec_test(hits, alpha)["statistic"]
    conditional = independence + unconditional if math.isfinite(independence) and math.isfinite(unconditional) else math.nan
    return {
        "independence_statistic": float(independence),
        "independence_p_value": float(stats.chi2.sf(independence, 1)) if math.isfinite(independence) else math.nan,
        "conditional_coverage_statistic": float(conditional),
        "conditional_coverage_p_value": (float(stats.chi2.sf(conditional, 2)) if math.isfinite(conditional) else math.nan),
        "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    }
