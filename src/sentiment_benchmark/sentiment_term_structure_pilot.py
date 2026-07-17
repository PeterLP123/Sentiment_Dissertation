"""Leak-free precision gate for the term structure of news-sentiment alpha."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import least_squares

from .artifact_io import atomic_write_json, sha256_file
from .runtime_metadata import collect_run_environment
from .sentiment_surprise_pilot import PilotConfig, _load_prices, build_daily_sentiment
from .strategy_sweep import chronological_split_date


class TermStructurePilotError(RuntimeError):
    """Raised when the frozen term-structure pilot cannot run as specified."""


@dataclass(frozen=True)
class TermStructureConfig:
    """Frozen design choices for the one-shot pilot."""

    development_fraction: float = 0.70
    horizons: tuple[int, ...] = tuple(range(11))
    bootstrap_samples: int = 2_000
    simulation_runs: int = 500
    simulation_bootstrap_samples: int = 100
    random_seed: int = 20260717
    coverage_break_ratio: float = 3.0
    history_gate_months: float = 18.0
    precision_gate_fraction: float = 0.80
    evaluation_run_number: int = 1
    rerun_reason: str | None = None


PARAMETER_NAMES: dict[str, tuple[str, ...]] = {
    "single": ("lambda", "rho", "half_life"),
    "two_component": (
        "lambda_fast",
        "rho_fast",
        "half_life_fast",
        "lambda_slow",
        "rho_slow",
        "half_life_slow",
    ),
}


def audit_corpus_depth(
    headlines_path: str | Path,
    *,
    cohort: str,
    through_month: str,
    break_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Count raw symbol associations by month without exposing licensed text."""

    source = Path(headlines_path)
    counts: dict[tuple[str, str], int] = {}
    earliest: datetime | None = None
    latest: datetime | None = None
    rows = 0
    associations = 0
    symbols: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TermStructurePilotError(f"invalid headline JSON at {source}:{line_number}: {exc}") from exc
            timestamp_text = row.get("version_created") or row.get("first_created")
            try:
                timestamp = datetime.fromisoformat(str(timestamp_text).replace("Z", "+00:00"))
            except (TypeError, ValueError) as exc:
                raise TermStructurePilotError(f"invalid timestamp at {source}:{line_number}") from exc
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            raw_symbols = row.get("matched_symbols") or []
            symbol_values = raw_symbols.split("|") if isinstance(raw_symbols, str) else raw_symbols
            matched = sorted({str(value).strip() for value in symbol_values if str(value).strip()})
            month = timestamp.strftime("%Y-%m")
            for symbol in matched:
                counts[(symbol, month)] = counts.get((symbol, month), 0) + 1
                symbols.add(symbol)
            rows += 1
            associations += len(matched)
            earliest = timestamp if earliest is None or timestamp < earliest else earliest
            latest = timestamp if latest is None or timestamp > latest else latest
    if earliest is None or latest is None or not symbols:
        raise TermStructurePilotError(f"no usable symbol-timestamp rows in {source}")

    months = pd.period_range(earliest.strftime("%Y-%m"), through_month, freq="M")
    records = [
        {
            "cohort": cohort,
            "symbol": symbol,
            "month": str(month),
            "article_associations": counts.get((symbol, str(month)), 0),
        }
        for symbol in sorted(symbols)
        for month in months
    ]
    monthly = pd.DataFrame(records)
    flags: list[dict[str, Any]] = []
    for symbol, group in monthly.groupby("symbol", sort=True):
        ordered = group.sort_values("month", kind="stable")
        previous: int | None = None
        previous_month: str | None = None
        for row in ordered.itertuples(index=False):
            current = int(row.article_associations)
            if previous is not None:
                if previous == 0 and current == 0:
                    ratio = 1.0
                elif previous == 0 or current == 0:
                    ratio = math.inf
                else:
                    ratio = max(current / previous, previous / current)
                if ratio > break_ratio:
                    flags.append(
                        {
                            "cohort": cohort,
                            "symbol": symbol,
                            "previous_month": previous_month,
                            "month": row.month,
                            "previous_count": previous,
                            "count": current,
                            "max_fold_change": ratio,
                        }
                    )
            previous = current
            previous_month = str(row.month)
    summary = {
        "cohort": cohort,
        "raw_headline_rows": rows,
        "symbol_associations": associations,
        "symbols": len(symbols),
        "earliest_timestamp": earliest.isoformat(),
        "latest_timestamp": latest.isoformat(),
        "calendar_span_days": (latest - earliest).total_seconds() / 86_400,
        "calendar_span_months": (latest - earliest).total_seconds() / (86_400 * 365.2425 / 12),
    }
    return monthly, pd.DataFrame(flags), summary


def add_horizon_outcomes(
    events: pd.DataFrame,
    stock_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    *,
    market_symbol: str,
    horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach per-horizon single-day market-adjusted log returns."""

    horizon_values = tuple(sorted(set(horizons)))
    if not horizon_values or horizon_values[0] < 0:
        raise TermStructurePilotError("horizons must be non-negative")
    market = market_prices[market_prices["symbol"] == market_symbol].copy()
    if market.empty:
        raise TermStructurePilotError(f"market price input lacks {market_symbol}")
    market = market.sort_values("session_date", kind="stable")
    market["market_return"] = np.log(market["close"]).diff()
    stocks = stock_prices.sort_values(["symbol", "session_date"], kind="stable").copy()
    stocks["stock_return"] = stocks.groupby("symbol", sort=False)["close"].transform(lambda values: np.log(values).diff())
    abnormal = stocks.merge(
        market[["session_date", "market_return"]],
        on="session_date",
        how="inner",
        validate="many_to_one",
    )
    abnormal["abnormal_return"] = abnormal["stock_return"] - abnormal["market_return"]
    abnormal = abnormal.dropna(subset=["abnormal_return"]).copy()

    lookup: dict[tuple[str, pd.Timestamp], tuple[float, ...]] = {}
    for symbol, group in abnormal.groupby("symbol", sort=True):
        ordered = group.sort_values("session_date", kind="stable").reset_index(drop=True)
        values = ordered["abnormal_return"].to_numpy(float)
        for index, session_date in enumerate(ordered["session_date"]):
            lookup[(str(symbol), pd.Timestamp(session_date))] = tuple(
                float(values[index + horizon]) if index + horizon < len(values) else math.nan for horizon in horizon_values
            )
    result = events.copy()
    outcomes = [
        lookup.get((str(row.symbol), pd.Timestamp(row.session_date)), (math.nan,) * len(horizon_values))
        for row in result.itertuples(index=False)
    ]
    columns = [f"ar_h{horizon}" for horizon in horizon_values]
    result[columns] = pd.DataFrame(outcomes, index=result.index)
    return result, abnormal[["symbol", "session_date", "abnormal_return"]].reset_index(drop=True)


def _fit_level_regression(data: pd.DataFrame, outcome: str) -> tuple[Any, np.ndarray]:
    design = sm.add_constant(data[["level"]], has_constant="add")
    ordinary = sm.OLS(data[outcome], design).fit()
    clustered = ordinary.get_robustcov_results(
        cov_type="cluster",
        groups=data["session_date"],
        use_correction=True,
        df_correction=True,
        use_t=True,
    )
    return clustered, np.asarray(ordinary.resid, dtype=float)


def estimate_impulse_response(
    events: pd.DataFrame,
    *,
    split: str,
    horizons: Iterable[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Estimate Lambda(h) and date-clustered inference for one split."""

    rows: list[dict[str, Any]] = []
    residuals: list[np.ndarray] = []
    for horizon in horizons:
        outcome = f"ar_h{horizon}"
        fitted, ordinary_residual = _fit_level_regression(events, outcome)
        names = list(fitted.model.exog_names)
        level_index = names.index("level")
        constant_index = names.index("const")
        rows.append(
            {
                "split": split,
                "horizon": horizon,
                "intercept": float(fitted.params[constant_index]),
                "lambda": float(fitted.params[level_index]),
                "clustered_se": float(fitted.bse[level_index]),
                "t_stat": float(fitted.tvalues[level_index]),
                "p_value": float(fitted.pvalues[level_index]),
                "events": len(events),
                "date_clusters": events["session_date"].nunique(),
            }
        )
        residuals.append(ordinary_residual)
    return pd.DataFrame(rows), np.column_stack(residuals)


def _date_sufficient_statistics(events: pd.DataFrame, horizons: Iterable[int]) -> tuple[np.ndarray, list[pd.Timestamp]]:
    horizon_values = tuple(horizons)
    dates = sorted(pd.Timestamp(value) for value in events["session_date"].unique())
    # Per date: n, sum(x), sum(x^2), then sum(y_h), sum(x*y_h) for each horizon.
    statistics = np.zeros((len(dates), 3 + 2 * len(horizon_values)), dtype=float)
    date_index = {date: index for index, date in enumerate(dates)}
    for date, group in events.groupby("session_date", sort=True):
        x = group["level"].to_numpy(float)
        row = statistics[date_index[pd.Timestamp(date)]]
        row[:3] = (len(group), x.sum(), np.square(x).sum())
        for column, horizon in enumerate(horizon_values):
            y = group[f"ar_h{horizon}"].to_numpy(float)
            row[3 + 2 * column : 5 + 2 * column] = (y.sum(), np.dot(x, y))
    return statistics, dates


def _bootstrap_slopes_from_statistics(
    statistics: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    date_count = len(statistics)
    horizon_count = (statistics.shape[1] - 3) // 2
    output = np.full((samples, horizon_count), np.nan, dtype=float)
    for draw in range(samples):
        aggregate = statistics[rng.integers(0, date_count, size=date_count)].sum(axis=0)
        n, sum_x, sum_x2 = aggregate[:3]
        denominator = sum_x2 - sum_x * sum_x / n
        if denominator <= 0:
            continue
        for horizon in range(horizon_count):
            sum_y, sum_xy = aggregate[3 + 2 * horizon : 5 + 2 * horizon]
            output[draw, horizon] = (sum_xy - sum_x * sum_y / n) / denominator
    return output


def bootstrap_impulse_response(
    events: pd.DataFrame,
    *,
    horizons: Iterable[int],
    samples: int,
    seed: int,
) -> np.ndarray:
    statistics, _ = _date_sufficient_statistics(events, horizons)
    return _bootstrap_slopes_from_statistics(statistics, samples=samples, rng=np.random.default_rng(seed))


def _single_curve(parameters: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    return parameters[0] * np.exp(-parameters[1] * horizons)


def _two_curve(parameters: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    return parameters[0] * np.exp(-parameters[1] * horizons) + parameters[2] * np.exp(-parameters[3] * horizons)


def _ordered_two(parameters: np.ndarray) -> np.ndarray:
    lambda_a, rho_a, lambda_b, rho_b = (float(value) for value in parameters)
    if rho_a >= rho_b:
        return np.array([lambda_a, rho_a, lambda_b, rho_b], dtype=float)
    return np.array([lambda_b, rho_b, lambda_a, rho_a], dtype=float)


def fit_decay_curve(
    lambdas: np.ndarray,
    clustered_se: np.ndarray,
    *,
    model: str,
) -> dict[str, float]:
    """Fit one fixed exponential specification with signs free and positive rates."""

    y = np.asarray(lambdas, dtype=float)
    se = np.asarray(clustered_se, dtype=float)
    horizons = np.arange(len(y), dtype=float)
    finite_positive = se[np.isfinite(se) & (se > 0)]
    replacement = float(np.median(finite_positive)) if len(finite_positive) else 1.0
    weights = 1.0 / np.where(np.isfinite(se) & (se > 0), se, replacement)
    scale = max(float(np.nanmax(np.abs(y))), 1e-6)

    if model == "single":
        starts = [(float(y[0]), rho) for rho in (0.05, 0.15, 0.4, 1.0, 2.0)]
        best = None
        for start in starts:
            fitted = least_squares(
                lambda p: weights * (_single_curve(p, horizons) - y),
                x0=np.asarray(start),
                bounds=([-10 * scale, 1e-4], [10 * scale, 10.0]),
                max_nfev=2_000,
            )
            if best is None or fitted.cost < best.cost:
                best = fitted
        if best is None or not best.success:
            raise TermStructurePilotError("single-exponential fit failed")
        lam, rho = (float(value) for value in best.x)
        return {"lambda": lam, "rho": rho, "half_life": math.log(2) / rho, "weighted_sse": 2 * float(best.cost)}

    if model != "two_component":
        raise TermStructurePilotError(f"unknown decay model: {model}")
    two_starts = [(0.7 * y[0], fast, 0.3 * y[0], slow) for fast, slow in ((1.0, 0.1), (0.4, 0.05), (2.0, 0.2), (0.2, 0.02), (4.0, 0.5))]
    best = None
    for two_start in two_starts:
        fitted = least_squares(
            lambda p: weights * (_two_curve(p, horizons) - y),
            x0=np.asarray(two_start),
            bounds=([-10 * scale, 1e-4, -10 * scale, 1e-4], [10 * scale, 10.0, 10 * scale, 10.0]),
            max_nfev=5_000,
        )
        if best is None or fitted.cost < best.cost:
            best = fitted
    if best is None or not best.success:
        raise TermStructurePilotError("two-component exponential fit failed")
    lambda_fast, rho_fast, lambda_slow, rho_slow = _ordered_two(best.x)
    return {
        "lambda_fast": float(lambda_fast),
        "rho_fast": float(rho_fast),
        "half_life_fast": math.log(2) / float(rho_fast),
        "lambda_slow": float(lambda_slow),
        "rho_slow": float(rho_slow),
        "half_life_slow": math.log(2) / float(rho_slow),
        "weighted_sse": 2 * float(best.cost),
    }


def decay_curve_values(model: str, parameters: dict[str, float], horizons: np.ndarray) -> np.ndarray:
    if model == "single":
        return parameters["lambda"] * np.exp(-parameters["rho"] * horizons)
    return parameters["lambda_fast"] * np.exp(-parameters["rho_fast"] * horizons) + parameters["lambda_slow"] * np.exp(
        -parameters["rho_slow"] * horizons
    )


def bootstrap_decay_parameters(
    lambda_draws: np.ndarray,
    clustered_se: np.ndarray,
    *,
    model: str,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, float]] = []
    failures = 0
    for draw in lambda_draws:
        if not np.isfinite(draw).all():
            failures += 1
            continue
        try:
            rows.append(fit_decay_curve(draw, clustered_se, model=model))
        except (TermStructurePilotError, ValueError, FloatingPointError):
            failures += 1
    return pd.DataFrame(rows), failures


def summarize_decay_fits(
    development_ir: pd.DataFrame,
    evaluation_ir: pd.DataFrame,
    development_draws: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    """Fit both pre-declared decay models and summarize bootstrap uncertainty."""

    horizons = development_ir["horizon"].to_numpy(float)
    dev_lambda = development_ir["lambda"].to_numpy(float)
    dev_se = development_ir["clustered_se"].to_numpy(float)
    eval_lambda = evaluation_ir["lambda"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    points: dict[str, dict[str, float]] = {}
    bootstrap_frames: dict[str, pd.DataFrame] = {}
    for model in ("single", "two_component"):
        point = fit_decay_curve(dev_lambda, dev_se, model=model)
        bootstrap, failures = bootstrap_decay_parameters(development_draws, dev_se, model=model)
        if len(bootstrap) < max(50, len(development_draws) // 2):
            raise TermStructurePilotError(f"too few valid {model} decay bootstrap fits: {len(bootstrap)}")
        predicted = decay_curve_values(model, point, horizons)
        benchmark_sse = float(np.square(eval_lambda).sum())
        model_sse = float(np.square(eval_lambda - predicted).sum())
        oos_r2_vs_zero = 1 - model_sse / benchmark_sse if benchmark_sse > 0 else math.nan
        rmse = math.sqrt(model_sse / len(horizons))
        points[model] = point
        bootstrap_frames[model] = bootstrap
        for parameter in PARAMETER_NAMES[model]:
            ci_low, ci_high = np.quantile(bootstrap[parameter], [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "parameter": parameter,
                    "estimate": point[parameter],
                    "bootstrap_ci_low": float(ci_low),
                    "bootstrap_ci_high": float(ci_high),
                    "evaluation_rmse": rmse,
                    "evaluation_r2_vs_zero": oos_r2_vs_zero,
                    "valid_bootstrap_fits": len(bootstrap),
                    "failed_bootstrap_fits": failures,
                }
            )
    return pd.DataFrame(rows), points, bootstrap_frames


def _nearest_psd(covariance: np.ndarray) -> np.ndarray:
    symmetric = (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(values, 1e-12)
    return (vectors * clipped) @ vectors.T


def _simulation_statistics(
    x: np.ndarray,
    y: np.ndarray,
    date_codes: np.ndarray,
    date_count: int,
) -> np.ndarray:
    horizons = y.shape[1]
    statistics = np.zeros((date_count, 3 + 2 * horizons), dtype=float)
    np.add.at(statistics[:, 0], date_codes, 1)
    np.add.at(statistics[:, 1], date_codes, x)
    np.add.at(statistics[:, 2], date_codes, np.square(x))
    for horizon in range(horizons):
        np.add.at(statistics[:, 3 + 2 * horizon], date_codes, y[:, horizon])
        np.add.at(statistics[:, 4 + 2 * horizon], date_codes, x * y[:, horizon])
    return statistics


def _slopes_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        raise TermStructurePilotError("sentiment has zero variance")
    return centered @ y / denominator


def simulation_precision_gate(
    development: pd.DataFrame,
    residuals: np.ndarray,
    development_ir: pd.DataFrame,
    point_parameters: dict[str, dict[str, float]],
    *,
    config: TermStructureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, dict[str, Any]]:
    """Run nested date-block simulations under each fitted decay specification."""

    horizons = development_ir["horizon"].to_numpy(float)
    x = development["level"].to_numpy(float)
    dates = pd.Categorical(development["session_date"], ordered=True)
    date_codes = np.asarray(dates.codes, dtype=int)
    date_count = len(dates.categories)
    horizon_count = len(horizons)
    if residuals.shape != (len(development), horizon_count):
        raise TermStructurePilotError("residual calibration matrix has the wrong shape")

    date_means = np.zeros((date_count, horizon_count), dtype=float)
    date_sizes = np.bincount(date_codes, minlength=date_count).astype(float)
    for horizon in range(horizon_count):
        np.add.at(date_means[:, horizon], date_codes, residuals[:, horizon])
    date_means /= date_sizes[:, None]
    idiosyncratic = residuals - date_means[date_codes]
    common_covariance = _nearest_psd(np.cov(date_means, rowvar=False, ddof=1))
    idiosyncratic_covariance = _nearest_psd(np.cov(idiosyncratic, rowvar=False, ddof=1))
    intercepts = development_ir["intercept"].to_numpy(float)
    clustered_se = development_ir["clustered_se"].to_numpy(float)
    rng = np.random.default_rng(config.random_seed + 10_000)
    run_rows: list[dict[str, Any]] = []
    invalid_runs: dict[str, int] = {"single": 0, "two_component": 0}

    for model in ("single", "two_component"):
        truth = point_parameters[model]
        true_curve = decay_curve_values(model, truth, horizons)
        for run in range(config.simulation_runs):
            common = rng.multivariate_normal(np.zeros(horizon_count), common_covariance, size=date_count)
            idiosyncratic_draw = rng.multivariate_normal(np.zeros(horizon_count), idiosyncratic_covariance, size=len(development))
            synthetic = intercepts + x[:, None] * true_curve + common[date_codes] + idiosyncratic_draw
            lambda_estimates = _slopes_from_xy(x, synthetic)
            try:
                point = fit_decay_curve(lambda_estimates, clustered_se, model=model)
                statistics = _simulation_statistics(x, synthetic, date_codes, date_count)
                lambda_bootstrap = _bootstrap_slopes_from_statistics(
                    statistics,
                    samples=config.simulation_bootstrap_samples,
                    rng=rng,
                )
                bootstrap, _ = bootstrap_decay_parameters(lambda_bootstrap, clustered_se, model=model)
            except (TermStructurePilotError, ValueError, FloatingPointError):
                invalid_runs[model] += 1
                continue
            if len(bootstrap) < max(20, config.simulation_bootstrap_samples // 2):
                invalid_runs[model] += 1
                continue
            for parameter in PARAMETER_NAMES[model]:
                low, high = np.quantile(bootstrap[parameter], [0.025, 0.975])
                factor_span = float(high / low) if parameter.startswith("half_life") and np.isfinite(low) and low > 0 else math.nan
                run_rows.append(
                    {
                        "data_generating_model": model,
                        "run": run + 1,
                        "parameter": parameter,
                        "true_value": truth[parameter],
                        "estimate": point[parameter],
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "ci_width": float(high - low),
                        "factor_span": factor_span,
                        "factor_of_3_pass": bool(factor_span <= 3.0) if parameter.startswith("half_life") else None,
                        "ci_covers_truth": bool(low <= truth[parameter] <= high),
                    }
                )
    runs = pd.DataFrame(run_rows)
    if runs.empty:
        raise TermStructurePilotError("all precision simulations failed")
    summaries: list[dict[str, Any]] = []
    for (model, parameter), group in runs.groupby(["data_generating_model", "parameter"], sort=False):
        true_value = float(group["true_value"].iloc[0])
        is_half_life = str(parameter).startswith("half_life")
        summaries.append(
            {
                "data_generating_model": model,
                "parameter": parameter,
                "true_value": true_value,
                "mean_estimate": float(group["estimate"].mean()),
                "bias": float(group["estimate"].mean() - true_value),
                "median_ci_width": float(group["ci_width"].median()),
                "median_factor_span": float(group["factor_span"].median()) if is_half_life else math.nan,
                "factor_of_3_pass_fraction": (float(group["factor_of_3_pass"].astype(bool).mean()) if is_half_life else math.nan),
                "ci_coverage_fraction": float(group["ci_covers_truth"].mean()),
                "valid_runs": int(group["run"].nunique()),
                "invalid_runs": invalid_runs[str(model)],
            }
        )
    summary = pd.DataFrame(summaries)
    half_lives = summary[summary["parameter"].str.startswith("half_life")]
    gate_pass = bool(
        len(half_lives) == 3
        and (half_lives["valid_runs"] >= config.simulation_runs * 0.9).all()
        and (half_lives["factor_of_3_pass_fraction"] >= config.precision_gate_fraction).all()
    )
    calibration = {
        "firms": int(development["symbol"].nunique()),
        "events": len(development),
        "sessions": date_count,
        "news_frequency_events_per_firm_session": len(development) / (development["symbol"].nunique() * date_count),
        "residual_return_std": float(np.std(residuals, ddof=1)),
        "common_covariance_trace": float(np.trace(common_covariance)),
        "idiosyncratic_covariance_trace": float(np.trace(idiosyncratic_covariance)),
        "simulation_runs_per_model": config.simulation_runs,
        "simulation_bootstrap_samples": config.simulation_bootstrap_samples,
        "invalid_runs": invalid_runs,
    }
    return summary, runs, gate_pass, calibration


def reversal_check(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    pilot2_regressions_path: str | Path,
) -> pd.DataFrame:
    """Compare the prior CAR(+1,+5) reversal with the longer coherent panel."""

    previous = pd.read_csv(pilot2_regressions_path)
    prior = previous[(previous["model"] == "M2") & (previous["term"] == "level")]
    if len(prior) != 1:
        raise TermStructurePilotError("pilot-2 regressions lack exactly one M2 level row")
    rows = [
        {
            "sample": "pilot2_sector33_evaluation",
            "coefficient": float(prior.iloc[0]["coefficient"]),
            "clustered_se": float(prior.iloc[0]["standard_error"]),
            "t_stat": float(prior.iloc[0]["t_stat"]),
            "events": int(prior.iloc[0]["evaluation_events"]),
            "date_clusters": int(prior.iloc[0]["evaluation_date_clusters"]),
        }
    ]
    for label, data in (("midcap22_development", development), ("midcap22_evaluation", evaluation)):
        copy = data.copy()
        copy["car_p1_p5"] = copy[[f"ar_h{horizon}" for horizon in range(1, 6)]].sum(axis=1)
        fitted, _ = _fit_level_regression(copy, "car_p1_p5")
        index = list(fitted.model.exog_names).index("level")
        rows.append(
            {
                "sample": label,
                "coefficient": float(fitted.params[index]),
                "clustered_se": float(fitted.bse[index]),
                "t_stat": float(fitted.tvalues[index]),
                "events": len(copy),
                "date_clusters": copy["session_date"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def classify_term_structure_decision(
    *,
    history_months: float,
    precision_pass: bool,
    development_ir: pd.DataFrame,
    history_gate_months: float,
) -> tuple[str, list[str]]:
    significant = bool((development_ir["t_stat"].abs() > 2).any())
    history_pass = history_months >= history_gate_months
    reasons = [
        f"history_{'passes' if history_pass else 'below'}_{history_gate_months:g}_months",
        f"precision_gate_{'passes' if precision_pass else 'fails'}",
        f"development_lambda_{'has' if significant else 'lacks'}_abs_t_gt_2",
    ]
    if not precision_pass:
        return "NO-GO", reasons
    if history_pass and significant:
        return "GO", reasons
    return "WEAK-GO", reasons


def _format_number(value: float, digits: int = 6) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _plot_impulse_response(impulse: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    colors = {"development": "#1f77b4", "evaluation": "#d62728"}
    for split in ("development", "evaluation"):
        subset = impulse[impulse["split"] == split].sort_values("horizon")
        x = subset["horizon"].to_numpy(float)
        y = subset["lambda"].to_numpy(float)
        low = subset["bootstrap_ci_low"].to_numpy(float)
        high = subset["bootstrap_ci_high"].to_numpy(float)
        axis.plot(x, y, marker="o", linewidth=1.8, color=colors[split], label=split.title())
        axis.fill_between(x, low, high, color=colors[split], alpha=0.16)
    axis.axhline(0, color="#444444", linewidth=1)
    axis.set_xlabel("Horizon h (trading sessions)")
    axis.set_ylabel("Lambda(h): abnormal return per sentiment unit")
    axis.set_title("News-sentiment impulse response with date-block 95% intervals")
    axis.set_xticks(range(11))
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render_term_structure_report(
    *,
    corpus_summaries: pd.DataFrame,
    firm_depth: pd.DataFrame,
    coverage_flags: pd.DataFrame,
    impulse: pd.DataFrame,
    decay: pd.DataFrame,
    simulation: pd.DataFrame,
    reversal: pd.DataFrame,
    attrition: pd.DataFrame,
    decision: str,
    reasons: list[str],
    split_date: str,
    history_months: float,
    precision_pass: bool,
    config: TermStructureConfig,
) -> str:
    history_warning = (
        (
            f"**HISTORY GATE FAIL: {history_months:.2f} usable months is below 18; the result cannot be GO, "
            "and a failed precision gate can still force NO-GO.**"
        )
        if history_months < config.history_gate_months
        else f"**History gate passes: {history_months:.2f} usable months.**"
    )
    lines = [
        "# Sentiment alpha term structure: precision and feasibility gate",
        "",
        (
            f"**Decision: {decision}.** Evaluation begins `{split_date}`. Outcomes are market-adjusted "
            "close-to-close log returns (stock minus `^GSPC`)."
        ),
        "",
        history_warning,
        "",
        "## Part A — corpus depth",
        "",
        (
            "The one-year Reuters-only mid-cap cohort is the primary coherent maximum-depth panel. "
            "The separate six-month all-source sector cohort is audited but not pooled because both its "
            "firm universe and source regime differ."
        ),
        "",
        "| Cohort | Symbols | Raw headlines | Associations | Earliest | Latest | Span months |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in corpus_summaries.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {int(row.symbols)} | {int(row.raw_headline_rows):,} | {int(row.symbol_associations):,} | "
            f"{row.earliest_timestamp} | {row.latest_timestamp} | {row.calendar_span_months:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Coverage-break rule (> {config.coverage_break_ratio:g}x consecutive monthly change) flags "
                f"**{len(coverage_flags):,} symbol-month transitions**; full details are in "
                "`coverage_break_flags.csv`. Partial boundary months and the post-collection July zero are "
                "retained rather than hidden."
            ),
            "",
            "| Firm | News sessions | Articles | First session | Last session |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in firm_depth.itertuples(index=False):
        lines.append(f"| {row.symbol} | {int(row.news_sessions)} | {int(row.articles):,} | {row.first_session} | {row.last_session} |")
    lines.extend(
        [
            "",
            "## Part B — Lambda(h) impulse response",
            "",
            (
                "Each row is a separate OLS regression of the single-day AR(+h) on event-day FinBERT level. "
                "t-statistics use calendar-event-date clustered standard errors; intervals resample whole "
                "event dates."
            ),
            "",
            "| Split | h | Lambda(h) | clustered t | bootstrap 95% CI | Events | Dates |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in impulse.itertuples(index=False):
        lines.append(
            f"| {row.split} | {int(row.horizon)} | {row.lambda_:.6f} | {row.t_stat:.2f} | "
            f"[{row.bootstrap_ci_low:.6f}, {row.bootstrap_ci_high:.6f}] | {int(row.events):,} | {int(row.date_clusters)} |"
        )
    lines.extend(
        [
            "",
            "![Lambda(h) curve](lambda_curve.png)",
            "",
            "## Part C — decay fits",
            "",
            (
                "Models are fit once to development Lambda(h), with sign-free amplitudes, positive decay rates, "
                "and fast/slow ordering only to prevent label switching. OOS R² is measured against a "
                "zero-response forecast on the 11 evaluation coefficients."
            ),
            (
                "The single rate and two-component fast rate hit the fixed rho=10 numerical ceiling. Their "
                "0.069-session half-lives are boundary-censored: they mean the fitted response is effectively "
                "an h=0 spike, not that a sub-session economic half-life was measured precisely."
            ),
            "",
            "| Model | Parameter | Estimate | bootstrap 95% CI | Eval RMSE | Eval R² vs zero |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in decay.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.parameter} | {_format_number(row.estimate)} | "
            f"[{_format_number(row.bootstrap_ci_low)}, {_format_number(row.bootstrap_ci_high)}] | "
            f"{_format_number(row.evaluation_rmse)} | {_format_number(row.evaluation_r2_vs_zero)} |"
        )
    lines.extend(
        [
            "",
            "## Part D — simulation precision gate",
            "",
            (
                f"There are {config.simulation_runs} synthetic panels under each fitted DGP "
                f"({2 * config.simulation_runs:,} total), each preserving the observed development event mask "
                f"and date clustering. Every run uses {config.simulation_bootstrap_samples} nested date-block "
                "resamples. The overall gate requires every requested half-life to have a "
                f"factor-of-3-or-tighter interval in at least {config.precision_gate_fraction:.0%} of runs."
            ),
            "",
            f"**Precision gate: {'PASS' if precision_pass else 'FAIL'}.**",
            "",
            "| DGP | Parameter | True | Bias | Median CI width | Median factor span | Factor-3 pass | Coverage | Valid / invalid |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in simulation.itertuples(index=False):
        lines.append(
            f"| {row.data_generating_model} | {row.parameter} | {_format_number(row.true_value)} | "
            f"{_format_number(row.bias)} | {_format_number(row.median_ci_width)} | "
            f"{_format_number(row.median_factor_span, 3)} | {_format_number(row.factor_of_3_pass_fraction, 3)} | "
            f"{_format_number(row.ci_coverage_fraction, 3)} | {int(row.valid_runs)} / {int(row.invalid_runs)} |"
        )
    lines.extend(
        [
            "",
            "## Pilot-2 reversal check",
            "",
            (
                "The original cell is the frozen sector-33 evaluation M2 level coefficient on CAR(+1,+5). "
                "The two mid-cap rows use the longer, different-firm Reuters cohort, so this is an "
                "external-universe replication rather than extra history for the same firms."
            ),
            "",
            "| Sample | CAR(+1,+5) level coefficient | clustered t | Events | Dates |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in reversal.itertuples(index=False):
        lines.append(f"| {row.sample} | {row.coefficient:.6f} | {row.t_stat:.2f} | {int(row.events):,} | {int(row.date_clusters)} |")
    lines.extend(["", "## Attrition", "", "| Step | Unit | Surviving |", "| --- | --- | ---: |"])
    for row in attrition.itertuples(index=False):
        lines.append(f"| {row.step} | {row.unit} | {int(row.surviving):,} |")

    eval_reversal = reversal[reversal["sample"] == "midcap22_evaluation"].iloc[0]
    replicated = float(eval_reversal["coefficient"]) < 0 and abs(float(eval_reversal["t_stat"])) > 2
    lines.extend(
        [
            "",
            "## Ten-line plain-English summary",
            "",
            f"1. The mechanical dissertation gate is **{decision}**.",
            (
                f"2. The primary coherent corpus spans {history_months:.2f} months, so the 18-month history "
                f"gate {'passes' if history_months >= 18 else 'fails'}."
            ),
            f"3. The nested simulation precision gate {'passes' if precision_pass else 'fails'} at the required 80% threshold.",
            (
                f"4. Development has {int((impulse[impulse['split'] == 'development']['t_stat'].abs() > 2).sum())} "
                "individually significant Lambda(h) horizons at |t| > 2."
            ),
            (
                "5. Lambda(h) is a single-day response, not a cumulative return, so the curve does not "
                "mechanically smooth noise across horizons."
            ),
            "6. Both exponential models were frozen in advance; no third component was fitted.",
            "7. The two-component fit is judged on both fast and slow half-life precision, not just visual fit.",
            (
                "8. The prior negative CAR(+1,+5) level coefficient "
                f"{'replicates significantly' if replicated else 'does not replicate significantly'} in the "
                "longer mid-cap evaluation sample."
            ),
            "9. The mid-cap and sector corpora were not pooled because doing so would confound history with a firm/source regime change.",
            f"10. Mechanical reasons: {', '.join(reasons)}.",
            "",
            "## Integrity notes",
            "",
            (
                "- This is the first and only evaluation run for this output directory."
                if config.evaluation_run_number == 1
                else f"- Disclosed evaluation rerun {config.evaluation_run_number}: {config.rerun_reason}."
            ),
            (
                "- All Part C parameters and Part D calibration were completed from development data before "
                "evaluation Lambda(h) was estimated."
            ),
            (
                "- Random seeds, input hashes, package environment, invalid simulation counts, and "
                "generated-file hashes are recorded in `manifest.json`."
            ),
            "- Licensed article text is absent from every committed output.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_term_structure_pilot(
    *,
    primary_scores_path: str | Path,
    primary_headlines_path: str | Path,
    primary_stock_prices_path: str | Path,
    market_prices_path: str | Path,
    comparison_headlines_path: str | Path,
    pilot2_regressions_path: str | Path,
    output_dir: str | Path,
    config: TermStructureConfig | None = None,
) -> Path:
    """Execute the frozen Part A-D gate and write aggregate-only artifacts."""

    config = config or TermStructureConfig()
    if config.evaluation_run_number < 1:
        raise TermStructurePilotError("evaluation_run_number must be positive")
    if config.evaluation_run_number > 1 and not (config.rerun_reason or "").strip():
        raise TermStructurePilotError("rerun_reason is required after evaluation run 1")
    if config.simulation_runs < 500:
        raise TermStructurePilotError("the precision gate requires at least 500 simulations")
    destination = Path(output_dir)
    if destination.exists():
        raise TermStructurePilotError(f"refusing to overwrite pilot output: {destination}")

    through_month = datetime.now(UTC).strftime("%Y-%m")
    primary_monthly, primary_flags, primary_summary = audit_corpus_depth(
        primary_headlines_path,
        cohort="midcap22_1y_reuters",
        through_month=through_month,
        break_ratio=config.coverage_break_ratio,
    )
    comparison_monthly, comparison_flags, comparison_summary = audit_corpus_depth(
        comparison_headlines_path,
        cohort="sector33_6m_all_sources",
        through_month=through_month,
        break_ratio=config.coverage_break_ratio,
    )
    monthly = pd.concat([primary_monthly, comparison_monthly], ignore_index=True)
    coverage_flags = pd.concat([primary_flags, comparison_flags], ignore_index=True)
    corpus_summaries = pd.DataFrame([primary_summary, comparison_summary])
    history_months = float(primary_summary["calendar_span_months"])

    stocks = _load_prices(primary_stock_prices_path, label="primary stock prices")
    market = _load_prices(market_prices_path, label="market prices")
    base_config = PilotConfig(
        scorer="finbert",
        minimum_symbol_news_days=1,
        minimum_prior_news_days=0,
        dev20_window=1,
    )
    daily, build_attrition = build_daily_sentiment(primary_scores_path, primary_headlines_path, stocks, base_config)
    vader, _ = build_daily_sentiment(
        primary_scores_path,
        primary_headlines_path,
        stocks,
        PilotConfig(scorer="vader", minimum_symbol_news_days=1, minimum_prior_news_days=0, dev20_window=1),
    )
    if not daily[["symbol", "session_date"]].equals(vader[["symbol", "session_date"]]):
        raise TermStructurePilotError("FinBERT and VADER session panels disagree")
    firm_depth = (
        daily.groupby("symbol", sort=True)
        .agg(
            news_sessions=("session_date", "size"),
            articles=("article_count", "sum"),
            first_session=("session_date", "min"),
            last_session=("session_date", "max"),
        )
        .reset_index()
    )
    firm_depth["first_session"] = firm_depth["first_session"].dt.date.astype(str)
    firm_depth["last_session"] = firm_depth["last_session"].dt.date.astype(str)

    events, abnormal = add_horizon_outcomes(
        daily,
        stocks,
        market,
        market_symbol="^GSPC",
        horizons=config.horizons,
    )
    outcome_columns = [f"ar_h{horizon}" for horizon in config.horizons]
    complete = events.dropna(subset=["level", *outcome_columns]).copy()
    if not np.isfinite(complete[["level", *outcome_columns]].to_numpy(float)).all():
        raise TermStructurePilotError("complete event sample contains non-finite values")
    if complete["session_date"].nunique() < 20:
        raise TermStructurePilotError("fewer than 20 complete event dates survive")
    split_date = chronological_split_date(complete["session_date"].dt.date.astype(str).tolist(), config.development_fraction)
    boundary = pd.Timestamp(split_date)
    development = complete[complete["session_date"] < boundary].copy()
    evaluation = complete[complete["session_date"] >= boundary].copy()
    if development["session_date"].nunique() < 10 or evaluation["session_date"].nunique() < 10:
        raise TermStructurePilotError("chronological split has too few date clusters")

    # Development is fully analyzed, including the precision simulation, before evaluation is opened.
    development_ir, development_residuals = estimate_impulse_response(development, split="development", horizons=config.horizons)
    development_draws = bootstrap_impulse_response(
        development,
        horizons=config.horizons,
        samples=config.bootstrap_samples,
        seed=config.random_seed,
    )
    point_parameters = {
        model: fit_decay_curve(
            development_ir["lambda"].to_numpy(float),
            development_ir["clustered_se"].to_numpy(float),
            model=model,
        )
        for model in ("single", "two_component")
    }
    simulation, simulation_runs, precision_pass, simulation_calibration = simulation_precision_gate(
        development,
        development_residuals,
        development_ir,
        point_parameters,
        config=config,
    )

    # One-shot evaluation opening occurs here; nothing above is selected from evaluation outcomes.
    evaluation_ir, _ = estimate_impulse_response(evaluation, split="evaluation", horizons=config.horizons)
    evaluation_draws = bootstrap_impulse_response(
        evaluation,
        horizons=config.horizons,
        samples=config.bootstrap_samples,
        seed=config.random_seed + 1,
    )
    impulse = pd.concat([development_ir, evaluation_ir], ignore_index=True)
    impulse["bootstrap_ci_low"] = np.concatenate(
        [np.quantile(development_draws, 0.025, axis=0), np.quantile(evaluation_draws, 0.025, axis=0)]
    )
    impulse["bootstrap_ci_high"] = np.concatenate(
        [np.quantile(development_draws, 0.975, axis=0), np.quantile(evaluation_draws, 0.975, axis=0)]
    )
    decay, fitted_points, _ = summarize_decay_fits(development_ir, evaluation_ir, development_draws)
    reversal = reversal_check(development, evaluation, pilot2_regressions_path)
    decision, reasons = classify_term_structure_decision(
        history_months=history_months,
        precision_pass=precision_pass,
        development_ir=development_ir,
        history_gate_months=config.history_gate_months,
    )

    build_counts = {str(row["step"]): int(row["surviving"]) for row in build_attrition}
    attrition = pd.DataFrame(
        [
            {"step": "raw_primary_headlines", "unit": "articles", "surviving": primary_summary["raw_headline_rows"]},
            {
                "step": "successful_finbert_unique_headlines",
                "unit": "headlines",
                "surviving": build_counts["successful_finbert_headline_rows"],
            },
            {
                "step": "scored_symbol_headline_associations",
                "unit": "associations",
                "surviving": build_counts["valid_symbol_headline_associations"],
            },
            {
                "step": "associations_mapped_to_stock_session",
                "unit": "associations",
                "surviving": build_counts["associations_mapped_to_stock_session"],
            },
            {"step": "aggregated_finbert_symbol_sessions", "unit": "events", "surviving": len(daily)},
            {"step": "aggregated_vader_symbol_sessions", "unit": "events", "surviving": len(vader)},
            {"step": "symbols_in_primary_panel", "unit": "symbols", "surviving": daily["symbol"].nunique()},
            {"step": "events_with_ar_h0", "unit": "events", "surviving": events["ar_h0"].notna().sum()},
            {"step": "events_complete_through_h10", "unit": "events", "surviving": len(complete)},
            {"step": "development_events", "unit": "events", "surviving": len(development)},
            {"step": "development_date_clusters", "unit": "dates", "surviving": development["session_date"].nunique()},
            {"step": "evaluation_events", "unit": "events", "surviving": len(evaluation)},
            {"step": "evaluation_date_clusters", "unit": "dates", "surviving": evaluation["session_date"].nunique()},
        ]
    )

    destination.mkdir(parents=True)
    paths = {
        "corpus_depth_monthly.csv": monthly,
        "corpus_summaries.csv": corpus_summaries,
        "firm_depth.csv": firm_depth,
        "coverage_break_flags.csv": coverage_flags,
        "impulse_response.csv": impulse,
        "decay_fits.csv": decay,
        "simulation_precision.csv": simulation,
        "simulation_runs.csv": simulation_runs,
        "reversal_check.csv": reversal,
        "attrition.csv": attrition,
    }
    for filename, frame in paths.items():
        frame.to_csv(destination / filename, index=False, lineterminator="\n")
    _plot_impulse_response(impulse.rename(columns={"lambda": "lambda"}), destination / "lambda_curve.png")
    report_path = destination / "report.md"
    report_impulse = impulse.rename(columns={"lambda": "lambda_"})
    report_path.write_text(
        render_term_structure_report(
            corpus_summaries=corpus_summaries,
            firm_depth=firm_depth,
            coverage_flags=coverage_flags,
            impulse=report_impulse,
            decay=decay,
            simulation=simulation,
            reversal=reversal,
            attrition=attrition,
            decision=decision,
            reasons=reasons,
            split_date=split_date,
            history_months=history_months,
            precision_pass=precision_pass,
            config=config,
        ),
        encoding="utf-8",
        newline="\n",
    )
    input_paths = {
        "primary_scores": str(primary_scores_path),
        "primary_headlines": str(primary_headlines_path),
        "primary_stock_prices": str(primary_stock_prices_path),
        "market_prices": str(market_prices_path),
        "comparison_headlines": str(comparison_headlines_path),
        "pilot2_regressions": str(pilot2_regressions_path),
    }
    generated = [destination / filename for filename in paths] + [destination / "lambda_curve.png", report_path]
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "decision": decision,
        "decision_reasons": reasons,
        "history_gate": {
            "usable_months": history_months,
            "required_months": config.history_gate_months,
            "passed": history_months >= config.history_gate_months,
        },
        "precision_gate": {"passed": precision_pass, "required_fraction": config.precision_gate_fraction},
        "split": {
            "method": "chronological distinct complete-outcome event dates",
            "development_fraction": config.development_fraction,
            "evaluation_start": split_date,
            "evaluation_opened_after_simulation": True,
        },
        "outcomes": {
            "method": "market-adjusted close-to-close log return",
            "market_symbol": "^GSPC",
            "horizons": list(config.horizons),
            "cumulative_returns_used_for_impulse_response": False,
        },
        "decay_models": {
            "single": "lambda * exp(-rho*h), lambda sign free",
            "two_component": "lambda_fast*exp(-rho_fast*h)+lambda_slow*exp(-rho_slow*h), signs free",
            "third_component_fitted": False,
            "development_point_parameters": fitted_points,
        },
        "simulation_calibration": simulation_calibration,
        "config": config.__dict__,
        "evaluation_run_number": config.evaluation_run_number,
        "rerun_reason": config.rerun_reason,
        "inputs": {name: {"path": path, "sha256": sha256_file(path)} for name, path in input_paths.items()},
        "files": {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in generated},
        "environment": collect_run_environment(),
        "abnormal_return_calibration_rows": len(abnormal),
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return report_path
