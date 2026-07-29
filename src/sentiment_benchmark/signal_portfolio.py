"""Sharpe-driven combination of sentiment signal return series.

This module answers four questions about a panel of daily signal return series:

1. which two signals maximise the Sharpe ratio of their equal-weighted average;
2. which three signals do;
3. which subset of unspecified size is *add/drop stable* — adding any one further
   signal lowers Sharpe and removing any one held signal also lowers it; and
4. for two signals, what weights maximise Sharpe under a sum-to-one budget,
   solved with Lagrange multipliers.

The signal panel is produced by the existing Week 6 accounting engine
(:mod:`sentiment_benchmark.week6_pnl`) so that every series shares one price
panel, one cost rate, one session convention and one chronological split. Signal
rules are held **fixed** across scorers (one threshold, one holding period) so
that no per-scorer development grid search is embedded in the panel.

Selection is a development-only operation. The chosen subsets and weights are
frozen and then reported once on the evaluation block. ``docs/cli_reference.md``
forbids choosing a best-Sharpe scorer on the same period used to report
performance, so the selection stage never reads evaluation returns.

Because a Sharpe-maximising search over subsets is itself a multiple-comparison
procedure, the module always reports two honesty diagnostics beside the winner:
the Lo (2002) standard error of an annualised Sharpe, and a demeaned
moving-block bootstrap distribution of the *maximum* subset Sharpe under a
no-edge null. The second quantifies how much of the winning Sharpe is
attributable to selecting the best of many correlated candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .week6_pnl import Week6PnlError, annualized_sharpe, build_portfolio_daily, build_stock_daily_pnl, load_and_validate_inputs

MAX_EXHAUSTIVE_SIGNALS = 20
RETURN_VARIANTS = ("gross", "net")


class SignalPortfolioError(ValueError):
    """Raised when signal-portfolio inputs or selection invariants are invalid."""


@dataclass(frozen=True)
class SignalPortfolioConfig:
    """Frozen assumptions for one signal-combination study.

    ``threshold`` and ``holding_period`` are deliberately shared by every scorer.
    Calibrating them per scorer would embed a development grid search inside the
    panel, which is the contamination this study is designed to avoid.
    """

    run_id: str
    scorer_ids: tuple[str, ...]
    return_variant: str = "gross"
    threshold: float = 0.0
    holding_period: int = 1
    starting_capital: float = 100_000.0
    transaction_cost_bps_per_side: float = 10.0
    development_fraction: float = 0.7
    annualization_periods: int = 252
    annual_risk_free_rate: float = 0.0
    exchange_timezone: str = "America/New_York"
    drop_inactive_sessions: bool = True
    bootstrap_replications: int = 2_000
    bootstrap_block_length: int = 5
    random_seed: int = 20260729
    # Provenance disclosure. This study was specified after the Week 6 model
    # comparison had been observed, so the loader refuses to hide that.
    exploratory: bool = True
    specified_after_week6_results: bool = True

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise SignalPortfolioError("run_id must be a non-empty string")
        if len(set(self.scorer_ids)) != len(self.scorer_ids):
            raise SignalPortfolioError("scorer_ids must be unique")
        if len(self.scorer_ids) < 2:
            raise SignalPortfolioError("signal combination requires at least two scorers")
        if self.return_variant not in RETURN_VARIANTS:
            raise SignalPortfolioError(f"return_variant must be one of {RETURN_VARIANTS}")
        if self.threshold < 0:
            raise SignalPortfolioError("threshold must be non-negative")
        if self.holding_period < 1:
            raise SignalPortfolioError("holding_period must be at least one session")
        if self.starting_capital <= 0:
            raise SignalPortfolioError("starting_capital must be positive")
        if self.transaction_cost_bps_per_side < 0:
            raise SignalPortfolioError("transaction cost cannot be negative")
        if not 0.0 < self.development_fraction < 1.0:
            raise SignalPortfolioError("development_fraction must lie strictly between 0 and 1")
        if self.annualization_periods < 1:
            raise SignalPortfolioError("annualization_periods must be positive")
        if self.bootstrap_replications < 1:
            raise SignalPortfolioError("bootstrap_replications must be positive")
        if self.bootstrap_block_length < 1:
            raise SignalPortfolioError("bootstrap_block_length must be positive")
        if not self.exploratory:
            raise SignalPortfolioError("in-sample Sharpe-maximising selection must be recorded as exploratory")
        if not self.specified_after_week6_results:
            raise SignalPortfolioError("this study must disclose that it was specified after the Week 6 results")


@dataclass(frozen=True)
class SubsetSharpe:
    """Sharpe of one equal-weighted signal subset on both blocks."""

    signals: tuple[str, ...]
    size: int
    development_sharpe: float | None
    development_sharpe_standard_error: float | None
    development_mean_daily_return: float
    development_daily_volatility: float
    evaluation_sharpe: float | None
    evaluation_sharpe_standard_error: float | None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signals"] = "|".join(self.signals)
        return payload


@dataclass(frozen=True)
class StabilityCertificate:
    """Add/drop stationarity evidence for one candidate subset."""

    signals: tuple[str, ...]
    size: int
    sharpe: float
    is_add_drop_stable: bool
    best_addition: str | None
    best_addition_sharpe: float | None
    best_removal: str | None
    best_removal_sharpe: float | None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signals"] = "|".join(self.signals)
        return payload


@dataclass(frozen=True)
class TwoSignalWeightSolution:
    """Lagrange solution for the Sharpe-optimal weighted average of two signals."""

    signal_a: str
    signal_b: str
    mean_a: float
    mean_b: float
    volatility_a: float
    volatility_b: float
    correlation: float
    weight_a: float
    weight_b: float
    weight_a_closed_form: float
    closed_form_agrees: bool
    lagrange_multiplier: float
    budget_denominator: float
    stationary_point_is_maximum: bool
    optimal_sharpe: float | None
    equal_weight_sharpe: float | None
    best_single_sharpe: float | None
    gross_leverage: float
    covariance_condition_number: float
    grid_best_weight_a: float
    grid_best_sharpe: float
    long_only_weight_a: float
    long_only_sharpe: float | None
    evaluation_sharpe_at_frozen_weights: float | None
    evaluation_sharpe_long_only: float | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def sharpe_standard_error(returns: Sequence[float] | np.ndarray | pd.Series, *, periods_per_year: int = 252) -> float | None:
    """Lo (2002) standard error of an annualised Sharpe under IID returns.

    ``SE = sqrt(periods_per_year) * sqrt((1 + 0.5 * SR_period^2) / n)``. This is
    the optimistic case: it ignores autocorrelation and the selection performed
    to choose the series in the first place.
    """

    values = _finite_array(returns)
    if len(values) < 3:
        return None
    deviation = float(values.std(ddof=1))
    if deviation == 0.0:
        return None
    period_sharpe = float(values.mean()) / deviation
    return float(math.sqrt(periods_per_year) * math.sqrt((1.0 + 0.5 * period_sharpe**2) / len(values)))


def build_signal_panel(
    signals_path: str | Path,
    prices_path: str | Path,
    config: SignalPortfolioConfig,
) -> pd.DataFrame:
    """Build one daily return series per scorer on a single shared date grid.

    Every scorer is run through the same Week 6 accounting engine with identical
    threshold, holding period, universe, cost rate and split date, so the
    resulting series differ only by the sentiment model that produced the signal.
    """

    from .week6_model_comparison import common_comparison_split

    split_date = common_comparison_split(
        Path(signals_path),
        config.scorer_ids,
        config.development_fraction,
    )
    frames: list[pd.DataFrame] = []
    for scorer_id in config.scorer_ids:
        scorer_signals, prices = load_and_validate_inputs(signals_path, prices_path, scorer_id)
        stock_daily, _ = build_stock_daily_pnl(
            scorer_signals,
            prices,
            threshold=config.threshold,
            holding_period=config.holding_period,
            split_date=split_date,
            starting_capital=config.starting_capital,
            transaction_cost_bps_per_side=config.transaction_cost_bps_per_side,
            exchange_timezone=config.exchange_timezone,
        )
        symbols = sorted(stock_daily["symbol"].unique().tolist())
        portfolio = build_portfolio_daily(stock_daily, {"all_stock_equal_weight": symbols}, starting_capital=config.starting_capital)
        portfolio = portfolio.assign(signal=scorer_id, split_date=split_date)
        frames.append(
            portfolio[["date", "split", "split_date", "signal", "daily_gross_return", "daily_net_return", "turnover", "active_stock_count"]]
        )

    panel = pd.concat(frames, ignore_index=True)
    counts = panel.groupby("signal")["date"].nunique()
    if counts.nunique() != 1:
        raise SignalPortfolioError(f"scorers produced ragged date grids: {counts.to_dict()}")

    if config.drop_inactive_sessions:
        # Sessions where no scorer holds any position carry a structural zero
        # return for every signal. The ragged first price session and the forced
        # split-boundary liquidation are the two known cases. Keeping them
        # shrinks every mean and variance identically and inflates correlation.
        activity = panel.groupby("date")["active_stock_count"].sum()
        keep = set(activity.loc[activity > 0].index)
        dropped = sorted(set(activity.index) - keep)
        panel = panel.loc[panel["date"].isin(keep)].reset_index(drop=True)
        panel.attrs["dropped_inactive_sessions"] = dropped
    else:
        panel.attrs["dropped_inactive_sessions"] = []

    panel.attrs["split_date"] = split_date
    return panel.sort_values(["date", "signal"], kind="stable").reset_index(drop=True)


def panel_return_matrix(panel: pd.DataFrame, *, return_variant: str) -> tuple[pd.DataFrame, pd.Series]:
    """Pivot a signal panel into a date x signal return matrix plus split labels."""

    if return_variant not in RETURN_VARIANTS:
        raise SignalPortfolioError(f"return_variant must be one of {RETURN_VARIANTS}")
    column = f"daily_{return_variant}_return"
    matrix = panel.pivot(index="date", columns="signal", values=column).sort_index()
    if matrix.isna().to_numpy().any():
        raise SignalPortfolioError("signal return matrix contains gaps; scorers must share one date grid")
    splits = panel.groupby("date")["split"].first().reindex(matrix.index)
    return matrix, splits


def equal_weight_returns(matrix: pd.DataFrame, subset: Sequence[str]) -> pd.Series:
    """Equal-weighted average of the selected signal return series."""

    chosen = list(subset)
    if not chosen:
        # The Sharpe of an empty average is undefined, so callers must not ask.
        raise SignalPortfolioError("subset must contain at least one signal")
    missing = sorted(set(chosen) - set(matrix.columns))
    if missing:
        raise SignalPortfolioError(f"signals absent from the panel: {missing}")
    return matrix[chosen].mean(axis=1)


def enumerate_subset_sharpes(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    periods_per_year: int,
    annual_risk_free_rate: float,
) -> tuple[dict[tuple[str, ...], float], list[SubsetSharpe]]:
    """Score every non-empty signal subset on development, and report evaluation.

    Development Sharpe is the selection objective. Evaluation Sharpe is carried
    alongside purely for reporting and is never consulted by the search.
    """

    names = tuple(development.columns)
    if len(names) > MAX_EXHAUSTIVE_SIGNALS:
        raise SignalPortfolioError(
            f"exhaustive enumeration refuses {len(names)} signals; the 2^n subset space exceeds {MAX_EXHAUSTIVE_SIGNALS}"
        )
    objective: dict[tuple[str, ...], float] = {}
    rows: list[SubsetSharpe] = []
    for size in range(1, len(names) + 1):
        for subset in combinations(names, size):
            development_returns = equal_weight_returns(development, subset)
            evaluation_returns = equal_weight_returns(evaluation, subset)
            development_sharpe = annualized_sharpe(
                development_returns,
                periods_per_year=periods_per_year,
                annual_risk_free_rate=annual_risk_free_rate,
            )
            if development_sharpe is None:
                continue
            objective[subset] = development_sharpe
            rows.append(
                SubsetSharpe(
                    signals=subset,
                    size=size,
                    development_sharpe=development_sharpe,
                    development_sharpe_standard_error=sharpe_standard_error(development_returns, periods_per_year=periods_per_year),
                    development_mean_daily_return=float(development_returns.mean()),
                    development_daily_volatility=float(development_returns.std(ddof=1)),
                    evaluation_sharpe=annualized_sharpe(
                        evaluation_returns,
                        periods_per_year=periods_per_year,
                        annual_risk_free_rate=annual_risk_free_rate,
                    ),
                    evaluation_sharpe_standard_error=sharpe_standard_error(evaluation_returns, periods_per_year=periods_per_year),
                )
            )
    if not objective:
        raise SignalPortfolioError("no signal subset produced a defined Sharpe ratio")
    return objective, rows


def best_subset_of_size(objective: dict[tuple[str, ...], float], size: int) -> tuple[tuple[str, ...], float]:
    """Return the development-Sharpe-maximising subset of exactly ``size`` signals."""

    candidates = {subset: value for subset, value in objective.items() if len(subset) == size}
    if not candidates:
        raise SignalPortfolioError(f"no subset of size {size} has a defined Sharpe ratio")
    best = max(candidates, key=lambda subset: (candidates[subset], tuple(-ord(character) for character in "".join(subset))))
    return best, candidates[best]


def certify_add_drop_stability(
    objective: dict[tuple[str, ...], float],
    subset: tuple[str, ...],
    names: Sequence[str],
) -> StabilityCertificate:
    """Test the user's stopping rule on one subset.

    A subset is stable when every single-signal addition and every single-signal
    removal strictly lowers Sharpe. A singleton is vacuously removal-stable
    because the empty average has no defined Sharpe.
    """

    current = objective[subset]
    held = set(subset)
    additions = {name: objective[tuple(sorted(held | {name}))] for name in names if name not in held}
    removals = {name: objective[tuple(sorted(held - {name}))] for name in subset} if len(subset) > 1 else {}
    best_addition = max(additions, key=lambda key: additions[key]) if additions else None
    best_removal = max(removals, key=lambda key: removals[key]) if removals else None
    stable = all(value < current for value in additions.values()) and all(value < current for value in removals.values())
    return StabilityCertificate(
        signals=subset,
        size=len(subset),
        sharpe=current,
        is_add_drop_stable=stable,
        best_addition=best_addition,
        best_addition_sharpe=additions.get(best_addition) if best_addition else None,
        best_removal=best_removal,
        best_removal_sharpe=removals.get(best_removal) if best_removal else None,
    )


def stepwise_search(objective: dict[tuple[str, ...], float], names: Sequence[str], *, start: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Greedy add/drop hill climb, returning the visited path.

    The path terminates exactly when no single addition or removal improves
    Sharpe — that is, on an add/drop-stable subset. Recording the path shows
    whether a practitioner using stepwise selection would find the global
    optimum or stall on a local one.
    """

    current = tuple(sorted(start))
    if current not in objective:
        raise SignalPortfolioError(f"stepwise start subset is not scoreable: {current}")
    path = [current]
    seen = {current}
    while True:
        held = set(current)
        moves = [tuple(sorted(held | {name})) for name in names if name not in held]
        if len(current) > 1:
            moves.extend(tuple(sorted(held - {name})) for name in current)
        improvements = [move for move in moves if move in objective and objective[move] > objective[current] and move not in seen]
        if not improvements:
            return path
        current = max(improvements, key=lambda move: objective[move])
        seen.add(current)
        path.append(current)


def solve_two_signal_weights(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    signal_a: str,
    signal_b: str,
    *,
    periods_per_year: int,
    annual_risk_free_rate: float,
    grid_limit: float = 25.0,
    grid_points: int = 100_001,
) -> TwoSignalWeightSolution:
    r"""Maximise the Sharpe ratio of ``w_a * a + w_b * b`` subject to ``w_a + w_b = 1``.

    Lagrangian, with :math:`m = w^\top \mu` and :math:`v = w^\top \Sigma w`:

    .. math::
        \mathcal{L}(w, \lambda) = \frac{w^\top \mu}{\sqrt{v}} - \lambda(\mathbf{1}^\top w - 1)

    The stationarity condition is

    .. math::
        \frac{\mu}{\sqrt{v}} - \frac{m\,\Sigma w}{v^{3/2}} = \lambda \mathbf{1}.

    Pre-multiplying by :math:`w^\top` and using :math:`\mathbf{1}^\top w = 1`
    gives :math:`m - m = \lambda`, hence :math:`\lambda = 0`: the Sharpe ratio is
    scale invariant, so the budget constraint never binds. The condition then
    reduces to :math:`\Sigma w \propto \mu`, giving the tangency direction

    .. math::
        w^{*} = \frac{\Sigma^{-1}\mu}{\mathbf{1}^\top \Sigma^{-1}\mu}.

    The denominator :math:`d = \mathbf{1}^\top \Sigma^{-1}\mu` decides the nature
    of the stationary point. When ``d > 0`` the normalised point is the
    constrained maximum. When ``d < 0`` the same point is the constrained
    *minimum* and the supremum is only approached asymptotically as leverage
    grows without bound, so no maximum is attained; the reported solution is
    then flagged rather than presented as optimal.
    """

    for name in (signal_a, signal_b):
        if name not in development.columns:
            raise SignalPortfolioError(f"signal absent from the panel: {name}")
    if signal_a == signal_b:
        raise SignalPortfolioError("two-signal weighting requires two distinct signals")

    pair = [signal_a, signal_b]
    sample = development[pair].to_numpy(dtype=float)
    mean = sample.mean(axis=0)
    covariance = np.cov(sample, rowvar=False, ddof=1)
    if not np.all(np.isfinite(covariance)) or np.linalg.matrix_rank(covariance) < 2:
        raise SignalPortfolioError(f"covariance of {signal_a} and {signal_b} is singular; weights are not identified")

    inverse = np.linalg.inv(covariance)
    direction = inverse @ mean
    denominator = float(np.ones(2) @ direction)
    if denominator == 0.0:
        raise SignalPortfolioError("1' Sigma^-1 mu is zero; the sum-to-one normalisation is undefined")
    weights = direction / denominator

    volatility_a, volatility_b = (float(math.sqrt(covariance[0, 0])), float(math.sqrt(covariance[1, 1])))
    correlation = float(covariance[0, 1] / (volatility_a * volatility_b)) if volatility_a and volatility_b else float("nan")
    # Closed form for two assets, used as an independent check on the algebra.
    closed_numerator = mean[0] * volatility_b**2 - mean[1] * correlation * volatility_a * volatility_b
    closed_denominator = (
        mean[0] * volatility_b**2 + mean[1] * volatility_a**2 - correlation * volatility_a * volatility_b * (mean[0] + mean[1])
    )
    weight_a_closed_form = float(closed_numerator / closed_denominator) if closed_denominator else float("nan")

    def sharpe_at(weight_a: float, frame: pd.DataFrame) -> float | None:
        combined = frame[pair].to_numpy(dtype=float) @ np.array([weight_a, 1.0 - weight_a], dtype=float)
        return annualized_sharpe(combined, periods_per_year=periods_per_year, annual_risk_free_rate=annual_risk_free_rate)

    grid = np.linspace(-grid_limit, grid_limit, grid_points)
    grid_values = np.array([sharpe_at(value, development) or -np.inf for value in grid], dtype=float)
    grid_index = int(np.argmax(grid_values))
    optimal_sharpe = sharpe_at(float(weights[0]), development)
    # Long-only: the feasible set is the segment [0, 1], so the KKT solution is
    # the interior stationary point when it is feasible and a maximum, else the
    # better endpoint.
    long_only_candidates = [0.0, 1.0] + ([float(weights[0])] if 0.0 <= weights[0] <= 1.0 and denominator > 0 else [])
    long_only_weight = max(long_only_candidates, key=lambda value: sharpe_at(value, development) or -np.inf)

    return TwoSignalWeightSolution(
        signal_a=signal_a,
        signal_b=signal_b,
        mean_a=float(mean[0]),
        mean_b=float(mean[1]),
        volatility_a=volatility_a,
        volatility_b=volatility_b,
        correlation=correlation,
        weight_a=float(weights[0]),
        weight_b=float(weights[1]),
        weight_a_closed_form=weight_a_closed_form,
        closed_form_agrees=bool(np.isclose(weights[0], weight_a_closed_form, rtol=1e-9, atol=1e-12)),
        lagrange_multiplier=0.0,
        budget_denominator=denominator,
        stationary_point_is_maximum=denominator > 0,
        optimal_sharpe=optimal_sharpe,
        equal_weight_sharpe=sharpe_at(0.5, development),
        best_single_sharpe=max(
            (value for value in (sharpe_at(1.0, development), sharpe_at(0.0, development)) if value is not None),
            default=None,
        ),
        gross_leverage=float(np.abs(weights).sum()),
        covariance_condition_number=float(np.linalg.cond(covariance)),
        grid_best_weight_a=float(grid[grid_index]),
        grid_best_sharpe=float(grid_values[grid_index]),
        long_only_weight_a=long_only_weight,
        long_only_sharpe=sharpe_at(long_only_weight, development),
        evaluation_sharpe_at_frozen_weights=sharpe_at(float(weights[0]), evaluation),
        evaluation_sharpe_long_only=sharpe_at(long_only_weight, evaluation),
    )


def selection_null_distribution(
    development: pd.DataFrame,
    *,
    periods_per_year: int,
    replications: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    """Distribution of the best-of-all-subsets Sharpe under a no-edge null.

    Returns are demeaned so that the null is "no signal has an edge" while the
    contemporaneous covariance between signals is preserved. Resampling uses
    overlapping blocks to retain short-horizon dependence. Comparing the observed
    winning Sharpe against this distribution measures how much of it is explained
    by taking the maximum over many correlated candidates.
    """

    values = development.to_numpy(dtype=float)
    observations, width = values.shape
    if observations < block_length + 1:
        raise SignalPortfolioError("development block is too short for the requested bootstrap block length")
    centred = values - values.mean(axis=0)
    subsets = [list(subset) for size in range(1, width + 1) for subset in combinations(range(width), size)]
    generator = np.random.default_rng(seed)
    blocks = max(1, math.ceil(observations / block_length))
    maxima = np.empty(replications, dtype=float)
    scale = math.sqrt(periods_per_year)
    for replication in range(replications):
        starts = generator.integers(0, observations - block_length + 1, blocks)
        index = np.concatenate([np.arange(start, start + block_length) for start in starts])[:observations]
        resample = centred[index]
        best = -np.inf
        for subset in subsets:
            series = resample[:, subset].mean(axis=1)
            deviation = series.std(ddof=1)
            if deviation > 0:
                best = max(best, scale * float(series.mean()) / float(deviation))
        maxima[replication] = best
    finite = maxima[np.isfinite(maxima)]
    return {
        "replications": int(len(finite)),
        "block_length": block_length,
        "seed": seed,
        "subsets_searched": len(subsets),
        "mean_max_sharpe": float(finite.mean()),
        "median_max_sharpe": float(np.median(finite)),
        "percentile_95_max_sharpe": float(np.quantile(finite, 0.95)),
        "maximum_max_sharpe": float(finite.max()),
        "distribution": finite,
    }


def bootstrap_subset_sharpe(
    returns: pd.Series,
    *,
    periods_per_year: int,
    replications: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    """Moving-block bootstrap interval for one series' annualised Sharpe."""

    values = _finite_array(returns)
    if len(values) < block_length + 1:
        raise SignalPortfolioError("series is too short for the requested bootstrap block length")
    generator = np.random.default_rng(seed)
    blocks = max(1, math.ceil(len(values) / block_length))
    scale = math.sqrt(periods_per_year)
    estimates: list[float] = []
    for _ in range(replications):
        starts = generator.integers(0, len(values) - block_length + 1, blocks)
        index = np.concatenate([np.arange(start, start + block_length) for start in starts])[: len(values)]
        resample = values[index]
        deviation = resample.std(ddof=1)
        if deviation > 0:
            estimates.append(scale * float(resample.mean()) / float(deviation))
    if not estimates:
        raise SignalPortfolioError("bootstrap produced no defined Sharpe replications")
    array = np.asarray(estimates, dtype=float)
    return {
        "replications": int(len(array)),
        "block_length": block_length,
        "seed": seed,
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "confidence_interval_low": float(np.quantile(array, 0.025)),
        "confidence_interval_high": float(np.quantile(array, 0.975)),
        "share_at_or_below_zero": float(np.mean(array <= 0.0)),
    }


def breakeven_cost_bps_per_side(panel: pd.DataFrame, subset: Sequence[str], *, split: str = "development") -> float | None:
    """Cost rate at which the subset's mean net return reaches zero.

    The engine charges ``allocation * turnover * bps / 10_000`` per stock, so mean
    net return is linear in the cost rate and the breakeven rate is
    ``10_000 * mean(gross) / mean(turnover)``.
    """

    rows = panel.loc[panel["signal"].isin(list(subset)) & panel["split"].eq(split)]
    if rows.empty:
        return None
    gross = rows.groupby("date")["daily_gross_return"].mean().mean()
    turnover = rows.groupby("date")["turnover"].mean().mean()
    if not math.isfinite(gross) or not math.isfinite(turnover) or turnover <= 0:
        return None
    return float(10_000.0 * gross / turnover)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(cwd: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip())
        return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "branch": None, "dirty": None}


def _number(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


@dataclass(frozen=True)
class SignalPortfolioResult:
    """Paths and headline choices from a completed study."""

    output_dir: Path
    split_date: str
    best_pair: tuple[str, ...]
    best_triple: tuple[str, ...]
    stable_subsets: tuple[tuple[str, ...], ...]
    global_best_subset: tuple[str, ...]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _render_summary(
    config: SignalPortfolioConfig,
    split_date: str,
    panel: pd.DataFrame,
    subset_rows: list[SubsetSharpe],
    objective: dict[tuple[str, ...], float],
    best_pair: tuple[str, ...],
    best_triple: tuple[str, ...],
    certificates: list[StabilityCertificate],
    stepwise_paths: dict[str, list[tuple[str, ...]]],
    pair_solutions: list[TwoSignalWeightSolution],
    null_summary: dict[str, Any],
    bootstrap: dict[str, Any],
    breakeven: float | None,
    correlations: pd.DataFrame,
) -> str:
    by_size = {row.size: row for row in sorted(subset_rows, key=lambda item: (item.size, -(item.development_sharpe or -np.inf)))}
    stable = [certificate for certificate in certificates if certificate.is_add_drop_stable]
    global_best = max(objective, key=lambda subset: objective[subset])
    headline = pair_solutions[0] if pair_solutions else None
    observed_best = objective[global_best]
    null_share = float(np.mean(null_summary["distribution"] >= observed_best))

    lines = [
        f"# Signal portfolio optimisation — `{config.run_id}`",
        "",
        "> This is an exploratory, in-sample selection study on a previously explored cohort.",
        "> It is not deployable alpha, investment advice, or causal evidence. Subsets and",
        "> weights are chosen on the development block only; the evaluation block is",
        "> reported once and never used to select.",
        "",
        "## Frozen design",
        "",
        f"- Signals: {len(config.scorer_ids)} scorers, one daily return series each — "
        + ", ".join(f"`{name}`" for name in config.scorer_ids)
        + ".",
        f"- Return variant used for selection: **{config.return_variant}**.",
        f"- Shared signal rule: threshold `{config.threshold}`, holding period `{config.holding_period}` session(s). "
        "No per-scorer calibration, so the panel embeds no development grid search.",
        f"- Cost: `{config.transaction_cost_bps_per_side}` bps per side on `|change in position|`.",
        f"- Split: development before `{split_date}`, evaluation on and after it.",
        f"- Sharpe: annualised over `{config.annualization_periods}` periods at a "
        f"`{config.annual_risk_free_rate}` risk-free rate, sample volatility (`ddof=1`).",
        f"- Sessions dropped as structurally inactive: {panel.attrs.get('dropped_inactive_sessions') or 'none'}.",
        "",
        "## Signal correlations (development)",
        "",
        "| Signal | " + " | ".join(name.split("/")[-1] for name in correlations.columns) + " |",
        "| --- |" + " --- |" * len(correlations.columns),
    ]
    for name, row in correlations.iterrows():
        lines.append(f"| `{name}` | " + " | ".join(_number(value, 2) for value in row) + " |")

    lines.extend(
        [
            "",
            "## Best equal-weighted subset by cardinality",
            "",
            "| Size | Development Sharpe | Std. error | Evaluation Sharpe | Members |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for size in sorted(by_size):
        best_subset, best_value = best_subset_of_size(objective, size)
        row = next(item for item in subset_rows if item.signals == best_subset)
        lines.append(
            f"| {size} | {_number(best_value)} | {_number(row.development_sharpe_standard_error)} | "
            f"{_number(row.evaluation_sharpe)} | {', '.join(name.split('/')[-1] for name in best_subset)} |"
        )

    lines.extend(
        [
            "",
            "## Part 1 and 2 — best two and three signals",
            "",
            f"- **Best pair** (development Sharpe {_number(objective[best_pair])}): {', '.join(f'`{name}`' for name in best_pair)}.",
            f"- **Best triple** (development Sharpe {_number(objective[best_triple])}): {', '.join(f'`{name}`' for name in best_triple)}.",
            "",
            "## Part 3 — subset of unspecified size",
            "",
            "The stopping rule is add/drop stationarity: adding any one further signal lowers",
            "Sharpe, and removing any one held signal also lowers it. A singleton is vacuously",
            "removal-stable because the empty average has no defined Sharpe.",
            "",
            f"- Subsets searched exhaustively: **{len(objective)}**.",
            f"- Add/drop-stable subsets found: **{len(stable)}**.",
            "- Global development maximum: **"
            + ", ".join(name.split("/")[-1] for name in global_best)
            + f"** at Sharpe {_number(observed_best)}.",
            "",
            "| Stable subset | Size | Development Sharpe | Best forgone addition | Best forgone removal |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for certificate in sorted(stable, key=lambda item: -item.sharpe):
        addition = f"`{certificate.best_addition}` -> {_number(certificate.best_addition_sharpe)}" if certificate.best_addition else "n/a"
        removal = f"`{certificate.best_removal}` -> {_number(certificate.best_removal_sharpe)}" if certificate.best_removal else "n/a"
        lines.append(
            f"| {', '.join(name.split('/')[-1] for name in certificate.signals)} | {certificate.size} | "
            f"{_number(certificate.sharpe)} | {addition} | {removal} |"
        )

    lines.extend(["", "### Stepwise paths", ""])
    for label, path in stepwise_paths.items():
        rendered = " -> ".join("{" + ", ".join(name.split("/")[-1] for name in step) + "}" for step in path)
        lines.append(f"- From {label}: {rendered} (terminal Sharpe {_number(objective[path[-1]])}).")

    if headline is not None:
        lines.extend(
            [
                "",
                "## Part 4 — Sharpe-optimal weights for two signals (Lagrange)",
                "",
                "Maximising `w'mu / sqrt(w'Sigma w)` subject to `1'w = 1` gives a **zero multiplier**:",
                "the Sharpe ratio is scale invariant, so the budget constraint does not bind. The",
                "stationarity condition collapses to `Sigma w` proportional to `mu`, hence",
                "`w* = Sigma^-1 mu / (1' Sigma^-1 mu)`.",
                "",
                f"For the best pair `{headline.signal_a}` and `{headline.signal_b}`:",
                "",
                "| Quantity | Value |",
                "| --- | --- |",
                f"| Correlation | {_number(headline.correlation, 4)} |",
                f"| Mean daily return (bps) | {_number(headline.mean_a * 1e4, 3)} / {_number(headline.mean_b * 1e4, 3)} |",
                f"| Daily volatility (bps) | {_number(headline.volatility_a * 1e4, 3)} / {_number(headline.volatility_b * 1e4, 3)} |",
                f"| Optimal weights | {_number(headline.weight_a, 4)} / {_number(headline.weight_b, 4)} |",
                f"| Closed form agrees with linear algebra | {headline.closed_form_agrees} |",
                f"| Lagrange multiplier | {_number(headline.lagrange_multiplier, 6)} |",
                f"| `1' Sigma^-1 mu` | {_number(headline.budget_denominator, 4)} |",
                f"| Stationary point is a maximum | {headline.stationary_point_is_maximum} |",
                f"| Sharpe: equal weight | {_number(headline.equal_weight_sharpe)} |",
                f"| Sharpe: better single signal | {_number(headline.best_single_sharpe)} |",
                f"| Sharpe: optimal weights | {_number(headline.optimal_sharpe)} |",
                f"| Gross leverage `|w_a| + |w_b|` | {_number(headline.gross_leverage)} |",
                f"| Covariance condition number | {_number(headline.covariance_condition_number, 1)} |",
                f"| Grid check: best weight / Sharpe | {_number(headline.grid_best_weight_a)} / {_number(headline.grid_best_sharpe)} |",
                f"| Long-only weights | {_number(headline.long_only_weight_a, 4)} / {_number(1 - headline.long_only_weight_a, 4)} |",
                f"| Sharpe: long-only | {_number(headline.long_only_sharpe)} |",
                f"| Evaluation Sharpe at frozen optimal weights | {_number(headline.evaluation_sharpe_at_frozen_weights)} |",
                f"| Evaluation Sharpe at frozen long-only weights | {_number(headline.evaluation_sharpe_long_only)} |",
                "",
                "`pair_weight_solutions.csv` carries the same solve for every pair. Rows where",
                "`stationary_point_is_maximum` is false have `1' Sigma^-1 mu < 0`: there the",
                "stationary point is the constrained *minimum* and the supremum is approached only",
                "as leverage diverges, so no finite maximum exists.",
            ]
        )

    lines.extend(
        [
            "",
            "## How much of the winner is selection luck?",
            "",
            f"- Lo (2002) standard error of the winning development Sharpe: "
            f"{_number(next(item.development_sharpe_standard_error for item in subset_rows if item.signals == global_best))}.",
            f"- Moving-block bootstrap of the winning subset: mean {_number(bootstrap['mean'])}, "
            f"95% interval [{_number(bootstrap['confidence_interval_low'])}, {_number(bootstrap['confidence_interval_high'])}], "
            f"share at or below zero {_number(bootstrap['share_at_or_below_zero'], 3)}.",
            f"- Demeaned no-edge null, maximum over all {null_summary['subsets_searched']} subsets across "
            f"{null_summary['replications']} replications: mean {_number(null_summary['mean_max_sharpe'])}, "
            f"median {_number(null_summary['median_max_sharpe'])}, 95th percentile {_number(null_summary['percentile_95_max_sharpe'])}.",
            f"- Share of null replications at or above the observed winning Sharpe: **{_number(null_share, 3)}**.",
            "",
            f"- Breakeven cost for the winning subset: {_number(breakeven, 2)} bps per side "
            f"(charged rate {config.transaction_cost_bps_per_side} bps).",
            "",
            "## Limitations",
            "",
            "- Selection is in-sample by construction. The subset and weight searches maximise a",
            "  development statistic, and the null distribution above shows how large that",
            "  statistic becomes by chance alone when the best of many correlated candidates is",
            "  taken. Treat the winning Sharpe as an upper bound, not an estimate.",
            "- The signal universe is not independent. Several scorers are aggregations of the",
            "  same underlying lexicon over nested headline populations, so their return series",
            "  are near-duplicates. See the correlation table.",
            "- The development and evaluation blocks are short. At these sample sizes the standard",
            "  error of an annualised Sharpe is of the same order as the spread between candidate",
            "  subsets, so rankings are not statistically separated.",
            "- The cohort was already explored by earlier work in this repository. The evaluation",
            "  block is leakage-controlled within this command but is not a pristine confirmatory",
            "  holdout.",
            "- Unconstrained Lagrange weights are unstable when signals are highly correlated: the",
            "  covariance matrix is near-singular, and the solution takes large offsetting long and",
            "  short positions that are unlikely to survive out of sample. The long-only variant is",
            "  reported alongside for that reason.",
            "- Returns are price returns on a fixed notional book with a zero risk-free rate. Borrow",
            "  cost, financing, dividends, spread, market impact and capacity are not modelled.",
            "- Weights are estimated from a sample covariance matrix without shrinkage, so that the",
            "  Lagrange solution is reported exactly as derived. A shrinkage estimator would change",
            "  the weights and is a separate design choice.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_signal_portfolio(
    signals_path: str | Path,
    prices_path: str | Path,
    output_root: str | Path,
    config: SignalPortfolioConfig,
    *,
    command: str,
    repo_root: Path,
) -> SignalPortfolioResult:
    """Execute the complete signal-combination study into a new run directory."""

    signal_file = Path(signals_path).expanduser().resolve()
    price_file = Path(prices_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve() / config.run_id
    if destination.exists():
        raise SignalPortfolioError(f"output directory already exists and is immutable: {destination}")

    panel = build_signal_panel(signal_file, price_file, config)
    split_date = str(panel.attrs["split_date"])
    matrix, splits = panel_return_matrix(panel, return_variant=config.return_variant)
    development = matrix.loc[splits.eq("development")]
    evaluation = matrix.loc[splits.eq("evaluation")]
    if len(development) < 3 or len(evaluation) < 3:
        raise SignalPortfolioError("both blocks need at least three sessions to estimate a Sharpe ratio")

    names = tuple(matrix.columns)
    objective, subset_rows = enumerate_subset_sharpes(
        development,
        evaluation,
        periods_per_year=config.annualization_periods,
        annual_risk_free_rate=config.annual_risk_free_rate,
    )
    best_pair, _ = best_subset_of_size(objective, 2)
    best_triple, _ = best_subset_of_size(objective, 3)
    certificates = [certify_add_drop_stability(objective, subset, names) for subset in objective]
    stable = tuple(certificate.signals for certificate in certificates if certificate.is_add_drop_stable)
    global_best = max(objective, key=lambda subset: objective[subset])

    stepwise_paths = {
        "the empty-to-best forward start": stepwise_search(objective, names, start=best_subset_of_size(objective, 1)[0]),
        "the full set": stepwise_search(objective, names, start=names),
    }

    pair_solutions = [
        solve_two_signal_weights(
            development,
            evaluation,
            best_pair[0],
            best_pair[1],
            periods_per_year=config.annualization_periods,
            annual_risk_free_rate=config.annual_risk_free_rate,
        )
    ]
    pair_solutions.extend(
        solve_two_signal_weights(
            development,
            evaluation,
            pair[0],
            pair[1],
            periods_per_year=config.annualization_periods,
            annual_risk_free_rate=config.annual_risk_free_rate,
        )
        for pair in combinations(names, 2)
        if tuple(sorted(pair)) != tuple(sorted(best_pair))
    )

    null_summary = selection_null_distribution(
        development,
        periods_per_year=config.annualization_periods,
        replications=config.bootstrap_replications,
        block_length=config.bootstrap_block_length,
        seed=config.random_seed,
    )
    bootstrap = bootstrap_subset_sharpe(
        equal_weight_returns(development, global_best),
        periods_per_year=config.annualization_periods,
        replications=config.bootstrap_replications,
        block_length=config.bootstrap_block_length,
        seed=config.random_seed + 1,
    )
    breakeven = breakeven_cost_bps_per_side(panel, global_best)
    correlations = development.corr()

    staging = Path(tempfile.mkdtemp(prefix=f"{config.run_id}-", dir=destination.parent if destination.parent.exists() else None))
    try:
        _write_csv(panel, staging / "signal_panel.csv")
        _write_csv(pd.DataFrame([row.to_payload() for row in subset_rows]), staging / "subset_sharpe.csv")
        _write_csv(pd.DataFrame([item.to_payload() for item in certificates]), staging / "stability_certificates.csv")
        _write_csv(pd.DataFrame([item.to_payload() for item in pair_solutions]), staging / "pair_weight_solutions.csv")
        _write_csv(correlations.reset_index().rename(columns={"index": "signal"}), staging / "development_correlations.csv")
        _write_csv(
            pd.DataFrame(
                [
                    {"start": "|".join(path[0]), "step": index, "subset": "|".join(step), "development_sharpe": objective[step]}
                    for label, path in stepwise_paths.items()
                    for index, step in enumerate(path)
                ]
            ),
            staging / "stepwise_paths.csv",
        )
        summary = _render_summary(
            config,
            split_date,
            panel,
            subset_rows,
            objective,
            best_pair,
            best_triple,
            certificates,
            stepwise_paths,
            pair_solutions,
            null_summary,
            bootstrap,
            breakeven,
            correlations,
        )
        (staging / "summary.md").write_text(summary, encoding="utf-8")

        null_payload = {key: value for key, value in null_summary.items() if key != "distribution"}
        null_payload["share_at_or_above_observed"] = float(np.mean(null_summary["distribution"] >= objective[global_best]))
        manifest = {
            "command": command,
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration": asdict(config),
            "development_evaluation": {
                "split_date": split_date,
                "development_sessions": int(len(development)),
                "evaluation_sessions": int(len(evaluation)),
            },
            "inputs": {
                "daily_signals": {"path": str(signal_file), "sha256": _sha256(signal_file)},
                "prices": {"path": str(price_file), "sha256": _sha256(price_file)},
            },
            "selection": {
                "subsets_searched": len(objective),
                "best_pair": list(best_pair),
                "best_triple": list(best_triple),
                "global_best_subset": list(global_best),
                "global_best_development_sharpe": objective[global_best],
                "add_drop_stable_subsets": [list(subset) for subset in stable],
            },
            "inference": {
                "selection_null": null_payload,
                "winning_subset_bootstrap": bootstrap,
                "breakeven_cost_bps_per_side": breakeven,
            },
            "analysis": {
                "exploratory": config.exploratory,
                "preregistered": False,
                "causal_claim": False,
                "specified_after_week6_results": config.specified_after_week6_results,
                "interpretation_notes": (
                    "Subset and weight selection maximise a development Sharpe over many correlated candidates. "
                    "The demeaned no-edge null quantifies the resulting selection bias; the winning Sharpe is an "
                    "upper bound rather than an estimate."
                ),
            },
            "git": _git_metadata(repo_root),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        output_hashes = {path.name: _sha256(path) for path in sorted(staging.iterdir()) if path.is_file() and path.name != "manifest.json"}
        manifest["output_hashes"] = output_hashes
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(destination))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return SignalPortfolioResult(
        output_dir=destination,
        split_date=split_date,
        best_pair=best_pair,
        best_triple=best_triple,
        stable_subsets=stable,
        global_best_subset=global_best,
    )


__all__ = [
    "SignalPortfolioConfig",
    "SignalPortfolioError",
    "SignalPortfolioResult",
    "StabilityCertificate",
    "SubsetSharpe",
    "TwoSignalWeightSolution",
    "Week6PnlError",
    "best_subset_of_size",
    "bootstrap_subset_sharpe",
    "breakeven_cost_bps_per_side",
    "build_signal_panel",
    "certify_add_drop_stability",
    "enumerate_subset_sharpes",
    "equal_weight_returns",
    "panel_return_matrix",
    "run_signal_portfolio",
    "selection_null_distribution",
    "sharpe_standard_error",
    "solve_two_signal_weights",
    "stepwise_search",
]
