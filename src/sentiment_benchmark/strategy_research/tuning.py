"""Chronological folds, fixed candidate grid, and deterministic selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, median

import numpy as np


class TuningError(ValueError):
    """Raised when development history cannot support declared tuning."""


@dataclass(frozen=True)
class ChronologicalFold:
    fold_index: int
    training_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]


@dataclass(frozen=True)
class ChronologicalSplit:
    evaluation_start: str
    development_sessions: tuple[str, ...]
    evaluation_sessions: tuple[str, ...]


@dataclass(frozen=True)
class StrategyCandidate:
    half_life_sessions: float
    state_scale_quantile: float
    no_trade_band: float
    severity_power: float = 1.5
    reversal_reset: float = 0.75
    impulse_scale: float = 1.0
    state_cap: float = 3.0
    complexity_rank: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.half_life_sessions) or self.half_life_sessions <= 0:
            raise ValueError("half_life_sessions must be finite and positive")
        if not 0 <= self.state_scale_quantile <= 1:
            raise ValueError("state_scale_quantile must be in [0, 1]")
        if not 0 <= self.no_trade_band <= 1:
            raise ValueError("no_trade_band must be in [0, 1]")
        if not math.isfinite(self.severity_power) or self.severity_power <= 0:
            raise ValueError("severity_power must be finite and positive")
        if not math.isfinite(self.reversal_reset) or not 0 <= self.reversal_reset <= 1:
            raise ValueError("reversal_reset must be in [0, 1]")
        if not math.isfinite(self.impulse_scale) or self.impulse_scale < 0:
            raise ValueError("impulse_scale must be finite and non-negative")
        if not math.isfinite(self.state_cap) or self.state_cap <= 0:
            raise ValueError("state_cap must be finite and positive")
        if self.complexity_rank < 0:
            raise ValueError("complexity_rank must be non-negative")

    @property
    def identity(self) -> tuple[float, ...]:
        return (
            float(self.half_life_sessions),
            self.state_scale_quantile,
            self.no_trade_band,
            self.severity_power,
            self.reversal_reset,
            self.impulse_scale,
            self.state_cap,
        )


@dataclass(frozen=True)
class FoldMetric:
    fold_index: int
    sharpe: float | None
    average_turnover: float
    active_days: int
    supported_stocks: int
    exposure_violations: int = 0
    valid_price_coverage: bool = True
    maximum_single_name_contribution: float | None = None


@dataclass(frozen=True)
class CandidateGuardrails:
    minimum_active_days: int = 10
    minimum_supported_stocks: int = 25
    maximum_exposure_violations: int = 0
    maximum_average_turnover: float | None = None
    maximum_single_name_contribution: float | None = None

    def __post_init__(self) -> None:
        if self.minimum_active_days < 1:
            raise ValueError("minimum_active_days must be positive")
        if self.minimum_supported_stocks < 1:
            raise ValueError("minimum_supported_stocks must be positive")
        if self.maximum_exposure_violations < 0:
            raise ValueError("maximum_exposure_violations must be non-negative")
        if self.maximum_average_turnover is not None and self.maximum_average_turnover < 0:
            raise ValueError("maximum_average_turnover must be non-negative")
        if self.maximum_single_name_contribution is not None and self.maximum_single_name_contribution < 0:
            raise ValueError("maximum_single_name_contribution must be non-negative")


@dataclass(frozen=True)
class CandidateResult:
    candidate: StrategyCandidate
    folds: tuple[FoldMetric, ...]
    objective: float | None
    median_sharpe: float | None
    sharpe_iqr: float | None
    average_turnover: float | None
    valid: bool
    rejection_reasons: tuple[str, ...]


def build_expanding_folds(
    sessions: Sequence[str],
    *,
    requested_folds: int = 3,
    minimum_training_sessions: int = 60,
    requested_validation_sessions: int = 20,
    minimum_validation_sessions: int = 10,
) -> list[ChronologicalFold]:
    """Use all development sessions in the largest defensible 2–3 fold layout."""

    ordered = tuple(sessions)
    if len(ordered) != len(set(ordered)) or tuple(sorted(ordered)) != ordered:
        raise TuningError("sessions must be unique and chronological")
    if requested_folds < 2:
        raise ValueError("requested_folds must be at least two")
    if minimum_training_sessions < 1:
        raise ValueError("minimum_training_sessions must be positive")
    if requested_validation_sessions < minimum_validation_sessions or minimum_validation_sessions < 1:
        raise ValueError("validation lengths are inconsistent")

    available_after_minimum = len(ordered) - minimum_training_sessions
    if available_after_minimum < 2 * minimum_validation_sessions:
        raise TuningError(
            "development history cannot support two validation folds: "
            f"have {len(ordered)} sessions, require at least "
            f"{minimum_training_sessions + 2 * minimum_validation_sessions}"
        )

    exact_required = requested_folds * requested_validation_sessions
    if available_after_minimum >= exact_required:
        fold_count = requested_folds
        validation_length = requested_validation_sessions
    else:
        fold_count = min(requested_folds, available_after_minimum // minimum_validation_sessions)
        if fold_count < 2:
            raise TuningError("development history cannot support two validation folds")
        validation_length = available_after_minimum // fold_count

    initial_training_length = len(ordered) - fold_count * validation_length
    if initial_training_length < minimum_training_sessions:
        raise AssertionError("fold construction violated minimum training history")
    folds: list[ChronologicalFold] = []
    for fold_index in range(fold_count):
        validation_start = initial_training_length + fold_index * validation_length
        validation_end = validation_start + validation_length
        folds.append(
            ChronologicalFold(
                fold_index=fold_index,
                training_sessions=ordered[:validation_start],
                validation_sessions=ordered[validation_start:validation_end],
            )
        )
    return folds


def build_chronological_split(sessions: Sequence[str], evaluation_start: str) -> ChronologicalSplit:
    """Apply a pre-declared development/evaluation boundary without moving it."""

    ordered = tuple(sessions)
    if len(ordered) != len(set(ordered)) or tuple(sorted(ordered)) != ordered:
        raise TuningError("sessions must be unique and chronological")
    development = tuple(session for session in ordered if session < evaluation_start)
    evaluation = tuple(session for session in ordered if session >= evaluation_start)
    if not development:
        raise TuningError("evaluation boundary leaves no development sessions")
    if not evaluation:
        raise TuningError("evaluation boundary leaves no evaluation sessions")
    return ChronologicalSplit(
        evaluation_start=evaluation_start,
        development_sessions=development,
        evaluation_sessions=evaluation,
    )


def default_candidate_grid() -> list[StrategyCandidate]:
    """Return the declared 5 × 3 × 3 reset-strategy grid in stable order."""

    return build_candidate_grid(
        half_life_sessions=(1, 2, 3, 5, 10),
        state_scale_quantiles=(0.50, 0.75, 0.90),
        no_trade_bands=(0.00, 0.10, 0.20),
    )


def build_candidate_grid(
    *,
    half_life_sessions: Sequence[float],
    state_scale_quantiles: Sequence[float],
    no_trade_bands: Sequence[float],
    severity_power: float = 1.5,
    reversal_reset: float = 0.75,
    impulse_scale: float = 1.0,
    state_cap: float = 3.0,
) -> list[StrategyCandidate]:
    """Build a declared Cartesian grid, including smaller offline smoke grids."""

    candidates = [
        StrategyCandidate(
            half_life_sessions=half_life,
            state_scale_quantile=scale_quantile,
            no_trade_band=no_trade_band,
            severity_power=severity_power,
            reversal_reset=reversal_reset,
            impulse_scale=impulse_scale,
            state_cap=state_cap,
        )
        for half_life in half_life_sessions
        for scale_quantile in state_scale_quantiles
        for no_trade_band in no_trade_bands
    ]
    if not candidates:
        raise ValueError("candidate grid dimensions must not be empty")
    identities = [candidate.identity for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate grid contains duplicate configurations")
    return candidates


def annualized_sharpe(daily_net_returns: Sequence[float], *, periods_per_year: int = 252) -> float | None:
    """Calculate after-cost Sharpe with a zero risk-free rate."""

    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(tuple(daily_net_returns), dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return None
    volatility = float(np.std(values, ddof=1))
    if volatility <= 0:
        return None
    return float(np.mean(values) / volatility * math.sqrt(periods_per_year))


def evaluate_candidate(
    candidate: StrategyCandidate,
    fold_metrics: Sequence[FoldMetric],
    *,
    stability_penalty: float = 0.5,
    guardrails: CandidateGuardrails | None = None,
) -> CandidateResult:
    """Apply hard guardrails and the median-Sharpe-minus-IQR objective."""

    if not math.isfinite(stability_penalty) or stability_penalty < 0:
        raise ValueError("stability_penalty must be finite and non-negative")
    limits = guardrails or CandidateGuardrails()
    folds = tuple(sorted(fold_metrics, key=lambda metric: metric.fold_index))
    if not folds or len({metric.fold_index for metric in folds}) != len(folds):
        raise TuningError("fold metrics must contain unique fold indices")
    reasons: list[str] = []
    for metric in folds:
        prefix = f"fold_{metric.fold_index}"
        if metric.sharpe is None or not math.isfinite(metric.sharpe):
            reasons.append(f"{prefix}:undefined_sharpe")
        if not math.isfinite(metric.average_turnover) or metric.average_turnover < 0:
            reasons.append(f"{prefix}:invalid_turnover")
        elif limits.maximum_average_turnover is not None and metric.average_turnover > limits.maximum_average_turnover:
            reasons.append(f"{prefix}:excessive_turnover")
        if metric.active_days < limits.minimum_active_days:
            reasons.append(f"{prefix}:insufficient_active_days")
        if metric.supported_stocks < limits.minimum_supported_stocks:
            reasons.append(f"{prefix}:insufficient_supported_stocks")
        if metric.exposure_violations > limits.maximum_exposure_violations:
            reasons.append(f"{prefix}:exposure_violations")
        if not metric.valid_price_coverage:
            reasons.append(f"{prefix}:invalid_price_coverage")
        if (
            limits.maximum_single_name_contribution is not None
            and metric.maximum_single_name_contribution is not None
            and metric.maximum_single_name_contribution > limits.maximum_single_name_contribution
        ):
            reasons.append(f"{prefix}:excessive_single_name_contribution")

    valid_sharpes = [float(metric.sharpe) for metric in folds if metric.sharpe is not None and math.isfinite(metric.sharpe)]
    valid_turnovers = [
        metric.average_turnover
        for metric in folds
        if math.isfinite(metric.average_turnover) and metric.average_turnover >= 0
    ]
    average_turnover = fmean(valid_turnovers) if len(valid_turnovers) == len(folds) else None
    if reasons:
        objective = None
        median_sharpe = None
        sharpe_iqr = None
    else:
        median_sharpe = float(median(valid_sharpes))
        values = np.asarray(valid_sharpes, dtype=float)
        sharpe_iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        objective = median_sharpe - stability_penalty * sharpe_iqr
    return CandidateResult(
        candidate=candidate,
        folds=folds,
        objective=objective,
        median_sharpe=median_sharpe,
        sharpe_iqr=sharpe_iqr,
        average_turnover=average_turnover,
        valid=not reasons,
        rejection_reasons=tuple(reasons),
    )


def select_candidate(results: Sequence[CandidateResult]) -> CandidateResult:
    """Select one valid candidate using the declared deterministic tie-breaks."""

    valid = [result for result in results if result.valid and result.objective is not None and result.sharpe_iqr is not None]
    if not valid:
        raise TuningError("all strategy candidates failed development guardrails")
    return min(
        valid,
        key=lambda result: (
            -result.objective if result.objective is not None else math.inf,
            result.sharpe_iqr if result.sharpe_iqr is not None else math.inf,
            result.average_turnover if result.average_turnover is not None else math.inf,
            result.candidate.complexity_rank,
            result.candidate.identity,
        ),
    )
