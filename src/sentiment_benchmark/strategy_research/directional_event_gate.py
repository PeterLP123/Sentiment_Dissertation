"""Development-only directional event study for model-only strategy labels.

The gate deliberately stops before portfolio construction.  It collapses each
stock-session to one strongest eligible event, measures forward adjusted-open
returns, and requires direction separation in frozen validation folds.  No
evaluation-period return enters a calculation and no model call is made here.
"""

from __future__ import annotations

import csv
import html
import io
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from ..artifact_io import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from .market import load_adjusted_opens_csv
from .model_only_development import (
    MODEL_ONLY_CONTRACT,
    ModelOnlyDevelopmentError,
    validate_model_only_scoring_universe,
)
from .run_validation import (
    CompletedRunValidationError,
    load_completed_run_snapshot,
    validate_adjusted_open_price_panel,
    validate_completed_stage_output,
)
from .schemas import load_strategy_events

LEVEL_ORDER = {"none": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4}
DIRECTION_STRENGTH = {
    "negative": 1,
    "positive": 1,
    "very_negative": 2,
    "very_positive": 2,
}


class DirectionalEventGateError(RuntimeError):
    """Raised when the frozen event-study contract cannot be evaluated safely."""


@dataclass(frozen=True)
class DirectionalEventGateSpec:
    """Predeclared evidence requirements for the directional event gate."""

    horizons: tuple[int, ...] = (1, 2, 3, 5, 10)
    bootstrap_replications: int = 2_000
    block_length: int = 5
    seed: int = 20_260_715
    familywise_alpha: float = 0.05
    maximum_one_sided_p_value: float = 0.01
    minimum_observations_per_direction: int = 50
    minimum_observations_per_direction_per_fold: int = 10
    minimum_supported_stocks: int = 25
    required_positive_folds: int = 3

    def __post_init__(self) -> None:
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if any(value < 1 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if self.bootstrap_replications < 1 or self.block_length < 1:
            raise ValueError("bootstrap parameters must be positive")
        if not 0 < self.familywise_alpha < 1 or not 0 < self.maximum_one_sided_p_value < 1:
            raise ValueError("significance levels must be in (0, 1)")
        bonferroni_limit = self.familywise_alpha / len(self.horizons)
        if self.maximum_one_sided_p_value > bonferroni_limit + 1e-15:
            raise ValueError("maximum one-sided p-value must respect the Bonferroni horizon correction")
        if (
            self.minimum_observations_per_direction < 1
            or self.minimum_observations_per_direction_per_fold < 1
            or self.minimum_supported_stocks < 1
        ):
            raise ValueError("support requirements must be positive")
        if self.required_positive_folds < 1:
            raise ValueError("required_positive_folds must be positive")


@dataclass(frozen=True)
class SelectedSessionEvent:
    session: str
    symbol: str
    event_id: str
    direction: int
    direction_severity: str
    materiality: str
    novelty: str
    eligible_event_count: int


@dataclass(frozen=True)
class SelectionResult:
    selected_events: tuple[SelectedSessionEvent, ...]
    input_events: int
    eligible_events: int
    ineligible_events: int
    conflicting_tie_stock_sessions: int


@dataclass(frozen=True)
class HorizonObservation:
    horizon_sessions: int
    start_session: str
    end_session: str
    symbol: str
    direction: int
    raw_return: float
    market_return: float
    market_adjusted_return: float
    direction_severity: str = "unknown"
    materiality: str = "unknown"
    novelty: str = "unknown"


@dataclass(frozen=True)
class HorizonSpreadSummary:
    horizon_sessions: int
    positive_observations: int
    negative_observations: int
    supported_stocks: int
    positive_mean_market_adjusted_return: float | None
    negative_mean_market_adjusted_return: float | None
    spread: float | None
    positive_median_market_adjusted_return: float | None = None
    negative_median_market_adjusted_return: float | None = None


@dataclass(frozen=True)
class FoldSpreadSummary:
    horizon_sessions: int
    fold_index: int
    positive_observations: int
    negative_observations: int
    positive_mean_market_adjusted_return: float | None
    negative_mean_market_adjusted_return: float | None
    spread: float | None


@dataclass(frozen=True)
class BootstrapSpread:
    horizon_sessions: int
    observed_spread: float
    ci_low: float
    ci_high: float
    one_sided_p_value: float
    block_length: int
    replications: int
    seed: int
    valid_replications: int | None = None


@dataclass(frozen=True)
class SubgroupSpreadSummary:
    horizon_sessions: int
    dimension: str
    subgroup: str
    positive_observations: int
    negative_observations: int
    supported_stocks: int
    spread: float | None


@dataclass(frozen=True)
class HorizonAcceptance:
    horizon_sessions: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    positive_observations: int
    negative_observations: int
    supported_stocks: int
    positive_folds: int
    required_positive_folds: int
    one_sided_p_value: float | None
    confidence_interval_low: float | None


@dataclass(frozen=True)
class DirectionalEventGateRun:
    experiment_id: str
    output_dir: Path
    report_path: Path
    manifest_path: Path
    passed: bool
    selected_horizon: int | None
    acceptance: tuple[HorizonAcceptance, ...]
    reused: bool


def _direction(value: float) -> int:
    if value not in {-1.0, 1.0}:
        raise DirectionalEventGateError("eligible event score must be exactly -1 or +1")
    return -1 if value < 0 else 1


def select_strongest_session_events(
    events: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    development_sessions: Sequence[str],
) -> SelectionResult:
    """Select one strongest eligible event per stock-session deterministically.

    Strength is a lexicographic tuple of materiality, novelty, and absolute
    direction severity.  If equally strongest events disagree in direction,
    the entire stock-session is excluded rather than arbitrarily resolved.
    """

    allowed = set(development_sessions)
    if len(allowed) != len(development_sessions):
        raise DirectionalEventGateError("development sessions must be unique")
    event_by_id: dict[str, Mapping[str, Any]] = {}
    for event in events:
        session = str(event.get("eligible_execution_session") or "")
        if session not in allowed:
            continue
        event_id = str(event.get("event_id") or "")
        symbol = str(event.get("symbol") or "").upper()
        if not event_id or not symbol:
            raise DirectionalEventGateError("development event has no event_id or symbol")
        if event_id in event_by_id:
            raise DirectionalEventGateError(f"duplicate development event_id: {event_id}")
        event_by_id[event_id] = event

    score_by_id: dict[str, Mapping[str, Any]] = {}
    eligible_count = 0
    ineligible_count = 0
    for score in scores:
        event_id = str(score.get("event_id") or "")
        if not event_id or event_id in score_by_id:
            raise DirectionalEventGateError(f"missing or duplicate score event_id: {event_id}")
        if score.get("status") != "success":
            raise DirectionalEventGateError(f"model-only score is not successful: {event_id}")
        eligible = score.get("eligible")
        if not isinstance(eligible, bool):
            raise DirectionalEventGateError(f"model-only score has no boolean eligibility flag: {event_id}")
        value = float(score.get("score", math.nan))
        if eligible:
            _direction(value)
            eligible_count += 1
        elif value != 0.0:
            raise DirectionalEventGateError(f"ineligible score is not an explicit zero: {event_id}")
        else:
            ineligible_count += 1
        score_by_id[event_id] = score

    if set(score_by_id) != set(event_by_id):
        missing = sorted(set(event_by_id) - set(score_by_id))
        extra = sorted(set(score_by_id) - set(event_by_id))
        raise DirectionalEventGateError(f"event/score identities differ (missing={missing[:3]}, extra={extra[:3]})")
    if not event_by_id:
        raise DirectionalEventGateError("development event universe is empty")

    grouped: dict[tuple[str, str], list[tuple[tuple[int, int, int], Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for event_id, score in score_by_id.items():
        if not score["eligible"]:
            continue
        event = event_by_id[event_id]
        materiality = str(score.get("model_materiality") or "")
        novelty = str(score.get("model_novelty") or "")
        severity = str(score.get("model_direction_severity") or "")
        if materiality not in LEVEL_ORDER or novelty not in LEVEL_ORDER or severity not in DIRECTION_STRENGTH:
            raise DirectionalEventGateError(f"eligible score has invalid joint labels: {event_id}")
        rank = (LEVEL_ORDER[materiality], LEVEL_ORDER[novelty], DIRECTION_STRENGTH[severity])
        key = (str(event["eligible_execution_session"]), str(event["symbol"]).upper())
        grouped[key].append((rank, event, score))

    selected: list[SelectedSessionEvent] = []
    conflicts = 0
    for (session, symbol), candidates in sorted(grouped.items()):
        strongest_rank = max(row[0] for row in candidates)
        strongest = [row for row in candidates if row[0] == strongest_rank]
        directions = {_direction(float(row[2]["score"])) for row in strongest}
        if len(directions) != 1:
            conflicts += 1
            continue
        strongest.sort(
            key=lambda row: (
                str(row[1].get("available_at_utc") or ""),
                str(row[1].get("event_id") or ""),
            )
        )
        _, event, score = strongest[0]
        selected.append(
            SelectedSessionEvent(
                session=session,
                symbol=symbol,
                event_id=str(event["event_id"]),
                direction=next(iter(directions)),
                direction_severity=str(score["model_direction_severity"]),
                materiality=str(score["model_materiality"]),
                novelty=str(score["model_novelty"]),
                eligible_event_count=len(candidates),
            )
        )
    if not selected:
        raise DirectionalEventGateError("strongest-event selection produced no usable stock-sessions")
    return SelectionResult(
        selected_events=tuple(selected),
        input_events=len(event_by_id),
        eligible_events=eligible_count,
        ineligible_events=ineligible_count,
        conflicting_tie_stock_sessions=conflicts,
    )


def calculate_horizon_observations(
    selected: Sequence[SelectedSessionEvent],
    prices: Mapping[tuple[str, str], float],
    sessions: Sequence[str],
    symbols: Sequence[str],
    evaluation_start: str,
    horizons: Sequence[int],
) -> tuple[HorizonObservation, ...]:
    """Calculate leakage-safe stock and equal-weight-market forward returns."""

    ordered_sessions = tuple(sessions)
    if tuple(sorted(set(ordered_sessions))) != ordered_sessions:
        raise DirectionalEventGateError("price sessions must be unique and chronological")
    ordered_symbols = tuple(sorted(set(symbols)))
    if not ordered_symbols:
        raise DirectionalEventGateError("price universe is empty")
    index = {session: position for position, session in enumerate(ordered_sessions)}
    benchmark_cache: dict[tuple[str, str], float] = {}
    observations: list[HorizonObservation] = []
    for event in selected:
        if event.session not in index:
            raise DirectionalEventGateError(f"event session is absent from the price panel: {event.session}")
        start_index = index[event.session]
        for horizon in horizons:
            end_index = start_index + int(horizon)
            if end_index >= len(ordered_sessions):
                continue
            end_session = ordered_sessions[end_index]
            if end_session >= evaluation_start:
                continue
            benchmark_key = (event.session, end_session)
            if benchmark_key not in benchmark_cache:
                market_values: list[float] = []
                for symbol in ordered_symbols:
                    try:
                        start_price = float(prices[(symbol, event.session)])
                        end_price = float(prices[(symbol, end_session)])
                    except KeyError as exc:
                        raise DirectionalEventGateError(
                            f"price panel is incomplete for {symbol}: {event.session} -> {end_session}"
                        ) from exc
                    market_values.append(end_price / start_price - 1)
                benchmark_cache[benchmark_key] = fmean(market_values)
            try:
                raw_return = float(prices[(event.symbol, end_session)]) / float(prices[(event.symbol, event.session)]) - 1
            except KeyError as exc:
                raise DirectionalEventGateError(
                    f"selected event has incomplete prices: {event.symbol} {event.session} -> {end_session}"
                ) from exc
            market_return = benchmark_cache[benchmark_key]
            observations.append(
                HorizonObservation(
                    horizon_sessions=int(horizon),
                    start_session=event.session,
                    end_session=end_session,
                    symbol=event.symbol,
                    direction=event.direction,
                    raw_return=raw_return,
                    market_return=market_return,
                    market_adjusted_return=raw_return - market_return,
                    direction_severity=event.direction_severity,
                    materiality=event.materiality,
                    novelty=event.novelty,
                )
            )
    observations.sort(key=lambda row: (row.horizon_sessions, row.start_session, row.symbol))
    return tuple(observations)


def _spread_summary(rows: Sequence[HorizonObservation], horizon: int) -> HorizonSpreadSummary:
    positive = [row.market_adjusted_return for row in rows if row.direction > 0]
    negative = [row.market_adjusted_return for row in rows if row.direction < 0]
    positive_mean = fmean(positive) if positive else None
    negative_mean = fmean(negative) if negative else None
    spread = positive_mean - negative_mean if positive_mean is not None and negative_mean is not None else None
    return HorizonSpreadSummary(
        horizon_sessions=horizon,
        positive_observations=len(positive),
        negative_observations=len(negative),
        supported_stocks=len({row.symbol for row in rows}),
        positive_mean_market_adjusted_return=positive_mean,
        negative_mean_market_adjusted_return=negative_mean,
        spread=spread,
        positive_median_market_adjusted_return=median(positive) if positive else None,
        negative_median_market_adjusted_return=median(negative) if negative else None,
    )


def summarize_horizon_spreads(
    observations: Sequence[HorizonObservation],
    horizons: Sequence[int],
    *,
    allowed_sessions: Sequence[str] | None = None,
) -> tuple[HorizonSpreadSummary, ...]:
    """Summarize positive-minus-negative separation for each horizon."""

    allowed = set(allowed_sessions) if allowed_sessions is not None else None
    return tuple(
        _spread_summary(
            [row for row in observations if row.horizon_sessions == horizon and (allowed is None or row.start_session in allowed)],
            int(horizon),
        )
        for horizon in horizons
    )


def summarize_validation_folds(
    observations: Sequence[HorizonObservation],
    folds: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
) -> tuple[FoldSpreadSummary, ...]:
    """Summarize direction separation independently in frozen validation folds."""

    seen_sessions: set[str] = set()
    output: list[FoldSpreadSummary] = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        validation = tuple(str(value) for value in fold["validation_sessions"])
        overlap = seen_sessions.intersection(validation)
        if overlap:
            raise DirectionalEventGateError(f"validation folds overlap: {sorted(overlap)[:3]}")
        seen_sessions.update(validation)
        allowed = set(validation)
        for horizon in horizons:
            rows = [
                row
                for row in observations
                if row.horizon_sessions == horizon and row.start_session in allowed and row.end_session in allowed
            ]
            summary = _spread_summary(rows, int(horizon))
            output.append(
                FoldSpreadSummary(
                    horizon_sessions=int(horizon),
                    fold_index=fold_index,
                    positive_observations=summary.positive_observations,
                    negative_observations=summary.negative_observations,
                    positive_mean_market_adjusted_return=summary.positive_mean_market_adjusted_return,
                    negative_mean_market_adjusted_return=summary.negative_mean_market_adjusted_return,
                    spread=summary.spread,
                )
            )
    output.sort(key=lambda row: (row.horizon_sessions, row.fold_index))
    return tuple(output)


def session_block_bootstrap_spread(
    observations: Sequence[HorizonObservation],
    *,
    horizon_sessions: int | None = None,
    block_length: int,
    replications: int,
    seed: int,
    alpha: float | None = None,
    confidence_level: float | None = None,
    session_order: Sequence[str] | None = None,
) -> BootstrapSpread:
    """Bootstrap the mean direction spread in contiguous session clusters.

    Every sampled session carries all of its stock observations.  When a full
    validation-session calendar is supplied, no-news sessions remain in the
    block structure instead of being silently compressed away.
    """

    if not observations:
        raise DirectionalEventGateError("cannot bootstrap an empty horizon")
    horizons = {row.horizon_sessions for row in observations}
    if len(horizons) != 1:
        raise DirectionalEventGateError("bootstrap input must contain exactly one horizon")
    resolved_horizon = next(iter(horizons))
    if horizon_sessions is not None and horizon_sessions != resolved_horizon:
        raise DirectionalEventGateError("declared bootstrap horizon does not match observations")
    if alpha is not None and confidence_level is not None:
        raise ValueError("provide alpha or confidence_level, not both")
    if confidence_level is not None:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        tail_probability = (1 - confidence_level) / 2
    else:
        tail_probability = 0.01 if alpha is None else alpha
    if block_length < 1 or replications < 1 or not 0 < tail_probability < 1:
        raise ValueError("invalid bootstrap configuration")
    sessions = tuple(session_order) if session_order is not None else tuple(sorted({row.start_session for row in observations}))
    if not sessions or len(set(sessions)) != len(sessions):
        raise DirectionalEventGateError("bootstrap session order must be non-empty and unique")
    if tuple(sorted(sessions)) != sessions:
        raise DirectionalEventGateError("bootstrap session order must be chronological")
    if any(row.start_session not in set(sessions) for row in observations):
        raise DirectionalEventGateError("bootstrap observations fall outside the supplied session order")
    by_session: dict[str, list[HorizonObservation]] = defaultdict(list)
    for row in observations:
        by_session[row.start_session].append(row)
    observed = _spread_summary(observations, next(iter(horizons))).spread
    if observed is None:
        raise DirectionalEventGateError("bootstrap horizon has no observations on one direction")

    generator = np.random.default_rng(seed)
    block_count = math.ceil(len(sessions) / block_length)
    samples: list[float] = []
    for _ in range(replications):
        sampled_sessions = list(_draw_contiguous_session_blocks(sessions, block_length, block_count, generator))
        sampled_rows: list[HorizonObservation] = []
        for session in sampled_sessions[: len(sessions)]:
            sampled_rows.extend(by_session.get(session, ()))
        value = _spread_summary(sampled_rows, next(iter(horizons))).spread
        if value is not None:
            samples.append(value)
    if not samples:
        raise DirectionalEventGateError("every bootstrap draw omitted one direction")
    array = np.asarray(samples, dtype=float)
    return BootstrapSpread(
        horizon_sessions=next(iter(horizons)),
        observed_spread=observed,
        ci_low=float(np.quantile(array, tail_probability)),
        ci_high=float(np.quantile(array, 1 - tail_probability)),
        one_sided_p_value=(1 + int(np.sum(array <= 0))) / (len(samples) + 1),
        block_length=block_length,
        replications=replications,
        seed=seed,
        valid_replications=len(samples),
    )


def _draw_contiguous_session_blocks(
    sessions: Sequence[str],
    block_length: int,
    block_count: int,
    generator: np.random.Generator,
) -> tuple[str, ...]:
    """Draw ordinary moving blocks without wrapping the sample boundary."""

    if block_length < 1 or block_length > len(sessions) or block_count < 1:
        raise ValueError("block dimensions are incompatible with the session sample")
    maximum_start = len(sessions) - block_length
    sampled: list[str] = []
    for _ in range(block_count):
        start = int(generator.integers(0, maximum_start + 1))
        sampled.extend(sessions[start : start + block_length])
    return tuple(sampled)


def _optional_session_block_bootstrap_spread(
    observations: Sequence[HorizonObservation],
    *,
    block_length: int,
    replications: int,
    seed: int,
    alpha: float,
    session_order: Sequence[str],
) -> BootstrapSpread | None:
    """Return no estimate when sparse resamples cannot retain both directions."""

    if {row.direction for row in observations} != {-1, 1}:
        return None
    try:
        return session_block_bootstrap_spread(
            observations,
            block_length=block_length,
            replications=replications,
            seed=seed,
            alpha=alpha,
            session_order=session_order,
        )
    except DirectionalEventGateError as exc:
        if str(exc) == "every bootstrap draw omitted one direction":
            return None
        raise


def summarize_subgroups(
    observations: Sequence[HorizonObservation],
    horizons: Sequence[int],
) -> tuple[SubgroupSpreadSummary, ...]:
    """Produce diagnostics by materiality, novelty, and direction strength."""

    output: list[SubgroupSpreadSummary] = []
    dimensions = {
        "materiality": lambda row: row.materiality,
        "novelty": lambda row: row.novelty,
        "direction_strength": lambda row: "very" if row.direction_severity.startswith("very_") else "regular",
    }
    for horizon in horizons:
        horizon_rows = [row for row in observations if row.horizon_sessions == horizon]
        for dimension, getter in dimensions.items():
            for subgroup in sorted({getter(row) for row in horizon_rows}):
                rows = [row for row in horizon_rows if getter(row) == subgroup]
                summary = _spread_summary(rows, int(horizon))
                output.append(
                    SubgroupSpreadSummary(
                        horizon_sessions=int(horizon),
                        dimension=dimension,
                        subgroup=subgroup,
                        positive_observations=summary.positive_observations,
                        negative_observations=summary.negative_observations,
                        supported_stocks=summary.supported_stocks,
                        spread=summary.spread,
                    )
                )
    return tuple(output)


def validation_fold_observations(
    observations: Sequence[HorizonObservation],
    folds: Sequence[Mapping[str, Any]],
) -> tuple[HorizonObservation, ...]:
    """Retain observations whose start and endpoint belong to one validation fold."""

    validation_sets = [set(str(value) for value in fold["validation_sessions"]) for fold in folds]
    seen: set[str] = set()
    for validation in validation_sets:
        if seen.intersection(validation):
            raise DirectionalEventGateError("validation folds overlap")
        seen.update(validation)
    return tuple(
        row
        for row in observations
        if any(row.start_session in validation and row.end_session in validation for validation in validation_sets)
    )


def evaluate_directional_gate(
    horizon_summaries: Sequence[HorizonSpreadSummary],
    fold_summaries: Sequence[FoldSpreadSummary],
    bootstrap_results: Sequence[BootstrapSpread],
    spec: DirectionalEventGateSpec,
) -> tuple[tuple[HorizonAcceptance, ...], int | None]:
    """Apply the predeclared, multiplicity-corrected directional evidence gate."""

    summary_by_horizon = {row.horizon_sessions: row for row in horizon_summaries}
    bootstrap_by_horizon = {row.horizon_sessions: row for row in bootstrap_results}
    acceptance: list[HorizonAcceptance] = []
    supplied_horizons = sorted(summary_by_horizon)
    unexpected = set(supplied_horizons) - set(spec.horizons)
    if unexpected:
        raise DirectionalEventGateError(f"unexpected horizon summaries: {sorted(unexpected)}")
    for horizon in supplied_horizons:
        summary = summary_by_horizon[horizon]
        fold_rows = [row for row in fold_summaries if row.horizon_sessions == horizon]
        bootstrap = bootstrap_by_horizon.get(horizon)
        positive_folds = sum(row.spread is not None and row.spread > 0 for row in fold_rows)
        reasons: list[str] = []
        if summary.positive_observations < spec.minimum_observations_per_direction:
            reasons.append("insufficient_positive_observations")
        if summary.negative_observations < spec.minimum_observations_per_direction:
            reasons.append("insufficient_negative_observations")
        if summary.supported_stocks < spec.minimum_supported_stocks:
            reasons.append("insufficient_stock_support")
        if len(fold_rows) != spec.required_positive_folds or positive_folds != spec.required_positive_folds:
            reasons.append("direction_spread_not_positive_in_all_required_folds")
        if any(
            row.positive_observations < spec.minimum_observations_per_direction_per_fold
            or row.negative_observations < spec.minimum_observations_per_direction_per_fold
            for row in fold_rows
        ):
            reasons.append("insufficient_direction_observations_in_one_or_more_folds")
        if bootstrap is None:
            reasons.append("missing_bootstrap_result")
        else:
            if bootstrap.one_sided_p_value > spec.maximum_one_sided_p_value:
                reasons.append("bonferroni_one_sided_p_value_failed")
            if bootstrap.ci_low <= 0:
                reasons.append("familywise_confidence_lower_bound_not_positive")
            valid_replications = bootstrap.replications if bootstrap.valid_replications is None else bootstrap.valid_replications
            if valid_replications / bootstrap.replications < 0.99:
                reasons.append("fewer_than_99_percent_valid_bootstrap_draws")
        acceptance.append(
            HorizonAcceptance(
                horizon_sessions=horizon,
                passed=not reasons,
                rejection_reasons=tuple(reasons),
                positive_observations=summary.positive_observations,
                negative_observations=summary.negative_observations,
                supported_stocks=summary.supported_stocks,
                positive_folds=positive_folds,
                required_positive_folds=spec.required_positive_folds,
                one_sided_p_value=bootstrap.one_sided_p_value if bootstrap is not None else None,
                confidence_interval_low=bootstrap.ci_low if bootstrap is not None else None,
            )
        )
    selected_horizon = next((row.horizon_sessions for row in acceptance if row.passed), None)
    return tuple(acceptance), selected_horizon


def _csv(rows: Sequence[Any]) -> str:
    if not rows:
        return ""
    payloads: list[dict[str, Any]] = []
    for row in rows:
        payload = asdict(row)
        for key, value in payload.items():
            if isinstance(value, tuple):
                payload[key] = "|".join(str(item) for item in value)
        payloads.append(payload)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(payloads[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(payloads)
    return buffer.getvalue()


def _load_acceptance_csv(path: Path) -> tuple[HorizonAcceptance, ...]:
    def optional_float(value: str | None) -> float | None:
        return None if value is None or value == "" else float(value)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("passed") not in {"True", "False"} for row in rows):
        raise DirectionalEventGateError("hashed directional-gate acceptance artifact is empty or invalid")
    try:
        acceptance = tuple(
            HorizonAcceptance(
                horizon_sessions=int(row["horizon_sessions"]),
                passed=row["passed"] == "True",
                rejection_reasons=tuple(value for value in row["rejection_reasons"].split("|") if value),
                positive_observations=int(row["positive_observations"]),
                negative_observations=int(row["negative_observations"]),
                supported_stocks=int(row["supported_stocks"]),
                positive_folds=int(row["positive_folds"]),
                required_positive_folds=int(row["required_positive_folds"]),
                one_sided_p_value=optional_float(row.get("one_sided_p_value")),
                confidence_interval_low=optional_float(row.get("confidence_interval_low")),
            )
            for row in rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectionalEventGateError("hashed directional-gate acceptance artifact is invalid") from exc
    return acceptance


def _spread_svg(summaries: Sequence[HorizonSpreadSummary]) -> str:
    width, height = 760, 380
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 45, 70
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    values = [row.spread for row in summaries if row.spread is not None]
    bound = max(max((abs(value) for value in values), default=0.0), 0.001)
    zero_y = margin_top + chart_height / 2
    bar_width = chart_width / max(len(summaries), 1) * 0.58
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="sans-serif" font-size="18">Validation direction spread by horizon</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{width - margin_right}" y2="{zero_y:.2f}" stroke="#444"/>',
    ]
    for index, row in enumerate(summaries):
        value = row.spread
        x = margin_left + (index + 0.5) * chart_width / len(summaries) - bar_width / 2
        scaled = 0.0 if value is None else value / bound * chart_height / 2
        y = zero_y - max(scaled, 0)
        bar_height = abs(scaled)
        color = "#999999" if value is None else "#2a7f62" if value > 0 else "#b64949"
        value_label = "n/a" if value is None else f"{value:.2%}"
        elements.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>',
                (
                    f'<text x="{x + bar_width / 2:.2f}" y="{height - margin_bottom + 24}" '
                    f'text-anchor="middle" font-family="sans-serif" font-size="13">'
                    f"{row.horizon_sessions}</text>"
                ),
                (
                    f'<text x="{x + bar_width / 2:.2f}" '
                    f'y="{y - 6 if value is None or value >= 0 else y + bar_height + 16:.2f}" '
                    f'text-anchor="middle" font-family="sans-serif" font-size="11">'
                    f"{html.escape(value_label)}</text>"
                ),
            ]
        )
    elements.extend(
        [
            (
                f'<text x="{margin_left + chart_width / 2:.2f}" y="{height - 15}" '
                'text-anchor="middle" font-family="sans-serif" font-size="13">Forward sessions</text>'
            ),
            (
                f'<text x="18" y="{margin_top + chart_height / 2:.2f}" '
                f'transform="rotate(-90 18 {margin_top + chart_height / 2:.2f})" '
                'text-anchor="middle" font-family="sans-serif" font-size="13">'
                "Positive minus negative adjusted return</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def _report(
    experiment_id: str,
    selection: SelectionResult,
    summaries: Sequence[HorizonSpreadSummary],
    folds: Sequence[FoldSpreadSummary],
    bootstraps: Sequence[BootstrapSpread],
    acceptance: Sequence[HorizonAcceptance],
    selected_horizon: int | None,
    spec: DirectionalEventGateSpec,
) -> str:
    bootstrap_by_horizon = {row.horizon_sessions: row for row in bootstraps}
    fold_by_horizon: dict[int, list[FoldSpreadSummary]] = defaultdict(list)
    for fold_row in folds:
        fold_by_horizon[fold_row.horizon_sessions].append(fold_row)
    decision = (
        f"PASS: the shortest qualifying horizon is {selected_horizon} sessions. A separately frozen sparse strategy may be specified next."
        if selected_horizon is not None
        else "STOP: no horizon met the directional evidence gate. Do not construct or evaluate another strategy from these labels."
    )
    lines = [
        f"# Directional event-study gate: {experiment_id}",
        "",
        "> Development-only diagnostic. The locked evaluation block is absent from every return, bootstrap, and decision.",
        "",
        "## Material Passport",
        "",
        f"- Experiment ID: `{experiment_id}`",
        "- Type: deterministic, validation-fold directional event study",
        "- Status: completed",
        "- Verification status: ANALYZED until deterministic replay and full tests pass",
        "- Data movement: aggregate tables and figure only; no licensed article text is emitted",
        "",
        "## Frozen design",
        "",
        "- Unit: one strongest eligible model event per stock-session.",
        "- Ranking: materiality, then novelty, then absolute direction severity; equally strong conflicting directions are excluded.",
        f"- Horizons: `{', '.join(map(str, spec.horizons))}` adjusted-open sessions.",
        "- Outcome: stock forward return minus the contemporaneous equal-weight panel return.",
        (
            f"- Inference: contiguous {spec.block_length}-session block bootstrap, "
            f"{spec.bootstrap_replications:,} replications, seed `{spec.seed}`."
        ),
        (f"- Multiplicity: {len(spec.horizons)} predeclared horizons; one-sided p-value must be `<= {spec.maximum_one_sided_p_value:g}`."),
        (
            f"- Support: at least {spec.minimum_observations_per_direction} observations per direction, "
            f"{spec.minimum_observations_per_direction_per_fold} per direction in every fold, "
            f"{spec.minimum_supported_stocks} stocks, and positive spread in all "
            f"{spec.required_positive_folds} validation folds."
        ),
        "- Horizon choice: shortest passing horizon, never the largest observed return.",
        "",
        "## Attrition",
        "",
        f"- Development events: `{selection.input_events:,}`.",
        f"- Eligible events: `{selection.eligible_events:,}`.",
        f"- Ineligible explicit-zero events: `{selection.ineligible_events:,}`.",
        f"- Selected stock-sessions: `{len(selection.selected_events):,}`.",
        f"- Excluded tied conflicting stock-sessions: `{selection.conflicting_tie_stock_sessions:,}`.",
        "",
        "## Observations",
        "",
        (
            "| Horizon | Positive N | Negative N | Stocks | Positive adjusted | Negative adjusted | "
            "Spread | Fold spreads | Bootstrap p | Lower bound | Pass |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    acceptance_by_horizon = {acceptance_row.horizon_sessions: acceptance_row for acceptance_row in acceptance}
    for summary in summaries:
        bootstrap = bootstrap_by_horizon.get(summary.horizon_sessions)
        check = acceptance_by_horizon[summary.horizon_sessions]
        fold_values = ", ".join(
            "n/a" if item.spread is None else f"{item.spread:.2%}"
            for item in sorted(fold_by_horizon[summary.horizon_sessions], key=lambda item: item.fold_index)
        )
        positive = "n/a" if summary.positive_mean_market_adjusted_return is None else f"{summary.positive_mean_market_adjusted_return:.2%}"
        negative = "n/a" if summary.negative_mean_market_adjusted_return is None else f"{summary.negative_mean_market_adjusted_return:.2%}"
        spread = "n/a" if summary.spread is None else f"{summary.spread:.2%}"
        bootstrap_p = "n/a" if bootstrap is None else f"{bootstrap.one_sided_p_value:.4f}"
        bootstrap_low = "n/a" if bootstrap is None else f"{bootstrap.ci_low:.2%}"
        lines.append(
            f"| {summary.horizon_sessions} | {summary.positive_observations} | "
            f"{summary.negative_observations} | {summary.supported_stocks} | {positive} | "
            f"{negative} | {spread} | {fold_values} | "
            f"{bootstrap_p} | {bootstrap_low} | {'yes' if check.passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "![Validation direction spreads](figures/direction_spread.svg)",
            "",
            "## Decision",
            "",
            f"**{decision}**",
            "",
        ]
    )
    failed = [row for row in acceptance if not row.passed]
    if failed:
        lines.extend(["### Failed checks", ""])
        for failed_check in failed:
            lines.append(f"- {failed_check.horizon_sessions} sessions: `{', '.join(failed_check.rejection_reasons)}`.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "The table reports observed development-validation associations, not a trading result or causal effect. "
            "The model labels are an explicitly accepted research assumption and were not human validated. "
            "This gate was specified after earlier development controls failed, so its evidence is post hoc and exploratory. "
            "A pass would justify specifying one sparse development strategy; it would not establish profitable or deployable alpha.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_events_path(root: Path, config: Mapping[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    configured = Path(str(config["outputs"]["derived_root"])) / root.name / "events" / "events.jsonl"
    fallback = root / "events" / "events.jsonl"
    return configured if configured.is_file() else fallback


def run_directional_event_gate(
    run_dir: str | Path,
    universe_dir: str | Path,
    strategy_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    price_panel: str | Path | None = None,
    events_path_override: str | Path | None = None,
    spec: DirectionalEventGateSpec | None = None,
) -> DirectionalEventGateRun:
    """Run and immutably materialize the development-only direction gate."""

    root = Path(run_dir)
    try:
        snapshot = load_completed_run_snapshot(root)
    except CompletedRunValidationError as exc:
        raise DirectionalEventGateError(str(exc)) from exc
    report = snapshot.report
    config = snapshot.config
    evaluation_start = str(config["run"]["evaluation_start"])
    events_path = _resolve_events_path(root, config, events_path_override)
    price_path = Path(price_panel) if price_panel is not None else Path(str(config["prices"]["panel_path"]))
    folds_path = root / "tuning" / "fold_definitions.json"
    strategy_root = Path(strategy_dir)
    universe_root = Path(universe_dir)
    scores_path = strategy_root / "scores.jsonl"
    strategy_manifest_path = strategy_root / "manifest.json"
    universe_manifest_path = universe_root / "sample_manifest.json"
    universe_items_path = universe_root / "audit_items.csv"
    events_manifest_path = root / "manifests" / "events.json"
    tuning_manifest_path = root / "manifests" / "tuning.json"
    price_manifest_path = Path(str(config["prices"].get("manifest_path") or ""))
    required = (
        events_path,
        events_manifest_path,
        price_path,
        price_manifest_path,
        folds_path,
        tuning_manifest_path,
        scores_path,
        strategy_manifest_path,
        universe_manifest_path,
        universe_items_path,
    )
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise DirectionalEventGateError(f"required gate artifacts are missing: {missing}")

    try:
        events_manifest_hash = validate_completed_stage_output(
            snapshot,
            stage="events",
            output_name="events",
            output_path=events_path,
            root_output_suffix=f"{root.name}/events/events.jsonl",
        )
        tuning_manifest_hash = validate_completed_stage_output(
            snapshot,
            stage="tuning",
            output_name="fold_definitions",
            output_path=folds_path,
            root_output_suffix=f"{root.name}/tuning/fold_definitions.json",
        )
        price_manifest_hash = validate_adjusted_open_price_panel(snapshot, price_path)
    except CompletedRunValidationError as exc:
        raise DirectionalEventGateError(str(exc)) from exc

    strategy_manifest = read_json(strategy_manifest_path)
    expected_score_hash = strategy_manifest.get("outputs", {}).get(scores_path.name, {}).get("sha256")
    if (
        strategy_manifest.get("schema_version") != 1
        or strategy_manifest.get("status") != "completed"
        or strategy_manifest.get("identity_sha256")
        != sha256_text(canonical_json(strategy_manifest.get("identity")))
        or sha256_file(scores_path) != expected_score_hash
    ):
        raise DirectionalEventGateError("filtered strategy manifest is incomplete or its score artifact changed")
    universe_manifest = read_json(universe_manifest_path)
    try:
        validate_model_only_scoring_universe(universe_root)
    except ModelOnlyDevelopmentError as exc:
        raise DirectionalEventGateError(str(exc)) from exc
    expected_items_hash = universe_manifest.get("outputs", {}).get(universe_items_path.name, {}).get("sha256")
    if (
        universe_manifest.get("schema_version") != 1
        or universe_manifest.get("status") != "completed"
        or sha256_file(universe_items_path) != expected_items_hash
    ):
        raise DirectionalEventGateError("model-only universe manifest is incomplete or its item artifact changed")
    if strategy_manifest.get("identity", {}).get("universe_identity_sha256") != universe_manifest.get("identity_sha256"):
        raise DirectionalEventGateError("filtered strategy does not belong to the supplied model-only universe")
    universe_identity = universe_manifest.get("identity", {})
    if (
        strategy_manifest.get("identity", {}).get("contract") != MODEL_ONLY_CONTRACT
        or universe_identity.get("contract") != MODEL_ONLY_CONTRACT
        or universe_identity.get("source_run_id") != root.name
        or universe_identity.get("source_run_identity") != report.get("run_identity_sha256")
        or universe_identity.get("evaluation_start") != evaluation_start
        or universe_identity.get("events_sha256") != sha256_file(events_path)
    ):
        raise DirectionalEventGateError("model-only artifacts do not belong to the supplied source run")
    gate_spec = spec or DirectionalEventGateSpec()
    identity = {
        "analysis_schema_version": 4,
        "contract": "strongest_eligible_stock_session_direction_gate_v4",
        "source_run_id": root.name,
        "source_run_identity": report.get("run_identity_sha256"),
        "evaluation_start": evaluation_start,
        "events_sha256": sha256_file(events_path),
        "events_manifest_sha256": events_manifest_hash,
        "filtered_scores_sha256": sha256_file(scores_path),
        "filtered_strategy_manifest_sha256": sha256_file(strategy_manifest_path),
        "model_only_universe_identity": universe_manifest.get("identity_sha256"),
        "model_only_universe_manifest_sha256": sha256_file(universe_manifest_path),
        "model_only_items_sha256": sha256_file(universe_items_path),
        "prices_sha256": sha256_file(price_path),
        "price_manifest_sha256": price_manifest_hash,
        "folds_sha256": sha256_file(folds_path),
        "tuning_manifest_sha256": tuning_manifest_hash,
        "spec": asdict(gate_spec),
        "selection": "lexicographic materiality, novelty, absolute direction severity; conflicting strongest tie excluded",
        "market_adjustment": "stock adjusted-open forward return minus equal-weight panel forward return",
        "inference": "validation starts only; ordinary non-wrapping contiguous-session moving-block bootstrap",
        "boundary_rule": "every return endpoint must be strictly before evaluation_start",
        "output_policy": "aggregate-only; no event identifiers or article text",
    }
    identity_hash = sha256_text(canonical_json(identity))
    experiment_id = f"directional-event-gate-{identity_hash[:12]}"
    output_dir = Path(output_root) if output_root is not None else root / "directional_event_gate" / experiment_id
    manifest_path = output_dir / "manifest.json"
    report_output_path = output_dir / "report.md"
    horizon_path = output_dir / "horizon_summary.csv"
    fold_output_path = output_dir / "fold_summary.csv"
    subgroup_path = output_dir / "subgroup_summary.csv"
    bootstrap_path = output_dir / "bootstrap.csv"
    acceptance_path = output_dir / "acceptance_checks.csv"
    figure_path = output_dir / "figures" / "direction_spread.svg"
    expected = (
        report_output_path,
        horizon_path,
        fold_output_path,
        subgroup_path,
        bootstrap_path,
        acceptance_path,
        figure_path,
    )
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("schema_version") != 4 or existing.get("status") != "completed":
            raise DirectionalEventGateError(f"refusing to reuse incomplete directional-gate output: {output_dir}")
        if (
            existing.get("identity_sha256") != identity_hash
            or canonical_json(existing.get("identity")) != canonical_json(identity)
        ):
            raise DirectionalEventGateError(f"existing directional-gate identity mismatch: {output_dir}")
        actual_files = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()}
        expected_files = {path.relative_to(output_dir).as_posix() for path in expected} | {"manifest.json"}
        if actual_files != expected_files:
            raise DirectionalEventGateError(f"completed directional-gate directory contains unexpected or missing files: {output_dir}")
        for path in expected:
            expected_hash = existing.get("outputs", {}).get(path.name, {}).get("sha256")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise DirectionalEventGateError(f"completed directional-gate output changed: {path}")
        acceptance = _load_acceptance_csv(acceptance_path)
        selected_horizon = next((row.horizon_sessions for row in acceptance if row.passed), None)
        derived_passed = selected_horizon is not None
        stored_passed = existing.get("passed")
        stored_selected = existing.get("selected_horizon")
        if (
            not isinstance(stored_passed, bool)
            or (
                stored_selected is not None
                and (not isinstance(stored_selected, int) or isinstance(stored_selected, bool))
            )
            or stored_passed != derived_passed
            or stored_selected != selected_horizon
            or canonical_json(existing.get("acceptance")) != canonical_json([asdict(row) for row in acceptance])
        ):
            raise DirectionalEventGateError("directional-gate manifest decision fields disagree with hashed outputs")
        return DirectionalEventGateRun(
            experiment_id,
            output_dir,
            report_output_path,
            manifest_path,
            derived_passed,
            selected_horizon,
            acceptance,
            True,
        )
    if output_dir.exists():
        raise DirectionalEventGateError(f"refusing to reuse incomplete directional-gate output: {output_dir}")

    opens = load_adjusted_opens_csv(price_path)
    symbols = tuple(sorted({row.symbol for row in opens}))
    sessions = tuple(sorted({row.session for row in opens}))
    development_sessions = tuple(session for session in sessions if session < evaluation_start)
    try:
        events = [event.to_payload() for event in load_strategy_events(events_path)]
    except ValueError as exc:
        raise DirectionalEventGateError(f"source strategy events are invalid: {exc}") from exc
    scores = read_jsonl(scores_path)
    selection = select_strongest_session_events(events, scores, development_sessions)
    price_lookup = {(row.symbol, row.session): row.adjusted_open for row in opens}
    observations = calculate_horizon_observations(
        selection.selected_events,
        price_lookup,
        sessions,
        symbols,
        evaluation_start,
        gate_spec.horizons,
    )
    folds_payload = read_json(folds_path)
    folds_raw = folds_payload.get("folds")
    if not isinstance(folds_raw, list) or len(folds_raw) != gate_spec.required_positive_folds:
        raise DirectionalEventGateError(f"direction gate requires exactly {gate_spec.required_positive_folds} frozen validation folds")
    folds: tuple[Mapping[str, Any], ...] = tuple(folds_raw)
    validation_sessions = tuple(str(session) for fold in folds for session in fold["validation_sessions"])
    validation_observations = validation_fold_observations(observations, folds)
    horizon_summaries = summarize_horizon_spreads(validation_observations, gate_spec.horizons)
    fold_summaries = summarize_validation_folds(observations, folds, gate_spec.horizons)
    subgroups = summarize_subgroups(validation_observations, gate_spec.horizons)
    bootstraps: list[BootstrapSpread] = []
    for horizon in gate_spec.horizons:
        horizon_rows = tuple(row for row in validation_observations if row.horizon_sessions == horizon)
        bootstrap = _optional_session_block_bootstrap_spread(
            horizon_rows,
            block_length=gate_spec.block_length,
            replications=gate_spec.bootstrap_replications,
            seed=gate_spec.seed,
            alpha=gate_spec.maximum_one_sided_p_value,
            session_order=validation_sessions,
        )
        if bootstrap is not None:
            bootstraps.append(bootstrap)
    acceptance, selected_horizon = evaluate_directional_gate(
        horizon_summaries,
        fold_summaries,
        bootstraps,
        gate_spec,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(horizon_path, _csv(horizon_summaries))
    atomic_write_text(fold_output_path, _csv(fold_summaries))
    atomic_write_text(subgroup_path, _csv(subgroups))
    atomic_write_text(bootstrap_path, _csv(bootstraps))
    atomic_write_text(acceptance_path, _csv(acceptance))
    atomic_write_text(figure_path, _spread_svg(horizon_summaries))
    atomic_write_text(
        report_output_path,
        _report(
            experiment_id,
            selection,
            horizon_summaries,
            fold_summaries,
            bootstraps,
            acceptance,
            selected_horizon,
            gate_spec,
        ),
    )
    manifest = {
        "schema_version": 4,
        "status": "completed",
        "experiment_id": experiment_id,
        "identity_sha256": identity_hash,
        "identity": identity,
        "passed": selected_horizon is not None,
        "selected_horizon": selected_horizon,
        "selection_attrition": {
            "input_events": selection.input_events,
            "eligible_events": selection.eligible_events,
            "ineligible_events": selection.ineligible_events,
            "selected_stock_sessions": len(selection.selected_events),
            "conflicting_tie_stock_sessions": selection.conflicting_tie_stock_sessions,
        },
        "acceptance": [asdict(row) for row in acceptance],
        "row_counts": {
            "validation_horizon_summaries": len(horizon_summaries),
            "fold_summaries": len(fold_summaries),
            "subgroup_summaries": len(subgroups),
            "bootstrap_results": len(bootstraps),
        },
        "outputs": {
            path.name: {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in expected
        },
        "warning": (
            "Development-only association gate; evaluation rows were loaded only as panel metadata, excluded from "
            "every return calculation, and cannot support an alpha claim."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return DirectionalEventGateRun(
        experiment_id,
        output_dir,
        report_output_path,
        manifest_path,
        selected_horizon is not None,
        selected_horizon,
        acceptance,
        False,
    )
