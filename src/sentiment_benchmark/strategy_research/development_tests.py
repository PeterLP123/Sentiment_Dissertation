"""Development-only diagnostics for a sparse stock-session news strategy.

The formal evaluation block is never read as an outcome. Candidate ranking uses
the existing expanding development folds and excludes any return interval whose
endpoint reaches the locked evaluation boundary.
"""

from __future__ import annotations

import csv
import html
import io
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean, median
from typing import Any, Literal

import numpy as np
from scipy.stats import spearmanr

from ..artifact_io import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from .ledger import evaluation_rows, run_open_to_open_ledger
from .market import OpenToOpenReturn, calculate_open_to_open_returns, load_adjusted_opens_csv
from .portfolio import PositionTarget, TargetPortfolio

SignalKind = Literal["raw", "centered"]
AggregationKind = Literal["repeated_event", "stock_session_mean"]


class DevelopmentTestError(RuntimeError):
    """Raised when development diagnostics would violate their frozen contract."""


@dataclass(frozen=True)
class DevelopmentTestSpec:
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    thresholds: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    signal_kinds: tuple[SignalKind, ...] = ("raw", "centered")
    shrinkage_k: float = 50.0
    cost_bps_per_side: float = 10.0
    minimum_supported_stocks: int = 20
    minimum_active_fraction: float = 0.05
    stability_penalty: float = 0.5

    def __post_init__(self) -> None:
        if not self.horizons or any(value < 1 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if not self.thresholds or any(not 0 <= value <= 1 for value in self.thresholds):
            raise ValueError("thresholds must be in [0, 1]")
        if self.signal_kinds != ("raw", "centered"):
            raise ValueError("v1 development tests require raw and centered signals")
        if self.shrinkage_k < 0 or self.cost_bps_per_side < 0:
            raise ValueError("shrinkage and cost parameters must be non-negative")
        if self.minimum_supported_stocks < 1 or not 0 <= self.minimum_active_fraction <= 1:
            raise ValueError("candidate guardrails are invalid")


@dataclass(frozen=True)
class SessionSignal:
    session: str
    symbol: str
    event_count: int
    mean_score: float
    positive_events: int
    neutral_events: int
    negative_events: int


@dataclass(frozen=True)
class LabelHorizonResult:
    aggregation: AggregationKind
    horizon_sessions: int
    label: str
    observations: int
    unique_stock_sessions: int
    mean_raw_return: float
    median_raw_return: float
    raw_positive_rate: float
    mean_market_adjusted_return: float
    median_market_adjusted_return: float
    adjusted_positive_rate: float


@dataclass(frozen=True)
class InformationCoefficientResult:
    aggregation: AggregationKind
    horizon_sessions: int
    return_kind: str
    observations: int
    spearman_rho: float | None
    p_value: float | None
    fdr_q_value: float | None


@dataclass(frozen=True)
class FoldCandidateResult:
    signal_kind: SignalKind
    hold_sessions: int
    threshold: float
    fold_index: int
    validation_start: str
    validation_end: str
    eligible_intervals: int
    supported_stocks: int
    active_stock_days: int
    active_fraction: float
    mean_stock_net_return: float
    median_stock_net_return: float
    positive_stock_rate: float
    mean_stock_market_adjusted_net_return: float
    median_stock_market_adjusted_net_return: float
    average_stock_turnover: float
    long_gross_return_sum: float
    short_gross_return_sum: float
    long_observations: int
    short_observations: int


@dataclass(frozen=True)
class CandidateSummary:
    signal_kind: SignalKind
    hold_sessions: int
    threshold: float
    valid: bool
    rejection_reasons: tuple[str, ...]
    objective: float | None
    median_fold_market_adjusted_return: float
    fold_return_iqr: float
    median_fold_raw_return: float
    mean_active_fraction: float
    mean_supported_stocks: float
    mean_turnover: float
    positive_folds: int


@dataclass(frozen=True)
class DevelopmentTestRun:
    experiment_id: str
    output_dir: Path
    report_path: Path
    candidate_path: Path
    label_path: Path
    manifest_path: Path
    selected_candidate: CandidateSummary | None
    reused: bool


def aggregate_stock_session_scores(
    events: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    allowed_sessions: Sequence[str],
) -> tuple[SessionSignal, ...]:
    """Join immutable score records and average all events per stock-session."""

    score_by_event: dict[str, float] = {}
    for row in scores:
        if row.get("status") != "success":
            continue
        event_id = str(row.get("event_id") or "")
        if event_id in score_by_event:
            raise DevelopmentTestError(f"duplicate score event_id: {event_id}")
        score_by_event[event_id] = float(row["score"])
    allowed = set(allowed_sessions)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in events:
        session = str(event.get("eligible_execution_session") or "")
        if session not in allowed:
            continue
        event_id = str(event.get("event_id") or "")
        symbol = str(event.get("symbol") or "").upper()
        if event_id not in score_by_event:
            continue
        grouped[(session, symbol)].append(score_by_event[event_id])
    return tuple(
        SessionSignal(
            session=session,
            symbol=symbol,
            event_count=len(values),
            mean_score=fmean(values),
            positive_events=sum(value > 0 for value in values),
            neutral_events=sum(value == 0 for value in values),
            negative_events=sum(value < 0 for value in values),
        )
        for (session, symbol), values in sorted(grouped.items())
    )


def calculate_label_horizon_results(
    events: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    session_signals: Sequence[SessionSignal],
    prices: Mapping[tuple[str, str], float],
    sessions: Sequence[str],
    symbols: Sequence[str],
    evaluation_start: str,
    horizons: Sequence[int],
) -> tuple[tuple[LabelHorizonResult, ...], tuple[InformationCoefficientResult, ...]]:
    """Describe label-conditioned forward returns without using evaluation prices."""

    session_index = {session: index for index, session in enumerate(sessions)}
    score_by_event = {str(row["event_id"]): float(row["score"]) for row in scores if row.get("status") == "success"}
    repeated_rows = [
        (str(event["eligible_execution_session"]), str(event["symbol"]), score_by_event[str(event["event_id"])])
        for event in events
        if str(event.get("eligible_execution_session") or "") in session_index
        and str(event["eligible_execution_session"]) < evaluation_start
        and str(event.get("event_id") or "") in score_by_event
    ]
    session_rows = [(row.session, row.symbol, row.mean_score) for row in session_signals]
    all_results: list[LabelHorizonResult] = []
    all_ics: list[InformationCoefficientResult] = []
    for aggregation, rows in (("repeated_event", repeated_rows), ("stock_session_mean", session_rows)):
        for horizon in horizons:
            observations: list[tuple[str, str, float, float, float]] = []
            benchmark_cache: dict[tuple[str, str], float] = {}
            for session, symbol, score in rows:
                start = session_index[session]
                end = start + horizon
                if end >= len(sessions) or sessions[end] >= evaluation_start:
                    continue
                end_session = sessions[end]
                key = (session, end_session)
                if key not in benchmark_cache:
                    universe_returns = [prices[(candidate, end_session)] / prices[(candidate, session)] - 1 for candidate in symbols]
                    benchmark_cache[key] = fmean(universe_returns)
                raw_return = prices[(symbol, end_session)] / prices[(symbol, session)] - 1
                adjusted = raw_return - benchmark_cache[key]
                observations.append((session, symbol, score, raw_return, adjusted))
            for label, predicate in (
                ("negative", lambda value: value < 0),
                ("neutral", lambda value: value == 0),
                ("positive", lambda value: value > 0),
            ):
                selected = [row for row in observations if predicate(row[2])]
                if not selected:
                    continue
                raw_values = [row[3] for row in selected]
                adjusted_values = [row[4] for row in selected]
                all_results.append(
                    LabelHorizonResult(
                        aggregation=aggregation,  # type: ignore[arg-type]
                        horizon_sessions=horizon,
                        label=label,
                        observations=len(selected),
                        unique_stock_sessions=len({(row[0], row[1]) for row in selected}),
                        mean_raw_return=fmean(raw_values),
                        median_raw_return=median(raw_values),
                        raw_positive_rate=sum(value > 0 for value in raw_values) / len(raw_values),
                        mean_market_adjusted_return=fmean(adjusted_values),
                        median_market_adjusted_return=median(adjusted_values),
                        adjusted_positive_rate=sum(value > 0 for value in adjusted_values) / len(adjusted_values),
                    )
                )
            for return_kind, position in (("raw", 3), ("market_adjusted", 4)):
                if len(observations) < 3 or len({row[2] for row in observations}) < 2:
                    rho = p_value = None
                else:
                    result = spearmanr([row[2] for row in observations], [row[position] for row in observations])
                    rho = float(result.statistic) if math.isfinite(float(result.statistic)) else None
                    p_value = float(result.pvalue) if math.isfinite(float(result.pvalue)) else None
                all_ics.append(
                    InformationCoefficientResult(
                        aggregation=aggregation,  # type: ignore[arg-type]
                        horizon_sessions=horizon,
                        return_kind=return_kind,
                        observations=len(observations),
                        spearman_rho=rho,
                        p_value=p_value,
                        fdr_q_value=None,
                    )
                )
    finite = sorted(
        ((index, row.p_value) for index, row in enumerate(all_ics) if row.p_value is not None),
        key=lambda item: item[1],
    )
    q_values: dict[int, float] = {}
    running = 1.0
    test_count = len(finite)
    for reverse_rank, (index, p_value) in enumerate(reversed(finite), start=1):
        rank = test_count - reverse_rank + 1
        running = min(running, float(p_value) * test_count / rank)
        q_values[index] = min(1.0, running)
    adjusted_ics = tuple(replace(row, fdr_q_value=q_values.get(index)) for index, row in enumerate(all_ics))
    return tuple(all_results), adjusted_ics


def _training_centers(
    session_signals: Sequence[SessionSignal],
    training_sessions: Sequence[str],
    symbols: Sequence[str],
    shrinkage_k: float,
) -> dict[str, float]:
    training = set(training_sessions)
    pooled_values = [row.mean_score for row in session_signals if row.session in training]
    pooled = fmean(pooled_values) if pooled_values else 0.0
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for row in session_signals:
        if row.session in training:
            by_symbol[row.symbol].append(row.mean_score)
    centers = {}
    for symbol in symbols:
        values = by_symbol[symbol]
        weight = len(values) / (len(values) + shrinkage_k) if values else 0.0
        centers[symbol] = weight * fmean(values) + (1 - weight) * pooled if values else pooled
    return centers


def _target(session: str, symbol: str, exposure: float) -> TargetPortfolio:
    position = PositionTarget(symbol, exposure, None, exposure, exposure)
    return TargetPortfolio(
        session=session,
        positions=(position,),
        gross_exposure=abs(exposure),
        net_exposure=exposure,
        long_exposure=max(exposure, 0.0),
        short_exposure=max(-exposure, 0.0),
        cash_weight=1 - exposure,
    )


def _candidate_targets(
    sessions: Sequence[str],
    symbols: Sequence[str],
    session_signals: Sequence[SessionSignal],
    centers: Mapping[str, float],
    *,
    signal_kind: SignalKind,
    hold_sessions: int,
    threshold: float,
) -> dict[str, list[TargetPortfolio]]:
    ordinal = {session: index for index, session in enumerate(sessions)}
    signal_lookup = {(row.session, row.symbol): row.mean_score for row in session_signals}
    targets: dict[str, list[TargetPortfolio]] = defaultdict(list)
    for symbol in symbols:
        active_direction = 0.0
        signal_ordinal: int | None = None
        for session in sessions:
            value = signal_lookup.get((session, symbol))
            if value is not None:
                candidate = value if signal_kind == "raw" else value - centers[symbol]
                if abs(candidate) >= threshold and candidate != 0:
                    active_direction = math.copysign(1.0, candidate)
                    signal_ordinal = ordinal[session]
            if signal_ordinal is None or ordinal[session] - signal_ordinal >= hold_sessions:
                active_direction = 0.0
            targets[symbol].append(_target(session, symbol, active_direction))
    return targets


def evaluate_candidate_folds(
    session_signals: Sequence[SessionSignal],
    returns: Sequence[OpenToOpenReturn],
    sessions: Sequence[str],
    symbols: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
    evaluation_start: str,
    spec: DevelopmentTestSpec,
) -> tuple[tuple[FoldCandidateResult, ...], tuple[CandidateSummary, ...]]:
    """Evaluate every candidate on expanding development-validation folds."""

    return_by_key = {(row.session, row.symbol): row for row in returns}
    valid_returns = [row for row in returns if row.next_session < evaluation_start]
    market_by_session: dict[str, float] = {}
    for session in sessions:
        values = [return_by_key[(session, symbol)].value for symbol in symbols if (session, symbol) in return_by_key]
        if len(values) == len(symbols):
            market_by_session[session] = fmean(values)
    fold_rows: list[FoldCandidateResult] = []
    for signal_kind in spec.signal_kinds:
        for hold_sessions in spec.horizons:
            for threshold in spec.thresholds:
                for fold in folds:
                    training = tuple(str(value) for value in fold["training_sessions"])
                    validation = tuple(str(value) for value in fold["validation_sessions"])
                    validation_intervals = tuple(
                        session
                        for session in validation
                        if session in market_by_session and return_by_key[(session, symbols[0])].next_session < evaluation_start
                    )
                    if not validation_intervals:
                        raise DevelopmentTestError("a development fold has no non-leaking return intervals")
                    combined = tuple(session for session in sessions if session <= validation_intervals[-1])
                    centers = _training_centers(session_signals, training, symbols, spec.shrinkage_k)
                    targets = _candidate_targets(
                        combined,
                        symbols,
                        session_signals,
                        centers,
                        signal_kind=signal_kind,
                        hold_sessions=hold_sessions,
                        threshold=threshold,
                    )
                    stock_net: list[float] = []
                    stock_adjusted: list[float] = []
                    stock_turnover: list[float] = []
                    supported = active_days = long_obs = short_obs = 0
                    long_sum = short_sum = 0.0
                    for symbol in symbols:
                        symbol_returns = [row for row in valid_returns if row.symbol == symbol]
                        ledger = run_open_to_open_ledger(
                            targets[symbol],
                            symbol_returns,
                            cost_rate_per_side=spec.cost_bps_per_side / 10_000,
                            initial_nav_usd=1.0,
                            accounting_reset_session=validation[0],
                            force_final_liquidation=True,
                        )
                        selected = [
                            row
                            for row in evaluation_rows(ledger, validation[0])
                            if row.session <= validation_intervals[-1] or row.final_liquidation
                        ]
                        interval_rows = [row for row in selected if not row.final_liquidation]
                        active = [row for row in interval_rows if row.active_names]
                        if active:
                            supported += 1
                        active_days += len(active)
                        raw_net = math.prod(1 + row.net_return for row in selected) - 1
                        adjusted_returns = []
                        for row in selected:
                            if row.final_liquidation:
                                adjusted = -row.transaction_cost
                            else:
                                weight = dict(row.target_weights).get(symbol, 0.0)
                                adjusted = weight * (return_by_key[(row.session, symbol)].value - market_by_session[row.session])
                                adjusted -= row.transaction_cost
                                if weight > 0:
                                    long_sum += row.gross_return
                                    long_obs += 1
                                elif weight < 0:
                                    short_sum += row.gross_return
                                    short_obs += 1
                            adjusted_returns.append(adjusted)
                        stock_net.append(raw_net)
                        stock_adjusted.append(math.prod(1 + value for value in adjusted_returns) - 1)
                        stock_turnover.append(sum(row.turnover for row in selected))
                    eligible = len(validation_intervals) * len(symbols)
                    fold_rows.append(
                        FoldCandidateResult(
                            signal_kind=signal_kind,
                            hold_sessions=hold_sessions,
                            threshold=threshold,
                            fold_index=int(fold["fold_index"]),
                            validation_start=validation[0],
                            validation_end=validation_intervals[-1],
                            eligible_intervals=len(validation_intervals),
                            supported_stocks=supported,
                            active_stock_days=active_days,
                            active_fraction=active_days / eligible,
                            mean_stock_net_return=fmean(stock_net),
                            median_stock_net_return=median(stock_net),
                            positive_stock_rate=sum(value > 0 for value in stock_net) / len(stock_net),
                            mean_stock_market_adjusted_net_return=fmean(stock_adjusted),
                            median_stock_market_adjusted_net_return=median(stock_adjusted),
                            average_stock_turnover=fmean(stock_turnover),
                            long_gross_return_sum=long_sum,
                            short_gross_return_sum=short_sum,
                            long_observations=long_obs,
                            short_observations=short_obs,
                        )
                    )
    grouped: dict[tuple[SignalKind, int, float], list[FoldCandidateResult]] = defaultdict(list)
    for fold_result in fold_rows:
        grouped[(fold_result.signal_kind, fold_result.hold_sessions, fold_result.threshold)].append(fold_result)
    summaries: list[CandidateSummary] = []
    for (signal_kind, hold_sessions, threshold), candidate_folds in sorted(grouped.items()):
        adjusted_values = np.asarray(
            [fold.median_stock_market_adjusted_net_return for fold in candidate_folds], dtype=float
        )
        raw_values = [fold.median_stock_net_return for fold in candidate_folds]
        active_fraction_mean = fmean(fold.active_fraction for fold in candidate_folds)
        supported_mean = fmean(fold.supported_stocks for fold in candidate_folds)
        rejections = []
        if min(fold.supported_stocks for fold in candidate_folds) < spec.minimum_supported_stocks:
            rejections.append("minimum_supported_stocks")
        if min(fold.active_fraction for fold in candidate_folds) < spec.minimum_active_fraction:
            rejections.append("minimum_active_fraction")
        fold_iqr = float(np.quantile(adjusted_values, 0.75) - np.quantile(adjusted_values, 0.25))
        median_adjusted = float(np.median(adjusted_values))
        objective = median_adjusted - spec.stability_penalty * fold_iqr if not rejections else None
        summaries.append(
            CandidateSummary(
                signal_kind=signal_kind,
                hold_sessions=hold_sessions,
                threshold=threshold,
                valid=not rejections,
                rejection_reasons=tuple(rejections),
                objective=objective,
                median_fold_market_adjusted_return=median_adjusted,
                fold_return_iqr=fold_iqr,
                median_fold_raw_return=median(raw_values),
                mean_active_fraction=active_fraction_mean,
                mean_supported_stocks=supported_mean,
                mean_turnover=fmean(fold.average_stock_turnover for fold in candidate_folds),
                positive_folds=sum(fold.median_stock_market_adjusted_net_return > 0 for fold in candidate_folds),
            )
        )
    summaries.sort(
        key=lambda row: (
            not row.valid,
            -(row.objective if row.objective is not None else -math.inf),
            row.fold_return_iqr,
            row.mean_turnover,
            row.signal_kind,
            row.hold_sessions,
            row.threshold,
        )
    )
    return tuple(fold_rows), tuple(summaries)


def run_development_tests(
    run_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    spec: DevelopmentTestSpec | None = None,
) -> DevelopmentTestRun:
    """Execute and immutably materialize the complete development test battery."""

    root = Path(run_dir)
    report_manifest = read_json(root / "manifests" / "report.json")
    if report_manifest.get("status") != "completed":
        raise DevelopmentTestError("source run report stage is not completed")
    config = report_manifest["config"]
    evaluation_start = str(config["run"]["evaluation_start"])
    price_path = Path(str(config["prices"]["panel_path"]))
    derived_root = Path(str(config["outputs"]["derived_root"])) / root.name
    events_path = derived_root / "events" / "events.jsonl"
    scores_path = derived_root / "scores" / "scores.jsonl"
    folds_path = root / "tuning" / "fold_definitions.json"
    required = (price_path, events_path, scores_path, folds_path)
    if any(not path.is_file() for path in required):
        missing = [path.as_posix() for path in required if not path.is_file()]
        raise DevelopmentTestError(f"required artifacts are missing: {missing}")
    test_spec = spec or DevelopmentTestSpec(cost_bps_per_side=float(config["execution"]["cost_bps_per_side"]))
    identity_payload = {
        "analysis_schema_version": 3,
        "source_run_id": root.name,
        "source_run_identity": report_manifest.get("run_identity_sha256"),
        "evaluation_start": evaluation_start,
        "events_sha256": sha256_file(events_path),
        "scores_sha256": sha256_file(scores_path),
        "prices_sha256": sha256_file(price_path),
        "folds_sha256": sha256_file(folds_path),
        "spec": asdict(test_spec),
        "market_adjustment": "stock forward return minus equal-weight 33-stock contemporaneous forward return",
        "boundary_rule": "return endpoint must be strictly before evaluation_start",
    }
    identity_hash = sha256_text(canonical_json(identity_payload))
    experiment_id = f"news-surprise-dev-{identity_hash[:12]}"
    output_dir = Path(output_root) if output_root is not None else root / "development_tests" / experiment_id
    report_path = output_dir / "report.md"
    candidate_path = output_dir / "candidate_summary.csv"
    folds_output_path = output_dir / "candidate_fold_results.csv"
    label_path = output_dir / "label_horizon_results.csv"
    ic_path = output_dir / "information_coefficients.csv"
    signal_path = output_dir / "stock_session_signals.csv"
    figure_path = output_dir / "figures" / "label_horizon_market_adjusted.svg"
    candidate_figure_path = output_dir / "figures" / "candidate_objectives.svg"
    manifest_path = output_dir / "manifest.json"
    expected = (report_path, candidate_path, folds_output_path, label_path, ic_path, signal_path, figure_path, candidate_figure_path)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("identity_sha256") != identity_hash:
            raise DevelopmentTestError(f"existing development-test identity mismatch: {output_dir}")
        for path in expected:
            expected_hash = existing.get("outputs", {}).get(path.name, {}).get("sha256")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise DevelopmentTestError(f"completed development-test output is missing or changed: {path}")
        selected_payload = existing.get("selected_candidate")
        selected = CandidateSummary(**selected_payload) if isinstance(selected_payload, dict) else None
        return DevelopmentTestRun(experiment_id, output_dir, report_path, candidate_path, label_path, manifest_path, selected, True)

    opens = load_adjusted_opens_csv(price_path)
    price_lookup = {(row.symbol, row.session): row.adjusted_open for row in opens}
    symbols = tuple(sorted({row.symbol for row in opens}))
    sessions = tuple(sorted({row.session for row in opens}))
    development_sessions = tuple(
        session for session in sessions if session < evaluation_start and session >= min(config_session(report_manifest))
    )
    events = read_jsonl(events_path)
    scores = read_jsonl(scores_path)
    session_signals = aggregate_stock_session_scores(events, scores, development_sessions)
    labels, ics = calculate_label_horizon_results(
        events,
        scores,
        session_signals,
        price_lookup,
        sessions,
        symbols,
        evaluation_start,
        test_spec.horizons,
    )
    folds = read_json(folds_path)["folds"]
    returns = calculate_open_to_open_returns(opens)
    fold_results, candidates = evaluate_candidate_folds(
        session_signals,
        returns,
        development_sessions,
        symbols,
        folds,
        evaluation_start,
        test_spec,
    )
    selected = next((row for row in candidates if row.valid), None)
    if selected is None:
        raise DevelopmentTestError("all development candidates failed guardrails")
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(signal_path, _csv(session_signals))
    atomic_write_text(label_path, _csv(labels))
    atomic_write_text(ic_path, _csv(ics))
    atomic_write_text(folds_output_path, _csv(fold_results))
    atomic_write_text(candidate_path, _csv(candidates))
    atomic_write_text(figure_path, _label_horizon_svg(labels))
    atomic_write_text(candidate_figure_path, _candidate_svg(candidates))
    atomic_write_text(report_path, _report(experiment_id, test_spec, labels, ics, candidates, fold_results, selected))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": experiment_id,
        "identity_sha256": identity_hash,
        "identity": identity_payload,
        "selected_candidate": asdict(selected),
        "row_counts": {
            "stock_session_signals": len(session_signals),
            "label_horizon_results": len(labels),
            "information_coefficients": len(ics),
            "candidate_fold_results": len(fold_results),
            "candidate_summaries": len(candidates),
        },
        "outputs": {
            path.name: {"path": path.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in expected
        },
        "warning": "Development-only selection evidence; the prior evaluation block was not used and must not be rerun for selection.",
    }
    atomic_write_json(manifest_path, manifest)
    return DevelopmentTestRun(experiment_id, output_dir, report_path, candidate_path, label_path, manifest_path, selected, False)


def config_session(report_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    selected_path = Path(str(report_manifest["outputs"]["summary"]["path"])).parents[1] / "tuning" / "selected_config.json"
    selected = read_json(selected_path)
    return tuple(str(value) for value in selected["development_sessions"])


def _csv(rows: Sequence[Any]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    payloads = [asdict(row) for row in rows]
    writer = csv.DictWriter(buffer, fieldnames=list(payloads[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(payloads)
    return buffer.getvalue()


def _report(
    experiment_id: str,
    spec: DevelopmentTestSpec,
    labels: Sequence[LabelHorizonResult],
    ics: Sequence[InformationCoefficientResult],
    candidates: Sequence[CandidateSummary],
    folds: Sequence[FoldCandidateResult],
    selected: CandidateSummary,
) -> str:
    session_labels = [row for row in labels if row.aggregation == "stock_session_mean"]
    best_ics = sorted(
        (row for row in ics if row.aggregation == "stock_session_mean" and row.return_kind == "market_adjusted"),
        key=lambda row: abs(row.spearman_rho or 0),
        reverse=True,
    )
    selected_folds = [
        row
        for row in folds
        if row.signal_kind == selected.signal_kind and row.hold_sessions == selected.hold_sessions and row.threshold == selected.threshold
    ]
    repeated_count = sum(row.observations for row in labels if row.aggregation == "repeated_event")
    session_count = sum(row.observations for row in labels if row.aggregation == "stock_session_mean")
    lines = [
        f"# Development-only news-signal tests: {experiment_id}",
        "",
        "> The locked evaluation block is excluded from every return, fit, rank, and chart in this report.",
        "",
        "## Material Passport",
        "",
        f"- Experiment ID: `{experiment_id}`",
        "- Type: deterministic development-only analysis",
        "- Status: completed",
        "- Verification status: ANALYZED until deterministic replay and full tests pass",
        "",
        "## Tested design",
        "",
        f"- Horizons/holds: `{', '.join(map(str, spec.horizons))}` sessions.",
        f"- Thresholds: `{', '.join(f'{value:g}' for value in spec.thresholds)}`.",
        "- Signal units: repeated events for duplication diagnosis; one mean score per stock-session for strategy tests.",
        "- Centering: each stock's training mean shrunk toward the pooled training mean with `n/(n+50)`.",
        "- Market adjustment: stock return minus the contemporaneous equal-weight 33-stock return.",
        f"- Costs: `{spec.cost_bps_per_side:g}` bps per side; expanding validation folds only.",
        "- Boundary: a development return is retained only when its endpoint is before the evaluation start.",
        "",
        "## Label-conditioned forward returns",
        "",
        f"Repeated-event tables contain {repeated_count:,} horizon observations versus {session_count:,} stock-session observations. "
        "The repeated version reuses the same stock return for every same-session article and is diagnostic only.",
        "",
        "| Horizon | Label | N stock-sessions | Mean raw return | Mean market-adjusted return | Adjusted positive rate |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for label_result in sorted(session_labels, key=lambda item: (item.horizon_sessions, item.label)):
        lines.append(
            f"| {label_result.horizon_sessions} | {label_result.label} | {label_result.observations} | "
            f"{label_result.mean_raw_return:.3%} | {label_result.mean_market_adjusted_return:.3%} | "
            f"{label_result.adjusted_positive_rate:.1%} |"
        )
    lines.extend(["", "![Label horizon results](figures/label_horizon_market_adjusted.svg)", ""])
    if best_ics:
        lines.extend(
            [
                "### Directional ordering",
                "",
                "| Horizon | Spearman rho | Unadjusted p-value | BH-FDR q-value | N |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for ic_result in sorted(best_ics, key=lambda item: item.horizon_sessions):
            rho = "n/a" if ic_result.spearman_rho is None else f"{ic_result.spearman_rho:.3f}"
            p_value = "n/a" if ic_result.p_value is None else f"{ic_result.p_value:.4f}"
            q_value = "n/a" if ic_result.fdr_q_value is None else f"{ic_result.fdr_q_value:.4f}"
            lines.append(
                f"| {ic_result.horizon_sessions} | {rho} | {p_value} | {q_value} | {ic_result.observations} |"
            )
    lines.extend(
        [
            "",
            "## Expanding-fold strategy candidates",
            "",
            f"The highest-ranked valid development candidate uses **{selected.signal_kind}** stock-session scores, "
            f"a **{selected.hold_sessions}-session hold**, and threshold **{selected.threshold:g}**.",
            "",
            f"- Stability-penalized objective: `{selected.objective:.6f}`.",
            f"- Median fold market-adjusted stock return: `{selected.median_fold_market_adjusted_return:.3%}`.",
            f"- Median fold raw stock return: `{selected.median_fold_raw_return:.3%}`.",
            f"- Mean active stock-day fraction: `{selected.mean_active_fraction:.1%}`.",
            f"- Mean supported stocks: `{selected.mean_supported_stocks:.1f}`.",
            f"- Positive validation folds: `{selected.positive_folds}/3`.",
            "",
            (
                "| Fold | Validation | Median raw | Median market-adjusted | Positive stocks | Active fraction | "
                "Stocks | Mean long/short gross return |"
            ),
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold_result in sorted(selected_folds, key=lambda item: item.fold_index):
        lines.append(
            f"| {fold_result.fold_index} | {fold_result.validation_start} to {fold_result.validation_end} | "
            f"{fold_result.median_stock_net_return:.3%} | "
            f"{fold_result.median_stock_market_adjusted_net_return:.3%} | "
            f"{fold_result.positive_stock_rate:.1%} | {fold_result.active_fraction:.1%} | "
            f"{fold_result.supported_stocks} | "
            f"{fold_result.long_gross_return_sum/fold_result.long_observations:.3%}/"
            f"{fold_result.short_gross_return_sum/fold_result.short_observations:.3%} |"
        )
    lines.extend(
        [
            "",
            "![Candidate objectives](figures/candidate_objectives.svg)",
            "",
            "## Interpretation limits",
            "",
            (
                f"- `{len(candidates)}` candidates and multiple label/horizon contrasts were inspected; "
                "p-values are descriptive and unadjusted."
            ),
            "- Overlapping forward horizons and same-date stocks are dependent observations.",
            "- The equal-weight 33-stock adjustment is an internal benchmark, not a broad-market index or factor model.",
            "- Development folds are short, so a positive objective is not proof of stable or deployable alpha.",
            "- This report can motivate one future lock, but the already observed evaluation block cannot validate that lock.",
            "",
            "### Statistical fallacy scan — 11/11 checked",
            "",
            (
                "- **Simpson's paradox:** open; sector-level reversals were not tested because the locked stock diagnostics "
                "lack a canonical sector field."
            ),
            "- **Ecological fallacy:** no individual-investor or firm-level causal inference is made from aggregate stock-session results.",
            "- **Berkson's paradox:** caution; the fixed 33-company sample is selected rather than a random market sample.",
            "- **Collider bias:** no post-signal controls were introduced in these tests.",
            "- **Base-rate neglect:** label counts and active fractions are reported alongside outcomes.",
            "- **Regression to the mean:** stocks were not selected from extreme development outcomes.",
            "- **Survivorship bias:** caution; the fixed contemporary company universe can contain survivorship bias.",
            (
                "- **Look-elsewhere effect:** caution; all 32 candidates and all information-coefficient tests are retained, "
                "with BH-FDR q-values for the latter."
            ),
            "- **Garden of forking paths:** caution; the grid is explicit and immutable here but was not prospectively preregistered.",
            "- **Correlation versus causation:** the results are described as associations, not causal effects.",
            (
                "- **Reverse causality:** timestamps precede measured returns, but the observational design still does not "
                "establish causality."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _label_horizon_svg(rows: Sequence[LabelHorizonResult]) -> str:
    selected = [row for row in rows if row.aggregation == "stock_session_mean"]
    width, height = 900, 520
    left, right, top, bottom = 75, 30, 55, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [row.mean_market_adjusted_return for row in selected] or [0.0]
    bound = max(max(abs(value) for value in values), 0.001)
    labels = ("negative", "neutral", "positive")
    colors = {"negative": "#e09045", "neutral": "#9a9a9a", "positive": "#2864a8"}
    grouped = {(row.horizon_sessions, row.label): row for row in selected}
    group_width = plot_width / 4
    bar_width = group_width / 4
    zero_y = top + plot_height / 2
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{left}" y="27" font-family="sans-serif" font-size="19">'
            "Mean market-adjusted forward return by stock-session label</text>"
        ),
        (
            f'<text x="{left}" y="46" font-family="sans-serif" font-size="12" fill="#555">'
            "Development only; one averaged signal per stock-session</text>"
        ),
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}" stroke="#333"/>',
        f'<text x="8" y="{top + 8}" font-family="sans-serif" font-size="11">+{bound:.2%}</text>',
        f'<text x="8" y="{top + plot_height}" font-family="sans-serif" font-size="11">-{bound:.2%}</text>',
    ]
    for group_index, horizon in enumerate((1, 3, 5, 10)):
        center = left + (group_index + 0.5) * group_width
        for label_index, label in enumerate(labels):
            row = grouped[(horizon, label)]
            value = row.mean_market_adjusted_return
            value_y = zero_y - (value / bound) * (plot_height / 2)
            x = center + (label_index - 1) * bar_width - bar_width * 0.38
            pieces.append(
                f'<rect x="{x:.2f}" y="{min(zero_y, value_y):.2f}" width="{bar_width * 0.76:.2f}" '
                f'height="{max(1, abs(value_y - zero_y)):.2f}" fill="{colors[label]}"/>'
            )
        pieces.append(
            f'<text x="{center:.2f}" y="{height - bottom + 24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{horizon} sessions</text>'
        )
    for index, label in enumerate(labels):
        x = left + index * 105
        pieces.append(f'<rect x="{x}" y="{height - 28}" width="12" height="12" fill="{colors[label]}"/>')
        pieces.append(f'<text x="{x + 17}" y="{height - 18}" font-family="sans-serif" font-size="11">{label}</text>')
    pieces.append("</svg>\n")
    return "\n".join(pieces)


def _candidate_svg(rows: Sequence[CandidateSummary]) -> str:
    valid = [row for row in rows if row.valid][:12]
    width, height = 950, 520
    left, right, top, bottom = 230, 35, 55, 35
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [row.objective or 0.0 for row in valid] or [0.0]
    low, high = min(values + [0.0]), max(values + [0.0])
    span = max(high - low, 0.001)
    zero_x = left + plot_width * (0 - low) / span
    row_height = plot_height / max(1, len(valid))
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="27" font-family="sans-serif" font-size="19">Top valid development candidate objectives</text>',
        (
            f'<text x="{left}" y="46" font-family="sans-serif" font-size="12" fill="#555">'
            "Median fold market-adjusted return minus 0.5 × fold IQR</text>"
        ),
        f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{height - bottom}" stroke="#555"/>',
        f'<text x="{left}" y="{height - 12}" font-family="sans-serif" font-size="11">{low:.2%}</text>',
        f'<text x="{width-right}" y="{height - 12}" text-anchor="end" font-family="sans-serif" font-size="11">{high:.2%}</text>',
    ]
    for index, row in enumerate(valid):
        value = row.objective or 0.0
        value_x = left + plot_width * (value - low) / span
        y = top + index * row_height + row_height * 0.2
        label = f"{row.signal_kind}, {row.hold_sessions}s, t={row.threshold:g}"
        pieces.append(
            f'<text x="{left - 8}" y="{y + row_height * 0.45:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{html.escape(label)}</text>'
        )
        pieces.append(
            f'<rect x="{min(zero_x, value_x):.2f}" y="{y:.2f}" width="{max(1, abs(value_x - zero_x)):.2f}" '
            f'height="{row_height * 0.6:.2f}" fill="{"#2864a8" if value >= 0 else "#e09045"}"/>'
        )
    pieces.append("</svg>\n")
    return "\n".join(pieces)
