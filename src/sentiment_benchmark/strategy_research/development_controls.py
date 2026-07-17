"""Development-only controls for the non-refreshing stock-session strategy.

The module compares one pre-declared v2 rule with timing, refresh, inversion,
buy-and-hold, and within-stock shuffled-label controls. No interval may reach
the already-observed formal evaluation boundary.
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
from typing import Any, Literal

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
from .development_tests import SessionSignal, aggregate_stock_session_scores, config_session
from .ledger import evaluation_rows, run_open_to_open_ledger
from .market import OpenToOpenReturn, calculate_open_to_open_returns, load_adjusted_opens_csv
from .model_only_development import (
    MODEL_ONLY_CONTRACT,
    ModelOnlyDevelopmentError,
    validate_model_only_scoring_universe,
)
from .portfolio import PositionTarget, TargetPortfolio
from .run_validation import (
    CompletedRunValidationError,
    load_completed_run_snapshot,
    validate_adjusted_open_price_panel,
    validate_completed_stage_output,
)

ControlName = Literal["sentiment_v2", "refreshing_signal", "news_timing_long", "inverted_sentiment"]
ScoreAggregationMode = Literal["all_scored_events", "eligible_nonzero_events"]


class DevelopmentControlError(RuntimeError):
    """Raised when a development control cannot be evaluated safely."""


@dataclass(frozen=True)
class DevelopmentControlSpec:
    threshold: float = 0.75
    hold_sessions: int = 10
    cost_bps_per_side: float = 10.0
    shuffle_replications: int = 500
    seed: int = 20_260_715
    stability_penalty: float = 0.5
    maximum_active_fraction: float = 0.40
    minimum_supported_stocks: int = 25
    maximum_shuffle_p_value: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if self.hold_sessions < 1 or self.shuffle_replications < 1:
            raise ValueError("hold_sessions and shuffle_replications must be positive")
        if self.cost_bps_per_side < 0 or not 0 <= self.maximum_active_fraction <= 1:
            raise ValueError("cost and activity constraints are invalid")
        if self.minimum_supported_stocks < 1 or not 0 < self.maximum_shuffle_p_value < 1:
            raise ValueError("support and shuffle constraints are invalid")


@dataclass(frozen=True)
class StockFoldOutcome:
    control: ControlName
    fold_index: int
    symbol: str
    observations: int
    active_days: int
    active_fraction: float
    net_return: float
    market_adjusted_net_return: float
    buy_and_hold_return: float
    excess_vs_buy_and_hold: float
    turnover: float
    long_observations: int
    short_observations: int
    mean_long_gross_return: float | None
    mean_short_gross_return: float | None


@dataclass(frozen=True)
class ControlFoldResult:
    control: ControlName
    fold_index: int
    validation_start: str
    validation_end: str
    eligible_intervals: int
    supported_stocks: int
    active_fraction: float
    median_stock_net_return: float
    median_stock_market_adjusted_net_return: float
    median_excess_vs_buy_and_hold: float
    positive_stock_rate: float
    outperform_buy_and_hold_rate: float
    mean_stock_turnover: float
    mean_long_gross_return: float | None
    mean_short_gross_return: float | None


@dataclass(frozen=True)
class ControlSummary:
    control: ControlName
    objective: float
    median_fold_market_adjusted_return: float
    fold_return_iqr: float
    median_fold_raw_return: float
    median_fold_excess_vs_buy_and_hold: float
    mean_active_fraction: float
    mean_supported_stocks: float
    mean_turnover: float
    positive_folds: int
    long_positive_folds: int
    short_positive_folds: int


@dataclass(frozen=True)
class ShuffleResult:
    replication: int
    seed: int
    objective: float
    median_fold_market_adjusted_return: float
    fold_return_iqr: float
    mean_active_fraction: float
    mean_supported_stocks: float
    positive_folds: int


@dataclass(frozen=True)
class AcceptanceCheck:
    check: str
    passed: bool
    observed: str
    requirement: str


@dataclass(frozen=True)
class DevelopmentControlRun:
    experiment_id: str
    output_dir: Path
    report_path: Path
    manifest_path: Path
    v2_summary: ControlSummary
    acceptance: tuple[AcceptanceCheck, ...]
    reused: bool


def select_control_scores(
    scores: Sequence[Mapping[str, Any]],
    aggregation_mode: ScoreAggregationMode,
) -> tuple[dict[str, Any], ...]:
    """Select the event records admitted to stock-session aggregation.

    The eligible-only contract is deliberately strict: every source row must be
    a successful filtered-strategy record, eligible rows must be sign-only, and
    ineligible rows must be explicit zeros. This prevents missing or malformed
    model output from disappearing as if it were neutral news.
    """

    copied = tuple(dict(row) for row in scores)
    if aggregation_mode == "all_scored_events":
        return copied
    if aggregation_mode != "eligible_nonzero_events":
        raise DevelopmentControlError(f"unsupported score aggregation mode: {aggregation_mode}")
    selected: list[dict[str, Any]] = []
    for row in copied:
        event_id = str(row.get("event_id") or "")
        if row.get("status") != "success" or not event_id:
            raise DevelopmentControlError("eligible-only aggregation requires successful identified score rows")
        eligible = row.get("eligible")
        if not isinstance(eligible, bool):
            raise DevelopmentControlError(f"eligible-only score has no boolean eligibility flag: {event_id}")
        score = float(row["score"])
        if eligible and score not in {-1.0, 1.0}:
            raise DevelopmentControlError(f"eligible-only score is not sign-only: {event_id}")
        if not eligible and score != 0.0:
            raise DevelopmentControlError(f"ineligible score is not an explicit zero: {event_id}")
        if eligible:
            selected.append(row)
    if not selected:
        raise DevelopmentControlError("eligible-only aggregation selected no events")
    return tuple(selected)


def build_control_exposures(
    sessions: Sequence[str],
    symbols: Sequence[str],
    signals: Sequence[SessionSignal],
    *,
    threshold: float,
    hold_sessions: int,
    control: ControlName,
) -> dict[str, tuple[float, ...]]:
    """Create direct exposures while making refresh semantics explicit."""

    if control not in {"sentiment_v2", "refreshing_signal", "news_timing_long", "inverted_sentiment"}:
        raise ValueError(f"unsupported control: {control}")
    ordered_sessions = tuple(sessions)
    ordinal = {session: index for index, session in enumerate(ordered_sessions)}
    signal_lookup = {(row.session, row.symbol): row.mean_score for row in signals}
    output: dict[str, tuple[float, ...]] = {}
    for symbol in symbols:
        current = 0.0
        expires_at: int | None = None
        symbol_exposures: list[float] = []
        for session in ordered_sessions:
            index = ordinal[session]
            if expires_at is not None and index >= expires_at:
                current = 0.0
                expires_at = None
            score = signal_lookup.get((session, symbol))
            direction = 0.0
            if score is not None:
                if control == "news_timing_long":
                    direction = 1.0
                elif abs(score) >= threshold and score != 0:
                    direction = math.copysign(1.0, score)
                    if control == "inverted_sentiment":
                        direction *= -1
            if direction:
                if current == 0 or direction != current:
                    current = direction
                    expires_at = index + hold_sessions
                elif control == "refreshing_signal":
                    expires_at = index + hold_sessions
            symbol_exposures.append(current)
        output[symbol] = tuple(symbol_exposures)
    return output


def shuffle_scores_within_stock(
    events: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    allowed_sessions: Sequence[str],
    *,
    seed: int,
) -> tuple[SessionSignal, ...]:
    """Shuffle event scores within stock, preserving dates, counts, and score multisets."""

    allowed = set(allowed_sessions)
    score_by_event = {str(row["event_id"]): float(row["score"]) for row in scores if row.get("status") == "success"}
    by_symbol: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for event in events:
        session = str(event.get("eligible_execution_session") or "")
        if session not in allowed:
            continue
        event_id = str(event.get("event_id") or "")
        symbol = str(event.get("symbol") or "").upper()
        if event_id not in score_by_event:
            continue
        by_symbol[symbol].append((session, event_id))
    random = np.random.default_rng(seed)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for symbol in sorted(by_symbol):
        event_rows = sorted(by_symbol[symbol], key=lambda row: (row[0], row[1]))
        values = np.asarray([score_by_event[event_id] for _, event_id in event_rows], dtype=float)
        shuffled = random.permutation(values)
        for (session, _), value in zip(event_rows, shuffled, strict=True):
            grouped[(session, symbol)].append(float(value))
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


def _evaluate_control(
    control: ControlName,
    signals: Sequence[SessionSignal],
    returns: Sequence[OpenToOpenReturn],
    sessions: Sequence[str],
    symbols: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
    evaluation_start: str,
    spec: DevelopmentControlSpec,
    *,
    keep_stock_rows: bool,
) -> tuple[tuple[ControlFoldResult, ...], tuple[StockFoldOutcome, ...]]:
    return_lookup = {(row.session, row.symbol): row for row in returns}
    valid_returns = [row for row in returns if row.next_session < evaluation_start]
    market_by_session: dict[str, float] = {}
    for session in sessions:
        values = [return_lookup[(session, symbol)].value for symbol in symbols if (session, symbol) in return_lookup]
        if len(values) == len(symbols):
            market_by_session[session] = fmean(values)
    fold_results: list[ControlFoldResult] = []
    stock_results: list[StockFoldOutcome] = []
    for fold in folds:
        validation = tuple(str(value) for value in fold["validation_sessions"])
        validation_intervals = tuple(
            session
            for session in validation
            if session in market_by_session and return_lookup[(session, symbols[0])].next_session < evaluation_start
        )
        if not validation_intervals:
            raise DevelopmentControlError("a fold has no non-leaking validation intervals")
        combined = tuple(session for session in sessions if session <= validation_intervals[-1])
        exposures = build_control_exposures(
            combined,
            symbols,
            signals,
            threshold=spec.threshold,
            hold_sessions=spec.hold_sessions,
            control=control,
        )
        fold_stock_rows: list[StockFoldOutcome] = []
        for symbol in symbols:
            targets = [_target(session, symbol, exposure) for session, exposure in zip(combined, exposures[symbol], strict=True)]
            symbol_returns = [row for row in valid_returns if row.symbol == symbol]
            ledger = run_open_to_open_ledger(
                targets,
                symbol_returns,
                cost_rate_per_side=spec.cost_bps_per_side / 10_000,
                initial_nav_usd=1.0,
                accounting_reset_session=validation[0],
                force_final_liquidation=True,
            )
            selected = [
                row for row in evaluation_rows(ledger, validation[0]) if row.session <= validation_intervals[-1] or row.final_liquidation
            ]
            interval_rows = [row for row in selected if not row.final_liquidation]
            active = [row for row in interval_rows if row.active_names]
            adjusted_returns: list[float] = []
            long_returns: list[float] = []
            short_returns: list[float] = []
            for row in selected:
                if row.final_liquidation:
                    adjusted_returns.append(-row.transaction_cost)
                    continue
                exposure = dict(row.target_weights).get(symbol, 0.0)
                adjusted = exposure * (return_lookup[(row.session, symbol)].value - market_by_session[row.session])
                adjusted_returns.append(adjusted - row.transaction_cost)
                if exposure > 0:
                    long_returns.append(row.gross_return)
                elif exposure < 0:
                    short_returns.append(row.gross_return)
            net_return = math.prod(1 + row.net_return for row in selected) - 1
            adjusted_net = math.prod(1 + value for value in adjusted_returns) - 1
            buy_and_hold = math.prod(1 + return_lookup[(row.session, symbol)].value for row in interval_rows) - 1
            fold_stock_rows.append(
                StockFoldOutcome(
                    control=control,
                    fold_index=int(fold["fold_index"]),
                    symbol=symbol,
                    observations=len(selected),
                    active_days=len(active),
                    active_fraction=len(active) / len(validation_intervals),
                    net_return=net_return,
                    market_adjusted_net_return=adjusted_net,
                    buy_and_hold_return=buy_and_hold,
                    excess_vs_buy_and_hold=net_return - buy_and_hold,
                    turnover=sum(row.turnover for row in selected),
                    long_observations=len(long_returns),
                    short_observations=len(short_returns),
                    mean_long_gross_return=fmean(long_returns) if long_returns else None,
                    mean_short_gross_return=fmean(short_returns) if short_returns else None,
                )
            )
        supported = [row for row in fold_stock_rows if row.active_days]
        long_values = [row.mean_long_gross_return for row in fold_stock_rows if row.mean_long_gross_return is not None]
        short_values = [row.mean_short_gross_return for row in fold_stock_rows if row.mean_short_gross_return is not None]
        fold_results.append(
            ControlFoldResult(
                control=control,
                fold_index=int(fold["fold_index"]),
                validation_start=validation[0],
                validation_end=validation_intervals[-1],
                eligible_intervals=len(validation_intervals),
                supported_stocks=len(supported),
                active_fraction=sum(row.active_days for row in fold_stock_rows) / (len(validation_intervals) * len(symbols)),
                median_stock_net_return=median(row.net_return for row in fold_stock_rows),
                median_stock_market_adjusted_net_return=median(row.market_adjusted_net_return for row in fold_stock_rows),
                median_excess_vs_buy_and_hold=median(row.excess_vs_buy_and_hold for row in fold_stock_rows),
                positive_stock_rate=sum(row.net_return > 0 for row in fold_stock_rows) / len(fold_stock_rows),
                outperform_buy_and_hold_rate=sum(row.excess_vs_buy_and_hold > 0 for row in fold_stock_rows) / len(fold_stock_rows),
                mean_stock_turnover=fmean(row.turnover for row in fold_stock_rows),
                mean_long_gross_return=fmean(long_values) if long_values else None,
                mean_short_gross_return=fmean(short_values) if short_values else None,
            )
        )
        if keep_stock_rows:
            stock_results.extend(fold_stock_rows)
    return tuple(fold_results), tuple(stock_results)


def _summarize(control: ControlName, folds: Sequence[ControlFoldResult], stability_penalty: float) -> ControlSummary:
    adjusted = np.asarray([row.median_stock_market_adjusted_net_return for row in folds], dtype=float)
    iqr = float(np.quantile(adjusted, 0.75) - np.quantile(adjusted, 0.25))
    median_adjusted = float(np.median(adjusted))
    return ControlSummary(
        control=control,
        objective=median_adjusted - stability_penalty * iqr,
        median_fold_market_adjusted_return=median_adjusted,
        fold_return_iqr=iqr,
        median_fold_raw_return=median(row.median_stock_net_return for row in folds),
        median_fold_excess_vs_buy_and_hold=median(row.median_excess_vs_buy_and_hold for row in folds),
        mean_active_fraction=fmean(row.active_fraction for row in folds),
        mean_supported_stocks=fmean(row.supported_stocks for row in folds),
        mean_turnover=fmean(row.mean_stock_turnover for row in folds),
        positive_folds=sum(row.median_stock_market_adjusted_net_return > 0 for row in folds),
        long_positive_folds=sum((row.mean_long_gross_return or 0.0) > 0 for row in folds),
        short_positive_folds=sum((row.mean_short_gross_return or 0.0) > 0 for row in folds),
    )


def _acceptance(
    v2: ControlSummary,
    controls: Mapping[ControlName, ControlSummary],
    shuffles: Sequence[ShuffleResult],
    spec: DevelopmentControlSpec,
) -> tuple[AcceptanceCheck, ...]:
    null_objectives = np.asarray([row.objective for row in shuffles], dtype=float)
    shuffle_p = (1 + int(np.sum(null_objectives >= v2.objective))) / (len(shuffles) + 1)
    return (
        AcceptanceCheck("all_folds_positive", v2.positive_folds == 3, f"{v2.positive_folds}/3", "3/3"),
        AcceptanceCheck(
            "sparse_activity",
            v2.mean_active_fraction <= spec.maximum_active_fraction,
            f"{v2.mean_active_fraction:.1%}",
            f"<= {spec.maximum_active_fraction:.0%}",
        ),
        AcceptanceCheck(
            "stock_support",
            v2.mean_supported_stocks >= spec.minimum_supported_stocks,
            f"{v2.mean_supported_stocks:.1f}",
            f">= {spec.minimum_supported_stocks}",
        ),
        AcceptanceCheck(
            "beats_news_timing",
            v2.objective > controls["news_timing_long"].objective,
            f"{v2.objective:.4%} vs {controls['news_timing_long'].objective:.4%}",
            "strictly higher objective",
        ),
        AcceptanceCheck(
            "beats_refreshing_signal",
            v2.objective > controls["refreshing_signal"].objective,
            f"{v2.objective:.4%} vs {controls['refreshing_signal'].objective:.4%}",
            "strictly higher objective",
        ),
        AcceptanceCheck(
            "beats_shuffled_labels",
            shuffle_p <= spec.maximum_shuffle_p_value,
            f"p={shuffle_p:.4f}",
            f"p <= {spec.maximum_shuffle_p_value:.2f}",
        ),
        AcceptanceCheck(
            "long_side_not_consistently_losing",
            v2.long_positive_folds >= 2,
            f"{v2.long_positive_folds}/3 positive folds",
            ">= 2/3",
        ),
        AcceptanceCheck(
            "short_side_not_consistently_losing",
            v2.short_positive_folds >= 2,
            f"{v2.short_positive_folds}/3 positive folds",
            ">= 2/3",
        ),
        AcceptanceCheck(
            "lower_turnover_than_refreshing",
            v2.mean_turnover < controls["refreshing_signal"].mean_turnover,
            f"{v2.mean_turnover:.3f} vs {controls['refreshing_signal'].mean_turnover:.3f}",
            "strictly lower",
        ),
    )


def run_development_controls(
    run_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    spec: DevelopmentControlSpec | None = None,
    scores_path_override: str | Path | None = None,
    universe_dir_override: str | Path | None = None,
    candidate_label: str = "sentiment v2",
    aggregation_mode: ScoreAggregationMode = "all_scored_events",
) -> DevelopmentControlRun:
    """Run and immutably materialize controls and the shuffled-label null."""

    root = Path(run_dir)
    try:
        snapshot = load_completed_run_snapshot(root)
    except CompletedRunValidationError as exc:
        raise DevelopmentControlError(str(exc)) from exc
    report_manifest = snapshot.report
    config = snapshot.config
    evaluation_start = str(config["run"]["evaluation_start"])
    price_path = Path(str(config["prices"]["panel_path"]))
    derived_root = Path(str(config["outputs"]["derived_root"])) / root.name
    events_path = derived_root / "events" / "events.jsonl"
    scores_path = Path(scores_path_override) if scores_path_override is not None else derived_root / "scores" / "scores.jsonl"
    folds_path = root / "tuning" / "fold_definitions.json"
    required = (price_path, events_path, scores_path, folds_path)
    if any(not path.is_file() for path in required):
        raise DevelopmentControlError("one or more required completed-run artifacts are missing")
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
        raise DevelopmentControlError(str(exc)) from exc
    score_manifest_path: Path | None = None
    score_manifest: Mapping[str, Any] | None = None
    universe_manifest_path: Path | None = None
    universe_manifest: Mapping[str, Any] | None = None
    if scores_path_override is not None:
        score_manifest_path = scores_path.parent / "manifest.json"
        if not score_manifest_path.is_file():
            raise DevelopmentControlError("explicit model-only scores have no completed manifest")
        loaded_manifest = read_json(score_manifest_path)
        expected_score_hash = loaded_manifest.get("outputs", {}).get(scores_path.name, {}).get("sha256")
        if (
            loaded_manifest.get("schema_version") != 1
            or loaded_manifest.get("status") != "completed"
            or loaded_manifest.get("identity_sha256")
            != sha256_text(canonical_json(loaded_manifest.get("identity")))
            or loaded_manifest.get("identity", {}).get("contract") != MODEL_ONLY_CONTRACT
            or sha256_file(scores_path) != expected_score_hash
        ):
            raise DevelopmentControlError("explicit model-only score artifact is incomplete or changed")
        score_manifest = loaded_manifest
        universe_root = Path(universe_dir_override) if universe_dir_override is not None else scores_path.parent.parent.parent
        try:
            validate_model_only_scoring_universe(universe_root)
        except ModelOnlyDevelopmentError as exc:
            raise DevelopmentControlError(str(exc)) from exc
        universe_manifest_path = universe_root / "sample_manifest.json"
        universe_items_path = universe_root / "audit_items.csv"
        if not universe_manifest_path.is_file() or not universe_items_path.is_file():
            raise DevelopmentControlError("explicit model-only scores have no verifiable source universe")
        loaded_universe = read_json(universe_manifest_path)
        expected_items_hash = loaded_universe.get("outputs", {}).get(universe_items_path.name, {}).get("sha256")
        universe_identity = loaded_universe.get("identity", {})
        if (
            loaded_universe.get("schema_version") != 1
            or loaded_universe.get("status") != "completed"
            or sha256_file(universe_items_path) != expected_items_hash
            or loaded_manifest.get("identity", {}).get("universe_identity_sha256") != loaded_universe.get("identity_sha256")
            or universe_identity.get("contract") != MODEL_ONLY_CONTRACT
            or universe_identity.get("source_run_id") != root.name
            or universe_identity.get("source_run_identity") != report_manifest.get("run_identity_sha256")
            or universe_identity.get("evaluation_start") != evaluation_start
            or universe_identity.get("events_sha256") != sha256_file(events_path)
        ):
            raise DevelopmentControlError("explicit model-only strategy does not belong to the source run")
        universe_manifest = loaded_universe
    control_spec = spec or DevelopmentControlSpec(cost_bps_per_side=float(config["execution"]["cost_bps_per_side"]))
    identity_payload = {
        "analysis_schema_version": 5 if scores_path_override is not None else 3,
        "source_run_id": root.name,
        "source_run_identity": report_manifest.get("run_identity_sha256"),
        "evaluation_start": evaluation_start,
        "events_sha256": sha256_file(events_path),
        "events_manifest_sha256": events_manifest_hash,
        "scores_sha256": sha256_file(scores_path),
        "prices_sha256": sha256_file(price_path),
        "price_manifest_sha256": price_manifest_hash,
        "folds_sha256": sha256_file(folds_path),
        "tuning_manifest_sha256": tuning_manifest_hash,
        "spec": asdict(control_spec),
        "shuffle_contract": "permute event scores within stock; preserve event dates, counts, and per-stock score multiset",
        "boundary_rule": "return endpoint must be strictly before evaluation_start",
    }
    if scores_path_override is not None:
        assert (
            score_manifest_path is not None
            and score_manifest is not None
            and universe_manifest_path is not None
            and universe_manifest is not None
        )
        identity_payload["candidate_label"] = candidate_label
        identity_payload["scores_source"] = "explicit model-only filtered score artifact"
        identity_payload["scores_manifest_sha256"] = sha256_file(score_manifest_path)
        identity_payload["scores_artifact_identity"] = score_manifest.get("identity_sha256")
        identity_payload["universe_manifest_sha256"] = sha256_file(universe_manifest_path)
        identity_payload["universe_identity"] = universe_manifest.get("identity_sha256")
    if aggregation_mode != "all_scored_events":
        identity_payload["aggregation_mode"] = aggregation_mode
        identity_payload["aggregation_contract"] = (
            "mean only eligible sign-only events per stock-session; ineligible successful events are absent, not zeros"
        )
        identity_payload["shuffle_contract"] = (
            "permute eligible sign-only event scores within stock; preserve eligible event dates, counts, and sign multiset"
        )
    identity_hash = sha256_text(canonical_json(identity_payload))
    if aggregation_mode == "eligible_nonzero_events":
        prefix = "model-only-eligible-controls"
    else:
        prefix = "model-only-filtered-controls" if scores_path_override is not None else "sentiment-v2-controls"
    experiment_id = f"{prefix}-{identity_hash[:12]}"
    output_dir = Path(output_root) if output_root is not None else root / "development_controls" / experiment_id
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"
    control_summary_path = output_dir / "control_summary.csv"
    fold_path = output_dir / "control_fold_results.csv"
    stock_path = output_dir / "stock_fold_results.csv"
    shuffle_path = output_dir / "shuffled_null.csv"
    acceptance_path = output_dir / "acceptance_checks.csv"
    control_figure = output_dir / "figures" / "control_objectives.svg"
    shuffle_figure = output_dir / "figures" / "shuffled_null.svg"
    expected = (
        report_path,
        control_summary_path,
        fold_path,
        stock_path,
        shuffle_path,
        acceptance_path,
        control_figure,
        shuffle_figure,
    )
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("schema_version") != 1 or existing.get("status") != "completed":
            raise DevelopmentControlError(f"refusing to reuse incomplete control output: {output_dir}")
        if (
            existing.get("identity_sha256") != identity_hash
            or canonical_json(existing.get("identity")) != canonical_json(identity_payload)
        ):
            raise DevelopmentControlError(f"existing control identity mismatch: {output_dir}")
        actual_files = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()}
        expected_files = {path.relative_to(output_dir).as_posix() for path in expected} | {"manifest.json"}
        if actual_files != expected_files:
            raise DevelopmentControlError(f"completed control directory contains unexpected or missing files: {output_dir}")
        for path in expected:
            expected_hash = existing.get("outputs", {}).get(path.name, {}).get("sha256")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise DevelopmentControlError(f"completed control output is missing or changed: {path}")
        cached_summaries = _load_control_summary_csv(control_summary_path)
        try:
            v2 = next(row for row in cached_summaries if row.control == "sentiment_v2")
        except StopIteration as exc:
            raise DevelopmentControlError("hashed control summary has no candidate row") from exc
        acceptance = _load_acceptance_csv(acceptance_path)
        if (
            canonical_json(existing.get("v2_summary")) != canonical_json(asdict(v2))
            or canonical_json(existing.get("acceptance")) != canonical_json([asdict(row) for row in acceptance])
        ):
            raise DevelopmentControlError("control manifest decision fields disagree with hashed outputs")
        return DevelopmentControlRun(experiment_id, output_dir, report_path, manifest_path, v2, acceptance, True)
    if output_dir.exists():
        raise DevelopmentControlError(f"refusing to reuse incomplete control output: {output_dir}")

    opens = load_adjusted_opens_csv(price_path)
    symbols = tuple(sorted({row.symbol for row in opens}))
    all_sessions = tuple(sorted({row.session for row in opens}))
    development_start = min(config_session(report_manifest))
    sessions = tuple(session for session in all_sessions if development_start <= session < evaluation_start)
    returns = calculate_open_to_open_returns(opens)
    events = read_jsonl(events_path)
    scores = read_jsonl(scores_path)
    if scores_path_override is not None:
        if any(not str(event.get("eligible_execution_session") or "") for event in events):
            raise DevelopmentControlError("source events contain a missing eligible execution session")
        development_event_ids = [
            str(event.get("event_id") or "") for event in events if str(event.get("eligible_execution_session") or "") < evaluation_start
        ]
        score_event_ids = [str(score.get("event_id") or "") for score in scores]
        if (
            any(not event_id for event_id in development_event_ids + score_event_ids)
            or len(development_event_ids) != len(set(development_event_ids))
            or len(score_event_ids) != len(set(score_event_ids))
            or set(development_event_ids) != set(score_event_ids)
        ):
            raise DevelopmentControlError("model-only event and score identities do not match exactly")
    analysis_scores = select_control_scores(scores, aggregation_mode)
    folds = read_json(folds_path)["folds"]
    signals = aggregate_stock_session_scores(events, analysis_scores, sessions)
    summaries: list[ControlSummary] = []
    all_folds: list[ControlFoldResult] = []
    all_stocks: list[StockFoldOutcome] = []
    controls: tuple[ControlName, ...] = (
        "sentiment_v2",
        "refreshing_signal",
        "news_timing_long",
        "inverted_sentiment",
    )
    for control in controls:
        fold_rows, stock_rows = _evaluate_control(
            control,
            signals,
            returns,
            sessions,
            symbols,
            folds,
            evaluation_start,
            control_spec,
            keep_stock_rows=True,
        )
        all_folds.extend(fold_rows)
        all_stocks.extend(stock_rows)
        summaries.append(_summarize(control, fold_rows, control_spec.stability_penalty))
    summary_by_control = {row.control: row for row in summaries}
    v2 = summary_by_control["sentiment_v2"]

    shuffles: list[ShuffleResult] = []
    for replication in range(control_spec.shuffle_replications):
        shuffle_seed = control_spec.seed + replication
        shuffled_signals = shuffle_scores_within_stock(events, analysis_scores, sessions, seed=shuffle_seed)
        shuffle_folds, _ = _evaluate_control(
            "sentiment_v2",
            shuffled_signals,
            returns,
            sessions,
            symbols,
            folds,
            evaluation_start,
            control_spec,
            keep_stock_rows=False,
        )
        summary = _summarize("sentiment_v2", shuffle_folds, control_spec.stability_penalty)
        shuffles.append(
            ShuffleResult(
                replication=replication,
                seed=shuffle_seed,
                objective=summary.objective,
                median_fold_market_adjusted_return=summary.median_fold_market_adjusted_return,
                fold_return_iqr=summary.fold_return_iqr,
                mean_active_fraction=summary.mean_active_fraction,
                mean_supported_stocks=summary.mean_supported_stocks,
                positive_folds=summary.positive_folds,
            )
        )
    acceptance = _acceptance(v2, summary_by_control, shuffles, control_spec)
    row_counts = {
        "control_summaries": len(summaries),
        "control_fold_results": len(all_folds),
        "stock_fold_results": len(all_stocks),
        "shuffle_replications": len(shuffles),
    }
    if aggregation_mode != "all_scored_events":
        row_counts.update(
            {
                "source_score_rows": len(scores),
                "aggregation_score_rows": len(analysis_scores),
                "stock_session_signals": len(signals),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    control_figure.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(control_summary_path, _csv(summaries))
    atomic_write_text(fold_path, _csv(all_folds))
    atomic_write_text(stock_path, _csv(all_stocks))
    atomic_write_text(shuffle_path, _csv(shuffles))
    atomic_write_text(acceptance_path, _csv(acceptance))
    atomic_write_text(control_figure, _control_svg(summaries))
    atomic_write_text(shuffle_figure, _shuffle_svg(v2, shuffles))
    atomic_write_text(
        report_path,
        _report(
            experiment_id,
            control_spec,
            summaries,
            all_folds,
            shuffles,
            acceptance,
            candidate_label,
            aggregation_mode,
        ),
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": experiment_id,
        "identity_sha256": identity_hash,
        "identity": identity_payload,
        "v2_summary": asdict(v2),
        "acceptance": [asdict(row) for row in acceptance],
        "row_counts": row_counts,
        "outputs": {
            path.name: {"path": path.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in expected
        },
        "warning": (
            "Development controls only; evaluation rows were loaded only as panel metadata and were excluded from "
            "every return interval and decision."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return DevelopmentControlRun(experiment_id, output_dir, report_path, manifest_path, v2, acceptance, False)


def _csv(rows: Sequence[Any]) -> str:
    if not rows:
        return ""
    payloads = [asdict(row) for row in rows]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(payloads[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(payloads)
    return buffer.getvalue()


def _load_control_summary_csv(path: Path) -> tuple[ControlSummary, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        summaries = tuple(
            ControlSummary(
                control=row["control"],  # type: ignore[arg-type]
                objective=float(row["objective"]),
                median_fold_market_adjusted_return=float(row["median_fold_market_adjusted_return"]),
                fold_return_iqr=float(row["fold_return_iqr"]),
                median_fold_raw_return=float(row["median_fold_raw_return"]),
                median_fold_excess_vs_buy_and_hold=float(row["median_fold_excess_vs_buy_and_hold"]),
                mean_active_fraction=float(row["mean_active_fraction"]),
                mean_supported_stocks=float(row["mean_supported_stocks"]),
                mean_turnover=float(row["mean_turnover"]),
                positive_folds=int(row["positive_folds"]),
                long_positive_folds=int(row["long_positive_folds"]),
                short_positive_folds=int(row["short_positive_folds"]),
            )
            for row in rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DevelopmentControlError("hashed control summary artifact is invalid") from exc
    if not summaries:
        raise DevelopmentControlError("hashed control summary artifact is empty")
    return summaries


def _load_acceptance_csv(path: Path) -> tuple[AcceptanceCheck, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("passed") not in {"True", "False"} for row in rows):
        raise DevelopmentControlError("hashed acceptance artifact is empty or invalid")
    try:
        return tuple(
            AcceptanceCheck(
                check=row["check"],
                passed=row["passed"] == "True",
                observed=row["observed"],
                requirement=row["requirement"],
            )
            for row in rows
        )
    except (KeyError, TypeError) as exc:
        raise DevelopmentControlError("hashed acceptance artifact is invalid") from exc


def _report(
    experiment_id: str,
    spec: DevelopmentControlSpec,
    summaries: Sequence[ControlSummary],
    folds: Sequence[ControlFoldResult],
    shuffles: Sequence[ShuffleResult],
    acceptance: Sequence[AcceptanceCheck],
    candidate_label: str = "sentiment v2",
    aggregation_mode: ScoreAggregationMode = "all_scored_events",
) -> str:
    by_control = {row.control: row for row in summaries}
    v2 = by_control["sentiment_v2"]
    null = np.asarray([row.objective for row in shuffles], dtype=float)
    shuffle_p = (1 + int(np.sum(null >= v2.objective))) / (len(shuffles) + 1)
    null_95 = float(np.quantile(null, 0.95))
    passed = sum(row.passed for row in acceptance)
    aggregation_text = (
        "stock-session mean over eligible non-zero model events only"
        if aggregation_mode == "eligible_nonzero_events"
        else "stock-session mean"
    )
    timing_text = (
        "Eligible-event timing control goes long after a candidate-eligible stock-session and ignores direction."
        if aggregation_mode == "eligible_nonzero_events"
        else "News-timing control goes long after any screened stock-session news and ignores sentiment."
    )
    lines = [
        f"# Development controls for {candidate_label}: {experiment_id}",
        "",
        "> All fits, controls, shuffles, and returns stop before the previously observed evaluation boundary.",
        "",
        "## Material Passport",
        "",
        f"- Experiment ID: `{experiment_id}`",
        "- Type: deterministic controlled development experiment",
        "- Status: completed",
        "- Verification status: ANALYZED until deterministic replay and full tests pass",
        "",
        "## Frozen rules",
        "",
        f"- Candidate ({candidate_label}): {aggregation_text}, `|score| >= {spec.threshold:g}`, "
        f"non-refreshing {spec.hold_sessions}-session hold.",
        "- Same-direction signals during a live hold are ignored; an opposite strong signal reverses and resets the clock.",
        "- Refresh control restarts the clock on same-direction signals.",
        f"- {timing_text}",
        "- Inverted control reverses every qualifying sentiment direction.",
        f"- Null: `{spec.shuffle_replications}` within-stock event-label shuffles, seed sequence starting `{spec.seed}`.",
        f"- Costs: `{spec.cost_bps_per_side:g}` bps per side; primary objective includes the stability penalty.",
        "",
        "## Control comparison",
        "",
        (
            "| Rule | Objective | Median adjusted | Positive folds | Activity | Stocks | Turnover | "
            "Median excess vs B&H | Long/short positive folds |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda item: (-item.objective, item.control)):
        lines.append(
            f"| {row.control} | {row.objective:.3%} | {row.median_fold_market_adjusted_return:.3%} | "
            f"{row.positive_folds}/3 | {row.mean_active_fraction:.1%} | {row.mean_supported_stocks:.1f} | "
            f"{row.mean_turnover:.2f}x | {row.median_fold_excess_vs_buy_and_hold:.3%} | "
            f"{row.long_positive_folds}/3 / {row.short_positive_folds}/3 |"
        )
    lines.extend(
        [
            "",
            "![Control objectives](figures/control_objectives.svg)",
            "",
            "## Shuffled-label placebo",
            "",
            f"- Observed candidate objective: `{v2.objective:.3%}`.",
            f"- Shuffled-null 95th percentile: `{null_95:.3%}`.",
            f"- One-sided randomization p-value: `{shuffle_p:.4f}`.",
            "",
            "![Shuffled-label null](figures/shuffled_null.svg)",
            "",
            "## Pre-declared acceptance checks",
            "",
            f"The candidate passed **{passed}/{len(acceptance)}** checks.",
            "",
            "| Check | Result | Observed | Requirement |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in acceptance:
        lines.append(f"| {check.check} | {'PASS' if check.passed else 'FAIL'} | {check.observed} | {check.requirement} |")
    lines.extend(
        [
            "",
            "## Fold detail",
            "",
            "| Rule | Fold | Validation | Median adjusted | Median raw | Excess vs B&H | Activity | Stocks | Long mean | Short mean |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold_result in sorted(folds, key=lambda item: (item.control, item.fold_index)):
        long_mean = "n/a" if fold_result.mean_long_gross_return is None else f"{fold_result.mean_long_gross_return:.3%}"
        short_mean = "n/a" if fold_result.mean_short_gross_return is None else f"{fold_result.mean_short_gross_return:.3%}"
        lines.append(
            f"| {fold_result.control} | {fold_result.fold_index} | "
            f"{fold_result.validation_start} to {fold_result.validation_end} | "
            f"{fold_result.median_stock_market_adjusted_net_return:.3%} | "
            f"{fold_result.median_stock_net_return:.3%} | {fold_result.median_excess_vs_buy_and_hold:.3%} | "
            f"{fold_result.active_fraction:.1%} | {fold_result.supported_stocks} | "
            f"{long_mean} | {short_mean} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Passing a shuffled-label test would show association beyond this placebo, not deployable or causal alpha.",
            "- Same-date stocks and overlapping holds are dependent; the fold-level objective is the primary comparison unit.",
            "- The news-timing control is long-only and intentionally exposes market-direction contamination.",
            "- The 33-company universe is fixed and can contain survivorship and selection bias.",
            "- No result here authorizes reopening the already-observed evaluation block.",
            "",
            "### Statistical fallacy scan — 11/11 checked",
            "",
            "- Simpson's paradox remains open because no canonical sector field is present.",
            "- No ecological or individual-investor inference is made.",
            "- Berkson/selection bias remains possible in the fixed company universe.",
            "- No post-signal collider controls were introduced.",
            "- Base rates are shown through activity, support, and side counts.",
            "- No stocks were selected from extreme outcomes, limiting regression-to-mean selection.",
            "- Survivorship bias remains possible in the contemporary universe.",
            (
                "- The full control set and all 500 shuffles are retained, limiting selective reporting while not removing "
                "look-elsewhere risk."
            ),
            "- The rule and acceptance gates are explicit but were not prospectively preregistered, so forking-path caution remains.",
            "- Results are associational, not causal.",
            "- Timestamps precede returns, but observational reverse-causality and confounding concerns remain.",
            "",
        ]
    )
    return "\n".join(lines)


def _control_svg(rows: Sequence[ControlSummary]) -> str:
    ordered = sorted(rows, key=lambda row: (row.objective, row.control))
    width, height = 900, 420
    left, right, top, bottom = 190, 35, 55, 45
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [row.objective for row in ordered] + [0.0]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.08, 0.0005)
    low, high = low - padding, high + padding
    span = high - low
    zero_x = left + plot_width * (0 - low) / span
    row_height = plot_height / len(ordered)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="27" font-family="sans-serif" font-size="19">Development control objectives</text>',
        (
            f'<text x="{left}" y="46" font-family="sans-serif" font-size="12" fill="#555">'
            "Higher is better; market-adjusted and stability-penalized</text>"
        ),
        f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{height - bottom}" stroke="#555"/>',
    ]
    for index, row in enumerate(ordered):
        value_x = left + plot_width * (row.objective - low) / span
        y = top + index * row_height + row_height * 0.2
        pieces.append(
            f'<text x="{left - 8}" y="{y + row_height * 0.42:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{html.escape(row.control)}</text>'
        )
        pieces.append(
            f'<rect x="{min(zero_x, value_x):.2f}" y="{y:.2f}" width="{max(1, abs(value_x - zero_x)):.2f}" '
            f'height="{row_height * 0.58:.2f}" fill="{"#2864a8" if row.objective >= 0 else "#e09045"}"/>'
        )
    pieces.extend(
        [
            f'<text x="{left}" y="{height - 12}" font-family="sans-serif" font-size="11">{low:.2%}</text>',
            f'<text x="{width - right}" y="{height - 12}" text-anchor="end" font-family="sans-serif" font-size="11">{high:.2%}</text>',
            "</svg>\n",
        ]
    )
    return "\n".join(pieces)


def _shuffle_svg(v2: ControlSummary, rows: Sequence[ShuffleResult]) -> str:
    values = np.asarray([row.objective for row in rows], dtype=float)
    counts, edges = np.histogram(values, bins=24)
    width, height = 900, 430
    left, right, top, bottom = 75, 35, 55, 55
    plot_width, plot_height = width - left - right, height - top - bottom
    low, high = float(edges[0]), float(edges[-1])
    if v2.objective < low:
        low = v2.objective
    if v2.objective > high:
        high = v2.objective
    span = max(high - low, 0.001)
    maximum = max(int(counts.max()), 1)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="27" font-family="sans-serif" font-size="19">Within-stock shuffled-label null</text>',
        (
            f'<text x="{left}" y="46" font-family="sans-serif" font-size="12" fill="#555">'
            "500 deterministic replications; vertical line is observed candidate</text>"
        ),
    ]
    for index, count in enumerate(counts):
        edge_low, edge_high = float(edges[index]), float(edges[index + 1])
        x = left + plot_width * (edge_low - low) / span
        next_x = left + plot_width * (edge_high - low) / span
        bar_height = plot_height * int(count) / maximum
        pieces.append(
            f'<rect x="{x:.2f}" y="{top + plot_height - bar_height:.2f}" width="{max(1, next_x - x - 1):.2f}" '
            f'height="{bar_height:.2f}" fill="#9a9a9a"/>'
        )
    observed_x = left + plot_width * (v2.objective - low) / span
    pieces.extend(
        [
            f'<line x1="{observed_x:.2f}" y1="{top}" x2="{observed_x:.2f}" y2="{top + plot_height}" stroke="#2864a8" stroke-width="3"/>',
            f'<text x="{left}" y="{height - 15}" font-family="sans-serif" font-size="11">{low:.2%}</text>',
            f'<text x="{width - right}" y="{height - 15}" text-anchor="end" font-family="sans-serif" font-size="11">{high:.2%}</text>',
            "</svg>\n",
        ]
    )
    return "\n".join(pieces)
