"""Stage orchestration for the separate historical strategy-research pipeline."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import date, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from ..artifact_io import canonical_json, read_json, sha256_file, sha256_text
from ..prompts import load_prompts
from ..providers import make_llm_client, normalize_provider
from .artifacts import StageManifestStore, write_immutable_json, write_immutable_jsonl, write_immutable_text
from .config import (
    OutputSettings,
    RunPaths,
    StrategyConfigurationError,
    StrategyResearchConfig,
    compute_run_identity,
    resolve_run_paths,
)
from .diagnostics import (
    PortfolioMetrics,
    StockDiagnostic,
    calculate_portfolio_metrics,
    calculate_stock_diagnostics,
    contribution_concentration,
    leave_one_stock_out_returns,
    returns_by_time_block,
)
from .inference import PairedBootstrapResult, paired_block_bootstrap
from .ledger import DailyLedgerRow, assert_no_boundary_crossing, evaluation_rows, run_open_to_open_ledger
from .market import (
    AdjustedOpen,
    LaggedVolatility,
    OpenToOpenReturn,
    calculate_lagged_volatility,
    calculate_open_to_open_returns,
    load_adjusted_opens_csv,
    price_coverage,
    validate_exchange_sessions,
    volatility_mapping,
)
from .portfolio import PortfolioConstraints, TargetPortfolio, project_target_weights
from .prompts import strategy_prompt
from .reporting import (
    cumulative_return_svg,
    exposure_turnover_svg,
    leakage_payload,
    render_summary,
)
from .schemas import ScoreRecord, StrategyEvent, load_score_records, load_strategy_events, write_score_records, write_strategy_events
from .score_mapping import map_score_label, score_mapping_hash
from .scoring import (
    ScoringIdentity,
    import_completed_score_cache,
    inspect_score_cache,
    score_cache_key,
    score_events,
)
from .sources import LsegEventSettings, build_strategy_events, write_event_build_artifacts
from .state import (
    ScaleEstimate,
    StateAction,
    StateConfig,
    StateEvent,
    StateTransition,
    apply_state_scales,
    build_state_table,
    estimate_state_scales,
)
from .tuning import (
    CandidateGuardrails,
    CandidateResult,
    ChronologicalFold,
    FoldMetric,
    StrategyCandidate,
    annualized_sharpe,
    build_candidate_grid,
    build_chronological_split,
    build_expanding_folds,
    evaluate_candidate,
    select_candidate,
)

PipelineStage = Literal["events", "scores", "tuning", "state", "backtest", "report"]


class StrategyPipelineError(RuntimeError):
    """Raised when a strategy stage cannot be completed safely."""


@dataclass(frozen=True)
class PricePanel:
    rows: tuple[AdjustedOpen, ...]
    returns: tuple[OpenToOpenReturn, ...]
    sessions: tuple[str, ...]
    symbols: tuple[str, ...]
    panel_sha256: str
    manifest_sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DryRunReport:
    resolved_run_id: str
    source: str
    event_count: int | None
    event_date_range: tuple[str, str] | None
    unique_symbols: int | None
    expected_score_calls: int | None
    cached_scores: int | None
    missing_scores: int | None
    score_success_rate: float | None
    price_rows: int | None
    price_symbols: int | None
    price_date_range: tuple[str, str] | None
    development_sessions: int | None
    evaluation_sessions: int | None
    feasible_folds: int | None
    executable_events: int | None
    supported_event_symbols: int | None
    tuning_candidates: int
    derived_dir: str
    results_dir: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class StrategyInputs:
    events: tuple[StrategyEvent, ...]
    scores: tuple[ScoreRecord, ...]
    prices: PricePanel
    state_events: tuple[StateEvent, ...]
    decision_sessions: tuple[str, ...]
    symbols: tuple[str, ...]
    development_sessions: tuple[str, ...]
    evaluation_sessions: tuple[str, ...]
    event_build_counts: dict[str, int]
    attrition: dict[str, int]


@dataclass(frozen=True)
class VariantRun:
    variant: str
    candidate: StrategyCandidate | None
    transitions: tuple[StateTransition, ...]
    scale_estimate: ScaleEstimate
    actions: tuple[StateAction, ...]
    volatilities: tuple[LaggedVolatility, ...]
    targets: tuple[TargetPortfolio, ...]
    ledger: tuple[DailyLedgerRow, ...]
    development_metrics: PortfolioMetrics
    evaluation_metrics: PortfolioMetrics


@dataclass(frozen=True)
class TuningRun:
    folds: tuple[ChronologicalFold, ...]
    candidates: tuple[CandidateResult, ...]
    selected: CandidateResult


@dataclass(frozen=True)
class ScoreProgress:
    total_events: int
    canary_eligible_events: int
    successful_scores: int
    missing_scores: int
    calls_made: int
    complete: bool


@dataclass(frozen=True)
class PipelineResult:
    paths: RunPaths
    events_path: Path
    scores_path: Path
    selected_config_path: Path | None
    evaluation_metrics_path: Path | None
    summary_path: Path | None
    reused_stages: tuple[str, ...]
    score_progress: ScoreProgress | None = None


def resolved_scoring_identity(config: StrategyResearchConfig) -> ScoringIdentity:
    """Resolve and cross-check the versioned prompt used by a scoring run."""

    _verify_competence_evidence(config)
    built_in = strategy_prompt(config.scoring.scheme)
    if config.scoring.prompt_id != built_in.prompt_id:
        raise StrategyPipelineError(
            f"scoring.prompt_id {config.scoring.prompt_id!r} does not match the {config.scoring.scheme} v1 prompt "
            f"{built_in.prompt_id!r}"
        )
    configured = load_prompts(config.scoring.prompts_path).get(config.scoring.prompt_id)
    if configured is None:
        raise StrategyPipelineError(f"configured strategy prompt does not exist: {config.scoring.prompt_id}")
    if configured != built_in:
        raise StrategyPipelineError(
            "configured strategy prompt content does not match the code-locked v1 prompt; create a new prompt ID and pipeline identity"
        )
    return ScoringIdentity(
        scheme=config.scoring.scheme,
        provider=config.scoring.provider,
        model_id=config.scoring.model,
        model_digest=config.scoring.model_digest,
        endpoint=config.scoring.endpoint,
        temperature=config.scoring.temperature,
        sample_count=config.scoring.samples,
        max_completion_tokens=config.scoring.max_completion_tokens,
        ollama_think=config.scoring.ollama_think,
        retries=config.scoring.retries,
        agreement_conditioning=config.agreement_conditioning.enabled,
        prompt_config=configured,
    )


def _verify_competence_evidence(config: StrategyResearchConfig) -> str | None:
    """Validate the frozen benchmark evidence used to choose a formal scorer."""

    path = config.scoring.competence_evidence_path
    experiment_id = config.scoring.competence_experiment_id
    if path is None or experiment_id is None:
        return None
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise StrategyPipelineError(f"cannot read scorer competence evidence {path}: {exc}") from exc
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise StrategyPipelineError(f"scorer competence evidence contains no experiment registry: {path}")
    matches = [item for item in experiments if isinstance(item, dict) and item.get("id") == experiment_id]
    if len(matches) != 1:
        raise StrategyPipelineError(
            f"scorer competence experiment {experiment_id!r} must occur exactly once in {path}"
        )
    experiment = matches[0]
    if experiment.get("status") != "completed":
        raise StrategyPipelineError(f"scorer competence experiment {experiment_id!r} is not completed")
    models = experiment.get("models")
    model_ids = models.get("ids") if isinstance(models, dict) else None
    if not isinstance(model_ids, list) or config.scoring.model not in model_ids:
        raise StrategyPipelineError(
            f"scorer competence experiment {experiment_id!r} does not include model {config.scoring.model!r}"
        )
    return sha256_file(path)


def resolve_pipeline_paths(config: StrategyResearchConfig) -> RunPaths:
    scoring = resolved_scoring_identity(config)
    identity = compute_run_identity(
        config,
        additional_inputs={
            "event_rules_hash": _event_settings(config).identity_hash,
            "resolved_prompt_hash": scoring.prompt.prompt_hash,
            "score_mapping_hash": score_mapping_hash(config.scoring.scheme),
            "resolved_scoring_config_hash": scoring.config_hash,
        },
    )
    return resolve_run_paths(config, identity)


def inspect_pipeline(config: StrategyResearchConfig) -> DryRunReport:
    """Inspect a full run without creating clients, directories, or artifacts."""

    blockers = list(config.readiness_issues(check_files=True))
    warnings: list[str] = []
    try:
        paths = resolve_pipeline_paths(config)
    except Exception as exc:
        identity = compute_run_identity(config)
        try:
            paths = resolve_run_paths(config, identity)
        except StrategyConfigurationError:
            paths = resolve_run_paths(replace(config, outputs=OutputSettings()), identity)
        blockers.append(str(exc))

    events: tuple[StrategyEvent, ...] = ()
    event_count: int | None = None
    event_range: tuple[str, str] | None = None
    unique_symbols: int | None = None
    if config.data.events_path is not None and config.data.events_path.is_file():
        try:
            events = load_strategy_events(config.data.events_path)
            event_count = len(events)
            unique_symbols = len({event.symbol for event in events})
            if events:
                dates = [event.version_created_utc.date().isoformat() for event in events]
                event_range = min(dates), max(dates)
        except Exception as exc:
            blockers.append(f"cannot load prebuilt events: {exc}")
    elif config.data.corpus_manifest is not None and config.data.corpus_manifest.is_file():
        try:
            result = build_strategy_events(
                config.data.corpus_manifest,
                settings=_event_settings(config),
            )
            if config.data.fail_on_invalid_timestamp and result.attrition.get("invalid_timestamp", 0):
                blockers.append(
                    "formal event construction encountered invalid timestamps while "
                    "data.fail_on_invalid_timestamp=true: "
                    f"{result.attrition['invalid_timestamp']} rows"
                )
            events = result.events
            event_count = len(events)
            unique_symbols = len({event.symbol for event in events})
            if events:
                dates = [event.version_created_utc.date().isoformat() for event in events]
                event_range = min(dates), max(dates)
        except Exception as exc:
            blockers.append(f"cannot inspect canonical events: {exc}")

    expected_calls: int | None = None
    cached_scores: int | None = None
    missing_scores: int | None = None
    score_success_rate: float | None = None
    fixed_successful_event_ids: set[str] | None = None
    if events:
        try:
            scoring_identity = resolved_scoring_identity(config)
            cache_inspection = inspect_score_cache(events, scoring_identity, paths.derived_stage("scores") / "cache")
            expected_calls = cache_inspection.expected_score_calls
            cached_scores = cache_inspection.cache_hits
            missing_scores = len(cache_inspection.missing_event_ids)
            if cache_inspection.expected_score_calls == 0:
                fixed_successful_event_ids = {event.event_id for event in events}
            if config.scoring.scores_path is not None and config.scoring.scores_path.is_file():
                fixture_records = load_score_records(config.scoring.scores_path)
                validated = _validate_scores(events, fixture_records, config, scoring_identity)
                valid_ids = {record.event_id for record in validated if record.status == "success"}
                fixed_successful_event_ids = valid_ids
                cached_scores = len(valid_ids)
                missing_scores = len(events) - cached_scores
                expected_calls = 0
            if cached_scores is not None:
                score_success_rate = cached_scores / len(events)
                if score_success_rate < config.scoring.minimum_success_rate:
                    blockers.append(
                        f"score success rate {score_success_rate:.3f} is below configured minimum "
                        f"{config.scoring.minimum_success_rate:.3f}"
                    )
        except Exception as exc:
            blockers.append(f"cannot inspect score identities: {exc}")

    price_rows: int | None = None
    price_symbols: int | None = None
    price_range: tuple[str, str] | None = None
    development_sessions: int | None = None
    evaluation_sessions: int | None = None
    feasible_folds: int | None = None
    executable_events: int | None = None
    supported_event_symbols: int | None = None
    if config.prices.panel_path is not None and config.prices.panel_path.is_file() and config.prices.manifest_path is not None:
        try:
            panel = load_verified_price_panel(config)
            price_rows = len(panel.rows)
            price_symbols = len(panel.symbols)
            price_range = (panel.sessions[0], panel.sessions[-1]) if panel.sessions else None
            if config.run.evaluation_start is not None and events:
                panel_symbols = set(panel.symbols)
                return_sessions = {row.session for row in panel.returns}
                scored_or_pending_events = [
                    event
                    for event in events
                    if fixed_successful_event_ids is None or event.event_id in fixed_successful_event_ids
                ]
                executable = [
                    event
                    for event in scored_or_pending_events
                    if event.symbol in panel_symbols
                    and event.eligible_execution_session.isoformat() in return_sessions
                ]
                executable_events = len(executable)
                supported_event_symbols = len({event.symbol for event in executable})
                if not executable:
                    blockers.append("no scored event has executable price coverage")
                else:
                    first_session = min(
                        event.eligible_execution_session.isoformat()
                        for event in executable
                    )
                    if config.run.mode == "formal":
                        insufficient_warmup = _predecision_warmup_gaps(config, panel, first_session)
                        if insufficient_warmup:
                            raise StrategyPipelineError(
                                "formal price panel lacks pre-decision volatility warm-up for: "
                                + ", ".join(insufficient_warmup[:10])
                            )
                    decision_sessions = tuple(
                        session
                        for session in sorted({row.session for row in panel.returns})
                        if session >= first_session
                    )
                    split = build_chronological_split(
                        decision_sessions,
                        config.run.evaluation_start.isoformat(),
                    )
                    if config.run.evaluation_start.isoformat() not in decision_sessions:
                        raise StrategyPipelineError(
                            "evaluation_start must be an executable XNYS decision session"
                        )
                    development_sessions = len(split.development_sessions)
                    evaluation_sessions = len(split.evaluation_sessions)
                    folds = build_expanding_folds(
                        split.development_sessions,
                        requested_folds=config.tuning.folds,
                        minimum_training_sessions=config.tuning.minimum_training_sessions,
                        requested_validation_sessions=config.tuning.validation_sessions,
                        minimum_validation_sessions=config.tuning.minimum_validation_sessions,
                    )
                    feasible_folds = len(folds)
                    if fixed_successful_event_ids is None:
                        warnings.append(
                            "split feasibility uses the screened event universe before hosted scoring exclusions"
                        )
        except Exception as exc:
            blockers.append(f"cannot verify price panel or chronological split: {exc}")

    return DryRunReport(
        resolved_run_id=paths.identity.resolved_run_id,
        source=config.data.source,
        event_count=event_count,
        event_date_range=event_range,
        unique_symbols=unique_symbols,
        expected_score_calls=expected_calls,
        cached_scores=cached_scores,
        missing_scores=missing_scores,
        score_success_rate=score_success_rate,
        price_rows=price_rows,
        price_symbols=price_symbols,
        price_date_range=price_range,
        development_sessions=development_sessions,
        evaluation_sessions=evaluation_sessions,
        feasible_folds=feasible_folds,
        executable_events=executable_events,
        supported_event_symbols=supported_event_symbols,
        tuning_candidates=config.tuning.candidate_count,
        derived_dir=paths.derived_dir.as_posix(),
        results_dir=paths.results_dir.as_posix(),
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def load_verified_price_panel(config: StrategyResearchConfig) -> PricePanel:
    """Load a hash-manifested price panel and fail closed on field ambiguity."""

    panel_path = config.prices.panel_path
    manifest_path = config.prices.manifest_path
    if panel_path is None or manifest_path is None:
        raise StrategyPipelineError("both prices.panel_path and prices.manifest_path are required")
    if config.prices.execution_field != "adjusted_open" or config.prices.return_convention != "open_to_open":
        raise StrategyPipelineError("pipeline v1 implements adjusted_open/open_to_open only; close execution requires a separate version")
    if not manifest_path.is_file():
        raise StrategyPipelineError(f"price manifest does not exist: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise StrategyPipelineError(f"price manifest is not completed: {manifest_path}")
    if manifest.get("calendar") != config.data.exchange_calendar:
        raise StrategyPipelineError(
            f"price manifest calendar must be {config.data.exchange_calendar}: {manifest_path}"
        )
    if manifest.get("timezone") != config.data.exchange_timezone:
        raise StrategyPipelineError(
            f"price manifest timezone must be {config.data.exchange_timezone}: {manifest_path}"
        )
    if manifest.get("execution_field") != "adjusted_open":
        raise StrategyPipelineError("price manifest must explicitly declare execution_field='adjusted_open'")
    if manifest.get("return_convention") != "open_to_open":
        raise StrategyPipelineError("price manifest must explicitly declare return_convention='open_to_open'")
    if manifest.get("adjustment_supported") is not True:
        raise StrategyPipelineError("price manifest must explicitly set adjustment_supported=true")
    if manifest.get("split_adjusted") is not True:
        raise StrategyPipelineError("price manifest must explicitly set split_adjusted=true")
    if manifest.get("dividend_adjusted") is not False:
        raise StrategyPipelineError("price manifest must explicitly set dividend_adjusted=false")
    adjustment_convention = str(manifest.get("adjustment_convention") or "").lower()
    if "split-adjusted" not in adjustment_convention or "dividend" not in adjustment_convention:
        raise StrategyPipelineError(
            "price manifest adjustment_convention must state split adjustment and dividend treatment"
        )
    provenance = manifest.get("adjustment_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise StrategyPipelineError("price manifest must record non-empty adjustment_provenance")
    price_source = manifest.get("source")
    if not isinstance(price_source, str) or not price_source.strip():
        raise StrategyPipelineError("price manifest must record a non-empty source")
    expected_hash = _manifest_panel_hash(manifest, panel_path)
    if not panel_path.is_file():
        raise StrategyPipelineError(f"price panel does not exist: {panel_path}")
    actual_hash = sha256_file(panel_path)
    if actual_hash != expected_hash:
        raise StrategyPipelineError(
            f"price panel hash mismatch for {panel_path}: expected {expected_hash}, got {actual_hash}"
        )
    rows = load_adjusted_opens_csv(
        panel_path,
        price_column="adjusted_open",
        adjustment_supported=True,
    )
    if not rows:
        raise StrategyPipelineError("price panel is empty")
    validate_exchange_sessions(rows, calendar_name=config.data.exchange_calendar)
    sessions = tuple(sorted({row.session for row in rows}))
    symbols = tuple(sorted({row.symbol for row in rows}))
    import exchange_calendars as xcals

    calendar = xcals.get_calendar(config.data.exchange_calendar)
    complete_sessions = tuple(
        str(session.date())
        for session in calendar.sessions_in_range(sessions[0], sessions[-1])
    )
    if sessions != complete_sessions:
        missing_sessions = sorted(set(complete_sessions) - set(sessions))
        raise StrategyPipelineError(
            "price panel session union is not a complete XNYS spine; missing: "
            + ", ".join(missing_sessions[:10])
        )
    declared_symbols = manifest.get("symbols")
    if not isinstance(declared_symbols, list) or not all(
        isinstance(symbol, str) and symbol.strip() for symbol in declared_symbols
    ):
        raise StrategyPipelineError("price manifest must contain a non-empty symbols list")
    normalized_declared_symbols = tuple(sorted({symbol.strip().upper() for symbol in declared_symbols}))
    if normalized_declared_symbols != symbols or len(normalized_declared_symbols) != len(declared_symbols):
        raise StrategyPipelineError("price manifest symbols do not exactly match the panel symbols")
    observed_cells = {(row.symbol, row.session) for row in rows}
    missing_cells = [
        (symbol, session)
        for symbol in symbols
        for session in complete_sessions
        if (symbol, session) not in observed_cells
    ]
    if missing_cells:
        preview = ", ".join(f"{symbol}@{session}" for symbol, session in missing_cells[:10])
        raise StrategyPipelineError(
            "price panel is not a complete symbol-by-XNYS-session panel; missing: " + preview
        )
    if config.run.mode == "formal" and len(symbols) != 33:
        raise StrategyPipelineError(
            f"formal price panel must contain exactly 33 companies; found {len(symbols)}"
        )
    if config.run.mode == "formal":
        if config.data.corpus_manifest is None:
            raise StrategyPipelineError("formal symbol verification requires data.corpus_manifest")
        corpus_manifest = read_json(config.data.corpus_manifest)
        companies = (corpus_manifest.get("config") or {}).get("companies") or []
        corpus_symbols = tuple(
            sorted(
                {
                    str(company.get("symbol") or "").strip().upper()
                    for company in companies
                    if isinstance(company, dict) and str(company.get("symbol") or "").strip()
                }
            )
        )
        if len(corpus_symbols) != 33 or corpus_symbols != symbols:
            raise StrategyPipelineError(
                "formal price-panel symbols do not exactly match the 33-company canonical corpus universe"
            )
    declared_symbol_count = manifest.get("symbol_count")
    if declared_symbol_count is not None and declared_symbol_count != len(symbols):
        raise StrategyPipelineError(
            f"price manifest symbol_count={declared_symbol_count!r} does not match panel count {len(symbols)}"
        )
    counts_by_symbol = Counter(row.symbol for row in rows)
    insufficient = sorted(
        symbol
        for symbol, count in counts_by_symbol.items()
        if count < config.prices.volatility_minimum_sessions + 2
    )
    if insufficient:
        raise StrategyPipelineError(
            "price panel lacks per-symbol volatility warm-up for: " + ", ".join(insufficient[:10])
        )
    global_next = {current: following for current, following in zip(sessions, sessions[1:], strict=False)}
    returns = tuple(
        row
        for row in calculate_open_to_open_returns(rows)
        if global_next.get(row.session) == row.next_session
    )
    if len(sessions) < config.prices.volatility_minimum_sessions + 2:
        raise StrategyPipelineError(
            "price panel lacks volatility warm-up: "
            f"have {len(sessions)} sessions, require at least {config.prices.volatility_minimum_sessions + 2}"
        )
    return PricePanel(
        rows=tuple(rows),
        returns=returns,
        sessions=sessions,
        symbols=symbols,
        panel_sha256=actual_hash,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
    )


def _manifest_panel_hash(manifest: Mapping[str, Any], panel_path: Path) -> str:
    direct = manifest.get("file")
    if isinstance(direct, dict) and direct.get("sha256"):
        declared = str(direct.get("path") or "")
        if declared and Path(declared).name != panel_path.name:
            raise StrategyPipelineError(
                f"price manifest declares {declared!r}, not configured panel {panel_path.name!r}"
            )
        return str(direct["sha256"])
    files = manifest.get("files")
    if isinstance(files, dict):
        for key in ("price_panel", "prices_csv", "panel"):
            entry = files.get(key)
            if isinstance(entry, dict) and entry.get("sha256"):
                declared = str(entry.get("path") or "")
                if declared and Path(declared).name != panel_path.name:
                    raise StrategyPipelineError(
                        f"price manifest declares {declared!r}, not configured panel {panel_path.name!r}"
                    )
                return str(entry["sha256"])
    raise StrategyPipelineError("price manifest does not contain a hashed panel file entry")


def _predecision_warmup_gaps(
    config: StrategyResearchConfig,
    panel: PricePanel,
    earliest_session: str,
) -> list[str]:
    """Return symbols lacking point-in-time volatility history before a decision."""

    return [
        symbol
        for symbol in panel.symbols
        if sum(
            row.next_session < earliest_session
            for row in panel.returns
            if row.symbol == symbol
        )
        < config.prices.volatility_minimum_sessions
    ]


def _labels(config: StrategyResearchConfig) -> tuple[str, ...]:
    if config.scoring.scheme == "three_class":
        return ("negative", "neutral", "positive")
    return ("very_negative", "negative", "neutral", "positive", "very_positive")


def _event_settings(config: StrategyResearchConfig) -> LsegEventSettings:
    return LsegEventSettings(
        processing_buffer_minutes=config.data.processing_buffer_minutes,
        calendar_name=config.data.exchange_calendar,
        exchange_timezone=config.data.exchange_timezone,
        overrides_path=config.data.screening_overrides_path,
    )


def _command(command: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(command or sys.argv or ("sentiment-bench", "strategy"))


def _stage_store(paths: RunPaths, stage: PipelineStage, *, repo_root: Path) -> StageManifestStore:
    return StageManifestStore(
        paths.stage_manifest(stage),
        identity=paths.identity,
        stage=stage,
        repo_root=repo_root,
    )


def _input_identities(paths: RunPaths, **extra: str) -> dict[str, str]:
    return {**paths.identity.input_identities, **dict(sorted(extra.items()))}


def build_events_stage(
    config: StrategyResearchConfig,
    paths: RunPaths,
    *,
    repo_root: Path,
    command: Sequence[str] | None = None,
) -> tuple[Path, bool]:
    """Build or validate the full unsampled strategy event universe."""

    output_dir = paths.derived_stage("events")
    events_path = output_dir / "events.jsonl"
    store = _stage_store(paths, "events", repo_root=repo_root)
    inputs = _input_identities(paths)
    inspection = store.inspect(config=config, input_identities=inputs, verify_outputs=True)
    if inspection.reusable:
        return events_path, True
    opened = store.begin(config=config, input_identities=inputs, command=_command(command))
    if config.data.source == "synthetic":
        if config.data.events_path is None:
            raise StrategyPipelineError("synthetic event stage requires data.events_path")
        events = load_strategy_events(config.data.events_path)
        write_strategy_events(events_path, list(events))
        attrition = {
            "input_records": len(events),
            "selected_events": len(events),
            "screening_rows": len(events),
        }
        screening_path, _ = write_immutable_jsonl(output_dir / "event_screening.jsonl", [])
        attrition_path, _ = write_immutable_json(
            output_dir / "event_attrition.json",
            {"source": "synthetic", "counts": attrition},
        )
        event_rules_hash = sha256_text(
            canonical_json(
                {
                    "source": "synthetic",
                    "processing_buffer_minutes": config.data.processing_buffer_minutes,
                    "calendar": config.data.exchange_calendar,
                }
            )
        )
    else:
        if config.data.corpus_manifest is None:
            raise StrategyPipelineError("LSEG event stage requires data.corpus_manifest")
        result = build_strategy_events(config.data.corpus_manifest, settings=_event_settings(config))
        if config.data.fail_on_invalid_timestamp and result.attrition.get("invalid_timestamp", 0):
            raise StrategyPipelineError(
                "formal event construction encountered invalid timestamps while data.fail_on_invalid_timestamp=true: "
                f"{result.attrition['invalid_timestamp']} rows"
            )
        events_path, screening_path, attrition_path = write_event_build_artifacts(output_dir, result)
        events = result.events
        attrition = result.attrition
        event_rules_hash = result.event_rules_hash
    store.complete(
        outputs={
            "events": events_path,
            "event_screening": screening_path,
            "event_attrition": attrition_path,
        },
        row_counts={"events": len(events)},
        exclusions={key: value for key, value in attrition.items() if key not in {"input_records", "selected_events", "screening_rows"}},
        warnings=("synthetic fixture; not research evidence",) if config.data.source == "synthetic" else (),
        deviations=(f"event_rules_hash={event_rules_hash}",),
    )
    return events_path, opened.action == "reuse"


def scores_stage(
    config: StrategyResearchConfig,
    paths: RunPaths,
    events_path: Path,
    *,
    repo_root: Path,
    command: Sequence[str] | None = None,
    allow_paid: bool = False,
    max_new_scores: int | None = None,
    cache_from: Path | None = None,
) -> tuple[Path, bool, ScoreProgress]:
    """Materialize frozen scores from a fixture or explicitly-authorized provider."""

    output_dir = paths.derived_stage("scores")
    scores_path = output_dir / "scores.jsonl"
    coverage_path = output_dir / "score_coverage.json"
    identity = resolved_scoring_identity(config)
    events = load_strategy_events(events_path)
    if max_new_scores is not None:
        if max_new_scores < 1:
            raise StrategyPipelineError("max_new_scores must be positive")
        if config.scoring.provider != "ollama" or config.scoring.scores_path is not None:
            raise StrategyPipelineError("bounded canary scoring is available only for live local Ollama runs")
        if config.run.evaluation_start is None:
            raise StrategyPipelineError(
                "bounded canary scoring requires a frozen run.evaluation_start so only development events are used"
            )
    canary_events = events
    if max_new_scores is not None:
        boundary = config.run.evaluation_start
        assert boundary is not None  # validated above; narrows the type for static checking
        canary_events = tuple(event for event in events if event.eligible_execution_session < boundary)
        if not canary_events:
            raise StrategyPipelineError("no development-period events are available for bounded canary scoring")
    inputs = _input_identities(
        paths,
        events_sha256=sha256_file(events_path),
        resolved_scoring_config=identity.config_hash,
    )
    store = _stage_store(paths, "scores", repo_root=repo_root)
    inspection = store.inspect(config=config, input_identities=inputs, verify_outputs=True)
    if inspection.reusable:
        records = load_score_records(scores_path)
        successful = sum(record.status == "success" for record in records)
        progress = ScoreProgress(
            len(events),
            len(canary_events),
            successful,
            len(events) - successful,
            0,
            True,
        )
        return scores_path, True, progress
    cache_import_path = output_dir / "cache_import.json"
    if cache_from is not None:
        cache_import = import_completed_score_cache(events, identity, cache_from, output_dir / "cache")
        write_immutable_json(cache_import_path, cache_import)
    opened = store.begin(config=config, input_identities=inputs, command=_command(command))

    if config.scoring.scores_path is not None:
        fixture_records = load_score_records(config.scoring.scores_path)
        records = _validate_scores(events, fixture_records, config, identity)
        calls_made = 0
        cache_hits = len(records)
    else:
        cache_dir = output_dir / "cache"
        cache_inspection = inspect_score_cache(events, identity, cache_dir)
        if cache_inspection.expected_score_calls and not allow_paid:
            raise StrategyPipelineError(
                f"{cache_inspection.expected_score_calls} score calls are missing; rerun `strategy score` or `strategy run` "
                "with --allow-paid after reviewing the dry-run identity"
            )
        if cache_inspection.expected_score_calls:
            provider = normalize_provider(config.scoring.provider)

            async def execute() -> Any:
                endpoint = identity.resolved_endpoint
                client = make_llm_client(
                    provider,
                    base_url=endpoint,
                    ollama_host=endpoint,
                    cerebras_base_url=endpoint,
                    structured_label_output=True,
                    ollama_think=identity.ollama_think,
                )
                async with client:
                    if provider == "ollama":
                        resolver = getattr(client, "model_digest", None)
                        if not callable(resolver):
                            raise StrategyPipelineError("Ollama client cannot verify the configured model digest")
                        actual_digest = await resolver(identity.model_id, retries=identity.retries)
                        if actual_digest != identity.model_digest:
                            raise StrategyPipelineError(
                                f"Ollama model digest mismatch for {identity.model_id!r}: "
                                f"expected {identity.model_digest!r}, got {actual_digest!r}"
                            )
                    return await score_events(
                        canary_events,
                        identity,
                        cache_dir,
                        client,
                        allow_calls=True,
                        max_new_scores=max_new_scores,
                    )

            batch = asyncio.run(execute())
        else:
            batch = asyncio.run(score_events(events, identity, cache_dir, allow_calls=False))
        calls_made = batch.calls_made
        cache_hits = batch.cache_hits

        final_cache = inspect_score_cache(events, identity, cache_dir)
        successful = final_cache.cache_hits
        missing = final_cache.expected_score_calls
        cache_hits = successful
        progress = ScoreProgress(
            total_events=len(events),
            canary_eligible_events=len(canary_events),
            successful_scores=successful,
            missing_scores=missing,
            calls_made=calls_made,
            complete=missing == 0,
        )
        if max_new_scores is not None and missing:
            return scores_path, False, progress
        completed_batch = asyncio.run(score_events(events, identity, cache_dir, allow_calls=False))
        records = _validate_scores(events, completed_batch.records, config, identity)

    event_ids = {event.event_id for event in events}
    successful_ids = {record.event_id for record in records if record.status == "success"}
    success_rate = len(successful_ids) / len(event_ids) if event_ids else 0.0
    missing_ids = sorted(event_ids - successful_ids)
    if success_rate < config.scoring.minimum_success_rate:
        raise StrategyPipelineError(
            f"score success rate {success_rate:.3f} is below configured minimum {config.scoring.minimum_success_rate:.3f}"
        )
    write_score_records(scores_path, list(records))
    coverage_path, _ = write_immutable_json(
        coverage_path,
        {
            "events": len(event_ids),
            "successful": len(successful_ids),
            "missing": len(missing_ids),
            "success_rate": success_rate,
            "missing_event_ids": missing_ids,
            "cache_hits": cache_hits,
            "calls_made": calls_made,
            "provider": identity.provider,
            "model": identity.model_id,
            "model_digest": identity.model_digest,
            "ollama_think": identity.ollama_think,
            "endpoint": identity.resolved_endpoint,
            "prompt_id": identity.prompt.prompt_id,
            "prompt_hash": identity.prompt.prompt_hash,
            "scoring_config_hash": identity.config_hash,
        },
    )
    outputs = {"scores": scores_path, "score_coverage": coverage_path}
    if cache_import_path.is_file():
        outputs["score_cache_import"] = cache_import_path
    store.complete(
        outputs=outputs,
        row_counts={"scores": len(records), "successful_scores": len(successful_ids)},
        exclusions={"missing_score": len(missing_ids)},
        warnings=("fixture scores; no model call was made",) if config.scoring.scores_path is not None else (),
        deviations=("validated score cache imported from a prior run identity",) if cache_import_path.is_file() else (),
    )
    progress = ScoreProgress(
        total_events=len(event_ids),
        canary_eligible_events=len(canary_events),
        successful_scores=len(successful_ids),
        missing_scores=len(missing_ids),
        calls_made=calls_made,
        complete=True,
    )
    return scores_path, opened.action == "reuse", progress


def _validate_scores(
    events: Sequence[StrategyEvent],
    records: Sequence[ScoreRecord],
    config: StrategyResearchConfig,
    identity: ScoringIdentity,
) -> tuple[ScoreRecord, ...]:
    events_by_id = {event.event_id: event for event in events}
    if len(events_by_id) != len(events):
        raise StrategyPipelineError("event input contains duplicate event IDs")
    seen: set[str] = set()
    validated: list[ScoreRecord] = []
    for record in records:
        if record.event_id in seen:
            raise StrategyPipelineError(f"score input contains duplicate event ID: {record.event_id}")
        seen.add(record.event_id)
        event = events_by_id.get(record.event_id)
        if event is None:
            raise StrategyPipelineError(f"score input contains unknown event ID: {record.event_id}")
        expected = {
            "symbol": event.symbol,
            "content_hash": event.text_sha256,
            "target_identity": event.target_identity,
            "scoring_scheme": config.scoring.scheme,
            "scorer_family": identity.scorer_family,
            "provider": config.scoring.provider,
            "model_id": config.scoring.model,
            "model_digest": identity.model_digest,
            "prompt_id": config.scoring.prompt_id,
            "prompt_hash": identity.prompt.prompt_hash,
            "temperature": identity.temperature,
            "sample_count": identity.sample_count,
            "config_hash": identity.config_hash,
            "cache_key": score_cache_key(event, identity),
        }
        for field, value in expected.items():
            if getattr(record, field) != value:
                raise StrategyPipelineError(
                    f"score identity mismatch for {record.event_id}: {field}={getattr(record, field)!r}, expected {value!r}"
                )
        if record.status == "success":
            if record.raw_label is None or record.score != map_score_label(record.raw_label, config.scoring.scheme):
                raise StrategyPipelineError(f"score mapping mismatch for {record.event_id}")
        validated.append(record)
    validated.sort(key=lambda row: (row.event_id, row.symbol, row.cache_key))
    return tuple(validated)


def prepare_strategy_inputs(
    config: StrategyResearchConfig,
    events_path: Path,
    scores_path: Path,
) -> StrategyInputs:
    """Join frozen events, scores, and prices without imputing missing evidence."""

    if config.run.evaluation_start is None:
        raise StrategyPipelineError("run.evaluation_start is required")
    events = load_strategy_events(events_path)
    scores = load_score_records(scores_path)
    score_by_event = {record.event_id: record for record in scores if record.status == "success"}
    if len(score_by_event) != sum(record.status == "success" for record in scores):
        raise StrategyPipelineError("successful score input contains duplicate event IDs")
    prices = load_verified_price_panel(config)
    return_sessions = {row.session for row in prices.returns}
    price_symbols = set(prices.symbols)
    attrition: Counter[str] = Counter()
    state_events: list[StateEvent] = []
    earliest_session: str | None = None
    for event in events:
        score = score_by_event.get(event.event_id)
        if score is None or score.score is None:
            attrition["missing_score"] += 1
            continue
        execution_session = event.eligible_execution_session.isoformat()
        if event.symbol not in price_symbols:
            attrition["missing_price_history"] += 1
            continue
        if execution_session not in return_sessions:
            attrition["no_executable_return_interval"] += 1
            continue
        earliest_session = execution_session if earliest_session is None else min(earliest_session, execution_session)
        state_events.append(
            StateEvent(
                event_id=event.event_id,
                symbol=event.symbol,
                session=execution_session,
                available_at_utc=event.available_at_utc,
                score=score.score,
            )
        )
    if earliest_session is None:
        raise StrategyPipelineError("no scored events have executable price coverage")
    decision_sessions = tuple(session for session in sorted(return_sessions) if session >= earliest_session)
    symbols = tuple(sorted({event.symbol for event in state_events}))
    boundary = config.run.evaluation_start.isoformat()
    if boundary not in decision_sessions:
        raise StrategyPipelineError(
            f"evaluation_start {boundary} must be an executable XNYS decision session"
        )
    if config.run.mode == "formal":
        insufficient_warmup = _predecision_warmup_gaps(config, prices, earliest_session)
        if insufficient_warmup:
            raise StrategyPipelineError(
                "formal price panel lacks pre-decision volatility warm-up for: "
                + ", ".join(insufficient_warmup[:10])
            )
    split = build_chronological_split(decision_sessions, boundary)
    return StrategyInputs(
        events=events,
        scores=scores,
        prices=prices,
        state_events=tuple(sorted(state_events, key=lambda row: (row.available_at_utc, row.event_id))),
        decision_sessions=decision_sessions,
        symbols=symbols,
        development_sessions=split.development_sessions,
        evaluation_sessions=split.evaluation_sessions,
        event_build_counts={},
        attrition=dict(sorted(attrition.items())),
    )


def _load_event_build_counts(paths: RunPaths) -> dict[str, int]:
    payload = read_json(paths.derived_stage("events") / "event_attrition.json")
    raw_counts = payload.get("counts")
    if not isinstance(raw_counts, dict):
        raise StrategyPipelineError("event attrition artifact is missing its counts object")
    counts: dict[str, int] = {}
    for name, value in raw_counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StrategyPipelineError(f"event attrition count {name!r} is invalid")
        counts[str(name)] = value
    return dict(sorted(counts.items()))


def _reported_attrition(inputs: StrategyInputs) -> dict[str, int]:
    return {
        **{f"event_build.{name}": count for name, count in inputs.event_build_counts.items()},
        **{f"downstream.{name}": count for name, count in inputs.attrition.items()},
    }


def _exclusion_counts(inputs: StrategyInputs) -> dict[str, int]:
    accounting = {"input_records", "expanded_candidates", "selected_events", "screening_rows"}
    return {
        **{
            f"event_build.{name}": count
            for name, count in inputs.event_build_counts.items()
            if name not in accounting
        },
        **{f"downstream.{name}": count for name, count in inputs.attrition.items()},
    }


def simulate_variant(
    config: StrategyResearchConfig,
    inputs: StrategyInputs,
    *,
    variant: str,
    candidate: StrategyCandidate | None = None,
    scale_training_sessions: Sequence[str] | None = None,
    target_end_session: str | None = None,
) -> VariantRun:
    """Run one fixed variant over one declared session history."""

    half_life = candidate.half_life_sessions if candidate is not None else config.signal.half_life_sessions
    scale_quantile = candidate.state_scale_quantile if candidate is not None else config.signal.state_scale_quantile
    no_trade_band = candidate.no_trade_band if candidate is not None else config.signal.no_trade_band
    state_config = StateConfig(
        variant=variant,  # type: ignore[arg-type]
        fixed_hold_sessions=config.signal.fixed_hold_sessions,
        half_life_sessions=half_life,
        severity_power=candidate.severity_power if candidate is not None else config.signal.severity_power,
        reversal_reset=candidate.reversal_reset if candidate is not None else config.signal.reversal_reset,
        impulse_scale=candidate.impulse_scale if candidate is not None else config.signal.impulse_scale,
        state_cap=candidate.state_cap if candidate is not None else config.signal.state_cap,
    )
    global_ordinals = {session: index for index, session in enumerate(inputs.prices.sessions)}
    transitions = build_state_table(
        inputs.decision_sessions,
        inputs.symbols,
        inputs.state_events,
        state_config,
        session_ordinals={session: global_ordinals[session] for session in inputs.decision_sessions},
        run_id=config.run.id,
    )
    training_sessions = tuple(scale_training_sessions or inputs.development_sessions)
    scales = estimate_state_scales(
        transitions,
        training_sessions,
        quantile=scale_quantile,
        shrinkage_k=config.signal.scale_shrinkage_k,
        minimum_scale=config.signal.minimum_state_scale,
    )
    actions = apply_state_scales(transitions, scales, no_trade_band=no_trade_band)
    volatilities = calculate_lagged_volatility(
        inputs.prices.returns,
        inputs.decision_sessions,
        inputs.symbols,
        window_sessions=config.prices.volatility_window_sessions,
        minimum_sessions=config.prices.volatility_minimum_sessions,
    )
    action_lookup = {(row.session, row.symbol): row.action for row in actions}
    volatility_lookup = volatility_mapping(volatilities)
    executable = {(row.session, row.symbol) for row in inputs.prices.returns}
    limits = PortfolioConstraints(
        gross_limit=config.portfolio.gross_limit,
        net_limit=config.portfolio.net_limit,
        single_name_limit=config.portfolio.single_name_limit,
        volatility_floor=config.prices.volatility_floor,
    )
    target_sessions = tuple(
        session
        for session in inputs.decision_sessions
        if target_end_session is None or session <= target_end_session
    )
    targets: list[TargetPortfolio] = []
    for session in target_sessions:
        session_actions = {symbol: action_lookup[(session, symbol)] for symbol in inputs.symbols}
        session_volatilities = {
            symbol: volatility_lookup[(session, symbol)] if (session, symbol) in executable else None
            for symbol in inputs.symbols
        }
        targets.append(project_target_weights(session, session_actions, session_volatilities, limits))
    ledger = run_open_to_open_ledger(
        targets,
        inputs.prices.returns,
        cost_rate_per_side=config.execution.cost_bps_per_side / 10_000,
        accounting_reset_session=(
            config.run.evaluation_start.isoformat()
            if config.run.evaluation_start is not None
            and config.run.evaluation_start.isoformat() in target_sessions
            else None
        ),
        force_final_liquidation=True,
    )
    boundary = config.run.evaluation_start.isoformat() if config.run.evaluation_start else "9999-12-31"
    assert_no_boundary_crossing(ledger, boundary)
    development_rows = [row for row in ledger if row.session < boundary]
    eval_rows = evaluation_rows(ledger, boundary)
    return VariantRun(
        variant=variant,
        candidate=candidate,
        transitions=tuple(transitions),
        scale_estimate=scales,
        actions=tuple(actions),
        volatilities=tuple(volatilities),
        targets=tuple(targets),
        ledger=tuple(ledger),
        development_metrics=calculate_portfolio_metrics(development_rows),
        evaluation_metrics=calculate_portfolio_metrics(eval_rows),
    )


def run_tuning(config: StrategyResearchConfig, inputs: StrategyInputs) -> TuningRun:
    """Evaluate every declared reset-strategy candidate on development folds only."""

    folds = build_expanding_folds(
        inputs.development_sessions,
        requested_folds=config.tuning.folds,
        minimum_training_sessions=config.tuning.minimum_training_sessions,
        requested_validation_sessions=config.tuning.validation_sessions,
        minimum_validation_sessions=config.tuning.minimum_validation_sessions,
    )
    candidates = build_candidate_grid(
        half_life_sessions=config.tuning.half_life_grid,
        state_scale_quantiles=config.tuning.state_scale_quantile_grid,
        no_trade_bands=config.tuning.no_trade_band_grid,
        severity_power=config.signal.severity_power,
        reversal_reset=config.signal.reversal_reset,
        impulse_scale=config.signal.impulse_scale,
        state_cap=config.signal.state_cap,
    )
    guardrails = CandidateGuardrails(
        minimum_active_days=config.tuning.minimum_active_validation_days,
        minimum_supported_stocks=config.tuning.minimum_supported_stocks,
        maximum_exposure_violations=config.tuning.maximum_exposure_violations,
    )
    candidate_results: list[CandidateResult] = []
    for candidate in candidates:
        fold_metrics: list[FoldMetric] = []
        for fold in folds:
            variant = simulate_variant(
                config,
                inputs,
                variant="decay_with_reset",
                candidate=candidate,
                scale_training_sessions=fold.training_sessions,
                target_end_session=fold.validation_sessions[-1],
            )
            validation_start = fold.validation_sessions[0]
            validation_end = fold.validation_sessions[-1]
            validation_rows = [
                row
                for row in variant.ledger
                if validation_start <= row.session <= validation_end
                or (row.final_liquidation and row.session > validation_end)
            ]
            validation_session_set = set(fold.validation_sessions)
            eligible_volatility_sessions: dict[str, set[str]] = defaultdict(set)
            for volatility in variant.volatilities:
                if volatility.session in validation_session_set and volatility.eligible:
                    eligible_volatility_sessions[volatility.symbol].add(volatility.session)
            supported_symbols = {
                symbol
                for symbol, sessions in eligible_volatility_sessions.items()
                if sessions == validation_session_set
            }
            exposure_violations = sum(
                row.gross_exposure > config.portfolio.gross_limit + 1e-12
                or abs(row.net_exposure) > config.portfolio.net_limit + 1e-12
                or any(abs(weight) > config.portfolio.single_name_limit + 1e-12 for _, weight in row.target_weights)
                for row in validation_rows
            )
            fold_metrics.append(
                FoldMetric(
                    fold_index=fold.fold_index,
                    sharpe=annualized_sharpe([row.net_return for row in validation_rows]),
                    average_turnover=fmean(row.turnover for row in validation_rows) if validation_rows else math.inf,
                    active_days=sum(row.active_names > 0 for row in validation_rows),
                    supported_stocks=len(supported_symbols),
                    exposure_violations=exposure_violations,
                    valid_price_coverage=(
                        bool(validation_rows)
                        and len({row.session for row in validation_rows if not row.final_liquidation})
                        == len(validation_session_set)
                    ),
                    maximum_single_name_contribution=_maximum_single_name_contribution(validation_rows),
                )
            )
        candidate_results.append(
            evaluate_candidate(
                candidate,
                fold_metrics,
                stability_penalty=config.tuning.stability_penalty,
                guardrails=guardrails,
            )
        )
    selected = select_candidate(candidate_results)
    return TuningRun(tuple(folds), tuple(candidate_results), selected)


def _maximum_single_name_contribution(rows: Sequence[DailyLedgerRow]) -> float | None:
    by_symbol: dict[str, float] = defaultdict(float)
    total = 0.0
    for row in rows:
        for contribution in row.contributions:
            by_symbol[contribution.symbol] += contribution.net_pnl_usd
            total += contribution.net_pnl_usd
    if not by_symbol or total == 0:
        return None
    return max(abs(value / total) for value in by_symbol.values())


def tuning_stage(
    config: StrategyResearchConfig,
    paths: RunPaths,
    inputs: StrategyInputs,
    *,
    events_path: Path,
    scores_path: Path,
    repo_root: Path,
    command: Sequence[str] | None = None,
) -> tuple[TuningRun, Path, bool]:
    output_dir = paths.results_stage("tuning")
    selected_path = output_dir / "selected_config.json"
    history_path = output_dir / "tuning_history.jsonl"
    folds_path = output_dir / "fold_metrics.jsonl"
    fold_definitions_path = output_dir / "fold_definitions.json"
    inputs_hashes = _input_identities(
        paths,
        events_sha256=sha256_file(events_path),
        scores_sha256=sha256_file(scores_path),
        price_panel_sha256=inputs.prices.panel_sha256,
        price_manifest_sha256=inputs.prices.manifest_sha256,
    )
    store = _stage_store(paths, "tuning", repo_root=repo_root)
    inspection = store.inspect(config=config, input_identities=inputs_hashes, verify_outputs=True)
    if inspection.reusable:
        tuning = _load_tuning_run(history_path, fold_definitions_path, selected_path)
        return tuning, selected_path, True
    opened = store.begin(config=config, input_identities=inputs_hashes, command=_command(command))
    tuning = run_tuning(config, inputs)
    history_rows = [_candidate_payload(result) for result in tuning.candidates]
    fold_rows = [
        {
            "candidate_identity": list(result.candidate.identity),
            **_jsonable(metric),
        }
        for result in tuning.candidates
        for metric in result.folds
    ]
    write_immutable_jsonl(history_path, history_rows)
    write_immutable_jsonl(folds_path, fold_rows)
    write_immutable_json(
        fold_definitions_path,
        {"folds": [_jsonable(fold) for fold in tuning.folds]},
    )
    selected_payload = {
        "candidate": _jsonable(tuning.selected.candidate),
        "candidate_identity": list(tuning.selected.candidate.identity),
        "objective": tuning.selected.objective,
        "selection_hash": sha256_text(canonical_json(_candidate_payload(tuning.selected))),
        "development_sessions": list(inputs.development_sessions),
        "evaluation_start": config.run.evaluation_start.isoformat() if config.run.evaluation_start else None,
        "candidate_count": len(tuning.candidates),
        "fold_definitions_sha256": sha256_file(fold_definitions_path),
    }
    write_immutable_json(selected_path, selected_payload)
    store.complete(
        outputs={
            "tuning_history": history_path,
            "fold_metrics": folds_path,
            "fold_definitions": fold_definitions_path,
            "selected_config": selected_path,
        },
        row_counts={
            "candidates": len(tuning.candidates),
            "fold_results": len(fold_rows),
            "folds": len(tuning.folds),
        },
        exclusions={"invalid_candidates": sum(not result.valid for result in tuning.candidates)},
    )
    return tuning, selected_path, opened.action == "reuse"


def _candidate_payload(result: CandidateResult) -> dict[str, Any]:
    return {
        "candidate": _jsonable(result.candidate),
        "candidate_identity": list(result.candidate.identity),
        "folds": [_jsonable(fold) for fold in result.folds],
        "objective": result.objective,
        "median_sharpe": result.median_sharpe,
        "sharpe_iqr": result.sharpe_iqr,
        "average_turnover": result.average_turnover,
        "valid": result.valid,
        "rejection_reasons": list(result.rejection_reasons),
    }


def _load_tuning_run(
    history_path: Path,
    fold_definitions_path: Path,
    selected_path: Path,
) -> TuningRun:
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected_payload = read_json(selected_path)
    candidates = tuple(_candidate_result_from_payload(row) for row in history)
    selected_identity = tuple(float(value) for value in selected_payload["candidate_identity"])
    selected = next((result for result in candidates if result.candidate.identity == selected_identity), None)
    if selected is None:
        raise StrategyPipelineError("selected tuning identity is absent from tuning history")
    definitions = read_json(fold_definitions_path)
    raw_folds = definitions.get("folds")
    if not isinstance(raw_folds, list):
        raise StrategyPipelineError("tuning fold definitions are missing")
    folds = tuple(
        ChronologicalFold(
            fold_index=int(row["fold_index"]),
            training_sessions=tuple(str(value) for value in row["training_sessions"]),
            validation_sessions=tuple(str(value) for value in row["validation_sessions"]),
        )
        for row in raw_folds
    )
    if sha256_file(fold_definitions_path) != selected_payload.get("fold_definitions_sha256"):
        raise StrategyPipelineError("selected tuning configuration does not match its fold definitions")
    return TuningRun(folds, candidates, selected)


def _candidate_result_from_payload(payload: Mapping[str, Any]) -> CandidateResult:
    raw_candidate = payload["candidate"]
    candidate = StrategyCandidate(**raw_candidate)
    folds = tuple(FoldMetric(**row) for row in payload.get("folds", []))
    return CandidateResult(
        candidate=candidate,
        folds=folds,
        objective=float(payload["objective"]) if payload.get("objective") is not None else None,
        median_sharpe=float(payload["median_sharpe"]) if payload.get("median_sharpe") is not None else None,
        sharpe_iqr=float(payload["sharpe_iqr"]) if payload.get("sharpe_iqr") is not None else None,
        average_turnover=(
            float(payload["average_turnover"])
            if payload.get("average_turnover") is not None
            else None
        ),
        valid=bool(payload["valid"]),
        rejection_reasons=tuple(str(value) for value in payload.get("rejection_reasons", [])),
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    return value


def state_stage(
    config: StrategyResearchConfig,
    paths: RunPaths,
    inputs: StrategyInputs,
    selected: CandidateResult,
    *,
    selected_config_path: Path,
    events_path: Path,
    scores_path: Path,
    repo_root: Path,
    command: Sequence[str] | None = None,
) -> tuple[VariantRun, bool]:
    output_dir = paths.results_stage("state")
    states_path = output_dir / "states.jsonl"
    actions_path = output_dir / "actions.jsonl"
    scales_path = output_dir / "state_scales.json"
    stage_inputs = _input_identities(
        paths,
        selected_config_sha256=sha256_file(selected_config_path),
        events_sha256=sha256_file(events_path),
        scores_sha256=sha256_file(scores_path),
        price_panel_sha256=inputs.prices.panel_sha256,
        price_manifest_sha256=inputs.prices.manifest_sha256,
    )
    store = _stage_store(paths, "state", repo_root=repo_root)
    inspection = store.inspect(config=config, input_identities=stage_inputs, verify_outputs=True)
    variant = simulate_variant(
        config,
        inputs,
        variant="decay_with_reset",
        candidate=selected.candidate,
    )
    if inspection.reusable:
        return variant, True
    opened = store.begin(config=config, input_identities=stage_inputs, command=_command(command))
    write_immutable_jsonl(states_path, [_jsonable(row) for row in variant.transitions])
    write_immutable_jsonl(actions_path, [_jsonable(row) for row in variant.actions])
    write_immutable_json(scales_path, _jsonable(variant.scale_estimate))
    store.complete(
        outputs={"states": states_path, "actions": actions_path, "state_scales": scales_path},
        row_counts={"states": len(variant.transitions), "actions": len(variant.actions)},
        exclusions=_exclusion_counts(inputs),
    )
    return variant, opened.action == "reuse"


def backtest_stage(
    config: StrategyResearchConfig,
    paths: RunPaths,
    inputs: StrategyInputs,
    selected: CandidateResult,
    *,
    selected_config_path: Path,
    events_path: Path,
    scores_path: Path,
    repo_root: Path,
    command: Sequence[str] | None = None,
) -> tuple[dict[str, VariantRun], dict[str, PairedBootstrapResult], Path, bool]:
    output_dir = paths.results_stage("backtest")
    positions_path = output_dir / "positions.jsonl"
    orders_path = output_dir / "orders.jsonl"
    pnl_path = output_dir / "daily_pnl.jsonl"
    stock_path = output_dir / "stock_metrics.csv"
    metrics_path = output_dir / "evaluation_metrics.json"
    diagnostics_path = output_dir / "diagnostics.json"
    bootstrap_path = output_dir / "bootstrap.csv"
    stage_inputs = _input_identities(
        paths,
        selected_config_sha256=sha256_file(selected_config_path),
        events_sha256=sha256_file(events_path),
        scores_sha256=sha256_file(scores_path),
        price_panel_sha256=inputs.prices.panel_sha256,
        price_manifest_sha256=inputs.prices.manifest_sha256,
        state_manifest_sha256=sha256_file(paths.stage_manifest("state")),
    )
    store = _stage_store(paths, "backtest", repo_root=repo_root)
    inspection = store.inspect(config=config, input_identities=stage_inputs, verify_outputs=True)

    variants = {
        "cash": simulate_variant(config, inputs, variant="cash"),
        "last_event_fixed_hold": simulate_variant(config, inputs, variant="last_event_fixed_hold"),
        "additive_decay": simulate_variant(config, inputs, variant="additive_decay"),
        "decay_with_reset": simulate_variant(
            config,
            inputs,
            variant="decay_with_reset",
            candidate=selected.candidate,
        ),
    }
    comparisons = _bootstrap_comparisons(config, variants)
    position_rows = [
        {"variant": name, "session": target.session, **_jsonable(position)}
        for name, variant in variants.items()
        for target in variant.targets
        for position in target.positions
    ]
    order_rows = [
        {"variant": name, **_jsonable(order)}
        for name, variant in variants.items()
        for row in variant.ledger
        for order in row.orders
    ]
    pnl_rows = [
        {"variant": name, **_ledger_summary(row)}
        for name, variant in variants.items()
        for row in variant.ledger
    ]
    evaluation_metrics = {
        name: _jsonable(variant.evaluation_metrics)
        for name, variant in variants.items()
    }
    evaluation_payload = {
        "evaluation_start": config.run.evaluation_start.isoformat() if config.run.evaluation_start else None,
        "selected_variant": "decay_with_reset",
        "selected_candidate": _jsonable(selected.candidate),
        "metrics": evaluation_metrics,
        "interpretation": "chronological evaluation block; sample-specific research backtest, not deployable alpha",
    }
    if inspection.reusable:
        _verify_recomputed_jsonl(positions_path, position_rows, artifact_name="positions")
        _verify_recomputed_jsonl(orders_path, order_rows, artifact_name="orders")
        _verify_recomputed_jsonl(pnl_path, pnl_rows, artifact_name="daily PnL")
        if read_json(metrics_path) != evaluation_payload:
            raise StrategyPipelineError("recomputed evaluation metrics differ from the frozen backtest artifact")
        if bootstrap_path.read_text(encoding="utf-8") != _render_bootstrap_csv(comparisons):
            raise StrategyPipelineError("recomputed bootstrap results differ from the frozen backtest artifact")
        return variants, comparisons, metrics_path, True
    opened = store.begin(config=config, input_identities=stage_inputs, command=_command(command))

    write_immutable_jsonl(positions_path, position_rows)
    write_immutable_jsonl(orders_path, order_rows)
    write_immutable_jsonl(pnl_path, pnl_rows)
    write_immutable_json(metrics_path, evaluation_payload)

    selected_rows = evaluation_rows(variants["decay_with_reset"].ledger, config.run.evaluation_start.isoformat())  # type: ignore[union-attr]
    stock_rows = calculate_stock_diagnostics(selected_rows)
    _write_stock_metrics(stock_path, stock_rows, inputs, variants["decay_with_reset"])
    leave_one_out = leave_one_stock_out_returns(selected_rows)
    diagnostics_payload = {
        "event_build_counts": inputs.event_build_counts,
        "downstream_attrition": inputs.attrition,
        "price_coverage": dict(price_coverage(inputs.prices.returns, inputs.decision_sessions, inputs.symbols)),
        "contribution_concentration": _jsonable(contribution_concentration(stock_rows)),
        "leave_one_stock_out": {
            symbol: {
                "observations": len(values),
                "cumulative_net_return": _compound(values.values()),
                "after_cost_sharpe": annualized_sharpe(list(values.values())),
            }
            for symbol, values in leave_one_out.items()
        },
        "time_blocks": dict(returns_by_time_block(selected_rows)),
        "long_net_pnl_usd": sum(row.long_pnl_usd for row in stock_rows),
        "short_net_pnl_usd": sum(row.short_pnl_usd for row in stock_rows),
        "limitations": _limitations(),
    }
    write_immutable_json(diagnostics_path, diagnostics_payload)
    _write_bootstrap_csv(bootstrap_path, comparisons)
    store.complete(
        outputs={
            "positions": positions_path,
            "orders": orders_path,
            "daily_pnl": pnl_path,
            "stock_metrics": stock_path,
            "evaluation_metrics": metrics_path,
            "diagnostics": diagnostics_path,
            "bootstrap": bootstrap_path,
        },
        row_counts={
            "positions": len(position_rows),
            "orders": len(order_rows),
            "daily_pnl": len(pnl_rows),
            "stock_metrics": len(stock_rows),
        },
        exclusions=_exclusion_counts(inputs),
        warnings=tuple(_limitations()),
    )
    return variants, comparisons, metrics_path, opened.action == "reuse"


def _bootstrap_comparisons(
    config: StrategyResearchConfig,
    variants: Mapping[str, VariantRun],
) -> dict[str, PairedBootstrapResult]:
    boundary = config.run.evaluation_start.isoformat() if config.run.evaluation_start else "9999-12-31"
    selected = {row.session: row.net_return for row in evaluation_rows(variants["decay_with_reset"].ledger, boundary)}
    results: dict[str, PairedBootstrapResult] = {}
    for name in ("cash", "last_event_fixed_hold", "additive_decay"):
        comparator = {row.session: row.net_return for row in evaluation_rows(variants[name].ledger, boundary)}
        dates = sorted(set(selected) | set(comparator))
        selected_aligned = {session: selected.get(session, 0.0) for session in dates}
        comparator_aligned = {session: comparator.get(session, 0.0) for session in dates}
        if len(dates) < config.inference.block_length_sessions:
            raise StrategyPipelineError(
                f"evaluation has {len(dates)} paired dates, fewer than bootstrap block length "
                f"{config.inference.block_length_sessions}"
            )
        results[name] = paired_block_bootstrap(
            selected_aligned,
            comparator_aligned,
            block_length=config.inference.block_length_sessions,
            replications=config.inference.replications,
            seed=config.inference.seed,
        )
    return results


def _verify_recomputed_jsonl(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    artifact_name: str,
) -> None:
    expected = "".join(canonical_json(row) + "\n" for row in rows)
    if path.read_text(encoding="utf-8") != expected:
        raise StrategyPipelineError(
            f"recomputed {artifact_name} differs from the frozen backtest artifact"
        )


def _ledger_summary(row: DailyLedgerRow) -> dict[str, Any]:
    return {
        "session": row.session,
        "next_session": row.next_session,
        "start_nav_usd": row.start_nav_usd,
        "gross_return": row.gross_return,
        "transaction_cost": row.transaction_cost,
        "net_return": row.net_return,
        "end_nav_usd": row.end_nav_usd,
        "turnover": row.turnover,
        "gross_exposure": row.gross_exposure,
        "long_exposure": row.long_exposure,
        "short_exposure": row.short_exposure,
        "net_exposure": row.net_exposure,
        "cash_weight": row.cash_weight,
        "active_names": row.active_names,
        "final_liquidation": row.final_liquidation,
    }


def _write_stock_metrics(
    path: Path,
    stock_rows: Sequence[StockDiagnostic],
    inputs: StrategyInputs,
    selected: VariantRun,
) -> None:
    event_counts = Counter(event.symbol for event in inputs.state_events)
    states: dict[str, list[float]] = defaultdict(list)
    active_state_days: Counter[str] = Counter()
    for row in selected.transitions:
        if row.session in set(inputs.evaluation_sessions):
            states[row.symbol].append(abs(row.final_state))
            active_state_days[row.symbol] += row.final_state != 0
    fields = [
        "symbol",
        "event_count",
        "active_days",
        "active_state_days",
        "average_absolute_state",
        "average_absolute_position",
        "gross_pnl_usd",
        "transaction_cost_usd",
        "net_pnl_usd",
        "turnover",
        "maximum_weight",
        "long_pnl_usd",
        "short_pnl_usd",
    ]
    rows = []
    for stock in stock_rows:
        payload = _jsonable(stock)
        payload.update(
            {
                "event_count": event_counts[stock.symbol],
                "active_state_days": active_state_days[stock.symbol],
                "average_absolute_state": fmean(states[stock.symbol]) if states[stock.symbol] else 0.0,
            }
        )
        rows.append(payload)
    _write_csv(path, fields, rows)


def _write_bootstrap_csv(path: Path, comparisons: Mapping[str, PairedBootstrapResult]) -> None:
    write_immutable_text(path, _render_bootstrap_csv(comparisons))


def _render_bootstrap_csv(comparisons: Mapping[str, PairedBootstrapResult]) -> str:
    fields = [
        "comparator",
        "mean_daily_difference",
        "confidence_interval_low",
        "confidence_interval_high",
        "mean_direction",
        "confidence_direction",
        "effective_dates",
        "block_length",
        "replications",
        "seed",
    ]
    rows = [
        {
            "comparator": name,
            "mean_daily_difference": result.mean_daily_difference,
            "confidence_interval_low": result.confidence_interval_low,
            "confidence_interval_high": result.confidence_interval_high,
            "mean_direction": result.mean_direction,
            "confidence_direction": result.confidence_direction,
            "effective_dates": len(result.effective_dates),
            "block_length": result.block_length,
            "replications": result.replications,
            "seed": result.seed,
        }
        for name, result in sorted(comparisons.items())
    ]
    return _render_csv(fields, rows)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    write_immutable_text(path, _render_csv(fields, rows))


def _render_csv(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue()


def _compound(values: Iterable[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1 + value
    return result - 1


def _limitations() -> list[str]:
    return [
        "Execution prices are split-adjusted price returns; dividends are not back-adjusted.",
        "Short borrow, financing, liquidity, capacity, and market impact are not modelled.",
        "Daily prices leave residual intraday timing and execution uncertainty.",
        "The fixed company universe may contain survivorship bias.",
        "The chronological evaluation period has had prior exploratory exposure.",
        "Results are sample-specific research backtests, not deployable alpha or causal evidence.",
    ]


def report_stage(
    config: StrategyResearchConfig,
    paths: RunPaths,
    inputs: StrategyInputs,
    tuning: TuningRun,
    variants: Mapping[str, VariantRun],
    comparisons: Mapping[str, PairedBootstrapResult],
    *,
    selected_config_path: Path,
    evaluation_metrics_path: Path,
    repo_root: Path,
    command: Sequence[str] | None = None,
) -> tuple[Path, bool]:
    output_dir = paths.results_stage("report")
    summary_path = output_dir / "summary.md"
    leakage_path = output_dir / "leakage_checklist.json"
    manifest_path = paths.results_dir / "manifest.json"
    figures_dir = output_dir / "figures"
    cumulative_path = figures_dir / "cumulative_net_return.svg"
    exposure_path = figures_dir / "exposure_turnover.svg"
    stage_inputs = _input_identities(
        paths,
        selected_config_sha256=sha256_file(selected_config_path),
        evaluation_metrics_sha256=sha256_file(evaluation_metrics_path),
        backtest_manifest_sha256=sha256_file(paths.stage_manifest("backtest")),
    )
    store = _stage_store(paths, "report", repo_root=repo_root)
    inspection = store.inspect(config=config, input_identities=stage_inputs, verify_outputs=True)
    if inspection.reusable:
        return summary_path, True
    opened = store.begin(config=config, input_identities=stage_inputs, command=_command(command))
    evaluation_metrics = {
        name: _jsonable(variant.evaluation_metrics)
        for name, variant in variants.items()
    }
    summary = render_summary(
        config,
        resolved_run_id=paths.identity.resolved_run_id,
        selected=tuning.selected,
        evaluation_metrics=evaluation_metrics,
        comparisons=comparisons,
        attrition=_reported_attrition(inputs),
        warnings=_limitations(),
    )
    write_immutable_text(summary_path, summary)
    leakage_checks = leakage_payload(config)
    write_immutable_json(leakage_path, {"checks": leakage_checks})
    boundary = config.run.evaluation_start.isoformat() if config.run.evaluation_start else "9999-12-31"
    evaluation_ledger = evaluation_rows(variants["decay_with_reset"].ledger, boundary)
    write_immutable_text(
        cumulative_path,
        cumulative_return_svg(evaluation_ledger, title="Selected strategy cumulative net return"),
    )
    write_immutable_text(
        exposure_path,
        exposure_turnover_svg(evaluation_ledger, title="Selected strategy exposure and turnover"),
    )
    output_hashes = _collect_output_hashes(paths, repo_root=repo_root, exclude={manifest_path, store.path})
    backtest_manifest = read_json(paths.stage_manifest("backtest"))
    master_manifest = {
        "schema_version": 1,
        "status": "completed",
        "run_id": paths.identity.resolved_run_id,
        "logical_run_id": config.run.id,
        "run_identity_sha256": paths.identity.identity_sha256,
        "config_sha256": config.config_sha256,
        "config": config.to_payload(),
        "command": list(_command(command)),
        "input_identities": paths.identity.input_identities,
        "event_identity": config.data.event_identity,
        "processing_buffer_minutes": config.data.processing_buffer_minutes,
        "calendar": config.data.exchange_calendar,
        "exchange_timezone": config.data.exchange_timezone,
        "scorer": {
            "provider": config.scoring.provider,
            "model": config.scoring.model,
            "model_digest": config.scoring.model_digest,
            "endpoint": resolved_scoring_identity(config).resolved_endpoint,
            "prompt_id": config.scoring.prompt_id,
            "prompt_hash": resolved_scoring_identity(config).prompt.prompt_hash,
            "scheme": config.scoring.scheme,
            "competence_experiment_id": config.scoring.competence_experiment_id,
            "competence_evidence_sha256": _verify_competence_evidence(config),
        },
        "prices": {
            "panel_sha256": inputs.prices.panel_sha256,
            "manifest_sha256": inputs.prices.manifest_sha256,
            "execution_field": config.prices.execution_field,
            "return_convention": config.prices.return_convention,
            "adjustment_limitation": "split-adjusted price returns; dividends not back-adjusted",
        },
        "split": {
            "development_first": inputs.development_sessions[0],
            "development_last": inputs.development_sessions[-1],
            "evaluation_start": inputs.evaluation_sessions[0],
            "evaluation_last": inputs.evaluation_sessions[-1],
        },
        "tuning_grid": {
            "half_life_sessions": list(config.tuning.half_life_grid),
            "state_scale_quantiles": list(config.tuning.state_scale_quantile_grid),
            "no_trade_bands": list(config.tuning.no_trade_band_grid),
            "candidate_count": len(tuning.candidates),
        },
        "selected_parameters": _jsonable(tuning.selected.candidate),
        "cost_model": {"bps_per_side": config.execution.cost_bps_per_side},
        "risk_constraints": _jsonable(config.portfolio),
        "inference": _jsonable(config.inference),
        "event_build_counts": inputs.event_build_counts,
        "downstream_attrition": inputs.attrition,
        "warnings": _limitations(),
        "runtime": backtest_manifest.get("runtime", {}),
        "outputs": output_hashes,
    }
    write_immutable_json(manifest_path, master_manifest)
    store.complete(
        outputs={
            "summary": summary_path,
            "leakage_checklist": leakage_path,
            "cumulative_figure": cumulative_path,
            "exposure_figure": exposure_path,
            "manifest": manifest_path,
        },
        row_counts={"leakage_checks": len(leakage_checks), "figures": 2},
        exclusions=_exclusion_counts(inputs),
        warnings=tuple(_limitations()),
    )
    return summary_path, opened.action == "reuse"


def _collect_output_hashes(paths: RunPaths, *, repo_root: Path, exclude: set[Path]) -> dict[str, dict[str, Any]]:
    excluded = {path.resolve() for path in exclude}
    records: dict[str, dict[str, Any]] = {}
    for root in (paths.derived_dir, paths.results_dir):
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if (
                path.resolve() in excluded
                or paths.manifests_dir.resolve() in path.resolve().parents
                or path.name.endswith(".lock")
                or "/cache/" in path.as_posix()
                or "/attempts/" in path.as_posix()
            ):
                continue
            try:
                display = path.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                display = path.resolve().as_posix()
            records[display] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return dict(sorted(records.items()))


def execute_pipeline(
    config: StrategyResearchConfig,
    *,
    through: PipelineStage = "report",
    allow_paid: bool = False,
    max_new_scores: int | None = None,
    score_cache_from: str | Path | None = None,
    repo_root: str | Path = ".",
    command: Sequence[str] | None = None,
) -> PipelineResult:
    """Execute prerequisites through one stage, validating and reusing matches."""

    repo = Path(repo_root).resolve()
    paths = resolve_pipeline_paths(config)
    stage_order: tuple[PipelineStage, ...] = ("events", "scores", "tuning", "state", "backtest", "report")
    if through not in stage_order:
        raise StrategyPipelineError(f"unsupported pipeline stage: {through}")
    if max_new_scores is not None and through != "scores":
        raise StrategyPipelineError("max_new_scores is supported only by the explicit strategy score command")
    if score_cache_from is not None and through != "scores":
        raise StrategyPipelineError("score_cache_from is supported only by the explicit strategy score command")
    if stage_order.index(through) >= stage_order.index("tuning"):
        config.require_ready(check_files=True)
    elif through == "events":
        required = config.data.events_path if config.data.source == "synthetic" else config.data.corpus_manifest
        if required is None or not required.is_file():
            raise StrategyPipelineError(f"event source does not exist: {required}")
    else:
        for required in (config.scoring.prompts_path,):
            if not required.is_file():
                raise StrategyPipelineError(f"scoring input does not exist: {required}")

    reused: list[str] = []
    events_path, was_reused = build_events_stage(config, paths, repo_root=repo, command=command)
    if was_reused:
        reused.append("events")
    if through == "events":
        return PipelineResult(paths, events_path, Path(), None, None, None, tuple(reused))

    scores_path, was_reused, score_progress = scores_stage(
        config,
        paths,
        events_path,
        repo_root=repo,
        command=command,
        allow_paid=allow_paid,
        max_new_scores=max_new_scores,
        cache_from=Path(score_cache_from) if score_cache_from is not None else None,
    )
    if was_reused:
        reused.append("scores")
    if through == "scores":
        return PipelineResult(
            paths,
            events_path,
            scores_path,
            None,
            None,
            None,
            tuple(reused),
            score_progress,
        )

    inputs = prepare_strategy_inputs(config, events_path, scores_path)
    inputs = replace(inputs, event_build_counts=_load_event_build_counts(paths))
    tuning, selected_path, was_reused = tuning_stage(
        config,
        paths,
        inputs,
        events_path=events_path,
        scores_path=scores_path,
        repo_root=repo,
        command=command,
    )
    if was_reused:
        reused.append("tuning")
    if through == "tuning":
        return PipelineResult(paths, events_path, scores_path, selected_path, None, None, tuple(reused))

    _, was_reused = state_stage(
        config,
        paths,
        inputs,
        tuning.selected,
        selected_config_path=selected_path,
        events_path=events_path,
        scores_path=scores_path,
        repo_root=repo,
        command=command,
    )
    if was_reused:
        reused.append("state")
    if through == "state":
        return PipelineResult(paths, events_path, scores_path, selected_path, None, None, tuple(reused))

    variants, comparisons, metrics_path, was_reused = backtest_stage(
        config,
        paths,
        inputs,
        tuning.selected,
        selected_config_path=selected_path,
        events_path=events_path,
        scores_path=scores_path,
        repo_root=repo,
        command=command,
    )
    if was_reused:
        reused.append("backtest")
    if through == "backtest":
        return PipelineResult(paths, events_path, scores_path, selected_path, metrics_path, None, tuple(reused))

    summary_path, was_reused = report_stage(
        config,
        paths,
        inputs,
        tuning,
        variants,
        comparisons,
        selected_config_path=selected_path,
        evaluation_metrics_path=metrics_path,
        repo_root=repo,
        command=command,
    )
    if was_reused:
        reused.append("report")
    return PipelineResult(paths, events_path, scores_path, selected_path, metrics_path, summary_path, tuple(reused))
