from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from ..artifact_io import canonical_json, sha256_file, sha256_text

PIPELINE_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
FORMAL_DERIVED_ROOT = Path("Data/derived/strategy_research")
FORMAL_RESULTS_ROOT = Path("results/strategy_research")


class StrategyConfigurationError(ValueError):
    """Raised when a strategy-research configuration is invalid."""


@dataclass(frozen=True)
class RunSettings:
    id: str
    title: str
    mode: Literal["smoke", "formal"] = "formal"
    evaluation_start: date | None = None


@dataclass(frozen=True)
class DataSettings:
    source: Literal["lseg", "synthetic"] = "lseg"
    corpus_manifest: Path | None = None
    events_path: Path | None = None
    screening_overrides_path: Path | None = None
    event_identity: str = "earliest_story_family_symbol"
    text_unit: str = "headline_plus_lead"
    exchange_calendar: str = "XNYS"
    exchange_timezone: str = "America/New_York"
    processing_buffer_minutes: int = 15
    fail_on_invalid_timestamp: bool = True


@dataclass(frozen=True)
class ScoringSettings:
    provider: str = "cerebras"
    model: str = "gemma-4-31b"
    endpoint: str | None = None
    model_digest: str | None = None
    competence_evidence_path: Path | None = None
    competence_experiment_id: str | None = None
    scheme: Literal["three_class", "five_level"] = "three_class"
    prompt_id: str = "strategy_target_direction_3class_v1"
    prompts_path: Path = Path("configs/strategy_research/prompts.toml")
    scores_path: Path | None = None
    temperature: float = 0.0
    samples: int = 1
    target_specific: bool = True
    allow_missing_as_neutral: bool = False
    minimum_success_rate: float = 0.98
    max_completion_tokens: int = 64
    concurrency: int = 1
    retries: int = 3


@dataclass(frozen=True)
class SignalSettings:
    variant: Literal["cash", "last_event_fixed_hold", "additive_decay", "decay_with_reset"] = "decay_with_reset"
    fixed_hold_sessions: int = 3
    half_life_sessions: float = 3.0
    severity_power: float = 1.5
    reversal_reset: float = 0.75
    impulse_scale: float = 1.0
    state_cap: float = 3.0
    state_scale_quantile: float = 0.75
    scale_shrinkage_k: float = 50.0
    minimum_state_scale: float = 0.05
    no_trade_band: float = 0.10


@dataclass(frozen=True)
class PriceSettings:
    panel_path: Path | None = None
    manifest_path: Path | None = None
    execution_field: Literal["adjusted_open", "adjusted_close"] = "adjusted_open"
    return_convention: Literal["open_to_open", "close_to_close"] = "open_to_open"
    volatility_window_sessions: int = 20
    volatility_minimum_sessions: int = 15
    volatility_floor: float = 0.005
    fail_on_unsupported_adjustment: bool = True


@dataclass(frozen=True)
class ExecutionSettings:
    fill_rule: str = "next_eligible_open"
    cost_bps_per_side: float = 10.0
    allow_same_session_open_only_if_available_before_open: bool = True


@dataclass(frozen=True)
class PortfolioSettings:
    mode: str = "long_short"
    gross_limit: float = 1.0
    net_limit: float = 0.20
    single_name_limit: float = 0.05
    volatility_scaling: bool = True
    preserve_signal_sign: bool = True


@dataclass(frozen=True)
class TuningSettings:
    method: str = "expanding_chronological"
    folds: int = 3
    minimum_training_sessions: int = 60
    validation_sessions: int = 20
    minimum_validation_sessions: int = 10
    half_life_grid: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0)
    state_scale_quantile_grid: tuple[float, ...] = (0.50, 0.75, 0.90)
    no_trade_band_grid: tuple[float, ...] = (0.0, 0.10, 0.20)
    stability_penalty: float = 0.5
    selection_metric: str = "median_sharpe_minus_iqr_penalty"
    maximum_exposure_violations: int = 0
    minimum_supported_stocks: int = 25
    minimum_active_validation_days: int = 10

    @property
    def candidate_count(self) -> int:
        return len(self.half_life_grid) * len(self.state_scale_quantile_grid) * len(self.no_trade_band_grid)


@dataclass(frozen=True)
class InferenceSettings:
    method: str = "paired_block_bootstrap"
    block_length_sessions: int = 5
    replications: int = 2000
    seed: int = 20260715


@dataclass(frozen=True)
class ReportingSettings:
    write_intermediate_artifacts: bool = True
    include_gross_and_net: bool = True
    include_stock_diagnostics: bool = True
    include_leave_one_stock_out: bool = True


@dataclass(frozen=True)
class AgreementConditioningSettings:
    enabled: bool = False


@dataclass(frozen=True)
class OutputSettings:
    derived_root: Path = FORMAL_DERIVED_ROOT
    results_root: Path = FORMAL_RESULTS_ROOT


def _uses_isolated_formal_roots(outputs: OutputSettings) -> bool:
    return (
        outputs.derived_root.resolve() == FORMAL_DERIVED_ROOT.resolve()
        and outputs.results_root.resolve() == FORMAL_RESULTS_ROOT.resolve()
    )


@dataclass(frozen=True)
class StrategyResearchConfig:
    run: RunSettings
    data: DataSettings
    scoring: ScoringSettings
    signal: SignalSettings
    prices: PriceSettings
    execution: ExecutionSettings
    portfolio: PortfolioSettings
    tuning: TuningSettings
    inference: InferenceSettings
    reporting: ReportingSettings
    agreement_conditioning: AgreementConditioningSettings
    outputs: OutputSettings

    def to_payload(self) -> dict[str, Any]:
        return _normalize(asdict(self))

    def to_identity_payload(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["run"].pop("id", None)
        payload["run"].pop("title", None)
        payload.pop("outputs", None)
        return {"pipeline_schema_version": PIPELINE_SCHEMA_VERSION, "config": payload}

    @property
    def config_sha256(self) -> str:
        return sha256_text(canonical_json(self.to_payload()))

    def declared_input_paths(self) -> dict[str, Path]:
        values = {
            "corpus_manifest": self.data.corpus_manifest,
            "events": self.data.events_path,
            "screening_overrides": self.data.screening_overrides_path,
            "prompts": self.scoring.prompts_path,
            "scores": self.scoring.scores_path,
            "competence_evidence": self.scoring.competence_evidence_path,
            "price_panel": self.prices.panel_path,
            "price_manifest": self.prices.manifest_path,
        }
        return {name: path for name, path in values.items() if path is not None}

    def readiness_issues(self, *, check_files: bool = True) -> tuple[str, ...]:
        issues: list[str] = []
        for package, version in strategy_dependency_versions().items():
            if version is None:
                issues.append(f"required strategy dependency is not installed: {package}")
        if self.run.evaluation_start is None:
            issues.append("run.evaluation_start is required before a historical run can execute")
        if self.run.mode == "formal":
            if self.data.source != "lseg":
                issues.append("formal mode requires data.source = 'lseg'")
            if self.data.corpus_manifest is None:
                issues.append("formal mode requires data.corpus_manifest")
            if self.data.events_path is not None:
                issues.append("data.events_path is a smoke-only prebuilt fixture override")
            if self.scoring.scores_path is not None:
                issues.append("scoring.scores_path is a smoke-only prebuilt fixture override")
            if self.scoring.competence_evidence_path is None:
                issues.append("formal mode requires scoring.competence_evidence_path")
            if self.scoring.competence_experiment_id is None:
                issues.append("formal mode requires scoring.competence_experiment_id")
            if self.prices.manifest_path is None:
                issues.append("formal mode requires prices.manifest_path")
            if not _uses_isolated_formal_roots(self.outputs):
                issues.append(
                    "formal strategy outputs must remain under Data/derived/strategy_research and results/strategy_research"
                )
        elif self.data.source != "synthetic":
            issues.append("smoke mode requires data.source = 'synthetic'")
        if self.data.source == "synthetic" and self.data.events_path is None:
            issues.append("synthetic data requires data.events_path")
        if self.run.mode == "smoke" and self.scoring.scores_path is None:
            issues.append("smoke mode requires scoring.scores_path so it cannot make provider calls")
        if self.prices.panel_path is None:
            issues.append("prices.panel_path is required")
        if check_files:
            for name, path in self.declared_input_paths().items():
                if not path.is_file():
                    issues.append(f"{name} file does not exist: {path}")
        return tuple(dict.fromkeys(issues))

    def require_ready(self, *, check_files: bool = True) -> None:
        issues = self.readiness_issues(check_files=check_files)
        if issues:
            raise StrategyConfigurationError("strategy configuration is not ready:\n- " + "\n- ".join(issues))


@dataclass(frozen=True)
class RunIdentity:
    logical_run_id: str
    resolved_run_id: str
    config_sha256: str
    identity_sha256: str
    input_identities: dict[str, str]
    missing_inputs: tuple[str, ...]


@dataclass(frozen=True)
class RunPaths:
    identity: RunIdentity
    derived_dir: Path
    results_dir: Path

    @property
    def manifests_dir(self) -> Path:
        return self.results_dir / "manifests"

    def stage_manifest(self, stage: str) -> Path:
        normalized = _safe_component(stage, "stage")
        return self.manifests_dir / f"{normalized}.json"

    def derived_stage(self, stage: str) -> Path:
        return self.derived_dir / _safe_component(stage, "stage")

    def results_stage(self, stage: str) -> Path:
        return self.results_dir / _safe_component(stage, "stage")


def inspect_declared_inputs(config: StrategyResearchConfig) -> tuple[dict[str, str], tuple[str, ...]]:
    """Hash existing declared inputs without creating or changing any files."""

    identities: dict[str, str] = {}
    missing: list[str] = []
    for name, path in sorted(config.declared_input_paths().items()):
        if path.is_file():
            identities[name] = sha256_file(path)
        else:
            identities[name] = f"missing:{path.as_posix()}"
            missing.append(name)
    for package, version in strategy_dependency_versions().items():
        name = f"dependency:{package}"
        if version is None:
            identities[name] = f"missing:{package}"
            missing.append(name)
        else:
            identities[name] = version
    return identities, tuple(missing)


def strategy_dependency_versions() -> dict[str, str | None]:
    """Resolve behavior-affecting strategy dependencies for identity and provenance."""

    try:
        exchange_calendars_version: str | None = metadata.version("exchange-calendars")
    except metadata.PackageNotFoundError:
        exchange_calendars_version = None
    return {"exchange-calendars": exchange_calendars_version}


def compute_run_identity(
    config: StrategyResearchConfig,
    *,
    additional_inputs: Mapping[str, str] | None = None,
) -> RunIdentity:
    input_identities, missing = inspect_declared_inputs(config)
    for name, value in sorted((additional_inputs or {}).items()):
        clean_name = str(name).strip()
        clean_value = str(value).strip()
        if not clean_name or not clean_value:
            raise StrategyConfigurationError("additional input identities require non-empty names and values")
        if clean_name in input_identities and input_identities[clean_name] != clean_value:
            raise StrategyConfigurationError(f"additional input identity conflicts with declared input {clean_name!r}")
        input_identities[clean_name] = clean_value
    identity_payload = config.to_identity_payload() | {"input_identities": dict(sorted(input_identities.items()))}
    identity_sha256 = sha256_text(canonical_json(identity_payload))
    resolved = f"{config.run.id}-{identity_sha256[:12]}"
    return RunIdentity(
        logical_run_id=config.run.id,
        resolved_run_id=resolved,
        config_sha256=config.config_sha256,
        identity_sha256=identity_sha256,
        input_identities=dict(sorted(input_identities.items())),
        missing_inputs=missing,
    )


def resolve_run_paths(config: StrategyResearchConfig, identity: RunIdentity | None = None) -> RunPaths:
    """Resolve run paths without making directories or touching the filesystem."""

    if config.run.mode == "formal" and not _uses_isolated_formal_roots(config.outputs):
        raise StrategyConfigurationError(
            "formal strategy outputs must remain under Data/derived/strategy_research and results/strategy_research"
        )
    resolved_identity = identity or compute_run_identity(config)
    if resolved_identity.logical_run_id != config.run.id:
        raise StrategyConfigurationError("run identity does not belong to this logical run ID")
    return RunPaths(
        identity=resolved_identity,
        derived_dir=config.outputs.derived_root / resolved_identity.resolved_run_id,
        results_dir=config.outputs.results_root / resolved_identity.resolved_run_id,
    )


def load_strategy_config(path: str | Path) -> StrategyResearchConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise StrategyConfigurationError(f"cannot load strategy config {config_path}: {exc}") from exc
    try:
        return _build_config(raw)
    except StrategyConfigurationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategyConfigurationError(f"invalid strategy config {config_path}: {exc}") from exc


def _build_config(raw: dict[str, Any]) -> StrategyResearchConfig:
    allowed_sections = {
        "run",
        "data",
        "scoring",
        "signal",
        "prices",
        "execution",
        "portfolio",
        "tuning",
        "inference",
        "reporting",
        "agreement_conditioning",
        "outputs",
    }
    _reject_unknown(raw, allowed_sections, "root")
    run_raw = _table(raw, "run", required=True)
    data_raw = _table(raw, "data")
    scoring_raw = _table(raw, "scoring")
    signal_raw = _table(raw, "signal")
    prices_raw = _table(raw, "prices")
    execution_raw = _table(raw, "execution")
    portfolio_raw = _table(raw, "portfolio")
    tuning_raw = _table(raw, "tuning")
    inference_raw = _table(raw, "inference")
    reporting_raw = _table(raw, "reporting")
    agreement_raw = _table(raw, "agreement_conditioning")
    outputs_raw = _table(raw, "outputs")

    _reject_unknown(run_raw, {"id", "title", "mode", "evaluation_start"}, "run")
    run_id = _string(run_raw, "id", section="run", required=True)
    if not _RUN_ID.fullmatch(run_id):
        raise StrategyConfigurationError("run.id must be 1-64 characters using letters, digits, '.', '_' or '-'")
    evaluation_value = _string(run_raw, "evaluation_start", section="run")
    try:
        evaluation_start = date.fromisoformat(evaluation_value) if evaluation_value else None
    except ValueError as exc:
        raise StrategyConfigurationError("run.evaluation_start must be an ISO date in YYYY-MM-DD form") from exc
    run = RunSettings(
        id=run_id,
        title=_string(run_raw, "title", section="run") or run_id,
        mode=_choice(run_raw, "mode", {"smoke", "formal"}, "formal", "run"),  # type: ignore[arg-type]
        evaluation_start=evaluation_start,
    )

    _reject_unknown(
        data_raw,
        {
            "source",
            "corpus_manifest",
            "events_path",
            "screening_overrides_path",
            "event_identity",
            "text_unit",
            "exchange_calendar",
            "exchange_timezone",
            "processing_buffer_minutes",
            "fail_on_invalid_timestamp",
        },
        "data",
    )
    data = DataSettings(
        source=_choice(data_raw, "source", {"lseg", "synthetic"}, "lseg", "data"),  # type: ignore[arg-type]
        corpus_manifest=_path(data_raw.get("corpus_manifest")),
        events_path=_path(data_raw.get("events_path")),
        screening_overrides_path=_path(data_raw.get("screening_overrides_path")),
        event_identity=_string(data_raw, "event_identity", section="data") or "earliest_story_family_symbol",
        text_unit=_string(data_raw, "text_unit", section="data") or "headline_plus_lead",
        exchange_calendar=_string(data_raw, "exchange_calendar", section="data") or "XNYS",
        exchange_timezone=_string(data_raw, "exchange_timezone", section="data") or "America/New_York",
        processing_buffer_minutes=_integer(data_raw, "processing_buffer_minutes", 15, "data"),
        fail_on_invalid_timestamp=_boolean(data_raw, "fail_on_invalid_timestamp", True, "data"),
    )
    if data.event_identity != "earliest_story_family_symbol":
        raise StrategyConfigurationError("data.event_identity must be 'earliest_story_family_symbol' in pipeline v1")
    if data.text_unit != "headline_plus_lead":
        raise StrategyConfigurationError("data.text_unit must be 'headline_plus_lead' in pipeline v1")
    if data.exchange_calendar != "XNYS":
        raise StrategyConfigurationError("data.exchange_calendar must be 'XNYS' for the current US universe")
    try:
        ZoneInfo(data.exchange_timezone)
    except Exception as exc:
        raise StrategyConfigurationError(f"unknown data.exchange_timezone: {data.exchange_timezone}") from exc
    if data.processing_buffer_minutes < 0:
        raise StrategyConfigurationError("data.processing_buffer_minutes cannot be negative")
    if not data.fail_on_invalid_timestamp:
        raise StrategyConfigurationError("data.fail_on_invalid_timestamp must remain true in pipeline v1")

    _reject_unknown(
        scoring_raw,
        {
            "provider",
            "model",
            "endpoint",
            "model_digest",
            "competence_evidence_path",
            "competence_experiment_id",
            "scheme",
            "prompt_id",
            "prompts_path",
            "scores_path",
            "temperature",
            "samples",
            "target_specific",
            "allow_missing_as_neutral",
            "minimum_success_rate",
            "max_completion_tokens",
            "concurrency",
            "retries",
        },
        "scoring",
    )
    scheme = _choice(scoring_raw, "scheme", {"three_class", "five_level"}, "three_class", "scoring")
    scoring = ScoringSettings(
        provider=_string(scoring_raw, "provider", section="scoring") or "cerebras",
        model=_string(scoring_raw, "model", section="scoring") or "gemma-4-31b",
        endpoint=_string(scoring_raw, "endpoint", section="scoring") or None,
        model_digest=_string(scoring_raw, "model_digest", section="scoring") or None,
        competence_evidence_path=_path(scoring_raw.get("competence_evidence_path")),
        competence_experiment_id=_string(scoring_raw, "competence_experiment_id", section="scoring") or None,
        scheme=scheme,  # type: ignore[arg-type]
        prompt_id=(
            _string(scoring_raw, "prompt_id", section="scoring")
            or f"strategy_target_direction_{'3class' if scheme == 'three_class' else '5level'}_v1"
        ),
        prompts_path=_path(scoring_raw.get("prompts_path")) or Path("configs/strategy_research/prompts.toml"),
        scores_path=_path(scoring_raw.get("scores_path")),
        temperature=_number(scoring_raw, "temperature", 0.0, "scoring"),
        samples=_integer(scoring_raw, "samples", 1, "scoring"),
        target_specific=_boolean(scoring_raw, "target_specific", True, "scoring"),
        allow_missing_as_neutral=_boolean(scoring_raw, "allow_missing_as_neutral", False, "scoring"),
        minimum_success_rate=_number(scoring_raw, "minimum_success_rate", 0.98, "scoring"),
        max_completion_tokens=_integer(scoring_raw, "max_completion_tokens", 64, "scoring"),
        concurrency=_integer(scoring_raw, "concurrency", 1, "scoring"),
        retries=_integer(scoring_raw, "retries", 3, "scoring"),
    )
    if not scoring.target_specific:
        raise StrategyConfigurationError("scoring.target_specific must remain true")
    if scoring.allow_missing_as_neutral:
        raise StrategyConfigurationError("scoring.allow_missing_as_neutral must remain false")
    expected_prompt_id = f"strategy_target_direction_{'3class' if scheme == 'three_class' else '5level'}_v1"
    if scoring.prompt_id != expected_prompt_id:
        raise StrategyConfigurationError(f"scoring.prompt_id must be {expected_prompt_id!r} for scheme {scheme!r}")
    if (
        scoring.temperature < 0
        or scoring.samples < 1
        or scoring.max_completion_tokens < 1
        or scoring.concurrency < 1
        or scoring.retries < 0
    ):
        raise StrategyConfigurationError("scoring numeric settings must be non-negative and counts must be positive")
    if not 0 < scoring.minimum_success_rate <= 1:
        raise StrategyConfigurationError("scoring.minimum_success_rate must be in (0, 1]")
    if scoring.temperature != 0.0 or scoring.samples != 1:
        raise StrategyConfigurationError("strategy scoring v1 requires temperature = 0 and samples = 1")
    if scoring.concurrency != 1:
        raise StrategyConfigurationError("strategy scoring v1 requires concurrency = 1")
    if scoring.provider not in {"cerebras", "openrouter", "ollama", "fixture"}:
        raise StrategyConfigurationError("scoring.provider must be cerebras, openrouter, ollama, or fixture")
    if scoring.provider == "ollama" and scoring.model_digest is None:
        raise StrategyConfigurationError(
            "strategy scoring with Ollama requires an explicit scoring.model_digest"
        )
    if (scoring.competence_evidence_path is None) != (scoring.competence_experiment_id is None):
        raise StrategyConfigurationError(
            "scoring.competence_evidence_path and scoring.competence_experiment_id must be set together"
        )

    _reject_unknown(signal_raw, set(SignalSettings.__dataclass_fields__), "signal")
    signal = SignalSettings(
        variant=_choice(
            signal_raw,
            "variant",
            {"cash", "last_event_fixed_hold", "additive_decay", "decay_with_reset"},
            "decay_with_reset",
            "signal",
        ),  # type: ignore[arg-type]
        fixed_hold_sessions=_integer(signal_raw, "fixed_hold_sessions", 3, "signal"),
        half_life_sessions=_number(signal_raw, "half_life_sessions", 3.0, "signal"),
        severity_power=_number(signal_raw, "severity_power", 1.5, "signal"),
        reversal_reset=_number(signal_raw, "reversal_reset", 0.75, "signal"),
        impulse_scale=_number(signal_raw, "impulse_scale", 1.0, "signal"),
        state_cap=_number(signal_raw, "state_cap", 3.0, "signal"),
        state_scale_quantile=_number(signal_raw, "state_scale_quantile", 0.75, "signal"),
        scale_shrinkage_k=_number(signal_raw, "scale_shrinkage_k", 50.0, "signal"),
        minimum_state_scale=_number(signal_raw, "minimum_state_scale", 0.05, "signal"),
        no_trade_band=_number(signal_raw, "no_trade_band", 0.10, "signal"),
    )
    if min(signal.fixed_hold_sessions, signal.half_life_sessions, signal.severity_power, signal.impulse_scale, signal.state_cap) <= 0:
        raise StrategyConfigurationError("signal durations, powers, scale, and cap must be positive")
    if not 0 <= signal.reversal_reset <= 1 or not 0 < signal.state_scale_quantile <= 1 or not 0 <= signal.no_trade_band < 1:
        raise StrategyConfigurationError("signal reset, quantile, or no-trade band is outside its valid range")
    if signal.scale_shrinkage_k < 0 or signal.minimum_state_scale <= 0:
        raise StrategyConfigurationError("signal scale shrinkage must be non-negative and its floor positive")

    _reject_unknown(prices_raw, set(PriceSettings.__dataclass_fields__), "prices")
    execution_field = _choice(prices_raw, "execution_field", {"adjusted_open", "adjusted_close"}, "adjusted_open", "prices")
    return_convention = _choice(prices_raw, "return_convention", {"open_to_open", "close_to_close"}, "open_to_open", "prices")
    prices = PriceSettings(
        panel_path=_path(prices_raw.get("panel_path")),
        manifest_path=_path(prices_raw.get("manifest_path")),
        execution_field=execution_field,  # type: ignore[arg-type]
        return_convention=return_convention,  # type: ignore[arg-type]
        volatility_window_sessions=_integer(prices_raw, "volatility_window_sessions", 20, "prices"),
        volatility_minimum_sessions=_integer(prices_raw, "volatility_minimum_sessions", 15, "prices"),
        volatility_floor=_number(prices_raw, "volatility_floor", 0.005, "prices"),
        fail_on_unsupported_adjustment=_boolean(prices_raw, "fail_on_unsupported_adjustment", True, "prices"),
    )
    if (prices.execution_field, prices.return_convention) != ("adjusted_open", "open_to_open"):
        raise StrategyConfigurationError(
            "strategy pipeline v1 requires adjusted_open with open_to_open returns"
        )
    if prices.volatility_window_sessions < 2 or not 1 <= prices.volatility_minimum_sessions <= prices.volatility_window_sessions:
        raise StrategyConfigurationError("prices volatility history must satisfy 1 <= minimum <= window")
    if prices.volatility_floor <= 0:
        raise StrategyConfigurationError("prices.volatility_floor must be positive")
    if not prices.fail_on_unsupported_adjustment:
        raise StrategyConfigurationError("prices.fail_on_unsupported_adjustment must remain true")

    _reject_unknown(execution_raw, set(ExecutionSettings.__dataclass_fields__), "execution")
    execution = ExecutionSettings(
        fill_rule=_string(execution_raw, "fill_rule", section="execution") or "next_eligible_open",
        cost_bps_per_side=_number(execution_raw, "cost_bps_per_side", 10.0, "execution"),
        allow_same_session_open_only_if_available_before_open=_boolean(
            execution_raw, "allow_same_session_open_only_if_available_before_open", True, "execution"
        ),
    )
    if execution.fill_rule != "next_eligible_open" or execution.cost_bps_per_side < 0:
        raise StrategyConfigurationError("execution requires next_eligible_open and a non-negative cost")
    if not execution.allow_same_session_open_only_if_available_before_open:
        raise StrategyConfigurationError(
            "execution.allow_same_session_open_only_if_available_before_open must remain true in pipeline v1"
        )

    _reject_unknown(portfolio_raw, set(PortfolioSettings.__dataclass_fields__), "portfolio")
    portfolio = PortfolioSettings(
        mode=_string(portfolio_raw, "mode", section="portfolio") or "long_short",
        gross_limit=_number(portfolio_raw, "gross_limit", 1.0, "portfolio"),
        net_limit=_number(portfolio_raw, "net_limit", 0.20, "portfolio"),
        single_name_limit=_number(portfolio_raw, "single_name_limit", 0.05, "portfolio"),
        volatility_scaling=_boolean(portfolio_raw, "volatility_scaling", True, "portfolio"),
        preserve_signal_sign=_boolean(portfolio_raw, "preserve_signal_sign", True, "portfolio"),
    )
    if portfolio.mode != "long_short" or min(portfolio.gross_limit, portfolio.net_limit, portfolio.single_name_limit) <= 0:
        raise StrategyConfigurationError("portfolio v1 requires positive long-short exposure limits")
    if portfolio.net_limit > portfolio.gross_limit or portfolio.single_name_limit > portfolio.gross_limit:
        raise StrategyConfigurationError("portfolio net and single-name limits cannot exceed the gross limit")
    if not portfolio.volatility_scaling or not portfolio.preserve_signal_sign:
        raise StrategyConfigurationError("portfolio v1 requires volatility scaling and signal-sign preservation")

    _reject_unknown(tuning_raw, set(TuningSettings.__dataclass_fields__) - {"candidate_count"}, "tuning")
    tuning = TuningSettings(
        method=_string(tuning_raw, "method", section="tuning") or "expanding_chronological",
        folds=_integer(tuning_raw, "folds", 3, "tuning"),
        minimum_training_sessions=_integer(tuning_raw, "minimum_training_sessions", 60, "tuning"),
        validation_sessions=_integer(tuning_raw, "validation_sessions", 20, "tuning"),
        minimum_validation_sessions=_integer(tuning_raw, "minimum_validation_sessions", 10, "tuning"),
        half_life_grid=_number_tuple(tuning_raw, "half_life_grid", (1, 2, 3, 5, 10), "tuning"),
        state_scale_quantile_grid=_number_tuple(
            tuning_raw, "state_scale_quantile_grid", (0.50, 0.75, 0.90), "tuning"
        ),
        no_trade_band_grid=_number_tuple(tuning_raw, "no_trade_band_grid", (0.0, 0.10, 0.20), "tuning"),
        stability_penalty=_number(tuning_raw, "stability_penalty", 0.5, "tuning"),
        selection_metric=_string(tuning_raw, "selection_metric", section="tuning") or "median_sharpe_minus_iqr_penalty",
        maximum_exposure_violations=_integer(tuning_raw, "maximum_exposure_violations", 0, "tuning"),
        minimum_supported_stocks=_integer(tuning_raw, "minimum_supported_stocks", 25, "tuning"),
        minimum_active_validation_days=_integer(tuning_raw, "minimum_active_validation_days", 10, "tuning"),
    )
    if tuning.method != "expanding_chronological" or tuning.selection_metric != "median_sharpe_minus_iqr_penalty":
        raise StrategyConfigurationError("tuning v1 requires expanding chronological median-Sharpe/IQR selection")
    if not 2 <= tuning.folds <= 3 or min(
        tuning.minimum_training_sessions,
        tuning.validation_sessions,
        tuning.minimum_validation_sessions,
        tuning.minimum_supported_stocks,
        tuning.minimum_active_validation_days,
    ) < 1:
        raise StrategyConfigurationError("tuning folds and minimum sample counts are invalid")
    if tuning.minimum_validation_sessions > tuning.validation_sessions:
        raise StrategyConfigurationError("tuning.minimum_validation_sessions cannot exceed validation_sessions")
    if tuning.stability_penalty < 0 or tuning.maximum_exposure_violations < 0:
        raise StrategyConfigurationError("tuning penalties and allowed violations cannot be negative")
    if any(value <= 0 for value in tuning.half_life_grid):
        raise StrategyConfigurationError("tuning.half_life_grid values must be positive")
    if any(not 0 < value <= 1 for value in tuning.state_scale_quantile_grid):
        raise StrategyConfigurationError("tuning.state_scale_quantile_grid values must be in (0, 1]")
    if any(not 0 <= value < 1 for value in tuning.no_trade_band_grid):
        raise StrategyConfigurationError("tuning.no_trade_band_grid values must be in [0, 1)")
    for field_name in ("half_life_grid", "state_scale_quantile_grid", "no_trade_band_grid"):
        values = getattr(tuning, field_name)
        if len(values) != len(set(values)):
            raise StrategyConfigurationError(f"tuning.{field_name} cannot contain duplicate candidates")

    _reject_unknown(inference_raw, set(InferenceSettings.__dataclass_fields__), "inference")
    inference = InferenceSettings(
        method=_string(inference_raw, "method", section="inference") or "paired_block_bootstrap",
        block_length_sessions=_integer(inference_raw, "block_length_sessions", 5, "inference"),
        replications=_integer(inference_raw, "replications", 2000, "inference"),
        seed=_integer(inference_raw, "seed", 20260715, "inference"),
    )
    if inference.method != "paired_block_bootstrap" or min(inference.block_length_sessions, inference.replications) < 1:
        raise StrategyConfigurationError("inference requires a positive paired block bootstrap")

    _reject_unknown(reporting_raw, set(ReportingSettings.__dataclass_fields__), "reporting")
    reporting = ReportingSettings(
        write_intermediate_artifacts=_boolean(reporting_raw, "write_intermediate_artifacts", True, "reporting"),
        include_gross_and_net=_boolean(reporting_raw, "include_gross_and_net", True, "reporting"),
        include_stock_diagnostics=_boolean(reporting_raw, "include_stock_diagnostics", True, "reporting"),
        include_leave_one_stock_out=_boolean(reporting_raw, "include_leave_one_stock_out", True, "reporting"),
    )
    if not reporting.include_gross_and_net:
        raise StrategyConfigurationError("reporting.include_gross_and_net must remain true")

    _reject_unknown(agreement_raw, {"enabled"}, "agreement_conditioning")
    agreement = AgreementConditioningSettings(enabled=_boolean(agreement_raw, "enabled", False, "agreement_conditioning"))
    if agreement.enabled:
        raise StrategyConfigurationError("agreement conditioning is reserved for a later pipeline version and must remain disabled")

    _reject_unknown(outputs_raw, {"derived_root", "results_root"}, "outputs")
    outputs = OutputSettings(
        derived_root=_path(outputs_raw.get("derived_root")) or FORMAL_DERIVED_ROOT,
        results_root=_path(outputs_raw.get("results_root")) or FORMAL_RESULTS_ROOT,
    )
    if outputs.derived_root == outputs.results_root:
        raise StrategyConfigurationError("outputs.derived_root and outputs.results_root must be separate")
    if run.mode == "formal" and not _uses_isolated_formal_roots(outputs):
        raise StrategyConfigurationError(
            "formal strategy outputs must remain under Data/derived/strategy_research and results/strategy_research"
        )

    config = StrategyResearchConfig(
        run=run,
        data=data,
        scoring=scoring,
        signal=signal,
        prices=prices,
        execution=execution,
        portfolio=portfolio,
        tuning=tuning,
        inference=inference,
        reporting=reporting,
        agreement_conditioning=agreement,
        outputs=outputs,
    )
    if config.run.mode == "formal" and (config.data.events_path is not None or config.scoring.scores_path is not None):
        raise StrategyConfigurationError("prebuilt event and score paths are allowed only in smoke mode")
    if config.run.mode == "formal" and config.scoring.provider == "fixture":
        raise StrategyConfigurationError("formal mode cannot use the synthetic fixture scoring provider")
    return config


def _table(raw: Mapping[str, Any], name: str, *, required: bool = False) -> dict[str, Any]:
    value = raw.get(name)
    if value is None:
        if required:
            raise StrategyConfigurationError(f"[{name}] table is required")
        return {}
    if not isinstance(value, dict):
        raise StrategyConfigurationError(f"[{name}] must be a TOML table")
    return value


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        prefix = "" if section == "root" else f"{section}."
        raise StrategyConfigurationError(f"unknown configuration key(s): {', '.join(prefix + key for key in unknown)}")


def _string(raw: Mapping[str, Any], key: str, *, section: str, required: bool = False) -> str:
    value = raw.get(key)
    if value is None:
        if required:
            raise StrategyConfigurationError(f"{section}.{key} is required")
        return ""
    if not isinstance(value, str):
        raise StrategyConfigurationError(f"{section}.{key} must be a string")
    value = value.strip()
    if required and not value:
        raise StrategyConfigurationError(f"{section}.{key} cannot be blank")
    return value


def _choice(raw: Mapping[str, Any], key: str, choices: set[str], default: str, section: str) -> str:
    value = _string(raw, key, section=section) or default
    if value not in choices:
        raise StrategyConfigurationError(f"{section}.{key} must be one of {sorted(choices)}, got {value!r}")
    return value


def _boolean(raw: Mapping[str, Any], key: str, default: bool, section: str) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise StrategyConfigurationError(f"{section}.{key} must be true or false")
    return value


def _integer(raw: Mapping[str, Any], key: str, default: int, section: str) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrategyConfigurationError(f"{section}.{key} must be an integer")
    return value


def _number(raw: Mapping[str, Any], key: str, default: float, section: str) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyConfigurationError(f"{section}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise StrategyConfigurationError(f"{section}.{key} must be finite")
    return result


def _number_tuple(raw: Mapping[str, Any], key: str, default: tuple[float, ...], section: str) -> tuple[float, ...]:
    value = raw.get(key, list(default))
    if not isinstance(value, list) or not value:
        raise StrategyConfigurationError(f"{section}.{key} must be a non-empty list of numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise StrategyConfigurationError(f"{section}.{key} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise StrategyConfigurationError(f"{section}.{key} must contain only finite numbers")
        result.append(number)
    return tuple(result)


def _path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise StrategyConfigurationError("configured paths must be non-empty strings")
    return Path(value.strip())


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    return value


def _safe_component(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not _RUN_ID.fullmatch(normalized):
        raise StrategyConfigurationError(f"{field_name} must be a safe path component")
    return normalized
