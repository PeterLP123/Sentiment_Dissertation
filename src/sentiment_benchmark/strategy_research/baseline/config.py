"""Strict configuration contract for the literature trading baseline."""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from ...artifact_io import canonical_json, sha256_text

SampleStatus = Literal["previously_explored"]
ScorerRole = Literal["primary", "comparator"]
FINBERT_V1_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"


class BaselineConfigurationError(ValueError):
    """Raised when the frozen baseline configuration is incomplete or ambiguous."""


def _table(payload: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise BaselineConfigurationError(f"missing [{name}] table")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BaselineConfigurationError(f"unknown [{name}] keys: {unknown}")
    return value


def _required_string(table: dict[str, Any], key: str, section: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BaselineConfigurationError(f"[{section}].{key} must be a non-empty string")
    return value.strip()


def _path(table: dict[str, Any], key: str, section: str) -> Path:
    return Path(_required_string(table, key, section))


def _bool(table: dict[str, Any], key: str, section: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise BaselineConfigurationError(f"[{section}].{key} must be true or false")
    return value


def _integer(table: dict[str, Any], key: str, section: str, *, minimum: int = 0) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BaselineConfigurationError(f"[{section}].{key} must be an integer >= {minimum}")
    return value


def _number(table: dict[str, Any], key: str, section: str, *, minimum: float = 0.0) -> float:
    value = table.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise BaselineConfigurationError(f"[{section}].{key} must be a number >= {minimum}")
    return float(value)


@dataclass(frozen=True)
class RunConfig:
    id: str
    evaluation_start: date
    sample_status: SampleStatus

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,79}", self.id):
            raise BaselineConfigurationError("[run].id must be a lowercase hyphenated identifier")
        if self.sample_status != "previously_explored":
            raise BaselineConfigurationError(
                "the initial baseline must declare sample_status='previously_explored'"
            )


@dataclass(frozen=True)
class ReplicationConfig:
    paper_id: str
    doi: str
    reference_version: str
    reference_url: str
    fidelity: str
    deviations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fidelity != "conceptual_adaptation":
            raise BaselineConfigurationError(
                "[replication].fidelity must be 'conceptual_adaptation'; this is not an exact replication"
            )
        if not self.deviations:
            raise BaselineConfigurationError("[replication].deviations must document at least one deviation")
        if self.reference_version != "arXiv:2304.07619v6":
            raise BaselineConfigurationError("the v1 reference must be pinned to arXiv:2304.07619v6")
        if self.reference_url != "https://arxiv.org/pdf/2304.07619v6":
            raise BaselineConfigurationError("the v1 reference URL must be the immutable arXiv v6 PDF")


@dataclass(frozen=True)
class DataConfig:
    collection_root: Path
    corpus_manifest: Path
    score_path: Path
    score_manifest: Path
    processing_buffer_minutes: int
    calendar: str
    timezone: str


@dataclass(frozen=True)
class PriceConfig:
    panel_path: Path
    manifest_path: Path
    execution_field: str
    return_convention: str

    def __post_init__(self) -> None:
        if self.execution_field != "adjusted_open" or self.return_convention != "open_to_open":
            raise BaselineConfigurationError(
                "prices must use execution_field='adjusted_open' and return_convention='open_to_open'"
            )


@dataclass(frozen=True)
class ScorerConfig:
    id: str
    role: ScorerRole
    model_id: str
    revision: str | None
    score_definition: str

    def __post_init__(self) -> None:
        if self.id not in {"finbert", "vader"}:
            raise BaselineConfigurationError(f"unsupported baseline scorer: {self.id!r}")
        if self.role not in {"primary", "comparator"}:
            raise BaselineConfigurationError(f"invalid scorer role: {self.role!r}")
        if self.id == "finbert":
            if self.model_id != "ProsusAI/finbert":
                raise BaselineConfigurationError("FinBERT model_id must be 'ProsusAI/finbert'")
            if self.revision != FINBERT_V1_REVISION:
                raise BaselineConfigurationError(
                    f"the v1 FinBERT revision must be exactly {FINBERT_V1_REVISION}"
                )
            if self.score_definition != "p_positive_minus_p_negative":
                raise BaselineConfigurationError("FinBERT score definition must be p_positive_minus_p_negative")
        elif self.score_definition != "vader_pos_minus_neg":
            raise BaselineConfigurationError(
                "the reusable VADER share-argmax artifact stores pos - neg, not canonical VADER compound"
            )


@dataclass(frozen=True)
class ScreeningConfig:
    require_explicit_target: bool
    require_single_target: bool
    exclude_market_price_technical: bool


@dataclass(frozen=True)
class SignalConfig:
    aggregation: str
    minimum_events: int
    holding_sessions: int
    carry: str

    def __post_init__(self) -> None:
        if self.aggregation != "sign_mean_hard_label":
            raise BaselineConfigurationError("signal aggregation must be sign_mean_hard_label")
        if self.minimum_events < 1:
            raise BaselineConfigurationError("signal minimum_events must be positive")
        if self.holding_sessions != 1 or self.carry != "none":
            raise BaselineConfigurationError("the baseline is frozen to one session with no carry")


@dataclass(frozen=True)
class PortfolioConfig:
    construction: str
    gross_exposure: float
    minimum_names_per_side: int
    initial_nav_usd: float
    cost_bps_per_side: float
    force_final_liquidation: bool

    def __post_init__(self) -> None:
        if self.construction != "equal_weight_dollar_neutral":
            raise BaselineConfigurationError("portfolio construction must be equal_weight_dollar_neutral")
        if not 0 < self.gross_exposure <= 1:
            raise BaselineConfigurationError("gross_exposure must be in (0, 1]")
        if self.minimum_names_per_side < 1:
            raise BaselineConfigurationError("minimum_names_per_side must be positive")
        if self.initial_nav_usd <= 0:
            raise BaselineConfigurationError("initial_nav_usd must be positive")
        if self.cost_bps_per_side < 0:
            raise BaselineConfigurationError("cost_bps_per_side cannot be negative")

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10_000


@dataclass(frozen=True)
class InferenceConfig:
    block_length_sessions: int
    replications: int
    seed: int

    def __post_init__(self) -> None:
        if min(self.block_length_sessions, self.replications) < 1 or self.seed < 0:
            raise BaselineConfigurationError("inference block length/replications must be positive and seed non-negative")


@dataclass(frozen=True)
class OutputConfig:
    derived_root: Path
    results_root: Path


@dataclass(frozen=True)
class LiteratureBaselineConfig:
    path: Path
    run: RunConfig
    replication: ReplicationConfig
    data: DataConfig
    prices: PriceConfig
    scorers: tuple[ScorerConfig, ...]
    screening: ScreeningConfig
    signal: SignalConfig
    portfolio: PortfolioConfig
    inference: InferenceConfig
    outputs: OutputConfig

    def __post_init__(self) -> None:
        if len(self.scorers) != 2 or {item.id for item in self.scorers} != {"finbert", "vader"}:
            raise BaselineConfigurationError("the v1 baseline requires exactly FinBERT and VADER")
        if sum(item.role == "primary" for item in self.scorers) != 1:
            raise BaselineConfigurationError("exactly one scorer must have role='primary'")
        primary = next(item for item in self.scorers if item.role == "primary")
        if primary.id != "finbert":
            raise BaselineConfigurationError("FinBERT must remain the predeclared primary scorer")
        if self.data.calendar != "XNYS" or self.data.timezone != "America/New_York":
            raise BaselineConfigurationError("the v1 baseline is frozen to XNYS/America/New_York")
        if self.data.processing_buffer_minutes != 15:
            raise BaselineConfigurationError("the v1 baseline processing buffer is frozen at 15 minutes")
        if self.screening != ScreeningConfig(True, True, True):
            raise BaselineConfigurationError("the v1 screening flags are frozen to explicit, single-target, non-technical events")
        if self.signal != SignalConfig("sign_mean_hard_label", 1, 1, "none"):
            raise BaselineConfigurationError("the v1 signal contract cannot be changed in place")
        if (
            self.portfolio.construction != "equal_weight_dollar_neutral"
            or not math.isclose(self.portfolio.gross_exposure, 1.0)
            or self.portfolio.minimum_names_per_side != 1
            or not math.isclose(self.portfolio.initial_nav_usd, 1_000_000.0)
            or not math.isclose(self.portfolio.cost_bps_per_side, 10.0)
            or not self.portfolio.force_final_liquidation
        ):
            raise BaselineConfigurationError("the v1 portfolio, NAV, cost, and liquidation rules are frozen")
        if self.inference != InferenceConfig(5, 2_000, 20_260_718):
            raise BaselineConfigurationError("the v1 bootstrap length, replications, and seed are frozen")
        if self.outputs.derived_root != Path("Data/derived/strategy_research/baselines"):
            raise BaselineConfigurationError("derived_root must preserve the isolated baseline namespace")
        if self.outputs.results_root != Path("results/strategy_research/baselines"):
            raise BaselineConfigurationError("results_root must preserve the isolated baseline namespace")

    @property
    def primary_scorer(self) -> ScorerConfig:
        return next(item for item in self.scorers if item.role == "primary")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run": {
                "id": self.run.id,
                "evaluation_start": self.run.evaluation_start.isoformat(),
                "sample_status": self.run.sample_status,
            },
            "replication": {
                "paper_id": self.replication.paper_id,
                "doi": self.replication.doi,
                "reference_version": self.replication.reference_version,
                "reference_url": self.replication.reference_url,
                "fidelity": self.replication.fidelity,
                "deviations": list(self.replication.deviations),
            },
            "data": {
                "collection_root": self.data.collection_root.as_posix(),
                "corpus_manifest": self.data.corpus_manifest.as_posix(),
                "score_path": self.data.score_path.as_posix(),
                "score_manifest": self.data.score_manifest.as_posix(),
                "processing_buffer_minutes": self.data.processing_buffer_minutes,
                "calendar": self.data.calendar,
                "timezone": self.data.timezone,
            },
            "prices": {
                "panel_path": self.prices.panel_path.as_posix(),
                "manifest_path": self.prices.manifest_path.as_posix(),
                "execution_field": self.prices.execution_field,
                "return_convention": self.prices.return_convention,
            },
            "scorers": [
                {
                    "id": item.id,
                    "role": item.role,
                    "model_id": item.model_id,
                    "revision": item.revision,
                    "score_definition": item.score_definition,
                }
                for item in self.scorers
            ],
            "screening": {
                "require_explicit_target": self.screening.require_explicit_target,
                "require_single_target": self.screening.require_single_target,
                "exclude_market_price_technical": self.screening.exclude_market_price_technical,
            },
            "signal": {
                "aggregation": self.signal.aggregation,
                "minimum_events": self.signal.minimum_events,
                "holding_sessions": self.signal.holding_sessions,
                "carry": self.signal.carry,
            },
            "portfolio": {
                "construction": self.portfolio.construction,
                "gross_exposure": self.portfolio.gross_exposure,
                "minimum_names_per_side": self.portfolio.minimum_names_per_side,
                "initial_nav_usd": self.portfolio.initial_nav_usd,
                "cost_bps_per_side": self.portfolio.cost_bps_per_side,
                "force_final_liquidation": self.portfolio.force_final_liquidation,
            },
            "inference": {
                "block_length_sessions": self.inference.block_length_sessions,
                "replications": self.inference.replications,
                "seed": self.inference.seed,
            },
            "outputs": {
                "derived_root": self.outputs.derived_root.as_posix(),
                "results_root": self.outputs.results_root.as_posix(),
            },
        }

    @property
    def config_sha256(self) -> str:
        return sha256_text(canonical_json(self.to_payload()))


def load_baseline_config(path: str | Path) -> LiteratureBaselineConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BaselineConfigurationError(f"cannot load baseline config {config_path}: {exc}") from exc
    allowed_sections = {
        "run",
        "replication",
        "data",
        "prices",
        "scorers",
        "screening",
        "signal",
        "portfolio",
        "inference",
        "outputs",
    }
    unknown_sections = sorted(set(payload) - allowed_sections)
    if unknown_sections:
        raise BaselineConfigurationError(f"unknown top-level sections: {unknown_sections}")

    run = _table(payload, "run", {"id", "evaluation_start", "sample_status"})
    try:
        evaluation_start = date.fromisoformat(_required_string(run, "evaluation_start", "run"))
    except ValueError as exc:
        raise BaselineConfigurationError("[run].evaluation_start must be an ISO date") from exc
    replication = _table(
        payload,
        "replication",
        {"paper_id", "doi", "reference_version", "reference_url", "fidelity", "deviations"},
    )
    deviations = replication.get("deviations")
    if not isinstance(deviations, list) or not all(isinstance(item, str) and item.strip() for item in deviations):
        raise BaselineConfigurationError("[replication].deviations must be a non-empty string list")
    data = _table(
        payload,
        "data",
        {
            "collection_root",
            "corpus_manifest",
            "score_path",
            "score_manifest",
            "processing_buffer_minutes",
            "calendar",
            "timezone",
        },
    )
    prices = _table(payload, "prices", {"panel_path", "manifest_path", "execution_field", "return_convention"})
    scorer_payloads = payload.get("scorers")
    if not isinstance(scorer_payloads, list) or not scorer_payloads:
        raise BaselineConfigurationError("at least one [[scorers]] table is required")
    scorers: list[ScorerConfig] = []
    for index, item in enumerate(scorer_payloads, start=1):
        if not isinstance(item, dict):
            raise BaselineConfigurationError(f"scorer {index} must be a table")
        unknown = sorted(set(item) - {"id", "role", "model_id", "revision", "score_definition"})
        if unknown:
            raise BaselineConfigurationError(f"unknown scorer {index} keys: {unknown}")
        role = _required_string(item, "role", f"scorers[{index}]")
        if role not in {"primary", "comparator"}:
            raise BaselineConfigurationError(f"invalid scorer role: {role!r}")
        revision = item.get("revision")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise BaselineConfigurationError(f"scorer {index} revision must be a string when set")
        scorers.append(
            ScorerConfig(
                id=_required_string(item, "id", f"scorers[{index}]"),
                role=role,  # type: ignore[arg-type]
                model_id=_required_string(item, "model_id", f"scorers[{index}]"),
                revision=revision.strip() if isinstance(revision, str) else None,
                score_definition=_required_string(item, "score_definition", f"scorers[{index}]"),
            )
        )
    screening = _table(
        payload,
        "screening",
        {"require_explicit_target", "require_single_target", "exclude_market_price_technical"},
    )
    signal = _table(payload, "signal", {"aggregation", "minimum_events", "holding_sessions", "carry"})
    portfolio = _table(
        payload,
        "portfolio",
        {
            "construction",
            "gross_exposure",
            "minimum_names_per_side",
            "initial_nav_usd",
            "cost_bps_per_side",
            "force_final_liquidation",
        },
    )
    inference = _table(payload, "inference", {"block_length_sessions", "replications", "seed"})
    outputs = _table(payload, "outputs", {"derived_root", "results_root"})

    return LiteratureBaselineConfig(
        path=config_path,
        run=RunConfig(
            id=_required_string(run, "id", "run"),
            evaluation_start=evaluation_start,
            sample_status=_required_string(run, "sample_status", "run"),  # type: ignore[arg-type]
        ),
        replication=ReplicationConfig(
            paper_id=_required_string(replication, "paper_id", "replication"),
            doi=_required_string(replication, "doi", "replication"),
            reference_version=_required_string(replication, "reference_version", "replication"),
            reference_url=_required_string(replication, "reference_url", "replication"),
            fidelity=_required_string(replication, "fidelity", "replication"),
            deviations=tuple(item.strip() for item in deviations),
        ),
        data=DataConfig(
            collection_root=_path(data, "collection_root", "data"),
            corpus_manifest=_path(data, "corpus_manifest", "data"),
            score_path=_path(data, "score_path", "data"),
            score_manifest=_path(data, "score_manifest", "data"),
            processing_buffer_minutes=_integer(data, "processing_buffer_minutes", "data"),
            calendar=_required_string(data, "calendar", "data"),
            timezone=_required_string(data, "timezone", "data"),
        ),
        prices=PriceConfig(
            panel_path=_path(prices, "panel_path", "prices"),
            manifest_path=_path(prices, "manifest_path", "prices"),
            execution_field=_required_string(prices, "execution_field", "prices"),
            return_convention=_required_string(prices, "return_convention", "prices"),
        ),
        scorers=tuple(scorers),
        screening=ScreeningConfig(
            require_explicit_target=_bool(screening, "require_explicit_target", "screening"),
            require_single_target=_bool(screening, "require_single_target", "screening"),
            exclude_market_price_technical=_bool(screening, "exclude_market_price_technical", "screening"),
        ),
        signal=SignalConfig(
            aggregation=_required_string(signal, "aggregation", "signal"),
            minimum_events=_integer(signal, "minimum_events", "signal", minimum=1),
            holding_sessions=_integer(signal, "holding_sessions", "signal", minimum=1),
            carry=_required_string(signal, "carry", "signal"),
        ),
        portfolio=PortfolioConfig(
            construction=_required_string(portfolio, "construction", "portfolio"),
            gross_exposure=_number(portfolio, "gross_exposure", "portfolio"),
            minimum_names_per_side=_integer(portfolio, "minimum_names_per_side", "portfolio", minimum=1),
            initial_nav_usd=_number(portfolio, "initial_nav_usd", "portfolio"),
            cost_bps_per_side=_number(portfolio, "cost_bps_per_side", "portfolio"),
            force_final_liquidation=_bool(portfolio, "force_final_liquidation", "portfolio"),
        ),
        inference=InferenceConfig(
            block_length_sessions=_integer(inference, "block_length_sessions", "inference", minimum=1),
            replications=_integer(inference, "replications", "inference", minimum=1),
            seed=_integer(inference, "seed", "inference"),
        ),
        outputs=OutputConfig(
            derived_root=_path(outputs, "derived_root", "outputs"),
            results_root=_path(outputs, "results_root", "outputs"),
        ),
    )
