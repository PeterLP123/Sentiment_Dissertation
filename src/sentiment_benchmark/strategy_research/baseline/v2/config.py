"""Strict frozen configuration for the v2 sentiment trading baseline.

The v2 protocol changes exactly four things relative to v1, each predeclared
here and enforced by the loader:

1. The comparator is canonical compound-threshold VADER (``+/-0.05``), loaded
   from its own score artifact, instead of the degenerate share-argmax variant.
2. Portfolios require at least two names per side (the reference paper's rule),
   capping single-name weight at a quarter of NAV.
3. A predeclared firm-session (event-level) bootstrap is the primary inference,
   with the portfolio ledger retained as the economics illustration.
4. Two predeclared signal variants are reported alongside the primary:
   magnitude-weighted FinBERT legs and an agreement-conditioned arm that trades
   only firm-sessions where FinBERT and canonical VADER agree.

Everything else (sample, screening, buffer, holding period, costs, universe)
is byte-identical to the v1 contract. The v2 protocol was specified after the
v1 result on this cohort was observed; the loader forces that admission to be
recorded so no run can present itself as confirmatory.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ....artifact_io import canonical_json, sha256_text
from ..config import (
    FINBERT_V1_REVISION,
    DataConfig,
    OutputConfig,
    PriceConfig,
    ReplicationConfig,
    ScreeningConfig,
    _bool,
    _integer,
    _number,
    _path,
    _required_string,
    _table,
)


class BaselineV2ConfigurationError(ValueError):
    """Raised when a v2 baseline configuration violates the frozen contract."""


V2_ARMS = ("finbert", "vader_compound", "finbert_magnitude", "agreement")
V2_COST_GRID_BPS = (5.0, 10.0, 20.0)
V2_SEED = 20_260_721


@dataclass(frozen=True)
class RunV2Config:
    id: str
    evaluation_start: date
    sample_status: str
    specified_after_v1_results: bool

    def __post_init__(self) -> None:
        import re

        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,79}", self.id):
            raise BaselineV2ConfigurationError("[run].id must be a lowercase hyphenated identifier")
        if self.sample_status != "previously_explored":
            raise BaselineV2ConfigurationError("the v2 cohort is previously explored; sample_status is frozen")
        if self.specified_after_v1_results is not True:
            raise BaselineV2ConfigurationError(
                "the v2 protocol was specified after v1 results were observed; "
                "specified_after_v1_results must be recorded as true"
            )


@dataclass(frozen=True)
class ScorerV2Config:
    id: str
    role: str
    model_id: str
    revision: str | None
    score_definition: str

    def __post_init__(self) -> None:
        if self.id not in {"finbert", "vader_compound"}:
            raise BaselineV2ConfigurationError(f"unsupported v2 scorer: {self.id!r}")
        if self.role not in {"primary", "comparator"}:
            raise BaselineV2ConfigurationError(f"invalid scorer role: {self.role!r}")
        if self.id == "finbert":
            if self.model_id != "ProsusAI/finbert":
                raise BaselineV2ConfigurationError("FinBERT model_id must be 'ProsusAI/finbert'")
            if self.revision != FINBERT_V1_REVISION:
                raise BaselineV2ConfigurationError(
                    f"the v2 FinBERT revision must remain exactly {FINBERT_V1_REVISION}"
                )
            if self.score_definition != "p_positive_minus_p_negative":
                raise BaselineV2ConfigurationError("FinBERT score definition must be p_positive_minus_p_negative")
        else:
            if self.model_id != "nltk.sentiment.vader":
                raise BaselineV2ConfigurationError("the v2 comparator model_id must be 'nltk.sentiment.vader'")
            if self.revision is not None:
                raise BaselineV2ConfigurationError("the VADER-compound comparator does not take a revision")
            if self.score_definition != "vader_compound_threshold_0p05":
                raise BaselineV2ConfigurationError(
                    "the v2 comparator is canonical compound-threshold VADER at 0.05"
                )


@dataclass(frozen=True)
class DataV2Config:
    base: DataConfig
    vader_compound_path: Path
    vader_compound_manifest: Path


@dataclass(frozen=True)
class SignalV2Config:
    arms: tuple[str, ...]
    minimum_events: int
    holding_sessions: int
    carry: str

    def __post_init__(self) -> None:
        if self.arms != V2_ARMS:
            raise BaselineV2ConfigurationError(f"the v2 arms are frozen to {list(V2_ARMS)}")
        if self.minimum_events != 1 or self.holding_sessions != 1 or self.carry != "none":
            raise BaselineV2ConfigurationError("the v2 signal is frozen to one event, one session, no carry")


@dataclass(frozen=True)
class PortfolioV2Config:
    construction: str
    gross_exposure: float
    minimum_names_per_side: int
    initial_nav_usd: float
    cost_bps_per_side: float
    cost_grid_bps: tuple[float, ...]
    force_final_liquidation: bool

    def __post_init__(self) -> None:
        if self.construction != "dollar_neutral_two_name_minimum":
            raise BaselineV2ConfigurationError("portfolio construction must be dollar_neutral_two_name_minimum")
        if not math.isclose(self.gross_exposure, 1.0):
            raise BaselineV2ConfigurationError("the v2 gross exposure is frozen at 1.0")
        if self.minimum_names_per_side != 2:
            raise BaselineV2ConfigurationError("the v2 baseline requires at least two names per side")
        if not math.isclose(self.initial_nav_usd, 1_000_000.0):
            raise BaselineV2ConfigurationError("the v2 initial NAV is frozen at $1,000,000")
        if not math.isclose(self.cost_bps_per_side, 10.0):
            raise BaselineV2ConfigurationError("the v2 headline cost assumption is frozen at 10 bps per side")
        if tuple(self.cost_grid_bps) != V2_COST_GRID_BPS:
            raise BaselineV2ConfigurationError(f"the v2 reporting cost grid is frozen to {list(V2_COST_GRID_BPS)}")
        if not self.force_final_liquidation:
            raise BaselineV2ConfigurationError("the v2 baseline must liquidate at the final open")

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10_000


@dataclass(frozen=True)
class InferenceV2Config:
    block_length_sessions: int
    replications: int
    seed: int
    event_level_primary: bool

    def __post_init__(self) -> None:
        if self.block_length_sessions != 5 or self.replications != 2_000 or self.seed != V2_SEED:
            raise BaselineV2ConfigurationError("the v2 bootstrap block length, replications, and seed are frozen")
        if self.event_level_primary is not True:
            raise BaselineV2ConfigurationError("the v2 primary inference is the firm-session event-level bootstrap")


@dataclass(frozen=True)
class LiteratureBaselineV2Config:
    path: Path
    run: RunV2Config
    replication: ReplicationConfig
    data: DataV2Config
    prices: PriceConfig
    scorers: tuple[ScorerV2Config, ...]
    screening: ScreeningConfig
    signal: SignalV2Config
    portfolio: PortfolioV2Config
    inference: InferenceV2Config
    outputs: OutputConfig

    def __post_init__(self) -> None:
        if len(self.scorers) != 2 or {item.id for item in self.scorers} != {"finbert", "vader_compound"}:
            raise BaselineV2ConfigurationError("the v2 baseline requires exactly FinBERT and VADER-compound")
        if sum(item.role == "primary" for item in self.scorers) != 1:
            raise BaselineV2ConfigurationError("exactly one scorer must have role='primary'")
        primary = next(item for item in self.scorers if item.role == "primary")
        if primary.id != "finbert":
            raise BaselineV2ConfigurationError("FinBERT must remain the predeclared primary scorer")
        comparator = next(item for item in self.scorers if item.role == "comparator")
        if comparator.score_definition != "vader_compound_threshold_0p05":
            raise BaselineV2ConfigurationError("the v2 comparator is frozen to compound thresholds at 0.05")
        if self.data.base.calendar != "XNYS" or self.data.base.timezone != "America/New_York":
            raise BaselineV2ConfigurationError("the v2 baseline is frozen to XNYS/America/New_York")
        if self.data.base.processing_buffer_minutes != 15:
            raise BaselineV2ConfigurationError("the v2 processing buffer is frozen at 15 minutes")
        if self.screening != ScreeningConfig(True, True, True):
            raise BaselineV2ConfigurationError("the v2 screening flags are frozen to explicit, single-target, non-technical")
        if self.outputs.derived_root != Path("Data/derived/strategy_research/baselines"):
            raise BaselineV2ConfigurationError("derived_root must preserve the isolated baseline namespace")
        if self.outputs.results_root != Path("results/strategy_research/baselines"):
            raise BaselineV2ConfigurationError("results_root must preserve the isolated baseline namespace")

    @property
    def primary_scorer(self) -> ScorerV2Config:
        return next(item for item in self.scorers if item.role == "primary")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "run": {
                "id": self.run.id,
                "evaluation_start": self.run.evaluation_start.isoformat(),
                "sample_status": self.run.sample_status,
                "specified_after_v1_results": self.run.specified_after_v1_results,
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
                "collection_root": self.data.base.collection_root.as_posix(),
                "corpus_manifest": self.data.base.corpus_manifest.as_posix(),
                "score_path": self.data.base.score_path.as_posix(),
                "score_manifest": self.data.base.score_manifest.as_posix(),
                "vader_compound_path": self.data.vader_compound_path.as_posix(),
                "vader_compound_manifest": self.data.vader_compound_manifest.as_posix(),
                "processing_buffer_minutes": self.data.base.processing_buffer_minutes,
                "calendar": self.data.base.calendar,
                "timezone": self.data.base.timezone,
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
                "arms": list(self.signal.arms),
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
                "cost_grid_bps": list(self.portfolio.cost_grid_bps),
                "force_final_liquidation": self.portfolio.force_final_liquidation,
            },
            "inference": {
                "block_length_sessions": self.inference.block_length_sessions,
                "replications": self.inference.replications,
                "seed": self.inference.seed,
                "event_level_primary": self.inference.event_level_primary,
            },
            "outputs": {
                "derived_root": self.outputs.derived_root.as_posix(),
                "results_root": self.outputs.results_root.as_posix(),
            },
        }

    @property
    def config_sha256(self) -> str:
        return sha256_text(canonical_json(self.to_payload()))


def load_baseline_v2_config(path: str | Path) -> LiteratureBaselineV2Config:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BaselineV2ConfigurationError(f"cannot load baseline v2 config {config_path}: {exc}") from exc
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
        raise BaselineV2ConfigurationError(f"unknown top-level sections: {unknown_sections}")
    try:
        run = _table(payload, "run", {"id", "evaluation_start", "sample_status", "specified_after_v1_results"})
        try:
            evaluation_start = date.fromisoformat(_required_string(run, "evaluation_start", "run"))
        except ValueError as exc:
            raise BaselineV2ConfigurationError("[run].evaluation_start must be an ISO date") from exc
        replication = _table(
            payload,
            "replication",
            {"paper_id", "doi", "reference_version", "reference_url", "fidelity", "deviations"},
        )
        deviations = replication.get("deviations")
        if not isinstance(deviations, list) or not all(isinstance(item, str) and item.strip() for item in deviations):
            raise BaselineV2ConfigurationError("[replication].deviations must be a non-empty string list")
        data = _table(
            payload,
            "data",
            {
                "collection_root",
                "corpus_manifest",
                "score_path",
                "score_manifest",
                "vader_compound_path",
                "vader_compound_manifest",
                "processing_buffer_minutes",
                "calendar",
                "timezone",
            },
        )
        prices = _table(payload, "prices", {"panel_path", "manifest_path", "execution_field", "return_convention"})
        scorer_payloads = payload.get("scorers")
        if not isinstance(scorer_payloads, list) or not scorer_payloads:
            raise BaselineV2ConfigurationError("at least one [[scorers]] table is required")
        scorers: list[ScorerV2Config] = []
        for index, item in enumerate(scorer_payloads, start=1):
            if not isinstance(item, dict):
                raise BaselineV2ConfigurationError(f"scorer {index} must be a table")
            unknown = sorted(set(item) - {"id", "role", "model_id", "revision", "score_definition"})
            if unknown:
                raise BaselineV2ConfigurationError(f"unknown scorer {index} keys: {unknown}")
            role = _required_string(item, "role", f"scorers[{index}]")
            if role not in {"primary", "comparator"}:
                raise BaselineV2ConfigurationError(f"invalid scorer role: {role!r}")
            revision = item.get("revision")
            if revision is not None and (not isinstance(revision, str) or not revision.strip()):
                raise BaselineV2ConfigurationError(f"scorer {index} revision must be a string when set")
            scorers.append(
                ScorerV2Config(
                    id=_required_string(item, "id", f"scorers[{index}]"),
                    role=role,
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
        signal = _table(payload, "signal", {"arms", "minimum_events", "holding_sessions", "carry"})
        arms = signal.get("arms")
        if not isinstance(arms, list) or not all(isinstance(item, str) for item in arms):
            raise BaselineV2ConfigurationError("[signal].arms must be a string list")
        portfolio = _table(
            payload,
            "portfolio",
            {
                "construction",
                "gross_exposure",
                "minimum_names_per_side",
                "initial_nav_usd",
                "cost_bps_per_side",
                "cost_grid_bps",
                "force_final_liquidation",
            },
        )
        cost_grid = portfolio.get("cost_grid_bps")
        if not isinstance(cost_grid, list) or not all(isinstance(item, int | float) for item in cost_grid):
            raise BaselineV2ConfigurationError("[portfolio].cost_grid_bps must be a numeric list")
        inference = _table(
            payload,
            "inference",
            {"block_length_sessions", "replications", "seed", "event_level_primary"},
        )
        outputs = _table(payload, "outputs", {"derived_root", "results_root"})

        return LiteratureBaselineV2Config(
            path=config_path,
            run=RunV2Config(
                id=_required_string(run, "id", "run"),
                evaluation_start=evaluation_start,
                sample_status=_required_string(run, "sample_status", "run"),
                specified_after_v1_results=_bool(run, "specified_after_v1_results", "run"),
            ),
            replication=ReplicationConfig(
                paper_id=_required_string(replication, "paper_id", "replication"),
                doi=_required_string(replication, "doi", "replication"),
                reference_version=_required_string(replication, "reference_version", "replication"),
                reference_url=_required_string(replication, "reference_url", "replication"),
                fidelity=_required_string(replication, "fidelity", "replication"),
                deviations=tuple(item.strip() for item in deviations),
            ),
            data=DataV2Config(
                base=DataConfig(
                    collection_root=_path(data, "collection_root", "data"),
                    corpus_manifest=_path(data, "corpus_manifest", "data"),
                    score_path=_path(data, "score_path", "data"),
                    score_manifest=_path(data, "score_manifest", "data"),
                    processing_buffer_minutes=_integer(data, "processing_buffer_minutes", "data"),
                    calendar=_required_string(data, "calendar", "data"),
                    timezone=_required_string(data, "timezone", "data"),
                ),
                vader_compound_path=_path(data, "vader_compound_path", "data"),
                vader_compound_manifest=_path(data, "vader_compound_manifest", "data"),
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
            signal=SignalV2Config(
                arms=tuple(arms),
                minimum_events=_integer(signal, "minimum_events", "signal", minimum=1),
                holding_sessions=_integer(signal, "holding_sessions", "signal", minimum=1),
                carry=_required_string(signal, "carry", "signal"),
            ),
            portfolio=PortfolioV2Config(
                construction=_required_string(portfolio, "construction", "portfolio"),
                gross_exposure=_number(portfolio, "gross_exposure", "portfolio"),
                minimum_names_per_side=_integer(portfolio, "minimum_names_per_side", "portfolio", minimum=1),
                initial_nav_usd=_number(portfolio, "initial_nav_usd", "portfolio"),
                cost_bps_per_side=_number(portfolio, "cost_bps_per_side", "portfolio"),
                cost_grid_bps=tuple(float(item) for item in cost_grid),
                force_final_liquidation=_bool(portfolio, "force_final_liquidation", "portfolio"),
            ),
            inference=InferenceV2Config(
                block_length_sessions=_integer(inference, "block_length_sessions", "inference", minimum=1),
                replications=_integer(inference, "replications", "inference", minimum=1),
                seed=_integer(inference, "seed", "inference"),
                event_level_primary=_bool(inference, "event_level_primary", "inference"),
            ),
            outputs=OutputConfig(
                derived_root=_path(outputs, "derived_root", "outputs"),
                results_root=_path(outputs, "results_root", "outputs"),
            ),
        )
    except BaselineV2ConfigurationError:
        raise
    except ValueError as exc:
        # The reused v1 table/field helpers raise BaselineConfigurationError,
        # which is a ValueError; re-badge so callers see one error type.
        raise BaselineV2ConfigurationError(str(exc)) from exc
