"""Strict configuration for the post-v2 FinBERT rank-reversal strategy.

The v3 rules are intentionally narrow. They were specified after the 33-stock
sentiment-level pilot showed a negative five-session relation, so the loader
requires that provenance to be disclosed and forbids silent parameter changes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ....artifact_io import canonical_json, sha256_text


class FinbertV3ConfigurationError(ValueError):
    """Raised when a v3 config differs from the frozen protocol."""


@dataclass(frozen=True)
class RunConfig:
    id: str
    evaluation_start: date
    sample_status: str
    specified_after_level_reversal_results: bool
    confirmation_required: bool


@dataclass(frozen=True)
class DataConfig:
    collection_root: Path
    corpus_manifest: Path
    score_path: Path
    score_manifest: Path
    processing_buffer_minutes: int
    calendar: str
    timezone: str
    allow_legacy_revision_enforcement_metadata: bool


@dataclass(frozen=True)
class PriceConfig:
    panel_path: Path
    manifest_path: Path
    execution_field: str
    return_convention: str


@dataclass(frozen=True)
class ScorerConfig:
    id: str
    model_id: str
    revision: str
    score_definition: str


@dataclass(frozen=True)
class ScreeningConfig:
    require_explicit_target: bool
    require_single_target: bool
    exclude_market_price_technical: bool


@dataclass(frozen=True)
class SignalConfig:
    construction: str
    formation_lag_sessions: int
    lookback_sessions: int
    tail_fraction: float
    minimum_ranked_firms: int
    minimum_names_per_side: int
    direction: str


@dataclass(frozen=True)
class PortfolioConfig:
    gross_exposure: float
    initial_nav_usd: float
    cost_bps_per_side: float
    cost_grid_bps: tuple[float, ...]
    force_final_liquidation: bool


@dataclass(frozen=True)
class InferenceConfig:
    block_length_sessions: int
    replications: int
    seed: int


@dataclass(frozen=True)
class OutputConfig:
    derived_root: Path
    results_root: Path


@dataclass(frozen=True)
class FinbertV3Config:
    path: Path
    config_sha256: str
    run: RunConfig
    data: DataConfig
    prices: PriceConfig
    scorer: ScorerConfig
    screening: ScreeningConfig
    signal: SignalConfig
    portfolio: PortfolioConfig
    inference: InferenceConfig
    outputs: OutputConfig

    def to_payload(self) -> dict[str, Any]:
        return {
            "run": {
                "id": self.run.id,
                "evaluation_start": self.run.evaluation_start.isoformat(),
                "sample_status": self.run.sample_status,
                "specified_after_level_reversal_results": self.run.specified_after_level_reversal_results,
                "confirmation_required": self.run.confirmation_required,
            },
            "data": {
                "collection_root": self.data.collection_root.as_posix(),
                "corpus_manifest": self.data.corpus_manifest.as_posix(),
                "score_path": self.data.score_path.as_posix(),
                "score_manifest": self.data.score_manifest.as_posix(),
                "processing_buffer_minutes": self.data.processing_buffer_minutes,
                "calendar": self.data.calendar,
                "timezone": self.data.timezone,
                "allow_legacy_revision_enforcement_metadata": self.data.allow_legacy_revision_enforcement_metadata,
            },
            "prices": {
                "panel_path": self.prices.panel_path.as_posix(),
                "manifest_path": self.prices.manifest_path.as_posix(),
                "execution_field": self.prices.execution_field,
                "return_convention": self.prices.return_convention,
            },
            "scorer": {
                "id": self.scorer.id,
                "model_id": self.scorer.model_id,
                "revision": self.scorer.revision,
                "score_definition": self.scorer.score_definition,
            },
            "screening": self.screening.__dict__,
            "signal": self.signal.__dict__,
            "portfolio": {
                **self.portfolio.__dict__,
                "cost_grid_bps": list(self.portfolio.cost_grid_bps),
            },
            "inference": self.inference.__dict__,
            "outputs": {
                "derived_root": self.outputs.derived_root.as_posix(),
                "results_root": self.outputs.results_root.as_posix(),
            },
        }


def _table(data: dict[str, Any], name: str, keys: set[str]) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise FinbertV3ConfigurationError(f"missing [{name}] table")
    extra = set(value) - keys
    missing = keys - set(value)
    if extra or missing:
        raise FinbertV3ConfigurationError(f"[{name}] keys differ from frozen schema; missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinbertV3ConfigurationError(message)


def _date(value: Any, field: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise FinbertV3ConfigurationError(f"{field} must be an ISO date") from exc


def load_finbert_v3_config(path: str | Path) -> FinbertV3Config:
    """Load the exact frozen v3 protocol and reject unregistered variants."""

    source = Path(path)
    try:
        raw = source.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise FinbertV3ConfigurationError(f"cannot load v3 config {source}: {exc}") from exc
    expected_tables = {"run", "data", "prices", "scorer", "screening", "signal", "portfolio", "inference", "outputs"}
    _require(set(data) == expected_tables, "top-level v3 config tables differ from the frozen schema")

    run = _table(
        data,
        "run",
        {"id", "evaluation_start", "sample_status", "specified_after_level_reversal_results", "confirmation_required"},
    )
    data_table = _table(
        data,
        "data",
        {
            "collection_root",
            "corpus_manifest",
            "score_path",
            "score_manifest",
            "processing_buffer_minutes",
            "calendar",
            "timezone",
            "allow_legacy_revision_enforcement_metadata",
        },
    )
    prices = _table(data, "prices", {"panel_path", "manifest_path", "execution_field", "return_convention"})
    scorer = _table(data, "scorer", {"id", "model_id", "revision", "score_definition"})
    screening = _table(
        data,
        "screening",
        {"require_explicit_target", "require_single_target", "exclude_market_price_technical"},
    )
    signal = _table(
        data,
        "signal",
        {
            "construction",
            "formation_lag_sessions",
            "lookback_sessions",
            "tail_fraction",
            "minimum_ranked_firms",
            "minimum_names_per_side",
            "direction",
        },
    )
    portfolio = _table(
        data,
        "portfolio",
        {"gross_exposure", "initial_nav_usd", "cost_bps_per_side", "cost_grid_bps", "force_final_liquidation"},
    )
    inference = _table(data, "inference", {"block_length_sessions", "replications", "seed"})
    outputs = _table(data, "outputs", {"derived_root", "results_root"})

    _require(run["sample_status"] == "previously_explored", "v3 may not label this cohort as untouched")
    _require(run["specified_after_level_reversal_results"] is True, "v3 must disclose post-result specification")
    _require(run["confirmation_required"] is True, "v3 must reserve confirmation for an independent cohort")
    _require(data_table["processing_buffer_minutes"] == 15, "v3 processing buffer is frozen at 15 minutes")
    _require(data_table["calendar"] == "XNYS", "v3 calendar is frozen at XNYS")
    _require(data_table["timezone"] == "America/New_York", "v3 timezone is frozen")
    _require(prices["execution_field"] == "adjusted_open", "v3 requires adjusted-open execution")
    _require(prices["return_convention"] == "open_to_open", "v3 requires open-to-open returns")
    _require(scorer["id"] == "finbert", "v3 has exactly one FinBERT scorer")
    _require(scorer["model_id"] == "ProsusAI/finbert", "v3 model ID is frozen")
    _require(scorer["score_definition"] == "p_positive_minus_p_negative", "v3 score definition is frozen")
    revision = str(scorer["revision"])
    _require(len(revision) == 40 and all(char in "0123456789abcdef" for char in revision), "invalid FinBERT revision")
    _require(all(screening.values()), "all v3 screening safeguards must remain enabled")
    _require(signal["construction"] == "lagged_cross_sectional_rank_reversal", "v3 construction is frozen")
    _require(signal["formation_lag_sessions"] == 1, "v3 formation lag is frozen at one session")
    _require(signal["lookback_sessions"] == 5, "v3 lookback is frozen at five sessions")
    _require(float(signal["tail_fraction"]) == 0.20, "v3 tail fraction is frozen at 20 percent")
    _require(signal["minimum_ranked_firms"] == 10, "v3 requires at least ten ranked firms")
    _require(signal["minimum_names_per_side"] == 2, "v3 requires at least two names per side")
    _require(signal["direction"] == "contrarian", "v3 direction is frozen as contrarian")
    _require(float(portfolio["gross_exposure"]) == 1.0, "v3 gross exposure is frozen at 1.0")
    _require(float(portfolio["initial_nav_usd"]) == 1_000_000.0, "v3 initial NAV is frozen")
    _require(float(portfolio["cost_bps_per_side"]) == 10.0, "v3 headline cost is frozen at 10 bps")
    cost_grid = tuple(float(value) for value in portfolio["cost_grid_bps"])
    _require(cost_grid == (5.0, 10.0, 20.0), "v3 cost grid is frozen at 5/10/20 bps")
    _require(portfolio["force_final_liquidation"] is True, "v3 requires final liquidation")
    _require(inference["block_length_sessions"] == 5, "v3 bootstrap block is frozen at five sessions")
    _require(inference["replications"] == 2000, "v3 bootstrap replications are frozen at 2000")
    _require(inference["seed"] == 20260722, "v3 bootstrap seed is frozen")

    config = FinbertV3Config(
        path=source,
        config_sha256=sha256_text(raw.decode("utf-8")),
        run=RunConfig(
            id=str(run["id"]),
            evaluation_start=_date(run["evaluation_start"], "run.evaluation_start"),
            sample_status=str(run["sample_status"]),
            specified_after_level_reversal_results=True,
            confirmation_required=True,
        ),
        data=DataConfig(
            collection_root=Path(data_table["collection_root"]),
            corpus_manifest=Path(data_table["corpus_manifest"]),
            score_path=Path(data_table["score_path"]),
            score_manifest=Path(data_table["score_manifest"]),
            processing_buffer_minutes=15,
            calendar="XNYS",
            timezone="America/New_York",
            allow_legacy_revision_enforcement_metadata=bool(data_table["allow_legacy_revision_enforcement_metadata"]),
        ),
        prices=PriceConfig(
            panel_path=Path(prices["panel_path"]),
            manifest_path=Path(prices["manifest_path"]),
            execution_field="adjusted_open",
            return_convention="open_to_open",
        ),
        scorer=ScorerConfig(
            id="finbert",
            model_id="ProsusAI/finbert",
            revision=revision,
            score_definition="p_positive_minus_p_negative",
        ),
        screening=ScreeningConfig(**screening),
        signal=SignalConfig(
            construction="lagged_cross_sectional_rank_reversal",
            formation_lag_sessions=1,
            lookback_sessions=5,
            tail_fraction=0.20,
            minimum_ranked_firms=10,
            minimum_names_per_side=2,
            direction="contrarian",
        ),
        portfolio=PortfolioConfig(
            gross_exposure=1.0,
            initial_nav_usd=1_000_000.0,
            cost_bps_per_side=10.0,
            cost_grid_bps=cost_grid,
            force_final_liquidation=True,
        ),
        inference=InferenceConfig(block_length_sessions=5, replications=2000, seed=20260722),
        outputs=OutputConfig(derived_root=Path(outputs["derived_root"]), results_root=Path(outputs["results_root"])),
    )
    # Hashing the structured payload catches platform-specific path surprises in tests.
    canonical_json(config.to_payload())
    return config
