from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from ..artifact_io import atomic_write_text, canonical_json, read_jsonl, sha256_text

ScoringScheme = Literal["three_class", "five_level"]
ScoreStatus = Literal[
    "success",
    "invalid",
    "api_error",
    "transport_error",
    "malformed_response",
    "client_error",
    "skipped",
]


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def parse_aware_utc(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid ISO 8601 timestamp: {value!r}") from exc
    return _aware_utc(parsed, field_name)


@dataclass(frozen=True)
class StrategyEvent:
    event_id: str
    story_family_id: str
    revision_id: str
    symbol: str
    target_company_name: str | None
    version_created_utc: datetime
    available_at_utc: datetime
    source: str
    headline: str
    lead_or_body: str
    text_sha256: str
    target_relevance: str
    event_session: date
    eligible_execution_session: date
    exclusion_reason: str | None
    input_manifest_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        for field_name in ("event_id", "story_family_id", "revision_id", "symbol", "source", "text_sha256", "input_manifest_hash"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be blank")
        version_created = _aware_utc(self.version_created_utc, "version_created_utc")
        available_at = _aware_utc(self.available_at_utc, "available_at_utc")
        object.__setattr__(self, "version_created_utc", version_created)
        object.__setattr__(self, "available_at_utc", available_at)
        if available_at < version_created:
            raise ValueError("available_at_utc cannot precede version_created_utc")
        if not self.headline.strip() and not self.lead_or_body.strip():
            raise ValueError("an event must contain a headline or body")
        if self.text_sha256 != sha256_text(self.scoring_text):
            raise ValueError("text_sha256 does not match headline and lead_or_body")

    @property
    def target_identity(self) -> str:
        return f"{self.symbol}|{self.target_company_name or self.symbol}"

    @property
    def scoring_text(self) -> str:
        parts = [value.strip() for value in (self.headline, self.lead_or_body) if value.strip()]
        return "\n\n".join(parts)

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "story_family_id": self.story_family_id,
            "revision_id": self.revision_id,
            "symbol": self.symbol,
            "target_company_name": self.target_company_name,
            "version_created_utc": self.version_created_utc.isoformat(),
            "available_at_utc": self.available_at_utc.isoformat(),
            "source": self.source,
            "headline": self.headline,
            "lead_or_body": self.lead_or_body,
            "text_sha256": self.text_sha256,
            "target_relevance": self.target_relevance,
            "event_session": self.event_session.isoformat(),
            "eligible_execution_session": self.eligible_execution_session.isoformat(),
            "exclusion_reason": self.exclusion_reason,
            "input_manifest_hash": self.input_manifest_hash,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StrategyEvent:
        try:
            event_session = date.fromisoformat(str(payload["event_session"]))
            execution_session = date.fromisoformat(str(payload["eligible_execution_session"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("strategy event has invalid session dates") from exc
        return cls(
            event_id=str(payload.get("event_id") or ""),
            story_family_id=str(payload.get("story_family_id") or ""),
            revision_id=str(payload.get("revision_id") or ""),
            symbol=str(payload.get("symbol") or "").upper(),
            target_company_name=str(payload["target_company_name"]) if payload.get("target_company_name") else None,
            version_created_utc=parse_aware_utc(payload.get("version_created_utc"), "version_created_utc"),
            available_at_utc=parse_aware_utc(payload.get("available_at_utc"), "available_at_utc"),
            source=str(payload.get("source") or ""),
            headline=str(payload.get("headline") or ""),
            lead_or_body=str(payload.get("lead_or_body") or ""),
            text_sha256=str(payload.get("text_sha256") or ""),
            target_relevance=str(payload.get("target_relevance") or ""),
            event_session=event_session,
            eligible_execution_session=execution_session,
            exclusion_reason=str(payload["exclusion_reason"]) if payload.get("exclusion_reason") else None,
            input_manifest_hash=str(payload.get("input_manifest_hash") or ""),
        )


@dataclass(frozen=True)
class EventScreeningRecord:
    story_family_id: str
    revision_id: str
    story_id: str
    symbol: str | None
    version_created: str | None
    status: str
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "story_family_id": self.story_family_id,
            "revision_id": self.revision_id,
            "story_id": self.story_id,
            "symbol": self.symbol,
            "version_created": self.version_created,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EventBuildResult:
    source_manifest_path: Path
    source_manifest_hash: str
    event_rules: dict[str, Any]
    event_rules_hash: str
    events: tuple[StrategyEvent, ...]
    screening: tuple[EventScreeningRecord, ...]
    attrition: dict[str, int]


@dataclass(frozen=True)
class ScoreRecord:
    event_id: str
    symbol: str
    raw_label: str | None
    score: float | None
    scoring_scheme: ScoringScheme
    scorer_family: str
    provider: str
    model_id: str
    model_digest: str | None
    prompt_id: str
    prompt_hash: str
    content_hash: str
    target_identity: str
    temperature: float
    sample_count: int
    status: ScoreStatus
    attempt_count: int
    latency_ms: float | None
    created_at_utc: datetime
    config_hash: str
    cache_key: str
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "created_at_utc", _aware_utc(self.created_at_utc, "created_at_utc"))
        for field_name in (
            "event_id",
            "symbol",
            "scorer_family",
            "provider",
            "model_id",
            "prompt_id",
            "prompt_hash",
            "content_hash",
            "target_identity",
            "config_hash",
            "cache_key",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be blank")
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be at least one")
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least one")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.status == "success":
            if self.raw_label is None or self.score is None:
                raise ValueError("successful scores require a raw label and mapped score")
            if not -1.0 <= self.score <= 1.0:
                raise ValueError("score must be in [-1, 1]")
        elif self.score is not None:
            raise ValueError("non-successful scores cannot carry a mapped score")

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "symbol": self.symbol,
            "raw_label": self.raw_label,
            "score": self.score,
            "scoring_scheme": self.scoring_scheme,
            "scorer_family": self.scorer_family,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_digest": self.model_digest,
            "prompt_id": self.prompt_id,
            "prompt_hash": self.prompt_hash,
            "content_hash": self.content_hash,
            "target_identity": self.target_identity,
            "temperature": self.temperature,
            "sample_count": self.sample_count,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "latency_ms": self.latency_ms,
            "created_at_utc": self.created_at_utc.isoformat(),
            "config_hash": self.config_hash,
            "cache_key": self.cache_key,
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ScoreRecord:
        scheme = str(payload.get("scoring_scheme") or "")
        if scheme not in {"three_class", "five_level"}:
            raise ValueError(f"unknown scoring scheme: {scheme!r}")
        status = str(payload.get("status") or "")
        valid_statuses = {"success", "invalid", "api_error", "transport_error", "malformed_response", "client_error", "skipped"}
        if status not in valid_statuses:
            raise ValueError(f"unknown score status: {status!r}")
        return cls(
            event_id=str(payload.get("event_id") or ""),
            symbol=str(payload.get("symbol") or "").upper(),
            raw_label=str(payload["raw_label"]) if payload.get("raw_label") is not None else None,
            score=float(payload["score"]) if payload.get("score") is not None else None,
            scoring_scheme=scheme,  # type: ignore[arg-type]
            scorer_family=str(payload.get("scorer_family") or ""),
            provider=str(payload.get("provider") or ""),
            model_id=str(payload.get("model_id") or ""),
            model_digest=str(payload["model_digest"]) if payload.get("model_digest") else None,
            prompt_id=str(payload.get("prompt_id") or ""),
            prompt_hash=str(payload.get("prompt_hash") or ""),
            content_hash=str(payload.get("content_hash") or ""),
            target_identity=str(payload.get("target_identity") or ""),
            temperature=float(payload.get("temperature", 0.0)),
            sample_count=int(payload.get("sample_count", 1)),
            status=status,  # type: ignore[arg-type]
            attempt_count=int(payload.get("attempt_count", 1)),
            latency_ms=float(payload["latency_ms"]) if payload.get("latency_ms") is not None else None,
            created_at_utc=parse_aware_utc(payload.get("created_at_utc"), "created_at_utc"),
            config_hash=str(payload.get("config_hash") or ""),
            cache_key=str(payload.get("cache_key") or ""),
            error=str(payload["error"]) if payload.get("error") else None,
        )


def write_strategy_events(path: str | Path, events: tuple[StrategyEvent, ...] | list[StrategyEvent]) -> Path:
    target = Path(path)
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("strategy events contain duplicate event IDs")
    ordered = sorted(events, key=lambda event: (event.available_at_utc, event.event_id))
    rendered = "".join(canonical_json(event.to_payload()) + "\n" for event in ordered)
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite different completed strategy events: {target}")
        return target
    return atomic_write_text(target, rendered)


def load_strategy_events(path: str | Path) -> tuple[StrategyEvent, ...]:
    events = tuple(StrategyEvent.from_payload(payload) for payload in read_jsonl(path))
    ordered = tuple(sorted(events, key=lambda event: (event.available_at_utc, event.event_id)))
    if events != ordered:
        raise ValueError(f"strategy events are not in deterministic availability order: {path}")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError(f"strategy events contain duplicate event IDs: {path}")
    return events


def write_score_records(path: str | Path, records: tuple[ScoreRecord, ...] | list[ScoreRecord]) -> Path:
    target = Path(path)
    if len({record.cache_key for record in records}) != len(records):
        raise ValueError("strategy scores contain duplicate cache keys")
    ordered = sorted(records, key=lambda record: (record.event_id, record.symbol, record.cache_key))
    rendered = "".join(canonical_json(record.to_payload()) + "\n" for record in ordered)
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite different completed strategy scores: {target}")
        return target
    return atomic_write_text(target, rendered)


def load_score_records(path: str | Path) -> tuple[ScoreRecord, ...]:
    records = tuple(ScoreRecord.from_payload(payload) for payload in read_jsonl(path))
    ordered = tuple(sorted(records, key=lambda record: (record.event_id, record.symbol, record.cache_key)))
    if records != ordered:
        raise ValueError(f"strategy scores are not in deterministic event order: {path}")
    if len({record.cache_key for record in records}) != len(records):
        raise ValueError(f"strategy scores contain duplicate cache keys: {path}")
    return records
