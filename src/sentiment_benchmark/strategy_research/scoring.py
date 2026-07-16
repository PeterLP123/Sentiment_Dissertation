from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..artifact_io import atomic_write_json, canonical_json, directory_digest, read_json, sha256_text
from ..constants import DEFAULT_BASE_URL, DEFAULT_CEREBRAS_BASE_URL, DEFAULT_OLLAMA_HOST
from ..models import BlindExample, LLMResponseRecord, PromptConfig
from .prompts import strategy_prompt
from .schemas import ScoreRecord, ScoreStatus, ScoringScheme, StrategyEvent, write_score_records
from .score_mapping import labels_for_scheme, map_score_label, score_mapping_hash


@dataclass(frozen=True)
class ScoringIdentity:
    scheme: ScoringScheme
    provider: str
    model_id: str
    model_digest: str | None = None
    endpoint: str | None = None
    prompt_config: PromptConfig | None = None
    scorer_family: str = "llm"
    temperature: float = 0.0
    sample_count: int = 1
    max_completion_tokens: int = 16
    ollama_think: bool | None = None
    retries: int = 3
    agreement_conditioning: bool = False

    def __post_init__(self) -> None:
        labels_for_scheme(self.scheme)
        if not self.provider.strip() or not self.model_id.strip():
            raise ValueError("provider and model_id are required")
        if self.sample_count != 1:
            raise ValueError("strategy scoring v1 supports exactly one deterministic sample")
        if self.temperature != 0.0:
            raise ValueError("strategy scoring v1 requires temperature=0")
        if self.max_completion_tokens < 1 or self.retries < 0:
            raise ValueError("completion tokens must be positive and retries cannot be negative")
        if self.agreement_conditioning:
            raise ValueError("agreement-conditioned impulses are not enabled in strategy v1")
        if self.provider == "ollama" and self.ollama_think is not False:
            raise ValueError("strategy Ollama scoring requires ollama_think=False")
        if self.provider != "ollama" and self.ollama_think is not None:
            raise ValueError("ollama_think is valid only for the Ollama provider")
        if self.prompt.output_mode != "label_only":
            raise ValueError("strategy scoring v1 requires a strict label_only prompt")
        _ = self.resolved_endpoint

    @property
    def prompt(self) -> PromptConfig:
        return self.prompt_config or strategy_prompt(self.scheme)

    @property
    def resolved_endpoint(self) -> str:
        if self.endpoint is not None:
            endpoint = self.endpoint.strip().rstrip("/")
            if not endpoint:
                raise ValueError("strategy scoring endpoint cannot be blank")
            return endpoint
        defaults = {
            "cerebras": DEFAULT_CEREBRAS_BASE_URL,
            "openrouter": DEFAULT_BASE_URL,
            "ollama": DEFAULT_OLLAMA_HOST,
            "fixture": "fixture://local",
        }
        try:
            return defaults[self.provider]
        except KeyError as exc:
            raise ValueError(f"no locked strategy endpoint for provider {self.provider!r}") from exc

    @property
    def config_hash(self) -> str:
        payload = {
            "scheme": self.scheme,
            "score_mapping_hash": score_mapping_hash(self.scheme),
            "provider": self.provider,
            "model_id": self.model_id,
            "model_digest": self.model_digest,
            "scorer_family": self.scorer_family,
            "prompt_id": self.prompt.prompt_id,
            "prompt_hash": self.prompt.prompt_hash,
            "temperature": self.temperature,
            "sample_count": self.sample_count,
            "max_completion_tokens": self.max_completion_tokens,
            "retries": self.retries,
            "agreement_conditioning": self.agreement_conditioning,
        }
        # Fixture score hashes predate endpoint routing and make no network
        # request. Hosted/local-model identities always bind the exact endpoint.
        if self.provider != "fixture":
            payload["endpoint"] = self.resolved_endpoint
        if self.provider == "ollama":
            payload["ollama_think"] = self.ollama_think
        return sha256_text(canonical_json(payload))


@dataclass(frozen=True)
class ScoreCacheInspection:
    event_count: int
    unique_target_count: int
    expected_score_calls: int
    cache_hits: int
    missing_event_ids: tuple[str, ...]
    provider: str
    model_id: str
    prompt_id: str
    prompt_hash: str


@dataclass(frozen=True)
class ScoreBatchResult:
    records: tuple[ScoreRecord, ...]
    missing_event_ids: tuple[str, ...]
    cache_hits: int
    calls_made: int


def import_completed_score_cache(
    events: tuple[StrategyEvent, ...] | list[StrategyEvent],
    identity: ScoringIdentity,
    source_cache_dir: str | Path,
    target_cache_dir: str | Path,
) -> dict[str, Any]:
    """Import only fully validated immutable successes into another run cache."""

    source = Path(source_cache_dir).resolve()
    target = Path(target_cache_dir).resolve()
    if source == target:
        raise ValueError("source and target score caches must differ")
    ordered_events = sorted(events, key=lambda value: (value.available_at_utc, value.event_id))
    records: list[ScoreRecord] = []
    source_paths: list[Path] = []
    for event in ordered_events:
        record = _load_completed(event, identity, source)
        if record is None:
            raise ValueError(f"source score cache is incomplete for event {event.event_id}")
        records.append(record)
        source_paths.append(_completed_path(source, record.cache_key))
    source_digest = directory_digest(source, source_paths)
    for record in records:
        _write_completed(target, record)
    inspection = inspect_score_cache(events, identity, target)
    if inspection.expected_score_calls:
        raise ValueError("target score cache remained incomplete after import")
    return {
        "schema_version": 1,
        "status": "completed",
        "source_cache": source.as_posix(),
        "source_completed_sha256": source_digest,
        "imported_records": len(records),
        "event_count": len(events),
        "scoring_config_hash": identity.config_hash,
        "model_id": identity.model_id,
        "model_digest": identity.model_digest,
        "prompt_hash": identity.prompt.prompt_hash,
    }


def score_cache_key(event: StrategyEvent, identity: ScoringIdentity) -> str:
    return sha256_text(
        canonical_json(
            {
                "event_id": event.event_id,
                "content_hash": event.text_sha256,
                "target_identity": event.target_identity,
                "provider": identity.provider,
                "model_id": identity.model_id,
                "model_digest": identity.model_digest,
                "prompt_hash": identity.prompt.prompt_hash,
                "scoring_scheme": identity.scheme,
                "temperature": identity.temperature,
                "sample_count": identity.sample_count,
                "scoring_config_hash": identity.config_hash,
            }
        )
    )


def score_input_text(event: StrategyEvent) -> str:
    company = event.target_company_name or event.symbol
    return f"Target company: {company}\nTarget symbol: {event.symbol}\n\nNews:\n{event.scoring_text}"


def _completed_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "completed" / key[:2] / f"{key}.json"


def _attempt_dir(cache_dir: Path, key: str) -> Path:
    return cache_dir / "attempts" / key[:2] / key


def _validate_record_identity(
    record: ScoreRecord,
    event: StrategyEvent,
    identity: ScoringIdentity,
    *,
    path: Path,
    require_success: bool,
) -> None:
    key = score_cache_key(event, identity)
    expected = {
        "cache_key": key,
        "event_id": event.event_id,
        "symbol": event.symbol,
        "content_hash": event.text_sha256,
        "target_identity": event.target_identity,
        "config_hash": identity.config_hash,
        "scoring_scheme": identity.scheme,
        "scorer_family": identity.scorer_family,
        "provider": identity.provider,
        "model_id": identity.model_id,
        "model_digest": identity.model_digest,
        "prompt_id": identity.prompt.prompt_id,
        "prompt_hash": identity.prompt.prompt_hash,
        "temperature": identity.temperature,
        "sample_count": identity.sample_count,
    }
    actual = {field: getattr(record, field) for field in expected}
    if actual != expected:
        raise ValueError(f"incompatible or corrupt score cache record: {path}")
    if require_success and record.status != "success":
        raise ValueError(f"completed score cache record is not successful: {path}")
    if record.status == "success":
        expected_score = map_score_label(record.raw_label or "", identity.scheme)
        if record.score != expected_score:
            raise ValueError(f"completed score cache record has an invalid numeric mapping: {path}")


def _load_completed(event: StrategyEvent, identity: ScoringIdentity, cache_dir: Path) -> ScoreRecord | None:
    key = score_cache_key(event, identity)
    path = _completed_path(cache_dir, key)
    if not path.exists():
        return None
    record = ScoreRecord.from_payload(read_json(path))
    _validate_record_identity(record, event, identity, path=path, require_success=True)
    return record


def _recoverable_attempt(event: StrategyEvent, identity: ScoringIdentity, cache_dir: Path) -> ScoreRecord | None:
    attempts_dir = _attempt_dir(cache_dir, score_cache_key(event, identity))
    recovered: ScoreRecord | None = None
    for path in sorted(attempts_dir.glob("*.json")) if attempts_dir.exists() else ():
        record = ScoreRecord.from_payload(read_json(path))
        _validate_record_identity(record, event, identity, path=path, require_success=False)
        if record.status == "success":
            recovered = record
    return recovered


def inspect_score_cache(
    events: tuple[StrategyEvent, ...] | list[StrategyEvent],
    identity: ScoringIdentity,
    cache_dir: str | Path,
) -> ScoreCacheInspection:
    root = Path(cache_dir)
    missing: list[str] = []
    hits = 0
    for event in sorted(events, key=lambda value: (value.available_at_utc, value.event_id)):
        if _load_completed(event, identity, root) is None and _recoverable_attempt(event, identity, root) is None:
            missing.append(event.event_id)
        else:
            hits += 1
    prompt = identity.prompt
    return ScoreCacheInspection(
        event_count=len(events),
        unique_target_count=len({event.target_identity for event in events}),
        expected_score_calls=len(missing),
        cache_hits=hits,
        missing_event_ids=tuple(missing),
        provider=identity.provider,
        model_id=identity.model_id,
        prompt_id=prompt.prompt_id,
        prompt_hash=prompt.prompt_hash,
    )


def _previous_attempt_count(cache_dir: Path, key: str) -> int:
    attempts_dir = _attempt_dir(cache_dir, key)
    if not attempts_dir.exists():
        return 0
    cumulative = 0
    for path in sorted(attempts_dir.glob("*.json")):
        record = ScoreRecord.from_payload(read_json(path))
        if record.cache_key != key:
            raise ValueError(f"incompatible score attempt journal entry: {path}")
        cumulative = max(cumulative, record.attempt_count)
    return cumulative


def _record_from_response(
    event: StrategyEvent,
    identity: ScoringIdentity,
    response: LLMResponseRecord,
    *,
    cumulative_attempts: int,
) -> ScoreRecord:
    label = response.normalized_label
    score: float | None = None
    status: ScoreStatus
    error = response.error
    if response.status == "success" and response.parse_status == "valid" and label is not None:
        try:
            score = map_score_label(label, identity.scheme)
            status = "success"
        except ValueError as exc:
            status = "invalid"
            error = str(exc)
    elif response.status == "success":
        status = "invalid"
        error = error or "model output did not match the strict strategy enum"
    else:
        status = response.status
    return ScoreRecord(
        event_id=event.event_id,
        symbol=event.symbol,
        raw_label=label,
        score=score,
        scoring_scheme=identity.scheme,
        scorer_family=identity.scorer_family,
        provider=identity.provider,
        model_id=identity.model_id,
        model_digest=identity.model_digest,
        prompt_id=identity.prompt.prompt_id,
        prompt_hash=identity.prompt.prompt_hash,
        content_hash=event.text_sha256,
        target_identity=event.target_identity,
        temperature=identity.temperature,
        sample_count=identity.sample_count,
        status=status,
        attempt_count=cumulative_attempts + response.attempt_count,
        latency_ms=response.latency_ms,
        created_at_utc=datetime.now(UTC),
        config_hash=identity.config_hash,
        cache_key=score_cache_key(event, identity),
        error=error,
    )


def _write_attempt(cache_dir: Path, record: ScoreRecord) -> Path:
    attempts_dir = _attempt_dir(cache_dir, record.cache_key)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    sequence = len(list(attempts_dir.glob("*.json"))) + 1
    path = attempts_dir / f"{sequence:06d}.json"
    if path.exists():
        raise FileExistsError(f"score attempt journal entry already exists: {path}")
    return atomic_write_json(path, record.to_payload())


def _write_completed(cache_dir: Path, record: ScoreRecord) -> Path:
    if record.status != "success":
        raise ValueError("only successful score records can enter the completed cache")
    path = _completed_path(cache_dir, record.cache_key)
    if path.exists():
        existing = ScoreRecord.from_payload(read_json(path))
        if existing != record:
            raise FileExistsError(f"refusing to overwrite different completed score cache record: {path}")
        return path
    return atomic_write_json(path, record.to_payload())


async def score_events(
    events: tuple[StrategyEvent, ...] | list[StrategyEvent],
    identity: ScoringIdentity,
    cache_dir: str | Path,
    client: Any | None = None,
    *,
    allow_calls: bool = False,
    max_new_scores: int | None = None,
) -> ScoreBatchResult:
    """Reuse completed scores and optionally make explicitly authorised calls.

    The default is cache-only. A caller must pass both ``allow_calls=True`` and
    an existing provider client before any model request can occur.
    """

    if max_new_scores is not None and max_new_scores < 1:
        raise ValueError("max_new_scores must be positive when supplied")
    root = Path(cache_dir)
    records: list[ScoreRecord] = []
    missing: list[str] = []
    cache_hits = 0
    calls_made = 0
    for row_number, event in enumerate(sorted(events, key=lambda value: (value.available_at_utc, value.event_id)), start=1):
        cached = _load_completed(event, identity, root)
        if cached is None:
            cached = _recoverable_attempt(event, identity, root)
            if cached is not None:
                _write_completed(root, cached)
        if cached is not None:
            records.append(cached)
            cache_hits += 1
            continue
        if not allow_calls:
            missing.append(event.event_id)
            continue
        if max_new_scores is not None and calls_made >= max_new_scores:
            missing.append(event.event_id)
            continue
        if client is None:
            raise ValueError("a provider client is required when allow_calls=True")
        key = score_cache_key(event, identity)
        previous_attempts = _previous_attempt_count(root, key)
        response = await client.classify(
            identity.model_id,
            identity.prompt,
            BlindExample(row_number=row_number, sentence=score_input_text(event)),
            temperature=identity.temperature,
            max_completion_tokens=identity.max_completion_tokens,
            retries=identity.retries,
            allowed_labels=labels_for_scheme(identity.scheme),
        )
        calls_made += 1
        record = _record_from_response(event, identity, response, cumulative_attempts=previous_attempts)
        _write_attempt(root, record)
        records.append(record)
        if record.status == "success":
            _write_completed(root, record)
        else:
            missing.append(event.event_id)
    records.sort(key=lambda record: (record.event_id, record.symbol, record.cache_key))
    return ScoreBatchResult(tuple(records), tuple(missing), cache_hits, calls_made)


def materialize_scores(path: str | Path, batch: ScoreBatchResult) -> Path:
    return write_score_records(path, list(batch.records))
