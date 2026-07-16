from __future__ import annotations

import re
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from ..artifact_io import atomic_write_json, atomic_write_text, canonical_json, sha256_file, sha256_text
from ..lseg_corpus import load_verified_lseg_corpus
from ..lseg_source import LsegNewsError
from .schemas import EventBuildResult, EventScreeningRecord, StrategyEvent, parse_aware_utc, write_strategy_events

ATTRITION_REASONS = (
    "invalid_timestamp",
    "missing_symbol",
    "duplicate_revision",
    "relevance_exclusion",
    "missing_usable_text",
    "missing_score",
    "missing_price_history",
    "no_eligible_execution_session",
    "other",
)


@dataclass(frozen=True)
class LsegEventSettings:
    processing_buffer_minutes: int = 15
    calendar_name: str = "XNYS"
    exchange_timezone: str = "America/New_York"
    lead_window_chars: int = 2000
    min_body_mentions: int = 2
    max_text_chars: int = 12000
    overrides_path: Path | None = None

    def __post_init__(self) -> None:
        if self.processing_buffer_minutes < 0:
            raise ValueError("processing_buffer_minutes cannot be negative")
        if min(self.lead_window_chars, self.min_body_mentions, self.max_text_chars) < 1:
            raise ValueError("lead, mention, and text limits must be positive")
        ZoneInfo(self.exchange_timezone)

    def to_identity_payload(self) -> dict[str, Any]:
        overrides_hash = None
        if self.overrides_path is not None:
            if not self.overrides_path.is_file():
                raise LsegNewsError(f"strategy event overrides file does not exist: {self.overrides_path}")
            overrides_hash = sha256_file(self.overrides_path)
        return {
            "event_rule_version": 1,
            "event_identity": "earliest_story_family_symbol",
            "text_unit": "headline_plus_lead",
            "processing_buffer_minutes": self.processing_buffer_minutes,
            "calendar_name": self.calendar_name,
            "exchange_timezone": self.exchange_timezone,
            "lead_window_chars": self.lead_window_chars,
            "min_body_mentions": self.min_body_mentions,
            "max_text_chars": self.max_text_chars,
            "overrides_path": str(self.overrides_path) if self.overrides_path else None,
            "overrides_sha256": overrides_hash,
        }

    @property
    def identity_hash(self) -> str:
        return sha256_text(canonical_json(self.to_identity_payload()))


class HistoricalNewsSource(Protocol):
    """Historical, point-in-time source contract.

    Future prospective sources may implement this protocol, but the LSEG source
    is the only historical implementation in v1.
    """

    def build_events(self) -> EventBuildResult: ...


@dataclass(frozen=True)
class _Target:
    symbol: str
    name: str | None
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class _Candidate:
    record: dict[str, Any]
    target: _Target
    version_created: datetime
    relevance_reason: str


def _targets(manifest: dict[str, Any]) -> dict[str, _Target]:
    companies = ((manifest.get("config") or {}).get("companies") or [])
    targets: dict[str, _Target] = {}
    for company in companies:
        if not isinstance(company, dict):
            continue
        symbol = str(company.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        name = str(company.get("name") or "").strip() or None
        aliases = [str(value).strip() for value in company.get("aliases") or [] if str(value).strip()]
        if name:
            aliases.append(name)
        targets[symbol] = _Target(symbol, name, tuple(dict.fromkeys(aliases or [symbol])))
    return targets


def _alias_matches(text: str, aliases: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for alias in aliases:
        flags = 0 if alias.isupper() and len(alias) <= 5 else re.IGNORECASE
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, flags):
            matches.append(alias)
    return matches


def _relevance(record: Mapping[str, Any], aliases: tuple[str, ...], settings: LsegEventSettings) -> tuple[str, str]:
    headline = str(record.get("headline") or "")
    body = str(record.get("clean_text") or "")
    if _alias_matches(headline, aliases):
        return "include", "target_alias_in_headline"
    body_matches = _alias_matches(body, aliases)
    lead_matches = _alias_matches(body[: settings.lead_window_chars], aliases)
    mention_count = sum(
        len(
            re.findall(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                body,
                0 if alias.isupper() and len(alias) <= 5 else re.IGNORECASE,
            )
        )
        for alias in aliases
    )
    if lead_matches and mention_count >= settings.min_body_mentions:
        return "include", "target_alias_in_lead_and_body"
    if body_matches:
        return "review", "target_alias_only_in_body"
    return "exclude", "no_target_alias"


def _load_overrides(path: Path | None) -> dict[tuple[str, str], tuple[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise LsegNewsError(f"strategy event overrides file does not exist: {path}")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    overrides: dict[tuple[str, str], tuple[str, str]] = {}
    for index, item in enumerate(payload.get("overrides") or [], start=1):
        family = str(item.get("story_family_id") or "").strip()
        symbol = str(item.get("symbol") or "").strip().upper()
        decision = str(item.get("decision") or "").strip().lower()
        reason = str(item.get("reason") or "").strip()
        if not family or not symbol or decision not in {"include", "review", "exclude"} or not reason:
            raise LsegNewsError(f"invalid strategy event override {index}")
        overrides[(family, symbol)] = decision, reason
    return overrides


def _screen(
    record: Mapping[str, Any],
    symbol: str | None,
    status: str,
    reason: str,
) -> EventScreeningRecord:
    return EventScreeningRecord(
        story_family_id=str(record.get("story_family_id") or ""),
        revision_id=str(record.get("revision_id") or ""),
        story_id=str(record.get("story_id") or ""),
        symbol=symbol,
        version_created=str(record["version_created"]) if record.get("version_created") else None,
        status=status,
        reason=reason,
    )


def _calendar_sessions(
    version_created: datetime,
    available_at: datetime,
    *,
    calendar: Any,
    timezone: ZoneInfo,
) -> tuple[Any, Any]:
    local_date = version_created.astimezone(timezone).date()
    event_session = calendar.date_to_session(pd.Timestamp(local_date), direction="next")
    next_open = calendar.next_open(pd.Timestamp(available_at))
    if next_open.to_pydatetime().astimezone(UTC) <= available_at:
        raise ValueError("calendar returned a non-strict execution open")
    execution_session = calendar.minute_to_session(next_open, direction="none")
    return event_session.date(), execution_session.date()


def build_strategy_events(
    corpus_manifest: str | Path,
    *,
    settings: LsegEventSettings | None = None,
    calendar: Any | None = None,
) -> EventBuildResult:
    """Build the full, unsampled LSEG strategy event universe."""

    resolved_settings = settings or LsegEventSettings()
    manifest_path = Path(corpus_manifest)
    manifest, records = load_verified_lseg_corpus(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    targets = _targets(manifest)
    overrides = _load_overrides(resolved_settings.overrides_path)
    exchange_calendar = calendar or xcals.get_calendar(resolved_settings.calendar_name)
    timezone = ZoneInfo(resolved_settings.exchange_timezone)
    calendar_timezone = str(getattr(exchange_calendar, "tz", ""))
    if calendar_timezone and calendar_timezone != resolved_settings.exchange_timezone:
        raise ValueError(
            f"calendar {resolved_settings.calendar_name} uses {calendar_timezone}, not {resolved_settings.exchange_timezone}"
        )

    counts: Counter[str] = Counter({reason: 0 for reason in ATTRITION_REASONS})
    counts["input_records"] = len(records)
    screening: list[EventScreeningRecord] = []
    candidates: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
    ordered_records = sorted(
        records,
        key=lambda record: (
            str(record.get("version_created") or ""),
            str(record.get("story_id") or ""),
            str(record.get("revision_id") or ""),
        ),
    )
    for record in ordered_records:
        raw_symbols = record.get("matched_symbols")
        symbols = sorted({str(value).strip().upper() for value in raw_symbols or [] if str(value).strip()})
        if not symbols:
            counts["missing_symbol"] += 1
            screening.append(_screen(record, None, "excluded", "missing_symbol"))
            continue
        counts["expanded_candidates"] += len(symbols)
        for symbol in symbols:
            try:
                version_created = parse_aware_utc(record.get("version_created"), "version_created")
            except ValueError:
                counts["invalid_timestamp"] += 1
                screening.append(_screen(record, symbol, "excluded", "invalid_timestamp"))
                continue
            clean_text = str(record.get("clean_text") or "").strip()
            headline = str(record.get("headline") or "").strip()
            if not record.get("scoring_eligible") or not (clean_text or headline):
                counts["missing_usable_text"] += 1
                screening.append(_screen(record, symbol, "excluded", "missing_usable_text"))
                continue
            target = targets.get(symbol, _Target(symbol, None, (symbol,)))
            decision, relevance_reason = _relevance(record, target.aliases, resolved_settings)
            family = str(record.get("story_family_id") or "").strip()
            override = overrides.get((family, symbol))
            if override is not None:
                decision, override_reason = override
                relevance_reason = f"manual_override:{override_reason}"
            if decision != "include":
                counts["relevance_exclusion"] += 1
                screening.append(_screen(record, symbol, "excluded", f"relevance_{decision}:{relevance_reason}"))
                continue
            candidates[(family, symbol)].append(_Candidate(dict(record), target, version_created, relevance_reason))

    events: list[StrategyEvent] = []
    for key in sorted(candidates):
        values = sorted(
            candidates[key],
            key=lambda candidate: (
                candidate.version_created,
                str(candidate.record.get("story_id") or ""),
                str(candidate.record.get("revision_id") or ""),
            ),
        )
        chosen = values[0]
        for duplicate in values[1:]:
            counts["duplicate_revision"] += 1
            screening.append(_screen(duplicate.record, duplicate.target.symbol, "excluded", "duplicate_revision"))
        available_at = chosen.version_created + timedelta(minutes=resolved_settings.processing_buffer_minutes)
        try:
            event_session, execution_session = _calendar_sessions(
                chosen.version_created,
                available_at,
                calendar=exchange_calendar,
                timezone=timezone,
            )
        except (ValueError, TypeError, OverflowError, IndexError) as exc:
            counts["no_eligible_execution_session"] += 1
            screening.append(
                _screen(chosen.record, chosen.target.symbol, "excluded", f"no_eligible_execution_session:{type(exc).__name__}")
            )
            continue
        headline = str(chosen.record.get("headline") or "").strip()
        source_text_limit = int(chosen.record.get("max_scoring_chars") or resolved_settings.max_text_chars)
        text_limit = min(source_text_limit, resolved_settings.max_text_chars, resolved_settings.lead_window_chars)
        lead_or_body = str(chosen.record.get("clean_text") or "").strip()[:text_limit]
        scoring_text = "\n\n".join(value.strip() for value in (headline, lead_or_body) if value.strip())
        family, symbol = key
        revision_id = str(chosen.record.get("revision_id") or "").strip()
        event_id = sha256_text(f"strategy-event-v1|lseg|{family}|{revision_id}|{symbol}")
        events.append(
            StrategyEvent(
                event_id=event_id,
                story_family_id=family,
                revision_id=revision_id,
                symbol=symbol,
                target_company_name=chosen.target.name,
                version_created_utc=chosen.version_created,
                available_at_utc=available_at,
                source="lseg",
                headline=headline,
                lead_or_body=lead_or_body,
                text_sha256=sha256_text(scoring_text),
                target_relevance="include",
                event_session=event_session,
                eligible_execution_session=execution_session,
                exclusion_reason=None,
                input_manifest_hash=manifest_hash,
            )
        )
        screening.append(_screen(chosen.record, symbol, "included", chosen.relevance_reason))

    events.sort(key=lambda event: (event.available_at_utc, event.event_id))
    screening.sort(
        key=lambda row: (
            row.version_created or "",
            row.story_id,
            row.symbol or "",
            row.revision_id,
            row.status,
        )
    )
    counts["selected_events"] = len(events)
    counts["screening_rows"] = len(screening)
    event_rules = resolved_settings.to_identity_payload()
    return EventBuildResult(
        source_manifest_path=manifest_path,
        source_manifest_hash=manifest_hash,
        event_rules=event_rules,
        event_rules_hash=sha256_text(canonical_json(event_rules)),
        events=tuple(events),
        screening=tuple(screening),
        attrition=dict(sorted(counts.items())),
    )


@dataclass(frozen=True)
class LsegHistoricalNewsSource:
    corpus_manifest: Path
    settings: LsegEventSettings = LsegEventSettings()

    def build_events(self) -> EventBuildResult:
        return build_strategy_events(self.corpus_manifest, settings=self.settings)


def write_event_build_artifacts(output_dir: str | Path, result: EventBuildResult) -> tuple[Path, Path, Path]:
    """Write deterministic local event artifacts without overwriting differences."""

    root = Path(output_dir)
    events_path = write_strategy_events(root / "events.jsonl", result.events)
    screening_path = root / "event_screening.jsonl"
    screening_payload = [row.to_payload() for row in result.screening]
    screening_rendered = "".join(canonical_json(row) + "\n" for row in screening_payload)
    if screening_path.exists():
        if screening_path.read_text(encoding="utf-8") != screening_rendered:
            raise FileExistsError(f"refusing to overwrite different event screening: {screening_path}")
    else:
        atomic_write_text(screening_path, screening_rendered)
    attrition_path = root / "event_attrition.json"
    attrition_payload = {
        "source_manifest": str(result.source_manifest_path),
        "source_manifest_sha256": result.source_manifest_hash,
        "event_rules": result.event_rules,
        "event_rules_sha256": result.event_rules_hash,
        "counts": result.attrition,
    }
    if attrition_path.exists():
        import json

        if json.loads(attrition_path.read_text(encoding="utf-8")) != attrition_payload:
            raise FileExistsError(f"refusing to overwrite different event attrition: {attrition_path}")
    else:
        atomic_write_json(attrition_path, attrition_payload)
    return events_path, screening_path, attrition_path
