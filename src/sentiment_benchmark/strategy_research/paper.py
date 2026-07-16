from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..artifact_io import canonical_json, sha256_text

PAPER_JOURNAL_SCHEMA_VERSION = 1
PaperRecordType = Literal["event", "proposed_order", "outcome"]


class PaperJournalError(RuntimeError):
    """Raised when the append-only prospective journal is invalid or unsafe."""


@dataclass(frozen=True)
class PaperJournalRecord:
    sequence: int
    record_type: PaperRecordType
    record_id: str
    event_id: str
    recorded_at_utc: str
    payload: dict[str, Any]
    previous_record_sha256: str | None
    record_sha256: str
    schema_version: int = PAPER_JOURNAL_SCHEMA_VERSION

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "event_id": self.event_id,
            "recorded_at_utc": self.recorded_at_utc,
            "payload": self.payload,
            "previous_record_sha256": self.previous_record_sha256,
        }

    def to_payload(self) -> dict[str, Any]:
        return self.unsigned_payload() | {"record_sha256": self.record_sha256}

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        record_type: PaperRecordType,
        record_id: str,
        event_id: str,
        recorded_at_utc: str,
        payload: dict[str, Any],
        previous_record_sha256: str | None,
    ) -> PaperJournalRecord:
        unsigned = {
            "schema_version": PAPER_JOURNAL_SCHEMA_VERSION,
            "sequence": sequence,
            "record_type": record_type,
            "record_id": record_id,
            "event_id": event_id,
            "recorded_at_utc": recorded_at_utc,
            "payload": payload,
            "previous_record_sha256": previous_record_sha256,
        }
        return cls(
            sequence=sequence,
            record_type=record_type,
            record_id=record_id,
            event_id=event_id,
            recorded_at_utc=recorded_at_utc,
            payload=payload,
            previous_record_sha256=previous_record_sha256,
            record_sha256=sha256_text(canonical_json(unsigned)),
        )

    @classmethod
    def from_payload(cls, payload: Any) -> PaperJournalRecord:
        if not isinstance(payload, dict):
            raise PaperJournalError("paper journal rows must be JSON objects")
        if payload.get("schema_version") != PAPER_JOURNAL_SCHEMA_VERSION:
            raise PaperJournalError("unsupported paper-journal schema version")
        record_type = str(payload.get("record_type") or "")
        if record_type not in {"event", "proposed_order", "outcome"}:
            raise PaperJournalError(f"unknown paper record type: {record_type!r}")
        try:
            sequence = int(payload["sequence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperJournalError("paper journal sequence must be an integer") from exc
        record_id = _required(payload, "record_id")
        event_id = _required(payload, "event_id")
        recorded_at = _utc_timestamp(_required(payload, "recorded_at_utc"), "recorded_at_utc")
        body = payload.get("payload")
        if not isinstance(body, dict):
            raise PaperJournalError("paper journal payload must be an object")
        previous = payload.get("previous_record_sha256")
        if previous is not None and (not isinstance(previous, str) or len(previous) != 64):
            raise PaperJournalError("previous_record_sha256 must be null or a SHA-256 digest")
        digest = _required(payload, "record_sha256")
        if len(digest) != 64:
            raise PaperJournalError("record_sha256 must be a SHA-256 digest")
        return cls(
            sequence=sequence,
            record_type=record_type,  # type: ignore[arg-type]
            record_id=record_id,
            event_id=event_id,
            recorded_at_utc=recorded_at,
            payload=body,
            previous_record_sha256=previous,
            record_sha256=digest,
        )


def prospective_availability(first_seen_at_utc: str | datetime, scoring_completed_at_utc: str | datetime) -> datetime:
    first_seen = _parse_datetime(first_seen_at_utc, "first_seen_at_utc")
    scoring_completed = _parse_datetime(scoring_completed_at_utc, "scoring_completed_at_utc")
    return max(first_seen, scoring_completed)


class PaperJournal:
    """Hash-chained append-only journal; it has no broker or order-routing API."""

    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self.path = Path(path)
        self._clock = clock or (lambda: datetime.now(UTC))

    def verify(self) -> tuple[PaperJournalRecord, ...]:
        """Verify the entire chain without creating the journal when it is absent."""

        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise PaperJournalError(f"paper journal is not a file: {self.path}")
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise PaperJournalError(f"cannot read paper journal {self.path}: {exc}") from exc
        return self._decode_and_verify(raw)

    def append_event(
        self,
        *,
        event_id: str,
        source: str,
        source_url: str,
        target_symbol: str,
        published_at_utc: str | datetime,
        retrieved_at_utc: str | datetime,
        first_seen_at_utc: str | datetime,
        content_sha256: str,
    ) -> PaperJournalRecord:
        published = _utc_timestamp(published_at_utc, "published_at_utc")
        retrieved = _utc_timestamp(retrieved_at_utc, "retrieved_at_utc")
        first_seen = _utc_timestamp(first_seen_at_utc, "first_seen_at_utc")
        if _parse_datetime(retrieved, "retrieved_at_utc") < _parse_datetime(first_seen, "first_seen_at_utc"):
            raise PaperJournalError("retrieved_at_utc cannot precede first_seen_at_utc")
        digest = _sha256(content_sha256, "content_sha256")
        payload = {
            "source": _nonblank(source, "source"),
            "source_url": _nonblank(source_url, "source_url"),
            "target_symbol": _nonblank(target_symbol, "target_symbol").upper(),
            "published_at_utc": published,
            "retrieved_at_utc": retrieved,
            "first_seen_at_utc": first_seen,
            "content_sha256": digest.lower(),
        }
        return self._append(
            record_type="event",
            record_id=f"event:{_nonblank(event_id, 'event_id')}",
            event_id=event_id,
            payload=payload,
        )

    def append_proposed_order(
        self,
        *,
        proposal_id: str,
        event_id: str,
        scoring_completed_at_utc: str | datetime,
        proposed_at_utc: str | datetime,
        target_weight: float,
        order_delta: float,
        scorer_identity: str,
        prompt_hash: str,
    ) -> PaperJournalRecord:
        records = self.verify()
        event = next((record for record in records if record.record_type == "event" and record.event_id == event_id), None)
        if event is None:
            raise PaperJournalError(f"cannot propose an order for unknown event: {event_id}")
        scoring_completed = _utc_timestamp(scoring_completed_at_utc, "scoring_completed_at_utc")
        proposed = _utc_timestamp(proposed_at_utc, "proposed_at_utc")
        available = prospective_availability(str(event.payload["first_seen_at_utc"]), scoring_completed)
        if _parse_datetime(proposed, "proposed_at_utc") < available:
            raise PaperJournalError("proposed_at_utc cannot precede prospective event availability")
        if not math.isfinite(target_weight) or not math.isfinite(order_delta):
            raise PaperJournalError("paper target weights and order deltas must be finite")
        if abs(target_weight) > 1 or abs(order_delta) > 2:
            raise PaperJournalError("paper target weight or order delta is outside a plausible normalized range")
        payload = {
            "scoring_completed_at_utc": scoring_completed,
            "available_at_utc": available.isoformat(),
            "proposed_at_utc": proposed,
            "target_weight": float(target_weight),
            "order_delta": float(order_delta),
            "scorer_identity": _nonblank(scorer_identity, "scorer_identity"),
            "prompt_hash": _sha256(prompt_hash, "prompt_hash"),
        }
        return self._append(
            record_type="proposed_order",
            record_id=f"proposal:{_nonblank(proposal_id, 'proposal_id')}",
            event_id=event_id,
            payload=payload,
        )

    def append_outcome(
        self,
        *,
        outcome_id: str,
        proposal_id: str,
        observed_at_utc: str | datetime,
        gross_return: float,
        net_return: float,
    ) -> PaperJournalRecord:
        records = self.verify()
        proposal_record_id = f"proposal:{proposal_id}"
        proposal = next(
            (record for record in records if record.record_type == "proposed_order" and record.record_id == proposal_record_id),
            None,
        )
        if proposal is None:
            raise PaperJournalError(f"cannot append an outcome for unknown proposal: {proposal_id}")
        observed = _utc_timestamp(observed_at_utc, "observed_at_utc")
        proposed = _parse_datetime(proposal.payload["proposed_at_utc"], "proposed_at_utc")
        if _parse_datetime(observed, "observed_at_utc") <= proposed:
            raise PaperJournalError("an outcome must be observed after its proposal")
        if not math.isfinite(gross_return) or not math.isfinite(net_return):
            raise PaperJournalError("paper outcome returns must be finite")
        payload = {
            "proposal_record_id": proposal_record_id,
            "observed_at_utc": observed,
            "gross_return": float(gross_return),
            "net_return": float(net_return),
        }
        return self._append(
            record_type="outcome",
            record_id=f"outcome:{_nonblank(outcome_id, 'outcome_id')}",
            event_id=proposal.event_id,
            payload=payload,
        )

    def _append(
        self,
        *,
        record_type: PaperRecordType,
        record_id: str,
        event_id: str,
        payload: dict[str, Any],
    ) -> PaperJournalRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a+b") as handle:
                _lock(handle)
                handle.seek(0)
                existing = self._decode_and_verify(handle.read())
                if any(record.record_id == record_id for record in existing):
                    raise PaperJournalError(f"paper journal already contains record_id {record_id!r}")
                normalized_recorded = _utc_timestamp(self._clock(), "recorded_at_utc")
                record = PaperJournalRecord.create(
                    sequence=len(existing) + 1,
                    record_type=record_type,
                    record_id=record_id,
                    event_id=_nonblank(event_id, "event_id"),
                    recorded_at_utc=normalized_recorded,
                    payload=payload,
                    previous_record_sha256=existing[-1].record_sha256 if existing else None,
                )
                _validate_record_semantics([*existing, record])
                handle.seek(0, os.SEEK_END)
                handle.write((canonical_json(record.to_payload()) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
                return record
        except PaperJournalError:
            raise
        except OSError as exc:
            raise PaperJournalError(f"cannot append paper journal {self.path}: {exc}") from exc

    @staticmethod
    def _decode_and_verify(raw: bytes) -> tuple[PaperJournalRecord, ...]:
        if not raw:
            return ()
        if not raw.endswith(b"\n"):
            raise PaperJournalError("paper journal ends with a partial record")
        records: list[PaperJournalRecord] = []
        previous: str | None = None
        seen_ids: set[str] = set()
        for line_number, raw_line in enumerate(raw.splitlines(), start=1):
            try:
                import json

                payload = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PaperJournalError(f"invalid paper journal JSON at line {line_number}") from exc
            record = PaperJournalRecord.from_payload(payload)
            if record.sequence != line_number:
                raise PaperJournalError(f"paper journal sequence mismatch at line {line_number}")
            if record.previous_record_sha256 != previous:
                raise PaperJournalError(f"paper journal hash chain mismatch at line {line_number}")
            if record.record_sha256 != sha256_text(canonical_json(record.unsigned_payload())):
                raise PaperJournalError(f"paper journal record hash mismatch at line {line_number}")
            if record.record_id in seen_ids:
                raise PaperJournalError(f"duplicate paper journal record_id at line {line_number}")
            seen_ids.add(record.record_id)
            records.append(record)
            previous = record.record_sha256
        _validate_record_semantics(records)
        return tuple(records)


def _lock(handle: Any) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI/users
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        return
    raise PaperJournalError(f"paper journal writes are unsupported without file locking on platform {os.name!r}")


def _parse_datetime(value: str | datetime, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise PaperJournalError(f"{field_name} is not a valid ISO 8601 timestamp") from exc
    else:
        raise PaperJournalError(f"{field_name} must be a timezone-aware ISO 8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PaperJournalError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _utc_timestamp(value: str | datetime, field_name: str) -> str:
    return _parse_datetime(value, field_name).isoformat()


def _required(payload: dict[str, Any], key: str) -> str:
    return _nonblank(payload.get(key), key)


def _nonblank(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperJournalError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, field_name: str) -> str:
    digest = _nonblank(value, field_name).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PaperJournalError(f"{field_name} must be a 64-character hexadecimal digest")
    return digest


def _validate_record_semantics(records: list[PaperJournalRecord]) -> None:
    events: dict[str, PaperJournalRecord] = {}
    proposals: dict[str, PaperJournalRecord] = {}
    previous_recorded: datetime | None = None
    for record in records:
        recorded = _parse_datetime(record.recorded_at_utc, "recorded_at_utc")
        if previous_recorded is not None and recorded < previous_recorded:
            raise PaperJournalError("paper journal recorded_at_utc must be monotonic")
        previous_recorded = recorded
        if record.record_type == "event":
            if not record.record_id.startswith("event:") or record.event_id in events:
                raise PaperJournalError("paper event IDs must be unique and use the event: record prefix")
            for key in ("source", "source_url", "target_symbol"):
                _nonblank(record.payload.get(key), key)
            _sha256(record.payload.get("content_sha256"), "content_sha256")
            _parse_datetime(_nonblank(record.payload.get("published_at_utc"), "published_at_utc"), "published_at_utc")
            first_seen = _parse_datetime(
                _nonblank(record.payload.get("first_seen_at_utc"), "first_seen_at_utc"), "first_seen_at_utc"
            )
            retrieved = _parse_datetime(
                _nonblank(record.payload.get("retrieved_at_utc"), "retrieved_at_utc"), "retrieved_at_utc"
            )
            if retrieved < first_seen or recorded < first_seen:
                raise PaperJournalError("paper event retrieval and recording cannot precede first-seen time")
            events[record.event_id] = record
            continue
        if record.record_type == "proposed_order":
            if not record.record_id.startswith("proposal:") or record.event_id not in events:
                raise PaperJournalError("paper proposals must reference an earlier event and use the proposal: prefix")
            if any(key in record.payload for key in ("gross_return", "net_return", "realized_return", "outcome")):
                raise PaperJournalError("paper proposals cannot contain subsequent return outcomes")
            scoring_completed = _parse_datetime(
                _nonblank(record.payload.get("scoring_completed_at_utc"), "scoring_completed_at_utc"),
                "scoring_completed_at_utc",
            )
            proposed = _parse_datetime(_nonblank(record.payload.get("proposed_at_utc"), "proposed_at_utc"), "proposed_at_utc")
            first_seen = _parse_datetime(
                _nonblank(events[record.event_id].payload.get("first_seen_at_utc"), "first_seen_at_utc"),
                "first_seen_at_utc",
            )
            available = max(first_seen, scoring_completed)
            stored_available = _parse_datetime(
                _nonblank(record.payload.get("available_at_utc"), "available_at_utc"), "available_at_utc"
            )
            if stored_available != available or proposed < available or recorded < proposed:
                raise PaperJournalError("paper proposal availability or recording chronology is invalid")
            target_weight = _finite_number(record.payload.get("target_weight"), "target_weight")
            order_delta = _finite_number(record.payload.get("order_delta"), "order_delta")
            if abs(target_weight) > 1 or abs(order_delta) > 2:
                raise PaperJournalError("paper proposal weights are outside normalized bounds")
            _nonblank(record.payload.get("scorer_identity"), "scorer_identity")
            _sha256(record.payload.get("prompt_hash"), "prompt_hash")
            proposals[record.record_id] = record
            continue
        proposal_id = _nonblank(record.payload.get("proposal_record_id"), "proposal_record_id")
        proposal = proposals.get(proposal_id)
        if not record.record_id.startswith("outcome:") or proposal is None or proposal.event_id != record.event_id:
            raise PaperJournalError("paper outcomes must reference an earlier proposal for the same event")
        observed = _parse_datetime(_nonblank(record.payload.get("observed_at_utc"), "observed_at_utc"), "observed_at_utc")
        proposed = _parse_datetime(
            _nonblank(proposal.payload.get("proposed_at_utc"), "proposed_at_utc"), "proposed_at_utc"
        )
        if observed <= proposed or recorded < observed:
            raise PaperJournalError("paper outcome chronology is invalid")
        _finite_number(record.payload.get("gross_return"), "gross_return")
        _finite_number(record.payload.get("net_return"), "net_return")


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PaperJournalError(f"{field_name} must be a finite number")
    return float(value)
