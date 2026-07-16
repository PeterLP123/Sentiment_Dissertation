from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..artifact_io import atomic_write_json, atomic_write_text, canonical_json, read_json, sha256_file, sha256_text
from ..runtime_metadata import collect_run_environment
from .config import PIPELINE_SCHEMA_VERSION, RunIdentity, StrategyResearchConfig, strategy_dependency_versions

STAGE_MANIFEST_SCHEMA_VERSION = 1
StageStatus = Literal["in_progress", "completed"]
StageAction = Literal["started", "resume", "reuse"]


class StrategyArtifactError(RuntimeError):
    """Raised when immutable strategy artifacts cannot be safely reused."""


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    size_bytes: int

    def to_payload(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_payload(cls, payload: Any, *, name: str) -> ArtifactRecord:
        if not isinstance(payload, dict):
            raise StrategyArtifactError(f"stage output {name!r} must be an object")
        path = str(payload.get("path") or "").strip()
        digest = str(payload.get("sha256") or "").strip()
        try:
            size = int(payload["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StrategyArtifactError(f"stage output {name!r} has an invalid size") from exc
        if not path or len(digest) != 64 or size < 0:
            raise StrategyArtifactError(f"stage output {name!r} is incomplete")
        return cls(path=path, sha256=digest, size_bytes=size)


@dataclass(frozen=True)
class StageManifest:
    run_id: str
    run_identity_sha256: str
    stage: str
    status: StageStatus
    created_at_utc: str
    completed_at_utc: str | None
    command: tuple[str, ...]
    config_sha256: str
    config: dict[str, Any]
    input_identities: dict[str, str]
    outputs: dict[str, ArtifactRecord]
    row_counts: dict[str, int]
    exclusions: dict[str, int]
    warnings: tuple[str, ...]
    deviations: tuple[str, ...]
    runtime: dict[str, Any]
    manifest_schema_version: int = STAGE_MANIFEST_SCHEMA_VERSION
    pipeline_schema_version: int = PIPELINE_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "pipeline_schema_version": self.pipeline_schema_version,
            "run_id": self.run_id,
            "run_identity_sha256": self.run_identity_sha256,
            "stage": self.stage,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "command": list(self.command),
            "config_sha256": self.config_sha256,
            "config": self.config,
            "input_identities": dict(sorted(self.input_identities.items())),
            "outputs": {name: record.to_payload() for name, record in sorted(self.outputs.items())},
            "row_counts": dict(sorted(self.row_counts.items())),
            "exclusions": dict(sorted(self.exclusions.items())),
            "warnings": list(self.warnings),
            "deviations": list(self.deviations),
            "runtime": self.runtime,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StageManifest:
        status = str(payload.get("status") or "")
        if status not in {"in_progress", "completed"}:
            raise StrategyArtifactError(f"unknown stage status: {status!r}")
        if int(payload.get("manifest_schema_version", -1)) != STAGE_MANIFEST_SCHEMA_VERSION:
            raise StrategyArtifactError("unsupported strategy stage-manifest schema version")
        if int(payload.get("pipeline_schema_version", -1)) != PIPELINE_SCHEMA_VERSION:
            raise StrategyArtifactError("stage manifest belongs to an incompatible pipeline schema")
        outputs_raw = payload.get("outputs") or {}
        if not isinstance(outputs_raw, dict):
            raise StrategyArtifactError("stage outputs must be an object")
        inputs = _string_mapping(payload.get("input_identities"), "input_identities")
        config = payload.get("config")
        runtime = payload.get("runtime")
        if not isinstance(config, dict) or not isinstance(runtime, dict):
            raise StrategyArtifactError("stage manifest config and runtime must be objects")
        config_sha256 = _required_string(payload, "config_sha256")
        if config_sha256 != sha256_text(canonical_json(config)):
            raise StrategyArtifactError("stage manifest config hash does not match its normalized config content")
        command = payload.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
            raise StrategyArtifactError("stage command must be a non-empty string list")
        completed_at = payload.get("completed_at_utc")
        if (status == "completed") != bool(completed_at):
            raise StrategyArtifactError("completed_at_utc must be set exactly when a stage is completed")
        created_timestamp = _required_timestamp(payload, "created_at_utc")
        completed_timestamp = _timestamp(completed_at, "completed_at_utc") if completed_at else None
        if completed_timestamp and _parse_timestamp(completed_timestamp) < _parse_timestamp(created_timestamp):
            raise StrategyArtifactError("completed_at_utc cannot precede created_at_utc")
        return cls(
            run_id=_required_string(payload, "run_id"),
            run_identity_sha256=_required_string(payload, "run_identity_sha256"),
            stage=_required_string(payload, "stage"),
            status=status,  # type: ignore[arg-type]
            created_at_utc=created_timestamp,
            completed_at_utc=completed_timestamp,
            command=tuple(command),
            config_sha256=config_sha256,
            config=config,
            input_identities=inputs,
            outputs={str(name): ArtifactRecord.from_payload(value, name=str(name)) for name, value in outputs_raw.items()},
            row_counts=_count_mapping(payload.get("row_counts"), "row_counts"),
            exclusions=_count_mapping(payload.get("exclusions"), "exclusions"),
            warnings=_string_tuple(payload.get("warnings"), "warnings"),
            deviations=_string_tuple(payload.get("deviations"), "deviations"),
            runtime=runtime,
        )


@dataclass(frozen=True)
class StageOpenResult:
    action: StageAction
    manifest: StageManifest


@dataclass(frozen=True)
class StageInspection:
    exists: bool
    manifest: StageManifest | None
    reusable: bool
    reason: str


class StageManifestStore:
    """Own one resumable stage manifest and enforce completed-stage immutability."""

    def __init__(
        self,
        path: str | Path,
        *,
        identity: RunIdentity,
        stage: str,
        repo_root: str | Path = ".",
    ) -> None:
        self.path = Path(path)
        self.identity = identity
        self.stage = stage.strip()
        self.repo_root = Path(repo_root).resolve()
        if not self.stage:
            raise StrategyArtifactError("stage cannot be blank")

    def inspect(
        self,
        *,
        config: StrategyResearchConfig | None = None,
        input_identities: Mapping[str, str] | None = None,
        verify_outputs: bool = False,
    ) -> StageInspection:
        """Inspect stage state without creating directories or writing files."""

        if not self.path.exists():
            return StageInspection(False, None, False, "manifest does not exist")
        manifest = self.load()
        mismatch = self._mismatch_reason(manifest, config=config, input_identities=input_identities)
        if mismatch:
            return StageInspection(True, manifest, False, mismatch)
        if manifest.status != "completed":
            return StageInspection(True, manifest, False, "stage is in progress and may be resumed")
        if verify_outputs:
            try:
                self.verify_outputs(manifest)
            except StrategyArtifactError as exc:
                return StageInspection(True, manifest, False, str(exc))
        return StageInspection(True, manifest, True, "completed stage matches")

    def load(self) -> StageManifest:
        try:
            manifest = StageManifest.from_payload(read_json(self.path))
        except (OSError, ValueError, StrategyArtifactError) as exc:
            raise StrategyArtifactError(f"cannot load strategy stage manifest {self.path}: {exc}") from exc
        self._validate_owner(manifest)
        return manifest

    def begin(
        self,
        *,
        config: StrategyResearchConfig,
        input_identities: Mapping[str, str],
        command: Iterable[str],
        runtime: dict[str, Any] | None = None,
        created_at_utc: str | None = None,
    ) -> StageOpenResult:
        normalized_inputs = _clean_identities(input_identities)
        normalized_command = tuple(str(item) for item in command if str(item))
        if not normalized_command:
            raise StrategyArtifactError("stage command cannot be empty")
        with _exclusive_file_lock(self.path):
            if self.path.exists():
                existing = self.load()
                mismatch = self._mismatch_reason(existing, config=config, input_identities=normalized_inputs)
                if mismatch:
                    raise StrategyArtifactError(f"refusing incompatible stage reuse: {mismatch}")
                if existing.status == "completed":
                    self.verify_outputs(existing)
                    return StageOpenResult("reuse", existing)
                return StageOpenResult("resume", existing)
            manifest = StageManifest(
                run_id=self.identity.resolved_run_id,
                run_identity_sha256=self.identity.identity_sha256,
                stage=self.stage,
                status="in_progress",
                created_at_utc=_timestamp(created_at_utc or _utc_now(), "created_at_utc"),
                completed_at_utc=None,
                command=normalized_command,
                config_sha256=config.config_sha256,
                config=config.to_payload(),
                input_identities=normalized_inputs,
                outputs={},
                row_counts={},
                exclusions={},
                warnings=(),
                deviations=(),
                runtime=runtime if runtime is not None else _strategy_runtime(self.repo_root),
            )
            atomic_write_json(self.path, manifest.to_payload())
            return StageOpenResult("started", manifest)

    def complete(
        self,
        *,
        outputs: Mapping[str, str | Path],
        row_counts: Mapping[str, int] | None = None,
        exclusions: Mapping[str, int] | None = None,
        warnings: Iterable[str] = (),
        deviations: Iterable[str] = (),
        completed_at_utc: str | None = None,
    ) -> StageManifest:
        with _exclusive_file_lock(self.path):
            if not self.path.exists():
                raise StrategyArtifactError("cannot complete a stage that has not begun")
            existing = self.load()
            records = {name: self._record(path) for name, path in sorted(outputs.items())}
            if not records:
                raise StrategyArtifactError("a completed stage must declare at least one output")
            normalized_rows = _clean_counts(row_counts or {}, "row_counts")
            normalized_exclusions = _clean_counts(exclusions or {}, "exclusions")
            normalized_warnings = _clean_strings(warnings)
            normalized_deviations = _clean_strings(deviations)
            if existing.status == "completed":
                self.verify_outputs(existing)
                if (
                    existing.outputs != records
                    or existing.row_counts != normalized_rows
                    or existing.exclusions != normalized_exclusions
                    or existing.warnings != normalized_warnings
                    or existing.deviations != normalized_deviations
                ):
                    raise StrategyArtifactError("refusing to change metadata or outputs for a completed stage")
                return existing
            normalized_completed_at = _timestamp(completed_at_utc or _utc_now(), "completed_at_utc")
            if _parse_timestamp(normalized_completed_at) < _parse_timestamp(existing.created_at_utc):
                raise StrategyArtifactError("completed_at_utc cannot precede the stage creation time")
            completed = replace(
                existing,
                status="completed",
                completed_at_utc=normalized_completed_at,
                outputs=records,
                row_counts=normalized_rows,
                exclusions=normalized_exclusions,
                warnings=normalized_warnings,
                deviations=normalized_deviations,
            )
            atomic_write_json(self.path, completed.to_payload())
            return completed

    def verify_outputs(self, manifest: StageManifest | None = None) -> None:
        current = manifest or self.load()
        if current.status != "completed":
            raise StrategyArtifactError("cannot verify outputs for an incomplete stage")
        for name, record in current.outputs.items():
            path = Path(record.path)
            resolved = path if path.is_absolute() else self.repo_root / path
            if not resolved.is_file():
                raise StrategyArtifactError(f"completed stage output is missing: {name} ({record.path})")
            if resolved.stat().st_size != record.size_bytes or sha256_file(resolved) != record.sha256:
                raise StrategyArtifactError(f"completed stage output hash mismatch: {name} ({record.path})")

    def _validate_owner(self, manifest: StageManifest) -> None:
        if manifest.run_id != self.identity.resolved_run_id:
            raise StrategyArtifactError("stage manifest belongs to a different resolved run ID")
        if manifest.run_identity_sha256 != self.identity.identity_sha256:
            raise StrategyArtifactError("stage manifest run identity hash does not match")
        if manifest.stage != self.stage:
            raise StrategyArtifactError("stage manifest belongs to a different stage")

    def _mismatch_reason(
        self,
        manifest: StageManifest,
        *,
        config: StrategyResearchConfig | None,
        input_identities: Mapping[str, str] | None,
    ) -> str | None:
        if config is not None:
            if manifest.config_sha256 != config.config_sha256 or manifest.config != config.to_payload():
                return "configuration content or hash differs"
        if input_identities is not None and manifest.input_identities != _clean_identities(input_identities):
            return "input identities differ"
        return None

    def _record(self, value: str | Path) -> ArtifactRecord:
        path = Path(value)
        resolved = path.resolve()
        if not resolved.is_file():
            raise StrategyArtifactError(f"stage output does not exist or is not a file: {path}")
        try:
            display = resolved.relative_to(self.repo_root).as_posix()
        except ValueError:
            display = resolved.as_posix()
        return ArtifactRecord(path=display, sha256=sha256_file(resolved), size_bytes=resolved.stat().st_size)


def write_immutable_text(path: str | Path, value: str) -> tuple[Path, bool]:
    """Write once, or verify byte-identical content. Return ``(path, reused)``."""

    target = Path(path)
    with _exclusive_file_lock(target):
        if target.exists():
            if not target.is_file() or target.read_bytes() != value.encode("utf-8"):
                raise StrategyArtifactError(f"refusing to overwrite a different immutable artifact: {target}")
            return target, True
        atomic_write_text(target, value)
        return target, False


def write_immutable_json(path: str | Path, value: Any) -> tuple[Path, bool]:
    rendered = _pretty_json(value)
    return write_immutable_text(path, rendered)


def write_immutable_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> tuple[Path, bool]:
    rendered = "".join(canonical_json(row) + "\n" for row in rows)
    return write_immutable_text(path, rendered)


def _pretty_json(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _strategy_runtime(repo_root: Path) -> dict[str, Any]:
    runtime = collect_run_environment(repo_root)
    runtime["strategy_dependencies"] = strategy_dependency_versions()
    return runtime


def _timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyArtifactError(f"{field_name} must be a timezone-aware ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise StrategyArtifactError(f"{field_name} is not a valid ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StrategyArtifactError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC).isoformat()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _required_timestamp(payload: Mapping[str, Any], key: str) -> str:
    return _timestamp(payload.get(key), key)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StrategyArtifactError(f"stage manifest {key} must be a non-empty string")
    return value.strip()


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise StrategyArtifactError(f"{field_name} must be an object")
    return _clean_identities({str(key): str(item) for key, item in value.items()})


def _clean_identities(value: Mapping[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for name, identity in sorted(value.items()):
        clean_name = str(name).strip()
        clean_identity = str(identity).strip()
        if not clean_name or not clean_identity:
            raise StrategyArtifactError("input identities require non-empty names and values")
        cleaned[clean_name] = clean_identity
    return cleaned


def _count_mapping(value: Any, field_name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise StrategyArtifactError(f"{field_name} must be an object")
    return _clean_counts(value, field_name)


def _clean_counts(value: Mapping[str, int], field_name: str) -> dict[str, int]:
    cleaned: dict[str, int] = {}
    for name, count in sorted(value.items()):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StrategyArtifactError(f"{field_name}.{name} must be a non-negative integer")
        cleaned[str(name)] = count
    return cleaned


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise StrategyArtifactError(f"{field_name} must be a string list")
    return _clean_strings(value)


def _clean_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


@contextmanager
def _exclusive_file_lock(target: Path) -> Iterator[None]:
    """Serialize a write transaction with a stable sibling lock file."""

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    try:
        with lock_path.open("a+b") as handle:
            _lock_handle(handle)
            try:
                yield
            finally:
                _unlock_handle(handle)
    except StrategyArtifactError:
        raise
    except OSError as exc:
        raise StrategyArtifactError(f"cannot lock immutable artifact transaction for {target}: {exc}") from exc


def _lock_handle(handle: Any) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI/users
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        return
    raise StrategyArtifactError(f"immutable writes are unsupported without file locking on platform {os.name!r}")


def _unlock_handle(handle: Any) -> None:
    if os.name == "posix":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI/users
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
