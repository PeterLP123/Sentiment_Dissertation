"""Validation helpers for immutable completed strategy-research runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifact_io import canonical_json, read_json, sha256_file, sha256_text


class CompletedRunValidationError(RuntimeError):
    """Raised when a completed run or one of its frozen inputs changed."""


@dataclass(frozen=True)
class CompletedRunSnapshot:
    root: Path
    report: Mapping[str, Any]
    config: Mapping[str, Any]
    root_manifest: Mapping[str, Any]
    root_manifest_sha256: str


def load_completed_run_snapshot(run_dir: str | Path) -> CompletedRunSnapshot:
    """Load a completed run and cross-check its report and root manifest."""

    root = Path(run_dir)
    report_path = root / "manifests" / "report.json"
    root_manifest_path = root / "manifest.json"
    if not report_path.is_file() or not root_manifest_path.is_file():
        raise CompletedRunValidationError("completed source-run report or root manifest is missing")
    report = read_json(report_path)
    root_manifest = read_json(root_manifest_path)
    config = report.get("config")
    root_digest = sha256_file(root_manifest_path)
    if (
        report.get("manifest_schema_version") != 1
        or report.get("pipeline_schema_version") != 1
        or report.get("status") != "completed"
        or report.get("run_id") != root.name
        or not isinstance(config, dict)
        or report.get("config_sha256") != sha256_text(canonical_json(config))
        or report.get("outputs", {}).get("manifest", {}).get("sha256") != root_digest
    ):
        raise CompletedRunValidationError("source-run report is incomplete, internally inconsistent, or changed")
    if (
        root_manifest.get("schema_version") != 1
        or root_manifest.get("status") != "completed"
        or root_manifest.get("run_id") != root.name
        or root_manifest.get("run_identity_sha256") != report.get("run_identity_sha256")
        or root_manifest.get("config_sha256") != report.get("config_sha256")
        or root_manifest.get("config") != config
    ):
        raise CompletedRunValidationError("source-run root manifest is incomplete or does not match its report")
    return CompletedRunSnapshot(root, report, config, root_manifest, root_digest)


def _root_output_hash(root_manifest: Mapping[str, Any], suffix: str) -> str:
    normalized = suffix.lstrip("/")
    matches = [
        str(payload.get("sha256") or "")
        for path, payload in root_manifest.get("outputs", {}).items()
        if isinstance(payload, dict) and (str(path) == normalized or str(path).endswith(f"/{normalized}"))
    ]
    if len(matches) != 1 or not matches[0]:
        raise CompletedRunValidationError(f"root manifest does not uniquely freeze output: {normalized}")
    return matches[0]


def validate_completed_stage_output(
    snapshot: CompletedRunSnapshot,
    *,
    stage: str,
    output_name: str,
    output_path: Path,
    root_output_suffix: str,
) -> str:
    """Validate an adjacent stage manifest against the frozen run root."""

    manifest_path = snapshot.root / "manifests" / f"{stage}.json"
    if not manifest_path.is_file() or not output_path.is_file():
        raise CompletedRunValidationError(f"completed {stage} stage manifest or output is missing")
    manifest = read_json(manifest_path)
    digest = sha256_file(output_path)
    adjacent_hash = manifest.get("outputs", {}).get(output_name, {}).get("sha256")
    root_hash = _root_output_hash(snapshot.root_manifest, root_output_suffix)
    if (
        manifest.get("manifest_schema_version") != 1
        or manifest.get("pipeline_schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("run_id") != snapshot.root.name
        or manifest.get("run_identity_sha256") != snapshot.report.get("run_identity_sha256")
        or adjacent_hash != digest
        or root_hash != digest
    ):
        raise CompletedRunValidationError(
            f"{stage} stage is incomplete, belongs to another run, or its {output_name} output changed"
        )
    return sha256_file(manifest_path)


def validate_adjusted_open_price_panel(snapshot: CompletedRunSnapshot, price_path: Path) -> str:
    """Validate adjusted opens against both their manifest and run identity."""

    manifest_path = Path(str(snapshot.config["prices"].get("manifest_path") or ""))
    if not manifest_path.is_file() or not price_path.is_file():
        raise CompletedRunValidationError("completed adjusted-open price panel or manifest is missing")
    manifest = read_json(manifest_path)
    panel_digest = sha256_file(price_path)
    manifest_digest = sha256_file(manifest_path)
    identities = snapshot.root_manifest.get("input_identities", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("adjustment_supported") is not True
        or manifest.get("execution_field") != "adjusted_open"
        or manifest.get("files", {}).get("price_panel", {}).get("sha256") != panel_digest
        or identities.get("price_panel") != panel_digest
        or identities.get("price_manifest") != manifest_digest
    ):
        raise CompletedRunValidationError("adjusted-open price panel is unsupported, incomplete, or changed")
    return manifest_digest
