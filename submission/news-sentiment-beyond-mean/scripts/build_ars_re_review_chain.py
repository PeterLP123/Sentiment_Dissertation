"""Build the hash-bound ARS re-review evidence bundle and input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def artifact(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256(path)}


def entry(path: Path) -> dict[str, Any]:
    return {
        "present": True,
        "path_or_passport_ref": f"path:{path.name}",
        "sha256": sha256(path),
        "version_label": None,
        "origin_date": None,
    }


def array_entry(paths: list[Path]) -> dict[str, Any]:
    return {
        "present": True,
        "items": [
            {
                "path_or_passport_ref": f"path:{path.name}",
                "sha256": sha256(path),
                "version_label": None,
                "origin_date": None,
            }
            for path in paths
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact_dir.resolve()
    original = root / "original_manuscript.md"
    block_manifest = root / "original_manuscript.block-manifest.json"
    integrity_receipt = root / "integrity-pass.json"
    roadmap = root / "revision-roadmap.json"
    claim_surface = root / "claim-surface-manifest.json"
    author = root / "author-adjudication.json"
    patch = root / "revision-patch.json"
    apply_report = root / "apply-report.json"
    revised = root / "revised_manuscript.md"
    round1_findings = root / "round1_findings.md"

    required = [
        original,
        block_manifest,
        integrity_receipt,
        roadmap,
        claim_surface,
        author,
        patch,
        apply_report,
        revised,
        round1_findings,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing chain artifacts: {missing}")

    bundle_path = root / "revision-evidence-bundle.json"
    bundle = {
        "schema_version": "revision-evidence-bundle/1.0",
        "chain_start": {
            "first_revision_round": 1,
            "draft": artifact(original),
            "block_manifest": artifact(block_manifest),
            "integrity_pass_receipt": artifact(integrity_receipt),
        },
        "rounds": [
            {
                "kind": "review_roadmap",
                "revision_round": 1,
                "pre_round_draft": artifact(original),
                "pre_round_block_manifest": artifact(block_manifest),
                "revision_roadmap": artifact(roadmap),
                "claim_surface_manifest": artifact(claim_surface),
                "author_adjudication": artifact(author),
                "revision_patch": artifact(patch),
                "apply_report": artifact(apply_report),
                "post_round_draft": artifact(revised),
            }
        ],
        "final_draft": artifact(revised),
    }
    write_json(bundle_path, bundle)

    manifest_path = root / "input-manifest.json"
    manifest = {
        "contract_version": "1.1",
        "round_id": "ars-re-review-2026-08-14-round-1",
        "cross_model_active": False,
        "artifacts": {
            "original_manuscript": entry(original),
            "revised_manuscript": entry(revised),
            "revision_roadmap": entry(roadmap),
            "author_adjudication": entry(author),
            "revision_evidence_bundle": entry(bundle_path),
            "editorial_decision_letter": {"present": False},
            "response_to_reviewers": {"present": False},
            "revision_patches": array_entry([patch]),
            "apply_reports": array_entry([apply_report]),
            "round1_findings": entry(round1_findings),
            "round1_config_cards": {"present": False},
        },
    }
    write_json(manifest_path, manifest)
    print(f"evidence bundle: {sha256(bundle_path)}")
    print(f"input manifest: {sha256(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
