"""Build the current-round and continuous two-round ARS evidence bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARS = ROOT / "reviews" / "ars" / "current_re_review"
ROUND2 = ARS / "round2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def bundle_artifact(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
    }


def manifest_entry(path: Path) -> dict[str, Any]:
    return {
        "present": True,
        "path_or_passport_ref": f"path:{path.name}",
        "sha256": sha256(path),
        "version_label": None,
        "origin_date": None,
    }


def manifest_array(paths: list[Path]) -> dict[str, Any]:
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


def round2_row(root: Path) -> dict[str, Any]:
    return {
        "kind": "review_roadmap",
        "revision_round": 2,
        "pre_round_draft": bundle_artifact(ROUND2 / "original_manuscript.md", root),
        "pre_round_block_manifest": bundle_artifact(
            ROUND2 / "original_manuscript.block-manifest.json", root
        ),
        "revision_roadmap": bundle_artifact(ROUND2 / "revision-roadmap.json", root),
        "claim_surface_manifest": bundle_artifact(
            ROUND2 / "claim-surface-manifest.json", root
        ),
        "author_adjudication": bundle_artifact(
            ROUND2 / "author-adjudication.json", root
        ),
        "revision_patch": bundle_artifact(ROUND2 / "revision-patch.json", root),
        "apply_report": bundle_artifact(
            ROUND2 / "revision-apply-report.json", root
        ),
        "post_round_draft": bundle_artifact(ROUND2 / "revised_manuscript.md", root),
    }


def main() -> int:
    original = ROUND2 / "original_manuscript.md"
    manifest = ROUND2 / "original_manuscript.block-manifest.json"
    integrity = ROUND2 / "integrity-pass.json"
    roadmap = ROUND2 / "revision-roadmap.json"
    author = ROUND2 / "author-adjudication.json"
    patch = ROUND2 / "revision-patch.json"
    report = ROUND2 / "revision-apply-report.json"
    revised = ROUND2 / "revised_manuscript.md"
    required = [
        original,
        manifest,
        integrity,
        roadmap,
        ROUND2 / "claim-surface-manifest.json",
        author,
        patch,
        report,
        revised,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Round-2 artifacts: {missing}")

    receipt = json.loads(integrity.read_text(encoding="utf-8"))
    if receipt["checked_draft_sha256"] != sha256(original):
        raise RuntimeError("Round-2 integrity receipt does not bind the original draft")

    current_bundle = {
        "schema_version": "revision-evidence-bundle/1.0",
        "chain_start": {
            "first_revision_round": 2,
            "draft": bundle_artifact(original, ROUND2),
            "block_manifest": bundle_artifact(manifest, ROUND2),
            "integrity_pass_receipt": bundle_artifact(integrity, ROUND2),
        },
        "rounds": [round2_row(ROUND2)],
        "final_draft": bundle_artifact(revised, ROUND2),
    }
    current_bundle_path = ROUND2 / "revision-evidence-bundle.json"
    write_json(current_bundle_path, current_bundle)

    prior_bundle_path = ARS / "revision-evidence-bundle.json"
    prior = json.loads(prior_bundle_path.read_text(encoding="utf-8"))
    if prior["final_draft"]["sha256"] != sha256(original):
        raise RuntimeError("Round-1 final draft does not equal the Round-2 original")
    continuous_bundle = {
        "schema_version": "revision-evidence-bundle/1.0",
        "chain_start": prior["chain_start"],
        "rounds": [*prior["rounds"], round2_row(ARS)],
        "final_draft": bundle_artifact(revised, ARS),
    }
    continuous_bundle_path = ARS / "two-round-revision-evidence-bundle.json"
    write_json(continuous_bundle_path, continuous_bundle)

    input_manifest = {
        "contract_version": "1.1",
        "round_id": "ars-re-review-2026-08-14-round-2",
        "cross_model_active": False,
        "artifacts": {
            "original_manuscript": manifest_entry(original),
            "revised_manuscript": manifest_entry(revised),
            "revision_roadmap": manifest_entry(roadmap),
            "author_adjudication": manifest_entry(author),
            "revision_evidence_bundle": manifest_entry(current_bundle_path),
            "editorial_decision_letter": {"present": False},
            "response_to_reviewers": {"present": False},
            "revision_patches": manifest_array([patch]),
            "apply_reports": manifest_array([report]),
            "round1_findings": {"present": False},
            "round1_config_cards": {"present": False},
        },
    }
    input_manifest_path = ROUND2 / "input-manifest.json"
    write_json(input_manifest_path, input_manifest)
    print(f"current bundle: {sha256(current_bundle_path)}")
    print(f"continuous bundle: {sha256(continuous_bundle_path)}")
    print(f"input manifest: {sha256(input_manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
