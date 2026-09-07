"""Build the authorized ARS 1.1 patch for the canonical manuscript snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fenced_blocks(parsed: Any) -> tuple[dict[str, str], dict[str, str]]:
    block_ids: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for index, block in enumerate(parsed.blocks[:-1]):
        if block.kind != "heading" or not block.first_line.startswith("## "):
            continue
        following = parsed.blocks[index + 1]
        if following.kind != "fence":
            continue
        key = block.first_line[3:]
        if key == "Binary artifact manifest":
            key = "__binary_artifact_manifest__"
        if following.block_id is not None:
            block_ids[key] = following.block_id
        normalized[key] = following.normalized_text
    return block_ids, normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--ars-root", type=Path, required=True)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    sys.path.insert(0, str(args.ars_root.resolve()))
    from scripts._block_parser import base_draft_hash, parse_document  # type: ignore[import-not-found]
    from scripts.revision_roadmap import author_decision_digest  # type: ignore[import-not-found]

    original_path = artifact_dir / "original_manuscript.md"
    expected_path = artifact_dir / "revised_manuscript.expected.md"
    manifest_path = artifact_dir / "original_manuscript.block-manifest.json"
    roadmap_path = artifact_dir / "revision-roadmap.json"
    author_path = artifact_dir / "author-adjudication.json"
    claim_surface_path = artifact_dir / "claim-surface-manifest.json"
    patch_path = artifact_dir / "revision-patch.json"

    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))
    author = json.loads(author_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    original = parse_document(original_path.read_text(encoding="utf-8"))
    expected = parse_document(expected_path.read_text(encoding="utf-8"))
    block_ids, original_norm = fenced_blocks(original)
    _, expected_norm = fenced_blocks(expected)
    manifest_by_id = {row["block_id"]: row for row in manifest["blocks"]}

    author_by_item = {row["item_id"]: row for row in author["author_adjudications"]}
    block_to_items: dict[str, list[str]] = {}
    for item in roadmap["items"]:
        item_id = item["id"]
        adjudication = author_by_item[item_id]
        if adjudication["author_triage"] != "will_address":
            continue
        authorized = {
            (target["block_id"], tuple(target["allowed_operations"]))
            for target in adjudication["authorized_targets"]
        }
        for target in item["proposed_targets"]:
            witness = (target["block_id"], tuple(target["allowed_operations"]))
            if witness not in authorized:
                raise RuntimeError(f"author authority does not match roadmap target: {item_id} {witness}")
            if "replace_block" in target["allowed_operations"]:
                block_to_items.setdefault(target["block_id"], []).append(item_id)

    ops: list[dict[str, Any]] = []
    changed_keys: list[str] = []
    for key, block_id in block_ids.items():
        if original_norm[key] == expected_norm.get(key):
            continue
        item_ids = block_to_items.get(block_id, [])
        if not item_ids:
            raise RuntimeError(f"changed block lacks explicit roadmap authority: {block_id} ({key})")
        changed_keys.append(key)
        ops.append(
            {
                "op": "replace_block",
                "block_id": block_id,
                "old_hash": manifest_by_id[block_id]["old_hash"],
                "new_text": expected_norm[key],
                "roadmap_item_ids": item_ids,
                "claim_strength_changes": [],
                "collateral_authorization_ids": [],
            }
        )

    if not ops:
        raise RuntimeError("expected at least one authorized changed block")

    patch = {
        "patch_format_version": "1.1",
        "authorization_context": "review_roadmap",
        "revision_round": roadmap["revision_round"],
        "base_draft_hash": base_draft_hash(original_path.read_bytes()),
        "roadmap_sha256": sha256(roadmap_path),
        "author_adjudication_sha256": sha256(author_path),
        "author_decision_digest": author_decision_digest(author),
        "claim_surface_manifest_sha256": sha256(claim_surface_path),
        "ops": ops,
        "emitted_by": "draft_writer_agent",
    }
    write_json(patch_path, patch)
    write_json(
        artifact_dir / "revision-patch-summary.json",
        {
            "patch_sha256": sha256(patch_path),
            "operation_count": len(ops),
            "changed_canonical_surfaces": changed_keys,
        },
    )
    print(f"revision patch built: {len(ops)} authorized replace_block operations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
