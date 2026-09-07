"""Build the non-authoritative scaffold for a formal ARS re-review.

The repository is a multi-file LaTeX project, while the ARS revision witness
operates on one anchored Markdown artifact.  This script creates canonical
source snapshots from Git HEAD and the current worktree, anchors the original,
and emits a reviewer-owned roadmap plus an exact author-authorization request.

It deliberately does not emit an author-adjudication sidecar, revision patch,
apply report, evidence bundle, or re-review verdict.  Those require a later
explicit author confirmation of the item/block/operation mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".tex", ".bib"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}
BINARY_MANIFEST_KEY = "__binary_artifact_manifest__"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def git_output(*args: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def git_head_bytes(path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def head_paths() -> set[str]:
    output = git_output("ls-tree", "-r", "--name-only", "HEAD", "--", "manuscript")
    return {line for line in output.decode("utf-8").splitlines() if line}


def worktree_paths() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "manuscript").rglob("*")
        if path.is_file()
    }


def ordered_source_paths(paths: set[str]) -> list[str]:
    selected = {
        path
        for path in paths
        if Path(path).suffix.lower() in SOURCE_SUFFIXES
        and (
            path == "manuscript/main.tex"
            or path == "manuscript/references.bib"
            or path.startswith("manuscript/chapters/")
            or path.startswith("manuscript/appendices/")
            or path.startswith("manuscript/tables/")
            or path.startswith("manuscript/artifacts/")
        )
    }

    def key(path: str) -> tuple[int, str]:
        if path == "manuscript/main.tex":
            return (0, path)
        if path.startswith("manuscript/chapters/"):
            return (1, path)
        if path.startswith("manuscript/appendices/"):
            return (2, path)
        if path.startswith("manuscript/tables/"):
            return (3, path)
        if path.startswith("manuscript/artifacts/"):
            return (4, path)
        if path == "manuscript/references.bib":
            return (5, path)
        return (6, path)

    return sorted(selected, key=key)


def binary_manifest(
    paths: list[str],
    *,
    revision: str,
) -> str:
    rows = ["path\tstatus\tsha256\tbytes"]
    for path in paths:
        if revision == "original":
            raw = git_head_bytes(path)
        else:
            candidate = ROOT / path
            raw = candidate.read_bytes() if candidate.is_file() else None
        if raw is None:
            rows.append(f"{path}\tabsent\t-\t0")
        else:
            rows.append(f"{path}\tpresent\t{sha256_bytes(raw)}\t{len(raw)}")
    return "\n".join(rows) + "\n"


def source_bytes(path: str, revision: str) -> bytes | None:
    if revision == "original":
        return git_head_bytes(path)
    candidate = ROOT / path
    return candidate.read_bytes() if candidate.is_file() else None


def render_snapshot(
    source_paths: list[str],
    binary_paths: list[str],
    *,
    revision: str,
) -> str:
    parts = [
        "# ARS canonical manuscript source snapshot\n\n",
        "This review artifact contains the manuscript source files and a hash manifest "
        "for image assets. It is not the submission PDF.\n\n",
    ]
    for path in source_paths:
        raw = source_bytes(path, revision)
        if raw is None:
            content = f"% ARS snapshot: {path} is absent in this revision.\n"
        else:
            content = raw.decode("utf-8").replace("\r\n", "\n")
            if not content.endswith("\n"):
                content += "\n"
        language = "bibtex" if path.endswith(".bib") else "latex"
        parts.extend(
            [
                f"## {path}\n\n",
                f"````{language}\n{content}````\n\n",
            ]
        )
    parts.extend(
        [
            "## Binary artifact manifest\n\n",
            "````text\n",
            binary_manifest(binary_paths, revision=revision),
            "````\n",
        ]
    )
    return "".join(parts)


def parse_block_map(
    ars_root: Path,
    snapshot: Path,
    *,
    require_ids: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    sys.path.insert(0, str(ars_root))
    from scripts._block_parser import parse_document  # type: ignore[import-not-found]

    parsed = parse_document(snapshot.read_text(encoding="utf-8"))
    block_ids: dict[str, str] = {}
    normalized: dict[str, str] = {}
    blocks = parsed.blocks
    for index, block in enumerate(blocks[:-1]):
        if block.kind != "heading" or not block.first_line.startswith("## "):
            continue
        key = block.first_line[3:]
        following = blocks[index + 1]
        if following.kind != "fence":
            raise RuntimeError(f"expected fenced content after heading {key!r}")
        if key == "Binary artifact manifest":
            key = BINARY_MANIFEST_KEY
        if require_ids and following.block_id is None:
            raise RuntimeError(f"expected anchored fenced content after heading {key!r}")
        if following.block_id is not None:
            block_ids[key] = following.block_id
        normalized[key] = following.normalized_text
    return block_ids, normalized


def targets(block_ids: dict[str, str], paths: list[str]) -> list[dict[str, Any]]:
    values = [
        {
            "block_id": block_ids[path],
            "allowed_operations": ["replace_block"],
        }
        for path in paths
    ]
    return sorted(values, key=lambda value: int(value["block_id"][1:]))


def roadmap_items(block_ids: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "id": "REV-ESTIMAND-ALIGNMENT",
            "source_refs": [
                {"seat": "EIC", "channel": "editorial", "ordinal": 1, "subclaim_ordinal": 0}
            ],
            "description": "The abstract and conclusion mix the training-period price-control coefficient with the baseline coefficient used in the frozen across-era contrast.",
            "reviewer": "EIC",
            "obligation_class": "must_fix",
            "severity": "major",
            "evidence_anchor": {
                "anchor_type": "text",
                "locator": "original abstract and conclusion",
                "quote": "With one-session, five-session and volatility controls, its coefficient is -0.00914 ... evaluation coefficient is instead +0.00583",
            },
            "confidence": 5,
            "competence_basis": "direct comparison of the stated training-period and frozen specifications",
            "cost_scope": {"kind": "section", "locator": "abstract, introduction, methodology, results and conclusion"},
            "consequence_if_unaddressed": {
                "code": "claim_scope_unsupported",
                "target": {"kind": "claim", "locator": "training-to-testing comparison"},
            },
            "target_section": "abstract through conclusion",
            "suggested_action": "Separate the training-period baseline, price-control robustness and frozen baseline replication wherever the estimates are compared.",
            "consensus_level": "CONSENSUS-4",
            "verification_criteria": "Every comparison states that -0.00831 is the training-period baseline paired with +0.00583, while -0.00914 remains separate training-period robustness.",
            "proposed_targets": targets(
                block_ids,
                [
                    "manuscript/main.tex",
                    "manuscript/chapters/01_introduction.tex",
                    "manuscript/chapters/04_methodology.tex",
                    "manuscript/chapters/05_results.tex",
                    "manuscript/chapters/06_conclusion.tex",
                ],
            ),
        },
        {
            "id": "REV-INTERPRETATION-BOUNDARY",
            "source_refs": [
                {"seat": "R1", "channel": "finding", "ordinal": 1, "subclaim_ordinal": 0}
            ],
            "description": "The manuscript describes temporal instability without consistently separating a measured pipeline change from market, sample-composition, timing or classifier explanations.",
            "reviewer": "R1",
            "obligation_class": "must_fix",
            "severity": "major",
            "evidence_anchor": {
                "anchor_type": "text",
                "locator": "original conclusion section 6.1",
                "quote": "Together, the failed families and direct contrast establish temporal non-replication and instability under the frozen design.",
            },
            "confidence": 5,
            "competence_basis": "statistical interpretation and transport-validity review",
            "cost_scope": {"kind": "section", "locator": "abstract, literature, data, results and conclusion"},
            "consequence_if_unaddressed": {
                "code": "interpretive_ambiguity_remains",
                "target": {"kind": "claim", "locator": "cause of the across-era coefficient change"},
            },
            "target_section": "temporal non-replication interpretation",
            "suggested_action": "State that the contrast identifies instability in the measured baseline pipeline association and does not identify its cause; name plausible composition and measurement alternatives.",
            "consensus_level": "CONSENSUS-4",
            "verification_criteria": "The revised manuscript consistently limits the finding to measured pipeline-level instability and explicitly leaves its cause unidentified.",
            "proposed_targets": targets(
                block_ids,
                [
                    "manuscript/main.tex",
                    "manuscript/chapters/02_literature.tex",
                    "manuscript/chapters/03_data.tex",
                    "manuscript/chapters/05_results.tex",
                    "manuscript/chapters/06_conclusion.tex",
                ],
            ),
        },
        {
            "id": "REV-HYPOTHESIS-LANGUAGE",
            "source_refs": [
                {"seat": "R1", "channel": "finding", "ordinal": 2, "subclaim_ordinal": 0}
            ],
            "description": "The results call H1b rejected even though the declared directional effect is not supported in the frozen evaluation.",
            "reviewer": "R1",
            "obligation_class": "should_fix",
            "severity": "minor",
            "evidence_anchor": {
                "anchor_type": "text",
                "locator": "original results section 5.2",
                "quote": "H1b is rejected under its recorded direction and correction.",
            },
            "confidence": 5,
            "competence_basis": "hypothesis-testing terminology review",
            "cost_scope": {"kind": "sentence", "locator": "hypothesis verdict statements"},
            "consequence_if_unaddressed": {
                "code": "reporting_requirement_unmet",
                "target": {"kind": "claim", "locator": "H1b verdict"},
            },
            "target_section": "results and hypothesis summary",
            "suggested_action": "Use not supported for H1b and keep hypothesis labels consistent in the final summary.",
            "consensus_level": "SINGLE-VERIFIER",
            "verification_criteria": "H1b is described as not supported, and the final hypothesis summary identifies the correct training-period hypothesis label.",
            "proposed_targets": targets(
                block_ids,
                [
                    "manuscript/chapters/01_introduction.tex",
                    "manuscript/chapters/05_results.tex",
                    "manuscript/chapters/06_conclusion.tex",
                ],
            ),
        },
        {
            "id": "REV-TEMPORAL-LITERATURE",
            "source_refs": [
                {"seat": "R2", "channel": "finding", "ordinal": 1, "subclaim_ordinal": 0}
            ],
            "description": "The literature review lacks direct empirical-finance context for out-of-sample predictor instability and post-publication attenuation.",
            "reviewer": "R2",
            "obligation_class": "should_fix",
            "severity": "major",
            "evidence_anchor": {
                "anchor_type": "absence",
                "locator": "original literature review",
                "absence_scope": "temporal transport and predictor-stability literature",
                "check_performed": "checked the literature chapter and bibliography for direct predictor-stability framing",
            },
            "confidence": 4,
            "competence_basis": "empirical-asset-pricing literature review",
            "cost_scope": {"kind": "section", "locator": "literature review and bibliography"},
            "consequence_if_unaddressed": {
                "code": "evidence_gap_remains",
                "target": {"kind": "section", "locator": "temporal transport positioning"},
            },
            "target_section": "literature review",
            "suggested_action": "Add verified predictor-stability studies and distinguish local predictability, out-of-sample failure and publication-related attenuation from the paper's measured non-replication.",
            "consensus_level": "SINGLE-VERIFIER",
            "verification_criteria": "The literature chapter cites verified primary metadata and uses it to bound, rather than overstate, the temporal contribution.",
            "proposed_targets": targets(
                block_ids,
                ["manuscript/chapters/02_literature.tex", "manuscript/references.bib"],
            ),
        },
        {
            "id": "REV-WORD-COUNT",
            "source_refs": [
                {"seat": "R3", "channel": "finding", "ordinal": 1, "subclaim_ordinal": 0}
            ],
            "description": "The six numbered chapters contain 12,107 words, materially above the programme's approximate 10,000-word target.",
            "reviewer": "R3",
            "obligation_class": "should_fix",
            "severity": "minor",
            "evidence_anchor": {
                "anchor_type": "dataset",
                "locator": "docs/word_count.md original count: 12,107 main-chapter words",
            },
            "confidence": 5,
            "competence_basis": "submission-readiness and document-structure review",
            "cost_scope": {"kind": "section", "locator": "six numbered chapters"},
            "consequence_if_unaddressed": {
                "code": "editorial_conformance_unmet",
                "target": {"kind": "manuscript", "locator": "main-chapter word count"},
            },
            "target_section": "numbered chapters",
            "suggested_action": "Compress repeated secondary trading, risk, transfer and prompt discussion while preserving estimates, uncertainty, nulls, multiplicity, costs and power.",
            "consensus_level": "CONSENSUS-3",
            "verification_criteria": "The six numbered chapters fall within ten percent of the approximate 10,000-word target without removing required evidence or limitations.",
            "proposed_targets": targets(
                block_ids,
                [
                    "manuscript/main.tex",
                    "manuscript/chapters/01_introduction.tex",
                    "manuscript/chapters/02_literature.tex",
                    "manuscript/chapters/03_data.tex",
                    "manuscript/chapters/04_methodology.tex",
                    "manuscript/chapters/05_results.tex",
                    "manuscript/chapters/06_conclusion.tex",
                ],
            ),
        },
        {
            "id": "REV-PRESENTATION",
            "source_refs": [
                {"seat": "DA", "channel": "finding", "ordinal": 1, "subclaim_ordinal": 0}
            ],
            "description": "Front-matter page numbering duplicates Arabic destinations and the schedule-shift figure uses the malformed label 99.1th.",
            "reviewer": "DA",
            "obligation_class": "should_fix",
            "severity": "minor",
            "evidence_anchor": {
                "anchor_type": "figure",
                "locator": "title/front matter and Figure 5.5 schedule-shift annotation",
            },
            "confidence": 5,
            "competence_basis": "PDF presentation and reproducibility review",
            "cost_scope": {"kind": "other", "surface_id": "presentation_assets", "locator": "front matter and schedule-shift figure"},
            "consequence_if_unaddressed": {
                "code": "editorial_conformance_unmet",
                "target": {"kind": "figure", "locator": "front matter and Figure 5.5"},
            },
            "target_section": "front matter and schedule-shift figure",
            "suggested_action": "Use Roman front-matter numbering and replace the malformed percentile suffix in the generated figure while retaining the underlying statistic.",
            "consensus_level": "SINGLE-VERIFIER",
            "verification_criteria": "The PDF has distinct Roman front matter, Arabic main matter and a timing figure labelled 99.1 percentile with unchanged evidence values.",
            "proposed_targets": targets(
                block_ids,
                ["manuscript/main.tex", BINARY_MANIFEST_KEY],
            ),
        },
    ]


def findings_markdown() -> str:
    return """# Reconstructed Round-1 Findings\n\nThis carrier reconstructs the six findings recorded in the completed review and in `docs/dissertation_quality_review.md`. The original chat-only reviewer cards were not persisted, so the current re-review must disclose `[YARDSTICK-REGENERATED: original manuscript — round-1 cards unavailable]`.\n\n1. Align the training-period baseline, full price-control robustness estimate and frozen baseline replication.\n2. Bound the temporal claim to instability in the measured pipeline association without identifying its cause.\n3. Replace the H1b rejection wording with a non-support verdict and keep hypothesis labels consistent.\n4. Add verified empirical-finance context for predictor stability and temporal transport.\n5. Reduce the numbered-chapter count from 12,107 toward the approximate 10,000-word target without removing required null, uncertainty, multiplicity, cost or power evidence.\n6. Correct front-matter page destinations and the malformed 99.1th schedule-shift label.\n\nThe editorial decision for this reconstructed set is Major Revision.\n"""


def authorization_request(
    roadmap: dict[str, Any],
    *,
    roadmap_sha256: str,
    claim_surface_sha256: str,
    changed_keys: list[str],
) -> str:
    lines = [
        "# ARS author-adjudication request\n",
        "Status: **AWAITING EXPLICIT AUTHOR CONFIRMATION**. This is not an `author-adjudication/1.0` sidecar.\n",
        f"- Roadmap SHA-256: `{roadmap_sha256}`",
        f"- Claim-surface manifest SHA-256: `{claim_surface_sha256}`",
        "- Proposed triage for every item: `will_address`",
        "- Proposed operation for every listed block: `replace_block`",
        "\n## Exact proposed authority\n",
        "| Item | Blocks | Operation |",
        "|---|---|---|",
    ]
    for item in roadmap["items"]:
        block_list = ", ".join(target["block_id"] for target in item["proposed_targets"])
        lines.append(f"| `{item['id']}` | `{block_list}` | `replace_block` |")
    lines.extend(
        [
            "\n## Changed canonical surfaces\n",
            *[f"- `{key}`" for key in changed_keys],
            "\nTo authorize the deterministic sidecar and patch application, the author must explicitly confirm this exact roadmap hash and every listed block/operation mapping.\n",
        ]
    )
    return "\n".join(lines)


def build(output_dir: Path, ars_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    head = head_paths()
    worktree = worktree_paths()
    source_paths = ordered_source_paths(head | worktree)
    binary_paths = sorted(
        path
        for path in head | worktree
        if Path(path).suffix.lower() in BINARY_SUFFIXES
    )

    original_raw = output_dir / "original_manuscript.raw.md"
    original = output_dir / "original_manuscript.md"
    revised_expected = output_dir / "revised_manuscript.expected.md"
    original_raw.write_text(
        render_snapshot(source_paths, binary_paths, revision="original"),
        encoding="utf-8",
    )
    revised_expected.write_text(
        render_snapshot(source_paths, binary_paths, revision="revised"),
        encoding="utf-8",
    )
    shutil.copyfile(original_raw, original)

    anchorizer = ars_root / "scripts" / "ars_anchorize_draft.py"
    manifest_path = output_dir / "original_manuscript.block-manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(anchorizer),
            str(original),
            "--manifest-out",
            str(manifest_path),
        ],
        cwd=ars_root,
        check=True,
    )

    original_ids, original_blocks = parse_block_map(
        ars_root,
        original,
        require_ids=True,
    )
    _revised_ids, revised_blocks = parse_block_map(
        ars_root,
        revised_expected,
        require_ids=False,
    )
    if set(original_ids) != set(revised_blocks):
        raise RuntimeError("original/revised canonical snapshot keys do not match")
    changed_keys = sorted(
        key for key in original_ids if original_blocks[key] != revised_blocks[key]
    )

    block_manifest_raw = manifest_path.read_bytes()
    roadmap = {
        "schema_version": "revision-roadmap/1.0",
        "revision_round": 1,
        "base_draft_sha256": sha256_bytes(original.read_bytes()),
        "block_manifest_sha256": sha256_bytes(block_manifest_raw),
        "items": roadmap_items(original_ids),
        "total_items": 6,
        "obligation_counts": {"must_fix": 2, "should_fix": 4, "consider": 0},
        "editorial_decision": "Major Revision",
        "consensus_summary": "The reconstructed review record requires aligned estimands and bounded interpretation, with supporting revisions to hypothesis language, literature context, length and presentation.",
        "dissenting_opinions": [],
    }
    roadmap_path = output_dir / "revision-roadmap.json"
    write_json(roadmap_path, roadmap)
    roadmap_sha = sha256_bytes(roadmap_path.read_bytes())

    claim_surface = {
        "schema_version": "claim-surface-manifest/1.0",
        "revision_round": 1,
        "roadmap_sha256": roadmap_sha,
        "base_draft_sha256": sha256_bytes(original.read_bytes()),
        "claim_intent_sources": [],
        "surfaces": [],
    }
    claim_surface_path = output_dir / "claim-surface-manifest.json"
    write_json(claim_surface_path, claim_surface)
    claim_surface_sha = sha256_bytes(claim_surface_path.read_bytes())

    findings_path = output_dir / "round1_findings.md"
    findings_path.write_text(findings_markdown(), encoding="utf-8")
    request_path = output_dir / "author-authorization-request.md"
    request_path.write_text(
        authorization_request(
            roadmap,
            roadmap_sha256=roadmap_sha,
            claim_surface_sha256=claim_surface_sha,
            changed_keys=changed_keys,
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "AWAITING_EXPLICIT_AUTHOR_CONFIRMATION",
        "original_manuscript_sha256": sha256_bytes(original.read_bytes()),
        "revised_expected_sha256": sha256_bytes(revised_expected.read_bytes()),
        "roadmap_sha256": roadmap_sha,
        "claim_surface_manifest_sha256": claim_surface_sha,
        "changed_canonical_surfaces": changed_keys,
        "source_file_count": len(source_paths),
        "binary_artifact_count": len(binary_paths),
    }
    write_json(output_dir / "scaffold-summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ars-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.ars_root.resolve())


if __name__ == "__main__":
    main()
