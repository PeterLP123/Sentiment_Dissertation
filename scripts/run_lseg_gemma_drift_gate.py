#!/usr/bin/env python3
"""Prepare and evaluate the frozen LSEG Gemma model-drift gate."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.backward_validation import (  # noqa: E402
    evaluate_gemma_drift,
    select_gemma_drift_sample,
)
from sentiment_benchmark.artifact_io import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    sha256_file,
    sha256_text,
)


def _prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"drift output directory is not empty: {output_dir}")
    source_manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    config = source_manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("reference manifest has no parsed collection config")

    sample = select_gemma_drift_sample(args.reference_scores)
    selected_hashes = set(sample["headline_sha256"].astype(str))
    metadata: dict[str, dict[str, str]] = {}
    with args.reference_scores.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            headline_hash = str(row.get("headline_sha256") or "")
            if row.get("status") != "success" or headline_hash not in selected_hashes:
                continue
            if headline_hash in metadata:
                raise ValueError(f"duplicate selected reference success: {headline_hash}")
            metadata[headline_hash] = row
    if set(metadata) != selected_hashes:
        raise ValueError("reference metadata does not cover the frozen drift sample")

    collection = config.get("collection") or {}
    fallback_timestamp = str(collection.get("start") or "")
    rows = []
    for item in sample.itertuples(index=False):
        source = metadata[str(item.headline_sha256)]
        matched_symbols = [value for value in str(source.get("matched_symbols") or "").split("|") if value]
        if not matched_symbols:
            matched_symbols = [str(config["companies"][0]["symbol"])]
        timestamp = str(source.get("first_timestamp") or fallback_timestamp)
        rows.append(
            {
                "story_id": f"gemma-drift-{item.headline_sha256}",
                "headline": str(item.headline),
                "matched_symbols": matched_symbols,
                "first_created": timestamp,
                "version_created": timestamp,
                "source_code": "frozen_gemma_drift_reference",
                "language": "en",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    headlines_path = atomic_write_jsonl(output_dir / "headlines.jsonl", rows)
    reference_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        reference_buffer,
        fieldnames=("headline_sha256", "reference_score", "stratum"),
    )
    writer.writeheader()
    writer.writerows(sample[["headline_sha256", "reference_score", "stratum"]].to_dict("records"))
    reference_path = atomic_write_text(output_dir / "reference_sample.csv", reference_buffer.getvalue())
    population_sha256 = sha256_text("\n".join(sorted(selected_hashes)))
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "kind": "frozen_gemma_model_drift_gate",
        "config": config,
        "sample": {
            "rows": len(sample),
            "endpoint_rows": int(sample["stratum"].eq("endpoint").sum()),
            "nonendpoint_control_rows": int(sample["stratum"].eq("nonendpoint_control").sum()),
            "population_sha256": population_sha256,
            "selection_seed_prefix": "20260806:",
        },
        "reference": {
            "scores_path": str(args.reference_scores.resolve()),
            "scores_sha256": sha256_file(args.reference_scores),
            "manifest_path": str(args.reference_manifest.resolve()),
            "manifest_sha256": sha256_file(args.reference_manifest),
        },
        "files": {
            "headlines_jsonl": {"path": headlines_path.name, "sha256": sha256_file(headlines_path)},
            "reference_sample_csv": {"path": reference_path.name, "sha256": sha256_file(reference_path)},
        },
        "prices_or_returns_loaded": False,
        "sharing": {
            "contains_licensed_headline_text": True,
            "source_control": False,
            "redistribute": False,
        },
    }
    manifest_path = atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "prepared", "manifest": str(manifest_path), "sample": manifest["sample"]}, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    reference = pd.read_csv(args.reference_sample)
    repeated = pd.read_csv(args.repeated_scores, low_memory=False)
    audit = evaluate_gemma_drift(reference, repeated)
    payload = {
        "schema_version": 1,
        "status": "passed" if audit.drift_gate_pass else "failed",
        "drift_gate_pass": audit.drift_gate_pass,
        "metrics": audit.metrics.iloc[0].to_dict(),
        "gates": audit.gate_table.to_dict("records"),
        "inputs": {
            "reference_sample": str(args.reference_sample.resolve()),
            "reference_sample_sha256": sha256_file(args.reference_sample),
            "repeated_scores": str(args.repeated_scores.resolve()),
            "repeated_scores_sha256": sha256_file(args.repeated_scores),
        },
        "prices_or_returns_loaded": False,
    }
    output = atomic_write_json(args.output, payload)
    print(json.dumps({**payload, "output": str(output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--reference-scores", type=Path, required=True)
    prepare.add_argument("--reference-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.set_defaults(func=_prepare)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--reference-sample", type=Path, required=True)
    evaluate.add_argument("--repeated-scores", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=_evaluate)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
