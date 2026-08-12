#!/usr/bin/env python3
"""Filter validated OpenRouter successes to one immutable LSEG population."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.openrouter_lseg import LSEG_COLUMNS  # noqa: E402
from final_experiments.lib.openrouter_validation import (  # noqa: E402
    MODEL_ID,
    PROMPT_HASH,
    PROVIDER_NAME,
)
from sentiment_benchmark.artifact_io import atomic_write_json, sha256_file, sha256_text  # noqa: E402
from sentiment_benchmark.headline_value import collect_scorable_headline_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--prior-successes", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".seed.json").exists():
        raise ValueError(f"refusing to overwrite an existing Gemma resume seed: {args.output}")
    records = collect_scorable_headline_records(args.collection_root)
    population = {record.headline_sha256 for record in records}
    population_sha256 = sha256_text("\n".join(sorted(population)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    included: set[str] = set()
    input_summaries: list[dict[str, object]] = []
    try:
        writer = csv.DictWriter(temporary, fieldnames=LSEG_COLUMNS)
        writer.writeheader()
        for path in args.prior_successes:
            artifact_seen: set[str] = set()
            successful = retained = excluded = 0
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != LSEG_COLUMNS:
                    raise ValueError(f"incompatible OpenRouter seed schema: {path}")
                for line_number, row in enumerate(reader, start=2):
                    if row.get("status") != "success":
                        continue
                    successful += 1
                    headline_hash = str(row.get("headline_sha256") or "")
                    if not headline_hash or headline_hash in artifact_seen:
                        raise ValueError(f"missing or duplicate successful hash at {path}:{line_number}")
                    artifact_seen.add(headline_hash)
                    if row.get("model_id") != MODEL_ID:
                        raise ValueError(f"seed model mismatch at {path}:{line_number}")
                    if str(row.get("provider") or "").casefold() != PROVIDER_NAME.casefold():
                        raise ValueError(f"seed provider mismatch at {path}:{line_number}")
                    if row.get("quantization") != "fp8" or row.get("prompt_hash") != PROMPT_HASH:
                        raise ValueError(f"seed scorer contract mismatch at {path}:{line_number}")
                    try:
                        probabilities = [float(row[name]) for name in ("p_positive", "p_negative", "p_neutral")]
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"invalid seed probabilities at {path}:{line_number}") from exc
                    if (
                        not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
                        or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
                    ):
                        raise ValueError(f"invalid seed probabilities at {path}:{line_number}")
                    if headline_hash not in population:
                        excluded += 1
                        continue
                    if headline_hash in included:
                        raise ValueError(f"overlapping retained success across seed artifacts: {headline_hash}")
                    included.add(headline_hash)
                    retained += 1
                    writer.writerow(row)
            input_summaries.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "successful_rows": successful,
                    "retained_rows": retained,
                    "excluded_outside_population": excluded,
                }
            )
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        os.replace(temporary_path, args.output)
    except Exception:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "kind": "openrouter_lseg_population_filtered_resume_seed",
        "population": len(population),
        "population_sha256": population_sha256,
        "retained_unique_successes": len(included),
        "pending_hashes": len(population) - len(included),
        "excluded_outside_population": sum(
            int(item["excluded_outside_population"]) for item in input_summaries
        ),
        "inputs": input_summaries,
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "prices_or_returns_loaded": False,
        "sharing": {
            "contains_licensed_headline_text": True,
            "source_control": False,
            "redistribute": False,
        },
    }
    manifest_path = atomic_write_json(args.output.with_suffix(args.output.suffix + ".seed.json"), manifest)
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
