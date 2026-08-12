#!/usr/bin/env python3
"""Build a target-metadata-canonical baseline resume seed from prior scores."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.lseg_labelling import prepare_reusable_baseline_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--prior-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = asdict(prepare_reusable_baseline_seed(args.collection_root, args.prior_scores, args.output))
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
