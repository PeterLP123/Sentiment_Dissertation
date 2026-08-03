#!/usr/bin/env python3
"""Seed expanded LSEG labelling with exact compatible prior baseline rows."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--prior-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = prepare_reusable_baseline_seed(args.collection_root, args.prior_scores, args.output)
    print(json.dumps({key: str(value) for key, value in asdict(summary).items()}, indent=2))


if __name__ == "__main__":
    main()
