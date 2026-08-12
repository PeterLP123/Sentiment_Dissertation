#!/usr/bin/env python3
"""Export validated FinBERT rows from a completed FinBERT/VADER score file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.lseg_labelling import export_finbert_only_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_finbert_only_scores(args.input, args.output)
    print(f"unique_headlines={result.unique_headlines}")
    print(f"output={result.output_path}")
    print(f"manifest={result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
