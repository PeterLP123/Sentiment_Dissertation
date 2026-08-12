#!/usr/bin/env python3
"""Materialize a compact immutable FinBERT input from terminal LSEG pages."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.incremental_finbert import materialize_incremental_finbert_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = asdict(materialize_incremental_finbert_snapshot(REPO_ROOT, args.config, args.output_dir))
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
