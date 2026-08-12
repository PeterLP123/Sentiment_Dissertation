#!/usr/bin/env python3
"""Evaluate the frozen human reliability gate without opening returns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.lseg_materiality import validate_human_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="final_experiments/outputs/68_lseg_materiality_measurement_pilot",
    )
    args = parser.parse_args()
    report = validate_human_audit(output_root=args.output_root)
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
