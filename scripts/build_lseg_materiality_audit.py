#!/usr/bin/env python3
"""Build the blinded 200/60 human audit packet after model scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.lseg_materiality import build_human_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="final_experiments/outputs/68_lseg_materiality_measurement_pilot",
    )
    args = parser.parse_args()
    manifest = build_human_audit(output_root=args.output_root)
    print(json.dumps({"status": manifest["status"], "sampling": manifest["sampling"]}, indent=2))


if __name__ == "__main__":
    main()
