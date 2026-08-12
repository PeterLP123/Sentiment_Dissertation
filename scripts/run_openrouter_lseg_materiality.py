#!/usr/bin/env python3
"""Resume authorised DeepInfra/Gemma scoring for the materiality pilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.lseg_materiality import score_materiality_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="final_experiments/outputs/68_lseg_materiality_measurement_pilot",
    )
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--confirm-authorized-transfer", action="store_true")
    args = parser.parse_args()
    manifest = asyncio.run(
        score_materiality_sample(
            output_root=args.output_root,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_failures=args.retry_failures,
            max_new_calls=args.max_new_calls,
            confirm_authorized_transfer=args.confirm_authorized_transfer,
        )
    )
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "usage": manifest["usage"]}, indent=2))


if __name__ == "__main__":
    main()
