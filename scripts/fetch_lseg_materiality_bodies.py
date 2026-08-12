#!/usr/bin/env python3
"""Resume selective LSEG body retrieval for the frozen materiality sample."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare_lseg_materiality_pilot import frozen_inputs  # noqa: E402

from final_experiments.lib.lseg_materiality import fetch_materiality_bodies  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="final_experiments/outputs/68_lseg_materiality_measurement_pilot",
    )
    parser.add_argument("--max-new-requests", type=int, default=2_000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=3.0)
    parser.add_argument("--retry-terminal-failures", action="store_true")
    args = parser.parse_args()
    manifest = asyncio.run(
        fetch_materiality_bodies(
            frozen_inputs(),
            output_root=args.output_root,
            max_new_requests=args.max_new_requests,
            concurrency=args.concurrency,
            requests_per_second=args.requests_per_second,
            retry_terminal_failures=args.retry_terminal_failures,
        )
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
