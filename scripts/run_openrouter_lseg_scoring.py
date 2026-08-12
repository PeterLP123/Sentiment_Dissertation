#!/usr/bin/env python3
"""Run or resume the licensed LSEG OpenRouter Gemma 4 scoring job."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.openrouter_lseg import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_POPULATION,
    score_lseg_openrouter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-population", type=int, default=DEFAULT_MAX_POPULATION)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seed-successes",
        type=Path,
        action="append",
        default=[],
        help="Immutable prior OpenRouter CSV whose validated successes should be reused by headline hash.",
    )
    parser.add_argument(
        "--confirm-licensed-external-processing",
        action="store_true",
        help="Required acknowledgement that the researcher confirmed hosted processing is permitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_licensed_external_processing:
        raise SystemExit("refusing hosted LSEG scoring without --confirm-licensed-external-processing")
    manifest = asyncio.run(
        score_lseg_openrouter(
            args.collection_root,
            args.output,
            concurrency=args.concurrency,
            retries=args.retries,
            max_population=args.max_population,
            limit=args.limit,
            seed_success_paths=tuple(args.seed_successes),
            callback=print,
        )
    )
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "usage": manifest["usage_this_run"]}, indent=2))


if __name__ == "__main__":
    main()
