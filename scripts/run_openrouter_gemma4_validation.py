#!/usr/bin/env python3
"""Prepare or run the frozen OpenRouter Gemma 4 public-benchmark gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.openrouter_validation import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    DEFAULT_LIMIT,
    prepare_manifest,
    run_validation,
    validation_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("Data/derived/labeled/financial_sentiment_v2.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("final_experiments/outputs/12_lseg_44_labelling/openrouter_gemma4_validation"),
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--retry-api-errors",
        action="store_true",
        help="Append new attempts only for rows whose latest checkpoint record failed; prior attempts remain intact.",
    )
    parser.add_argument("--confirm-paid", action="store_true", help="Required before any paid API request is sent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = validation_paths(args.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.confirm_paid:
        manifest = prepare_manifest(args.dataset, args.output_dir, args.limit)
        paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "prepared", "manifest": str(paths.manifest), "paid_requests_sent": False}, indent=2))
        return
    manifest = asyncio.run(
        run_validation(
            args.dataset,
            args.output_dir,
            limit=args.limit,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_failures=args.retry_api_errors,
        )
    )
    print(json.dumps({"status": manifest["status"], "results": manifest.get("results"), "manifest": str(paths.manifest)}, indent=2))


if __name__ == "__main__":
    main()
