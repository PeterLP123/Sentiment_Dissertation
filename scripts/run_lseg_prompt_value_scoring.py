#!/usr/bin/env python3
"""Build and score the frozen LSEG prompt-economic-value population."""

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

from final_experiments.lib.lseg_prompt_value import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_PROMPTS,
    OPERATIONAL_SPEND_CEILING_USD,
    build_prompt_candidate_population,
    score_prompt_variant,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--prompt-id",
        action="append",
        choices=sorted(EXPECTED_PROMPTS),
        help="Prompt to score; repeat as needed. Defaults to all five in frozen order.",
    )
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-authorized-transfer", action="store_true")
    args = parser.parse_args()

    manifest = build_prompt_candidate_population(
        frozen_inputs(),
        output_root=args.output_root,
    )
    print(json.dumps({"candidate_status": manifest["status"], "counts": manifest["counts"]}, indent=2))
    if args.prepare_only:
        return
    prompt_ids = args.prompt_id or list(EXPECTED_PROMPTS)
    for prompt_id in prompt_ids:
        result = asyncio.run(
            score_prompt_variant(
                prompt_id,
                output_root=args.output_root,
                concurrency=args.concurrency,
                retries=args.retries,
                retry_failures=args.retry_failures,
                max_new_calls=args.max_new_calls,
                max_total_reported_cost_usd=OPERATIONAL_SPEND_CEILING_USD,
                confirm_authorized_transfer=args.confirm_authorized_transfer,
            )
        )
        print(
            json.dumps(
                {
                    "prompt_id": prompt_id,
                    "status": result["status"],
                    "counts": result["counts"],
                    "usage": result["usage"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
