#!/usr/bin/env python3
"""Score the frozen LSEG no-newness structured reaction prompt."""

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
    build_prompt_candidate_population,
)
from final_experiments.lib.lseg_structured_reaction import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    score_structured_reaction,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--end-entry-session-exclusive")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--confirm-authorized-transfer", action="store_true")
    args = parser.parse_args()

    population = build_prompt_candidate_population(frozen_inputs())
    print(json.dumps({"candidate_status": population["status"], "counts": population["counts"]}, indent=2))
    result = asyncio.run(
        score_structured_reaction(
            output_root=args.output_root,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_failures=args.retry_failures,
            max_new_calls=args.max_new_calls,
            end_entry_session_exclusive=args.end_entry_session_exclusive,
            confirm_authorized_transfer=args.confirm_authorized_transfer,
        )
    )
    print(
        json.dumps(
            {
                "prompt_id": result["identity"]["prompt_id"],
                "status": result["status"],
                "counts": result["counts"],
                "usage": result["usage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
