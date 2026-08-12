#!/usr/bin/env python3
"""Freeze the return-blind 2,000-event LSEG materiality pilot sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.joined_lseg import JoinedLsegInputs  # noqa: E402
from final_experiments.lib.lseg_materiality import (  # noqa: E402
    MaterialityInputs,
    prepare_materiality_sample,
)

SNAPSHOT = "snapshot_20260811_collection_complete"
BACKWARD_DERIVED = Path(
    "Data/collections/lseg_us_sector_33_backward_20240101_20251026_headlines/derived"
)


def frozen_inputs() -> MaterialityInputs:
    return MaterialityInputs(
        joined_scores=JoinedLsegInputs(
            backward_finbert=BACKWARD_DERIVED
            / "incremental_finbert"
            / SNAPSHOT
            / "headline_scores_finbert4556_cacheonly.csv",
            backward_gemma_seed=BACKWARD_DERIVED
            / "incremental_gemma"
            / SNAPSHOT
            / "headline_scores_gemma4_26b_a4b_it_deepinfra_fp8_investor_headline_soft_label_v1_resume_seed.csv",
            backward_gemma_delta=BACKWARD_DERIVED
            / "incremental_gemma"
            / SNAPSHOT
            / "headline_scores_gemma4_26b_a4b_it_deepinfra_fp8_investor_headline_soft_label_v1_delta.csv",
            recent_finbert=Path(
                "Data/collections/lseg_us_sector_44_8m_headlines/derived/"
                "headline_scores_finbert4556_cacheonly_20260803.csv"
            ),
            recent_gemma=Path(
                "Data/collections/lseg_us_sector_44_8m_headlines/derived/"
                "headline_scores_gemma4_26b_a4b_it_deepinfra_fp8_investor_headline_soft_label_v1.csv"
            ),
        ),
        backward_headlines=BACKWARD_DERIVED / "incremental_finbert" / SNAPSHOT / "headlines.jsonl",
        recent_headlines=Path(
            "Data/collections/lseg_us_sector_44_8m_headlines/derived/merged/headlines.jsonl"
        ),
        recent_articles=Path(
            "Data/collections/lseg_us_sector_33_6m/derived/us_sector_33_6m/articles.jsonl"
        ),
        company_config=Path("configs/lseg_us_sector_33_6m.toml"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="final_experiments/outputs/68_lseg_materiality_measurement_pilot",
    )
    args = parser.parse_args()
    manifest = prepare_materiality_sample(frozen_inputs(), output_root=args.output_root)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
