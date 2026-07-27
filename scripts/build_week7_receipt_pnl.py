#!/usr/bin/env python3
"""Build the Week 7 receipt-date P/L view for the current FinBERT strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_benchmark.strategy_research.receipt_pnl import ReceiptPnlError, build_receipt_pnl_report

DEFAULT_RUN_ID = "recent-news-midcap-finbert-event-v1-dfdd88113a5f"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-pnl",
        type=Path,
        default=Path("results/strategy_research/baselines") / DEFAULT_RUN_ID / "daily_pnl.jsonl",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("results/strategy_research/baselines") / DEFAULT_RUN_ID / "manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/week7_pnl") / f"{DEFAULT_RUN_ID}-receipt-aligned",
    )
    parser.add_argument("--scorer", action="append", default=None, help="Scorer to export; repeat for multiple models.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = build_receipt_pnl_report(
            args.daily_pnl,
            args.source_manifest,
            args.output_dir,
            scorers=tuple(args.scorer or ("finbert",)),
        )
    except (OSError, ValueError, ReceiptPnlError) as exc:
        raise SystemExit(f"Week 7 P/L export failed: {exc}") from exc
    print(result.summary_path)
    print(result.portfolio_matrix_path)
    print(result.figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
