#!/usr/bin/env python3
"""Build the frozen historical-strategy adjusted-open panel."""

from __future__ import annotations

import argparse

from sentiment_benchmark.strategy_research.price_artifacts import build_strategy_price_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-panel", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--expected-symbol-count", type=int, default=33)
    args = parser.parse_args()
    result = build_strategy_price_artifact(
        args.source_panel,
        args.source_manifest,
        args.output_panel,
        args.output_manifest,
        expected_symbol_count=args.expected_symbol_count,
    )
    print(
        f"Built {result.row_count} rows for {result.symbol_count} symbols across "
        f"{result.session_count} XNYS sessions: {result.panel_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

