#!/usr/bin/env python3
"""Run the frozen FNSPID VADER scale smoke experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentiment_benchmark.fnspid_vader_smoke import load_config, run_fnspid_vader_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-archive", required=True, type=Path, action="append")
    parser.add_argument("--nasdaq-archive", required=True, type=Path)
    parser.add_argument("--price-archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run_fnspid_vader_smoke(
        news_archives=tuple(args.news_archive),
        nasdaq_archive=args.nasdaq_archive,
        price_archive=args.price_archive,
        output_dir=args.output_dir,
        config=load_config(args.config),
        command=tuple(sys.argv),
    )
    print(report)


if __name__ == "__main__":
    main()
