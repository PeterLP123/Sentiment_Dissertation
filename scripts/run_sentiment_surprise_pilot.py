#!/usr/bin/env python3
"""Run the frozen sentiment-surprise horse race from existing derived inputs."""

from __future__ import annotations

import argparse

from sentiment_benchmark.sentiment_surprise_pilot import PilotConfig, run_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--stock-prices", required=True)
    parser.add_argument("--market-prices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    config = PilotConfig(bootstrap_samples=args.bootstrap_samples, random_seed=args.seed)
    report = run_pilot(
        args.scores,
        args.stock_prices,
        args.market_prices,
        args.output_dir,
        config,
    )
    print(f"Pilot completed: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
