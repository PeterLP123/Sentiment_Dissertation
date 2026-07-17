#!/usr/bin/env python3
"""Run the frozen sentiment-alpha term-structure precision gate."""

from __future__ import annotations

import argparse

from sentiment_benchmark.sentiment_term_structure_pilot import TermStructureConfig, run_term_structure_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", required=True)
    parser.add_argument("--primary-headlines", required=True)
    parser.add_argument("--primary-stock-prices", required=True)
    parser.add_argument("--market-prices", required=True)
    parser.add_argument("--comparison-headlines", required=True)
    parser.add_argument("--pilot2-regressions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--simulation-runs", type=int, default=500)
    parser.add_argument("--simulation-bootstrap-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--evaluation-run-number", type=int, default=1)
    parser.add_argument("--rerun-reason")
    args = parser.parse_args()
    config = TermStructureConfig(
        bootstrap_samples=args.bootstrap_samples,
        simulation_runs=args.simulation_runs,
        simulation_bootstrap_samples=args.simulation_bootstrap_samples,
        random_seed=args.seed,
        evaluation_run_number=args.evaluation_run_number,
        rerun_reason=args.rerun_reason,
    )
    report = run_term_structure_pilot(
        primary_scores_path=args.primary_scores,
        primary_headlines_path=args.primary_headlines,
        primary_stock_prices_path=args.primary_stock_prices,
        market_prices_path=args.market_prices,
        comparison_headlines_path=args.comparison_headlines,
        pilot2_regressions_path=args.pilot2_regressions,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"Term-structure pilot completed: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
