"""Run local LSEG headline baselines and the aggregate return study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sentiment_benchmark.headline_return_study import (
    audit_headline_scores,
    run_headline_return_study,
    score_collection_baselines,
    score_collection_vader_compound,
    score_headline_baselines,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baselines = subparsers.add_parser("score-baselines", help="Score the frozen population with local baselines.")
    baseline_source = baselines.add_mutually_exclusive_group(required=True)
    baseline_source.add_argument("--gemma-scores", type=Path)
    baseline_source.add_argument(
        "--collection-root",
        type=Path,
        help="Score every unique in-window company-matched headline directly from the frozen collection.",
    )
    baselines.add_argument("--output", type=Path, required=True)
    baselines.add_argument("--finbert-batch-size", type=int, default=32)
    baselines.add_argument(
        "--finbert-revision",
        help="Exact ProsusAI/finbert commit to pass to Transformers model loading.",
    )

    compound = subparsers.add_parser(
        "score-vader-compound",
        help="Score the frozen population with canonical compound-threshold VADER.",
    )
    compound.add_argument("--collection-root", type=Path, required=True)
    compound.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser("analyze", help="Run the frozen next-open to following-open study.")
    analyze.add_argument("--gemma-scores", type=Path, required=True)
    analyze.add_argument("--baseline-scores", type=Path, required=True)
    analyze.add_argument("--prices", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--development-fraction", type=float, default=0.7)
    analyze.add_argument("--bootstrap-samples", type=int, default=1_000)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--event-records", type=Path)

    audit = subparsers.add_parser("audit-scores", help="Validate score reconciliation without printing text.")
    audit.add_argument("--scores", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--expected-rows", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "score-baselines":
        if args.collection_root is not None:
            summary = score_collection_baselines(
                args.collection_root,
                args.output,
                finbert_batch_size=args.finbert_batch_size,
                finbert_revision=args.finbert_revision,
            )
        else:
            summary = score_headline_baselines(
                args.gemma_scores,
                args.output,
                finbert_batch_size=args.finbert_batch_size,
                finbert_revision=args.finbert_revision,
            )
    elif args.command == "score-vader-compound":
        summary = score_collection_vader_compound(args.collection_root, args.output)
    elif args.command == "analyze":
        summary = run_headline_return_study(
            args.gemma_scores,
            args.baseline_scores,
            args.prices,
            args.output_dir,
            development_fraction=args.development_fraction,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            event_records_path=args.event_records,
        )
    else:
        summary = audit_headline_scores(args.scores, args.output, expected_rows=args.expected_rows)
    print(json.dumps({key: str(value) for key, value in asdict(summary).items()}, indent=2))


if __name__ == "__main__":
    main()
