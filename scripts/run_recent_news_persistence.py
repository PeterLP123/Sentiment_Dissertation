"""Freeze and evaluate the recent-news persistent FinBERT strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentiment_benchmark.strategy_research.artifacts import write_immutable_json
from sentiment_benchmark.strategy_research.persistent_finbert import develop, evaluate, load_experiment_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("develop", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.mode == "develop":
        payload = develop(config)
    else:
        if args.protocol is None:
            parser.error("evaluate requires --protocol")
        payload = evaluate(config, args.protocol)
    path, reused = write_immutable_json(args.output, payload)
    print(json.dumps({"path": path.as_posix(), "reused": reused, "status": payload["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
