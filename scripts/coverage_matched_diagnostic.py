"""Coverage-matched scorer diagnostic for the headline-value pilot.

The pilot scored a ~3.5% random subsample of headlines with an LLM, so the
LLM day-signal averaged ~5 headlines/day against the lexicon's ~145 — an
unfair comparison. This script rebuilds BOTH day-signals from exactly the
scored subset (same headlines, same company-day associations) and backtests
them identically, isolating scorer quality from coverage.

Usage:
    python scripts/coverage_matched_diagnostic.py \
        --scores Data/collections/lseg_us_sector_33_6m/derived/headline_scores_gemma-4-31b_soft.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import correlation, fmean

from sentiment_benchmark.artifact_io import read_json, sha256_text
from sentiment_benchmark.backtest import DailySignal, DecisionPolicyConfig, run_backtest
from sentiment_benchmark.headline_value import (
    _companies_from_config,
    _config_from_manifest,
    _filter_signals_with_price_horizon,
    _loads_jsonl_row,
    _parse_date,
    _parse_timestamp,
    _resolve_raw_dir,
    _trading_summary_rows,
    classify_headline,
    normalize_headline,
)
from sentiment_benchmark.strategies import get as get_strategy
from sentiment_benchmark.strategy_sweep import load_prices_csv


def load_scores(path: Path) -> tuple[str, dict[str, float]]:
    model_ids: set[str] = set()
    scores: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "success" or not row.get("score"):
                continue
            scores[str(row["headline_sha256"])] = float(row["score"])
            model_ids.add(str(row["model_id"]))
    if len(model_ids) != 1:
        raise SystemExit(f"expected one model in {path}, found {sorted(model_ids)}")
    return model_ids.pop(), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, default=Path("Data/collections/lseg_us_sector_33_6m"))
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=Path("Data/derived/prices/lseg_us_sector_33.csv"))
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    model_id, llm_scores = load_scores(args.scores)
    horizons = tuple(int(part) for part in args.horizons.split(","))

    raw_dir = _resolve_raw_dir(args.collection_root)
    config = _config_from_manifest(read_json(raw_dir / "manifest.json"))
    companies = _companies_from_config(config)
    symbols = {company.symbol for company in companies}
    start_date = _parse_date(config["collection"]["start"])
    end_date = _parse_date(config["collection"]["end"])

    # Aggregate BOTH scorers over exactly the scored associations.
    day_aggs: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"llm": [0.0, 0.0], "lex": [0.0, 0.0]}
    )
    with (raw_dir / "headlines.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _loads_jsonl_row(line, raw_dir / "headlines.jsonl", line_number)
            headline = str(row.get("headline") or "")
            normalized = normalize_headline(headline)
            if not normalized:
                continue
            llm_score = llm_scores.get(sha256_text(normalized))
            if llm_score is None:
                continue
            matched = tuple(symbol for symbol in (row.get("matched_symbols") or ()) if symbol in symbols)
            if not matched:
                continue
            timestamp = _parse_timestamp(row.get("version_created") or row.get("first_created"))
            if timestamp is None or not (start_date <= timestamp.date() < end_date):
                continue
            lex_score = classify_headline(headline).sentiment_score
            for symbol in matched:
                slots = day_aggs[(symbol, timestamp.date().isoformat())]
                slots["llm"][0] += llm_score
                slots["llm"][1] += 1
                slots["lex"][0] += lex_score
                slots["lex"][1] += 1

    signals: list[DailySignal] = []
    llm_means: list[float] = []
    lex_means: list[float] = []
    for (symbol, news_date), slots in day_aggs.items():
        for scorer_id, key in ((f"llm/{model_id}", "llm"), ("lexicon/matched_subset", "lex")):
            total, count = slots[key]
            mean_score = total / count
            signal_value = 1 if mean_score > 0 else -1 if mean_score < 0 else 0
            signals.append(
                DailySignal(
                    symbol=symbol,
                    news_date=news_date,
                    scorer_id=scorer_id,
                    article_count=int(count),
                    valid_count=int(count),
                    mean_score=mean_score,
                    signal={1: "positive", -1: "negative", 0: "neutral"}[signal_value],
                    signal_value=signal_value,
                    availability_timestamp=None,
                )
            )
        llm_means.append(slots["llm"][0] / slots["llm"][1])
        lex_means.append(slots["lex"][0] / slots["lex"][1])

    print(f"matched company-days: {len(day_aggs):,} | scored associations/day: {fmean(s['llm'][1] for s in day_aggs.values()):.1f}")
    print(f"day-mean correlation (llm vs lexicon, same headlines): {correlation(llm_means, lex_means):.3f}")

    price_rows = load_prices_csv(args.prices)
    strategy = get_strategy("headline_sentiment_threshold_v1")
    policy = DecisionPolicyConfig(
        min_valid_stories=1,
        threshold=0.0,
        transaction_cost_bps_per_side=args.cost_bps,
        policy_version=strategy.id,
    )
    signals, skipped = _filter_signals_with_price_horizon(signals, price_rows, max_horizon=max(horizons))
    if skipped:
        print(f"skipped {skipped} signals lacking a complete price horizon")
    result = run_backtest(
        signals,
        price_rows,
        policy,
        horizons=horizons,
        notional_usd=10_000.0,
        timezone="America/New_York",
        use_decision_policy=True,
        decision_fn=strategy.make_policy().decide,
    )
    summary_rows = _trading_summary_rows(result.returns)
    print(f"\n{'scorer':<28} {'h':>3} {'n':>6} {'gross%':>8} {'net%':>8} {'hit':>6}")
    for row in sorted(summary_rows, key=lambda r: (str(r['scorer_id']), int(r['horizon']))):
        print(
            f"{row['scorer_id']:<28} {row['horizon']:>3} {row['traded_rows']:>6} "
            f"{float(row['mean_gross_return_pct']):>+8.3f} {float(row['mean_net_return_pct']):>+8.3f} "
            f"{float(row['net_hit_rate']):>6.1%}"
        )

    output_dir = args.output_dir or args.collection_root / "derived" / "headline_value_llm_pilot" / "coverage_matched"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nwritten: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
