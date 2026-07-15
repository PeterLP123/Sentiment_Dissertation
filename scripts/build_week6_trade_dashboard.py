#!/usr/bin/env python3
"""Build a local, audited Week 6 trade explorer from frozen run artifacts.

The generated dashboard embeds licensed LSEG headline text and must remain a
local, ignored artifact.  Every displayed signal is reconstructed from the raw
headline collection and checked against the frozen daily signal before output.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_benchmark.headline_value import (
    _parse_timestamp,
    classify_headline,
    normalize_headline,
)

DATA_MARKER = "__WEEK6_DATA_B64__"
FRAGMENT_MARKER = "__WEEK6_FRAGMENT_HTML__"
RUN_META_MARKER = "__WEEK6_RUN_META__"
MAX_FRAGMENT_BYTES = 2_000_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPOSITORY_ROOT / "reports" / "week6_trade_explorer_template.html"
DEFAULT_STANDALONE_TEMPLATE = REPOSITORY_ROOT / "reports" / "week6_trade_explorer_standalone.html"


@dataclass
class HeadlineContribution:
    symbol: str
    news_date: str
    norm_hash: str
    headline: str
    score: float
    raw_count: int = 0
    first_timestamp: str = ""
    sources: set[str] = field(default_factory=set)
    story_families: set[str] = field(default_factory=set)

    def update(self, *, timestamp: str, source: str, story_id: str) -> None:
        self.raw_count += 1
        if timestamp and (not self.first_timestamp or timestamp < self.first_timestamp):
            self.first_timestamp = timestamp
        if source:
            self.sources.add(source)
        family = re.sub(r":\d+$", "", story_id)
        if family:
            self.story_families.add(family)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    return frame


def _signal_references(stock_daily: pd.DataFrame) -> set[tuple[str, str]]:
    signal_date = stock_daily["signal_date"].fillna("").astype(str)
    relevant = (_truthy(stock_daily["active_trade"]) | _truthy(stock_daily["trade_executed"])) & signal_date.ne("")
    return set(stock_daily.loc[relevant, ["symbol", "signal_date"]].drop_duplicates().itertuples(index=False, name=None))


def _aliases(raw_manifest: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    companies = raw_manifest.get("config", {}).get("companies", [])
    result: dict[str, tuple[str, ...]] = {}
    for company in companies:
        symbol = str(company.get("symbol") or "")
        if not symbol:
            continue
        aliases = tuple(str(value) for value in company.get("aliases", []) if str(value))
        ric = str(company.get("ric") or "")
        result[symbol] = aliases + ((ric,) if ric else ())
    return result


def _collect_contributions(
    raw_dir: Path,
    references: set[tuple[str, str]],
    score_lookup: dict[str, float] | None = None,
) -> tuple[dict[tuple[str, str, str], HeadlineContribution], dict[tuple[str, str], int]]:
    raw_manifest = _read_json(raw_dir / "manifest.json")
    aliases = _aliases(raw_manifest)
    contributions: dict[tuple[str, str, str], HeadlineContribution] = {}
    association_counts: dict[tuple[str, str], int] = defaultdict(int)

    with (raw_dir / "headlines.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on headline line {line_number}") from exc
            timestamp = _parse_timestamp(row.get("version_created") or row.get("first_created"))
            if timestamp is None:
                continue
            news_date = timestamp.date().isoformat()
            matched = tuple(dict.fromkeys(str(value) for value in (row.get("matched_symbols") or ())))
            headline = str(row.get("headline") or "")
            normalized = normalize_headline(headline)
            full_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""
            norm_hash = full_hash[:12] if full_hash else "blank"
            timestamp_text = timestamp.isoformat()
            source = str(row.get("source_code") or "unknown")
            story_id = str(row.get("story_id") or "")

            for symbol in matched:
                ref = (symbol, news_date)
                if ref not in references:
                    continue
                association_counts[ref] += 1
                if score_lookup is None:
                    score = classify_headline(
                        headline,
                        aliases=aliases.get(symbol, ()),
                        matched_symbol_count=len(matched),
                        duplicate_count=1,
                    ).sentiment_score
                else:
                    if full_hash not in score_lookup:
                        continue
                    score = score_lookup[full_hash]
                key = (symbol, news_date, norm_hash)
                contribution = contributions.get(key)
                if contribution is None:
                    contribution = HeadlineContribution(
                        symbol=symbol,
                        news_date=news_date,
                        norm_hash=norm_hash,
                        headline=headline,
                        score=float(score),
                    )
                    contributions[key] = contribution
                elif not math.isclose(contribution.score, float(score), abs_tol=1e-15):
                    raise ValueError(f"inconsistent score for normalized headline {key}")
                contribution.update(timestamp=timestamp_text, source=source, story_id=story_id)
    return contributions, dict(association_counts)


def _decision_payload(
    contributions: dict[tuple[str, str, str], HeadlineContribution],
    association_counts: dict[tuple[str, str], int],
    signals: pd.DataFrame,
    references: set[tuple[str, str]],
    *,
    scorer_id: str,
    nonzero_examples: int | None,
    neutral_examples: int,
) -> dict[str, Any]:
    selected = signals.loc[signals["scorer_id"].astype(str) == scorer_id].copy()
    selected["news_date"] = selected["news_date"].astype(str)
    selected = selected.set_index(["symbol", "news_date"], verify_integrity=True)
    by_reference: dict[tuple[str, str], list[HeadlineContribution]] = defaultdict(list)
    for contribution in contributions.values():
        by_reference[(contribution.symbol, contribution.news_date)].append(contribution)

    output: dict[str, Any] = {}
    for reference in sorted(references):
        if reference not in selected.index:
            raise ValueError(f"missing frozen signal for {reference}")
        frozen = selected.loc[reference]
        expected_count = int(frozen["article_count"])
        expected_valid_count = int(frozen["valid_count"])
        actual_count = int(association_counts.get(reference, 0))
        if actual_count != expected_count:
            raise ValueError(f"headline count mismatch for {reference}: reconstructed {actual_count}, frozen {expected_count}")

        items = by_reference.get(reference, [])
        actual_valid_count = sum(item.raw_count for item in items)
        if actual_valid_count != expected_valid_count:
            raise ValueError(
                f"valid headline count mismatch for {reference}: "
                f"reconstructed {actual_valid_count}, frozen {expected_valid_count}"
            )
        score_sum = sum(item.score * item.raw_count for item in items)
        reconstructed = score_sum / actual_valid_count if actual_valid_count else float("nan")
        recorded = float(frozen["mean_score"])
        both_missing = actual_count == 0 and math.isnan(reconstructed) and math.isnan(recorded)
        if not both_missing and not math.isclose(reconstructed, recorded, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"mean score mismatch for {reference}: reconstructed {reconstructed}, frozen {recorded}")

        positive = sum(item.raw_count for item in items if item.score > 0)
        negative = sum(item.raw_count for item in items if item.score < 0)
        zero = actual_valid_count - positive - negative
        unscored = actual_count - actual_valid_count
        nonzero = sorted(
            (item for item in items if item.score != 0),
            key=lambda item: (-abs(item.score * item.raw_count), item.first_timestamp, item.norm_hash),
        )
        neutral = sorted(
            (item for item in items if item.score == 0),
            key=lambda item: (-item.raw_count, item.first_timestamp, item.norm_hash),
        )
        retained_nonzero = nonzero if nonzero_examples is None else nonzero[:nonzero_examples]
        retained = [*retained_nonzero, *neutral[:neutral_examples]]
        headline_rows = []
        for item in retained:
            families = sorted(item.story_families)
            headline_rows.append(
                [
                    item.norm_hash,
                    item.headline,
                    round(item.score, 8),
                    item.raw_count,
                    item.first_timestamp,
                    "|".join(sorted(item.sources)),
                    len(families),
                    families[0] if families else "",
                ]
            )
        key = f"{reference[0]}|{reference[1]}"
        output[key] = {
            "n": actual_valid_count,
            "raw_n": actual_count,
            "x": unscored,
            "u": len(items),
            "p": positive,
            "m": negative,
            "z": zero,
            "mu": None if math.isnan(recorded) else round(recorded, 12),
            "sum": round(score_sum, 12),
            "ze": min(neutral_examples, len(neutral)),
            "zu": len(neutral),
            "h": headline_rows,
        }
    return output


def _portfolio_metrics(path: Path, *, scorer_id: str) -> list[dict[str, Any]]:
    frame = _required(
        path,
        {
            "portfolio",
            "period",
            "return_variant",
            "actual_sample_percentage_return",
            "annualized_sharpe",
            "maximum_drawdown",
        },
    )
    if "scorer_id" in frame.columns:
        frame = frame.loc[frame["scorer_id"].astype(str) == scorer_id]
    frame = frame.loc[frame["return_variant"].astype(str) == "net"]
    fields = [
        "portfolio",
        "period",
        "stock_count",
        "sample_total_profit",
        "actual_sample_percentage_return",
        "annualized_sharpe",
        "maximum_drawdown",
        "turnover",
        "active_days",
        "worst_daily_return_percentage",
    ]
    return frame[fields].where(pd.notna(frame[fields]), None).to_dict("records")


def _model_score_lookup(path: Path, scorer_id: str) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = set(pd.read_csv(path, nrows=0).columns)
    model_column = "baseline" if "baseline" in columns else "model_id"
    if model_column not in columns:
        raise ValueError(f"{path} lacks baseline/model_id")
    required = {"headline_sha256", "score", "status", model_column}
    missing = required - columns
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=sorted(required))
    model_name = scorer_id.removeprefix("llm/")
    selected = frame.loc[
        frame["status"].astype(str).eq("success") & frame[model_column].astype(str).eq(model_name)
    ].copy()
    if selected.empty:
        raise ValueError(f"no successful {model_name!r} scores in {path}")
    if selected["headline_sha256"].duplicated().any():
        raise ValueError(f"duplicated successful {model_name!r} headline hashes in {path}")
    return dict(zip(selected["headline_sha256"].astype(str), selected["score"].astype(float), strict=True))


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    model_run = (run_dir / "model_stock_daily_pnl.csv").is_file()
    stock_name = "model_stock_daily_pnl.csv" if model_run else "stock_daily_pnl.csv"
    stock_daily = _required(
        run_dir / stock_name,
        {
            "date",
            "symbol",
            "signal_date",
            "signal",
            "position",
            "previous_position",
            "gross_pnl",
            "transaction_cost",
            "net_pnl",
            "split",
            "active_trade",
            "trade_executed",
            "timing_attrition_status",
        },
    )
    if "scorer_id" in stock_daily.columns:
        stock_daily = stock_daily.loc[stock_daily["scorer_id"].astype(str) == args.scorer_id].copy()
    if stock_daily.empty:
        raise ValueError(f"no stock P&L rows for scorer {args.scorer_id!r}")
    signals = _required(
        args.signals.resolve(),
        {"symbol", "news_date", "scorer_id", "article_count", "valid_count", "mean_score"},
    )
    prices = _required(args.prices.resolve(), {"symbol", "session_date", "close"})
    references = _signal_references(stock_daily)
    score_lookup = _model_score_lookup(args.score_file.resolve(), args.scorer_id) if args.score_file else None
    if args.scorer_id != "headline/sentiment_all" and score_lookup is None:
        raise ValueError("non-lexicon viewers require --score-file for audited headline attribution")
    contributions, association_counts = _collect_contributions(
        args.raw_dir.resolve(),
        references,
        score_lookup,
    )
    decisions = _decision_payload(
        contributions,
        association_counts,
        signals,
        references,
        scorer_id=args.scorer_id,
        nonzero_examples=args.nonzero_examples,
        neutral_examples=args.neutral_examples,
    )

    symbols = sorted(stock_daily["symbol"].astype(str).unique())
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    dates = sorted(set(prices["session_date"].astype(str)) | set(stock_daily["date"].astype(str)))
    date_index = {value: index for index, value in enumerate(dates)}
    status_values = sorted(stock_daily["timing_attrition_status"].fillna("").astype(str).unique())
    status_index = {value: index for index, value in enumerate(status_values)}

    price_rows = [
        [symbol_index[str(row.symbol)], date_index[str(row.session_date)], round(float(row.close), 6)]
        for row in prices.itertuples(index=False)
    ]
    daily_rows = []
    for row in stock_daily.itertuples(index=False):
        signal_date = "" if pd.isna(row.signal_date) else str(row.signal_date)
        signal = None if pd.isna(row.signal) or row.signal == "" else round(float(row.signal), 12)
        daily_rows.append(
            [
                symbol_index[str(row.symbol)],
                date_index[str(row.date)],
                signal_date,
                signal,
                int(row.position),
                int(row.previous_position),
                round(float(row.gross_pnl), 6),
                round(float(row.transaction_cost), 6),
                round(float(row.net_pnl), 6),
                0 if str(row.split) == "development" else 1,
                int(str(row.active_trade).lower() == "true"),
                int(str(row.trade_executed).lower() == "true"),
                status_index[str(row.timing_attrition_status)],
            ]
        )

    run_manifest = _read_json(run_dir / "manifest.json")
    grid_name = "model_strategy_grid.csv" if model_run else "strategy_grid.csv"
    strategy_grid = _required(
        run_dir / grid_name,
        {"threshold", "holding_period", "selected", "selection_reason"},
    )
    if "scorer_id" in strategy_grid.columns:
        strategy_grid = strategy_grid.loc[strategy_grid["scorer_id"].astype(str) == args.scorer_id].copy()
    selected_candidates = strategy_grid.loc[_truthy(strategy_grid["selected"])]
    if len(selected_candidates) != 1:
        raise ValueError(f"expected exactly one selected strategy candidate, found {len(selected_candidates)}")
    selected_candidate = selected_candidates.iloc[0]
    if model_run:
        rule_rows = _required(
            run_dir / "selected_model_rules.csv",
            {"scorer_id", "selected_absolute_threshold", "selected_holding_period"},
        )
        rule_rows = rule_rows.loc[rule_rows["scorer_id"].astype(str) == args.scorer_id]
        if len(rule_rows) != 1:
            raise ValueError(f"expected one selected rule for {args.scorer_id!r}, found {len(rule_rows)}")
        selected_rule = rule_rows.iloc[0]
        selected_stocks = _required(
            run_dir / "selected_negative_portfolio_weights.csv",
            {"symbol", "scorer_id"},
        )
        selected_stocks = selected_stocks.loc[
            selected_stocks["scorer_id"].astype(str) == args.scorer_id, "symbol"
        ].astype(str).tolist()
        threshold = float(selected_rule["selected_absolute_threshold"])
        holding = int(selected_rule["selected_holding_period"])
        portfolio_name = "model_portfolio_performance.csv"
    else:
        selected_stocks = pd.read_csv(run_dir / "selected_low_correlation_stocks.csv")["symbol"].astype(str).tolist()
        threshold = float(run_manifest["selected_rule"]["threshold"])
        holding = int(run_manifest["selected_rule"]["holding_period"])
        portfolio_name = "portfolio_performance.csv"
    return {
        "symbols": symbols,
        "dates": dates,
        "statuses": status_values,
        "prices": price_rows,
        "daily": daily_rows,
        "decisions": decisions,
        "portfolio": _portfolio_metrics(run_dir / portfolio_name, scorer_id=args.scorer_id),
        "rule": {
            "threshold": threshold,
            "holding": holding,
            "capital": run_manifest["configuration"]["starting_capital"],
            "allocation": run_manifest["configuration"]["starting_capital"] / len(symbols),
            "cost_bps": run_manifest["configuration"]["transaction_cost_bps_per_side"],
            "split_date": run_manifest["development_evaluation"]["split_date"],
            "selected_stocks": selected_stocks,
            "scorer": args.scorer_id,
            "candidate_thresholds": sorted(strategy_grid["threshold"].astype(float).unique().tolist()),
            "candidate_holds": sorted(strategy_grid["holding_period"].astype(int).unique().tolist()),
            "selection_reason": str(selected_candidate["selection_reason"]),
        },
        "audit": {
            "signal_references": len(references),
            "raw_associations": sum(association_counts.values()),
            "retained_headline_rows": sum(len(item["h"]) for item in decisions.values()),
            "neutral_text_policy": (
                f"{'all' if args.nonzero_examples is None else args.nonzero_examples} highest-impact non-zero contributors "
                f"plus {args.neutral_examples} representative zero-score headlines per signal"
            ),
            "validated": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--scorer-id", default="headline/sentiment_all")
    parser.add_argument(
        "--score-file",
        type=Path,
        help="Model/baseline score CSV used to audit non-lexicon headline contributions.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"dashboard fragment template (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--standalone-output",
        type=Path,
        help="optional browser-ready HTML output; licensed headline text remains embedded locally",
    )
    parser.add_argument("--neutral-examples", type=int, default=5)
    parser.add_argument(
        "--nonzero-examples",
        type=int,
        help="Retain at most this many highest-impact non-zero headlines per signal (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.neutral_examples < 0:
        raise ValueError("--neutral-examples must be non-negative")
    if args.nonzero_examples is not None and args.nonzero_examples < 1:
        raise ValueError("--nonzero-examples must be positive")
    template = args.template.read_text(encoding="utf-8")
    payload = build_payload(args)
    compact_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(compact_json, 9, mtime=0)).decode("ascii")
    if template.count(DATA_MARKER) == 1:
        output_text = template.replace(DATA_MARKER, encoded)
    elif DATA_MARKER in template:
        raise ValueError(f"template must contain at most one {DATA_MARKER} marker")
    else:
        output_text, replacements = re.subn(
            r'(?<=const DATA_B64 = ")[A-Za-z0-9+/=]+(?=";)',
            encoded,
            template,
            count=1,
        )
        if replacements != 1:
            raise ValueError(f"template lacks both {DATA_MARKER} and an existing embedded payload")
    output_bytes = output_text.encode("utf-8")
    if len(output_bytes) >= MAX_FRAGMENT_BYTES:
        raise ValueError(f"dashboard fragment is {len(output_bytes):,} bytes; must remain below {MAX_FRAGMENT_BYTES:,}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8", newline="\n")
    standalone_path = None
    if args.standalone_output is not None:
        standalone_template = DEFAULT_STANDALONE_TEMPLATE.read_text(encoding="utf-8")
        if standalone_template.count(FRAGMENT_MARKER) != 1 or standalone_template.count(RUN_META_MARKER) != 1:
            raise ValueError("standalone template must contain exactly one fragment marker and one run-meta marker")
        execution_count = sum(row[11] for row in payload["daily"])
        run_meta = f"{payload['rule']['scorer']} · {len(payload['symbols'])} stocks · {execution_count} executions"
        standalone = standalone_template.replace(FRAGMENT_MARKER, html.escape(output_text, quote=True))
        standalone = standalone.replace(RUN_META_MARKER, html.escape(run_meta))
        args.standalone_output.parent.mkdir(parents=True, exist_ok=True)
        args.standalone_output.write_text(standalone, encoding="utf-8", newline="\n")
        standalone_path = str(args.standalone_output.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "standalone_output": standalone_path,
                "bytes": len(output_bytes),
                **payload["audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
