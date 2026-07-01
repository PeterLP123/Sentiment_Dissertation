from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from .artifact_io import atomic_write_json, atomic_write_text, read_json, sha256_file, sha256_text
from .backtest import DailySignal, DecisionPolicyConfig, ReturnRow, run_backtest
from .lseg_source import utc_now
from .runtime_metadata import collect_run_environment
from .strategies import get as get_strategy
from .strategy_sweep import load_prices_csv

HEADLINE_VALUE_SCHEMA_VERSION = 1

EVENT_TYPES = (
    "earnings_guidance",
    "analyst_rating",
    "mna_strategy",
    "legal_regulatory",
    "product_technology",
    "management_governance",
    "macro_sector",
    "commodity_rates_fx",
    "labor_esg",
    "market_price_technical",
    "generic_low_information",
    "other",
)

TRADABILITY_CLASSES = (
    "single_company_actionable",
    "company_relevant_context",
    "sector_macro_context",
    "duplicate_syndicated",
    "off_target_or_ambiguous",
    "low_information",
)

SENTIMENT_CLARITY = ("clear_positive", "clear_negative", "mixed", "neutral_factual", "unclear")
EXPECTED_HORIZONS = ("intraday", "1d", "5d", "10d_plus", "not_applicable")

SOURCE_REUTERS = "Reuters"
SOURCE_NON_REUTERS = "Non-Reuters"
SOURCE_ALL = "All sources"

SIGNAL_SCORERS = (
    "headline/sentiment_all",
    "headline/sentiment_non_reuters",
    "headline/sentiment_reuters",
    "headline/actionable_sentiment_all",
)


class HeadlineValueError(RuntimeError):
    """Raised when headline value analysis inputs or outputs are invalid."""


@dataclass(frozen=True)
class HeadlineValueResult:
    output_dir: Path
    manifest_path: Path
    summary_path: Path
    generated_files: tuple[Path, ...]
    trading_status: str


@dataclass(frozen=True)
class CompanyConfig:
    symbol: str
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Classification:
    event_type: str
    sentiment_clarity: str
    expected_horizon: str
    sentiment_score: float
    alias_match: bool


@dataclass
class DayAggregate:
    headline_count: int = 0
    unique_norms: set[str] = field(default_factory=set)
    sources: Counter[str] = field(default_factory=Counter)
    reuters_count: int = 0
    non_reuters_count: int = 0
    duplicate_association_count: int = 0
    sentiment_sum: float = 0.0
    sentiment_count: int = 0
    reuters_sentiment_sum: float = 0.0
    reuters_sentiment_count: int = 0
    non_reuters_sentiment_sum: float = 0.0
    non_reuters_sentiment_count: int = 0
    actionable_sentiment_sum: float = 0.0
    actionable_sentiment_count: int = 0
    event_type_counts: Counter[str] = field(default_factory=Counter)
    tradability_counts: Counter[str] = field(default_factory=Counter)
    clarity_counts: Counter[str] = field(default_factory=Counter)
    horizon_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class EventAggregate:
    symbol: str
    norm_hash: str
    first_seen: str
    last_seen: str
    raw_row_count: int = 0
    sources: Counter[str] = field(default_factory=Counter)
    story_families: set[str] = field(default_factory=set)
    event_type: str = "other"
    tradability_class: str = "company_relevant_context"
    sentiment_clarity: str = "neutral_factual"
    expected_horizon: str = "not_applicable"
    sentiment_score_sum: float = 0.0
    sentiment_score_count: int = 0


@dataclass
class SourceAggregate:
    headline_rows: int = 0
    company_associations: int = 0
    unique_company_headline_pairs: set[tuple[str, str]] = field(default_factory=set)
    company_days: set[tuple[str, date]] = field(default_factory=set)


EVENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "earnings_guidance",
        re.compile(r"\b(earnings?|results?|guidance|forecast|outlook|profit|revenue|sales|eps|quarter|q[1-4]|fy)\b", re.I),
    ),
    (
        "analyst_rating",
        re.compile(r"\b(upgrade|downgrade|initiates?|raises?|cuts?|price target|rating|analyst|brokerage|outperform|underperform)\b", re.I),
    ),
    (
        "mna_strategy",
        re.compile(
            r"\b(acquires?|acquisition|merger|stake|deal|bid|buyout|takeover|spin[- ]?off|strategic review|joint venture|partnership)\b",
            re.I,
        ),
    ),
    (
        "legal_regulatory",
        re.compile(
            r"\b(lawsuit|sues?|settlement|probe|investigation|regulator|court|antitrust|sec|doj|fda|approval|recall|tariffs?|sanctions?)\b",
            re.I,
        ),
    ),
    (
        "product_technology",
        re.compile(r"\b(launch|product|ai|chip|software|cloud|iphone|model|drug|trial|plant|factory|battery|platform|technology)\b", re.I),
    ),
    ("management_governance", re.compile(r"\b(ceo|cfo|chair|board|director|appoints?|resigns?|succession|executive|management)\b", re.I)),
    (
        "macro_sector",
        re.compile(r"\b(sector|industry|market|stocks|wall st|nasdaq|s&p|dow|economy|inflation|fed|rates|jobs|gdp|recession)\b", re.I),
    ),
    ("commodity_rates_fx", re.compile(r"\b(oil|gas|crude|gold|copper|dollar|yield|treasury|commodity|fx|currency)\b", re.I)),
    ("labor_esg", re.compile(r"\b(union|strike|labor|workers|layoffs?|climate|emissions|esg|sustainability)\b", re.I)),
    (
        "market_price_technical",
        re.compile(
            r"\b(shares?|stock|trading|premarket|after-hours|rises?|falls?|jumps?|slumps?|gains?|drops?|price|chart|volume)\b",
            re.I,
        ),
    ),
    (
        "generic_low_information",
        re.compile(r"\b(watch|roundup|brief|digest|morning bid|buzz|what to expect|too late to buy|why .+ today)\b", re.I),
    ),
)

POSITIVE_PATTERN = re.compile(
    r"\b(beat|beats|raises?|raised|win|wins|approved?|record|growth|profit|surges?|jumps?|rises?|gains?|upgrade|"
    r"outperform|strong|expands?|launches?|deal|contract|buyback|dividend|positive)\b",
    re.I,
)
NEGATIVE_PATTERN = re.compile(
    r"\b(miss|misses|cuts?|cut|falls?|drops?|slumps?|sues?|lawsuit|probe|recall|downgrade|underperform|weak|loss|"
    r"layoffs?|strike|antitrust|rejects?|warning|fraud|decline|negative)\b",
    re.I,
)

HORIZON_BY_EVENT = {
    "analyst_rating": "1d",
    "commodity_rates_fx": "1d",
    "earnings_guidance": "1d",
    "generic_low_information": "not_applicable",
    "labor_esg": "5d",
    "legal_regulatory": "5d",
    "macro_sector": "1d",
    "management_governance": "5d",
    "market_price_technical": "intraday",
    "mna_strategy": "10d_plus",
    "other": "not_applicable",
    "product_technology": "5d",
}


def normalize_headline(value: str) -> str:
    """Normalize headline text for exact duplicate suppression."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (value or "").lower())).strip()


def story_family_id(story_id: str) -> str:
    return re.sub(r":\d+$", "", story_id or "")


def classify_headline(
    headline: str,
    *,
    aliases: tuple[str, ...] = (),
    matched_symbol_count: int = 1,
    duplicate_count: int = 1,
) -> Classification:
    text = headline or ""
    normalized = normalize_headline(text)
    event_type = "other"
    for candidate, pattern in EVENT_PATTERNS:
        if pattern.search(text):
            event_type = candidate
            break

    positives = len(POSITIVE_PATTERN.findall(text))
    negatives = len(NEGATIVE_PATTERN.findall(text))
    if positives and negatives:
        clarity = "mixed"
        score = (positives - negatives) / (positives + negatives)
    elif positives:
        clarity = "clear_positive"
        score = min(1.0, positives / 3)
    elif negatives:
        clarity = "clear_negative"
        score = -min(1.0, negatives / 3)
    elif event_type == "generic_low_information" or len(normalized) < 12:
        clarity = "unclear"
        score = 0.0
    else:
        clarity = "neutral_factual"
        score = 0.0

    alias_match = _headline_alias_match(text, aliases)
    expected_horizon = HORIZON_BY_EVENT[event_type]
    if duplicate_count > 1 and event_type == "generic_low_information":
        expected_horizon = "not_applicable"
    if matched_symbol_count > 1 and event_type == "other" and not alias_match:
        expected_horizon = "not_applicable"
    return Classification(event_type, clarity, expected_horizon, score, alias_match)


def classify_tradability(
    classification: Classification,
    *,
    duplicate_count: int,
    matched_symbol_count: int,
) -> str:
    if duplicate_count > 1:
        return "duplicate_syndicated"
    if classification.event_type == "generic_low_information" or classification.sentiment_clarity == "unclear":
        return "low_information"
    if matched_symbol_count > 1 and not classification.alias_match:
        return "off_target_or_ambiguous"
    if classification.event_type in {"macro_sector", "commodity_rates_fx"}:
        return "sector_macro_context"
    if classification.alias_match or matched_symbol_count == 1:
        return "single_company_actionable"
    return "company_relevant_context"


def analyze_headline_value(
    collection_root: str | Path,
    *,
    prices: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_size: int = 2_000,
    seed: int = 42,
    timezone: str = "America/New_York",
    horizons: tuple[int, ...] = (1, 5, 10),
    transaction_cost_bps_per_side: float = 10.0,
    notional_usd: float = 10_000.0,
    overwrite: bool = False,
) -> HeadlineValueResult:
    root = Path(collection_root)
    raw_dir = _resolve_raw_dir(root)
    manifest_path = raw_dir / "manifest.json"
    headlines_path = raw_dir / "headlines.jsonl"
    if not headlines_path.exists():
        raise HeadlineValueError(f"missing headlines.jsonl: {headlines_path}")
    raw_manifest = read_json(manifest_path)
    config = _config_from_manifest(raw_manifest)
    companies = _companies_from_config(config)
    symbols = tuple(company.symbol for company in companies)
    aliases_by_symbol = {company.symbol: company.aliases for company in companies}
    start_date = _parse_date(config["collection"]["start"])
    end_date = _parse_date(config["collection"]["end"])
    dates = _date_range(start_date, end_date)
    possible_company_days = len(symbols) * len(dates)
    target_output = Path(output_dir) if output_dir is not None else root / "derived" / "headline_value_analysis"
    _prepare_output_dir(target_output, overwrite=overwrite)

    first_pass = _first_pass(headlines_path, symbols=symbols, start_date=start_date, end_date=end_date)
    low_volume_symbols = _low_volume_symbols(first_pass["company_associations"])
    top_sources = {source for source, _ in first_pass["source_counts"].most_common(15)}

    day_aggs: dict[tuple[str, date], DayAggregate] = {(symbol, day): DayAggregate() for symbol in symbols for day in dates}
    event_aggs: dict[tuple[str, str], EventAggregate] = {}
    source_aggs: dict[str, SourceAggregate] = defaultdict(SourceAggregate)
    category_assoc_counts: Counter[tuple[str, str, str]] = Counter()
    sample_candidates: list[dict[str, Any]] = []

    with headlines_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _loads_jsonl_row(line, headlines_path, line_number)
            headline = str(row.get("headline") or "")
            normalized = normalize_headline(headline)
            timestamp = _parse_timestamp(row.get("version_created") or row.get("first_created"))
            source_code = str(row.get("source_code") or "unknown")
            source_scope = SOURCE_REUTERS if source_code == "NS:RTRS" else SOURCE_NON_REUTERS
            matched_symbols = tuple(symbol for symbol in (row.get("matched_symbols") or ()) if symbol in aliases_by_symbol)
            story_family = story_family_id(str(row.get("story_id") or ""))

            if timestamp is not None:
                news_date = timestamp.date()
            else:
                news_date = None
            in_window = news_date is not None and start_date <= news_date < end_date

            if matched_symbols and sample_size > 0:
                sample_candidates.append(
                    _sample_candidate(
                        row=row,
                        headline=headline,
                        normalized=normalized,
                        source_code=source_code,
                        source_scope=source_scope,
                        matched_symbols=matched_symbols,
                        timestamp=timestamp,
                        low_volume_symbols=low_volume_symbols,
                        top_sources=top_sources,
                    )
                )

            source_aggs[source_code].headline_rows += 1
            for symbol in matched_symbols:
                duplicate_count = first_pass["company_norm_counts"][(symbol, normalized)] if normalized else 1
                classification = classify_headline(
                    headline,
                    aliases=aliases_by_symbol[symbol],
                    matched_symbol_count=len(matched_symbols),
                    duplicate_count=duplicate_count,
                )
                tradability_class = classify_tradability(
                    classification,
                    duplicate_count=duplicate_count,
                    matched_symbol_count=len(matched_symbols),
                )
                source_aggs[source_code].company_associations += 1
                if normalized:
                    source_aggs[source_code].unique_company_headline_pairs.add((symbol, normalized))
                if in_window and news_date is not None:
                    source_aggs[source_code].company_days.add((symbol, news_date))
                    _update_day_aggregate(
                        day_aggs[(symbol, news_date)],
                        normalized=normalized,
                        source_code=source_code,
                        source_scope=source_scope,
                        duplicate_count=duplicate_count,
                        classification=classification,
                        tradability_class=tradability_class,
                    )
                if normalized:
                    _update_event_aggregate(
                        event_aggs,
                        symbol=symbol,
                        normalized=normalized,
                        timestamp=timestamp,
                        source_code=source_code,
                        story_family=story_family,
                        classification=classification,
                        tradability_class=tradability_class,
                    )
                for scope in (SOURCE_ALL, source_scope):
                    category_assoc_counts[(scope, "event_type", classification.event_type)] += 1
                    category_assoc_counts[(scope, "tradability_class", tradability_class)] += 1
                    category_assoc_counts[(scope, "sentiment_clarity", classification.sentiment_clarity)] += 1
                    category_assoc_counts[(scope, "expected_horizon", classification.expected_horizon)] += 1

    panel_rows = _build_panel_rows(day_aggs, symbols=symbols, dates=dates)
    event_rows = _build_event_rows(event_aggs)
    source_rows = _build_source_rows(source_aggs, total_rows=first_pass["headline_rows"], possible_company_days=possible_company_days)
    category_rows = _build_category_rows(category_assoc_counts, event_rows)
    sample_rows = _build_sample_rows(sample_candidates, sample_size=sample_size, seed=seed)
    signal_rows = _build_signal_rows(panel_rows)

    company_day_panel_path = _write_csv(target_output / "company_day_panel.csv", panel_rows)
    event_table_path = _write_csv(target_output / "headline_events.csv", event_rows)
    category_summary_path = _write_csv(target_output / "category_summary.csv", category_rows)
    source_summary_path = _write_csv(target_output / "source_summary.csv", source_rows)
    daily_signals_path = _write_csv(target_output / "daily_signals.csv", signal_rows)
    sample_path = _write_csv(target_output / "sample_label_template.csv", sample_rows)

    trading_status, trading_paths, trading_metrics = _run_trading_value(
        signal_rows,
        prices=Path(prices) if prices is not None else None,
        output_dir=target_output,
        horizons=horizons,
        timezone=timezone,
        transaction_cost_bps_per_side=transaction_cost_bps_per_side,
        notional_usd=notional_usd,
    )

    counts = {
        "headline_rows": first_pass["headline_rows"],
        "outside_config_window_rows": first_pass["outside_window_rows"],
        "matched_company_associations": first_pass["matched_company_associations"],
        "unique_company_headline_pairs": len(first_pass["company_norm_counts"]),
        "company_day_panel_rows": len(panel_rows),
        "covered_company_days": sum(1 for row in panel_rows if int(row["headline_count"]) > 0),
        "headline_events": len(event_rows),
        "sample_rows": len(sample_rows),
        "source_count": len(first_pass["source_counts"]),
    }
    generated_files = [
        company_day_panel_path,
        event_table_path,
        category_summary_path,
        source_summary_path,
        daily_signals_path,
        sample_path,
        *trading_paths,
    ]
    summary_path = atomic_write_text(
        target_output / "summary.md",
        _summary_markdown(
            counts=counts,
            category_rows=category_rows,
            source_rows=source_rows,
            trading_status=trading_status,
            trading_metrics=trading_metrics,
            prices_path=Path(prices) if prices is not None else None,
        ),
    )
    generated_files.append(summary_path)
    manifest = {
        "schema_version": HEADLINE_VALUE_SCHEMA_VERSION,
        "status": "completed",
        "generated_at": utc_now(),
        "collection_root": str(root),
        "raw_dir": str(raw_dir),
        "headlines_sha256": sha256_file(headlines_path),
        "config": {
            "start": config["collection"]["start"],
            "end": config["collection"]["end"],
            "timezone": timezone,
            "sample_size": sample_size,
            "seed": seed,
            "horizons": list(horizons),
            "transaction_cost_bps_per_side": transaction_cost_bps_per_side,
            "notional_usd": notional_usd,
        },
        "counts": counts,
        "taxonomy": {
            "event_types": list(EVENT_TYPES),
            "tradability_classes": list(TRADABILITY_CLASSES),
            "sentiment_clarity": list(SENTIMENT_CLARITY),
            "expected_horizons": list(EXPECTED_HORIZONS),
        },
        "trading": trading_metrics,
        "files": {path.name: {"path": path.name, "sha256": sha256_file(path)} for path in generated_files},
        "sharing": {
            "licensed_headline_text": True,
            "redistribute": False,
            "public_outputs_exclude_raw_headlines": True,
            "text_bearing_local_file": sample_path.name,
        },
        "environment": collect_run_environment(),
    }
    manifest_path_out = atomic_write_json(target_output / "manifest.json", manifest)
    generated_files.append(manifest_path_out)
    return HeadlineValueResult(target_output, manifest_path_out, summary_path, tuple(generated_files), trading_status)


def _resolve_raw_dir(root: Path) -> Path:
    if (root / "headlines.jsonl").exists() and (root / "manifest.json").exists():
        return root
    raw_root = root / "raw"
    if raw_root.exists():
        candidates = sorted(path for path in raw_root.iterdir() if path.is_dir() and (path / "headlines.jsonl").exists())
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise HeadlineValueError(f"multiple LSEG raw collections under {raw_root}; pass the raw directory directly: {names}")
    raise HeadlineValueError(f"could not find LSEG raw directory under {root}")


def _config_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise HeadlineValueError("raw manifest is missing config")
    collection = config.get("collection")
    if not isinstance(collection, dict) or not collection.get("start") or not collection.get("end"):
        raise HeadlineValueError("raw manifest config is missing collection.start/end")
    companies = config.get("companies")
    if not isinstance(companies, list) or not companies:
        raise HeadlineValueError("raw manifest config is missing companies")
    return config


def _companies_from_config(config: dict[str, Any]) -> tuple[CompanyConfig, ...]:
    companies: list[CompanyConfig] = []
    for item in config["companies"]:
        if not isinstance(item, dict) or not item.get("symbol"):
            continue
        aliases = tuple(str(alias) for alias in item.get("aliases", []) if str(alias).strip())
        companies.append(CompanyConfig(str(item["symbol"]), str(item.get("name") or item["symbol"]), aliases))
    if not companies:
        raise HeadlineValueError("no valid company definitions found")
    return tuple(companies)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise HeadlineValueError(f"refusing to overwrite non-empty output directory without --overwrite: {path}")
        for child in path.iterdir():
            if child.is_dir():
                raise HeadlineValueError(f"refusing to overwrite directory child: {child}")
            child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).date()


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _date_range(start: date, end_exclusive: date) -> tuple[date, ...]:
    days = (end_exclusive - start).days
    if days <= 0:
        raise HeadlineValueError("collection end must be after start")
    return tuple(start + timedelta(days=offset) for offset in range(days))


def _loads_jsonl_row(line: str, path: Path, line_number: int) -> dict[str, Any]:
    import json

    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise HeadlineValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise HeadlineValueError(f"headline row must be an object at {path}:{line_number}")
    return value


def _first_pass(headlines_path: Path, *, symbols: tuple[str, ...], start_date: date, end_date: date) -> dict[str, Any]:
    allowed = set(symbols)
    source_counts: Counter[str] = Counter()
    company_counts: Counter[str] = Counter()
    company_norm_counts: Counter[tuple[str, str]] = Counter()
    matched_company_associations = 0
    outside_window_rows = 0
    headline_rows = 0
    with headlines_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _loads_jsonl_row(line, headlines_path, line_number)
            headline_rows += 1
            source_code = str(row.get("source_code") or "unknown")
            source_counts[source_code] += 1
            timestamp = _parse_timestamp(row.get("version_created") or row.get("first_created"))
            if timestamp is not None and not (start_date <= timestamp.date() < end_date):
                outside_window_rows += 1
            normalized = normalize_headline(str(row.get("headline") or ""))
            for symbol in (row.get("matched_symbols") or ()):
                if symbol not in allowed:
                    continue
                matched_company_associations += 1
                company_counts[str(symbol)] += 1
                if normalized:
                    company_norm_counts[(str(symbol), normalized)] += 1
    return {
        "headline_rows": headline_rows,
        "outside_window_rows": outside_window_rows,
        "source_counts": source_counts,
        "company_associations": company_counts,
        "company_norm_counts": company_norm_counts,
        "matched_company_associations": matched_company_associations,
    }


def _low_volume_symbols(company_counts: Counter[str]) -> set[str]:
    if not company_counts:
        return set()
    ordered = sorted(company_counts.values())
    cutoff = ordered[max(0, math.floor((len(ordered) - 1) * 0.25))]
    return {symbol for symbol, count in company_counts.items() if count <= cutoff}


def _headline_alias_match(headline: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        if not alias:
            continue
        escaped = re.escape(alias).replace(r"\ ", r"\s+")
        if re.search(rf"(?<!\w){escaped}(?!\w)", headline, re.I):
            return True
    return False


def _sample_candidate(
    *,
    row: dict[str, Any],
    headline: str,
    normalized: str,
    source_code: str,
    source_scope: str,
    matched_symbols: tuple[str, ...],
    timestamp: datetime | None,
    low_volume_symbols: set[str],
    top_sources: set[str],
) -> dict[str, Any]:
    rare_or_low = source_code not in top_sources or any(symbol in low_volume_symbols for symbol in matched_symbols)
    return {
        "sample_id": sha256_text(f"{row.get('story_id')}|{source_code}|{normalized}|{'|'.join(matched_symbols)}")[:16],
        "story_id": str(row.get("story_id") or ""),
        "version_created": timestamp.isoformat() if timestamp else "",
        "source_code": source_code,
        "source_scope": source_scope,
        "matched_symbols": "|".join(matched_symbols),
        "headline": headline,
        "rare_or_low_coverage": rare_or_low,
    }


def _update_day_aggregate(
    aggregate: DayAggregate,
    *,
    normalized: str,
    source_code: str,
    source_scope: str,
    duplicate_count: int,
    classification: Classification,
    tradability_class: str,
) -> None:
    aggregate.headline_count += 1
    if normalized:
        aggregate.unique_norms.add(normalized)
    aggregate.sources[source_code] += 1
    if source_scope == SOURCE_REUTERS:
        aggregate.reuters_count += 1
        aggregate.reuters_sentiment_sum += classification.sentiment_score
        aggregate.reuters_sentiment_count += 1
    else:
        aggregate.non_reuters_count += 1
        aggregate.non_reuters_sentiment_sum += classification.sentiment_score
        aggregate.non_reuters_sentiment_count += 1
    if duplicate_count > 1:
        aggregate.duplicate_association_count += 1
    aggregate.sentiment_sum += classification.sentiment_score
    aggregate.sentiment_count += 1
    if tradability_class == "single_company_actionable":
        aggregate.actionable_sentiment_sum += classification.sentiment_score
        aggregate.actionable_sentiment_count += 1
    aggregate.event_type_counts[classification.event_type] += 1
    aggregate.tradability_counts[tradability_class] += 1
    aggregate.clarity_counts[classification.sentiment_clarity] += 1
    aggregate.horizon_counts[classification.expected_horizon] += 1


def _update_event_aggregate(
    event_aggs: dict[tuple[str, str], EventAggregate],
    *,
    symbol: str,
    normalized: str,
    timestamp: datetime | None,
    source_code: str,
    story_family: str,
    classification: Classification,
    tradability_class: str,
) -> None:
    timestamp_text = timestamp.isoformat() if timestamp else ""
    key = (symbol, normalized)
    existing = event_aggs.get(key)
    if existing is None:
        existing = EventAggregate(
            symbol=symbol,
            norm_hash=sha256_text(normalized),
            first_seen=timestamp_text,
            last_seen=timestamp_text,
            event_type=classification.event_type,
            tradability_class=tradability_class,
            sentiment_clarity=classification.sentiment_clarity,
            expected_horizon=classification.expected_horizon,
        )
        event_aggs[key] = existing
    else:
        if timestamp_text and (not existing.first_seen or timestamp_text < existing.first_seen):
            existing.first_seen = timestamp_text
        if timestamp_text and timestamp_text > existing.last_seen:
            existing.last_seen = timestamp_text
        if existing.tradability_class != "duplicate_syndicated" and tradability_class == "duplicate_syndicated":
            existing.tradability_class = tradability_class
    existing.raw_row_count += 1
    existing.sources[source_code] += 1
    if story_family:
        existing.story_families.add(story_family)
    existing.sentiment_score_sum += classification.sentiment_score
    existing.sentiment_score_count += 1


def _build_panel_rows(
    day_aggs: dict[tuple[str, date], DayAggregate],
    *,
    symbols: tuple[str, ...],
    dates: tuple[date, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for day in dates:
            aggregate = day_aggs[(symbol, day)]
            row: dict[str, Any] = {
                "symbol": symbol,
                "news_date": day.isoformat(),
                "headline_count": aggregate.headline_count,
                "unique_headline_count": len(aggregate.unique_norms),
                "source_count": len(aggregate.sources),
                "reuters_count": aggregate.reuters_count,
                "non_reuters_count": aggregate.non_reuters_count,
                "duplicate_association_count": aggregate.duplicate_association_count,
                "duplicate_rate": _safe_divide(aggregate.duplicate_association_count, aggregate.headline_count),
                "top_source_count": max(aggregate.sources.values()) if aggregate.sources else 0,
                "top_source_share": _safe_divide(max(aggregate.sources.values()) if aggregate.sources else 0, aggregate.headline_count),
                "mean_sentiment_score": _safe_divide(aggregate.sentiment_sum, aggregate.sentiment_count),
                "mean_reuters_sentiment_score": _safe_divide(aggregate.reuters_sentiment_sum, aggregate.reuters_sentiment_count),
                "mean_non_reuters_sentiment_score": _safe_divide(
                    aggregate.non_reuters_sentiment_sum,
                    aggregate.non_reuters_sentiment_count,
                ),
                "mean_actionable_sentiment_score": _safe_divide(aggregate.actionable_sentiment_sum, aggregate.actionable_sentiment_count),
                "actionable_count": aggregate.tradability_counts["single_company_actionable"],
                "actionable_share": _safe_divide(aggregate.tradability_counts["single_company_actionable"], aggregate.headline_count),
                "low_information_count": aggregate.tradability_counts["low_information"],
                "low_information_share": _safe_divide(aggregate.tradability_counts["low_information"], aggregate.headline_count),
            }
            for event_type in EVENT_TYPES:
                row[f"event_type_count__{event_type}"] = aggregate.event_type_counts[event_type]
                row[f"event_type_share__{event_type}"] = _safe_divide(aggregate.event_type_counts[event_type], aggregate.headline_count)
            for tradability_class in TRADABILITY_CLASSES:
                row[f"tradability_count__{tradability_class}"] = aggregate.tradability_counts[tradability_class]
                row[f"tradability_share__{tradability_class}"] = _safe_divide(
                    aggregate.tradability_counts[tradability_class],
                    aggregate.headline_count,
                )
            for clarity in SENTIMENT_CLARITY:
                row[f"sentiment_clarity_count__{clarity}"] = aggregate.clarity_counts[clarity]
                row[f"sentiment_clarity_share__{clarity}"] = _safe_divide(aggregate.clarity_counts[clarity], aggregate.headline_count)
            rows.append(row)
    _add_company_zscores(
        rows,
        fields=(
            "headline_count",
            "unique_headline_count",
            "source_count",
            "reuters_count",
            "non_reuters_count",
            "duplicate_rate",
            "top_source_share",
            "mean_sentiment_score",
            "mean_reuters_sentiment_score",
            "mean_non_reuters_sentiment_score",
            "mean_actionable_sentiment_score",
            "actionable_share",
            "low_information_share",
        ),
    )
    return rows


def _build_event_rows(event_aggs: dict[tuple[str, str], EventAggregate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (symbol, normalized), aggregate in sorted(event_aggs.items(), key=lambda item: (item[1].first_seen, item[0][0], item[1].norm_hash)):
        sources = sorted(aggregate.sources)
        has_reuters = "NS:RTRS" in aggregate.sources
        source_scope = "Mixed" if has_reuters and len(sources) > 1 else SOURCE_REUTERS if has_reuters else SOURCE_NON_REUTERS
        event_id = sha256_text(f"headline_event|{symbol}|{normalized}|{aggregate.first_seen}")[:16]
        rows.append(
            {
                "event_id": event_id,
                "symbol": symbol,
                "normalized_headline_sha256": aggregate.norm_hash,
                "first_seen": aggregate.first_seen,
                "last_seen": aggregate.last_seen,
                "news_date": aggregate.first_seen[:10] if aggregate.first_seen else "",
                "raw_row_count": aggregate.raw_row_count,
                "duplicate_count": max(0, aggregate.raw_row_count - 1),
                "source_scope": source_scope,
                "source_count": len(sources),
                "source_codes": "|".join(sources),
                "story_family_count": len(aggregate.story_families),
                "event_type": aggregate.event_type,
                "tradability_class": aggregate.tradability_class,
                "sentiment_clarity": aggregate.sentiment_clarity,
                "expected_horizon": aggregate.expected_horizon,
                "sentiment_score": _safe_divide(aggregate.sentiment_score_sum, aggregate.sentiment_score_count),
            }
        )
    return rows


def _build_source_rows(source_aggs: dict[str, SourceAggregate], *, total_rows: int, possible_company_days: int) -> list[dict[str, Any]]:
    rows = []
    for source_code, aggregate in sorted(source_aggs.items(), key=lambda item: (-item[1].headline_rows, item[0])):
        rows.append(
            {
                "source_code": source_code,
                "source_scope": SOURCE_REUTERS if source_code == "NS:RTRS" else SOURCE_NON_REUTERS,
                "headline_rows": aggregate.headline_rows,
                "share_of_all_rows": _safe_divide(aggregate.headline_rows, total_rows),
                "company_associations": aggregate.company_associations,
                "unique_company_headline_pairs": len(aggregate.unique_company_headline_pairs),
                "duplicate_association_rate": 1.0
                - _safe_divide(len(aggregate.unique_company_headline_pairs), aggregate.company_associations),
                "covered_company_days": len(aggregate.company_days),
                "company_day_coverage": _safe_divide(len(aggregate.company_days), possible_company_days),
            }
        )
    return rows


def _build_category_rows(category_assoc_counts: Counter[tuple[str, str, str]], event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_counts: Counter[tuple[str, str, str]] = Counter()
    for row in event_rows:
        scope = str(row["source_scope"])
        scopes = (SOURCE_ALL, scope) if scope in {SOURCE_REUTERS, SOURCE_NON_REUTERS} else (SOURCE_ALL, "Mixed")
        for source_scope in scopes:
            event_counts[(source_scope, "event_type", str(row["event_type"]))] += 1
            event_counts[(source_scope, "tradability_class", str(row["tradability_class"]))] += 1
            event_counts[(source_scope, "sentiment_clarity", str(row["sentiment_clarity"]))] += 1
            event_counts[(source_scope, "expected_horizon", str(row["expected_horizon"]))] += 1
    totals: Counter[tuple[str, str]] = Counter()
    event_totals: Counter[tuple[str, str]] = Counter()
    for (scope, family, _), count in category_assoc_counts.items():
        totals[(scope, family)] += count
    for (scope, family, _), count in event_counts.items():
        event_totals[(scope, family)] += count
    rows = []
    keys = sorted(set(category_assoc_counts) | set(event_counts))
    for scope, family, value in keys:
        association_count = category_assoc_counts[(scope, family, value)]
        unique_event_count = event_counts[(scope, family, value)]
        rows.append(
            {
                "source_scope": scope,
                "category_family": family,
                "category": value,
                "association_count": association_count,
                "association_share": _safe_divide(association_count, totals[(scope, family)]),
                "unique_event_count": unique_event_count,
                "unique_event_share": _safe_divide(unique_event_count, event_totals[(scope, family)]),
            }
        )
    return rows


def _build_sample_rows(candidates: list[dict[str, Any]], *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or not candidates:
        return []
    rng = random.Random(seed)
    seen: set[str] = set()

    def take(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        available = [row for row in pool if row["sample_id"] not in seen]
        if len(available) <= n:
            chosen = available
        else:
            chosen = rng.sample(available, n)
        for row in chosen:
            seen.add(str(row["sample_id"]))
        return chosen

    reuters_target = min(400, sample_size)
    non_reuters_target = min(1_200, max(0, sample_size - reuters_target))
    rare_target = max(0, sample_size - reuters_target - non_reuters_target)
    reuters = take([row for row in candidates if row["source_scope"] == SOURCE_REUTERS], reuters_target)
    non_reuters = take([row for row in candidates if row["source_scope"] == SOURCE_NON_REUTERS], non_reuters_target)
    rare = take([row for row in candidates if row["rare_or_low_coverage"]], rare_target)
    remainder = take(candidates, sample_size - len(reuters) - len(non_reuters) - len(rare))
    rows = [*reuters, *non_reuters, *rare, *remainder]
    rows.sort(key=lambda row: (row["source_scope"], row["version_created"], row["sample_id"]))
    output = []
    for row in rows:
        output.append(
            {
                **row,
                "manual_event_type": "",
                "manual_tradability_class": "",
                "manual_sentiment_clarity": "",
                "manual_expected_horizon": "",
                "manual_notes": "",
                "llm_event_type": "",
                "llm_tradability_class": "",
                "llm_sentiment_clarity": "",
                "llm_expected_horizon": "",
            }
        )
    return output


def _build_signal_rows(panel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel in panel_rows:
        specs = (
            ("headline/sentiment_all", "headline_count", "mean_sentiment_score"),
            ("headline/sentiment_non_reuters", "non_reuters_count", "mean_non_reuters_sentiment_score"),
            ("headline/sentiment_reuters", "reuters_count", "mean_reuters_sentiment_score"),
            ("headline/actionable_sentiment_all", "actionable_count", "mean_actionable_sentiment_score"),
        )
        for scorer_id, count_field, score_field in specs:
            valid_count = int(panel[count_field])
            score = float(panel[score_field])
            mean_score: float | str
            signal_value: int | str
            if valid_count <= 0:
                signal = ""
                signal_value = ""
                mean_score = ""
            else:
                mean_score = score
                signal_value = 1 if score > 0 else -1 if score < 0 else 0
                signal = "positive" if signal_value == 1 else "negative" if signal_value == -1 else "neutral"
            rows.append(
                {
                    "symbol": panel["symbol"],
                    "news_date": panel["news_date"],
                    "scorer_id": scorer_id,
                    "article_count": panel["headline_count"],
                    "valid_count": valid_count,
                    "mean_score": mean_score,
                    "signal": signal,
                    "signal_value": signal_value,
                    "availability_timestamp": "",
                }
            )
    return rows


def _run_trading_value(
    signal_rows: list[dict[str, Any]],
    *,
    prices: Path | None,
    output_dir: Path,
    horizons: tuple[int, ...],
    timezone: str,
    transaction_cost_bps_per_side: float,
    notional_usd: float,
) -> tuple[str, list[Path], dict[str, Any]]:
    if prices is None or not prices.exists():
        blocked = {
            "status": "blocked_missing_prices",
            "reason": "Price CSV was not supplied or does not exist; coverage and information-value artifacts were still generated.",
            "prices_path": str(prices) if prices is not None else "",
        }
        path = _write_csv(output_dir / "trading_summary.csv", [blocked])
        note = atomic_write_text(output_dir / "trading_blocked.md", f"Trading value blocked: {blocked['reason']}\n")
        return "blocked_missing_prices", [path, note], blocked

    price_rows = load_prices_csv(prices)
    signals = _signals_from_rows(signal_rows)
    max_horizon = max(horizons)
    filtered, skipped = _filter_signals_with_price_horizon(signals, price_rows, max_horizon=max_horizon)
    strategy = get_strategy("headline_sentiment_threshold_v1")
    policy = DecisionPolicyConfig(
        min_valid_stories=1,
        threshold=0.0,
        transaction_cost_bps_per_side=transaction_cost_bps_per_side,
        policy_version=strategy.id,
    )
    result = run_backtest(
        filtered,
        price_rows,
        policy,
        horizons=horizons,
        notional_usd=notional_usd,
        timezone=timezone,
        use_decision_policy=True,
        decision_fn=strategy.make_policy().decide,
    )
    returns_path = _write_csv(output_dir / "trading_returns.csv", [_return_row_to_dict(row) for row in result.returns])
    decisions_path = _write_csv(output_dir / "trading_decisions.csv", [decision.__dict__ for decision in result.decisions])
    summary_rows = _trading_summary_rows(result.returns)
    if not summary_rows:
        summary_rows = [{"status": "completed_no_trades", "skipped_incomplete_price_horizon": skipped}]
    else:
        for row in summary_rows:
            row["status"] = "completed"
            row["skipped_incomplete_price_horizon"] = skipped
    summary_path = _write_csv(output_dir / "trading_summary.csv", summary_rows)
    metrics = {
        "status": "completed",
        "prices_path": str(prices),
        "signal_rows": len(signals),
        "signals_with_price_horizon": len(filtered),
        "skipped_incomplete_price_horizon": skipped,
        "return_rows": len(result.returns),
        "transaction_cost_bps_per_side": transaction_cost_bps_per_side,
        "notional_usd": notional_usd,
    }
    return "completed", [returns_path, decisions_path, summary_path], metrics


def _signals_from_rows(rows: list[dict[str, Any]]) -> list[DailySignal]:
    signals = []
    for row in rows:
        mean_score = None if row["mean_score"] == "" else float(row["mean_score"])
        signal_value = None if row["signal_value"] == "" else int(row["signal_value"])
        signals.append(
            DailySignal(
                symbol=str(row["symbol"]),
                news_date=str(row["news_date"]),
                scorer_id=str(row["scorer_id"]),
                article_count=int(row["article_count"]),
                valid_count=int(row["valid_count"]),
                mean_score=mean_score,
                signal=str(row["signal"]) or None,
                signal_value=signal_value,
                availability_timestamp=str(row.get("availability_timestamp") or "") or None,
            )
        )
    return signals


def _filter_signals_with_price_horizon(signals: list[DailySignal], prices: list[Any], *, max_horizon: int) -> tuple[list[DailySignal], int]:
    dates_by_symbol: dict[str, list[str]] = defaultdict(list)
    for price in prices:
        dates_by_symbol[price.symbol].append(price.session_date)
    for symbol in dates_by_symbol:
        dates_by_symbol[symbol] = sorted(set(dates_by_symbol[symbol]))
    filtered = []
    skipped = 0
    for signal in signals:
        if signal.mean_score is None or signal.mean_score == 0:
            filtered.append(signal)
            continue
        future = [session for session in dates_by_symbol.get(signal.symbol, []) if session > signal.news_date]
        if len(future) < max_horizon:
            skipped += 1
            continue
        filtered.append(signal)
    return filtered, skipped


def _return_row_to_dict(row: ReturnRow) -> dict[str, Any]:
    return row.__dict__


def _trading_summary_rows(returns: list[ReturnRow]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[str, int], list[ReturnRow]] = defaultdict(list)
    for row in returns:
        grouped[(row.scorer_id, row.horizon)].append(row)
    for (scorer_id, horizon), group in sorted(grouped.items()):
        net_returns = [row.net_strategy_return_pct for row in group]
        gross_returns = [row.strategy_return_pct for row in group]
        traded = [row for row in group if row.signal_value != 0]
        rows.append(
            {
                "scorer_id": scorer_id,
                "horizon": horizon,
                "return_rows": len(group),
                "traded_rows": len(traded),
                "mean_gross_return_pct": fmean(gross_returns) if gross_returns else 0.0,
                "mean_net_return_pct": fmean(net_returns) if net_returns else 0.0,
                "median_net_return_pct": sorted(net_returns)[len(net_returns) // 2] if net_returns else 0.0,
                "net_hit_rate": _safe_divide(sum(1 for value in net_returns if value > 0), len(net_returns)),
                "return_risk_ratio": _safe_divide(fmean(net_returns), pstdev(net_returns)) if len(net_returns) > 1 else 0.0,
                "total_net_pnl_usd": sum(row.net_pnl_usd for row in group),
            }
        )
    return rows


def _summary_markdown(
    *,
    counts: dict[str, Any],
    category_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    trading_status: str,
    trading_metrics: dict[str, Any],
    prices_path: Path | None,
) -> str:
    non_reuters_rows = sum(int(row["headline_rows"]) for row in source_rows if row["source_scope"] == SOURCE_NON_REUTERS)
    reuters_rows = sum(int(row["headline_rows"]) for row in source_rows if row["source_scope"] == SOURCE_REUTERS)
    top_source = source_rows[0] if source_rows else {}
    actionable = _category_lookup(category_rows, SOURCE_ALL, "tradability_class", "single_company_actionable")
    duplicate = _category_lookup(category_rows, SOURCE_ALL, "tradability_class", "duplicate_syndicated")
    low_info = _category_lookup(category_rows, SOURCE_ALL, "tradability_class", "low_information")
    lines = [
        "# Headline Value Analysis",
        "",
        "## Summary",
        "",
        f"- Headline rows profiled: {counts['headline_rows']:,}.",
        f"- Company-day panel rows: {counts['company_day_panel_rows']:,}; covered company-days: {counts['covered_company_days']:,}.",
        f"- Duplicate-adjusted event rows: {counts['headline_events']:,}.",
        f"- Reuters rows: {reuters_rows:,}; non-Reuters rows: {non_reuters_rows:,}.",
        f"- Largest source: {top_source.get('source_code', 'n/a')} ({float(top_source.get('share_of_all_rows', 0)):.1%} of rows).",
        "",
        "## Information value",
        "",
        f"- Single-company actionable association share: {float(actionable.get('association_share', 0)):.1%}.",
        f"- Duplicate/syndicated association share: {float(duplicate.get('association_share', 0)):.1%}.",
        f"- Low-information association share: {float(low_info.get('association_share', 0)):.1%}.",
        "",
        "## Trading value",
        "",
    ]
    if trading_status != "completed":
        lines.extend(
            [
                f"- Status: `{trading_status}`.",
                f"- Price file: `{prices_path}`.",
                f"- Reason: {trading_metrics.get('reason', 'Price data unavailable.')}",
            ]
        )
    else:
        lines.extend(
            [
                "- Status: `completed`.",
                f"- Return rows: {trading_metrics.get('return_rows', 0):,}.",
                "- Skipped signals near missing/incomplete price horizons: "
                f"{trading_metrics.get('skipped_incomplete_price_horizon', 0):,}.",
                "- See `trading_summary.csv` for net return, hit-rate, and return/risk by signal family and horizon.",
            ]
        )
    lines.extend(
        [
            "",
            "## Local calibration sample",
            "",
            "`sample_label_template.csv` contains raw licensed headline text for local manual/LLM calibration only. "
            "Do not commit or redistribute it.",
        ]
    )
    return "\n".join(lines) + "\n"


def _category_lookup(rows: list[dict[str, Any]], scope: str, family: str, category: str) -> dict[str, Any]:
    return next(
        (row for row in rows if row["source_scope"] == scope and row["category_family"] == family and row["category"] == category),
        {"association_share": 0, "association_count": 0},
    )


def _add_company_zscores(rows: list[dict[str, Any]], *, fields: tuple[str, ...]) -> None:
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symbol[str(row["symbol"])].append(row)
    for symbol_rows in by_symbol.values():
        for field_name in fields:
            values = [float(row[field_name]) for row in symbol_rows]
            mean = fmean(values) if values else 0.0
            spread = pstdev(values) if len(values) > 1 else 0.0
            for row in symbol_rows:
                row[f"{field_name}_z"] = 0.0 if spread == 0 else (float(row[field_name]) - mean) / spread


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        return atomic_write_text(path, "")
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for field_name in row:
            if field_name not in seen:
                seen.add(field_name)
                fieldnames.append(field_name)
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return atomic_write_text(path, output.getvalue())


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
