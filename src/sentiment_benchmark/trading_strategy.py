from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from zoneinfo import ZoneInfo

from .artifact_io import canonical_json, sha256_text
from .backtest import (
    DailySignal,
    DecisionPolicyConfig,
    IndexFallback,
    ReturnRow,
    TradingDecision,
    TradingStrategyError,
    calculate_returns,
    make_trading_decisions,
)
from .baselines import classify_finbert_texts, classify_vader_text
from .constants import DEFAULT_OLLAMA_HOST
from .entity_masking import mask_entities
from .lseg_corpus import load_verified_lseg_corpus
from .lseg_source import LsegNewsError
from .model_roster import is_post_cutoff, knowledge_cutoff, roster_cutoff
from .models import BlindExample, LLMResponseRecord, PromptConfig
from .news_source import NewsArticleRecord, make_news_fetch_config, write_news_corpus
from .newsapi_source import make_newsapi_fetch_config, write_newsapi_corpus
from .prices import PriceProviderError, PriceRow, make_price_provider
from .prompts import load_prompts
from .runtime_metadata import collect_run_environment

LABEL_VALUES = {"positive": 1, "neutral": 0, "negative": -1}
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "guccounter",
    "guce_referrer",
    "guce_referrer_sig",
    "mc_cid",
    "mc_eid",
}
PROMOTION_PATTERN = re.compile(
    r"\b(?:coupon|discount|record low|prime day|shopping deal|price drop|cheaper on|save \d+%|% off)\b",
    re.IGNORECASE,
)


# ``TradingStrategyError`` and the backtest-core types/functions now live in
# ``backtest``; imported above and re-exported here for backwards compatibility.


@dataclass(frozen=True)
class TradingCompany:
    symbol: str
    name: str
    query: str
    tavily_query_id: str
    family: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class PricesConfig:
    """Price-source settings. ``cache_dir`` is opt-in; ``None`` disables caching."""

    provider: str = "yfinance"
    cache_dir: Path | None = None
    adjusted: bool = True


@dataclass(frozen=True)
class CutoffConfig:
    """LLM knowledge-cutoff contamination policy.

    ``stratify`` (default) only annotates scores so results can be split pre/post
    cutoff downstream — the frozen primary cell is untouched. ``post_only``
    additionally restricts the traded signal to post-cutoff (contamination-free)
    scores. ``ignore`` disables annotation. ``overrides`` maps a model id (or its
    provider-suffix) to an authoritative ISO cutoff date.
    """

    policy: str = "stratify"
    overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TradingStrategyConfig:
    run_id: str
    title: str
    timezone: str
    dates: tuple[str, ...]
    horizons: tuple[int, ...]
    notional_usd: float
    provider: str
    models: tuple[str, ...]
    primary_model: str
    baselines: tuple[str, ...]
    consensus_enabled: bool
    masking_mode: str
    prompt_id: str
    prompts_path: Path
    temperature: float
    max_completion_tokens: int
    concurrency: int
    retries: int
    resume_scores_path: Path | None
    ollama_host: str
    ollama_keep_alive: str | int | None
    ollama_think: bool | None
    structured_output: bool
    tavily_package_manifest: Path | None
    lseg_corpus_manifest: Path | None
    newsapi_enabled: bool
    tavily_gap_fetch: bool
    newsapi_sources_file: Path | None
    screening_overrides_path: Path | None
    news_output_root: Path
    derived_output_root: Path
    results_output_root: Path
    experiment_registry: Path
    newsapi_max_pages: int
    index_fallback: IndexFallback | None
    decision_policy_enabled: bool
    decision_policy: DecisionPolicyConfig
    prices: PricesConfig
    cutoff: CutoffConfig
    companies: tuple[TradingCompany, ...]

    @property
    def derived_dir(self) -> Path:
        return self.derived_output_root / self.run_id

    @property
    def results_dir(self) -> Path:
        return self.results_output_root / self.run_id


@dataclass(frozen=True)
class ArticleCandidate:
    company: TradingCompany
    provider: str
    query: str
    source_corpus: str
    record: NewsArticleRecord


@dataclass
class MergedArticle:
    article_id: str
    symbol: str
    company_name: str
    title: str
    snippet: str
    scoring_text: str
    url: str
    normalized_url: str
    source_domain: str
    providers: list[str]
    queries: list[str]
    source_corpora: list[str]
    published_at: str
    published_date_local: str
    published_precision: str
    timezone: str
    article_text_available: bool
    article_text_sha256: str
    screening_decision: str
    screening_reason: str
    screening_method: str
    source_id: str = ""
    source_revision_id: str = ""
    scoring_text_sha256: str = ""
    scoring_text_truncated: bool = False


@dataclass(frozen=True)
class SentimentScore:
    article_id: str
    symbol: str
    news_date: str
    scorer_id: str
    scorer_kind: str
    label: str | None
    label_value: int | None
    status: str
    parse_status: str
    raw_output: str | None
    compound: float | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    generation_id: str | None
    total_cost_usd: float | None
    error: str | None
    source_revision_id: str | None = None
    content_sha256: str | None = None
    prompt_hash: str | None = None
    request_sha256: str | None = None
    model_digest: str | None = None
    scoring_text_truncated: bool = False
    # Contamination metadata, filled by annotate_scores_with_cutoff (see model_roster).
    model_knowledge_cutoff: str | None = None
    is_post_cutoff: bool | None = None
    # Entity-masking ablation metadata (see entity_masking).
    masking_mode: str = "off"
    entity_masked: bool = False
    n_masked_tokens: int | None = None


# ``DailySignal``, ``TradingDecision`` and ``ReturnRow`` now live in ``backtest``
# (``PriceRow`` in ``prices``); imported above and re-exported here for
# backwards compatibility with existing call sites/tests.


@dataclass(frozen=True)
class TradingRunResult:
    run_id: str
    derived_dir: Path
    results_dir: Path
    accepted_article_count: int
    sentiment_score_count: int
    return_count: int
    source_corpora: tuple[str, ...]


def _as_tuple(value: Any, *, field: str) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value:
        raise TradingStrategyError(f"{field} must be a non-empty list")
    return tuple(value)


def _optional_string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TradingStrategyError(f"{field} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


def load_trading_config(path: str | Path) -> TradingStrategyConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TradingStrategyError(f"cannot load trading config {config_path}: {exc}") from exc
    config = _build_trading_config(raw, config_path)
    _validate_trading_config(config)
    return config


def _build_trading_config(raw: dict[str, Any], config_path: Path) -> TradingStrategyConfig:
    run = raw.get("run") or {}
    scoring = raw.get("scoring") or {}
    sources = raw.get("sources") or {}
    outputs = raw.get("outputs") or {}
    signal_policy_raw = raw.get("signal_policy") or {}
    index_fallback_raw = raw.get("index_fallback") or {}
    company_items = raw.get("companies") or []
    try:
        dates = tuple(str(value) for value in _as_tuple(run["dates"], field="run.dates"))
        horizons = tuple(int(value) for value in _as_tuple(run["horizons"], field="run.horizons"))
        models = tuple(str(value) for value in _as_tuple(scoring["models"], field="scoring.models"))
        companies = tuple(
            TradingCompany(
                symbol=str(item["symbol"]).strip().upper(),
                name=str(item["name"]).strip(),
                query=str(item.get("query") or f"{item['name']} {item['symbol']} stock news").strip(),
                tavily_query_id=str(item.get("tavily_query_id") or "").strip(),
                family=str(item.get("family") or f"{item['symbol']} — {item['name']}").strip(),
                aliases=tuple(str(alias).strip() for alias in item["aliases"] if str(alias).strip()),
            )
            for item in company_items
        )
        provider = str(scoring.get("provider", "openrouter")).strip().lower()
        consensus_enabled = bool(scoring.get("consensus_enabled", True))
        primary_model = str(
            scoring.get("primary_model") or ("consensus/majority" if consensus_enabled else models[0])
        ).strip()
        baselines = _optional_string_tuple(scoring.get("baselines", ["vader"]), field="scoring.baselines")
        lseg_corpus_manifest = Path(sources["lseg_corpus_manifest"]) if sources.get("lseg_corpus_manifest") else None
        tavily_package_manifest = (
            Path(sources["tavily_package_manifest"]) if sources.get("tavily_package_manifest") else None
        )
        newsapi_enabled = bool(sources.get("newsapi_enabled", lseg_corpus_manifest is None))
        tavily_gap_fetch = bool(
            sources.get("tavily_gap_fetch", tavily_package_manifest is not None and newsapi_enabled)
        )
        decision_policy_enabled = "signal_policy" in raw
        decision_policy = DecisionPolicyConfig(
            min_valid_stories=int(signal_policy_raw.get("min_valid_stories", 1)),
            threshold=float(signal_policy_raw.get("threshold", 0.0)),
            transaction_cost_bps_per_side=float(signal_policy_raw.get("transaction_cost_bps_per_side", 0.0)),
            short_borrow_bps_per_day=float(signal_policy_raw.get("short_borrow_bps_per_day", 0.0)),
            policy_version=str(signal_policy_raw.get("policy_version", "sentiment_threshold_v1")).strip(),
        )
        index_fallback = (
            IndexFallback(
                symbol=str(index_fallback_raw.get("symbol", "")).strip(),
                min_texts=int(index_fallback_raw.get("min_texts", 3)),
            )
            if index_fallback_raw.get("enabled")
            else None
        )
        prices_raw = raw.get("prices") or {}
        prices = PricesConfig(
            provider=str(prices_raw.get("provider", "yfinance")).strip().lower(),
            cache_dir=Path(prices_raw["cache_dir"]) if prices_raw.get("cache_dir") else None,
            adjusted=bool(prices_raw.get("adjusted", True)),
        )
        cutoff_raw = raw.get("cutoff") or {}
        cutoff = CutoffConfig(
            policy=str(cutoff_raw.get("policy", "stratify")).strip().lower(),
            overrides={str(key): str(value) for key, value in (cutoff_raw.get("overrides") or {}).items()},
        )
        config = TradingStrategyConfig(
            run_id=str(run["id"]).strip(),
            title=str(run["title"]).strip(),
            timezone=str(run.get("timezone", "America/New_York")).strip(),
            dates=dates,
            horizons=horizons,
            notional_usd=float(run.get("notional_usd", 10_000)),
            provider=provider,
            models=models,
            primary_model=primary_model,
            baselines=baselines,
            consensus_enabled=consensus_enabled,
            masking_mode=str(scoring.get("masking_mode", "off")).strip().lower(),
            prompt_id=str(scoring["prompt_id"]).strip(),
            prompts_path=Path(scoring.get("prompts_path", "configs/default_prompts.toml")),
            temperature=float(scoring.get("temperature", 0.0)),
            max_completion_tokens=int(scoring.get("max_completion_tokens", 64)),
            concurrency=int(scoring.get("concurrency", 3)),
            retries=int(scoring.get("retries", 3)),
            resume_scores_path=Path(scoring["resume_scores_path"]) if scoring.get("resume_scores_path") else None,
            ollama_host=str(scoring.get("ollama_host") or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/"),
            ollama_keep_alive=scoring.get("ollama_keep_alive"),
            ollama_think=(bool(scoring["ollama_think"]) if "ollama_think" in scoring else (False if provider == "ollama" else None)),
            structured_output=bool(scoring.get("structured_output", provider == "ollama")),
            tavily_package_manifest=tavily_package_manifest,
            lseg_corpus_manifest=lseg_corpus_manifest,
            newsapi_enabled=newsapi_enabled,
            tavily_gap_fetch=tavily_gap_fetch,
            newsapi_sources_file=Path(sources["newsapi_sources_file"]) if sources.get("newsapi_sources_file") else None,
            screening_overrides_path=(
                Path(sources["screening_overrides_path"]) if sources.get("screening_overrides_path") else None
            ),
            news_output_root=Path(outputs.get("news_output_root", "Data/news")),
            derived_output_root=Path(outputs.get("derived_output_root", "Data/derived/trading")),
            results_output_root=Path(outputs.get("results_output_root", "results/trading")),
            experiment_registry=Path(outputs.get("experiment_registry", "experiments/manifest.toml")),
            newsapi_max_pages=int(sources.get("newsapi_max_pages", 10)),
            index_fallback=index_fallback,
            decision_policy_enabled=decision_policy_enabled,
            decision_policy=decision_policy,
            prices=prices,
            cutoff=cutoff,
            companies=companies,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TradingStrategyError(f"invalid trading config {config_path}: {exc}") from exc
    return config


def _validate_lseg_ollama_scoring(config: TradingStrategyConfig) -> None:
    if len(config.models) != 2:
        raise TradingStrategyError("LSEG Ollama scoring requires exactly one primary and one secondary model")
    if config.temperature != 0:
        raise TradingStrategyError("LSEG Ollama scoring requires temperature = 0")
    if config.concurrency != 1:
        raise TradingStrategyError("LSEG Ollama scoring requires concurrency = 1")
    if config.ollama_think is not False:
        raise TradingStrategyError("LSEG Ollama scoring requires ollama_think = false")
    if not config.structured_output:
        raise TradingStrategyError("LSEG Ollama scoring requires structured_output = true")


def _validate_trading_config(config: TradingStrategyConfig) -> None:
    if not config.run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", config.run_id):
        raise TradingStrategyError("run.id must contain only letters, numbers, dots, underscores, or hyphens")
    if not config.companies:
        raise TradingStrategyError("at least one [[companies]] entry is required")
    if config.cutoff.policy not in {"ignore", "stratify", "post_only"}:
        raise TradingStrategyError("cutoff.policy must be ignore, stratify, or post_only")
    if config.masking_mode not in {"off", "on", "both"}:
        raise TradingStrategyError("scoring.masking_mode must be off, on, or both")
    if len({company.symbol for company in config.companies}) != len(config.companies):
        raise TradingStrategyError("company symbols must be unique")
    if config.provider not in {"openrouter", "ollama"}:
        raise TradingStrategyError("scoring.provider must be openrouter or ollama")
    if len(set(config.models)) != len(config.models) or any(not model.strip() for model in config.models):
        raise TradingStrategyError("scoring.models must contain unique non-empty model IDs")
    if config.primary_model not in config.models and not (
        config.consensus_enabled and config.primary_model == "consensus/majority"
    ):
        raise TradingStrategyError(
            "scoring.primary_model must be included in scoring.models, or be consensus/majority when consensus is enabled"
        )
    if config.provider == "ollama" and any(":" not in model for model in config.models):
        raise TradingStrategyError("Ollama research models must use exact tags, for example model:tag")
    if config.provider == "ollama" and config.lseg_corpus_manifest is not None:
        _validate_lseg_ollama_scoring(config)
    unsupported_baselines = sorted(set(config.baselines) - {"vader", "finbert"})
    if unsupported_baselines:
        raise TradingStrategyError(f"unsupported trading baseline(s): {', '.join(unsupported_baselines)}")
    if config.tavily_package_manifest is None and config.lseg_corpus_manifest is None and not config.newsapi_enabled:
        raise TradingStrategyError("at least one of Tavily, LSEG, or NewsAPI must be enabled")
    for value in config.dates:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise TradingStrategyError(f"invalid run date: {value}") from exc
    if sorted(set(config.horizons)) != list(config.horizons) or config.horizons[0] < 1:
        raise TradingStrategyError("run.horizons must be unique, ascending positive integers")
    if config.notional_usd <= 0 or config.concurrency < 1 or config.max_completion_tokens < 1:
        raise TradingStrategyError("notional, concurrency, and completion-token settings must be positive")
    if config.decision_policy.min_valid_stories < 1:
        raise TradingStrategyError("signal_policy.min_valid_stories must be 1 or greater")
    if not 0 <= config.decision_policy.threshold <= 1:
        raise TradingStrategyError("signal_policy.threshold must be between 0 and 1")
    if config.decision_policy.transaction_cost_bps_per_side < 0 or config.decision_policy.short_borrow_bps_per_day < 0:
        raise TradingStrategyError("signal-policy cost assumptions cannot be negative")
    try:
        ZoneInfo(config.timezone)
    except Exception as exc:
        raise TradingStrategyError(f"unknown timezone: {config.timezone}") from exc
    if config.index_fallback is not None:
        if not config.index_fallback.symbol:
            raise TradingStrategyError("index_fallback.symbol is required when index_fallback.enabled is true")
        if config.index_fallback.min_texts < 1:
            raise TradingStrategyError("index_fallback.min_texts must be 1 or greater")


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS and not key.lower().startswith("utm_")
    ]
    return urlunparse(
        (
            parsed.scheme.lower() or "https",
            host,
            parsed.path.rstrip("/") or "/",
            "",
            urlencode(sorted(query)),
            "",
        )
    )


def normalize_headline(title: str) -> str:
    # Publisher suffixes and syndication bylines commonly turn one wire story
    # into several superficially different headlines. Remove those decorations
    # before exact-title deduplication while retaining the substantive title.
    substantive = re.split(r"\s+(?:by|[-|])\s+", title, maxsplit=1, flags=re.IGNORECASE)[0]
    substantive = substantive.replace("’s", "").replace("'s", "")
    return re.sub(r"[^a-z0-9]+", " ", substantive.lower()).strip()


def _parse_published(value: str | None, timezone: str) -> tuple[str, str, str]:
    if not value:
        return "", "", "missing"
    cleaned = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        return cleaned, cleaned, "date"
    try:
        parsed = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            return cleaned, "", "unparsed"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local = parsed.astimezone(ZoneInfo(timezone))
    return local.isoformat(), local.date().isoformat(), "timestamp"


def _alias_match(text: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE):
            return True
    return False


def _screen(title: str, aliases: tuple[str, ...]) -> tuple[str, str]:
    if not title.strip():
        return "exclude", "missing_title"
    if not _alias_match(title, aliases):
        return "exclude", "target_not_in_title"
    if PROMOTION_PATTERN.search(title):
        return "exclude", "consumer_promotion"
    return "include", "target_alias_in_title"


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _article_from_group(
    company: TradingCompany,
    candidates: list[ArticleCandidate],
    timezone: str,
) -> MergedArticle:
    titles = [_clean_text(candidate.record.title) for candidate in candidates]
    snippets = [_clean_text(candidate.record.snippet) for candidate in candidates]
    title = max(titles, key=len, default="")
    snippet = max(snippets, key=len, default="")
    dated = []
    for candidate in candidates:
        parsed, local_date, precision = _parse_published(candidate.record.published_date, timezone)
        if local_date:
            dated.append((parsed, local_date, precision))
    published_at, published_local, precision = min(dated, default=("", "", "missing"))
    canonical_urls = [canonicalize_url(candidate.record.url) for candidate in candidates]
    canonical_url = min(canonical_urls, key=len)
    article_id = hashlib.sha256(f"{company.symbol}|{canonical_url}".encode()).hexdigest()[:16]
    article_texts = [candidate.record.article_text or "" for candidate in candidates if candidate.record.article_text]
    article_text = max(article_texts, key=len, default="")
    decision, reason = _screen(title, company.aliases)
    scoring_text = f"Target company: {company.name} ({company.symbol})\nHeadline: {title}"
    if snippet:
        scoring_text += f"\nSummary: {snippet}"
    scoring_text_sha256 = hashlib.sha256(scoring_text.encode()).hexdigest()
    return MergedArticle(
        article_id=article_id,
        symbol=company.symbol,
        company_name=company.name,
        title=title,
        snippet=snippet,
        scoring_text=scoring_text,
        url=candidates[0].record.url,
        normalized_url=canonical_url,
        source_domain=urlparse(canonical_url).netloc,
        providers=sorted({candidate.provider for candidate in candidates}),
        queries=sorted({candidate.query for candidate in candidates}),
        source_corpora=sorted({candidate.source_corpus for candidate in candidates}),
        published_at=published_at,
        published_date_local=published_local,
        published_precision=precision,
        timezone=timezone,
        article_text_available=bool(article_text),
        article_text_sha256=hashlib.sha256(article_text.encode()).hexdigest() if article_text else "",
        screening_decision=decision,
        screening_reason=reason,
        screening_method="automatic_title_rule_v1",
        source_id=canonical_url,
        source_revision_id=candidates[0].record.record_id,
        scoring_text_sha256=scoring_text_sha256,
    )


def merge_article_candidates(
    candidates: list[ArticleCandidate],
    *,
    timezone: str,
    selected_dates: tuple[str, ...],
) -> list[MergedArticle]:
    by_company: dict[str, list[ArticleCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_company[candidate.company.symbol].append(candidate)
    merged: list[MergedArticle] = []
    for _symbol, company_candidates in sorted(by_company.items()):
        company = company_candidates[0].company
        by_url: dict[str, list[ArticleCandidate]] = defaultdict(list)
        for candidate in company_candidates:
            by_url[canonicalize_url(candidate.record.url)].append(candidate)
        url_groups = list(by_url.values())
        by_title: dict[str, list[ArticleCandidate]] = defaultdict(list)
        no_title: list[list[ArticleCandidate]] = []
        for group in url_groups:
            normalized_title = normalize_headline(max((_clean_text(item.record.title) for item in group), key=len, default=""))
            if normalized_title:
                by_title[normalized_title].extend(group)
            else:
                no_title.append(group)
        groups = list(by_title.values()) + no_title
        for group in groups:
            article = _article_from_group(company, group, timezone)
            if article.published_date_local in selected_dates:
                merged.append(article)
    return sorted(merged, key=lambda row: (row.published_date_local, row.symbol, row.article_id))


def apply_screening_overrides(
    articles: list[MergedArticle],
    overrides_path: Path | None,
) -> list[MergedArticle]:
    if overrides_path is None:
        return articles
    try:
        with overrides_path.open("rb") as file:
            raw = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TradingStrategyError(f"cannot load screening overrides {overrides_path}: {exc}") from exc
    overrides: dict[str, tuple[str, str]] = {}
    for item in raw.get("overrides", []):
        article_id = str(item.get("article_id") or "").strip()
        decision = str(item.get("decision") or "").strip().lower()
        reason = str(item.get("reason") or "manual_review").strip()
        if not article_id or decision not in {"include", "exclude"}:
            raise TradingStrategyError("each screening override requires article_id and decision=include|exclude")
        overrides[article_id] = (decision, reason)
    known = {article.article_id for article in articles}
    missing = sorted(set(overrides) - known)
    if missing:
        raise TradingStrategyError(f"screening override article id(s) not found: {', '.join(missing)}")
    for article in articles:
        override = overrides.get(article.article_id)
        if override:
            article.screening_decision, article.screening_reason = override
            article.screening_method = "manual_review_override_v1"
    return articles


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TradingStrategyError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _record_from_dict(value: dict[str, Any]) -> NewsArticleRecord:
    fields = NewsArticleRecord.__dataclass_fields__
    return NewsArticleRecord(**{name: value[name] for name in fields if name in value})


def load_tavily_candidates(config: TradingStrategyConfig) -> list[ArticleCandidate]:
    if config.tavily_package_manifest is None:
        return []
    try:
        package = json.loads(config.tavily_package_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TradingStrategyError(f"cannot load Tavily package manifest {config.tavily_package_manifest}: {exc}") from exc
    source_corpora = package.get("source_corpora")
    if not isinstance(source_corpora, list):
        raise TradingStrategyError("Tavily package manifest has no source_corpora list")
    companies = {company.tavily_query_id: company for company in config.companies}
    candidates: list[ArticleCandidate] = []
    for source in source_corpora:
        if not isinstance(source, dict):
            continue
        company = companies.get(str(source.get("query_id") or ""))
        if company is None:
            continue
        corpus = Path(str(source.get("path") or ""))
        articles_path = corpus / "articles.jsonl"
        if not articles_path.exists():
            raise TradingStrategyError(f"Tavily corpus is missing: {articles_path}")
        for row in _read_jsonl(articles_path):
            candidates.append(
                ArticleCandidate(
                    company=company,
                    provider="tavily",
                    query=str(source.get("query") or company.query),
                    source_corpus=str(corpus),
                    record=_record_from_dict(row),
                )
            )
    return candidates


def load_lseg_articles(config: TradingStrategyConfig) -> tuple[list[MergedArticle], list[str]]:
    if config.lseg_corpus_manifest is None:
        return [], []
    try:
        _manifest, records = load_verified_lseg_corpus(config.lseg_corpus_manifest)
    except LsegNewsError as exc:
        raise TradingStrategyError(str(exc)) from exc
    companies = {company.symbol: company for company in config.companies}
    articles: list[MergedArticle] = []
    for record in records:
        if not record.get("scoring_eligible"):
            continue
        clean_text = str(record.get("clean_text") or "")
        title = _clean_text(str(record.get("headline") or ""))
        version_created = str(record.get("version_created") or "")
        published_at, published_local, precision = _parse_published(version_created, config.timezone)
        if published_local not in config.dates:
            continue
        max_chars = 8_000
        scoring_body = clean_text[:max_chars]
        truncated = len(clean_text) > max_chars
        for symbol_value in record.get("matched_symbols") or []:
            symbol = str(symbol_value).upper()
            company = companies.get(symbol)
            if company is None:
                continue
            scoring_text = (
                f"Target company: {company.name} ({company.symbol})\n"
                f"Headline: {title}\n"
                "<news_document>\n"
                f"{scoring_body}\n"
                "</news_document>"
            )
            revision_id = str(record.get("revision_id") or "")
            article_id = hashlib.sha256(f"{symbol}|lseg|{revision_id}".encode()).hexdigest()[:16]
            article = MergedArticle(
                article_id=article_id,
                symbol=symbol,
                company_name=company.name,
                title=title,
                snippet="",
                scoring_text=scoring_text,
                url="",
                normalized_url="",
                source_domain="lseg",
                providers=["lseg"],
                queries=sorted(str(query) for query in record.get("matched_queries") or []),
                source_corpora=[str(config.lseg_corpus_manifest.parent)],
                published_at=published_at,
                published_date_local=published_local,
                published_precision=precision,
                timezone=config.timezone,
                article_text_available=True,
                article_text_sha256=str(record.get("clean_text_sha256") or ""),
                screening_decision="include",
                screening_reason="lseg_ric_query_and_text_quality",
                screening_method="lseg_metadata_quality_v1",
                source_id=str(record.get("story_family_id") or record.get("story_id") or ""),
                source_revision_id=revision_id,
                scoring_text_sha256=hashlib.sha256(scoring_text.encode()).hexdigest(),
                scoring_text_truncated=truncated,
            )
            articles.append(article)
    articles.sort(key=lambda row: (row.published_date_local, row.symbol, row.published_at, row.article_id))
    return articles, [str(config.lseg_corpus_manifest.parent)]


def _utc_newsapi_bounds(config: TradingStrategyConfig) -> tuple[str, str]:
    timezone = ZoneInfo(config.timezone)
    start = datetime.combine(min(date.fromisoformat(value) for value in config.dates), time.min, timezone)
    end = datetime.combine(max(date.fromisoformat(value) for value in config.dates) + timedelta(days=1), time.min, timezone)
    return start.astimezone(UTC).isoformat().replace("+00:00", "Z"), end.astimezone(UTC).isoformat().replace("+00:00", "Z")


async def fetch_newsapi_candidates(
    config: TradingStrategyConfig,
    client: Any,
) -> tuple[list[ArticleCandidate], list[str]]:
    if not config.newsapi_enabled:
        return [], []
    pointer_path = config.derived_dir / "newsapi_sources.json"
    restore_path = config.newsapi_sources_file or pointer_path
    candidates: list[ArticleCandidate] = []
    corpora: list[str] = []
    valid_pointers: list[dict[str, str]] = []
    restored_symbols: set[str] = set()
    if restore_path.exists():
        try:
            pointers = json.loads(restore_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pointers = None
        if isinstance(pointers, list):
            companies = {company.symbol: company for company in config.companies}
            for pointer in pointers:
                if not isinstance(pointer, dict):
                    continue
                company = companies.get(str(pointer.get("symbol") or ""))
                corpus = Path(str(pointer.get("path") or ""))
                if company is None or not (corpus / "articles.jsonl").exists():
                    continue
                for row in _read_jsonl(corpus / "articles.jsonl"):
                    candidates.append(
                        ArticleCandidate(company, "newsapi", company.query, str(corpus), _record_from_dict(row))
                    )
                corpora.append(str(corpus))
                valid_pointers.append({"symbol": company.symbol, "path": str(corpus)})
                restored_symbols.add(company.symbol)
            if len(restored_symbols) == len(config.companies):
                return candidates, corpora

    from_time, to_time = _utc_newsapi_bounds(config)
    for company in config.companies:
        if company.symbol in restored_symbols:
            continue
        fetch_config = make_newsapi_fetch_config(
            query=company.query,
            from_time=from_time,
            to_time=to_time,
            max_pages=config.newsapi_max_pages,
        )
        result = await client.fetch(fetch_config)
        paths = write_newsapi_corpus(result, config.news_output_root)
        corpora.append(str(paths.output_dir))
        valid_pointers.append({"symbol": company.symbol, "path": str(paths.output_dir)})
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(json.dumps(valid_pointers, indent=2, sort_keys=True), encoding="utf-8")
        for record in result.records:
            candidates.append(
                ArticleCandidate(
                    company=company,
                    provider="newsapi",
                    query=company.query,
                    source_corpus=str(paths.output_dir),
                    record=record,
                )
            )
    return candidates, corpora


async def fetch_tavily_gaps(
    config: TradingStrategyConfig,
    client: Any,
    current: list[ArticleCandidate],
) -> tuple[list[ArticleCandidate], list[str]]:
    if not config.tavily_gap_fetch:
        return [], []
    articles = merge_article_candidates(current, timezone=config.timezone, selected_dates=config.dates)
    articles = apply_screening_overrides(articles, config.screening_overrides_path)
    accepted = {(article.symbol, article.published_date_local) for article in articles if article.screening_decision == "include"}
    candidates: list[ArticleCandidate] = []
    corpora: list[str] = []
    for company in config.companies:
        for news_date in config.dates:
            if (company.symbol, news_date) in accepted:
                continue
            end_date = (date.fromisoformat(news_date) + timedelta(days=1)).isoformat()
            fetch_config = make_news_fetch_config(
                query=company.query,
                max_results=20,
                topic="news",
                time_range=None,
                search_depth="basic",
                extract=True,
                extract_depth="advanced",
                start_date=news_date,
                end_date=end_date,
                exclude_domains=["finance.yahoo.com", "ca.finance.yahoo.com"],
            )
            result = await client.fetch(fetch_config)
            paths = write_news_corpus(result, config.news_output_root)
            corpora.append(str(paths.output_dir))
            for record in result.records:
                candidates.append(
                    ArticleCandidate(company, "tavily", company.query, str(paths.output_dir), record)
                )
    return candidates, corpora


def _score_key(article_id: str, scorer_id: str) -> tuple[str, str]:
    return article_id, scorer_id


def _identity_score_key(score: SentimentScore) -> tuple[str, str, str, str, str, str, str]:
    return (
        score.article_id,
        score.scorer_id,
        score.source_revision_id or "",
        score.content_sha256 or "",
        score.prompt_hash or "",
        score.request_sha256 or "",
        score.model_digest or "",
    )


def _desired_score_key(
    article: MergedArticle,
    scorer_id: str,
    prompt_hash: str,
    request_sha256: str,
    model_digest: str | None,
) -> tuple[str, str, str, str, str, str, str]:
    return (
        article.article_id,
        scorer_id,
        article.source_revision_id,
        article.scoring_text_sha256 or hashlib.sha256(article.scoring_text.encode()).hexdigest(),
        prompt_hash,
        request_sha256,
        model_digest or "",
    )


def _sentiment_from_response(
    article: MergedArticle,
    scorer_id: str,
    record: LLMResponseRecord,
    cost: float | None,
    *,
    request_sha256: str,
    model_digest: str | None,
    masking_mode: str = "off",
    entity_masked: bool = False,
    n_masked_tokens: int | None = None,
) -> SentimentScore:
    label = record.normalized_label if record.normalized_label in LABEL_VALUES else None
    return SentimentScore(
        article_id=article.article_id,
        symbol=article.symbol,
        news_date=article.published_date_local,
        scorer_id=scorer_id,
        scorer_kind="llm",
        label=label,
        label_value=LABEL_VALUES.get(label) if label else None,
        status=record.status,
        parse_status=record.parse_status,
        raw_output=record.raw_content,
        compound=None,
        latency_ms=record.latency_ms,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        generation_id=record.generation_id,
        total_cost_usd=cost,
        error=record.error,
        source_revision_id=article.source_revision_id or None,
        content_sha256=article.scoring_text_sha256 or hashlib.sha256(article.scoring_text.encode()).hexdigest(),
        prompt_hash=record.prompt_hash,
        request_sha256=request_sha256,
        model_digest=model_digest,
        scoring_text_truncated=article.scoring_text_truncated,
        masking_mode=masking_mode,
        entity_masked=entity_masked,
        n_masked_tokens=n_masked_tokens,
    )


def _load_existing_scores(path: Path) -> list[SentimentScore]:
    if not path.exists():
        return []
    scores: list[SentimentScore] = []
    for value in _read_jsonl(path):
        try:
            score = SentimentScore(**value)
        except TypeError:
            continue
        scores.append(score)
    return scores


async def score_articles(
    articles: list[MergedArticle],
    *,
    models: tuple[str, ...],
    prompt: PromptConfig,
    client: Any,
    output_path: Path,
    resume_path: Path | None,
    temperature: float,
    max_completion_tokens: int,
    concurrency: int,
    retries: int,
    baselines: tuple[str, ...] = ("vader",),
    model_digests: dict[str, str | None] | None = None,
    request_settings: dict[str, Any] | None = None,
    strict_resume_identity: bool = False,
    scorer_suffix: str = "",
    masking_mode: str = "off",
    entity_masked: bool = False,
    masked_token_counts: dict[str, int] | None = None,
) -> list[SentimentScore]:
    accepted = [article for article in articles if article.screening_decision == "include"]
    existing_scores = _load_existing_scores(resume_path) if resume_path else []
    existing_scores.extend(_load_existing_scores(output_path))
    existing_legacy = {_score_key(score.article_id, score.scorer_id): score for score in existing_scores}
    existing_identity = {_identity_score_key(score): score for score in existing_scores}
    semaphore = asyncio.Semaphore(concurrency)
    resolved_request_settings = {
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        **(request_settings or {}),
    }
    request_sha256 = sha256_text(canonical_json(resolved_request_settings))
    resolved_model_digests = model_digests or {}

    async def classify(index: int, article: MergedArticle, model_id: str) -> SentimentScore:
        # API call uses the bare model id; the score is tagged with the (possibly
        # suffixed) scorer id so masked/unmasked arms resume independently.
        scorer_id = f"{model_id}{scorer_suffix}"
        if strict_resume_identity:
            key = _desired_score_key(
                article,
                scorer_id,
                prompt.prompt_hash,
                request_sha256,
                resolved_model_digests.get(model_id),
            )
            existing = existing_identity.get(key)
        else:
            existing = existing_legacy.get(_score_key(article.article_id, scorer_id))
        if existing is not None and existing.status == "success":
            return existing
        example = BlindExample(row_number=index, sentence=article.scoring_text)
        async with semaphore:
            response = await client.classify(
                model_id=model_id,
                prompt=prompt,
                example=example,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                retries=retries,
            )
            cost: float | None = None
            metadata_method = getattr(client, "get_generation_metadata", None)
            if response.status == "success" and response.generation_id and callable(metadata_method):
                try:
                    metadata = await metadata_method(response.generation_id, retries=retries)
                    raw_cost = metadata.get("total_cost") if isinstance(metadata, dict) else None
                    cost = float(raw_cost) if raw_cost is not None else None
                except Exception:
                    cost = None
            return _sentiment_from_response(
                article,
                scorer_id,
                response,
                cost,
                request_sha256=request_sha256,
                model_digest=resolved_model_digests.get(model_id),
                masking_mode=masking_mode,
                entity_masked=entity_masked,
                n_masked_tokens=(masked_token_counts or {}).get(article.article_id),
            )

    tasks = [
        classify(index, article, model_id)
        for index, article in enumerate(accepted, start=1)
        for model_id in models
    ]
    llm_scores = await asyncio.gather(*tasks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Checkpoint the paid LLM scores before running VADER: if the local baseline
    # raises (e.g. a missing lexicon), a resume reads these back instead of
    # re-issuing the API calls. The file is rewritten in full below.
    with output_path.open("w", encoding="utf-8") as file:
        for score in sorted(llm_scores, key=lambda row: (row.news_date, row.symbol, row.article_id, row.scorer_id)):
            file.write(json.dumps(asdict(score), sort_keys=True, ensure_ascii=False) + "\n")
    baseline_scores: list[SentimentScore] = []
    if "vader" in baselines:
        for article in accepted:
            result = classify_vader_text(article.scoring_text)
            baseline_scores.append(
                SentimentScore(
                    article_id=article.article_id,
                    symbol=article.symbol,
                    news_date=article.published_date_local,
                    scorer_id=f"baseline/vader{scorer_suffix}",
                    scorer_kind="baseline",
                    masking_mode=masking_mode,
                    entity_masked=entity_masked,
                    n_masked_tokens=(masked_token_counts or {}).get(article.article_id),
                    label=result.label,
                    label_value=LABEL_VALUES[result.label],
                    status="success",
                    parse_status="valid",
                    raw_output=result.label,
                    compound=result.compound,
                    latency_ms=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    generation_id=None,
                    total_cost_usd=0.0,
                    error=None,
                    source_revision_id=article.source_revision_id or None,
                    content_sha256=article.scoring_text_sha256 or hashlib.sha256(article.scoring_text.encode()).hexdigest(),
                    scoring_text_truncated=article.scoring_text_truncated,
                )
            )
    if "finbert" in baselines:
        predictions = classify_finbert_texts([article.scoring_text for article in accepted])
        for article, label in zip(accepted, predictions, strict=True):
            baseline_scores.append(
                SentimentScore(
                    article_id=article.article_id,
                    symbol=article.symbol,
                    news_date=article.published_date_local,
                    scorer_id=f"baseline/finbert{scorer_suffix}",
                    scorer_kind="baseline",
                    masking_mode=masking_mode,
                    entity_masked=entity_masked,
                    n_masked_tokens=(masked_token_counts or {}).get(article.article_id),
                    label=label,
                    label_value=LABEL_VALUES.get(label) if label else None,
                    status="success" if label else "malformed_response",
                    parse_status="valid" if label else "invalid",
                    raw_output=label,
                    compound=None,
                    latency_ms=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    generation_id=None,
                    total_cost_usd=0.0,
                    error=None if label else "FinBERT returned an unknown label",
                    source_revision_id=article.source_revision_id or None,
                    content_sha256=article.scoring_text_sha256 or hashlib.sha256(article.scoring_text.encode()).hexdigest(),
                    scoring_text_truncated=article.scoring_text_truncated,
                )
            )
    scores = sorted(
        llm_scores + baseline_scores,
        key=lambda row: (row.news_date, row.symbol, row.article_id, row.scorer_id),
    )
    with output_path.open("w", encoding="utf-8") as file:
        for score in scores:
            file.write(json.dumps(asdict(score), sort_keys=True, ensure_ascii=False) + "\n")
    return scores


def article_consensus(scores: list[SentimentScore], models: tuple[str, ...]) -> list[SentimentScore]:
    grouped: dict[str, list[SentimentScore]] = defaultdict(list)
    for score in scores:
        if score.scorer_id in models:
            grouped[score.article_id].append(score)
    consensus: list[SentimentScore] = []
    for article_scores in grouped.values():
        by_model = {score.scorer_id: score for score in article_scores if score.label in LABEL_VALUES and score.status == "success"}
        if set(by_model) != set(models):
            continue
        labels = [str(by_model[model].label) for model in models]
        counts = Counter(labels)
        label, count = counts.most_common(1)[0]
        if count < 2:
            label = "neutral"
        exemplar = article_scores[0]
        consensus.append(
            SentimentScore(
                article_id=exemplar.article_id,
                symbol=exemplar.symbol,
                news_date=exemplar.news_date,
                scorer_id="consensus/majority",
                scorer_kind="consensus",
                label=label,
                label_value=LABEL_VALUES[label],
                status="success",
                parse_status="valid",
                raw_output="|".join(str(value) for value in labels),
                compound=None,
                latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                generation_id=None,
                total_cost_usd=sum(score.total_cost_usd or 0.0 for score in article_scores),
                error=None,
                scoring_text_truncated=exemplar.scoring_text_truncated,
            )
        )
    return sorted(consensus, key=lambda row: (row.news_date, row.symbol, row.article_id))


def annotate_scores_with_cutoff(
    scores: list[SentimentScore],
    config: TradingStrategyConfig,
) -> list[SentimentScore]:
    """Stamp each score with its scorer's knowledge cutoff and ``is_post_cutoff``.

    Pure metadata — does not change labels or returns. A consensus scorer takes
    the most-conservative (latest) cutoff across the model roster; dictionary/ML
    baselines have no cutoff (``None``). Returns a new list; inputs are unchanged.
    """
    overrides = config.cutoff.overrides
    consensus_cut = roster_cutoff(config.models, overrides)
    annotated: list[SentimentScore] = []
    for score in scores:
        if score.scorer_id.startswith("consensus"):
            cutoff = consensus_cut
        elif score.scorer_id.startswith("baseline/"):
            cutoff = None
        else:
            cutoff = knowledge_cutoff(score.scorer_id, overrides)
        try:
            event = date.fromisoformat(score.news_date)
        except ValueError:
            event = None
        annotated.append(
            replace(
                score,
                model_knowledge_cutoff=cutoff.isoformat() if cutoff is not None else None,
                is_post_cutoff=is_post_cutoff(event, cutoff),
            )
        )
    return annotated


def _mask_articles(
    articles: list[MergedArticle],
    companies: dict[str, TradingCompany],
) -> tuple[list[MergedArticle], dict[str, int]]:
    """Entity-masked copies of ``articles`` (anonymised ``scoring_text`` and a
    refreshed hash) plus a per-article count of masked mentions. Articles whose
    symbol has no company entry pass through unmasked."""
    masked: list[MergedArticle] = []
    counts: dict[str, int] = {}
    for article in articles:
        company = companies.get(article.symbol)
        if company is None:
            masked.append(article)
            continue
        result = mask_entities(
            article.scoring_text,
            name=company.name,
            ticker=company.symbol,
            aliases=company.aliases,
        )
        counts[article.article_id] = result.n_masked
        masked.append(
            replace(
                article,
                scoring_text=result.masked_text,
                scoring_text_sha256=hashlib.sha256(result.masked_text.encode()).hexdigest(),
            )
        )
    return masked, counts


def daily_signals(
    articles: list[MergedArticle],
    scores: list[SentimentScore],
    scorer_ids: tuple[str, ...],
    dates: tuple[str, ...],
    symbols: tuple[str, ...],
) -> list[DailySignal]:
    accepted_counts = Counter(
        (article.symbol, article.published_date_local)
        for article in articles
        if article.screening_decision == "include"
    )
    article_by_id = {article.article_id: article for article in articles}
    grouped: dict[tuple[str, str, str], list[SentimentScore]] = defaultdict(list)
    for score in scores:
        grouped[(score.symbol, score.news_date, score.scorer_id)].append(score)
    results: list[DailySignal] = []
    for news_date in dates:
        for symbol in symbols:
            for scorer_id in scorer_ids:
                valid_scores = [
                    score
                    for score in grouped.get((symbol, news_date, scorer_id), [])
                    if score.label_value is not None
                ]
                values = [int(score.label_value) for score in valid_scores if score.label_value is not None]
                availability_values = [
                    article_by_id[score.article_id].published_at
                    for score in valid_scores
                    if score.article_id in article_by_id
                    and "lseg" in article_by_id[score.article_id].providers
                    and article_by_id[score.article_id].published_at
                ]
                mean = sum(values) / len(values) if values else None
                if mean is None:
                    signal = None
                    signal_value = None
                elif mean > 0:
                    signal, signal_value = "positive", 1
                elif mean < 0:
                    signal, signal_value = "negative", -1
                else:
                    signal, signal_value = "neutral", 0
                results.append(
                    DailySignal(
                        symbol=symbol,
                        news_date=news_date,
                        scorer_id=scorer_id,
                        article_count=accepted_counts[(symbol, news_date)],
                        valid_count=len(values),
                        mean_score=mean,
                        signal=signal,
                        signal_value=signal_value,
                        availability_timestamp=max(availability_values, default=None),
                    )
                )
    return results


def _price_window(config: TradingStrategyConfig) -> tuple[str, str]:
    """Inclusive start / exclusive end (ISO dates) covering every news date plus
    enough forward sessions to realise the longest horizon."""
    start = min(config.dates)
    requested_end = max(date.fromisoformat(value) for value in config.dates) + timedelta(days=max(config.horizons) * 3 + 7)
    end = min(requested_end, date.today() + timedelta(days=1)).isoformat()
    return start, end


def fetch_price_rows(config: TradingStrategyConfig) -> list[PriceRow]:
    """Adapter from config to the injectable ``price_loader`` seam: builds a
    :class:`~sentiment_benchmark.prices.PriceProvider` and fetches the window."""
    symbols = [company.symbol for company in config.companies]
    if config.index_fallback is not None and config.index_fallback.symbol not in symbols:
        symbols.append(config.index_fallback.symbol)
    start, end = _price_window(config)
    provider = make_price_provider(
        config.prices.provider,
        cache_dir=config.prices.cache_dir,
        adjusted=config.prices.adjusted,
    )
    try:
        return provider.fetch(symbols, start, end)
    except PriceProviderError as exc:
        raise TradingStrategyError(str(exc)) from exc


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as file:
        if not fields:
            return
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _public_article(article: MergedArticle) -> dict[str, Any]:
    value = asdict(article)
    value["providers"] = "; ".join(article.providers)
    value["queries"] = "; ".join(article.queries)
    value["source_corpora"] = "; ".join(article.source_corpora)
    return value


def write_articles(config: TradingStrategyConfig, articles: list[MergedArticle]) -> tuple[Path, Path]:
    config.derived_dir.mkdir(parents=True, exist_ok=True)
    articles_path = config.derived_dir / "articles.csv"
    screening_path = config.derived_dir / "screening.csv"
    _write_csv(articles_path, [_public_article(article) for article in articles])
    screening_fields = [
        "article_id",
        "symbol",
        "published_date_local",
        "title",
        "url",
        "providers",
        "screening_decision",
        "screening_reason",
        "screening_method",
    ]
    _write_csv(screening_path, [{key: _public_article(article).get(key, "") for key in screening_fields} for article in articles])
    manifest = {
        "schema_version": 1,
        "run_id": config.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "record_count": len(articles),
        "accepted_count": sum(article.screening_decision == "include" for article in articles),
        "notes": [
            "Scoring text is target-company context plus headline and provider snippet/description.",
            "Extracted Tavily bodies are not copied; only availability and SHA-256 are recorded.",
        ],
    }
    (config.derived_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return articles_path, screening_path


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(
    config: TradingStrategyConfig,
    articles: list[MergedArticle],
    signals: list[DailySignal],
    decisions: list[TradingDecision],
    returns: list[ReturnRow],
) -> Path:
    accepted = [article for article in articles if article.screening_decision == "include"]
    event_count = len({(row.symbol, row.news_date) for row in returns})
    source_counts = Counter((article.symbol, article.published_date_local) for article in accepted)
    decision_by_key = {(row.news_date, row.symbol, row.scorer_id): row for row in decisions}
    signal_rows = [
        [
            signal.news_date,
            signal.symbol,
            signal.scorer_id,
            str(signal.valid_count),
            "-" if signal.mean_score is None else f"{signal.mean_score:.3f}",
            signal.signal or "no signal",
            decision_by_key[(signal.news_date, signal.symbol, signal.scorer_id)].action,
            decision_by_key[(signal.news_date, signal.symbol, signal.scorer_id)].reason,
        ]
        for signal in signals
    ]
    mean_by_horizon: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in returns:
        mean_by_horizon[(row.scorer_id, row.horizon)].append(
            (row.strategy_return_pct, row.net_strategy_return_pct)
        )
    return_rows = [
        [
            scorer,
            str(horizon),
            f"{sum(value[0] for value in values) / len(values):.3f}%",
            f"{sum(value[1] for value in values) / len(values):.3f}%",
            str(len(values)),
        ]
        for (scorer, horizon), values in sorted(mean_by_horizon.items())
    ]
    source_rows = [
        [news_date, symbol, str(source_counts[(symbol, news_date)])]
        for news_date in config.dates
        for symbol in (company.symbol for company in config.companies)
    ]
    fallback_line = (
        f"\n- Thin-coverage fallback: company-days with fewer than {config.index_fallback.min_texts} accepted texts "
        f"trade {config.index_fallback.symbol} instead of the company's own stock"
        if config.index_fallback is not None
        else ""
    )
    consensus_line = (
        "\n- Consensus: per-article majority of the configured LLM labels; a split resolves to neutral"
        if config.consensus_enabled
        else ""
    )
    cost_bps = config.decision_policy.transaction_cost_bps_per_side
    text = f"""# {config.title}

Generated: {datetime.now(UTC).isoformat()}

This is an exploratory, non-preregistered research pilot. It is not causal evidence,
investment advice, or a funded portfolio backtest. Overlapping company-day events are
reported independently.

## Method

- Companies: {", ".join(company.symbol for company in config.companies)}
- News dates: {", ".join(config.dates)}
- News assignment timezone: `{config.timezone}`
- Entry: adjusted open of the first observed trading session after the news date
- Exit horizons: {", ".join(map(str, config.horizons))} trading sessions
- Position notional: ${config.notional_usd:,.0f}; transaction costs: {cost_bps:g} bps per side{fallback_line}
- LLM models: {", ".join(config.models)}
- Baselines: {", ".join(config.baselines) or "none"}{consensus_line}

## Accepted Article Yield

{_markdown_table(["News date", "Symbol", "Accepted articles"], source_rows)}

## Daily Signals

{_markdown_table(["News date", "Symbol", "Scorer", "Valid texts", "Mean", "Signal", "Action", "Reason"], signal_rows)}

## Equal-Weight Mean Event Return by Horizon

These are simple means of event-level returns, not returns on a capital-constrained portfolio.

{_markdown_table(["Scorer", "Horizon", "Gross mean", "Net mean", "Events"], return_rows)}

## Limitations

- News discovery and timestamp coverage differ by provider.
- Title-and-summary sentiment is a noisy proxy for market impact.
- The automated screen is conservative but may retain or reject borderline stories.
- {event_count} company-day observations are insufficient for statistical inference.
- The configured transaction-cost deduction is simplified; taxes, market impact, and borrow availability remain excluded.
"""
    path = config.results_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_charts(config: TradingStrategyConfig, articles: list[MergedArticle], returns: list[ReturnRow]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths: list[Path] = []
    accepted = [article for article in articles if article.screening_decision == "include"]
    counts = Counter(article.symbol for article in accepted)
    fig, axis = plt.subplots(figsize=(7, 4))
    symbols = [company.symbol for company in config.companies]
    axis.bar(symbols, [counts[symbol] for symbol in symbols], color="#2563eb")
    axis.set(title="Accepted news texts", ylabel="Article count")
    fig.tight_layout()
    path = config.results_dir / "source_yield.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in returns:
        grouped[(row.scorer_id, row.horizon)].append(row.strategy_return_pct)
    fig, axis = plt.subplots(figsize=(9, 5))
    for scorer in sorted({row.scorer_id for row in returns}):
        means = [sum(grouped[(scorer, horizon)]) / len(grouped[(scorer, horizon)]) for horizon in config.horizons]
        axis.plot(config.horizons, means, marker="o", label=scorer)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(title="Mean event return by trading-session horizon", xlabel="Horizon", ylabel="Strategy return (%)")
    axis.legend(fontsize=7)
    fig.tight_layout()
    path = config.results_dir / "horizon_returns.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_outputs(
    config: TradingStrategyConfig,
    config_path: Path,
    articles: list[MergedArticle],
    scores: list[SentimentScore],
    signals: list[DailySignal],
    decisions: list[TradingDecision],
    prices: list[PriceRow],
    returns: list[ReturnRow],
    source_corpora: list[str],
) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    score_csv = config.results_dir / "sentiment_scores.csv"
    signal_csv = config.results_dir / "daily_signals.csv"
    decision_csv = config.results_dir / "trading_decisions.csv"
    price_csv = config.results_dir / "prices.csv"
    return_csv = config.results_dir / "returns.csv"
    _write_csv(score_csv, [asdict(row) for row in scores])
    _write_csv(signal_csv, [asdict(row) for row in signals])
    _write_csv(decision_csv, [asdict(row) for row in decisions])
    _write_csv(price_csv, [asdict(row) for row in prices])
    _write_csv(return_csv, [asdict(row) for row in returns])
    summary_path = write_summary(config, articles, signals, decisions, returns)
    charts = write_charts(config, articles, returns)
    lseg_source_manifest: dict[str, Any] | None = None
    if config.lseg_corpus_manifest is not None:
        lseg_payload = json.loads(config.lseg_corpus_manifest.read_text(encoding="utf-8"))
        lseg_source_manifest = {
            "path": str(config.lseg_corpus_manifest),
            "sha256": _sha256(config.lseg_corpus_manifest),
            "build_fingerprint": lseg_payload.get("build_fingerprint"),
            "cleaner_version": lseg_payload.get("cleaner_version"),
        }
    files = [
        config.derived_dir / "articles.csv",
        config.derived_dir / "screening.csv",
        config.derived_dir / "manifest.json",
        config.results_dir / "sentiment_scores.jsonl",
        score_csv,
        signal_csv,
        decision_csv,
        price_csv,
        return_csv,
        summary_path,
        *charts,
    ]
    manifest = {
        "schema_version": 1,
        "run_id": config.run_id,
        "title": config.title,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "exploratory": True,
        "preregistered": False,
        "causal_claim": False,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_corpora": sorted(set(source_corpora)),
        "source_manifests": {
            "tavily": (
                {"path": str(config.tavily_package_manifest), "sha256": _sha256(config.tavily_package_manifest)}
                if config.tavily_package_manifest is not None
                else None
            ),
            "lseg": lseg_source_manifest,
        },
        "settings": {
            "dates": config.dates,
            "timezone": config.timezone,
            "symbols": [company.symbol for company in config.companies],
            "models": config.models,
            "primary_model": config.primary_model,
            "provider": config.provider,
            "baselines": config.baselines,
            "consensus_enabled": config.consensus_enabled,
            "prompt_id": config.prompt_id,
            "temperature": config.temperature,
            "max_completion_tokens": config.max_completion_tokens,
            "concurrency": config.concurrency,
            "ollama_host": config.ollama_host if config.provider == "ollama" else None,
            "ollama_keep_alive": config.ollama_keep_alive if config.provider == "ollama" else None,
            "ollama_think": config.ollama_think if config.provider == "ollama" else None,
            "structured_output": config.structured_output if config.provider == "ollama" else False,
            "model_digests": {
                scorer: next((row.model_digest for row in scores if row.scorer_id == scorer and row.model_digest), None)
                for scorer in config.models
            },
            "prompt_hashes": sorted({row.prompt_hash for row in scores if row.prompt_hash}),
            "request_hashes": sorted({row.request_sha256 for row in scores if row.request_sha256}),
            "horizons": config.horizons,
            "notional_usd": config.notional_usd,
            "entry_rule": "next_observed_session_adjusted_open",
            "price_source": "Yahoo Finance via yfinance",
            "decision_policy": asdict(config.decision_policy),
            "decision_policy_enabled": config.decision_policy_enabled,
            "index_fallback": (
                {"symbol": config.index_fallback.symbol, "min_texts": config.index_fallback.min_texts}
                if config.index_fallback is not None
                else None
            ),
        },
        "counts": {
            "articles_discovered": len(articles),
            "articles_accepted": sum(article.screening_decision == "include" for article in articles),
            "sentiment_scores": len(scores),
            "daily_signals": len(signals),
            "trading_decisions": len(decisions),
            "traded_decisions": sum(decision.action != "hold" for decision in decisions),
            "return_rows": len(returns),
            "index_fallback_events": len({(row.symbol, row.news_date) for row in returns if row.index_fallback}),
        },
        "environment": collect_run_environment(),
        "files": {str(path): _sha256(path) for path in files if path.exists()},
        "notes": [
            "Event returns overlap and are not combined into a funded portfolio.",
            "This exploratory pilot is not causal evidence or investment advice.",
        ],
    }
    (config.results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def register_completed_experiment(config: TradingStrategyConfig) -> None:
    registry = config.experiment_registry
    existing_text = registry.read_text(encoding="utf-8") if registry.exists() else ""
    try:
        existing = tomllib.loads(existing_text) if existing_text.strip() else {}
    except tomllib.TOMLDecodeError as exc:
        raise TradingStrategyError(f"cannot parse experiment registry {registry}: {exc}") from exc
    entries = existing.get("experiments") if isinstance(existing, dict) else None
    if isinstance(entries, list) and any(isinstance(entry, dict) and entry.get("id") == config.run_id for entry in entries):
        return
    environment = collect_run_environment()
    raw_git = environment.get("git")
    git: dict[str, Any] = raw_git if isinstance(raw_git, dict) else {}
    commit = str(git.get("commit") or "unknown")
    scorer_ids = [*config.models, *(f"baseline/{name}" for name in config.baselines)]
    if config.consensus_enabled:
        scorer_ids.append("consensus/majority")
    model_lines = ",\n  ".join(_toml_string(model) for model in scorer_ids)
    dates = ", ".join(_toml_string(value) for value in config.dates)
    horizons = ", ".join(str(value) for value in config.horizons)
    symbols = ", ".join(_toml_string(company.symbol) for company in config.companies)
    entry = f"""

[[experiments]]
id = {_toml_string(config.run_id)}
title = {_toml_string(config.title)}
status = "completed"
dissertation_relevance = "Reproducible news-sentiment trading research pipeline; non-causal."
commit_sha = {_toml_string(commit)}
export_path = {_toml_string(str(config.results_dir))}

[experiments.models]
ids = [
  {model_lines},
]
prompt_id = {_toml_string(config.prompt_id)}
provider = {_toml_string(config.provider)}
primary_model = {_toml_string(config.primary_model)}

[experiments.settings]
symbols = [{symbols}]
news_dates = [{dates}]
horizons = [{horizons}]
temperature = {config.temperature}
notional_usd = {config.notional_usd}
entry_rule = "next_observed_session_adjusted_open"
price_source = "Yahoo Finance via yfinance"
transaction_cost_bps_per_side = {config.decision_policy.transaction_cost_bps_per_side}
short_borrow_bps_per_day = {config.decision_policy.short_borrow_bps_per_day}
decision_policy_version = {_toml_string(config.decision_policy.policy_version)}

[experiments.analysis]
exploratory = true
preregistered = false
causal_claim = false
interpretation_notes = "Event-level and equal-weight mean returns only; overlapping events are not a funded portfolio."
"""
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(existing_text.rstrip() + entry, encoding="utf-8")


async def _resolve_model_digests(client: Any, models: tuple[str, ...], retries: int) -> dict[str, str | None]:
    resolver = getattr(client, "model_digest", None)
    if not callable(resolver):
        return {model: None for model in models}
    digests: dict[str, str | None] = {}
    for model in models:
        try:
            digests[model] = await resolver(model, retries=retries)
        except Exception as exc:
            raise TradingStrategyError(f"cannot resolve Ollama model digest for {model}: {exc}") from exc
    return digests


async def run_trading_strategy(
    config_path: str | Path,
    *,
    newsapi_client: Any,
    tavily_client: Any,
    llm_client: Any,
    price_loader: Callable[[TradingStrategyConfig], list[PriceRow]] = fetch_price_rows,
) -> TradingRunResult:
    path = Path(config_path)
    config = load_trading_config(path)
    completed_manifest = config.results_dir / "run_manifest.json"
    if completed_manifest.exists():
        try:
            completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            completed = None
        if isinstance(completed, dict) and completed.get("status") == "completed":
            if completed.get("config_sha256") != _sha256(path):
                raise TradingStrategyError(
                    f"completed run {config.run_id} used a different config; choose a new run.id instead of overwriting it"
                )
            register_completed_experiment(config)
            raw_counts = completed.get("counts")
            counts: dict[str, Any] = raw_counts if isinstance(raw_counts, dict) else {}
            return TradingRunResult(
                run_id=config.run_id,
                derived_dir=config.derived_dir,
                results_dir=config.results_dir,
                accepted_article_count=int(counts.get("articles_accepted", 0)),
                sentiment_score_count=int(counts.get("sentiment_scores", 0)),
                return_count=int(counts.get("return_rows", 0)),
                source_corpora=tuple(completed.get("source_corpora") or ()),
            )
    config.derived_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    tavily_candidates = load_tavily_candidates(config)
    newsapi_candidates, newsapi_corpora = await fetch_newsapi_candidates(config, newsapi_client)
    candidates = tavily_candidates + newsapi_candidates
    gap_candidates, gap_corpora = await fetch_tavily_gaps(config, tavily_client, candidates)
    candidates.extend(gap_candidates)
    web_articles = merge_article_candidates(candidates, timezone=config.timezone, selected_dates=config.dates)
    lseg_articles, lseg_corpora = load_lseg_articles(config)
    articles = sorted(
        [*web_articles, *lseg_articles],
        key=lambda row: (row.published_date_local, row.symbol, row.article_id),
    )
    articles = apply_screening_overrides(articles, config.screening_overrides_path)
    write_articles(config, articles)

    prompts = load_prompts(config.prompts_path)
    if config.prompt_id not in prompts:
        raise TradingStrategyError(f"prompt id not found: {config.prompt_id}")
    model_digests = await _resolve_model_digests(llm_client, config.models, config.retries)
    request_settings = {
        "provider": config.provider,
        "ollama_host": config.ollama_host if config.provider == "ollama" else None,
        "ollama_keep_alive": config.ollama_keep_alive if config.provider == "ollama" else None,
        "ollama_think": config.ollama_think if config.provider == "ollama" else None,
        "structured_output": config.structured_output if config.provider == "ollama" else False,
    }
    score_jsonl = config.results_dir / "sentiment_scores.jsonl"
    common_kwargs: dict[str, Any] = {
        "models": config.models,
        "prompt": prompts[config.prompt_id],
        "client": llm_client,
        "resume_path": config.resume_scores_path,
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "concurrency": config.concurrency,
        "retries": config.retries,
        "baselines": config.baselines,
        "model_digests": model_digests,
        "request_settings": request_settings,
        "strict_resume_identity": config.lseg_corpus_manifest is not None,
    }
    companies_by_symbol = {company.symbol: company for company in config.companies}
    masked_scores: list[SentimentScore] = []
    if config.masking_mode == "on":
        # Fully masked run: base scorer ids, anonymised text.
        masked_articles, masked_counts = _mask_articles(articles, companies_by_symbol)
        scores = await score_articles(
            masked_articles,
            output_path=score_jsonl,
            masking_mode="on",
            entity_masked=True,
            masked_token_counts=masked_counts,
            **common_kwargs,
        )
    else:
        scores = await score_articles(articles, output_path=score_jsonl, masking_mode=config.masking_mode, **common_kwargs)
        if config.masking_mode == "both":
            # Parallel masked arm under "#masked" scorer ids for the ablation.
            masked_articles, masked_counts = _mask_articles(articles, companies_by_symbol)
            masked_scores = await score_articles(
                masked_articles,
                output_path=config.results_dir / "sentiment_scores_masked.jsonl",
                scorer_suffix="#masked",
                masking_mode="both",
                entity_masked=True,
                masked_token_counts=masked_counts,
                **common_kwargs,
            )
    consensus = article_consensus(scores, config.models) if config.consensus_enabled else []
    all_scores = sorted(
        scores + masked_scores + consensus,
        key=lambda row: (row.news_date, row.symbol, row.article_id, row.scorer_id),
    )
    if config.cutoff.policy != "ignore":
        all_scores = annotate_scores_with_cutoff(all_scores, config)
    masked_arm = config.masking_mode == "both"
    scorer_ids = (
        *config.models,
        *(('consensus/majority',) if config.consensus_enabled else ()),
        *(f"baseline/{name}" for name in config.baselines),
        *((f"{model}#masked" for model in config.models) if masked_arm else ()),
        *((f"baseline/{name}#masked" for name in config.baselines) if masked_arm else ()),
    )
    # post_only restricts the *traded* signal to contamination-free scores; the
    # full annotated set is still written to outputs for auditing/stratification.
    signal_scores = (
        [score for score in all_scores if score.is_post_cutoff is not False]
        if config.cutoff.policy == "post_only"
        else all_scores
    )
    signals = daily_signals(
        articles,
        signal_scores,
        scorer_ids,
        config.dates,
        tuple(company.symbol for company in config.companies),
    )
    decisions = make_trading_decisions(signals, config.decision_policy)
    prices = price_loader(config)
    returns = calculate_returns(
        decisions if config.decision_policy_enabled else signals,
        prices,
        horizons=config.horizons,
        notional_usd=config.notional_usd,
        index_fallback=config.index_fallback,
        transaction_cost_bps_per_side=(
            config.decision_policy.transaction_cost_bps_per_side if config.decision_policy_enabled else 0.0
        ),
        short_borrow_bps_per_day=(
            config.decision_policy.short_borrow_bps_per_day if config.decision_policy_enabled else 0.0
        ),
        timezone=config.timezone,
    )
    source_corpora = sorted(
        {candidate.source_corpus for candidate in tavily_candidates}
        | set(newsapi_corpora)
        | set(gap_corpora)
        | set(lseg_corpora)
    )
    write_run_outputs(config, path, articles, all_scores, signals, decisions, prices, returns, source_corpora)
    register_completed_experiment(config)
    return TradingRunResult(
        run_id=config.run_id,
        derived_dir=config.derived_dir,
        results_dir=config.results_dir,
        accepted_article_count=sum(article.screening_decision == "include" for article in articles),
        sentiment_score_count=len(all_scores),
        return_count=len(returns),
        source_corpora=tuple(source_corpora),
    )


def describe_trading_plan(config: TradingStrategyConfig) -> dict[str, Any]:
    from_time, to_time = _utc_newsapi_bounds(config)
    return {
        "run_id": config.run_id,
        "companies": [company.symbol for company in config.companies],
        "dates": list(config.dates),
        "models": list(config.models),
        "provider": config.provider,
        "primary_model": config.primary_model,
        "baselines": list(config.baselines),
        "horizons": list(config.horizons),
        "newsapi_requests_before_pagination": len(config.companies) if config.newsapi_enabled else 0,
        "newsapi_from": from_time,
        "newsapi_to": to_time,
        "entry_rule": "next observed trading-session adjusted open",
        "lseg_corpus": str(config.lseg_corpus_manifest) if config.lseg_corpus_manifest else "disabled",
        "decision_policy": asdict(config.decision_policy) if config.decision_policy_enabled else "legacy signal rule",
        "index_fallback": (
            f"{config.index_fallback.symbol} when a company-day has fewer than {config.index_fallback.min_texts} texts"
            if config.index_fallback is not None
            else "disabled"
        ),
    }
