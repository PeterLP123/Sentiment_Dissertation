from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .env import load_env_file
from .news_source import (
    DEFAULT_NEWS_OUTPUT_DIR,
    TEXT_QUALITY_MISSING,
    TEXT_QUALITY_SNIPPET,
    NewsArticleRecord,
    NewsOutputPaths,
    article_record_id,
    normalize_url,
)

NEWSAPI_SCHEMA_VERSION = 1
NEWSAPI_BASE_URL = "https://newsapi.org/v2"
NEWSAPI_PAGE_SIZE = 100
NEWSAPI_MAX_PAGES = 10
NEWSAPI_TIMEOUT_SECONDS = 30.0
NEWSAPI_RETRIES = 3


class NewsApiError(RuntimeError):
    """Base error raised by NewsAPI sourcing."""


class NewsApiConfigurationError(NewsApiError):
    """Raised when NewsAPI credentials or configuration are missing."""


@dataclass(frozen=True)
class NewsApiFetchConfig:
    query: str
    from_time: str | None = None
    to_time: str | None = None
    language: str = "en"
    sort_by: str = "publishedAt"
    page_size: int = NEWSAPI_PAGE_SIZE
    max_pages: int = NEWSAPI_MAX_PAGES
    domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class NewsApiFetchResult:
    config: NewsApiFetchConfig
    fetched_at: str
    records: list[NewsArticleRecord]
    total_results: int
    pages_fetched: int


def _validate_iso8601(value: str | None, *, name: str) -> str | None:
    if value is None or not value.strip():
        return None
    cleaned = value.strip()
    try:
        datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 date or timestamp") from exc
    return cleaned


def _normalize_domains(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip().lower() for value in values or () if value.strip()))


def make_newsapi_fetch_config(
    *,
    query: str,
    from_time: str | None = None,
    to_time: str | None = None,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = NEWSAPI_PAGE_SIZE,
    max_pages: int = NEWSAPI_MAX_PAGES,
    domains: list[str] | tuple[str, ...] | None = None,
    exclude_domains: list[str] | tuple[str, ...] | None = None,
) -> NewsApiFetchConfig:
    query = query.strip()
    if not query:
        raise ValueError("query is required")
    if len(query) > 500:
        raise ValueError("query must be at most 500 characters")
    if page_size < 1 or page_size > NEWSAPI_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {NEWSAPI_PAGE_SIZE}")
    if max_pages < 1:
        raise ValueError("max_pages must be 1 or greater")
    if sort_by not in {"publishedAt", "relevancy", "popularity"}:
        raise ValueError("sort_by must be one of: publishedAt, relevancy, popularity")
    language = language.strip().lower()
    if len(language) != 2:
        raise ValueError("language must be a two-letter ISO-639-1 code")
    resolved_from = _validate_iso8601(from_time, name="from_time")
    resolved_to = _validate_iso8601(to_time, name="to_time")
    if resolved_from and resolved_to:
        start = datetime.fromisoformat(resolved_from.replace("Z", "+00:00"))
        end = datetime.fromisoformat(resolved_to.replace("Z", "+00:00"))
        if start > end:
            raise ValueError("from_time must be before or equal to to_time")
    return NewsApiFetchConfig(
        query=query,
        from_time=resolved_from,
        to_time=resolved_to,
        language=language,
        sort_by=sort_by,
        page_size=page_size,
        max_pages=max_pages,
        domains=_normalize_domains(domains),
        exclude_domains=_normalize_domains(exclude_domains),
    )


def _as_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


class NewsApiClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = NEWSAPI_BASE_URL,
        client: httpx.AsyncClient | None = None,
        timeout: float = NEWSAPI_TIMEOUT_SECONDS,
        retries: int = NEWSAPI_RETRIES,
    ) -> None:
        load_env_file()
        self.api_key = api_key or os.getenv("NEWSAPI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._client = client
        self._owns_client = client is None
        if self._client is None and not self.api_key:
            raise NewsApiConfigurationError("NEWSAPI_API_KEY is required for NewsAPI sourcing")

    async def __aenter__(self) -> NewsApiClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.retries + 1):
            try:
                response = await self._get_client().get(
                    f"{self.base_url}/everything",
                    params=params,
                    headers={"X-Api-Key": self.api_key or ""},
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise NewsApiError("NewsAPI returned a non-object response")
                if payload.get("status") != "ok":
                    code = payload.get("code") or "api_error"
                    message = payload.get("message") or "NewsAPI request failed"
                    raise NewsApiError(f"{code}: {message}")
                return payload
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(min(2**attempt, 8))
        raise NewsApiError("NewsAPI retry loop ended unexpectedly")

    async def check(self, query: str = "financial markets") -> NewsApiFetchResult:
        config = make_newsapi_fetch_config(query=query, page_size=1, max_pages=1)
        return await self.fetch(config)

    async def fetch(self, config: NewsApiFetchConfig) -> NewsApiFetchResult:
        base_params: dict[str, Any] = {
            "q": config.query,
            "language": config.language,
            "sortBy": config.sort_by,
            "pageSize": config.page_size,
        }
        if config.from_time:
            base_params["from"] = config.from_time
        if config.to_time:
            base_params["to"] = config.to_time
        if config.domains:
            base_params["domains"] = ",".join(config.domains)
        if config.exclude_domains:
            base_params["excludeDomains"] = ",".join(config.exclude_domains)

        records: list[NewsArticleRecord] = []
        seen_urls: set[str] = set()
        total_results = 0
        pages_fetched = 0
        for page in range(1, config.max_pages + 1):
            payload = await self._get({**base_params, "page": page})
            pages_fetched += 1
            raw_total = payload.get("totalResults")
            total_results = raw_total if isinstance(raw_total, int) else total_results
            articles = payload.get("articles")
            if not isinstance(articles, list):
                articles = []
            for rank, item in enumerate(articles, start=(page - 1) * config.page_size + 1):
                if not isinstance(item, dict):
                    continue
                url = _as_string(item.get("url"))
                if not url:
                    continue
                normalized = normalize_url(url)
                if normalized in seen_urls:
                    continue
                seen_urls.add(normalized)
                title = _as_string(item.get("title"))
                description = _as_string(item.get("description"))
                records.append(
                    NewsArticleRecord(
                        record_id=article_record_id(normalized),
                        url=url,
                        normalized_url=normalized,
                        title=title,
                        snippet=description,
                        article_text=None,
                        published_date=_as_string(item.get("publishedAt")),
                        published_date_source="search" if item.get("publishedAt") else "",
                        extraction_status="skipped",
                        text_quality=TEXT_QUALITY_SNIPPET if title or description else TEXT_QUALITY_MISSING,
                        search_rank=rank,
                        raw_search_result=item,
                    )
                )
            if not articles or page * config.page_size >= total_results:
                break

        return NewsApiFetchResult(
            config=config,
            fetched_at=datetime.now(UTC).isoformat(),
            records=records,
            total_results=total_results,
            pages_fetched=pages_fetched,
        )


def _slugify(value: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:limit].strip("-") or "news"


def _preview(value: str | None, limit: int = 500) -> str:
    collapsed = re.sub(r"\s+", " ", value or "").strip()
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def write_newsapi_corpus(
    result: NewsApiFetchResult,
    output_root: str | Path = DEFAULT_NEWS_OUTPUT_DIR,
) -> NewsOutputPaths:
    root = Path(output_root)
    timestamp = result.fetched_at.replace("+00:00", "Z").replace(":", "").replace("-", "").split(".")[0]
    base_name = f"newsapi_news_{timestamp}_{_slugify(result.config.query)}"
    output_dir = root / base_name
    suffix = 1
    while output_dir.exists():
        suffix += 1
        output_dir = root / f"{base_name}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)

    jsonl_path = output_dir / "articles.jsonl"
    csv_path = output_dir / "articles.csv"
    manifest_path = output_dir / "manifest.json"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in result.records:
            file.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["record_id", "title", "url", "published_date", "source_name", "text_quality", "snippet_preview"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in result.records:
            source = record.raw_search_result.get("source")
            source_name = source.get("name") if isinstance(source, dict) else ""
            writer.writerow(
                {
                    "record_id": record.record_id,
                    "title": record.title or "",
                    "url": record.url,
                    "published_date": record.published_date or "",
                    "source_name": source_name or "",
                    "text_quality": record.text_quality,
                    "snippet_preview": _preview(record.snippet),
                }
            )
    manifest = {
        "schema_version": NEWSAPI_SCHEMA_VERSION,
        "source": "newsapi",
        "fetched_at": result.fetched_at,
        "query": result.config.query,
        "parameters": asdict(result.config),
        "record_count": len(result.records),
        "total_results": result.total_results,
        "pages_fetched": result.pages_fetched,
        "files": {
            "articles_jsonl": jsonl_path.name,
            "articles_csv": csv_path.name,
            "manifest_json": manifest_path.name,
        },
        "notes": [
            "NewsAPI supplies titles, descriptions, and discovery metadata rather than full article bodies.",
            "Records are unlabeled source material and do not modify the benchmark dataset.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return NewsOutputPaths(output_dir, jsonl_path, csv_path, manifest_path)
