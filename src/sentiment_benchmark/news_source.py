from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import ParseResult, urlparse, urlunparse

from .env import load_env_file

NEWS_SCHEMA_VERSION = 1
DEFAULT_NEWS_OUTPUT_DIR = Path("Data/news")
DEFAULT_NEWS_TOPIC = "news"
DEFAULT_NEWS_TIME_RANGE = "week"
DEFAULT_NEWS_SEARCH_DEPTH = "basic"
DEFAULT_NEWS_EXTRACT_DEPTH = "basic"
DEFAULT_NEWS_EXTRACT_FORMAT = "text"
DEFAULT_NEWS_MAX_RESULTS = 10
MAX_TAVILY_RESULTS = 20
DEFAULT_MIN_ARTICLE_TEXT_CHARS = 500

# High-confidence openings of pages where the extractor was served an error,
# bot wall, or consent page instead of the article body.
ERROR_PAGE_SIGNATURES: tuple[str, ...] = (
    "oops, something went wrong",
    "are you a robot",
    "verify you are human",
    "access to this page has been denied",
    "access denied",
    "page not found",
    "404 not found",
    "please enable cookies",
    "please enable js",
    "enable javascript and cookies to continue",
    "javascript is disabled",
    "your browser is out of date",
)
ERROR_PAGE_SCAN_CHARS = 400

TEXT_QUALITY_OK = "ok"
TEXT_QUALITY_ERROR_PAGE = "error_page"
TEXT_QUALITY_TOO_SHORT = "too_short"
TEXT_QUALITY_MISSING = "missing"

NewsTopic = Literal["news", "finance", "general"]
NewsTimeRange = Literal["day", "week", "month", "year"]
NewsSearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]
NewsExtractDepth = Literal["basic", "advanced"]
NewsExtractFormat = Literal["text", "markdown"]

NEWS_TOPICS: tuple[NewsTopic, ...] = ("news", "finance", "general")
NEWS_TIME_RANGES: tuple[NewsTimeRange, ...] = ("day", "week", "month", "year")
NEWS_SEARCH_DEPTHS: tuple[NewsSearchDepth, ...] = ("basic", "advanced", "fast", "ultra-fast")
NEWS_EXTRACT_DEPTHS: tuple[NewsExtractDepth, ...] = ("basic", "advanced")
NEWS_EXTRACT_FORMATS: tuple[NewsExtractFormat, ...] = ("text", "markdown")


class TavilyNewsError(RuntimeError):
    """Base class for Tavily news sourcing errors."""


class TavilyNewsConfigurationError(TavilyNewsError):
    """Raised when Tavily credentials or configuration are missing."""


class TavilyNewsDependencyError(TavilyNewsError):
    """Raised when the Tavily SDK package is unavailable."""


@dataclass(frozen=True)
class NewsFetchConfig:
    query: str
    max_results: int = DEFAULT_NEWS_MAX_RESULTS
    topic: NewsTopic = "news"
    time_range: NewsTimeRange | None = "week"
    search_depth: NewsSearchDepth = "basic"
    extract: bool = True
    extract_depth: NewsExtractDepth = "basic"
    extract_format: NewsExtractFormat = "text"
    start_date: str | None = None
    end_date: str | None = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    min_text_chars: int = DEFAULT_MIN_ARTICLE_TEXT_CHARS


@dataclass(frozen=True)
class NewsArticleRecord:
    record_id: str
    url: str
    normalized_url: str
    title: str | None = None
    snippet: str | None = None
    article_text: str | None = None
    published_date: str | None = None
    published_date_source: str = ""
    score: float | None = None
    favicon: str | None = None
    extraction_status: str = "skipped"
    text_quality: str = TEXT_QUALITY_MISSING
    extract_error: str | None = None
    search_rank: int | None = None
    search_request_id: str | None = None
    extract_request_id: str | None = None
    search_usage: dict[str, Any] = field(default_factory=dict)
    extract_usage: dict[str, Any] = field(default_factory=dict)
    raw_search_result: dict[str, Any] = field(default_factory=dict)
    raw_extract_result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NewsFetchResult:
    config: NewsFetchConfig
    fetched_at: str
    records: list[NewsArticleRecord]
    search_request_id: str | None = None
    extract_request_id: str | None = None
    response_time: float | None = None
    search_usage: dict[str, Any] = field(default_factory=dict)
    extract_usage: dict[str, Any] = field(default_factory=dict)
    failed_extractions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class NewsOutputPaths:
    output_dir: Path
    articles_jsonl: Path
    articles_csv: Path
    manifest_json: Path


def _import_async_tavily_client() -> type:
    try:
        from tavily import AsyncTavilyClient
    except ImportError as exc:  # pragma: no cover - dependency absence is tested through constructor behavior
        raise TavilyNewsDependencyError(
            "The 'tavily-python' package is required for Tavily news sourcing. "
            "Install project dependencies or run: python -m pip install tavily-python"
        ) from exc
    return AsyncTavilyClient


def _get_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _to_jsonable(as_dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return repr(value)


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalize_domains(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    cleaned = []
    for value in values or []:
        item = value.strip().lower()
        if item:
            cleaned.append(item)
    return tuple(dict.fromkeys(cleaned))


def _validate_date(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format") from exc
    return value


def _validate_literal(value: str, allowed: tuple[str, ...], *, name: str) -> str:
    value = value.strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    return value


def make_news_fetch_config(
    *,
    query: str,
    max_results: int = DEFAULT_NEWS_MAX_RESULTS,
    topic: str = DEFAULT_NEWS_TOPIC,
    time_range: str | None = DEFAULT_NEWS_TIME_RANGE,
    search_depth: str = DEFAULT_NEWS_SEARCH_DEPTH,
    extract: bool = True,
    extract_depth: str = DEFAULT_NEWS_EXTRACT_DEPTH,
    extract_format: str = DEFAULT_NEWS_EXTRACT_FORMAT,
    start_date: str | None = None,
    end_date: str | None = None,
    include_domains: list[str] | tuple[str, ...] | None = None,
    exclude_domains: list[str] | tuple[str, ...] | None = None,
    min_text_chars: int = DEFAULT_MIN_ARTICLE_TEXT_CHARS,
) -> NewsFetchConfig:
    query = query.strip()
    if not query:
        raise ValueError("query is required")
    if max_results < 1 or max_results > MAX_TAVILY_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_TAVILY_RESULTS}")
    if min_text_chars < 0:
        raise ValueError("min_text_chars must be 0 or greater")
    resolved_topic = _validate_literal(topic, NEWS_TOPICS, name="topic")
    resolved_time_range = None
    if time_range is not None and time_range.strip():
        resolved_time_range = _validate_literal(time_range, NEWS_TIME_RANGES, name="time_range")
    resolved_search_depth = _validate_literal(search_depth, NEWS_SEARCH_DEPTHS, name="search_depth")
    resolved_extract_depth = _validate_literal(extract_depth, NEWS_EXTRACT_DEPTHS, name="extract_depth")
    resolved_extract_format = _validate_literal(extract_format, NEWS_EXTRACT_FORMATS, name="extract_format")
    return NewsFetchConfig(
        query=query,
        max_results=max_results,
        topic=resolved_topic,  # type: ignore[arg-type]
        time_range=resolved_time_range,  # type: ignore[arg-type]
        search_depth=resolved_search_depth,  # type: ignore[arg-type]
        extract=extract,
        extract_depth=resolved_extract_depth,  # type: ignore[arg-type]
        extract_format=resolved_extract_format,  # type: ignore[arg-type]
        start_date=_validate_date(start_date, name="start_date"),
        end_date=_validate_date(end_date, name="end_date"),
        include_domains=_normalize_domains(include_domains),
        exclude_domains=_normalize_domains(exclude_domains),
        min_text_chars=min_text_chars,
    )


def assess_article_text(
    article_text: str | None,
    *,
    min_chars: int = DEFAULT_MIN_ARTICLE_TEXT_CHARS,
) -> tuple[str, str | None]:
    """Classify extracted text quality; returns (quality, detail)."""
    if not article_text or not article_text.strip():
        return TEXT_QUALITY_MISSING, None
    head = re.sub(r"\s+", " ", article_text[:ERROR_PAGE_SCAN_CHARS]).strip().lower()
    for signature in ERROR_PAGE_SIGNATURES:
        if signature in head:
            return TEXT_QUALITY_ERROR_PAGE, f'extracted text matches error-page signature: "{signature}"'
    stripped_length = len(article_text.strip())
    if min_chars and stripped_length < min_chars:
        return TEXT_QUALITY_TOO_SHORT, f"extracted text has {stripped_length} characters (minimum {min_chars})"
    return TEXT_QUALITY_OK, None


_MONTH_NAME_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_URL_DATE_PATTERNS = (
    re.compile(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)"),
    re.compile(r"(?:^|[/_-])(20\d{2})-(\d{1,2})-(\d{1,2})(?:[/_.-]|$)"),
)
_TEXT_DATE_ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_TEXT_DATE_MDY = re.compile(rf"\b({_MONTH_NAME_PATTERN})\.?\s+(\d{{1,2}}),?\s+(20\d{{2}})\b", re.IGNORECASE)
_TEXT_DATE_DMY = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_NAME_PATTERN})\.?,?\s+(20\d{{2}})\b", re.IGNORECASE)
_MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
PUBLISHED_DATE_TEXT_SCAN_CHARS = 2000


def _iso_date_or_none(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def infer_published_date(url: str, article_text: str | None) -> tuple[str | None, str]:
    """Best-effort publication date recovery; returns (ISO date, source) where source is 'url' or 'text'."""
    path = urlparse(url).path
    for pattern in _URL_DATE_PATTERNS:
        match = pattern.search(path)
        if match:
            inferred = _iso_date_or_none(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if inferred:
                return inferred, "url"
    if article_text:
        head = article_text[:PUBLISHED_DATE_TEXT_SCAN_CHARS]
        match = _TEXT_DATE_ISO.search(head)
        if match:
            inferred = _iso_date_or_none(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if inferred:
                return inferred, "text"
        match = _TEXT_DATE_MDY.search(head)
        if match:
            month = _MONTH_NUMBERS[match.group(1)[:3].lower()]
            inferred = _iso_date_or_none(int(match.group(3)), month, int(match.group(2)))
            if inferred:
                return inferred, "text"
        match = _TEXT_DATE_DMY.search(head)
        if match:
            month = _MONTH_NUMBERS[match.group(2)[:3].lower()]
            inferred = _iso_date_or_none(int(match.group(3)), month, int(match.group(1)))
            if inferred:
                return inferred, "text"
    return None, ""


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    normalized = ParseResult(
        scheme=parsed.scheme.lower(),
        netloc=netloc,
        path=path,
        params="",
        query=parsed.query,
        fragment="",
    )
    return urlunparse(normalized)


def article_record_id(normalized_url: str) -> str:
    return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:16]


def _slugify(value: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:limit].strip("-") or "news"


def _preview(value: str | None, limit: int = 500) -> str:
    if not value:
        return ""
    collapsed = re.sub(r"\s+", " ", value).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _result_list(payload: Any, field_name: str) -> list[Any]:
    values = _get_field(payload, field_name, [])
    return list(values or []) if isinstance(values, list | tuple) else []


def _result_by_url(payload: Any) -> dict[str, dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    for item in _result_list(payload, "results"):
        raw = _to_jsonable(item)
        url = _get_field(item, "url")
        if isinstance(url, str) and url:
            by_url[normalize_url(url)] = raw if isinstance(raw, dict) else {"raw": raw}
    return by_url


def _failed_by_url(payload: Any) -> dict[str, dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    for item in _result_list(payload, "failed_results"):
        raw = _to_jsonable(item)
        url = _get_field(item, "url")
        if isinstance(url, str) and url:
            by_url[normalize_url(url)] = raw if isinstance(raw, dict) else {"raw": raw}
    return by_url


class TavilyNewsClient:
    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        load_env_file()
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.project_id = project_id or os.getenv("TAVILY_PROJECT") or None
        self._client = client
        self._owns_client = client is None
        if self._client is None and not self.api_key:
            raise TavilyNewsConfigurationError("TAVILY_API_KEY is required for Tavily news sourcing")

    async def __aenter__(self) -> TavilyNewsClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is None or not self._owns_client:
            return
        closer = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if callable(closer):
            result = closer()
            if hasattr(result, "__await__"):
                await result

    def _get_client(self) -> Any:
        if self._client is None:
            async_client = _import_async_tavily_client()
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.project_id:
                kwargs["project_id"] = self.project_id
            try:
                self._client = async_client(**kwargs)
            except TypeError:
                if self.project_id:
                    self._client = async_client(self.api_key, project_id=self.project_id)
                else:
                    self._client = async_client(self.api_key)
        return self._client

    async def check(self, query: str = "financial markets", max_results: int = 1) -> NewsFetchResult:
        config = make_news_fetch_config(query=query, max_results=max_results, extract=False)
        return await self.fetch(config)

    async def fetch(self, config: NewsFetchConfig) -> NewsFetchResult:
        client = self._get_client()
        search_kwargs: dict[str, Any] = {
            "query": config.query,
            "topic": config.topic,
            "search_depth": config.search_depth,
            "max_results": config.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_favicon": True,
        }
        if config.time_range:
            search_kwargs["time_range"] = config.time_range
        if config.start_date:
            search_kwargs["start_date"] = config.start_date
        if config.end_date:
            search_kwargs["end_date"] = config.end_date
        if config.include_domains:
            search_kwargs["include_domains"] = list(config.include_domains)
        if config.exclude_domains:
            search_kwargs["exclude_domains"] = list(config.exclude_domains)

        search_payload = await client.search(**search_kwargs)
        search_request_id = _get_field(search_payload, "request_id")
        search_usage = _get_field(search_payload, "usage", {}) or {}
        response_time = _as_optional_float(_get_field(search_payload, "response_time"))

        search_results = _result_list(search_payload, "results")
        unique_results: list[tuple[int, Any, dict[str, Any], str, str]] = []
        seen_urls: set[str] = set()
        for rank, item in enumerate(search_results, start=1):
            url = _get_field(item, "url")
            if not isinstance(url, str) or not url.strip():
                continue
            normalized = normalize_url(url)
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)
            raw_item = _to_jsonable(item)
            unique_results.append((rank, item, raw_item if isinstance(raw_item, dict) else {"raw": raw_item}, url, normalized))

        extract_payload = None
        extract_request_id = None
        extract_usage: dict[str, Any] = {}
        extract_results: dict[str, dict[str, Any]] = {}
        extract_failures: dict[str, dict[str, Any]] = {}
        if config.extract and unique_results:
            extract_payload = await client.extract(
                urls=[url for _, _, _, url, _ in unique_results],
                extract_depth=config.extract_depth,
                format=config.extract_format,
                include_favicon=True,
                include_usage=True,
            )
            extract_request_id = _get_field(extract_payload, "request_id")
            extract_usage = _get_field(extract_payload, "usage", {}) or {}
            extract_results = _result_by_url(extract_payload)
            extract_failures = _failed_by_url(extract_payload)

        records: list[NewsArticleRecord] = []
        for rank, item, raw_item, url, normalized in unique_results:
            extracted = extract_results.get(normalized, {})
            failed = extract_failures.get(normalized, {})
            article_text = extracted.get("raw_content") if isinstance(extracted.get("raw_content"), str) else None
            text_quality, quality_detail = assess_article_text(article_text, min_chars=config.min_text_chars)
            if article_text and text_quality == TEXT_QUALITY_OK:
                extraction_status = "success"
                extract_error = None
            elif article_text:
                # The extractor returned content, but it is an error page or
                # too short to be an article body; keep it only in
                # raw_extract_result so it never looks like usable text.
                extraction_status = "failed"
                extract_error = quality_detail
                article_text = None
            elif failed:
                extraction_status = "failed"
                extract_error = str(failed.get("error") or "Extraction failed")
            else:
                extraction_status = "skipped" if not config.extract else "failed"
                extract_error = "No extracted content returned" if config.extract else None

            published_date = _get_field(item, "published_date") or _get_field(item, "publishedDate")
            published_date_source = "search" if published_date else ""
            if not published_date:
                inferred_date, inferred_source = infer_published_date(url, article_text)
                if inferred_date:
                    published_date = inferred_date
                    published_date_source = inferred_source
            records.append(
                NewsArticleRecord(
                    record_id=article_record_id(normalized),
                    url=url,
                    normalized_url=normalized,
                    title=_get_field(item, "title"),
                    snippet=_get_field(item, "content"),
                    article_text=article_text,
                    published_date=published_date,
                    published_date_source=published_date_source,
                    score=_as_optional_float(_get_field(item, "score")),
                    favicon=_get_field(item, "favicon") or extracted.get("favicon"),
                    extraction_status=extraction_status,
                    text_quality=text_quality,
                    extract_error=extract_error,
                    search_rank=rank,
                    search_request_id=search_request_id if isinstance(search_request_id, str) else None,
                    extract_request_id=extract_request_id if isinstance(extract_request_id, str) else None,
                    search_usage=search_usage if isinstance(search_usage, dict) else {},
                    extract_usage=extract_usage if isinstance(extract_usage, dict) else {},
                    raw_search_result=raw_item,
                    raw_extract_result=extracted,
                )
            )

        return NewsFetchResult(
            config=config,
            fetched_at=datetime.now(UTC).isoformat(),
            records=records,
            search_request_id=search_request_id if isinstance(search_request_id, str) else None,
            extract_request_id=extract_request_id if isinstance(extract_request_id, str) else None,
            response_time=response_time,
            search_usage=search_usage if isinstance(search_usage, dict) else {},
            extract_usage=extract_usage if isinstance(extract_usage, dict) else {},
            failed_extractions=list(extract_failures.values()),
        )


def write_news_corpus(result: NewsFetchResult, output_root: str | Path = DEFAULT_NEWS_OUTPUT_DIR) -> NewsOutputPaths:
    output_root = Path(output_root)
    timestamp = result.fetched_at.replace("+00:00", "Z").replace(":", "").replace("-", "")
    timestamp = timestamp.split(".")[0]
    directory_name = f"tavily_news_{timestamp}_{_slugify(result.config.query)}"
    output_dir = output_root / directory_name
    suffix = 1
    while output_dir.exists():
        suffix += 1
        output_dir = output_root / f"{directory_name}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)

    jsonl_path = output_dir / "articles.jsonl"
    csv_path = output_dir / "articles.csv"
    manifest_path = output_dir / "manifest.json"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in result.records:
            file.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "record_id",
            "title",
            "url",
            "published_date",
            "published_date_source",
            "score",
            "extraction_status",
            "text_quality",
            "extract_error",
            "snippet_preview",
            "article_text_preview",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in result.records:
            writer.writerow(
                {
                    "record_id": record.record_id,
                    "title": record.title or "",
                    "url": record.url,
                    "published_date": record.published_date or "",
                    "published_date_source": record.published_date_source,
                    "score": record.score if record.score is not None else "",
                    "extraction_status": record.extraction_status,
                    "text_quality": record.text_quality,
                    "extract_error": record.extract_error or "",
                    "snippet_preview": _preview(record.snippet),
                    "article_text_preview": _preview(record.article_text),
                }
            )

    failed_count = sum(1 for record in result.records if record.extraction_status == "failed")
    text_quality_counts = dict(sorted(Counter(record.text_quality for record in result.records).items()))
    manifest = {
        "schema_version": NEWS_SCHEMA_VERSION,
        "source": "tavily",
        "fetched_at": result.fetched_at,
        "query": result.config.query,
        "parameters": asdict(result.config),
        "record_count": len(result.records),
        "usable_record_count": text_quality_counts.get(TEXT_QUALITY_OK, 0),
        "text_quality_counts": text_quality_counts,
        "failed_extraction_count": failed_count,
        "search_request_id": result.search_request_id,
        "extract_request_id": result.extract_request_id,
        "response_time": result.response_time,
        "search_usage": result.search_usage,
        "extract_usage": result.extract_usage,
        "files": {
            "articles_jsonl": jsonl_path.name,
            "articles_csv": csv_path.name,
            "manifest_json": manifest_path.name,
        },
        "notes": [
            "Records are unlabeled source material and are not benchmark rows.",
            "Data/data.csv is not modified by this workflow.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return NewsOutputPaths(
        output_dir=output_dir,
        articles_jsonl=jsonl_path,
        articles_csv=csv_path,
        manifest_json=manifest_path,
    )
