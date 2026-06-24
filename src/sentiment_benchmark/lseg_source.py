from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import inspect
import math
import re
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    directory_digest,
    read_json,
    sha256_file,
    sha256_text,
)
from .runtime_metadata import collect_run_environment

LSEG_RAW_SCHEMA_VERSION = 1
DEFAULT_LSEG_RAW_ROOT = Path("Data/news")
DEFAULT_LSEG_DERIVED_ROOT = Path("Data/derived/lseg")
DEFAULT_LSEG_PAGE_SIZE = 100
DEFAULT_LSEG_MAX_PAGES = 50
DEFAULT_LSEG_STORY_CONCURRENCY = 4
DEFAULT_LSEG_RETRIES = 3
DEFAULT_LSEG_MIN_TEXT_CHARS = 100
DEFAULT_LSEG_MAX_SCORING_CHARS = 8_000


class LsegNewsError(RuntimeError):
    """Base error for LSEG collection and corpus operations."""


class LsegConfigurationError(LsegNewsError):
    """Raised when an LSEG collection definition is invalid."""


class LsegDependencyError(LsegNewsError):
    """Raised when the optional LSEG SDK is not installed."""


@dataclass(frozen=True)
class LsegCompanyConfig:
    symbol: str
    name: str
    ric: str
    news_query: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class LsegCollectionConfig:
    collection_id: str
    start: str
    end: str
    language: str = "en"
    session: str = "desktop"
    page_size: int = DEFAULT_LSEG_PAGE_SIZE
    max_pages: int = DEFAULT_LSEG_MAX_PAGES
    story_concurrency: int = DEFAULT_LSEG_STORY_CONCURRENCY
    retries: int = DEFAULT_LSEG_RETRIES
    min_text_chars: int = DEFAULT_LSEG_MIN_TEXT_CHARS
    max_scoring_chars: int = DEFAULT_LSEG_MAX_SCORING_CHARS
    raw_output_root: Path = DEFAULT_LSEG_RAW_ROOT
    derived_output_root: Path = DEFAULT_LSEG_DERIVED_ROOT
    companies: tuple[LsegCompanyConfig, ...] = ()

    @property
    def raw_dir(self) -> Path:
        return self.raw_output_root / f"lseg_{self.collection_id}"

    @property
    def derived_dir(self) -> Path:
        return self.derived_output_root / self.collection_id

    def to_payload(self) -> dict[str, Any]:
        return {
            "collection": {
                "id": self.collection_id,
                "start": self.start,
                "end": self.end,
                "language": self.language,
                "session": self.session,
                "page_size": self.page_size,
                "max_pages": self.max_pages,
                "story_concurrency": self.story_concurrency,
                "retries": self.retries,
            },
            "cleaning": {
                "min_text_chars": self.min_text_chars,
                "max_scoring_chars": self.max_scoring_chars,
            },
            "outputs": {
                "raw_output_root": str(self.raw_output_root),
                "derived_output_root": str(self.derived_output_root),
            },
            "companies": [asdict(company) for company in self.companies],
        }

    @property
    def config_sha256(self) -> str:
        return sha256_text(canonical_json(self.to_payload()))


@dataclass(frozen=True)
class LsegHeadlinePage:
    rows: list[dict[str, Any]]
    next_cursor: str | None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LsegStoryResponse:
    story_id: str
    status: str
    body: str | None = None
    body_format: str | None = None
    web_url: str | None = None
    error: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LsegFetchResult:
    raw_dir: Path
    manifest_path: Path
    headline_count: int
    story_count: int
    failed_story_count: int
    resumed: bool = False


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_utc(value: str, *, name: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise LsegConfigurationError(f"{name} must be an ISO 8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise LsegConfigurationError(f"{name} must include the UTC offset Z or +00:00")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _nonempty(value: Any, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise LsegConfigurationError(f"{name} is required")
    return text


def _int_setting(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LsegConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise LsegConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def config_from_payload(payload: dict[str, Any]) -> LsegCollectionConfig:
    collection = payload.get("collection") or {}
    cleaning = payload.get("cleaning") or {}
    outputs = payload.get("outputs") or {}
    company_rows = payload.get("companies") or payload.get("company") or []
    if not isinstance(collection, dict) or not isinstance(cleaning, dict) or not isinstance(outputs, dict):
        raise LsegConfigurationError("collection, cleaning, and outputs must be TOML tables")
    if not isinstance(company_rows, list):
        raise LsegConfigurationError("companies must be an array of TOML tables")

    collection_id = _nonempty(collection.get("id"), name="collection.id")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", collection_id):
        raise LsegConfigurationError("collection.id must contain only letters, numbers, dots, underscores, or hyphens")
    start = _parse_utc(_nonempty(collection.get("start"), name="collection.start"), name="collection.start")
    end = _parse_utc(_nonempty(collection.get("end"), name="collection.end"), name="collection.end")
    if datetime.fromisoformat(start.replace("Z", "+00:00")) >= datetime.fromisoformat(end.replace("Z", "+00:00")):
        raise LsegConfigurationError("collection.start must be before collection.end")
    session = str(collection.get("session", "desktop")).strip().lower()
    if session != "desktop":
        raise LsegConfigurationError("version one supports only collection.session = 'desktop'")
    language = str(collection.get("language", "en")).strip().lower()
    if language not in {"en", "eng", "english"}:
        raise LsegConfigurationError("version one supports only English LSEG stories")

    companies: list[LsegCompanyConfig] = []
    for index, row in enumerate(company_rows):
        if not isinstance(row, dict):
            raise LsegConfigurationError(f"companies[{index}] must be a TOML table")
        symbol = _nonempty(row.get("symbol"), name=f"companies[{index}].symbol").upper()
        ric = _nonempty(row.get("ric"), name=f"companies[{index}].ric")
        raw_aliases = row.get("aliases") or []
        if not isinstance(raw_aliases, list) or not all(isinstance(alias, str) for alias in raw_aliases):
            raise LsegConfigurationError(f"companies[{index}].aliases must be a list of strings")
        aliases = tuple(dict.fromkeys(alias.strip() for alias in raw_aliases if alias.strip()))
        if not aliases:
            aliases = (symbol,)
        news_query = str(row.get("news_query") or f"R:{ric} and Language:LEN").strip()
        companies.append(
            LsegCompanyConfig(
                symbol=symbol,
                name=_nonempty(row.get("name"), name=f"companies[{index}].name"),
                ric=ric,
                news_query=news_query,
                aliases=aliases,
            )
        )
    if not companies:
        raise LsegConfigurationError("at least one [[companies]] entry is required")
    if len({company.symbol for company in companies}) != len(companies):
        raise LsegConfigurationError("company symbols must be unique")

    return LsegCollectionConfig(
        collection_id=collection_id,
        start=start,
        end=end,
        language="en",
        session=session,
        page_size=_int_setting(
            collection.get("page_size", DEFAULT_LSEG_PAGE_SIZE),
            name="collection.page_size",
            minimum=1,
            maximum=100,
        ),
        max_pages=_int_setting(
            collection.get("max_pages", DEFAULT_LSEG_MAX_PAGES),
            name="collection.max_pages",
            minimum=1,
            maximum=50,
        ),
        story_concurrency=_int_setting(
            collection.get("story_concurrency", DEFAULT_LSEG_STORY_CONCURRENCY),
            name="collection.story_concurrency",
            minimum=1,
            maximum=16,
        ),
        retries=_int_setting(
            collection.get("retries", DEFAULT_LSEG_RETRIES),
            name="collection.retries",
            minimum=0,
            maximum=10,
        ),
        min_text_chars=_int_setting(
            cleaning.get("min_text_chars", DEFAULT_LSEG_MIN_TEXT_CHARS),
            name="cleaning.min_text_chars",
            minimum=0,
            maximum=100_000,
        ),
        max_scoring_chars=_int_setting(
            cleaning.get("max_scoring_chars", DEFAULT_LSEG_MAX_SCORING_CHARS),
            name="cleaning.max_scoring_chars",
            minimum=1,
            maximum=1_000_000,
        ),
        raw_output_root=Path(outputs.get("raw_output_root", DEFAULT_LSEG_RAW_ROOT)),
        derived_output_root=Path(outputs.get("derived_output_root", DEFAULT_LSEG_DERIVED_ROOT)),
        companies=tuple(companies),
    )


def load_lseg_collection_config(path: str | Path) -> LsegCollectionConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as file:
            payload = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LsegConfigurationError(f"cannot load LSEG collection config {config_path}: {exc}") from exc
    return config_from_payload(payload)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump(mode="json"))
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _field(row: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _string(value: Any) -> str | None:
    normalized = _to_jsonable(value)
    if normalized is None:
        return None
    text = str(normalized).strip()
    return text or None


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        values = value
    else:
        values = [value]
    result: list[str] = []
    for item in values:
        if isinstance(item, dict):
            candidate = _field(item, "code", "name", "value", "$", "qcode", "_qcode")
        else:
            candidate = item
        text = _string(candidate)
        if text and text not in result:
            result.append(text)
    return result


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _raw_data(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    raw = getattr(data, "raw", None)
    value = _to_jsonable(raw)
    return value if isinstance(value, dict) else {"value": value}


def _dataframe_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    frame = getattr(data, "df", None)
    if frame is None:
        return []
    reset_index = getattr(frame, "reset_index", None)
    if callable(reset_index):
        frame = reset_index()
    to_dict = getattr(frame, "to_dict", None)
    if not callable(to_dict):
        return []
    rows = to_dict(orient="records")
    return [_to_jsonable(row) for row in rows if isinstance(row, dict)]


def _nested_string(value: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                if isinstance(item, (str, int, float, bool)):
                    text = _string(item)
                    if text:
                        return text
                else:
                    found = _nested_string(item, keys)
                    if found:
                        return found
        for item in value.values():
            found = _nested_string(item, keys)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _nested_string(item, keys)
            if found:
                return found
    return None


class _LsegSdkBackend:
    def __init__(self) -> None:
        self.ld: Any | None = None
        self.news: Any | None = None

    async def open(self) -> None:
        try:
            self.ld = importlib.import_module("lseg.data")
            self.news = importlib.import_module("lseg.data.content.news")
        except ImportError as exc:
            raise LsegDependencyError(
                "LSEG collection requires the optional SDK. Install with: python -m pip install -e '.[lseg]'"
            ) from exc
        await _await_if_needed(self.ld.open_session())

    async def close(self) -> None:
        if self.ld is None:
            return
        close_session = getattr(self.ld, "close_session", None)
        if callable(close_session):
            await _await_if_needed(close_session())

    async def headline_page(
        self,
        *,
        query: str,
        start: str,
        end: str,
        count: int,
        cursor: str | None,
    ) -> LsegHeadlinePage:
        if self.news is None:
            raise LsegNewsError("LSEG session is not open")
        if cursor:
            definition = self.news.headlines.Definition(query="", extended_params={"cursor": cursor})
        else:
            definition = self.news.headlines.Definition(
                query=query,
                date_from=start,
                date_to=end,
                count=count,
            )
        request = getattr(definition, "get_data_async", None) or definition.get_data
        response = await _await_if_needed(request())
        raw = _raw_data(response)
        next_cursor = _nested_string(raw, ("next",))
        return LsegHeadlinePage(_dataframe_rows(response), next_cursor, raw)

    async def story(self, story_id: str) -> LsegStoryResponse:
        if self.news is None:
            raise LsegNewsError("LSEG session is not open")
        story_namespace = getattr(self.news, "story", None)
        definition_type = getattr(story_namespace, "Definition", None)
        if callable(definition_type):
            definition = definition_type(story_id)
            request = getattr(definition, "get_data_async", None) or definition.get_data
            response = await _await_if_needed(request())
            data = getattr(response, "data", None)
            raw = _raw_data(response)
            body = _string(getattr(data, "news_story", None))
            if not body:
                body = _nested_string(raw, ("newsstory", "news_story", "story", "body", "content"))
        else:
            access_news = getattr(self.ld, "news", None)
            getter = getattr(access_news, "get_story", None)
            if not callable(getter):
                raise LsegNewsError("installed LSEG SDK exposes neither story.Definition nor ld.news.get_story")
            response = await _await_if_needed(getter(story_id))
            body = _string(response)
            raw = {"access_layer": "ld.news.get_story"}
        web_url = _nested_string(raw, ("weburl", "web_url"))
        if body and re.fullmatch(r"https?://\S+", body):
            web_url, body = body, None
        if not body and web_url:
            return LsegStoryResponse(story_id, "story_unavailable", web_url=web_url, raw_response=raw)
        if not body:
            return LsegStoryResponse(story_id, "missing", error="story response has no body", raw_response=raw)
        body_format = "html" if re.search(r"<[^>]+>", body) else "text"
        return LsegStoryResponse(story_id, "success", body=body, body_format=body_format, raw_response=raw)


class LsegNewsClient:
    """Async boundary around a Workspace desktop LSEG Data Library session."""

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend or _LsegSdkBackend()

    async def __aenter__(self) -> LsegNewsClient:
        opener = getattr(self._backend, "open", None)
        if callable(opener):
            await _await_if_needed(opener())
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        closer = getattr(self._backend, "close", None)
        if callable(closer):
            await _await_if_needed(closer())

    async def fetch_headline_page(
        self,
        *,
        query: str,
        start: str,
        end: str,
        count: int,
        cursor: str | None,
    ) -> LsegHeadlinePage:
        return await self._backend.headline_page(query=query, start=start, end=end, count=count, cursor=cursor)

    async def fetch_story(self, story_id: str) -> LsegStoryResponse:
        return await self._backend.story(story_id)


async def _retry(factory: Any, retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await factory()
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            error_name = type(exc).__name__.lower()
            transient = (
                isinstance(exc, TimeoutError | ConnectionError)
                or status_code in {429, 500, 502, 503, 504}
                or "timeout" in error_name
                or "connection" in error_name
            )
            if attempt >= retries or not transient:
                break
            await asyncio.sleep(min(2**attempt, 8))
    if last_error is not None:
        raise last_error
    raise LsegNewsError("LSEG retry loop ended unexpectedly")


def _normalize_headline(row: dict[str, Any], company: LsegCompanyConfig) -> dict[str, Any] | None:
    story_id = _string(_field(row, "storyId", "story_id"))
    if not story_id:
        return None
    return {
        "story_id": story_id,
        "headline": _string(_field(row, "headline", "title")) or "",
        "first_created": _string(_field(row, "firstCreated", "first_created")),
        "version_created": _string(_field(row, "versionCreated", "version_created", "index")),
        "source_code": _string(_field(row, "sourceCode", "source_code")),
        "language": _string(_field(row, "language", "languageCode", "language_code")),
        "subjects": _strings(_field(row, "subjects", "subject", "newsCodes", "news_codes")),
        "entities": _strings(_field(row, "entities", "entity", "permIds", "perm_ids")),
        "matched_symbols": [company.symbol],
        "matched_queries": [company.news_query],
        "matched_rics": [company.ric],
        "raw_rows": [_to_jsonable(row)],
    }


def _merge_headline(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key in ("matched_symbols", "matched_queries", "matched_rics", "subjects", "entities"):
        target[key] = sorted(set(target.get(key, [])) | set(incoming.get(key, [])))
    target["raw_rows"].extend(incoming.get("raw_rows", []))
    if len(str(incoming.get("headline") or "")) > len(str(target.get("headline") or "")):
        target["headline"] = incoming["headline"]
    incoming_first = str(incoming.get("first_created") or "")
    target_first = str(target.get("first_created") or "")
    if incoming_first and (not target_first or incoming_first < target_first):
        target["first_created"] = incoming_first
    incoming_version = str(incoming.get("version_created") or "")
    target_version = str(target.get("version_created") or "")
    if incoming_version and (not target_version or incoming_version > target_version):
        target["version_created"] = incoming_version
    for key in ("source_code", "language"):
        if not target.get(key) and incoming.get(key):
            target[key] = incoming[key]


def _load_completed_result(config: LsegCollectionConfig, manifest_path: Path) -> LsegFetchResult | None:
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("config_sha256") != config.config_sha256:
        raise LsegNewsError(
            f"collection {config.collection_id} already exists with a different configuration; choose a new collection.id"
        )
    if manifest.get("status") != "completed":
        return None
    files_value = manifest.get("files")
    files: dict[str, Any] = files_value if isinstance(files_value, dict) else {}
    for key in ("headlines_jsonl", "stories_jsonl"):
        entry = files.get(key)
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise LsegNewsError(f"completed LSEG manifest is missing a hash for {key}")
        artifact = config.raw_dir / str(entry["path"])
        if not artifact.exists() or sha256_file(artifact) != entry["sha256"]:
            raise LsegNewsError(f"completed LSEG collection artifact failed hash verification: {artifact}")
    counts_value = manifest.get("counts")
    counts: dict[str, Any] = counts_value if isinstance(counts_value, dict) else {}
    return LsegFetchResult(
        raw_dir=config.raw_dir,
        manifest_path=manifest_path,
        headline_count=int(counts.get("headlines", 0)),
        story_count=int(counts.get("stories", 0)),
        failed_story_count=int(counts.get("failed_stories", 0)),
        resumed=True,
    )


async def check_lseg_news(config: LsegCollectionConfig, client: LsegNewsClient) -> dict[str, Any]:
    company = config.companies[0]
    page = await _retry(
        lambda: client.fetch_headline_page(
            query=company.news_query,
            start=config.start,
            end=config.end,
            count=1,
            cursor=None,
        ),
        config.retries,
    )
    normalized = next(
        (record for row in page.rows if (record := _normalize_headline(row, company)) is not None),
        None,
    )
    story_status = "not_checked"
    story_id = None
    if normalized is not None:
        story_id = str(normalized["story_id"])
        story = await _retry(lambda: client.fetch_story(story_id), config.retries)
        story_status = story.status
    return {
        "query": company.news_query,
        "headline_count": len(page.rows),
        "story_id": story_id,
        "story_status": story_status,
    }


async def fetch_lseg_news(config: LsegCollectionConfig, client: LsegNewsClient) -> LsegFetchResult:
    raw_dir = config.raw_dir
    manifest_path = raw_dir / "manifest.json"
    completed = _load_completed_result(config, manifest_path)
    if completed is not None:
        return completed
    if raw_dir.exists() and any(raw_dir.iterdir()) and not manifest_path.exists():
        raise LsegNewsError(f"refusing to use non-empty collection directory without a manifest: {raw_dir}")

    pages_dir = raw_dir / "headline_pages"
    stories_dir = raw_dir / "stories"
    pages_dir.mkdir(parents=True, exist_ok=True)
    stories_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    atomic_write_json(
        manifest_path,
        {
            "schema_version": LSEG_RAW_SCHEMA_VERSION,
            "status": "in_progress",
            "config_sha256": config.config_sha256,
            "config": config.to_payload(),
            "started_at": started_at,
        },
    )

    headlines: dict[str, dict[str, Any]] = {}
    page_paths: list[Path] = []
    query_stats: dict[str, dict[str, Any]] = {}
    raw_headline_rows = 0
    for company_index, company in enumerate(config.companies, start=1):
        query_stats[company.symbol] = {"query": company.news_query, "ric": company.ric, "pages": 0, "rows": 0}
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for page_number in range(1, config.max_pages + 1):
            page_path = pages_dir / f"{company_index:03d}-{company.symbol}-page-{page_number:04d}.json"
            if page_path.exists():
                payload = read_json(page_path)
                if payload.get("query") != company.news_query or payload.get("cursor_in") != cursor:
                    raise LsegNewsError(f"checkpoint does not match requested page: {page_path}")
            else:
                page = await _retry(
                    lambda company=company, cursor=cursor: client.fetch_headline_page(
                        query=company.news_query,
                        start=config.start,
                        end=config.end,
                        count=config.page_size,
                        cursor=cursor,
                    ),
                    config.retries,
                )
                payload = {
                    "query": company.news_query,
                    "symbol": company.symbol,
                    "ric": company.ric,
                    "page_number": page_number,
                    "cursor_in": cursor,
                    "cursor_out": page.next_cursor,
                    "fetched_at": utc_now(),
                    "rows": [_to_jsonable(row) for row in page.rows],
                    "raw_response": _to_jsonable(page.raw_response),
                }
                atomic_write_json(page_path, payload)
            page_paths.append(page_path)
            rows_value = payload.get("rows")
            rows: list[Any] = rows_value if isinstance(rows_value, list) else []
            query_stats[company.symbol]["pages"] += 1
            query_stats[company.symbol]["rows"] += len(rows)
            raw_headline_rows += len(rows)
            for raw_row in rows:
                if not isinstance(raw_row, dict):
                    continue
                normalized = _normalize_headline(raw_row, company)
                if normalized is None:
                    continue
                story_id = str(normalized["story_id"])
                if story_id in headlines:
                    _merge_headline(headlines[story_id], normalized)
                else:
                    headlines[story_id] = normalized
            next_cursor = _string(payload.get("cursor_out"))
            if not next_cursor:
                cursor = None
                break
            if next_cursor == cursor or next_cursor in seen_cursors:
                raise LsegNewsError(f"LSEG pagination returned a repeated cursor for {company.news_query!r}")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        if cursor:
            raise LsegNewsError(
                f"LSEG query {company.news_query!r} exceeded collection.max_pages={config.max_pages}; "
                "increase the limit or narrow the date range"
            )

    headline_rows = [headlines[key] for key in sorted(headlines)]
    headlines_path = atomic_write_jsonl(raw_dir / "headlines.jsonl", headline_rows)

    semaphore = asyncio.Semaphore(config.story_concurrency)

    async def fetch_story_record(story_id: str) -> dict[str, Any]:
        story_path = stories_dir / f"{sha256_text(story_id)}.json"
        if story_path.exists():
            return read_json(story_path)
        async with semaphore:
            try:
                response = await _retry(lambda: client.fetch_story(story_id), config.retries)
                payload = {
                    "story_id": story_id,
                    "status": response.status,
                    "body": response.body,
                    "body_format": response.body_format,
                    "web_url": response.web_url,
                    "error": response.error,
                    "fetched_at": utc_now(),
                    "raw_response": _to_jsonable(response.raw_response),
                }
            except Exception as exc:
                payload = {
                    "story_id": story_id,
                    "status": "failed",
                    "body": None,
                    "body_format": None,
                    "web_url": None,
                    "error": str(exc),
                    "fetched_at": utc_now(),
                    "raw_response": {},
                }
            atomic_write_json(story_path, payload)
            return payload

    story_rows = await asyncio.gather(*(fetch_story_record(story_id) for story_id in sorted(headlines)))
    stories_path = atomic_write_jsonl(raw_dir / "stories.jsonl", story_rows)
    failed_story_count = sum(row.get("status") != "success" for row in story_rows)
    checkpoint_paths = [*page_paths, *sorted(stories_dir.glob("*.json"))]
    try:
        sdk_version = importlib.metadata.version("lseg-data")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "injected-or-unavailable"
    completed_at = utc_now()
    manifest = {
        "schema_version": LSEG_RAW_SCHEMA_VERSION,
        "status": "completed",
        "config_sha256": config.config_sha256,
        "config": config.to_payload(),
        "started_at": started_at,
        "completed_at": completed_at,
        "lseg_sdk_version": sdk_version,
        "environment": collect_run_environment(),
        "counts": {
            "headline_pages": len(page_paths),
            "headlines": len(headline_rows),
            "stories": len(story_rows),
            "failed_stories": failed_story_count,
            "successful_stories": len(story_rows) - failed_story_count,
            "raw_headline_rows": raw_headline_rows,
            "deduplicated_story_occurrences": raw_headline_rows - len(headline_rows),
            "ticker_query_associations": sum(len(row.get("matched_symbols") or []) for row in headline_rows),
        },
        "queries": query_stats,
        "failures": [
            {"story_id": row.get("story_id"), "status": row.get("status"), "error": row.get("error")}
            for row in story_rows
            if row.get("status") != "success"
        ],
        "files": {
            "headlines_jsonl": {"path": headlines_path.name, "sha256": sha256_file(headlines_path)},
            "stories_jsonl": {"path": stories_path.name, "sha256": sha256_file(stories_path)},
            "checkpoint_tree_sha256": directory_digest(raw_dir, checkpoint_paths),
        },
        "sharing": {
            "licensed_full_text": True,
            "redistribute": False,
            "note": "Workspace story content is local research material and must not be committed or shared.",
        },
    }
    atomic_write_json(manifest_path, manifest)
    return LsegFetchResult(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        headline_count=len(headline_rows),
        story_count=len(story_rows),
        failed_story_count=failed_story_count,
    )
