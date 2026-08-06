from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import inspect
import logging
import math
import re
import shutil
import tomllib
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, TypedDict

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
from .utils import utc_now

logger = logging.getLogger(__name__)

LSEG_RAW_SCHEMA_VERSION = 1
DEFAULT_LSEG_RAW_ROOT = Path("Data/news")
DEFAULT_LSEG_DERIVED_ROOT = Path("Data/derived/lseg")
DEFAULT_LSEG_PAGE_SIZE = 100
DEFAULT_LSEG_MAX_PAGES = 50
DEFAULT_LSEG_STORY_CONCURRENCY = 4
DEFAULT_LSEG_RETRIES = 3
DEFAULT_LSEG_REQUESTS_PER_SECOND = 3.0
DEFAULT_LSEG_MIN_TEXT_CHARS = 100
DEFAULT_LSEG_MAX_SCORING_CHARS = 8_000
DEFAULT_LSEG_WINDOW_DAYS: int | None = None


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
    requests_per_second: float = DEFAULT_LSEG_REQUESTS_PER_SECOND
    max_requests_per_run: int | None = None
    window_days: int | None = DEFAULT_LSEG_WINDOW_DAYS
    min_text_chars: int = DEFAULT_LSEG_MIN_TEXT_CHARS
    max_scoring_chars: int = DEFAULT_LSEG_MAX_SCORING_CHARS
    raw_output_root: Path = DEFAULT_LSEG_RAW_ROOT
    derived_output_root: Path = DEFAULT_LSEG_DERIVED_ROOT
    # Operational-only: when true, the per-story checkpoint shards under stories/
    # are deleted once the collection completes (they byte-for-byte duplicate
    # stories.jsonl). Deliberately excluded from to_payload()/config_sha256 so
    # toggling it never changes collection identity or disturbs an in-progress
    # resume.
    prune_story_shards_on_completion: bool = False
    # When non-empty, the story-fetch phase only retrieves bodies for headlines
    # whose source_code is in this allowlist (e.g. ["NS:RTRS"] for Reuters-only
    # stories). All headlines are still collected regardless; this narrows only
    # the expensive Phase 2. Empty means fetch every story.
    story_source_allowlist: tuple[str, ...] = ()
    # Headline-only collections skip Phase 2 entirely. The default remains true
    # for backwards compatibility; the false value is included in the config
    # identity so a headline-only collection cannot be mistaken for a full-text
    # corpus.
    fetch_story_bodies: bool = True
    companies: tuple[LsegCompanyConfig, ...] = ()

    @property
    def raw_dir(self) -> Path:
        return self.raw_output_root / f"lseg_{self.collection_id}"

    @property
    def derived_dir(self) -> Path:
        return self.derived_output_root / self.collection_id

    def to_payload(self) -> dict[str, Any]:
        collection = {
            "id": self.collection_id,
            "start": self.start,
            "end": self.end,
            "language": self.language,
            "session": self.session,
            "page_size": self.page_size,
            "max_pages": self.max_pages,
            "story_concurrency": self.story_concurrency,
            "retries": self.retries,
            "requests_per_second": self.requests_per_second,
        }
        if self.window_days is not None:
            collection["window_days"] = self.window_days
        if self.max_requests_per_run is not None:
            collection["max_requests_per_run"] = self.max_requests_per_run
        if self.story_source_allowlist:
            collection["story_source_allowlist"] = list(self.story_source_allowlist)
        if not self.fetch_story_bodies:
            collection["fetch_story_bodies"] = False
        return {
            "collection": collection,
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
    request_count: int = 0
    retry_count: int = 0
    pagination_anomaly_count: int = 0


@dataclass(frozen=True)
class LsegProgressUpdate:
    """Resume-aware progress snapshot emitted during an LSEG collection."""

    phase: str
    completed: int
    total: int
    current: str
    raw_headline_rows: int = 0
    unique_headlines: int = 0
    checkpointed: int = 0
    failed_stories: int = 0
    requests_started: int = 0
    paced_waits: int = 0
    paced_wait_seconds: float = 0.0
    requests_per_second: float = DEFAULT_LSEG_REQUESTS_PER_SECOND
    retries: int = 0
    retry_backoff_seconds: float = 0.0
    pagination_anomalies: int = 0


LsegProgressCallback = Callable[[LsegProgressUpdate], None]


@dataclass(frozen=True)
class LsegRequestMetrics:
    requests_started: int
    paced_waits: int
    paced_wait_seconds: float
    requests_per_second: float


class _RequestProgress(TypedDict):
    requests_started: int
    paced_waits: int
    paced_wait_seconds: float
    requests_per_second: float
    retries: int
    retry_backoff_seconds: float
    pagination_anomalies: int


class _RequestPacer:
    """Smooth request starts across concurrent headline and story calls."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        max_requests: int | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.requests_per_second = requests_per_second
        self._interval = 1.0 / requests_per_second
        self._max_requests = max_requests
        self._clock = clock
        self._sleeper = sleeper or asyncio.sleep
        self._lock = asyncio.Lock()
        self._next_start = 0.0
        self._requests_started = 0
        self._paced_waits = 0
        self._paced_wait_seconds = 0.0

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return asyncio.get_running_loop().time()

    async def acquire(self) -> None:
        async with self._lock:
            if self._max_requests is not None and self._requests_started >= self._max_requests:
                raise LsegNewsError(
                    f"LSEG request safety budget exhausted at {self._max_requests:,} requests; "
                    "saved checkpoints are intact, so raise the frozen budget or resume after the quota resets"
                )
            now = self._now()
            delay = max(0.0, self._next_start - now)
            if delay > 0.001:
                self._paced_waits += 1
                self._paced_wait_seconds += delay
                await self._sleeper(delay)
                now = self._now()
            self._next_start = max(self._next_start, now) + self._interval
            self._requests_started += 1

    def snapshot(self) -> LsegRequestMetrics:
        return LsegRequestMetrics(
            requests_started=self._requests_started,
            paced_waits=self._paced_waits,
            paced_wait_seconds=self._paced_wait_seconds,
            requests_per_second=self.requests_per_second,
        )


@dataclass(frozen=True)
class LsegCollectionWindow:
    index: int
    start: str
    end: str

    @property
    def checkpoint_label(self) -> str:
        start = self.start.replace("-", "").replace(":", "").replace("Z", "z")
        end = self.end.replace("-", "").replace(":", "").replace("Z", "z")
        return f"window-{self.index:04d}-{start}-{end}"


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


def _float_setting(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LsegConfigurationError(f"{name} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise LsegConfigurationError(f"{name} must be between {minimum:g} and {maximum:g}")
    return parsed


def _optional_int_setting(value: Any, *, name: str, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    return _int_setting(value, name=name, minimum=minimum, maximum=maximum)


def _bool_setting(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise LsegConfigurationError(f"{name} must be a boolean")


def _string_list_setting(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LsegConfigurationError(f"{name} must be a list of strings")
    return tuple(dict.fromkeys(item.strip() for item in value if item.strip()))


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_config_datetime(value: str, *, name: str) -> datetime:
    normalized = _parse_utc(value, name=name)
    return datetime.fromisoformat(normalized.replace("Z", "+00:00"))


def collection_windows(config: LsegCollectionConfig) -> list[LsegCollectionWindow]:
    """Return the query windows used for one LSEG collection."""
    if config.window_days is None:
        return [LsegCollectionWindow(1, config.start, config.end)]
    start = _parse_config_datetime(config.start, name="collection.start")
    end = _parse_config_datetime(config.end, name="collection.end")
    step = timedelta(days=config.window_days)
    windows: list[LsegCollectionWindow] = []
    current = start
    while current < end:
        window_end = min(current + step, end)
        windows.append(LsegCollectionWindow(len(windows) + 1, _format_utc(current), _format_utc(window_end)))
        current = window_end
    return windows


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
        requests_per_second=_float_setting(
            collection.get("requests_per_second", DEFAULT_LSEG_REQUESTS_PER_SECOND),
            name="collection.requests_per_second",
            minimum=0.1,
            maximum=5.0,
        ),
        max_requests_per_run=_optional_int_setting(
            collection.get("max_requests_per_run"),
            name="collection.max_requests_per_run",
            minimum=1,
            maximum=1_000_000,
        ),
        window_days=_optional_int_setting(
            collection.get("window_days"),
            name="collection.window_days",
            minimum=1,
            maximum=3660,
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
        prune_story_shards_on_completion=_bool_setting(
            collection.get("prune_story_shards_on_completion", False),
            name="collection.prune_story_shards_on_completion",
        ),
        story_source_allowlist=_string_list_setting(
            collection.get("story_source_allowlist"),
            name="collection.story_source_allowlist",
        ),
        fetch_story_bodies=_bool_setting(
            collection.get("fetch_story_bodies", True),
            name="collection.fetch_story_bodies",
        ),
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


def _story_content(data: Any) -> tuple[str | None, str | None, str | None]:
    story = getattr(data, "story", None)
    content = getattr(story, "content", None)
    html = _string(getattr(content, "html", None))
    text = _string(getattr(content, "text", None))
    web_url = _string(getattr(content, "web_url", None))
    if html:
        return html, "html", web_url
    if text:
        return text, "text", web_url
    return None, None, web_url


class _LsegSdkBackend:
    def __init__(self) -> None:
        self.ld: Any | None = None
        self.news: Any | None = None

    async def open(self) -> None:
        warnings.filterwarnings(
            "ignore",
            message=r"The behavior of (?:array|DataFrame) concatenation with empty.*",
            category=FutureWarning,
            module=r"lseg\.data\.content\.news\._df_builder",
        )
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
            # LSEG Data Library 2.1.1's asynchronous headline provider has a
            # malformed ``dict.update`` call on its archive branch.  Keep the
            # SDK's own age decision, but put archive parameters directly into
            # the query so that the broken branch is not entered.  This can be
            # removed once the SDK fixes its asynchronous archive routing.
            provider = getattr(definition, "_provider", None)
            archive_check = getattr(provider, "is_date_older_than_15months", None)
            if callable(archive_check) and bool(archive_check(start)):
                definition = self.news.headlines.Definition(
                    query=query,
                    count=count,
                    extended_params={
                        "dateFrom": start,
                        "dateTo": end,
                        "archive": True,
                    },
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
            body, body_format, web_url = _story_content(data)
            if not body:
                body = _string(getattr(data, "news_story", None))
                body_format = None
            if not body:
                body = _nested_string(raw, ("newsstory", "news_story", "story", "body", "content"))
                body_format = None
        else:
            access_news = getattr(self.ld, "news", None)
            getter = getattr(access_news, "get_story", None)
            if not callable(getter):
                raise LsegNewsError("installed LSEG SDK exposes neither story.Definition nor ld.news.get_story")
            response = await _await_if_needed(getter(story_id))
            body = _string(response)
            body_format = None
            raw = {"access_layer": "ld.news.get_story"}
            web_url = None
        web_url = web_url or _nested_string(raw, ("weburl", "web_url"))
        if body and re.fullmatch(r"https?://\S+", body):
            web_url, body = body, None
        if not body and web_url:
            return LsegStoryResponse(story_id, "story_unavailable", web_url=web_url, raw_response=raw)
        if not body:
            return LsegStoryResponse(story_id, "missing", error="story response has no body", raw_response=raw)
        body_format = body_format or ("html" if re.search(r"<[^>]+>", body) else "text")
        return LsegStoryResponse(story_id, "success", body=body, body_format=body_format, raw_response=raw)


class LsegNewsClient:
    """Async boundary around a Workspace desktop LSEG Data Library session."""

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend or _LsegSdkBackend()
        self._pacer = _RequestPacer(DEFAULT_LSEG_REQUESTS_PER_SECOND)

    def configure_request_pacing(self, requests_per_second: float, max_requests: int | None = None) -> None:
        self._pacer = _RequestPacer(requests_per_second, max_requests=max_requests)

    @property
    def request_metrics(self) -> LsegRequestMetrics:
        return self._pacer.snapshot()

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
        await self._pacer.acquire()
        try:
            return await self._backend.headline_page(query=query, start=start, end=end, count=count, cursor=cursor)
        except ValueError as exc:
            if "session is not opened" in str(exc).lower():
                raise LsegNewsError(
                    "cannot use the LSEG desktop session; start Workspace, sign in, and retry the same command"
                ) from exc
            raise

    async def fetch_story(self, story_id: str) -> LsegStoryResponse:
        await self._pacer.acquire()
        try:
            return await self._backend.story(story_id)
        except ValueError as exc:
            if "session is not opened" in str(exc).lower():
                raise LsegNewsError(
                    "cannot use the LSEG desktop session; start Workspace, sign in, and retry the same command"
                ) from exc
            raise


def _status_code_of(exc: Exception) -> int | None:
    """Best-effort HTTP status for an exception.

    LSEG's LDError exposes the code as ``.code``; most other libraries use
    ``.status_code``. Check both and coerce to int so 429/5xx stay detectable.
    """

    raw = getattr(exc, "status_code", None)
    if raw is None:
        raw = getattr(exc, "code", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _http_headers_from_exc(exc: Exception) -> dict[str, str] | None:
    """Lower-cased HTTP headers from a raised LDError, if reachable.

    The SDK attaches the originating ``Response`` to the error (``error.response``)
    and that response carries the underlying ``httpx.Response`` (or a list of them)
    on ``.raw``, whose ``.headers`` hold any rate-limit metadata LSEG returned.
    """

    response = getattr(exc, "response", None)
    raw = getattr(response, "raw", None)
    if isinstance(raw, list):
        raw = raw[-1] if raw else None
    headers = getattr(raw, "headers", None)
    if headers is None:
        return None
    try:
        return {str(key).lower(): str(value) for key, value in headers.items()}
    except Exception:
        return None


def _parse_retry_after(value: str) -> float | None:
    """Seconds to wait from a Retry-After header (numeric seconds or HTTP-date)."""

    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def _format_reset_clock(value: str) -> str | None:
    """Wall-clock for a RateLimit-Reset header (epoch seconds or seconds-from-now)."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1_000_000_000:  # looks like an absolute epoch timestamp
        reset_at = datetime.fromtimestamp(number, UTC)
    else:  # a relative seconds-until-reset value
        reset_at = datetime.now(UTC) + timedelta(seconds=number)
    return reset_at.strftime("%H:%M:%S UTC")


_RATELIMIT_WINDOW_UNITS = {1: "sec", 60: "min", 3600: "hour", 86400: "day", 604800: "week"}


def _format_ratelimit_policy(value: str) -> str | None:
    """Render an IETF RateLimit-Policy (e.g. ``5;w=1, 10000;w=86400``) readably."""

    rendered: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        quota, _, params = item.partition(";")
        quota = quota.strip()
        window: int | None = None
        for param in params.split(";"):
            param = param.strip()
            if param.startswith("w="):
                try:
                    window = int(param[2:])
                except ValueError:
                    window = None
        unit = _RATELIMIT_WINDOW_UNITS.get(window) if window is not None else None
        if unit is None:
            unit = f"{window}s" if window else "window"
        rendered.append(f"{quota}/{unit}")
    return ", ".join(rendered) if rendered else None


def _rate_limit_summary(exc: Exception) -> str | None:
    """Human-readable throttle detail extracted from a 429's HTTP headers.

    Surfaces whatever LSEG actually returns — Retry-After and/or standard
    RateLimit-* headers — so the otherwise-invisible limit, remaining budget,
    and reset time become visible. Returns None when no rate-limit headers exist.
    """

    headers = _http_headers_from_exc(exc)
    if not headers:
        return None
    parts: list[str] = []

    retry_after = headers.get("retry-after")
    if retry_after:
        wait_seconds = _parse_retry_after(retry_after)
        if wait_seconds is not None:
            reset_at = (datetime.now(UTC) + timedelta(seconds=wait_seconds)).strftime("%H:%M:%S UTC")
            parts.append(f"wait {wait_seconds:.0f}s (Retry-After) — resets ~{reset_at}")
        else:
            parts.append(f"Retry-After: {retry_after}")

    policy = headers.get("ratelimit-policy") or headers.get("x-ratelimit-policy")
    if policy:
        formatted_policy = _format_ratelimit_policy(policy)
        parts.append(f"limits {formatted_policy}" if formatted_policy else f"policy {policy}")

    limit = headers.get("x-ratelimit-limit") or headers.get("ratelimit-limit")
    remaining = headers.get("x-ratelimit-remaining") or headers.get("ratelimit-remaining")
    reset = headers.get("x-ratelimit-reset") or headers.get("ratelimit-reset")
    if limit is not None or remaining is not None:
        usage = f"{remaining} remaining" if remaining is not None else "limit reached"
        try:
            if limit is not None and remaining is not None:
                usage = f"{int(limit) - int(remaining)}/{int(limit)} used, {remaining} remaining"
        except ValueError:
            pass
        parts.append(usage)
    if reset is not None:
        clock = _format_reset_clock(reset)
        parts.append(f"resets ~{clock}" if clock else f"reset={reset}")

    known = {
        "retry-after",
        "ratelimit-policy",
        "x-ratelimit-policy",
        "x-ratelimit-limit",
        "ratelimit-limit",
        "x-ratelimit-remaining",
        "ratelimit-remaining",
        "x-ratelimit-reset",
        "ratelimit-reset",
    }
    for key, value in headers.items():
        if key not in known and any(hint in key for hint in ("ratelimit", "rate-limit", "quota")):
            parts.append(f"{key}={value}")

    return "; ".join(parts) if parts else None


async def _retry(
    factory: Any,
    retries: int,
    *,
    on_retry: Callable[[float, Exception], None] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await factory()
        except Exception as exc:
            last_error = exc
            status_code = _status_code_of(exc)
            error_name = type(exc).__name__.lower()
            transient = (
                isinstance(exc, TimeoutError | ConnectionError)
                or status_code in {429, 500, 502, 503, 504}
                or "timeout" in error_name
                or "connection" in error_name
            )
            if attempt >= retries or not transient:
                break
            # Rate-limit throttles (429) need a longer cooldown than a transient
            # network blip, so give them more breathing room within the same
            # retry budget.
            delay = min(5 * 2**attempt, 60) if status_code == 429 else min(2**attempt, 8)
            if status_code == 429:
                logger.warning(
                    "LSEG rate limit (HTTP 429): %s — backing off %.0fs (retry %d of %d)",
                    _rate_limit_summary(exc) or "no rate-limit headers returned by LSEG",
                    delay,
                    attempt + 1,
                    retries,
                )
            if on_retry is not None:
                on_retry(delay, exc)
            await asyncio.sleep(delay)
    if last_error is not None:
        if _status_code_of(last_error) == 429:
            summary = _rate_limit_summary(last_error)
            detail = f" {summary}." if summary else " LSEG returned no rate-limit headers."
            raise LsegNewsError(
                f"LSEG rate limit hit (HTTP 429) and did not clear after {retries} "
                f"retr{'y' if retries == 1 else 'ies'}.{detail} Wait for the limit to reset, "
                "then re-run the same command to resume from the last checkpoint."
            ) from last_error
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
    if manifest.get("config_sha256") != config.config_sha256 and not _safe_in_progress_operational_change(manifest, config):
        raise LsegNewsError(f"collection {config.collection_id} already exists with a different configuration; choose a new collection.id")
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
        request_count=0,
        retry_count=0,
        pagination_anomaly_count=int(counts.get("pagination_anomalies", 0)),
    )


def _prune_story_shards(config: LsegCollectionConfig, manifest_path: Path) -> int:
    """Delete per-story checkpoint shards once their content is durably consolidated.

    The files under ``stories/`` are byte-for-byte duplicates of ``stories.jsonl``;
    they exist only to make the multi-day story phase resumable. Once a collection
    is completed they are pure redundancy. Deletion is gated on the consolidated
    ``stories.jsonl`` existing and hash-matching the manifest, so the authoritative
    copy is always verified intact before any shard is removed. Idempotent: a no-op
    on a collection that is not completed or has already been pruned. Returns the
    number of shard files removed.
    """

    if not manifest_path.exists():
        return 0
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        return 0
    files_value = manifest.get("files")
    files = files_value if isinstance(files_value, dict) else {}
    entry = files.get("stories_jsonl")
    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
        return 0
    stories_jsonl = config.raw_dir / str(entry["path"])
    if not stories_jsonl.exists() or sha256_file(stories_jsonl) != entry["sha256"]:
        return 0
    stories_dir = config.raw_dir / "stories"
    if not stories_dir.is_dir():
        return 0
    removed = sum(1 for _ in stories_dir.glob("*.json"))
    shutil.rmtree(stories_dir)
    return removed


def _safe_in_progress_operational_change(manifest: dict[str, Any], config: LsegCollectionConfig) -> bool:
    """Allow safe pacing changes and page-cap increases on an unfinished collection."""

    if manifest.get("status") != "in_progress":
        return False
    stored = manifest.get("config")
    if not isinstance(stored, dict):
        return False
    stored_collection = stored.get("collection")
    if not isinstance(stored_collection, dict):
        return False
    stored_max_pages = stored_collection.get("max_pages")
    if stored_max_pages is None:
        return False
    try:
        old_max_pages = int(stored_max_pages)
    except (TypeError, ValueError):
        return False
    if config.max_pages < old_max_pages:
        return False
    config_payload = config.to_payload()
    collection_overrides = {
        **stored_collection,
        "max_pages": config.max_pages,
        "requests_per_second": config.requests_per_second,
    }
    # Narrowing or adopting the story-source allowlist is safe mid-collection: it
    # only affects which stories get bodies in the not-yet-run story phase and
    # leaves the headline checkpoints untouched.
    config_collection = config_payload["collection"]
    if "story_source_allowlist" in config_collection:
        collection_overrides["story_source_allowlist"] = config_collection["story_source_allowlist"]
    else:
        collection_overrides.pop("story_source_allowlist", None)
    normalized_stored = {**stored, "collection": collection_overrides}
    return canonical_json(normalized_stored) == canonical_json(config_payload)


async def check_lseg_news(
    config: LsegCollectionConfig,
    client: LsegNewsClient,
    *,
    all_companies: bool = False,
) -> dict[str, Any]:
    client.configure_request_pacing(config.requests_per_second, config.max_requests_per_run)
    companies = config.companies if all_companies else config.companies[:1]
    first_window = collection_windows(config)[0]
    checks: list[dict[str, Any]] = []
    for company in companies:
        start = config.start if all_companies else first_window.start
        end = config.end if all_companies else first_window.end
        page = await _retry(
            lambda company=company, start=start, end=end: client.fetch_headline_page(
                query=company.news_query,
                start=start,
                end=end,
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
        if normalized is not None and config.fetch_story_bodies:
            story_id = str(normalized["story_id"])
            story = await _retry(lambda story_id=story_id: client.fetch_story(story_id), config.retries)
            story_status = story.status
        elif normalized is not None:
            story_status = "skipped_headline_only"
        checks.append(
            {
                "symbol": company.symbol,
                "ric": company.ric,
                "query": company.news_query,
                "headline_count": len(page.rows),
                "story_id": story_id,
                "story_status": story_status,
            }
        )
    return {**checks[0], "checks": checks}


def _completed_window_checkpoints(pages_dir: Path, config: LsegCollectionConfig) -> set[tuple[str, int]]:
    expected_symbols = {company.symbol for company in config.companies}
    max_window_index = len(collection_windows(config))
    completed: set[tuple[str, int]] = set()
    for page_path in pages_dir.glob("*.json"):
        payload = read_json(page_path)
        symbol = str(payload.get("symbol") or "")
        try:
            window_index = int(payload.get("window_index", 1))
        except (TypeError, ValueError):
            continue
        if (
            symbol in expected_symbols
            and 1 <= window_index <= max_window_index
            and (not _string(payload.get("cursor_out")) or payload.get("pagination_terminal_reason"))
        ):
            completed.add((symbol, window_index))
    return completed


def _emit_progress(callback: LsegProgressCallback | None, update: LsegProgressUpdate) -> None:
    if callback is not None:
        callback(update)


async def fetch_lseg_news(
    config: LsegCollectionConfig,
    client: LsegNewsClient,
    *,
    progress_callback: LsegProgressCallback | None = None,
) -> LsegFetchResult:
    client.configure_request_pacing(config.requests_per_second, config.max_requests_per_run)
    retry_count = 0
    retry_backoff_seconds = 0.0
    pagination_anomalies: list[dict[str, Any]] = []

    def record_retry(delay: float, _exc: Exception) -> None:
        nonlocal retry_count, retry_backoff_seconds
        retry_count += 1
        retry_backoff_seconds += delay

    def request_progress() -> _RequestProgress:
        metrics = client.request_metrics
        return {
            "requests_started": metrics.requests_started,
            "paced_waits": metrics.paced_waits,
            "paced_wait_seconds": metrics.paced_wait_seconds,
            "requests_per_second": metrics.requests_per_second,
            "retries": retry_count,
            "retry_backoff_seconds": retry_backoff_seconds,
            "pagination_anomalies": len(pagination_anomalies),
        }

    raw_dir = config.raw_dir
    manifest_path = raw_dir / "manifest.json"
    completed = _load_completed_result(config, manifest_path)
    if completed is not None:
        if config.prune_story_shards_on_completion:
            _prune_story_shards(config, manifest_path)
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
    windows = collection_windows(config)
    total_windows = len(config.companies) * len(windows)
    checkpointed_windows = _completed_window_checkpoints(pages_dir, config)
    completed_windows = len(checkpointed_windows)
    _emit_progress(
        progress_callback,
        LsegProgressUpdate(
            phase="headlines",
            completed=completed_windows,
            total=total_windows,
            current="Loading saved checkpoints" if checkpointed_windows else "Starting headline collection",
            checkpointed=len(checkpointed_windows),
            **request_progress(),
        ),
    )
    for company_index, company in enumerate(config.companies, start=1):
        query_stats[company.symbol] = {"query": company.news_query, "ric": company.ric, "pages": 0, "rows": 0, "windows": len(windows)}
        for window in windows:
            cursor: str | None = None
            seen_cursors: set[str] = set()
            window_story_ids: set[str] = set()
            for page_number in range(1, config.max_pages + 1):
                if config.window_days is None:
                    page_path = pages_dir / f"{company_index:03d}-{company.symbol}-page-{page_number:04d}.json"
                else:
                    page_path = pages_dir / f"{company_index:03d}-{company.symbol}-{window.checkpoint_label}-page-{page_number:04d}.json"
                if page_path.exists():
                    payload = read_json(page_path)
                    checkpoint_matches = payload.get("query") == company.news_query and payload.get("cursor_in") == cursor
                    if config.window_days is not None:
                        checkpoint_matches = (
                            checkpoint_matches and payload.get("window_start") == window.start and payload.get("window_end") == window.end
                        )
                    if not checkpoint_matches:
                        raise LsegNewsError(f"checkpoint does not match requested page: {page_path}")
                else:
                    page = await _retry(
                        lambda company=company, cursor=cursor, window=window: client.fetch_headline_page(
                            query=company.news_query,
                            start=window.start,
                            end=window.end,
                            count=config.page_size,
                            cursor=cursor,
                        ),
                        config.retries,
                        on_retry=record_retry,
                    )
                    payload = {
                        "query": company.news_query,
                        "symbol": company.symbol,
                        "ric": company.ric,
                        "window_index": window.index,
                        "window_start": window.start,
                        "window_end": window.end,
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
                page_story_ids: set[str] = set()
                for raw_row in rows:
                    if not isinstance(raw_row, dict):
                        continue
                    normalized = _normalize_headline(raw_row, company)
                    if normalized is None:
                        continue
                    story_id = str(normalized["story_id"])
                    page_story_ids.add(story_id)
                    if story_id in headlines:
                        _merge_headline(headlines[story_id], normalized)
                    else:
                        headlines[story_id] = normalized
                new_window_story_ids = page_story_ids - window_story_ids
                window_story_ids.update(page_story_ids)
                terminal_reason = _string(payload.get("pagination_terminal_reason"))
                if terminal_reason:
                    pagination_anomalies.append(
                        {
                            "symbol": company.symbol,
                            "query": company.news_query,
                            "window_index": window.index,
                            "window_start": window.start,
                            "window_end": window.end,
                            "page_number": page_number,
                            "reason": terminal_reason,
                            "duplicate_rows": len(rows),
                        }
                    )
                next_cursor = None if terminal_reason else _string(payload.get("cursor_out"))
                if not next_cursor:
                    cursor = None
                    break
                if next_cursor == cursor or next_cursor in seen_cursors:
                    if new_window_story_ids:
                        raise LsegNewsError(
                            f"LSEG pagination returned a repeated cursor with {len(new_window_story_ids)} new stories "
                            f"for {company.news_query!r}"
                        )
                    terminal_reason = "repeated_cursor_duplicate_page"
                    payload["pagination_terminal_reason"] = terminal_reason
                    atomic_write_json(page_path, payload)
                    pagination_anomalies.append(
                        {
                            "symbol": company.symbol,
                            "query": company.news_query,
                            "window_index": window.index,
                            "window_start": window.start,
                            "window_end": window.end,
                            "page_number": page_number,
                            "reason": terminal_reason,
                            "duplicate_rows": len(rows),
                        }
                    )
                    cursor = None
                    break
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            if cursor:
                raise LsegNewsError(
                    f"LSEG query {company.news_query!r} exceeded collection.max_pages={config.max_pages} "
                    f"for {window.start} to {window.end}; increase the limit or narrow the date range"
                )
            checkpoint_key = (company.symbol, window.index)
            if checkpoint_key not in checkpointed_windows:
                completed_windows += 1
            _emit_progress(
                progress_callback,
                LsegProgressUpdate(
                    phase="headlines",
                    completed=completed_windows,
                    total=total_windows,
                    current=f"{company.symbol} {window.index}/{len(windows)}",
                    raw_headline_rows=raw_headline_rows,
                    unique_headlines=len(headlines),
                    checkpointed=len(checkpointed_windows),
                    **request_progress(),
                ),
            )

    headline_rows = [headlines[key] for key in sorted(headlines)]
    headlines_path = atomic_write_jsonl(raw_dir / "headlines.jsonl", headline_rows)

    # All headlines are kept; the story-fetch phase can be narrowed to specific
    # sources (e.g. Reuters-only) to stay within the request budget while still
    # preserving full headline coverage.
    if not config.fetch_story_bodies:
        story_targets: list[str] = []
    elif config.story_source_allowlist:
        allowed_sources = set(config.story_source_allowlist)
        story_targets = [
            story_id for story_id in sorted(headlines) if (headlines[story_id].get("source_code") or "") in allowed_sources
        ]
    else:
        story_targets = sorted(headlines)
    story_target_count = len(story_targets)

    semaphore = asyncio.Semaphore(config.story_concurrency)
    existing_story_records = {
        story_id: read_json(story_path)
        for story_id in story_targets
        if (story_path := stories_dir / f"{sha256_text(story_id)}.json").exists()
    }
    existing_story_ids = set(existing_story_records)
    completed_stories = len(existing_story_ids)
    failed_stories_in_progress = sum(payload.get("status") != "success" for payload in existing_story_records.values())
    _emit_progress(
        progress_callback,
        LsegProgressUpdate(
            phase="stories",
            completed=completed_stories,
            total=story_target_count,
            current="Loading saved stories" if existing_story_ids else "Starting full-story retrieval",
            raw_headline_rows=raw_headline_rows,
            unique_headlines=len(headlines),
            checkpointed=len(existing_story_ids),
            failed_stories=failed_stories_in_progress,
            **request_progress(),
        ),
    )

    async def fetch_story_record(story_id: str) -> dict[str, Any]:
        nonlocal completed_stories, failed_stories_in_progress
        story_path = stories_dir / f"{sha256_text(story_id)}.json"
        if story_path.exists():
            return existing_story_records[story_id]
        async with semaphore:
            try:
                response = await _retry(
                    lambda: client.fetch_story(story_id),
                    config.retries,
                    on_retry=record_retry,
                )
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
            completed_stories += 1
            if payload.get("status") != "success":
                failed_stories_in_progress += 1
            _emit_progress(
                progress_callback,
                LsegProgressUpdate(
                    phase="stories",
                    completed=completed_stories,
                    total=story_target_count,
                    current="Retrieving full stories",
                    raw_headline_rows=raw_headline_rows,
                    unique_headlines=len(headlines),
                    checkpointed=len(existing_story_ids),
                    failed_stories=failed_stories_in_progress,
                    **request_progress(),
                ),
            )
            return payload

    story_rows = await asyncio.gather(*(fetch_story_record(story_id) for story_id in story_targets))
    stories_path = atomic_write_jsonl(raw_dir / "stories.jsonl", story_rows)
    failed_story_count = sum(row.get("status") != "success" for row in story_rows)
    final_request_metrics = client.request_metrics
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
            "fetch_story_bodies": config.fetch_story_bodies,
            "story_source_allowlist": list(config.story_source_allowlist),
            "headlines_without_story_fetch": len(headline_rows) - len(story_rows),
            "failed_stories": failed_story_count,
            "successful_stories": len(story_rows) - failed_story_count,
            "raw_headline_rows": raw_headline_rows,
            "deduplicated_story_occurrences": raw_headline_rows - len(headline_rows),
            "ticker_query_associations": sum(len(row.get("matched_symbols") or []) for row in headline_rows),
            "pagination_anomalies": len(pagination_anomalies),
        },
        "queries": query_stats,
        "request_control": {
            "requests_per_second": final_request_metrics.requests_per_second,
            "requests_started": final_request_metrics.requests_started,
            "paced_waits": final_request_metrics.paced_waits,
            "paced_wait_seconds": final_request_metrics.paced_wait_seconds,
            "retries": retry_count,
            "retry_backoff_seconds": retry_backoff_seconds,
        },
        "pagination_anomalies": pagination_anomalies,
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
            "licensed_full_text": bool(story_rows),
            "licensed_headlines": True,
            "redistribute": False,
            "note": "Workspace news content is local research material and must not be committed or shared.",
        },
        "checkpoint_retention": {
            "story_shards": "pruned_on_completion" if config.prune_story_shards_on_completion else "retained",
            "authoritative_story_artifact": stories_path.name,
            "note": (
                "Per-story shards under stories/ duplicate stories.jsonl. When pruning is enabled they are "
                "deleted after this manifest is written; in-progress resume is unaffected because shards are "
                "only removed once status is completed."
            ),
        },
    }
    atomic_write_json(manifest_path, manifest)
    if config.prune_story_shards_on_completion:
        _prune_story_shards(config, manifest_path)
    return LsegFetchResult(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        headline_count=len(headline_rows),
        story_count=len(story_rows),
        failed_story_count=failed_story_count,
        request_count=final_request_metrics.requests_started,
        retry_count=retry_count,
        pagination_anomaly_count=len(pagination_anomalies),
    )
