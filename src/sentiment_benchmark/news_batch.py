from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .news_package import QueryMatrixEntry, load_query_matrix
from .news_source import NewsFetchConfig, NewsOutputPaths, make_news_fetch_config, write_news_corpus

MAX_WEEKLY_WINDOWS = 52


class NewsBatchError(RuntimeError):
    """Raised when Tavily batch news collection cannot be planned or run."""


@dataclass(frozen=True)
class DateWindow:
    start_date: str
    end_date: str


@dataclass(frozen=True)
class NewsBatchFetchPlan:
    query_entry: QueryMatrixEntry
    date_window: DateWindow
    config: NewsFetchConfig


@dataclass(frozen=True)
class NewsBatchResult:
    plans: list[NewsBatchFetchPlan]
    output_paths: list[NewsOutputPaths]


def parse_date_window(value: str) -> DateWindow:
    start, separator, end = value.partition(":")
    if not separator:
        raise NewsBatchError("date window must use START:END format, for example 2026-05-01:2026-05-07")
    start = start.strip()
    end = end.strip()
    try:
        config = make_news_fetch_config(query="date validation", start_date=start, end_date=end, time_range=None)
    except ValueError as exc:
        raise NewsBatchError(str(exc)) from exc
    if not config.start_date or not config.end_date:
        raise NewsBatchError("date window must include both start and end dates")
    if config.start_date > config.end_date:
        raise NewsBatchError("date window start_date must be before or equal to end_date")
    return DateWindow(start_date=config.start_date, end_date=config.end_date)


def build_weekly_date_windows(weeks: int, *, end_date: str | None = None) -> list[str]:
    """Generate `weeks` contiguous 7-day windows ending at end_date (default today), oldest first."""
    if weeks < 1:
        raise NewsBatchError("weeks must be 1 or greater")
    if weeks > MAX_WEEKLY_WINDOWS:
        raise NewsBatchError(f"weeks must be at most {MAX_WEEKLY_WINDOWS}")
    if end_date is None:
        final_end = date.today()
    else:
        try:
            final_end = date.fromisoformat(end_date.strip())
        except ValueError as exc:
            raise NewsBatchError("end date must use YYYY-MM-DD format") from exc
    windows: list[str] = []
    for index in range(weeks - 1, -1, -1):
        window_end = final_end - timedelta(days=7 * index)
        window_start = window_end - timedelta(days=6)
        windows.append(f"{window_start.isoformat()}:{window_end.isoformat()}")
    return windows


def build_news_batch_plan(
    *,
    query_matrix_path: str | Path,
    date_windows: list[str],
    query_ids: list[str] | None = None,
    extract: bool = True,
    extract_depth: str | None = None,
) -> list[NewsBatchFetchPlan]:
    matrix = load_query_matrix(query_matrix_path)
    if matrix is None:
        raise NewsBatchError(f"query matrix does not exist: {query_matrix_path}")
    windows = [parse_date_window(value) for value in date_windows]
    if not windows:
        raise NewsBatchError("at least one date window is required (pass --date-window or --weeks)")

    requested_ids = tuple(dict.fromkeys(query_ids or []))
    if requested_ids:
        missing = [query_id for query_id in requested_ids if query_id not in matrix.entries_by_id]
        if missing:
            raise NewsBatchError(f"unknown query id(s): {', '.join(missing)}")
        entries = [matrix.entries_by_id[query_id] for query_id in requested_ids]
    else:
        entries = list(matrix.entries)

    plans: list[NewsBatchFetchPlan] = []
    for entry in entries:
        for window in windows:
            try:
                config = make_news_fetch_config(
                    query=entry.query,
                    max_results=entry.max_results,
                    topic=entry.topic,
                    time_range=None,
                    search_depth=entry.search_depth,
                    extract=extract,
                    extract_depth=extract_depth or entry.extract_depth,
                    start_date=window.start_date,
                    end_date=window.end_date,
                    include_domains=entry.include_domains,
                    exclude_domains=entry.exclude_domains,
                )
            except ValueError as exc:
                raise NewsBatchError(str(exc)) from exc
            plans.append(NewsBatchFetchPlan(query_entry=entry, date_window=window, config=config))
    return plans


async def run_news_batch(
    *,
    client: Any,
    plans: list[NewsBatchFetchPlan],
    output_dir: str | Path,
) -> NewsBatchResult:
    output_paths: list[NewsOutputPaths] = []
    for plan in plans:
        result = await client.fetch(plan.config)
        output_paths.append(write_news_corpus(result, output_dir))
    return NewsBatchResult(plans=plans, output_paths=output_paths)
