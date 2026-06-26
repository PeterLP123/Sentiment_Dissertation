"""Self-contained Week 4 LSEG Workspace news downloader.

Dependencies: pandas and lseg-data. LSEG Workspace Desktop must be open and
signed in before running a download.
"""

from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_QUERY = "R:MSFT.O and Language:LEN"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_WINDOW_DAYS = 14

OUTPUT_COLUMNS = [
    "timestamp",
    "request_date",
    "request_start",
    "request_end",
    "company_symbol",
    "ric",
    "query",
    "storyId",
    "story_type",
    "sourceCode",
    "headline",
    "story_html",
    "story_error",
]

COMPANY_RICS = {
    "AAPL": "AAPL.O",
    "AMZN": "AMZN.O",
    "GOOGL": "GOOGL.O",
    "JPM": "JPM.N",
    "META": "META.O",
    "MSFT": "MSFT.O",
    "NVDA": "NVDA.O",
    "TSLA": "TSLA.O",
}
ld = None


def lseg_data():
    global ld
    if ld is None:
        try:
            import lseg.data as loaded_lseg_data
        except ImportError as exc:
            raise RuntimeError("Install lseg-data before downloading news: python -m pip install lseg-data") from exc
        ld = loaded_lseg_data
    return ld


def default_date_window(days: int = DEFAULT_WINDOW_DAYS) -> tuple[str, str]:
    end_day = date.today() - timedelta(days=1)
    start_day = end_day - timedelta(days=days - 1)
    return f"{start_day.isoformat()}T00:00:00", f"{end_day.isoformat()}T23:59:59"


def default_output_path(query: str, start: str, end: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", query).strip("_")[:80] or "lseg_news"
    start_slug = re.sub(r"[^0-9]+", "", start)[:8] or "start"
    end_slug = re.sub(r"[^0-9]+", "", end)[:8] or "end"
    return DEFAULT_OUTPUT_DIR / f"{slug}_{start_slug}_{end_slug}.csv"


def date_from_iso_datetime(value: str, *, name: str) -> date:
    match = re.match(r"^\s*(\d{4}-\d{2}-\d{2})", value)
    if not match:
        raise ValueError(f"{name} must start with YYYY-MM-DD")
    return date.fromisoformat(match.group(1))


def daily_windows(start: str, end: str) -> list[tuple[date, str, str]]:
    start_day = date_from_iso_datetime(start, name="--start")
    end_day = date_from_iso_datetime(end, name="--end")
    if start_day > end_day:
        raise ValueError("--start date must be on or before --end date")

    windows = []
    current = start_day
    while current <= end_day:
        day = current.isoformat()
        windows.append((current, f"{day}T00:00:00", f"{day}T23:59:59"))
        current += timedelta(days=1)
    return windows


def story_id_is_present(story_id: object) -> bool:
    return bool(pd.notna(story_id) and str(story_id).strip())


def story_type_from_id(story_id: object) -> str:
    if not story_id_is_present(story_id):
        return "unknown"
    value = str(story_id)
    if value.startswith("urn:link:webnews:"):
        return "webnews"
    if value.startswith("urn:newsml:social:"):
        return "social"
    if value.startswith("urn:newsml:reuters.com:"):
        return "reuters"
    if value.startswith("urn:newsml:newsroom:"):
        return "newsroom"
    if value.startswith("urn:"):
        parts = value.split(":")
        return ":".join(parts[1:3]) if len(parts) >= 3 else parts[1]
    return "unknown"


def query_for_company(company: str) -> tuple[str, str, str]:
    symbol = company.strip().upper()
    ric = COMPANY_RICS.get(symbol, company.strip())
    query = f"R:{ric} and Language:LEN"
    return symbol, ric, query


def built_in_companies() -> list[tuple[str, str, str]]:
    return [query_for_company(symbol) for symbol in sorted(COMPANY_RICS)]


def fetch_news(
    query: str,
    count: int,
    start: str,
    end: str,
    *,
    company_symbol: str = "",
    ric: str = "",
    request_date: date | None = None,
) -> pd.DataFrame:
    client = lseg_data()
    headlines = client.news.get_headlines(query=query, start=start, end=end, count=count)
    records: list[dict[str, object]] = []

    for timestamp, row in headlines.iterrows():
        story_id = row.get("storyId")
        story_html = None
        story_error = None
        if story_id_is_present(story_id):
            try:
                story_html = client.news.get_story(str(story_id))
            except Exception as exc:
                story_error = str(exc)

        records.append(
            {
                "timestamp": timestamp,
                "request_date": request_date.isoformat() if request_date else "",
                "request_start": start,
                "request_end": end,
                "company_symbol": company_symbol,
                "ric": ric,
                "query": query,
                "storyId": story_id,
                "story_type": story_type_from_id(story_id),
                "sourceCode": row.get("sourceCode"),
                "headline": row.get("headline"),
                "story_html": story_html,
                "story_error": story_error,
            }
        )

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def fetch_daily_news(
    query: str,
    per_day_count: int,
    start: str,
    end: str,
    *,
    company_symbol: str = "",
    ric: str = "",
) -> pd.DataFrame:
    frames = [
        fetch_news(
            query,
            per_day_count,
            request_start,
            request_end,
            company_symbol=company_symbol,
            ric=ric,
            request_date=request_date,
        )
        for request_date, request_start, request_end in daily_windows(start, end)
    ]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def dedupe_story_ids(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if dataframe.empty or "storyId" not in dataframe.columns:
        return dataframe, 0

    before = len(dataframe)
    story_ids = dataframe["storyId"].astype("string").str.strip()
    has_story_id = story_ids.notna() & story_ids.ne("")
    duplicate = has_story_id & dataframe.duplicated(subset=["company_symbol", "storyId"], keep="first")
    deduped = dataframe.loc[~duplicate].copy()
    return deduped, before - len(deduped)


def download_job(
    company_symbol: str,
    ric: str,
    query: str,
    output: Path,
    count: int,
    start: str,
    end: str,
    *,
    per_day_count: int | None = None,
    dedupe: bool = False,
) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)
    if per_day_count is None:
        dataframe = fetch_news(query, count, start, end, company_symbol=company_symbol, ric=ric)
    else:
        dataframe = fetch_daily_news(query, per_day_count, start, end, company_symbol=company_symbol, ric=ric)

    deduped_count = 0
    if dedupe:
        dataframe, deduped_count = dedupe_story_ids(dataframe)

    dataframe.to_csv(output, index=False)
    story_errors = int(dataframe["story_error"].notna().sum()) if "story_error" in dataframe else 0
    dedupe_note = f", {deduped_count} duplicate story rows removed" if dedupe else ""
    print(f"Wrote {len(dataframe)} rows to {output} ({story_errors} story errors{dedupe_note})")
    return dataframe


def run_downloads(
    jobs: list[tuple[str, str, str, Path]],
    count: int,
    start: str,
    end: str,
    *,
    per_day_count: int | None = None,
    dedupe: bool = False,
) -> None:
    client = lseg_data()
    client.open_session()
    try:
        for company_symbol, ric, query, output in jobs:
            download_job(
                company_symbol,
                ric,
                query,
                output,
                count,
                start,
                end,
                per_day_count=per_day_count,
                dedupe=dedupe,
            )
    finally:
        close_session = getattr(client, "close_session", None)
        if callable(close_session):
            close_session()


def parse_args() -> argparse.Namespace:
    default_start, default_end = default_date_window()
    parser = argparse.ArgumentParser(description="Download LSEG Workspace news headlines and full stories to CSV.")
    parser.add_argument("--query", help=f"LSEG news query. Default: {DEFAULT_QUERY}")
    parser.add_argument("--company", help="Week 4 shortcut for one company ticker, such as MSFT.")
    parser.add_argument(
        "--all-companies",
        action="store_true",
        help="Week 4 batch mode: download the built-in company list.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of headlines to request in one whole-window call. Use 50-100 per company.",
    )
    parser.add_argument(
        "--per-day-count",
        type=int,
        help="Week 4 balanced mode: request this many headlines for each calendar day in --start/--end. Use about 5.",
    )
    parser.add_argument(
        "--dedupe-story-ids",
        action="store_true",
        help="After fetching, remove duplicate rows with the same company_symbol and storyId.",
    )
    parser.add_argument("--start", default=default_start, help="Start datetime passed to LSEG.")
    parser.add_argument("--end", default=default_end, help="End datetime passed to LSEG.")
    parser.add_argument("--output", type=Path, help="CSV destination. Defaults to week_4_tasks/outputs/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.count <= 100:
        raise ValueError("--count should be between 1 and 100 for this Week 4 script")
    if args.per_day_count is not None and not 1 <= args.per_day_count <= 100:
        raise ValueError("--per-day-count should be between 1 and 100 for this Week 4 script")
    if args.output and args.all_companies:
        raise ValueError("--output can only be used with one --query or --company download")
    if sum(bool(value) for value in (args.query, args.company, args.all_companies)) > 1:
        raise ValueError("Use only one of --query, --company, or --all-companies")

    if args.per_day_count is None:
        print(f"Date window: {args.start} to {args.end} (count={args.count})")
    else:
        days = len(daily_windows(args.start, args.end))
        expected = days * args.per_day_count
        print(
            f"Date window: {args.start} to {args.end} "
            f"(balanced daily mode: {args.per_day_count}/day across {days} days, up to {expected} rows per job)"
        )

    if args.all_companies:
        jobs = [(symbol, ric, query, default_output_path(symbol, args.start, args.end)) for symbol, ric, query in built_in_companies()]
        print(f"Downloading news for {len(jobs)} built-in companies")
    elif args.company:
        symbol, ric, query = query_for_company(args.company)
        jobs = [(symbol, ric, query, args.output or default_output_path(symbol, args.start, args.end))]
    else:
        query = args.query or DEFAULT_QUERY
        jobs = [("", "", query, args.output or default_output_path(query, args.start, args.end))]

    run_downloads(
        jobs,
        args.count,
        args.start,
        args.end,
        per_day_count=args.per_day_count,
        dedupe=args.dedupe_story_ids,
    )


if __name__ == "__main__":
    main()
