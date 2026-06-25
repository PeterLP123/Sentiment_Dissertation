from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import lseg.data as ld
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
COMPANIES_CSV = Path(__file__).resolve().parent / "week4_companies.csv"
DEFAULT_WINDOW_DAYS = 14


def default_date_window(days: int = DEFAULT_WINDOW_DAYS) -> tuple[str, str]:
    """Return start/end for ~5 stories/day over a 50-100 story batch window."""
    end_day = date.today() - timedelta(days=1)
    start_day = end_day - timedelta(days=days - 1)
    return f"{start_day.isoformat()}T00:00:00", f"{end_day.isoformat()}T23:59:59"


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


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "lseg_news"


def default_output_path(symbol: str, start: str, end: str) -> Path:
    start_slug = re.sub(r"[^0-9]+", "", start)[:8] or "start"
    end_slug = re.sub(r"[^0-9]+", "", end)[:8] or "end"
    return OUTPUT_DIR / f"{safe_filename(symbol)}_{start_slug}_{end_slug}.csv"


def query_for_company(company: str) -> tuple[str, str, str]:
    symbol = company.strip().upper()
    ric = COMPANY_RICS.get(symbol, company.strip())
    query = f"R:{ric} and Language:LEN"
    return symbol, ric, query


def load_companies(path: Path = COMPANIES_CSV) -> list[dict[str, str]]:
    dataframe = pd.read_csv(path)
    required = {"symbol", "ric", "query"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Companies CSV missing columns: {sorted(missing)}")

    companies: list[dict[str, str]] = []
    for _, row in dataframe.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        if not symbol or symbol.lower() == "nan":
            continue
        companies.append(
            {
                "symbol": symbol,
                "ric": str(row["ric"]).strip(),
                "query": str(row["query"]).strip(),
            }
        )

    if not companies:
        raise ValueError(f"No companies found in {path}")
    return companies


def fetch_news(
    query: str,
    count: int,
    start: str,
    end: str,
    *,
    company_symbol: str = "",
    ric: str = "",
) -> pd.DataFrame:
    headlines = ld.news.get_headlines(
        query=query,
        start=start,
        end=end,
        count=count,
    )

    records = []
    for timestamp, row in headlines.iterrows():
        story_id = row.get("storyId")
        story = None
        story_error = None
        if story_id:
            try:
                story = ld.news.get_story(str(story_id))
            except Exception as exc:
                story_error = str(exc)

        records.append(
            {
                "timestamp": timestamp,
                "company_symbol": company_symbol,
                "ric": ric,
                "query": query,
                "storyId": story_id,
                "sourceCode": row.get("sourceCode"),
                "headline": row.get("headline"),
                "story": story,
                "story_error": story_error,
            }
        )

    return pd.DataFrame(records)


def get_news(
    query: str,
    count: int,
    start: str,
    end: str,
    output: Path,
    *,
    company_symbol: str = "",
    ric: str = "",
) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)

    ld.open_session()
    try:
        dataframe = fetch_news(
            query,
            count,
            start,
            end,
            company_symbol=company_symbol,
            ric=ric,
        )
        dataframe.to_csv(output, index=False)
        return dataframe
    finally:
        close_session = getattr(ld, "close_session", None)
        if callable(close_session):
            close_session()


def run_downloads(
    jobs: list[tuple[str, str, str, Path]],
    count: int,
    start: str,
    end: str,
) -> None:
    ld.open_session()
    try:
        for company_symbol, ric, query, output in jobs:
            output.parent.mkdir(parents=True, exist_ok=True)
            dataframe = fetch_news(
                query,
                count,
                start,
                end,
                company_symbol=company_symbol,
                ric=ric,
            )
            dataframe.to_csv(output, index=False)
            story_errors = int(dataframe["story_error"].notna().sum()) if "story_error" in dataframe else 0
            print(f"Wrote {len(dataframe)} rows to {output} ({story_errors} story errors)")
    finally:
        close_session = getattr(ld, "close_session", None)
        if callable(close_session):
            close_session()


def parse_args() -> argparse.Namespace:
    default_start, default_end = default_date_window()
    parser = argparse.ArgumentParser(description="Simple Week 4 LSEG/Eikon news downloader.")
    parser.add_argument(
        "--company",
        help="Download one company only. If omitted, all rows in week4_companies.csv are downloaded.",
    )
    parser.add_argument(
        "--companies-csv",
        type=Path,
        default=COMPANIES_CSV,
        help="CSV with symbol, ric, and query columns for batch downloads.",
    )
    parser.add_argument("--query", help="Direct LSEG query. If set, this overrides --company.")
    parser.add_argument("--count", type=int, default=50, help="Number of headlines to request. Use 50-100 per company.")
    parser.add_argument(
        "--start",
        default=default_start,
        help=f"Start datetime passed to LSEG. Default: {DEFAULT_WINDOW_DAYS} days before --end.",
    )
    parser.add_argument(
        "--end",
        default=default_end,
        help="End datetime passed to LSEG. Default: yesterday 23:59:59.",
    )
    parser.add_argument("--output", type=Path, help="CSV destination. Defaults to week_4_tasks/outputs/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.count <= 100:
        raise ValueError("--count should be between 1 and 100 for this Week 4 script")

    if args.output and not args.company and not args.query:
        raise ValueError("--output can only be used with --company or --query")

    print(f"Date window: {args.start} to {args.end} (count={args.count} per company)")

    if args.query:
        company_symbol = safe_filename(args.company or "custom").upper()
        ric = ""
        query = args.query
        output = args.output or default_output_path(company_symbol, args.start, args.end)
        run_downloads([(company_symbol, ric, query, output)], args.count, args.start, args.end)
        return

    if args.company:
        company_symbol, ric, query = query_for_company(args.company)
        output = args.output or default_output_path(company_symbol, args.start, args.end)
        run_downloads([(company_symbol, ric, query, output)], args.count, args.start, args.end)
        return

    companies = load_companies(args.companies_csv)
    jobs = [
        (
            company["symbol"],
            company["ric"],
            company["query"],
            default_output_path(company["symbol"], args.start, args.end),
        )
        for company in companies
    ]
    print(f"Downloading news for {len(jobs)} companies from {args.companies_csv}")
    run_downloads(jobs, args.count, args.start, args.end)


if __name__ == "__main__":
    main()
