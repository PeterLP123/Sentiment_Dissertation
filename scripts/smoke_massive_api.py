#!/usr/bin/env python3
"""Disposable Massive API smoke test for the dissertation study window.

The script deliberately saves a compact diagnostic report rather than a new
research dataset. It checks whether Massive can supply ticker-linked news
metadata and matching adjusted daily stock bars for a small sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

API_ROOT = "https://api.massive.com"
DEFAULT_FROM_DATE = "2025-12-26"
DEFAULT_TO_DATE = "2026-06-26"


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the shell environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Massive historical news and daily stock bars without modifying research data.",
    )
    parser.add_argument("--tickers", nargs="+", default=["AAPL"], help="Stock tickers to test (default: AAPL).")
    parser.add_argument("--from-date", default=DEFAULT_FROM_DATE, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--to-date", default=DEFAULT_TO_DATE, help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--news-limit", type=int, default=100, help="News rows requested per page (1-1000; default: 100).")
    parser.add_argument("--max-news-pages", type=int, default=1, help="Maximum news pages per ticker (default: 1).")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds (default: 30).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path (default: ignored results/massive_smoke_<timestamp>.json).",
    )
    parser.add_argument(
        "--save-rows",
        action="store_true",
        help="Include returned news and price rows in the ignored diagnostic report.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    try:
        start = datetime.strptime(args.from_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.to_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {exc}") from None
    if start > end:
        raise SystemExit("--from-date must not be after --to-date")
    if not 1 <= args.news_limit <= 1000:
        raise SystemExit("--news-limit must be between 1 and 1000")
    if args.max_news_pages < 1:
        raise SystemExit("--max-news-pages must be at least 1")
    args.tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in args.tickers if ticker.strip()))
    if not args.tickers:
        raise SystemExit("At least one non-empty ticker is required")


def with_api_key(url: str, api_key: str) -> str:
    """Attach the API key to first-page and Massive-provided pagination URLs."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["apiKey"] = api_key
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def safe_error(response: httpx.Response) -> str:
    """Extract a short API error without ever printing the authenticated URL."""
    try:
        payload = response.json()
    except ValueError:
        payload = response.text.strip()
    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("message") or payload.get("status") or payload
    else:
        detail = payload
    return str(detail)[:500]


def get_json(client: httpx.Client, url: str, api_key: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        response = client.get(with_api_key(url, api_key))
    except httpx.RequestError as exc:
        return None, {
            "ok": False,
            "status_code": None,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "error": f"{type(exc).__name__}: request failed before a response was received",
        }

    diagnostic: dict[str, Any] = {
        "ok": response.is_success,
        "status_code": response.status_code,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request_id": response.headers.get("x-request-id"),
    }
    if not response.is_success:
        diagnostic["error"] = safe_error(response)
        return None, diagnostic
    try:
        payload = response.json()
    except ValueError:
        diagnostic["ok"] = False
        diagnostic["error"] = "Successful HTTP response was not valid JSON"
        return None, diagnostic
    if not isinstance(payload, dict):
        diagnostic["ok"] = False
        diagnostic["error"] = "Expected a JSON object response"
        return None, diagnostic
    diagnostic["request_id"] = payload.get("request_id") or diagnostic["request_id"]
    return payload, diagnostic


def news_summary(rows: list[dict[str, Any]], ticker: str) -> dict[str, Any]:
    timestamps = sorted(str(row["published_utc"]) for row in rows if row.get("published_utc"))
    publishers = Counter(
        str(publisher.get("name")) for row in rows if isinstance((publisher := row.get("publisher")), dict) and publisher.get("name")
    )
    count = len(rows)

    def share(field: str) -> float:
        return round(sum(bool(row.get(field)) for row in rows) / count, 3) if count else 0.0

    ticker_tagged = sum(ticker in (row.get("tickers") or []) for row in rows)
    return {
        "rows": count,
        "oldest_published_utc": timestamps[0] if timestamps else None,
        "newest_published_utc": timestamps[-1] if timestamps else None,
        "unique_article_ids": len({row.get("id") for row in rows if row.get("id")}),
        "description_share": share("description"),
        "insights_share": share("insights"),
        "article_url_share": share("article_url"),
        "requested_ticker_tag_share": round(ticker_tagged / count, 3) if count else 0.0,
        "top_publishers": dict(publishers.most_common(10)),
        "sample_titles": [str(row.get("title", ""))[:180] for row in rows[:5]],
    }


def fetch_news(
    client: httpx.Client,
    api_key: str,
    ticker: str,
    from_date: str,
    to_date: str,
    limit: int,
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = urlencode(
        {
            "ticker": ticker,
            "published_utc.gte": from_date,
            "published_utc.lte": f"{to_date}T23:59:59Z",
            "order": "asc",
            "sort": "published_utc",
            "limit": limit,
        }
    )
    next_url: str | None = f"{API_ROOT}/v2/reference/news?{query}"
    rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    while next_url and len(requests) < max_pages:
        payload, diagnostic = get_json(client, next_url, api_key)
        requests.append(diagnostic)
        if payload is None:
            next_url = None
            break
        page_rows = payload.get("results") or []
        if not isinstance(page_rows, list):
            requests[-1]["ok"] = False
            requests[-1]["error"] = "News results was not an array"
            next_url = None
            break
        rows.extend(row for row in page_rows if isinstance(row, dict))
        next_url = payload.get("next_url")

    diagnostic = {
        "requests": requests,
        "pages_fetched": len(requests),
        "more_pages_available": bool(next_url),
        "summary": news_summary(rows, ticker),
    }
    return rows, diagnostic


def bars_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dated_rows = sorted((int(row["t"]), row) for row in rows if row.get("t") is not None)
    first_close = dated_rows[0][1].get("c") if dated_rows else None
    last_close = dated_rows[-1][1].get("c") if dated_rows else None
    total_return = None
    if isinstance(first_close, (int, float)) and isinstance(last_close, (int, float)) and first_close:
        total_return = round(last_close / first_close - 1, 6)
    return {
        "rows": len(rows),
        "first_bar_utc": datetime.fromtimestamp(dated_rows[0][0] / 1000, UTC).isoformat() if dated_rows else None,
        "last_bar_utc": datetime.fromtimestamp(dated_rows[-1][0] / 1000, UTC).isoformat() if dated_rows else None,
        "first_close": first_close,
        "last_close": last_close,
        "close_to_close_return": total_return,
        "adjusted_ohlcv_complete_share": round(
            sum(all(row.get(field) is not None for field in ("o", "h", "l", "c", "v")) for row in rows) / len(rows),
            3,
        )
        if rows
        else 0.0,
    }


def fetch_bars(
    client: httpx.Client,
    api_key: str,
    ticker: str,
    from_date: str,
    to_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = urlencode({"adjusted": "true", "sort": "asc", "limit": 50000})
    url = f"{API_ROOT}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}?{query}"
    payload, request = get_json(client, url, api_key)
    rows: list[dict[str, Any]] = []
    if payload is not None:
        candidate_rows = payload.get("results") or []
        if isinstance(candidate_rows, list):
            rows = [row for row in candidate_rows if isinstance(row, dict)]
        else:
            request["ok"] = False
            request["error"] = "Aggregate results was not an array"
    return rows, {"request": request, "summary": bars_summary(rows)}


def verdict(ticker_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    news_access = sum(bool(report["news"]["requests"] and report["news"]["requests"][0]["ok"]) for report in ticker_reports.values())
    bars_access = sum(report["bars"]["request"]["ok"] for report in ticker_reports.values())
    news_rows = sum(report["news"]["summary"]["rows"] for report in ticker_reports.values())
    bar_rows = sum(report["bars"]["summary"]["rows"] for report in ticker_reports.values())
    descriptions = [
        report["news"]["summary"]["description_share"] for report in ticker_reports.values() if report["news"]["summary"]["rows"]
    ]
    mean_description_share = round(sum(descriptions) / len(descriptions), 3) if descriptions else 0.0

    if news_access == len(ticker_reports) and bars_access == len(ticker_reports) and news_rows > 0 and bar_rows > 0:
        label = "promising" if mean_description_share >= 0.5 else "mixed"
        reason = "Both dissertation-relevant endpoints work; inspect coverage and descriptions before buying."
    elif news_access or bars_access:
        label = "limited"
        reason = "Only part of the required data was usable; inspect endpoint errors and plan access."
    else:
        label = "blocked"
        reason = "Neither endpoint produced usable data; check authentication and plan permissions."
    return {
        "label": label,
        "reason": reason,
        "news_rows": news_rows,
        "daily_bar_rows": bar_rows,
        "mean_description_share": mean_description_share,
        "important_limit": (
            "The news endpoint supplies metadata, titles, descriptions, URLs, and optional provider sentiment—not article bodies."
        ),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    load_dotenv(Path(".env"))
    api_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not api_key:
        print(
            "Missing MASSIVE_API_KEY. Add MASSIVE_API_KEY=... to .env or export it in the shell, then rerun.",
            file=sys.stderr,
        )
        return 2

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results") / f"massive_smoke_{timestamp}.json"
    ticker_reports: dict[str, dict[str, Any]] = {}

    with httpx.Client(
        timeout=args.timeout,
        follow_redirects=True,
        headers={"User-Agent": "sentiment-dissertation-massive-smoke/1"},
    ) as client:
        for ticker in args.tickers:
            print(f"Testing {ticker}: historical news...", flush=True)
            news_rows, news_diagnostic = fetch_news(
                client,
                api_key,
                ticker,
                args.from_date,
                args.to_date,
                args.news_limit,
                args.max_news_pages,
            )
            print(f"Testing {ticker}: adjusted daily bars...", flush=True)
            bar_rows, bars_diagnostic = fetch_bars(client, api_key, ticker, args.from_date, args.to_date)
            ticker_reports[ticker] = {"news": news_diagnostic, "bars": bars_diagnostic}
            if args.save_rows:
                ticker_reports[ticker]["news"]["rows"] = news_rows
                ticker_reports[ticker]["bars"]["rows"] = bar_rows

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Disposable purchase-decision smoke test; not a dissertation dataset.",
        "parameters": {
            "tickers": args.tickers,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "news_limit": args.news_limit,
            "max_news_pages": args.max_news_pages,
            "adjusted_daily_bars": True,
            "saved_rows": args.save_rows,
        },
        "tickers": ticker_reports,
        "verdict": verdict(ticker_reports),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report["verdict"], indent=2))
    print(f"Diagnostic report: {output.resolve()}")
    return 0 if report["verdict"]["label"] in {"promising", "mixed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
