from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from sentiment_benchmark.fnspid_feasibility import FnspidAuditConfig, _timestamp_parts, audit_fnspid_news


def _archive(path: Path, rows: list[dict[str, str]]) -> None:
    zstd = shutil.which("zstd")
    if zstd is None:
        pytest.skip("zstd is required")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=["Date", "Article_title", "Stock_symbol", "Url"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    subprocess.run([zstd, "-q", "-3", "-o", str(path)], input=text.getvalue().encode(), check=True)


def test_midnight_timestamp_is_not_primary_mappable() -> None:
    assert _timestamp_parts("2020-01-02 00:00:00 UTC") == ("2020-01-02", "2020-01", "2020", False)
    assert _timestamp_parts("2020-01-02 12:34:56 UTC") == ("2020-01-02", "2020-01", "2020", True)
    assert _timestamp_parts("not-a-date") is None


def test_streaming_audit_deduplicates_firm_events_and_preserves_cross_symbol_associations(tmp_path: Path) -> None:
    archive = tmp_path / "news.csv.zst"
    _archive(
        archive,
        [
            {
                "Date": "2020-01-02 12:00:00 UTC",
                "Article_title": "Shared story",
                "Stock_symbol": "AAA",
                "Url": "https://www.example.com/shared",
            },
            {
                "Date": "2020-01-02 12:00:00 UTC",
                "Article_title": "Shared story",
                "Stock_symbol": "AAA",
                "Url": "https://www.example.com/shared",
            },
            {
                "Date": "2020-01-02 12:00:00 UTC",
                "Article_title": "Shared story",
                "Stock_symbol": "BBB",
                "Url": "https://www.example.com/shared",
            },
            {
                "Date": "2020-01-03 00:00:00 UTC",
                "Article_title": "Ambiguous timestamp",
                "Stock_symbol": "AAA",
                "Url": "https://www.example.com/midnight",
            },
        ],
    )
    output = tmp_path / "audit"
    manifest_path = audit_fnspid_news((archive,), output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["rows"] == 4
    assert manifest["counts"]["mappable_deduplicated_firm_events"] == 2
    assert manifest["counts"]["exact_firm_event_duplicates"] == 1
    assert manifest["counts"]["cross_symbol_story_associations"] == 1
    assert manifest["counts"]["date_only_or_exact_midnight_rows"] == 1
    coverage = pd.read_csv(output / "symbol_year_coverage.csv")
    assert coverage.set_index("symbol")["mappable_news_days"].to_dict() == {"AAA": 1, "BBB": 1}


def test_date_only_policy_assigns_strictly_to_next_trading_session(tmp_path: Path) -> None:
    archive = tmp_path / "news.csv.zst"
    _archive(
        archive,
        [
            {
                "Date": "2020-01-03 00:00:00 UTC",
                "Article_title": "Friday date-only story",
                "Stock_symbol": "AAA",
                "Url": "https://example.com/friday",
            },
            {
                "Date": "2020-01-04",
                "Article_title": "Saturday date-only story",
                "Stock_symbol": "AAA",
                "Url": "https://example.com/saturday",
            },
        ],
    )
    output = tmp_path / "next-session-audit"
    manifest_path = audit_fnspid_news(
        (archive,),
        output,
        config=FnspidAuditConfig(date_only_policy="next_trading_session"),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["date_only_rows_assigned_next_session"] == 2
    assert manifest["counts"]["mappable_deduplicated_firm_events"] == 2
    daily = pd.read_csv(output / "daily_event_counts.csv")
    assert daily.to_dict("records") == [{"date": "2020-01-06", "mappable_deduplicated_events": 2}]
    coverage = pd.read_csv(output / "symbol_year_coverage.csv")
    assert coverage.to_dict("records") == [{"symbol": "AAA", "year": 2020, "mappable_news_days": 1}]


def test_full_datetime_policy_applies_exchange_close_rule(tmp_path: Path) -> None:
    archive = tmp_path / "news.csv.zst"
    _archive(
        archive,
        [
            {
                "Date": "2020-01-03 15:00:00 UTC",
                "Article_title": "Friday in-session story",
                "Stock_symbol": "AAA",
                "Url": "https://example.com/in-session",
            },
            {
                "Date": "2020-01-03 22:00:00 UTC",
                "Article_title": "Friday after-close story",
                "Stock_symbol": "AAA",
                "Url": "https://example.com/after-close",
            },
        ],
    )
    output = tmp_path / "close-rule-audit"
    audit_fnspid_news(
        (archive,),
        output,
        config=FnspidAuditConfig(full_datetime_policy="session_close"),
    )
    daily = pd.read_csv(output / "daily_event_counts.csv")
    assert daily.to_dict("records") == [
        {"date": "2020-01-03", "mappable_deduplicated_events": 1},
        {"date": "2020-01-06", "mappable_deduplicated_events": 1},
    ]


def test_candidate_window_requires_same_firms_in_every_year(tmp_path: Path) -> None:
    archive = tmp_path / "news.csv.zst"
    rows: list[dict[str, str]] = []
    symbols_by_year = {2020: ("AAA", "BBB"), 2021: ("AAA", "CCC"), 2022: ("AAA", "DDD")}
    for year, symbols in symbols_by_year.items():
        for symbol in symbols:
            rows.append(
                {
                    "Date": f"{year}-06-01 12:00:00 UTC",
                    "Article_title": f"{symbol} story {year}",
                    "Stock_symbol": symbol,
                    "Url": f"https://example.com/{symbol}/{year}",
                }
            )
    _archive(archive, rows)
    output = tmp_path / "coherent-window-audit"
    audit_fnspid_news(
        (archive,),
        output,
        config=FnspidAuditConfig(
            minimum_firms=2,
            minimum_news_days_per_firm_year=1,
            minimum_window_years=3,
            minimum_events=1,
        ),
    )
    candidates = pd.read_csv(output / "candidate_windows.csv")
    assert candidates.loc[0, "minimum_eligible_firms_in_any_year"] == 2
    assert candidates.loc[0, "firms_eligible_in_every_year"] == 1
    assert not bool(candidates.loc[0, "dimension_gate_pass"])
