from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from sentiment_benchmark.fnspid_feasibility import _timestamp_parts, audit_fnspid_news


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
