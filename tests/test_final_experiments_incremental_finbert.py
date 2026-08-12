from __future__ import annotations

import json
from pathlib import Path

from final_experiments.lib.incremental_finbert import materialize_incremental_finbert_snapshot


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_snapshot_includes_only_terminal_company_dates(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[collection]
id = "fixture"
start = "2024-01-01T00:00:00Z"
end = "2024-01-03T00:00:00Z"
language = "en"
session = "desktop"
page_size = 100
max_pages = 50
retries = 3
requests_per_second = 3.0
max_requests_per_run = 9500
window_days = 1
fetch_story_bodies = false

[cleaning]
min_text_chars = 0
max_scoring_chars = 1000

[outputs]
raw_output_root = "Data/raw"
derived_output_root = "Data/derived"

[[companies]]
symbol = "AAPL"
name = "Apple Inc."
ric = "AAPL.O"
news_query = "R:AAPL.O and Language:LEN"
aliases = ["Apple", "AAPL"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "Data/raw/lseg_fixture"
    _write(raw_dir / "manifest.json", {"status": "in_progress", "config": {}})
    pages = raw_dir / "headline_pages"
    common = {
        "query": "R:AAPL.O and Language:LEN",
        "symbol": "AAPL",
        "ric": "AAPL.O",
        "page_number": 1,
        "cursor_in": None,
    }
    _write(
        pages / "complete.json",
        {
            **common,
            "window_index": 1,
            "window_start": "2024-01-01T00:00:00Z",
            "window_end": "2024-01-02T00:00:00Z",
            "cursor_out": None,
            "rows": [
                {
                    "storyId": "story-1",
                    "headline": "Apple raises its revenue outlook",
                    "firstCreated": "2024-01-01T12:00:00Z",
                    "versionCreated": "2024-01-01T12:00:00Z",
                    "sourceCode": "NS:RTRS",
                }
            ],
        },
    )
    _write(
        pages / "incomplete.json",
        {
            **common,
            "window_index": 2,
            "window_start": "2024-01-02T00:00:00Z",
            "window_end": "2024-01-03T00:00:00Z",
            "cursor_out": "next-page",
            "rows": [
                {
                    "storyId": "story-2",
                    "headline": "Apple starts a new programme",
                    "firstCreated": "2024-01-02T12:00:00Z",
                    "versionCreated": "2024-01-02T12:00:00Z",
                    "sourceCode": "NS:RTRS",
                }
            ],
        },
    )

    summary = materialize_incremental_finbert_snapshot(tmp_path, config_path, tmp_path / "snapshot")

    rows = [json.loads(line) for line in summary.headlines_path.read_text(encoding="utf-8").splitlines()]
    assert summary.completed_company_dates == 1
    assert summary.source_pages == 1
    assert summary.unique_story_ids == 1
    assert summary.unique_scorable_headlines == 1
    assert [row["story_id"] for row in rows] == ["story-1"]
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["snapshot"]["partial_collection"] is True
    assert manifest["snapshot"]["prices_or_returns_loaded"] is False
    assert manifest["snapshot"]["population_sha256"] == summary.population_sha256

    incomplete_path = pages / "incomplete.json"
    completed_payload = json.loads(incomplete_path.read_text(encoding="utf-8"))
    completed_payload["cursor_out"] = None
    _write(incomplete_path, completed_payload)
    complete_summary = materialize_incremental_finbert_snapshot(
        tmp_path,
        config_path,
        tmp_path / "complete_snapshot",
    )
    complete_manifest = json.loads(complete_summary.manifest_path.read_text(encoding="utf-8"))
    assert complete_summary.completed_company_dates == 2
    assert complete_manifest["snapshot"]["partial_collection"] is False
