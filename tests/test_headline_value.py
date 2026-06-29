from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sentiment_benchmark.headline_value import (
    EVENT_TYPES,
    TRADABILITY_CLASSES,
    analyze_headline_value,
    classify_headline,
    classify_tradability,
    normalize_headline,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fixture_collection(tmp_path: Path) -> Path:
    root = tmp_path / "collection"
    raw = root / "raw" / "lseg_fixture"
    raw.mkdir(parents=True)
    manifest = {
        "status": "in_progress",
        "config": {
            "collection": {"id": "fixture", "start": "2026-06-01T00:00:00Z", "end": "2026-06-03T00:00:00Z"},
            "companies": [
                {"symbol": "AAPL", "name": "Apple Inc.", "aliases": ["Apple", "Apple Inc.", "AAPL"]},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "aliases": ["Microsoft", "MSFT"]},
            ],
        },
    }
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_jsonl(
        raw / "headlines.jsonl",
        [
            {
                "story_id": "urn:newsml:reuters.com:20260601:a:1",
                "headline": "Apple beats earnings estimates",
                "version_created": "2026-06-01T12:00:00",
                "source_code": "NS:RTRS",
                "matched_symbols": ["AAPL"],
            },
            {
                "story_id": "urn:web:20260601:b:1",
                "headline": "Apple beats earnings estimates",
                "version_created": "2026-06-01T13:00:00Z",
                "source_code": "NS:AAA",
                "matched_symbols": ["AAPL"],
            },
            {
                "story_id": "urn:web:20260601:c:1",
                "headline": "Apple shares rise after analyst upgrade",
                "version_created": "2026-06-01T14:00:00Z",
                "source_code": "NS:AAA",
                "matched_symbols": ["AAPL"],
            },
            {
                "story_id": "urn:web:20260601:d:1",
                "headline": "Microsoft faces antitrust probe",
                "version_created": "2026-06-01T15:00:00Z",
                "source_code": "NS:BBB",
                "matched_symbols": ["MSFT"],
            },
            {
                "story_id": "urn:web:20260602:e:1",
                "headline": "Market roundup: technology stocks mixed",
                "version_created": "2026-06-02T09:00:00Z",
                "source_code": "NS:CCC",
                "matched_symbols": ["MSFT"],
            },
            {
                "story_id": "urn:web:20260603:f:1",
                "headline": "Apple launches new product",
                "version_created": "2026-06-03T01:00:00Z",
                "source_code": "NS:AAA",
                "matched_symbols": ["AAPL"],
            },
        ],
    )
    return root


def _price_csv(tmp_path: Path) -> Path:
    path = tmp_path / "prices.csv"
    rows = []
    for symbol in ("AAPL", "MSFT"):
        for day, open_, close in [
            ("2026-06-02", 100.0, 110.0),
            ("2026-06-03", 110.0, 105.0),
            ("2026-06-04", 105.0, 115.0),
            ("2026-06-05", 115.0, 120.0),
        ]:
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": day,
                    "open": open_,
                    "high": max(open_, close),
                    "low": min(open_, close),
                    "close": close,
                    "volume": 1000,
                    "repaired": "False",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_headline_normalization_and_taxonomy_are_stable() -> None:
    assert normalize_headline("Apple  Beats!! Earnings") == "apple beats earnings"
    classification = classify_headline("Apple beats earnings estimates", aliases=("Apple",), duplicate_count=1)
    assert classification.event_type in EVENT_TYPES
    assert classification.event_type == "earnings_guidance"
    assert classification.sentiment_clarity == "clear_positive"
    tradability = classify_tradability(classification, duplicate_count=1, matched_symbol_count=1)
    assert tradability in TRADABILITY_CLASSES
    assert tradability == "single_company_actionable"


def test_analyze_headline_value_writes_safe_outputs_when_prices_missing(tmp_path: Path) -> None:
    root = _fixture_collection(tmp_path)
    output = tmp_path / "out"

    result = analyze_headline_value(root, prices=tmp_path / "missing_prices.csv", output_dir=output, sample_size=4, horizons=(1, 2))

    assert result.trading_status == "blocked_missing_prices"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"]["headline_rows"] == 6
    assert manifest["counts"]["outside_config_window_rows"] == 1
    assert manifest["counts"]["company_day_panel_rows"] == 4
    assert manifest["trading"]["status"] == "blocked_missing_prices"

    panel = _read_csv(output / "company_day_panel.csv")
    assert len(panel) == 4
    apple_day = next(row for row in panel if row["symbol"] == "AAPL" and row["news_date"] == "2026-06-01")
    assert apple_day["headline_count"] == "3"
    assert float(apple_day["duplicate_rate"]) == pytest.approx(2 / 3)

    events = _read_csv(output / "headline_events.csv")
    assert any(row["raw_row_count"] == "2" and row["duplicate_count"] == "1" for row in events)
    assert "Apple beats earnings estimates" not in (output / "summary.md").read_text()
    assert "Apple beats earnings estimates" not in (output / "headline_events.csv").read_text()

    sample = _read_csv(output / "sample_label_template.csv")
    assert sample
    assert "headline" in sample[0]
    assert "manual_event_type" in sample[0]


def test_analyze_headline_value_runs_trading_summary_with_prices(tmp_path: Path) -> None:
    root = _fixture_collection(tmp_path)
    output = tmp_path / "out"
    prices = _price_csv(tmp_path)

    result = analyze_headline_value(root, prices=prices, output_dir=output, sample_size=0, horizons=(1, 2))

    assert result.trading_status == "completed"
    summary = _read_csv(output / "trading_summary.csv")
    assert summary
    assert {row["scorer_id"] for row in summary if row.get("scorer_id")}
    returns = _read_csv(output / "trading_returns.csv")
    assert returns
