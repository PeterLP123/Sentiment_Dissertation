import json
from pathlib import Path

from sentiment_benchmark.artifact_io import atomic_write_json
from sentiment_benchmark.lseg_catalog import build_lseg_catalog


def test_lseg_catalog_writes_metadata_without_story_text(tmp_path: Path) -> None:
    corpus = tmp_path / "derived" / "corpus_a"
    manifest = corpus / "manifest.json"
    atomic_write_json(
        manifest,
        {
            "schema_version": 1,
            "status": "completed",
            "built_at": "2026-06-26T10:00:00Z",
            "cleaner_version": "lseg_html_v1",
            "config": {
                "collection": {
                    "id": "corpus_a",
                    "start": "2025-06-26T00:00:00Z",
                    "end": "2026-06-26T00:00:00Z",
                    "window_days": 1,
                },
                "companies": [
                    {"symbol": "AAPL"},
                    {"symbol": "MSFT"},
                ],
            },
            "counts": {
                "articles": 10,
                "eligible": 8,
                "ineligible": 2,
                "text_quality": {"ok": 8, "too_short": 2},
            },
            "sharing": {"redistribute": False, "licensed_full_text": True},
            "debug_should_not_appear": "Apple signed a large licensed story body.",
        },
    )

    result = build_lseg_catalog(tmp_path / "derived")

    catalog = json.loads(result.catalog_json.read_text())
    assert result.corpus_count == 1
    assert result.eligible_count == 8
    assert catalog["corpora"][0]["collection_id"] == "corpus_a"
    assert catalog["corpora"][0]["companies"] == ["AAPL", "MSFT"]
    assert catalog["corpora"][0]["text_quality"] == {"ok": 8, "too_short": 2}
    assert "Apple signed a large licensed story body" not in result.catalog_json.read_text()
    assert "Apple signed a large licensed story body" not in result.catalog_csv.read_text()
