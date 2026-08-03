import json
import runpy
from pathlib import Path

import pytest

_MERGER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "merge_lseg_headline_collections.py"))
build = _MERGER["build"]
sha256_file = _MERGER["sha256_file"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _collection(tmp_path: Path, collection_id: str, rows: list[dict]) -> Path:
    source = tmp_path / collection_id
    source.mkdir()
    headlines = source / "headlines.jsonl"
    _write_jsonl(headlines, rows)
    manifest = {
        "status": "completed",
        "config": {"collection": {"id": collection_id, "start": "2026-01-01", "end": "2026-02-01"}},
        "counts": {"headlines": len(rows)},
        "files": {"headlines_jsonl": {"path": headlines.name, "sha256": sha256_file(headlines)}},
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_build_merges_story_associations_and_is_idempotent(tmp_path: Path) -> None:
    shared_raw = {"storyId": "story-1", "headline": "Shared"}
    first = _collection(
        tmp_path,
        "first",
        [
            {
                "story_id": "story-1",
                "headline": "Short",
                "first_created": "2026-01-02T09:00:00",
                "version_created": "2026-01-02T09:05:00",
                "source_code": "NS:RTRS",
                "matched_symbols": ["AAA"],
                "matched_rics": ["AAA.N"],
                "matched_queries": ["R:AAA.N"],
                "raw_rows": [shared_raw],
            },
            {"story_id": "story-2", "headline": "Only first", "matched_symbols": ["AAA"]},
        ],
    )
    second = _collection(
        tmp_path,
        "second",
        [
            {
                "story_id": "story-1",
                "headline": "A longer shared headline",
                "first_created": "2026-01-02T08:55:00",
                "version_created": "2026-01-02T09:10:00",
                "source_code": "NS:RTRS",
                "matched_symbols": ["BBB"],
                "matched_rics": ["BBB.O"],
                "matched_queries": ["R:BBB.O"],
                "raw_rows": [shared_raw, {"storyId": "story-1", "headline": "Revised"}],
            }
        ],
    )
    output = tmp_path / "merged"

    manifest = build([first, second], output)
    rows = [json.loads(line) for line in (output / "headlines.jsonl").read_text().splitlines()]

    assert manifest["status"] == "completed"
    assert manifest["counts"] == {
        "duplicate_story_ids_across_collections": 1,
        "input_rows": 3,
        "output_rows": 2,
        "scalar_conflicts": {},
        "source_collections": 2,
        "source_rows": {"first": 2, "second": 1},
        "symbols": 2,
    }
    assert [row["story_id"] for row in rows] == ["story-1", "story-2"]
    shared = rows[0]
    assert shared["headline"] == "A longer shared headline"
    assert shared["first_created"] == "2026-01-02T08:55:00"
    assert shared["version_created"] == "2026-01-02T09:10:00"
    assert shared["matched_symbols"] == ["AAA", "BBB"]
    assert shared["source_collection_ids"] == ["first", "second"]
    assert len(shared["raw_rows"]) == 2
    assert build([first, second], output) == manifest


def test_build_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    source = _collection(tmp_path, "first", [{"story_id": "story-1", "headline": "Headline"}])
    with (source / "headlines.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="source headlines hash mismatch"):
        build([source, _collection(tmp_path, "second", [])], tmp_path / "merged")
