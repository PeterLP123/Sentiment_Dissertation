import csv
import json
import runpy
from pathlib import Path

import pytest

_EXPORTER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "build_lseg_clean_csvs.py"))
build = _EXPORTER["build"]
sha256_file = _EXPORTER["sha256_file"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _raw_collection(tmp_path: Path) -> Path:
    source = tmp_path / "raw"
    source.mkdir()
    headlines = source / "headlines.jsonl"
    stories = source / "stories.jsonl"
    _write_jsonl(
        headlines,
        [
            {
                "story_id": "urn:reuters:test:1",
                "headline": "  Acme   profits rise  ",
                "version_created": "2026-07-01T09:30:00",
                "source_code": "RTRS",
                "language": "en",
                "matched_symbols": ["ACME", "ACME"],
                "matched_rics": ["ACME.N"],
            },
            {
                "story_id": "urn:reuters:failed:1",
                "headline": "Acme publishes filing",
                "version_created": "2026-07-01T10:00:00Z",
                "source_code": "RTRS",
                "language": "en",
            },
        ],
    )
    _write_jsonl(
        stories,
        [
            {
                "story_id": "urn:reuters:test:1",
                "status": "success",
                "body_format": "html",
                "body": "<p>Acme profits rose sharply after strong demand.</p>",
            },
            {
                "story_id": "urn:reuters:failed:1",
                "status": "api_error",
                "body_format": "html",
                "body": "",
            },
        ],
    )
    manifest = {
        "status": "completed",
        "config": {
            "collection": {"language": "en"},
            "cleaning": {"min_text_chars": 20},
        },
        "files": {
            "headlines_jsonl": {"path": headlines.name, "sha256": sha256_file(headlines)},
            "stories_jsonl": {"path": stories.name, "sha256": sha256_file(stories)},
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_build_exports_verified_csvs_and_reuses_current_output(tmp_path: Path) -> None:
    source = _raw_collection(tmp_path)
    output = tmp_path / "derived"

    manifest = build(source, output)

    assert manifest["status"] == "completed"
    assert manifest["counts"]["headlines"]["written"] == 2
    assert manifest["counts"]["main_bodies"] == {
        "excluded_unsuccessful": 1,
        "source_rows": 2,
        "written": 1,
    }
    with (output / "headlines.csv").open(encoding="utf-8", newline="") as handle:
        headlines = list(csv.DictReader(handle))
    with (output / "main_bodies.csv").open(encoding="utf-8", newline="") as handle:
        bodies = list(csv.DictReader(handle))
    assert headlines[0]["headline"] == "Acme profits rise"
    assert headlines[0]["version_created"] == "2026-07-01T09:30:00+00:00"
    assert headlines[0]["story_family_id"] == "urn:reuters:test"
    assert headlines[0]["matched_symbols"] == "ACME"
    assert len(bodies) == 1
    assert bodies[0]["main_body"] == "Acme profits rose sharply after strong demand."
    assert bodies[0]["scoring_eligible"] == "true"
    assert sha256_file(output / "headlines.csv") == manifest["files"]["headlines_csv"]["sha256"]
    assert build(source, output) == manifest


def test_build_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    source = _raw_collection(tmp_path)
    with (source / "headlines.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="raw collection hash mismatch"):
        build(source, tmp_path / "derived")
