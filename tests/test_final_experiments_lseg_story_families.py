from __future__ import annotations

import json
from pathlib import Path

import pytest

from final_experiments.lib.lseg_story_families import (
    StoryFamilyError,
    load_family_revision_transitions,
    load_first_release_hashes,
    story_family_id,
)
from final_experiments.lib.novelty import headline_norm_sha256
from sentiment_benchmark.artifact_io import sha256_file


def _write_corpus(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "status": "completed",
        "files": {"headlines_jsonl": {"sha256": sha256_file(path)}},
    }
    path.with_name("manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_first_release_screen_removes_only_later_revision_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "merged" / "headlines.jsonl"
    rows = [
        {
            "story_id": "urn:test:family-a:1",
            "version_created": "2026-01-01T10:01:00Z",
            "headline": "Later revision",
        },
        {
            "story_id": "urn:test:family-a:0",
            "version_created": "2026-01-01T10:00:00Z",
            "headline": "First release",
        },
        {
            "story_id": "urn:test:family-b:0",
            "version_created": "2026-01-02T10:00:00Z",
            "headline": "Later revision",
        },
    ]
    _write_corpus(path, rows)
    first_hash = headline_norm_sha256("First release")
    reused_hash = headline_norm_sha256("Later revision")

    retained, audit = load_first_release_hashes(
        path, expected_hashes={first_hash, reused_hash}
    )

    assert retained == {first_hash, reused_hash}
    assert audit["families_with_multiple_rows"] == 1
    assert audit["removed_later_revision_only_hashes"] == 0
    assert audit["licensed_headline_text_emitted"] is False


def test_first_release_screen_removes_exclusive_later_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "merged" / "headlines.jsonl"
    rows = [
        {
            "story_id": "urn:test:family-a:0",
            "version_created": "2026-01-01T10:00:00Z",
            "headline": "First release",
        },
        {
            "story_id": "urn:test:family-a:1",
            "version_created": "2026-01-01T10:01:00Z",
            "headline": "Later revision",
        },
    ]
    _write_corpus(path, rows)
    first_hash = headline_norm_sha256("First release")
    later_hash = headline_norm_sha256("Later revision")

    retained, audit = load_first_release_hashes(
        path, expected_hashes={first_hash, later_hash}
    )

    assert retained == {first_hash}
    assert audit["removed_later_revision_only_hashes"] == 1
    assert audit["removed_share"] == pytest.approx(0.5)


def test_story_family_requires_numeric_revision_suffix() -> None:
    assert story_family_id("urn:test:family:12") == "urn:test:family"
    with pytest.raises(StoryFamilyError, match="numeric revision"):
        story_family_id("urn:test:family")


def test_revision_transitions_use_shared_symbols_and_emit_no_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "merged" / "headlines.jsonl"
    rows = [
        {
            "story_id": "urn:test:family-a:0",
            "version_created": "2026-01-01T10:00:00Z",
            "headline": "First release",
            "matched_symbols": ["AAA", "BBB"],
        },
        {
            "story_id": "urn:test:family-a:2",
            "version_created": "2026-01-01T10:05:00Z",
            "headline": "Changed release",
            "matched_symbols": ["AAA", "CCC"],
        },
        {
            "story_id": "urn:test:family-b:0",
            "version_created": "2026-01-02T10:00:00Z",
            "headline": "Single release",
            "matched_symbols": ["DDD"],
        },
    ]
    _write_corpus(path, rows)
    hashes = {headline_norm_sha256(str(row["headline"])) for row in rows}

    transitions, audit = load_family_revision_transitions(
        path, expected_hashes=hashes
    )

    assert len(transitions) == 1
    row = transitions.iloc[0]
    assert row["initial_headline_sha256"] == headline_norm_sha256("First release")
    assert row["current_headline_sha256"] == headline_norm_sha256("Changed release")
    assert row["shared_symbols"] == "AAA"
    assert bool(row["headline_changed"])
    assert "headline" not in transitions.columns
    assert "story_id" not in transitions.columns
    assert audit["families_with_multiple_distinct_revisions"] == 1
    assert audit["licensed_headline_text_emitted"] is False
    assert audit["raw_story_ids_emitted"] is False
