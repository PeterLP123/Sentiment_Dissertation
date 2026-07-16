import asyncio
import json
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.lseg_corpus import build_lseg_corpus, load_verified_lseg_corpus
from sentiment_benchmark.lseg_source import (
    LsegCollectionConfig,
    LsegCompanyConfig,
    LsegHeadlinePage,
    LsegNewsClient,
    LsegNewsError,
    LsegStoryResponse,
    fetch_lseg_news,
)


class CorpusBackend:
    async def headline_page(self, *, query, **_kwargs):
        return LsegHeadlinePage(
            [
                {
                    "storyId": "urn:test:valid:1",
                    "headline": "Apple wins contract",
                    "firstCreated": "2026-06-01T08:00:00Z",
                    "versionCreated": "2026-06-01T09:00:00Z",
                    "language": "en",
                },
                {
                    "storyId": "urn:test:naive-time:1",
                    "headline": "Apple publishes update",
                    "versionCreated": "2026-06-01T09:30:00",
                    "language": "en",
                },
                {
                    "storyId": "urn:test:invalid-time:1",
                    "headline": "Apple update",
                    "versionCreated": "not-a-time",
                    "language": "en",
                },
                {
                    "storyId": "urn:test:web:1",
                    "headline": "Apple web item",
                    "versionCreated": "2026-06-01T10:00:00Z",
                    "language": "en",
                },
                {
                    "storyId": "urn:test:french:1",
                    "headline": "Apple annonce un contrat",
                    "versionCreated": "2026-06-01T11:00:00Z",
                    "language": "fr",
                },
            ],
            None,
        )

    async def story(self, story_id):
        if story_id == "urn:test:web:1":
            return LsegStoryResponse(story_id, "story_unavailable", web_url="https://example.test")
        return LsegStoryResponse(
            story_id,
            "success",
            "<p>June 1 (Reuters) - Apple signed a large contract.</p>"
            "<p>(c) Copyright Thomson Reuters 2026.</p>",
            "html",
        )


def _config(tmp_path: Path) -> LsegCollectionConfig:
    return LsegCollectionConfig(
        collection_id="corpus",
        start="2026-06-01T00:00:00Z",
        end="2026-06-02T00:00:00Z",
        retries=0,
        min_text_chars=10,
        raw_output_root=tmp_path / "raw",
        derived_output_root=tmp_path / "derived",
        companies=(LsegCompanyConfig("AAPL", "Apple", "AAPL.O", "R:AAPL.O", ("Apple",)),),
    )


def test_build_corpus_preserves_full_clean_text_and_excludes_invalid_records(tmp_path: Path) -> None:
    config = _config(tmp_path)
    raw = asyncio.run(fetch_lseg_news(config, LsegNewsClient(CorpusBackend())))
    result = build_lseg_corpus(raw.raw_dir)

    assert result.article_count == 5
    assert result.eligible_count == 2
    manifest, articles = load_verified_lseg_corpus(result.manifest_path)
    valid = next(row for row in articles if row["story_id"] == "urn:test:valid:1")
    naive_time = next(row for row in articles if row["story_id"] == "urn:test:naive-time:1")
    invalid_time = next(row for row in articles if row["story_id"] == "urn:test:invalid-time:1")
    web = next(row for row in articles if row["story_id"] == "urn:test:web:1")
    french = next(row for row in articles if row["story_id"] == "urn:test:french:1")
    assert valid["version_created"] == "2026-06-01T09:00:00+00:00"
    assert valid["clean_text"] == "June 1 (Reuters) - Apple signed a large contract."
    assert valid["scoring_eligible"] is True
    assert naive_time["version_created"] == "2026-06-01T09:30:00+00:00"
    assert naive_time["scoring_eligible"] is True
    assert invalid_time["text_quality"] == "invalid_timestamp"
    assert web["text_quality"] == "story_unavailable"
    assert french["text_quality"] == "non_english"
    assert manifest["sharing"]["redistribute"] is False

    resumed = build_lseg_corpus(raw.raw_dir)
    assert resumed.resumed is True


def test_verified_corpus_rejects_content_hash_mismatch(tmp_path: Path) -> None:
    raw = asyncio.run(fetch_lseg_news(_config(tmp_path), LsegNewsClient(CorpusBackend())))
    result = build_lseg_corpus(raw.raw_dir)
    result.articles_path.write_text(result.articles_path.read_text() + json.dumps({"tampered": True}) + "\n")

    with pytest.raises(LsegNewsError, match="hash mismatch"):
        load_verified_lseg_corpus(result.manifest_path)


@pytest.mark.parametrize("tamper", ["unexpected_top_level", "screening_hash", "unverified_csv"])
def test_completed_corpus_resume_revalidates_the_closed_inventory(tmp_path: Path, tamper: str) -> None:
    raw = asyncio.run(fetch_lseg_news(_config(tmp_path), LsegNewsClient(CorpusBackend())))
    result = build_lseg_corpus(raw.raw_dir)
    if tamper == "unexpected_top_level":
        (result.derived_dir / "notes.txt").write_text("unexpected", encoding="utf-8")
    elif tamper == "screening_hash":
        result.screening_path.write_text(result.screening_path.read_text() + "tampered\n", encoding="utf-8")
    else:
        csv_dir = result.derived_dir / "csv"
        csv_dir.mkdir()
        (csv_dir / "extra.csv").write_text("unexpected", encoding="utf-8")

    with pytest.raises(LsegNewsError, match="hash mismatch|unexpected|unverified"):
        build_lseg_corpus(raw.raw_dir)


def test_build_corpus_allows_only_verified_csv_auxiliary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    raw = asyncio.run(fetch_lseg_news(config, LsegNewsClient(CorpusBackend())))
    csv_dir = config.derived_dir / "csv"
    csv_dir.mkdir(parents=True)
    headlines_csv = csv_dir / "headlines.csv"
    bodies_csv = csv_dir / "main_bodies.csv"
    headlines_csv.write_text("headline\nApple wins contract\n", encoding="utf-8")
    bodies_csv.write_text("main_body\nApple signed a contract.\n", encoding="utf-8")
    (csv_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "exporter_version": "lseg_clean_csv_v1",
                "source_manifest_sha256": sha256_file(raw.manifest_path),
                "files": {
                    "headlines_csv": {"path": headlines_csv.name, "sha256": sha256_file(headlines_csv)},
                    "main_bodies_csv": {"path": bodies_csv.name, "sha256": sha256_file(bodies_csv)},
                },
            }
        ),
        encoding="utf-8",
    )

    result = build_lseg_corpus(raw.raw_dir)

    assert result.article_count == 5
    assert result.manifest_path.exists()
    assert headlines_csv.exists()
    assert bodies_csv.exists()


@pytest.mark.parametrize("tamper", ["unexpected_top_level", "unmanifested_csv", "bad_csv_hash"])
def test_build_corpus_rejects_unverified_csv_auxiliary(tmp_path: Path, tamper: str) -> None:
    config = _config(tmp_path)
    raw = asyncio.run(fetch_lseg_news(config, LsegNewsClient(CorpusBackend())))
    csv_dir = config.derived_dir / "csv"
    csv_dir.mkdir(parents=True)
    output = csv_dir / "headlines.csv"
    output.write_text("headline\nApple wins contract\n", encoding="utf-8")
    (csv_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "exporter_version": "lseg_clean_csv_v1",
                "source_manifest_sha256": sha256_file(raw.manifest_path),
                "files": {
                    "headlines_csv": {"path": output.name, "sha256": sha256_file(output)},
                    "main_bodies_csv": {
                        "path": "main_bodies.csv",
                        "sha256": sha256_file(output),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (csv_dir / "main_bodies.csv").write_bytes(output.read_bytes())
    if tamper == "unexpected_top_level":
        (config.derived_dir / "notes.txt").write_text("unexpected", encoding="utf-8")
    elif tamper == "unmanifested_csv":
        (csv_dir / "extra.csv").write_text("unexpected", encoding="utf-8")
    else:
        output.write_text("tampered", encoding="utf-8")

    with pytest.raises(LsegNewsError, match="refusing to overwrite"):
        build_lseg_corpus(raw.raw_dir)
