import asyncio
import json
from pathlib import Path

import pytest

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

    assert result.article_count == 4
    assert result.eligible_count == 1
    manifest, articles = load_verified_lseg_corpus(result.manifest_path)
    valid = next(row for row in articles if row["story_id"] == "urn:test:valid:1")
    invalid_time = next(row for row in articles if row["story_id"] == "urn:test:invalid-time:1")
    web = next(row for row in articles if row["story_id"] == "urn:test:web:1")
    french = next(row for row in articles if row["story_id"] == "urn:test:french:1")
    assert valid["version_created"] == "2026-06-01T09:00:00+00:00"
    assert valid["clean_text"] == "June 1 (Reuters) - Apple signed a large contract."
    assert valid["scoring_eligible"] is True
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
