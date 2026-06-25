import asyncio
import json
from pathlib import Path

import pytest

import sentiment_benchmark.lseg_source as lseg_source
from sentiment_benchmark.lseg_source import (
    LsegCollectionConfig,
    LsegCompanyConfig,
    LsegHeadlinePage,
    LsegNewsClient,
    LsegNewsError,
    LsegStoryResponse,
    fetch_lseg_news,
    load_lseg_collection_config,
)


def _config(tmp_path: Path, *, collection_id: str = "test_lseg") -> LsegCollectionConfig:
    return LsegCollectionConfig(
        collection_id=collection_id,
        start="2026-06-01T00:00:00Z",
        end="2026-06-03T00:00:00Z",
        page_size=2,
        max_pages=3,
        story_concurrency=2,
        retries=0,
        min_text_chars=10,
        raw_output_root=tmp_path / "raw",
        derived_output_root=tmp_path / "derived",
        companies=(
            LsegCompanyConfig("AAPL", "Apple Inc.", "AAPL.O", "R:AAPL.O", ("Apple", "AAPL")),
            LsegCompanyConfig("MSFT", "Microsoft Corp.", "MSFT.O", "R:MSFT.O", ("Microsoft", "MSFT")),
        ),
    )


class FakeBackend:
    def __init__(self) -> None:
        self.page_calls: list[tuple[str, str | None]] = []
        self.story_calls: list[str] = []

    async def headline_page(self, *, query, start, end, count, cursor):
        self.page_calls.append((query, cursor))
        if query == "R:AAPL.O" and cursor is None:
            return LsegHeadlinePage(
                [
                    {
                        "storyId": "urn:test:apple:1",
                        "headline": "Apple reports growth",
                        "firstCreated": "2026-06-01T09:00:00Z",
                        "versionCreated": "2026-06-01T09:05:00Z",
                        "sourceCode": "NS:RTRS",
                        "language": "en",
                    }
                ],
                "next-aapl",
                {"meta": {"next": "next-aapl"}},
            )
        if query == "R:AAPL.O" and cursor == "next-aapl":
            return LsegHeadlinePage(
                [
                    {
                        "storyId": "urn:test:shared:1",
                        "headline": "Apple and Microsoft sign agreement",
                        "versionCreated": "2026-06-01T10:00:00Z",
                    }
                ],
                None,
            )
        if query == "R:MSFT.O":
            return LsegHeadlinePage(
                [
                    {
                        "storyId": "urn:test:shared:1",
                        "headline": "Apple and Microsoft sign a major agreement",
                        "versionCreated": "2026-06-01T10:00:00Z",
                    },
                    {
                        "storyId": "urn:test:web:1",
                        "headline": "Microsoft external report",
                        "versionCreated": "2026-06-01T11:00:00Z",
                    },
                ],
                None,
            )
        raise AssertionError((query, cursor, start, end, count))

    async def story(self, story_id):
        self.story_calls.append(story_id)
        if story_id == "urn:test:web:1":
            return LsegStoryResponse(story_id, "story_unavailable", web_url="https://publisher.test/story")
        return LsegStoryResponse(story_id, "success", f"<p>{story_id} body text with enough content.</p>", "html")


def test_load_lseg_config_requires_utc_and_desktop(tmp_path: Path) -> None:
    path = tmp_path / "lseg.toml"
    path.write_text(
        """
[collection]
id = "x"
start = "2026-06-01T00:00:00Z"
end = "2026-06-02T00:00:00Z"
[[companies]]
symbol = "AAPL"
name = "Apple"
ric = "AAPL.O"
aliases = ["Apple"]
"""
    )

    config = load_lseg_collection_config(path)

    assert config.companies[0].news_query == "R:AAPL.O and Language:LEN"
    assert config.session == "desktop"
    assert config.start.endswith("Z")


def test_story_content_reads_lseg_sdk_story_content() -> None:
    class Content:
        html = "<p>Full story body.</p>"
        text = "Full story body."
        web_url = "https://publisher.test/story"

    class Story:
        content = Content()

    class Data:
        story = Story()

    body, body_format, web_url = lseg_source._story_content(Data())

    assert body == "<p>Full story body.</p>"
    assert body_format == "html"
    assert web_url == "https://publisher.test/story"


def test_fetch_paginates_deduplicates_and_resumes_without_calls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = FakeBackend()
    result = asyncio.run(fetch_lseg_news(config, LsegNewsClient(backend)))

    assert result.headline_count == 3
    assert result.story_count == 3
    assert result.failed_story_count == 1
    assert backend.story_calls.count("urn:test:shared:1") == 1
    headlines = [json.loads(line) for line in (result.raw_dir / "headlines.jsonl").read_text().splitlines()]
    shared = next(row for row in headlines if row["story_id"] == "urn:test:shared:1")
    assert shared["matched_symbols"] == ["AAPL", "MSFT"]
    assert shared["headline"] == "Apple and Microsoft sign a major agreement"

    class NoCalls:
        async def headline_page(self, **kwargs):
            raise AssertionError(kwargs)

        async def story(self, story_id):
            raise AssertionError(story_id)

    resumed = asyncio.run(fetch_lseg_news(config, LsegNewsClient(NoCalls())))
    assert resumed.resumed is True


def test_completed_collection_rejects_changed_configuration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    asyncio.run(fetch_lseg_news(config, LsegNewsClient(FakeBackend())))
    changed = LsegCollectionConfig(**{**config.__dict__, "end": "2026-06-04T00:00:00Z"})

    with pytest.raises(LsegNewsError, match="different configuration"):
        asyncio.run(fetch_lseg_news(changed, LsegNewsClient(FakeBackend())))


def test_repeated_cursor_is_rejected(tmp_path: Path) -> None:
    class Repeating:
        async def headline_page(self, **kwargs):
            return LsegHeadlinePage([], "same")

        async def story(self, story_id):
            raise AssertionError(story_id)

    config = LsegCollectionConfig(**{**_config(tmp_path).__dict__, "companies": (_config(tmp_path).companies[0],)})
    with pytest.raises(LsegNewsError, match="repeated cursor"):
        asyncio.run(fetch_lseg_news(config, LsegNewsClient(Repeating())))


def test_retry_uses_exponential_backoff(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("transient")
        return "ok"

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(lseg_source.asyncio, "sleep", fake_sleep)

    assert asyncio.run(lseg_source._retry(operation, 3)) == "ok"
    assert delays == [1, 2]
