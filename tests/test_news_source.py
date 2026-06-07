import asyncio
import json
from pathlib import Path

import pytest

from sentiment_benchmark.news_source import (
    NewsArticleRecord,
    NewsFetchResult,
    TavilyNewsClient,
    TavilyNewsConfigurationError,
    article_record_id,
    make_news_fetch_config,
    normalize_url,
    write_news_corpus,
)


def run(coro):
    return asyncio.run(coro)


class FakeTavilyClient:
    def __init__(self, *, search_payload: dict, extract_payload: dict | None = None) -> None:
        self.search_payload = search_payload
        self.extract_payload = extract_payload or {"results": [], "failed_results": [], "request_id": "extract-1"}
        self.search_calls: list[dict] = []
        self.extract_calls: list[dict] = []
        self.closed = False

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_payload

    async def extract(self, **kwargs):
        self.extract_calls.append(kwargs)
        return self.extract_payload

    async def aclose(self) -> None:
        self.closed = True


def test_tavily_news_fetch_merges_search_and_extract_and_dedupes() -> None:
    async def scenario() -> None:
        search_payload = {
            "request_id": "search-1",
            "usage": {"credits": 1},
            "response_time": 0.5,
            "results": [
                {
                    "title": "Bank earnings rise",
                    "url": "https://example.com/article#section",
                    "content": "Snippet A",
                    "score": 0.9,
                    "published_date": "2026-06-01",
                    "favicon": "https://example.com/favicon.ico",
                },
                {
                    "title": "Duplicate",
                    "url": "https://example.com/article",
                    "content": "Snippet duplicate",
                    "score": 0.8,
                },
            ],
        }
        extract_payload = {
            "request_id": "extract-1",
            "usage": {"credits": 1},
            "results": [
                {
                    "url": "https://example.com/article",
                    "raw_content": "Full article text",
                    "favicon": "https://example.com/favicon.ico",
                }
            ],
            "failed_results": [],
        }
        fake = FakeTavilyClient(search_payload=search_payload, extract_payload=extract_payload)
        config = make_news_fetch_config(query="bank earnings", max_results=2)

        result = await TavilyNewsClient(client=fake).fetch(config)

        assert len(result.records) == 1
        record = result.records[0]
        assert record.record_id == article_record_id(normalize_url("https://example.com/article"))
        assert record.title == "Bank earnings rise"
        assert record.article_text == "Full article text"
        assert record.extraction_status == "success"
        assert record.search_request_id == "search-1"
        assert record.extract_request_id == "extract-1"
        assert fake.search_calls[0]["topic"] == "news"
        assert fake.extract_calls[0]["format"] == "text"

    run(scenario())


def test_tavily_news_missing_api_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(TavilyNewsConfigurationError):
        TavilyNewsClient()


def test_tavily_news_failed_extraction_is_preserved() -> None:
    async def scenario() -> None:
        fake = FakeTavilyClient(
            search_payload={
                "request_id": "search-1",
                "results": [{"title": "Article", "url": "https://example.com/a", "content": "Snippet"}],
            },
            extract_payload={
                "request_id": "extract-1",
                "results": [],
                "failed_results": [{"url": "https://example.com/a", "error": "blocked"}],
            },
        )
        config = make_news_fetch_config(query="market news", max_results=1)

        result = await TavilyNewsClient(client=fake).fetch(config)

        assert result.records[0].extraction_status == "failed"
        assert result.records[0].extract_error == "blocked"
        assert result.failed_extractions == [{"url": "https://example.com/a", "error": "blocked"}]

    run(scenario())


def test_tavily_news_snippet_only_skips_extract() -> None:
    async def scenario() -> None:
        fake = FakeTavilyClient(
            search_payload={
                "request_id": "search-1",
                "results": [{"title": "Article", "url": "https://example.com/a", "content": "Snippet"}],
            }
        )
        config = make_news_fetch_config(query="market news", max_results=1, extract=False)

        result = await TavilyNewsClient(client=fake).fetch(config)

        assert fake.extract_calls == []
        assert result.records[0].article_text is None
        assert result.records[0].extraction_status == "skipped"

    run(scenario())


def test_write_news_corpus_writes_jsonl_csv_and_manifest(tmp_path: Path) -> None:
    config = make_news_fetch_config(query="bank earnings", max_results=1)
    normalized = normalize_url("https://example.com/a")
    result = NewsFetchResult(
        config=config,
        fetched_at="2026-06-07T12:00:00+00:00",
        records=[
            NewsArticleRecord(
                record_id=article_record_id(normalized),
                url="https://example.com/a",
                normalized_url=normalized,
                title="Article",
                snippet="Snippet",
                article_text="Full text",
                extraction_status="success",
            )
        ],
        search_request_id="search-1",
    )

    paths = write_news_corpus(result, tmp_path)

    assert paths.articles_jsonl.exists()
    assert paths.articles_csv.exists()
    assert paths.manifest_json.exists()
    rows = paths.articles_jsonl.read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[0])["article_text"] == "Full text"
    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 1
    assert manifest["failed_extraction_count"] == 0
