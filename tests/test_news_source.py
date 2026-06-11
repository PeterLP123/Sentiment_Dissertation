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
    assess_article_text,
    infer_published_date,
    is_listing_url,
    make_news_fetch_config,
    normalize_url,
    write_news_corpus,
)


def run(coro):
    return asyncio.run(coro)


LONG_ARTICLE_TEXT = "Bank earnings rose sharply this quarter on stronger lending margins. " * 20


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
                    "raw_content": LONG_ARTICLE_TEXT,
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
        assert record.article_text == LONG_ARTICLE_TEXT
        assert record.extraction_status == "success"
        assert record.text_quality == "ok"
        assert record.published_date == "2026-06-01"
        assert record.published_date_source == "search"
        assert record.search_request_id == "search-1"
        assert record.extract_request_id == "extract-1"
        assert fake.search_calls[0]["topic"] == "news"
        assert fake.search_calls[0]["timeout"] == 90.0
        assert fake.extract_calls[0]["format"] == "text"
        # The Tavily SDK rejects timeouts above 120 seconds.
        assert 1 <= fake.extract_calls[0]["timeout"] <= 120

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


def test_assess_article_text_classifies_quality() -> None:
    assert assess_article_text(None) == ("missing", None)
    assert assess_article_text("   ") == ("missing", None)

    quality, detail = assess_article_text("Oops, something went wrong\nSkip to navigation" + "x" * 600)
    assert quality == "error_page"
    assert "error-page signature" in detail

    quality, detail = assess_article_text("Short stub.")
    assert quality == "too_short"
    assert "minimum 500" in detail

    assert assess_article_text("Short stub.", min_chars=0) == ("ok", None)
    assert assess_article_text(LONG_ARTICLE_TEXT) == ("ok", None)


def test_tavily_news_error_page_extraction_is_quality_gated() -> None:
    async def scenario() -> None:
        error_page = "Oops, something went wrong\nSkip to navigation\n" + ("Menu item\n" * 100)
        fake = FakeTavilyClient(
            search_payload={
                "request_id": "search-1",
                "results": [{"title": "Article", "url": "https://example.com/a", "content": "Snippet"}],
            },
            extract_payload={
                "request_id": "extract-1",
                "results": [{"url": "https://example.com/a", "raw_content": error_page}],
                "failed_results": [],
            },
        )
        config = make_news_fetch_config(query="market news", max_results=1)

        result = await TavilyNewsClient(client=fake).fetch(config)

        record = result.records[0]
        assert record.extraction_status == "failed"
        assert record.text_quality == "error_page"
        assert record.article_text is None
        assert "error-page signature" in record.extract_error
        # Raw extractor output is preserved for auditing.
        assert record.raw_extract_result["raw_content"] == error_page

    run(scenario())


def test_tavily_news_short_extraction_is_quality_gated_unless_disabled() -> None:
    async def scenario() -> None:
        search_payload = {
            "request_id": "search-1",
            "results": [{"title": "Article", "url": "https://example.com/a", "content": "Snippet"}],
        }
        extract_payload = {
            "request_id": "extract-1",
            "results": [{"url": "https://example.com/a", "raw_content": "Tiny body."}],
            "failed_results": [],
        }

        fake = FakeTavilyClient(search_payload=search_payload, extract_payload=extract_payload)
        gated = await TavilyNewsClient(client=fake).fetch(make_news_fetch_config(query="market news", max_results=1))
        assert gated.records[0].extraction_status == "failed"
        assert gated.records[0].text_quality == "too_short"
        assert gated.records[0].article_text is None

        fake = FakeTavilyClient(search_payload=search_payload, extract_payload=extract_payload)
        ungated = await TavilyNewsClient(client=fake).fetch(make_news_fetch_config(query="market news", max_results=1, min_text_chars=0))
        assert ungated.records[0].extraction_status == "success"
        assert ungated.records[0].text_quality == "ok"
        assert ungated.records[0].article_text == "Tiny body."

    run(scenario())


def test_is_listing_url_flags_index_pages_not_articles() -> None:
    assert is_listing_url("https://www.bloomberg.com/professional/insights/category/markets/page/4?pg=53")
    assert is_listing_url("https://www.wsj.com/topics/place/portugal")
    assert is_listing_url("https://www.bloomberg.com/professional/products/indices/quote/LF98TRUU:IND")
    assert is_listing_url("https://www.cnbc.com/video/2026/06/08/rbi-hiking-cycle.html")
    assert is_listing_url("https://example.com/tag/banking/")
    assert not is_listing_url("https://www.cnbc.com/2026/06/05/jobs-report-may-2026.html")
    assert not is_listing_url("https://www.reuters.com/markets/rates-bonds/some-story-2026-06-05/")
    # Words like "page" or "tags" inside slugs are not listing markers.
    assert not is_listing_url("https://example.com/news/front-page-news-roundup")
    assert not is_listing_url("https://example.com/news/price-tags-rise")


def test_tavily_news_listing_page_is_quality_gated() -> None:
    async def scenario() -> None:
        url = "https://example.com/category/markets/page/2"
        fake = FakeTavilyClient(
            search_payload={
                "request_id": "search-1",
                "results": [{"title": "Markets | Page 2", "url": url, "content": "Snippet"}],
            },
            extract_payload={
                "request_id": "extract-1",
                "results": [{"url": url, "raw_content": LONG_ARTICLE_TEXT}],
                "failed_results": [],
            },
        )
        config = make_news_fetch_config(query="market news", max_results=1)

        result = await TavilyNewsClient(client=fake).fetch(config)

        record = result.records[0]
        assert record.text_quality == "non_article"
        assert record.extraction_status == "failed"
        assert record.article_text is None
        assert "listing" in record.extract_error

    run(scenario())


def test_infer_published_date_from_url_and_text() -> None:
    assert infer_published_date("https://example.com/2026/05/27/banks-rally", None) == ("2026-05-27", "url")
    assert infer_published_date("https://example.com/news/2026-06-03-markets", None) == ("2026-06-03", "url")
    assert infer_published_date("https://example.com/article", "Published May 27, 2026 6:39 am ET") == ("2026-05-27", "text")
    assert infer_published_date("https://example.com/article", "Updated 3 June 2026") == ("2026-06-03", "text")
    assert infer_published_date("https://example.com/article", "By Jane Doe, 2026-05-22") == ("2026-05-22", "text")
    assert infer_published_date("https://example.com/article", "No date anywhere here") == (None, "")
    # Invalid calendar dates are rejected rather than guessed.
    assert infer_published_date("https://example.com/2026/13/40/story", None) == (None, "")


def test_tavily_news_backfills_published_date_from_url() -> None:
    async def scenario() -> None:
        fake = FakeTavilyClient(
            search_payload={
                "request_id": "search-1",
                "results": [{"title": "Article", "url": "https://example.com/2026/05/27/banks", "content": "Snippet"}],
            },
            extract_payload={
                "request_id": "extract-1",
                "results": [{"url": "https://example.com/2026/05/27/banks", "raw_content": LONG_ARTICLE_TEXT}],
                "failed_results": [],
            },
        )
        config = make_news_fetch_config(query="bank earnings", max_results=1)

        result = await TavilyNewsClient(client=fake).fetch(config)

        record = result.records[0]
        assert record.published_date == "2026-05-27"
        assert record.published_date_source == "url"

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
                text_quality="ok",
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
    assert manifest["usable_record_count"] == 1
    assert manifest["text_quality_counts"] == {"ok": 1}
