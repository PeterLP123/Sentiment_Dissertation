import asyncio
import json

import httpx
import pytest

from sentiment_benchmark.newsapi_source import (
    NewsApiClient,
    NewsApiConfigurationError,
    NewsApiError,
    make_newsapi_fetch_config,
    write_newsapi_corpus,
)


def test_newsapi_fetch_paginates_deduplicates_and_uses_header() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        page = int(request.url.params["page"])
        article = {
            "source": {"id": "example", "name": "Example"},
            "title": f"Article {page}",
            "description": "Company news summary",
            "url": f"https://example.com/{page}",
            "publishedAt": "2026-06-10T12:00:00Z",
        }
        articles = [article]
        if page == 2:
            articles.append({**article, "title": "Duplicate"})
        return httpx.Response(200, json={"status": "ok", "totalResults": 3, "articles": articles})

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = NewsApiClient(api_key="secret", client=http_client, retries=0)
            config = make_newsapi_fetch_config(
                query="Apple AAPL",
                from_time="2026-06-08T04:00:00Z",
                to_time="2026-06-11T04:00:00Z",
                page_size=1,
                max_pages=5,
            )
            return await client.fetch(config)

    result = asyncio.run(run())
    assert result.pages_fetched == 3
    assert len(result.records) == 3
    assert requests[0].headers["X-Api-Key"] == "secret"
    assert requests[0].url.params["from"] == "2026-06-08T04:00:00Z"
    assert requests[0].url.params["language"] == "en"
    assert "secret" not in str(requests[0].url)


def test_newsapi_missing_api_key_raises(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEWSAPI_API_KEY", raising=False)
    with pytest.raises(NewsApiConfigurationError, match="NEWSAPI_API_KEY"):
        NewsApiClient()


def test_newsapi_writer_records_provenance(tmp_path) -> None:
    async def run():
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "totalResults": 1,
                    "articles": [
                        {
                            "source": {"id": None, "name": "Wire"},
                            "title": "Tesla expands production",
                            "description": "Tesla announced an expansion.",
                            "url": "https://example.com/tesla?utm_source=test",
                            "publishedAt": "2026-06-10T12:00:00Z",
                        }
                    ],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = NewsApiClient(api_key="key", client=http_client, retries=0)
            return await client.fetch(make_newsapi_fetch_config(query="Tesla", max_pages=1))

    paths = write_newsapi_corpus(asyncio.run(run()), tmp_path)
    manifest = json.loads(paths.manifest_json.read_text())
    assert manifest["source"] == "newsapi"
    assert manifest["pages_fetched"] == 1
    assert manifest["record_count"] == 1
    row = json.loads(paths.articles_jsonl.read_text().strip())
    assert row["text_quality"] == "snippet_only"
    assert row["article_text"] is None


def test_newsapi_retries_transient_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("sentiment_benchmark.newsapi_source.asyncio.sleep", no_sleep)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, json={"status": "error"})
        return httpx.Response(200, json={"status": "ok", "totalResults": 0, "articles": []})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = NewsApiClient(api_key="key", client=http_client, retries=1)
            return await client.fetch(make_newsapi_fetch_config(query="markets"))

    result = asyncio.run(run())
    assert calls == 2
    assert result.records == []


def test_newsapi_surfaces_api_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "error", "code": "parameterInvalid", "message": "Bad query"},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = NewsApiClient(api_key="key", client=http_client, retries=0)
            return await client.fetch(make_newsapi_fetch_config(query="markets"))

    with pytest.raises(NewsApiError, match="parameterInvalid: Bad query"):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": ""}, "query is required"),
        ({"query": "x", "page_size": 101}, "page_size"),
        ({"query": "x", "from_time": "bad"}, "ISO 8601"),
        (
            {"query": "x", "from_time": "2026-06-11", "to_time": "2026-06-10"},
            "from_time",
        ),
    ],
)
def test_newsapi_config_validation(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_newsapi_fetch_config(**kwargs)
