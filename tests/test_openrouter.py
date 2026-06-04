import asyncio

import httpx

from sentiment_benchmark.models import BlindExample
from sentiment_benchmark.openrouter import OpenRouterClient
from sentiment_benchmark.prompts import make_prompt


def run(coro):
    return asyncio.run(coro)


def make_client(handler) -> OpenRouterClient:
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    return OpenRouterClient(api_key="test-key", base_url="https://openrouter.test/api/v1", client=async_client)


def test_openrouter_success_and_generation_metadata() -> None:
    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/chat/completions"):
                return httpx.Response(
                    200,
                    json={
                        "id": "gen-1",
                        "choices": [{"message": {"content": "positive"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                    },
                )
            if request.url.path.endswith("/generation"):
                return httpx.Response(200, json={"data": {"id": "gen-1", "total_cost": 0.001, "provider_name": "Mock"}})
            return httpx.Response(404)

        client = make_client(handler)
        prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")
        record = await client.classify("test/model", prompt, BlindExample(2, "sentence"))
        metadata = await client.get_generation_metadata("gen-1")
        await client._client.aclose()  # type: ignore[union-attr]
        assert record.status == "success"
        assert record.normalized_label == "positive"
        assert record.total_tokens == 11
        assert metadata and metadata["total_cost"] == 0.001

    run(scenario())


def test_openrouter_retries_429_then_success() -> None:
    async def scenario():
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, json={"id": "gen-2", "choices": [{"message": {"content": "neutral"}}]})

        client = make_client(handler)
        prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")
        record = await client.classify("test/model", prompt, BlindExample(2, "sentence"), retries=1)
        await client._client.aclose()  # type: ignore[union-attr]
        assert attempts["count"] == 2
        assert record.normalized_label == "neutral"

    run(scenario())


def test_openrouter_invalid_model_returns_api_error() -> None:
    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="invalid model")

        client = make_client(handler)
        prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")
        record = await client.classify("bad/model", prompt, BlindExample(2, "sentence"), retries=1)
        await client._client.aclose()  # type: ignore[union-attr]
        assert record.status == "api_error"
        assert "HTTP 400" in (record.error or "")

    run(scenario())

