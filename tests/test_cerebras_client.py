import asyncio
import json
import time

import httpx

from sentiment_benchmark.cerebras_client import CerebrasClient, cerebras_requests_per_minute
from sentiment_benchmark.models import BlindExample
from sentiment_benchmark.prompts import make_prompt


def run(coro):
    return asyncio.run(coro)


def make_client(handler) -> CerebrasClient:
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    return CerebrasClient(api_key="test-key", base_url="https://cerebras.test/v1", client=async_client)


def test_cerebras_lists_models_and_uses_cerebras_auth() -> None:
    async def scenario() -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization", "")
            return httpx.Response(
                200,
                json={"data": [{"id": "gemma-4-31b", "object": "model", "owned_by": "Cerebras"}]},
            )

        client = make_client(handler)
        models = await client.list_models()

        assert seen["authorization"] == "Bearer test-key"
        assert models[0].model_id == "gemma-4-31b"
        assert models[0].name == "gemma-4-31b"

    run(scenario())


def test_cerebras_classification_uses_short_non_streaming_completion() -> None:
    async def scenario() -> None:
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "choices": [{"finish_reason": "stop", "message": {"content": "negative"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
                    "time_info": {"total_time": 0.02},
                },
            )

        client = make_client(handler)
        prompt = make_prompt("test", "Return one label.", "{sentence}", "label_only")
        record = await client.classify(
            "gpt-oss-120b",
            prompt,
            BlindExample(7, "Profits fell."),
            max_completion_tokens=64,
        )

        assert record.status == "success"
        assert record.normalized_label == "negative"
        assert record.total_tokens == 21
        assert payloads[0]["max_completion_tokens"] == 64
        assert payloads[0]["stream"] is False
        assert await client.get_generation_metadata("chatcmpl-test") is None

    run(scenario())


def test_cerebras_retries_429_using_retry_header() -> None:
    async def scenario() -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, text="rate limited", headers={"retry-after": "0.05"})
            return httpx.Response(200, json={"id": "ok", "choices": [{"message": {"content": "neutral"}}]})

        client = make_client(handler)
        prompt = make_prompt("test", "Return one label.", "{sentence}", "label_only")
        started = time.monotonic()
        record = await client.classify("gpt-oss-120b", prompt, BlindExample(1, "Flat."), retries=1)
        wall_ms = (time.monotonic() - started) * 1000

        assert attempts == 2
        assert record.normalized_label == "neutral"
        assert record.attempt_count == 2
        assert wall_ms >= 45
        assert record.latency_ms is not None and record.latency_ms < 45

    run(scenario())


def test_cerebras_uses_recorded_model_limits_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv("CEREBRAS_MAX_RPM", raising=False)
    assert cerebras_requests_per_minute("gemma-4-31b") == 500
    assert cerebras_requests_per_minute("gpt-oss-120b") == 1_000
    assert cerebras_requests_per_minute("zai-glm-4.7") == 500
    assert cerebras_requests_per_minute("future-model") == 500

    monkeypatch.setenv("CEREBRAS_MAX_RPM", "750")
    assert cerebras_requests_per_minute("gpt-oss-120b") == 750


def test_cerebras_calibrates_to_live_api_key_limits(monkeypatch) -> None:
    monkeypatch.delenv("CEREBRAS_MAX_RPM", raising=False)

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={
                    "x-ratelimit-limit-requests-minute": "5",
                    "x-ratelimit-remaining-requests-minute": "4",
                    "x-ratelimit-limit-requests-hour": "150",
                    "x-ratelimit-remaining-requests-hour": "149",
                    "x-ratelimit-limit-requests-day": "2400",
                    "x-ratelimit-remaining-requests-day": "2399",
                },
                json={"id": "ok", "choices": [{"message": {"content": "positive"}}]},
            )

        client = make_client(handler)
        prompt = make_prompt("test", "Return one label.", "{sentence}", "label_only")
        record = await client.classify("gemma-4-31b", prompt, BlindExample(1, "Up."))

        assert record.normalized_label == "positive"
        assert client._rate_limiters["gemma-4-31b"].snapshot == {
            "requests_minute": 5,
            "requests_hour": 150,
            "requests_day": 2400,
        }

    run(scenario())


def test_cerebras_refills_at_server_reset_not_full_window(monkeypatch) -> None:
    monkeypatch.delenv("CEREBRAS_MAX_RPM", raising=False)

    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={
                    "x-ratelimit-limit-requests-hour": "10",
                    "x-ratelimit-remaining-requests-hour": "0",
                    "x-ratelimit-reset-requests-hour": "0.05",
                },
                json={"id": "ok", "choices": [{"message": {"content": "neutral"}}]},
            )

        client = make_client(handler)
        prompt = make_prompt("test", "Return one label.", "{sentence}", "label_only")
        # The probe reports the hour quota as already spent, resetting in 50ms.
        await client.classify("gemma-4-31b", prompt, BlindExample(1, "Flat."))

        started = time.monotonic()
        # Assuming a full hour window from now would block this call until the
        # wait_for timeout; honoring the reset header releases it in ~50ms.
        record = await asyncio.wait_for(
            client.classify("gemma-4-31b", prompt, BlindExample(2, "Flat.")),
            timeout=2,
        )
        waited = time.monotonic() - started

        assert record.normalized_label == "neutral"
        assert waited >= 0.04

    run(scenario())


def test_cerebras_serializes_initial_quota_probe() -> None:
    async def scenario() -> None:
        calls = 0
        active = 0
        max_active = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls, active, max_active
            calls += 1
            active += 1
            max_active = max(max_active, active)
            if calls == 1:
                await asyncio.sleep(0.03)
            else:
                await asyncio.sleep(0.005)
            active -= 1
            return httpx.Response(
                200,
                headers={
                    "x-ratelimit-limit-requests-minute": "500",
                    "x-ratelimit-remaining-requests-minute": str(500 - calls),
                },
                json={"id": f"ok-{calls}", "choices": [{"message": {"content": "neutral"}}]},
            )

        client = make_client(handler)
        prompt = make_prompt("test", "Return one label.", "{sentence}", "label_only")
        records = await asyncio.gather(
            *(client.classify("gemma-4-31b", prompt, BlindExample(index, "Flat.")) for index in range(8))
        )

        assert all(record.normalized_label == "neutral" for record in records)
        assert calls == 8
        assert max_active > 1

    run(scenario())
