"""Cerebras Inference adapter with proactive, model-aware rate pacing.

Cerebras exposes an OpenAI-compatible API, so response parsing stays shared with
the OpenRouter adapter.  This class only changes authentication, model metadata,
generation-metadata behavior, and request admission/retry timing.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .constants import DEFAULT_CEREBRAS_BASE_URL
from .models import ModelConfig
from .openrouter import OpenRouterClient

# Project limits recorded for the dissertation account on 2026-07-02.
# CEREBRAS_MAX_RPM deliberately overrides this map if the project limits change.
_DEFAULT_MODEL_RPM = 500
_MODEL_RPM = {
    "gemma-4-31b": 500,
    "gpt-oss-120b": 1_000,
    "zai-glm-4.7": 500,
}


_RATE_WINDOWS = {"minute": 60.0, "hour": 3_600.0, "day": 86_400.0}
_LATENCY_EXTENSION = "sentiment_benchmark_request_latency_ms"
_RATE_LIMIT_EXTENSION = "sentiment_benchmark_rate_limits"


@dataclass
class _QuotaBucket:
    capacity: float
    tokens: float
    window_seconds: float
    reset_at: float

    def reset_if_due(self, now: float) -> None:
        if now >= self.reset_at:
            self.tokens = self.capacity
            self.reset_at = now + self.window_seconds


class _AdaptiveRequestLimiter:
    """Probe once, then enforce the request quotas returned by this API key."""

    def __init__(self, fallback_rpm: int) -> None:
        self.fallback_rpm = fallback_rpm
        self._condition = asyncio.Condition()
        self._calibrated = False
        self._probe_in_flight = False
        self._buckets: dict[str, _QuotaBucket] = {}
        self._blocked_until = 0.0
        self.snapshot: dict[str, int] = {}

    async def acquire(self) -> None:
        while True:
            async with self._condition:
                if not self._calibrated:
                    if not self._probe_in_flight:
                        self._probe_in_flight = True
                        return
                    await self._condition.wait()
                    continue
                now = time.monotonic()
                delays: list[float] = []
                if now < self._blocked_until:
                    delays.append(self._blocked_until - now)
                for bucket in self._buckets.values():
                    bucket.reset_if_due(now)
                    if bucket.tokens < 1.0:
                        delays.append(max(0.0, bucket.reset_at - now))
                if not delays:
                    for bucket in self._buckets.values():
                        bucket.tokens -= 1.0
                    return
                delay = max(delays)
            await asyncio.sleep(delay)

    async def observe(self, headers: httpx.Headers) -> None:
        observed: dict[str, tuple[int, int | None, float | None]] = {}
        for window in _RATE_WINDOWS:
            limit = _header_int(headers, f"x-ratelimit-limit-requests-{window}")
            remaining = _header_int(headers, f"x-ratelimit-remaining-requests-{window}")
            reset_seconds = _header_float(headers, f"x-ratelimit-reset-requests-{window}")
            if limit is not None and limit > 0:
                observed[window] = (limit, remaining, reset_seconds)

        async with self._condition:
            now = time.monotonic()
            if observed:
                rpm_ceiling = _positive_int_env("CEREBRAS_MAX_RPM")
                self.snapshot = {f"requests_{window}": limit for window, (limit, _remaining, _reset) in observed.items()}
                for window, (reported_limit, remaining, reset_seconds) in observed.items():
                    limit = min(reported_limit, rpm_ceiling) if window == "minute" and rpm_ceiling else reported_limit
                    # Refill when the server window actually resets. Assuming a
                    # full window from now over-blocks: a run started mid-window
                    # would stall for up to an hour/day on partially spent quota.
                    server_reset_at = now + reset_seconds if reset_seconds is not None and reset_seconds > 0 else None
                    current = self._buckets.get(window)
                    if current is None or current.capacity != float(limit):
                        tokens = float(limit - 1 if remaining is None else min(remaining, limit))
                        self._buckets[window] = _QuotaBucket(
                            capacity=float(limit),
                            tokens=max(0.0, tokens),
                            window_seconds=_RATE_WINDOWS[window],
                            reset_at=server_reset_at if server_reset_at is not None else now + _RATE_WINDOWS[window],
                        )
                    else:
                        current.reset_if_due(now)
                        if server_reset_at is not None:
                            current.reset_at = min(current.reset_at, server_reset_at)
                        if remaining is not None:
                            current.tokens = min(current.tokens, float(remaining))
            elif not self._calibrated:
                limit = self.fallback_rpm
                self._buckets["minute"] = _QuotaBucket(
                    capacity=float(limit),
                    tokens=float(max(0, limit - 1)),
                    window_seconds=60.0,
                    reset_at=now + 60.0,
                )
            self._calibrated = True
            self._probe_in_flight = False
            self._condition.notify_all()

    async def probe_failed(self) -> None:
        await self.observe(httpx.Headers())

    async def block_for(self, delay: float) -> None:
        async with self._condition:
            self._blocked_until = max(self._blocked_until, time.monotonic() + max(0.0, delay))
            self._condition.notify_all()


def _header_int(headers: httpx.Headers, name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _header_float(headers: httpx.Headers, name: str) -> float | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _request_quota_exhausted(headers: httpx.Headers) -> bool:
    return any(_header_int(headers, f"x-ratelimit-remaining-requests-{window}") == 0 for window in _RATE_WINDOWS)


def _positive_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def cerebras_requests_per_minute(model_id: str) -> int:
    """Return the configured request pace for a model.

    CEREBRAS_MAX_RPM should be set to the model/project limit shown in the
    Cerebras console. Without an override, the repository's recorded project
    limits are used so high TUI concurrency reaches quota without a retry storm.
    """

    override = _positive_int_env("CEREBRAS_MAX_RPM")
    if override is not None:
        return override
    return _MODEL_RPM.get(model_id, _DEFAULT_MODEL_RPM)


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                return max(0.0, retry_at.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    for header in ("x-ratelimit-reset-tokens-minute", "x-ratelimit-reset-requests-minute"):
        raw = response.headers.get(header)
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
    return float(min(2**attempt, 8))


class CerebrasClient(OpenRouterClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_CEREBRAS_BASE_URL,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_key_env="CEREBRAS_API_KEY",
            base_url=base_url,
            timeout=timeout,
            client=client,
        )
        self._rate_limiters: dict[str, _AdaptiveRequestLimiter] = {}

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("CEREBRAS_API_KEY is required for Cerebras requests")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _limiter(self, model_id: str) -> _AdaptiveRequestLimiter:
        limiter = self._rate_limiters.get(model_id)
        if limiter is None:
            limiter = _AdaptiveRequestLimiter(cerebras_requests_per_minute(model_id))
            self._rate_limiters[model_id] = limiter
        return limiter

    async def _request_with_retries(
        self,
        method: str,
        path: str,
        retries: int,
        **kwargs: Any,
    ) -> httpx.Response:
        payload = kwargs.get("json")
        model_id = str(payload.get("model")) if isinstance(payload, dict) and payload.get("model") else None
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            limiter = None
            if method.upper() == "POST" and path.strip("/") == "chat/completions" and model_id:
                limiter = self._limiter(model_id)
                await limiter.acquire()
            try:
                attempt_started = time.monotonic()
                response = await self._get_client().request(method, self._url(path), headers=self._headers(), **kwargs)
                response.extensions[_LATENCY_EXTENSION] = (time.monotonic() - attempt_started) * 1000
                if limiter is not None:
                    await limiter.observe(response.headers)
                    response.extensions[_RATE_LIMIT_EXTENSION] = dict(limiter.snapshot)
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < retries:
                    if response.status_code == 429 and limiter is not None and _request_quota_exhausted(response.headers):
                        await limiter.block_for(_retry_delay(response, attempt))
                        continue
                    await asyncio.sleep(_retry_delay(response, attempt))
                    continue
                return response
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if limiter is not None:
                    await limiter.probe_failed()
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Request retry loop ended unexpectedly")

    async def list_models(self, retries: int = 3) -> list[ModelConfig]:
        models = await super().list_models(retries=retries)
        return [
            ModelConfig(
                model_id=model.model_id,
                name=model.name or model.model_id,
                context_length=model.context_length,
                pricing=model.pricing,
                raw_metadata=model.raw_metadata,
            )
            for model in models
        ]

    async def get_generation_metadata(self, generation_id: str, retries: int = 3) -> dict[str, Any] | None:
        # Cerebras includes usage, timing, service tier, and fingerprint in the
        # chat response itself and does not expose OpenRouter's /generation API.
        return None
