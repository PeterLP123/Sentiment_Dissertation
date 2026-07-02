from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from .constants import DEFAULT_APP_TITLE, DEFAULT_BASE_URL
from .env import load_env_file
from .models import BlindExample, LLMResponseRecord, ModelConfig, PromptConfig
from .parser import parse_model_response
from .prompts import render_messages


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "OPENROUTER_API_KEY",
        http_referer: str | None = None,
        app_title: str = DEFAULT_APP_TITLE,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        load_env_file()
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env)
        self.base_url = base_url.rstrip("/")
        self.http_referer = http_referer or os.getenv("OPENROUTER_HTTP_REFERER") or None
        self.app_title = app_title
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is required for {self.__class__.__name__} requests")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.app_title,
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    async def _request_with_retries(
        self,
        method: str,
        path: str,
        retries: int,
        **kwargs: Any,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = await self._get_client().request(method, self._url(path), headers=self._headers(), **kwargs)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                return response
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < retries:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Request retry loop ended unexpectedly")

    async def list_models(self, retries: int = 3) -> list[ModelConfig]:
        response = await self._request_with_retries("GET", "models", retries)
        response.raise_for_status()
        payload = response.json()
        models: list[ModelConfig] = []
        for item in payload.get("data", []):
            models.append(
                ModelConfig(
                    model_id=item.get("id") or item.get("canonical_slug"),
                    name=item.get("name"),
                    context_length=item.get("context_length"),
                    pricing=item.get("pricing") or {},
                    raw_metadata=item,
                )
            )
        return [model for model in models if model.model_id]

    async def classify(
        self,
        model_id: str,
        prompt: PromptConfig,
        example: BlindExample,
        temperature: float = 0.0,
        max_completion_tokens: int = 64,
        retries: int = 3,
    ) -> LLMResponseRecord:
        payload = {
            "model": model_id,
            "messages": render_messages(prompt, example),
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
        }
        start = time.monotonic()
        try:
            response = await self._request_with_retries("POST", "chat/completions", retries, json=payload)
            latency_ms = float(response.extensions.get("sentiment_benchmark_request_latency_ms", (time.monotonic() - start) * 1000))
            if response.status_code >= 400:
                return LLMResponseRecord(
                    row_number=example.row_number,
                    model_id=model_id,
                    prompt_hash=prompt.prompt_hash,
                    raw_content=None,
                    normalized_label=None,
                    parse_status="error",
                    status="api_error",
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: {response.text[:500]}",
                )

            raw_json = response.json()
            choice = (raw_json.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            raw_content = message.get("content")
            if not isinstance(raw_content, str):
                finish_reason = choice.get("finish_reason")
                native_finish_reason = choice.get("native_finish_reason")
                error = "Response did not contain choices[0].message.content"
                if finish_reason or native_finish_reason:
                    error += f" (finish_reason={finish_reason}, native_finish_reason={native_finish_reason})"
                if finish_reason == "length" or native_finish_reason == "MAX_TOKENS":
                    error += "; increase max_completion_tokens"
                return LLMResponseRecord(
                    row_number=example.row_number,
                    model_id=model_id,
                    prompt_hash=prompt.prompt_hash,
                    raw_content=None,
                    normalized_label=None,
                    parse_status="error",
                    status="malformed_response",
                    raw_response_json=raw_json,
                    latency_ms=latency_ms,
                    error=error,
                    generation_id=raw_json.get("id"),
                )
            if not raw_content.strip():
                finish_reason = choice.get("finish_reason")
                native_finish_reason = choice.get("native_finish_reason")
                error = "Model returned an empty completion"
                if finish_reason or native_finish_reason:
                    error += f" (finish_reason={finish_reason}, native_finish_reason={native_finish_reason})"
                if finish_reason == "length" or native_finish_reason == "MAX_TOKENS":
                    error += "; increase max_completion_tokens"
                return LLMResponseRecord(
                    row_number=example.row_number,
                    model_id=model_id,
                    prompt_hash=prompt.prompt_hash,
                    raw_content=raw_content,
                    normalized_label=None,
                    parse_status="error",
                    status="malformed_response",
                    raw_response_json=raw_json,
                    latency_ms=latency_ms,
                    error=error,
                    generation_id=raw_json.get("id"),
                )
            parsed = parse_model_response(raw_content, prompt.output_mode)
            usage = raw_json.get("usage") or {}
            return LLMResponseRecord(
                row_number=example.row_number,
                model_id=model_id,
                prompt_hash=prompt.prompt_hash,
                raw_content=raw_content,
                normalized_label=parsed.normalized_label,
                parse_status=parsed.parse_status,
                status="success",
                explanation=parsed.explanation,
                label_probabilities=parsed.label_probabilities,
                raw_response_json=raw_json,
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                generation_id=raw_json.get("id"),
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return LLMResponseRecord(
                row_number=example.row_number,
                model_id=model_id,
                prompt_hash=prompt.prompt_hash,
                raw_content=None,
                normalized_label=None,
                parse_status="error",
                status="transport_error",
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )
        except ValueError as exc:
            return LLMResponseRecord(
                row_number=example.row_number,
                model_id=model_id,
                prompt_hash=prompt.prompt_hash,
                raw_content=None,
                normalized_label=None,
                parse_status="error",
                status="malformed_response",
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )

    async def get_generation_metadata(self, generation_id: str, retries: int = 3) -> dict[str, Any] | None:
        response = await self._request_with_retries("GET", "generation", retries, params={"id": generation_id})
        if response.status_code >= 400:
            return None
        payload = response.json()
        data = payload.get("data")
        return data if isinstance(data, dict) else None
