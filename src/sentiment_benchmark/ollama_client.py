from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from .constants import DEFAULT_OLLAMA_HOST
from .env import load_env_file
from .models import BlindExample, LLMResponseRecord, ModelConfig, PromptConfig
from .parser import parse_model_response
from .prompts import render_messages

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class OllamaDependencyError(RuntimeError):
    """Raised when the official Ollama Python package is not installed."""


def _import_async_client() -> type:
    try:
        from ollama import AsyncClient
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise OllamaDependencyError(
            "The 'ollama' Python package is required for --provider ollama. "
            "Install project dependencies or run: python -m pip install ollama"
        ) from exc
    return AsyncClient


def _get_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _to_jsonable(as_dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return repr(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _format_ollama_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    message = getattr(exc, "error", None) or str(exc)
    if isinstance(status_code, int):
        return f"Ollama HTTP {status_code}: {message}"
    return message


def _is_response_error(exc: Exception) -> bool:
    return isinstance(getattr(exc, "status_code", None), int)


def _is_transport_error(exc: Exception) -> bool:
    return isinstance(exc, httpx.TimeoutException | httpx.TransportError)


class OllamaClient:
    """Async adapter around the official Ollama Python client."""

    def __init__(
        self,
        host: str | None = None,
        timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        load_env_file()
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._owns_client or self._client is None:
            return
        closer = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if callable(closer):
            result = closer()
            if inspect.isawaitable(result):
                await result
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            async_client = _import_async_client()
            try:
                self._client = async_client(host=self.host, timeout=self.timeout)
            except TypeError:
                self._client = async_client(host=self.host)
        return self._client

    async def _call_with_retries(
        self,
        factory: Callable[[], Awaitable[Any]],
        retries: int,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await factory()
            except OllamaDependencyError:
                raise
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                retryable_status = isinstance(status_code, int) and status_code in _TRANSIENT_STATUS_CODES
                if (retryable_status or _is_transport_error(exc)) and attempt < retries:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Ollama retry loop ended unexpectedly")

    async def list_models(self, retries: int = 3) -> list[ModelConfig]:
        payload = await self._call_with_retries(lambda: self._get_client().list(), retries)
        items = _get_field(payload, "models", [])
        models: list[ModelConfig] = []
        for item in items or []:
            raw = _to_jsonable(item)
            model_id = _get_field(item, "model") or _get_field(item, "name")
            if not isinstance(model_id, str) or not model_id:
                continue
            name = _get_field(item, "name")
            details = _get_field(item, "details", {}) or {}
            context_length = _as_int(_get_field(details, "context_length")) or _as_int(_get_field(item, "context_length"))
            models.append(
                ModelConfig(
                    model_id=model_id,
                    name=name if isinstance(name, str) else model_id,
                    context_length=context_length,
                    pricing={},
                    raw_metadata=raw if isinstance(raw, dict) else {"raw": raw},
                )
            )
        return models

    async def classify(
        self,
        model_id: str,
        prompt: PromptConfig,
        example: BlindExample,
        temperature: float = 0.0,
        max_completion_tokens: int = 64,
        retries: int = 3,
        ollama_think: bool | None = None,
    ) -> LLMResponseRecord:
        options = {
            "temperature": temperature,
            "num_predict": max_completion_tokens,
        }
        messages = render_messages(prompt, example)
        chat_kwargs = {
            "model": model_id,
            "messages": messages,
            "options": options,
            "stream": False,
        }
        if ollama_think is not None:
            chat_kwargs["think"] = ollama_think
        start = time.monotonic()
        try:
            response = await self._call_with_retries(
                lambda: self._get_client().chat(**chat_kwargs),
                retries,
            )
            latency_ms = (time.monotonic() - start) * 1000
            raw_json = _to_jsonable(response)
            message = _get_field(response, "message", {})
            raw_content = _get_field(message, "content")
            if not isinstance(raw_content, str):
                return LLMResponseRecord(
                    row_number=example.row_number,
                    model_id=model_id,
                    prompt_hash=prompt.prompt_hash,
                    raw_content=None,
                    normalized_label=None,
                    parse_status="error",
                    status="malformed_response",
                    raw_response_json=raw_json if isinstance(raw_json, dict) else {"raw": raw_json},
                    latency_ms=latency_ms,
                    error="Ollama response did not contain message.content",
                )

            parsed = parse_model_response(raw_content, prompt.output_mode)
            prompt_tokens = _as_int(_get_field(response, "prompt_eval_count"))
            completion_tokens = _as_int(_get_field(response, "eval_count"))
            total_tokens = _as_int(_get_field(response, "total_tokens"))
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            return LLMResponseRecord(
                row_number=example.row_number,
                model_id=model_id,
                prompt_hash=prompt.prompt_hash,
                raw_content=raw_content,
                normalized_label=parsed.normalized_label,
                parse_status=parsed.parse_status,
                status="success",
                explanation=parsed.explanation,
                raw_response_json=raw_json if isinstance(raw_json, dict) else {"raw": raw_json},
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except OllamaDependencyError:
            raise
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
        except Exception as exc:
            return LLMResponseRecord(
                row_number=example.row_number,
                model_id=model_id,
                prompt_hash=prompt.prompt_hash,
                raw_content=None,
                normalized_label=None,
                parse_status="error",
                status="api_error" if _is_response_error(exc) else "transport_error",
                latency_ms=(time.monotonic() - start) * 1000,
                error=_format_ollama_error(exc),
            )

    async def get_generation_metadata(self, generation_id: str, retries: int = 3) -> dict[str, Any] | None:
        return None
