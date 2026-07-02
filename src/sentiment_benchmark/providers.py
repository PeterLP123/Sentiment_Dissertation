from __future__ import annotations

import os
from typing import Any, cast

from .cerebras_client import CerebrasClient
from .constants import DEFAULT_BASE_URL, DEFAULT_CEREBRAS_BASE_URL, DEFAULT_OLLAMA_HOST
from .models import Provider
from .ollama_client import OllamaClient
from .openrouter import OpenRouterClient

PROVIDERS: tuple[Provider, ...] = ("openrouter", "ollama", "cerebras")


def normalize_provider(value: str) -> Provider:
    provider = value.strip().lower()
    if provider not in PROVIDERS:
        available = ", ".join(PROVIDERS)
        raise ValueError(f"provider must be one of: {available}")
    return cast(Provider, provider)


def endpoint_for_provider(
    provider: Provider,
    *,
    base_url: str = DEFAULT_BASE_URL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    cerebras_base_url: str | None = None,
) -> str:
    if provider == "ollama":
        return ollama_host.rstrip("/")
    if provider == "cerebras":
        return (cerebras_base_url or os.getenv("CEREBRAS_BASE_URL") or DEFAULT_CEREBRAS_BASE_URL).rstrip("/")
    return base_url.rstrip("/")


def make_llm_client(
    provider: Provider,
    *,
    base_url: str = DEFAULT_BASE_URL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    cerebras_base_url: str | None = None,
    ollama_keep_alive: str | int | None = None,
    ollama_think: bool | None = None,
    structured_label_output: bool = False,
) -> Any:
    if provider == "ollama":
        return OllamaClient(
            host=ollama_host,
            keep_alive=ollama_keep_alive,
            structured_label_output=structured_label_output,
            default_think=ollama_think,
        )
    if provider == "cerebras":
        endpoint = cerebras_base_url or os.getenv("CEREBRAS_BASE_URL") or DEFAULT_CEREBRAS_BASE_URL
        return CerebrasClient(base_url=endpoint)
    return OpenRouterClient(base_url=base_url)
