"""Discovery of Ollama Cloud models.

Ollama's local ``/api/tags`` only lists models that are already pulled, so cloud
models you have not set up never show up. Ollama does, however, publish its live
cloud catalogue at a public, OpenAI-compatible endpoint
(``https://ollama.com/v1/models``), so we fetch that and map each base id to the
``-cloud`` tag the signed-in local daemon routes to cloud.

Resolution order:

1. ``OLLAMA_CLOUD_MODELS`` env var (comma-separated tags) -- explicit override.
2. The live ``ollama.com`` catalogue (and we cache it to disk on success).
3. The last cached catalogue from a previous successful fetch.
4. A small built-in list, as a final offline fallback.

The live fetch is async (:func:`fetch_ollama_cloud_models`); everything else is
available synchronously via :func:`ollama_cloud_catalog` with no network access.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

from .models import ModelConfig

CLOUD_CATALOG_URL = "https://ollama.com/v1/models"
# Overridable so tests don't touch the real results/ directory.
_CACHE_PATH = Path("results/ollama_cloud_cache.json")
_ENV_VAR = "OLLAMA_CLOUD_MODELS"

# Final offline fallback if the network, cache, and env override are all absent.
# Tags already carry the -cloud suffix.
_BUILTIN_CLOUD_MODELS: list[tuple[str, str]] = [
    ("gpt-oss:20b-cloud", "gpt-oss:20b"),
    ("gpt-oss:120b-cloud", "gpt-oss:120b"),
    ("deepseek-v3.1:671b-cloud", "deepseek-v3.1:671b"),
    ("qwen3-coder:480b-cloud", "qwen3-coder:480b"),
    ("kimi-k2:1t-cloud", "kimi-k2:1t"),
    ("glm-4.6:cloud", "glm-4.6"),
]


def _to_cloud_tag(model_id: str) -> str:
    """Map a base model id (e.g. ``gpt-oss:120b``) to its cloud-routed tag."""
    lowered = model_id.lower()
    if lowered.endswith("-cloud") or lowered.endswith(":cloud"):
        return model_id
    return f"{model_id}-cloud"


def _models_from(entries: list[tuple[str, str]], source: str) -> list[ModelConfig]:
    return [
        ModelConfig(
            model_id=tag,
            name=name,
            context_length=None,
            pricing={},
            raw_metadata={"cloud": True, "source": source},
        )
        for tag, name in entries
    ]


def _parse_cloud_payload(payload: dict) -> list[tuple[str, str]]:
    """Turn an OpenAI-style ``/v1/models`` payload into (cloud tag, base name) pairs."""
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in payload.get("data", []) or []:
        base_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(base_id, str) or not base_id:
            continue
        tag = _to_cloud_tag(base_id)
        if tag in seen:
            continue
        seen.add(tag)
        entries.append((tag, base_id))
    return entries


def _env_override_entries() -> list[tuple[str, str]] | None:
    raw = os.getenv(_ENV_VAR, "").strip()
    if not raw:
        return None
    tags = [tag.strip() for tag in raw.split(",") if tag.strip()]
    return [(tag, tag) for tag in tags] or None


def cloud_catalog_overridden() -> bool:
    """True when the user has pinned the catalogue via ``OLLAMA_CLOUD_MODELS``."""
    return _env_override_entries() is not None


def _write_cache(entries: list[tuple[str, str]]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(
            json.dumps(
                {"fetched_at": time.time(), "models": [{"id": tag, "name": name} for tag, name in entries]},
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _read_cache_entries() -> list[tuple[str, str]] | None:
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    entries = [
        (str(model["id"]), str(model.get("name") or model["id"]))
        for model in models
        if isinstance(model, dict) and model.get("id")
    ]
    return entries or None


async def fetch_ollama_cloud_models(timeout: float = 10.0) -> list[ModelConfig]:
    """Fetch the live cloud catalogue from ollama.com and cache it.

    Returns models tagged for cloud routing. Raises on any network/parse error so
    the caller can fall back to :func:`ollama_cloud_catalog`.
    """
    override = _env_override_entries()
    if override is not None:
        return _models_from(override, "env")
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(CLOUD_CATALOG_URL)
        response.raise_for_status()
        payload = response.json()
    entries = _parse_cloud_payload(payload)
    if not entries:
        raise ValueError(f"{CLOUD_CATALOG_URL} returned no models")
    _write_cache(entries)
    return _models_from(entries, "live")


def ollama_cloud_catalog() -> list[ModelConfig]:
    """Offline catalogue with no network access: env override, else cache, else built-in."""
    override = _env_override_entries()
    if override is not None:
        return _models_from(override, "env")
    cached = _read_cache_entries()
    if cached is not None:
        return _models_from(cached, "cache")
    return _models_from(_BUILTIN_CLOUD_MODELS, "builtin")
