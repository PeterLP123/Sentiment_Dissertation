import asyncio

import pytest

from sentiment_benchmark import ollama_cloud
from sentiment_benchmark.ollama_cloud import (
    _parse_cloud_payload,
    _to_cloud_tag,
    cloud_catalog_overridden,
    fetch_ollama_cloud_models,
    ollama_cloud_catalog,
)


def test_to_cloud_tag_appends_suffix_without_doubling() -> None:
    assert _to_cloud_tag("gpt-oss:120b") == "gpt-oss:120b-cloud"
    assert _to_cloud_tag("deepseek-v3.1:671b") == "deepseek-v3.1:671b-cloud"
    # Already-cloud tags are left alone.
    assert _to_cloud_tag("foo:cloud") == "foo:cloud"
    assert _to_cloud_tag("bar-cloud") == "bar-cloud"


def test_parse_cloud_payload_maps_ids_and_dedupes() -> None:
    payload = {"data": [{"id": "gpt-oss:120b"}, {"id": "glm-5"}, {"id": "gpt-oss:120b"}, {"id": ""}, {}]}
    entries = _parse_cloud_payload(payload)
    assert entries == [("gpt-oss:120b-cloud", "gpt-oss:120b"), ("glm-5-cloud", "glm-5")]


def test_builtin_fallback_when_no_env_or_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_CLOUD_MODELS", raising=False)
    monkeypatch.setattr(ollama_cloud, "_CACHE_PATH", tmp_path / "missing.json")
    catalog = ollama_cloud_catalog()
    ids = {model.model_id for model in catalog}
    assert "gpt-oss:120b-cloud" in ids
    assert all(model.raw_metadata.get("cloud") is True for model in catalog)
    assert all("cloud" in model.model_id.lower() for model in catalog)


def test_env_override_takes_precedence(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_cloud, "_CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setenv("OLLAMA_CLOUD_MODELS", "foo:cloud,  bar:cloud ,")
    assert cloud_catalog_overridden() is True
    assert [m.model_id for m in ollama_cloud_catalog()] == ["foo:cloud", "bar:cloud"]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    payload = {"data": [{"id": "gpt-oss:120b"}, {"id": "glm-5"}, {"id": "already:cloud"}]}

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def get(self, url: str) -> _FakeResponse:
        return _FakeResponse(self.payload)


def test_fetch_live_maps_tags_caches_and_offline_reads_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache.json"
    monkeypatch.delenv("OLLAMA_CLOUD_MODELS", raising=False)
    monkeypatch.setattr(ollama_cloud, "_CACHE_PATH", cache)
    monkeypatch.setattr(ollama_cloud.httpx, "AsyncClient", _FakeClient)

    models = asyncio.run(fetch_ollama_cloud_models())
    ids = [model.model_id for model in models]
    assert ids == ["gpt-oss:120b-cloud", "glm-5-cloud", "already:cloud"]
    assert all(model.raw_metadata.get("source") == "live" for model in models)
    assert cache.exists()  # successful fetch is cached

    # With the network gone, the offline catalogue serves the cached list.
    monkeypatch.delattr(ollama_cloud.httpx, "AsyncClient")
    offline = ollama_cloud_catalog()
    assert [model.model_id for model in offline] == ids
    assert all(model.raw_metadata.get("source") == "cache" for model in offline)
