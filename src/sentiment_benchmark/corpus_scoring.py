from __future__ import annotations

import asyncio
import csv
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_io import atomic_write_json, canonical_json, sha256_file, sha256_text
from .lseg_source import LsegNewsError, utc_now
from .models import BlindExample, PromptConfig
from .prompts import load_prompts

MATRIX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MatrixModel:
    model_id: str
    provider: str


@dataclass(frozen=True)
class MatrixConfig:
    models: tuple[MatrixModel, ...]
    target_prompts: tuple[str, ...]
    benchmark_prompts: tuple[str, ...]
    samples: int = 5
    temperature: float = 0.7
    max_completion_tokens: int = 128
    concurrency: int = 3
    retries: int = 3


@dataclass(frozen=True)
class MatrixItem:
    item_id: str
    content: str
    content_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MatrixPlan:
    item_count: int
    subset_count: int
    base_calls: int
    variant_calls: int
    total_calls: int
    hosted_calls: int
    local_calls: int


def load_matrix_config(path: str | Path) -> MatrixConfig:
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    raw = payload.get("matrix")
    if not isinstance(raw, dict):
        raise LsegNewsError("matrix config must contain a [matrix] table")
    models = tuple(
        MatrixModel(str(item.get("id") or "").strip(), str(item.get("provider") or "").strip().lower())
        for item in payload.get("models") or []
    )
    if not models or any(not model.model_id or model.provider not in {"openrouter", "ollama"} for model in models):
        raise LsegNewsError("matrix config requires valid [[models]] entries")
    target = tuple(str(value) for value in raw.get("target_prompts") or [])
    benchmark = tuple(str(value) for value in raw.get("benchmark_prompts") or [])
    if len(target) != 3 or len(benchmark) != 3:
        raise LsegNewsError("matrix config requires exactly three target and benchmark prompts")
    config = MatrixConfig(
        models=models,
        target_prompts=target,
        benchmark_prompts=benchmark,
        samples=int(raw.get("samples", 5)),
        temperature=float(raw.get("temperature", 0.7)),
        max_completion_tokens=int(raw.get("max_completion_tokens", 128)),
        concurrency=int(raw.get("concurrency", 3)),
        retries=int(raw.get("retries", 3)),
    )
    if config.samples < 1 or config.concurrency < 1:
        raise LsegNewsError("matrix samples and concurrency must be positive")
    return config


def frozen_design_call_counts() -> dict[str, int]:
    return {
        "lseg": 3000 * 5 * 5 + 500 * 5 * 5 * 2,
        "benchmark": 900 * 5 * 5 + 500 * 5 * 5 * 2,
        "hosted": 88500,
        "local": 59000,
        "total": 147500,
    }


def load_matrix_items(path: str | Path, kind: str) -> list[MatrixItem]:
    source = Path(path)
    items: list[MatrixItem] = []
    if kind == "lseg":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, row in enumerate(rows, start=1):
            item_id = f"{row['story_family_id']}|{row['symbol']}"
            content = (
                f"Target company: {row['symbol']}\nHeadline: {row.get('headline', '')}\n"
                f"<news_document>\n{row.get('clean_text', '')}\n</news_document>"
            )
            items.append(MatrixItem(item_id, content, sha256_text(content), {**row, "row_number": index}))
    elif kind == "benchmark":
        with source.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                content = str(row.get("Sentence") or "")
                item_id = str(row.get("item_id") or row.get("article_id") or f"benchmark-{index}")
                items.append(MatrixItem(item_id, content, sha256_text(content), {**row, "row_number": index}))
    else:
        raise LsegNewsError("matrix input kind must be lseg or benchmark")
    if len({item.item_id for item in items}) != len(items):
        raise LsegNewsError("matrix input contains duplicate item ids")
    return items


def matrix_plan(config: MatrixConfig, items: list[MatrixItem], subset_ids: set[str]) -> MatrixPlan:
    base_calls = len(items) * len(config.models) * config.samples
    variant_calls = len(subset_ids) * len(config.models) * config.samples * 2
    hosted = sum(model.provider == "openrouter" for model in config.models)
    local = len(config.models) - hosted
    calls_per_model = len(items) * config.samples + len(subset_ids) * config.samples * 2
    return MatrixPlan(
        len(items),
        len(subset_ids),
        base_calls,
        variant_calls,
        base_calls + variant_calls,
        hosted * calls_per_model,
        local * calls_per_model,
    )


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(record["item_id"]),
        str(record["content_sha256"]),
        str(record["model_id"]),
        str(record["prompt_hash"]),
        int(record["sample_index"]),
    )


async def _model_digests(config: MatrixConfig, clients: dict[str, Any]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for model in config.models:
        if model.provider == "ollama":
            resolver = getattr(clients[model.provider], "model_digest", None)
            digest = await resolver(model.model_id, retries=config.retries) if callable(resolver) else None
            if not digest:
                raise LsegNewsError(f"cannot verify required Ollama model tag: {model.model_id}")
            digests[model.model_id] = str(digest)
        else:
            digests[model.model_id] = model.model_id
    return digests


async def score_corpus_matrix(
    *,
    input_path: str | Path,
    subset_path: str | Path,
    input_kind: str,
    config_path: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    clients: dict[str, Any],
) -> MatrixPlan:
    config = load_matrix_config(config_path)
    items = load_matrix_items(input_path, input_kind)
    subset_ids = {item.item_id for item in load_matrix_items(subset_path, input_kind)}
    unknown = subset_ids - {item.item_id for item in items}
    if unknown:
        raise LsegNewsError("matrix subset contains items absent from the main input")
    plan = matrix_plan(config, items, subset_ids)
    prompts = load_prompts(Path(prompts_path))
    prompt_ids = config.target_prompts if input_kind == "lseg" else config.benchmark_prompts
    selected_prompts: list[PromptConfig] = []
    for prompt_id in prompt_ids:
        if prompt_id not in prompts:
            raise LsegNewsError(f"matrix prompt does not exist: {prompt_id}")
        selected_prompts.append(prompts[prompt_id])
    digests = await _model_digests(config, clients)
    destination = Path(output_dir)
    config_hash = sha256_file(config_path)
    identity = {
        "config_sha256": config_hash,
        "input_sha256": sha256_file(input_path),
        "subset_sha256": sha256_file(subset_path),
        "input_kind": input_kind,
        "model_digests": digests,
        "prompt_hashes": {prompt.prompt_id: prompt.prompt_hash for prompt in selected_prompts},
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("identity") != identity:
            raise LsegNewsError("matrix resume identity does not match existing output")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise LsegNewsError(f"refusing non-empty matrix output without manifest: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, {"schema_version": MATRIX_SCHEMA_VERSION, "status": "in_progress", "identity": identity})
    scores_path = destination / "scores.jsonl"
    existing: set[tuple[str, str, str, str, int]] = set()
    if scores_path.exists():
        existing = {
            _record_key(json.loads(line))
            for line in scores_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    semaphore = asyncio.Semaphore(config.concurrency)
    write_lock = asyncio.Lock()

    async def score_one(item: MatrixItem, model: MatrixModel, prompt: PromptConfig, sample_index: int) -> None:
        key = (item.item_id, item.content_sha256, model.model_id, prompt.prompt_hash, sample_index)
        if key in existing:
            return
        async with semaphore:
            response = await clients[model.provider].classify(
                model.model_id,
                prompt,
                BlindExample(int(item.metadata["row_number"]), item.content),
                temperature=config.temperature,
                max_completion_tokens=config.max_completion_tokens,
                retries=config.retries,
            )
        record = {
            "item_id": item.item_id,
            "content_sha256": item.content_sha256,
            "model_id": model.model_id,
            "provider": model.provider,
            "model_digest": digests[model.model_id],
            "prompt_id": prompt.prompt_id,
            "prompt_hash": prompt.prompt_hash,
            "sample_index": sample_index,
            "normalized_label": response.normalized_label,
            "label_probabilities": response.label_probabilities,
            "raw_content": response.raw_content,
            "parse_status": response.parse_status,
            "status": response.status,
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "generation_id": response.generation_id,
            "error": response.error,
            "metadata": item.metadata,
        }
        async with write_lock:
            with scores_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(record) + "\n")
                handle.flush()
            existing.add(key)

    tasks = []
    for item in items:
        item_prompts = selected_prompts if item.item_id in subset_ids else selected_prompts[:1]
        for model in config.models:
            for prompt in item_prompts:
                for sample_index in range(config.samples):
                    tasks.append(score_one(item, model, prompt, sample_index))
    await asyncio.gather(*tasks)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "status": "completed",
            "completed_at": utc_now(),
            "identity": identity,
            "plan": plan.__dict__,
            "scores": {"path": scores_path.name, "sha256": sha256_file(scores_path), "rows": len(existing)},
        },
    )
    return plan
