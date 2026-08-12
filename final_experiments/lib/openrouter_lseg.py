"""Bounded, append-only OpenRouter scoring for the expanded LSEG corpus."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentiment_benchmark.artifact_io import atomic_write_json, sha256_file, sha256_text
from sentiment_benchmark.headline_value import ScorableHeadline, collect_scorable_headline_records
from sentiment_benchmark.models import DatasetRow
from sentiment_benchmark.prompts import load_prompts

from .openrouter_validation import (
    DEFAULT_RETRIES,
    MODEL_ID,
    PROMPT_HASH,
    PROMPT_ID,
    PROVIDER_NAME,
    PinnedOpenRouterClient,
    request_contract,
)

LICENCE_CONFIRMATION_DATE = "2026-08-10"
DEFAULT_CONCURRENCY = 10
DEFAULT_MAX_POPULATION = 888_155
FLUSH_SIZE = 50
LSEG_COLUMNS = (
    "headline_sha256",
    "headline",
    "matched_symbols",
    "first_timestamp",
    "explicit_target",
    "contextual",
    "market_price_technical",
    "model_id",
    "provider",
    "quantization",
    "prompt_hash",
    "label",
    "p_positive",
    "p_negative",
    "p_neutral",
    "score",
    "score_100",
    "status",
    "error",
    "latency_ms",
    "attempt_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reported_cost_usd",
    "generation_id",
    "config_hash",
)


def _headlines_path(collection_root: Path) -> Path:
    direct = collection_root / "headlines.jsonl"
    if direct.is_file():
        return direct
    matches = list(collection_root.rglob("headlines.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one headlines.jsonl below {collection_root}, found {len(matches)}")
    return matches[0]


def _existing_successes(output: Path, config_hash: str, population: set[str] | None = None) -> set[str]:
    if not output.exists():
        return set()
    successes: set[str] = set()
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LSEG_COLUMNS:
            raise ValueError("existing OpenRouter LSEG checkpoint has an incompatible schema")
        for line_number, row in enumerate(reader, start=2):
            if row.get("config_hash") != config_hash:
                raise ValueError(f"checkpoint configuration mismatch at row {line_number}")
            headline_hash = str(row.get("headline_sha256") or "")
            if row.get("status") == "success":
                if population is not None and headline_hash not in population:
                    raise ValueError(f"checkpoint success is outside the frozen population at row {line_number}")
                if headline_hash in successes:
                    raise ValueError(f"duplicate successful headline at row {line_number}: {headline_hash}")
                successes.add(headline_hash)
    return successes


def _seed_successes(paths: tuple[Path, ...], population: set[str]) -> tuple[set[str], list[dict[str, Any]]]:
    """Validate reusable successes from immutable earlier-population artifacts."""

    successes: set[str] = set()
    metadata: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"seed success artifact is missing: {path}")
        artifact_successes: set[str] = set()
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != LSEG_COLUMNS:
                raise ValueError(f"seed success artifact has an incompatible schema: {path}")
            for line_number, row in enumerate(reader, start=2):
                if row.get("status") != "success":
                    continue
                headline_hash = str(row.get("headline_sha256") or "")
                if headline_hash not in population:
                    raise ValueError(f"seed success is outside the frozen population at {path}:{line_number}")
                if row.get("model_id") != MODEL_ID:
                    raise ValueError(f"seed model mismatch at {path}:{line_number}")
                if str(row.get("provider") or "").casefold() != PROVIDER_NAME.casefold():
                    raise ValueError(f"seed provider mismatch at {path}:{line_number}")
                if row.get("quantization") != "fp8":
                    raise ValueError(f"seed quantization mismatch at {path}:{line_number}")
                if row.get("prompt_hash") != PROMPT_HASH:
                    raise ValueError(f"seed prompt mismatch at {path}:{line_number}")
                try:
                    probabilities = [float(row[name]) for name in ("p_positive", "p_negative", "p_neutral")]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid seed probabilities at {path}:{line_number}") from exc
                if (
                    not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
                    or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
                ):
                    raise ValueError(f"invalid seed probabilities at {path}:{line_number}")
                if headline_hash in artifact_successes:
                    raise ValueError(f"duplicate seed success at {path}:{line_number}: {headline_hash}")
                artifact_successes.add(headline_hash)
        overlap = successes & artifact_successes
        if overlap:
            raise ValueError(f"seed success artifacts overlap on {len(overlap):,} headline hashes")
        successes.update(artifact_successes)
        metadata.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "successful_headlines": len(artifact_successes),
            }
        )
    return successes, metadata


def _contract(
    collection_root: Path,
    headlines: Path,
    records: list[ScorableHeadline],
    output: Path,
    concurrency: int,
    retries: int,
    seed_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = load_prompts()[PROMPT_ID]
    if prompt.prompt_hash != PROMPT_HASH:
        raise ValueError(f"frozen prompt hash changed: expected {PROMPT_HASH}, got {prompt.prompt_hash}")
    source_manifest = headlines.parent / "manifest.json"
    population_hash = sha256_text("\n".join(record.headline_sha256 for record in records))
    identity = {
        "schema_version": 1,
        "input_headlines_sha256": sha256_file(headlines),
        "input_manifest_sha256": sha256_file(source_manifest) if source_manifest.exists() else None,
        "population_sha256": population_hash,
        "population": len(records),
        "model": MODEL_ID,
        "provider": PROVIDER_NAME,
        "quantization": "fp8",
        "prompt_id": PROMPT_ID,
        "prompt_hash": PROMPT_HASH,
        "request": request_contract(require_fp8=True),
        "seed_success_artifacts": [
            {"sha256": item["sha256"], "successful_headlines": item["successful_headlines"]}
            for item in seed_artifacts
        ],
    }
    config_hash = sha256_text(json.dumps(identity, sort_keys=True, separators=(",", ":")))
    return {
        **identity,
        "config_hash": config_hash,
        "collection_root": str(collection_root),
        "output": str(output),
        "execution": {"concurrency": concurrency, "retries": retries, "bounded_queue": True},
        "reuse": {"seed_success_artifacts": seed_artifacts},
        "licence": {
            "external_processing_permitted": True,
            "confirmed_by": "researcher",
            "confirmed_at": LICENCE_CONFIRMATION_DATE,
            "scope": "LSEG headline text sent for hosted sentiment inference",
            "redistribution_permitted": False,
        },
        "sharing": {"contains_licensed_headline_text": True, "source_control": False, "redistribute": False},
    }


def _append_rows(output: Path, rows: list[dict[str, Any]]) -> None:
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LSEG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


async def score_lseg_openrouter(
    collection_root: str | Path,
    output_path: str | Path,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    retries: int = DEFAULT_RETRIES,
    max_population: int = DEFAULT_MAX_POPULATION,
    limit: int | None = None,
    seed_success_paths: tuple[str | Path, ...] = (),
    client: Any | None = None,
    callback: Any = None,
) -> dict[str, Any]:
    """Score the frozen hash-ordered LSEG population with bounded concurrency."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    root = Path(collection_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    headlines = _headlines_path(root)
    records = collect_scorable_headline_records(root)
    if len(records) != max_population:
        raise ValueError(f"frozen population changed: expected {max_population:,}, found {len(records):,}")
    population = {record.headline_sha256 for record in records}
    seeds, seed_artifacts = _seed_successes(tuple(Path(path) for path in seed_success_paths), population)
    contract = _contract(root, headlines, records, output, concurrency, retries, seed_artifacts)
    config_hash = str(contract["config_hash"])
    output_completed = _existing_successes(output, config_hash, population)
    overlap = seeds & output_completed
    if overlap:
        raise ValueError(f"seed and current output overlap on {len(overlap):,} successful headline hashes")
    completed = seeds | output_completed
    pending_count = sum(record.headline_sha256 not in completed for record in records)
    attempt_limit = pending_count if limit is None else min(limit, pending_count)
    started_at = datetime.now(UTC)
    manifest = {
        **contract,
        "status": "running",
        "started_at": started_at.isoformat(),
        "counts": {
            "population": len(records),
            "already_successful": len(completed),
            "seeded_successful": len(seeds),
            "output_successful_before_run": len(output_completed),
            "pending_before_run": pending_count,
            "attempt_limit": attempt_limit,
        },
    }
    atomic_write_json(manifest_path, manifest)

    owned_client = client is None
    api_client = client or PinnedOpenRouterClient(require_fp8=True)
    queue: asyncio.Queue[tuple[int, ScorableHeadline] | None] = asyncio.Queue(maxsize=max(2, concurrency * 4))
    write_lock = asyncio.Lock()
    buffer: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    prompt_tokens = completion_tokens = 0
    reported_cost = 0.0

    async def flush(force: bool = False) -> None:
        async with write_lock:
            if not buffer or (not force and len(buffer) < FLUSH_SIZE):
                return
            batch = list(buffer)
            buffer.clear()
            await asyncio.to_thread(_append_rows, output, batch)

    async def producer() -> None:
        queued = 0
        for index, record in enumerate(records, start=1):
            if record.headline_sha256 in completed:
                continue
            if queued >= attempt_limit:
                break
            await queue.put((index, record))
            queued += 1
        for _ in range(concurrency):
            await queue.put(None)

    async def worker() -> None:
        nonlocal prompt_tokens, completion_tokens, reported_cost
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            index, record = item
            try:
                benchmark_row = DatasetRow(index, record.headline, "neutral", False, False, 1)
                try:
                    result = await api_client.classify(benchmark_row, retries=retries)
                except Exception as exc:  # Preserve progress if one unexpected client error occurs.
                    result = {
                        "status": "client_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "attempt_count": 1,
                    }
                probabilities = result.get("probabilities") or {}
                score = (
                    float(probabilities["positive"]) - float(probabilities["negative"])
                    if set(probabilities) == {"positive", "negative", "neutral"}
                    else None
                )
                row = {
                    "headline_sha256": record.headline_sha256,
                    "headline": record.headline,
                    "matched_symbols": "|".join(record.matched_symbols),
                    "first_timestamp": record.first_timestamp,
                    "explicit_target": record.explicit_target,
                    "contextual": record.contextual,
                    "market_price_technical": record.market_price_technical,
                    "model_id": MODEL_ID,
                    "provider": result.get("returned_provider") or "",
                    "quantization": "fp8",
                    "prompt_hash": PROMPT_HASH,
                    "label": result.get("normalized_label") or "",
                    "p_positive": probabilities.get("positive", ""),
                    "p_negative": probabilities.get("negative", ""),
                    "p_neutral": probabilities.get("neutral", ""),
                    "score": "" if score is None else score,
                    "score_100": "" if score is None else 100 * score,
                    "status": result.get("status") or "client_error",
                    "error": str(result.get("error") or "")[:1_000],
                    "latency_ms": result.get("latency_ms") or "",
                    "attempt_count": result.get("attempt_count") or 1,
                    "prompt_tokens": result.get("prompt_tokens") or "",
                    "completion_tokens": result.get("completion_tokens") or "",
                    "total_tokens": result.get("total_tokens") or "",
                    "reported_cost_usd": result.get("reported_cost_usd") or "",
                    "generation_id": result.get("generation_id") or "",
                    "config_hash": config_hash,
                }
                async with write_lock:
                    counters[str(row["status"])] += 1
                    prompt_tokens += int(result.get("prompt_tokens") or 0)
                    completion_tokens += int(result.get("completion_tokens") or 0)
                    reported_cost += float(result.get("reported_cost_usd") or 0.0)
                    buffer.append(row)
                    should_flush = len(buffer) >= FLUSH_SIZE
                    attempted = sum(counters.values())
                if attempted % 250 == 0 and callback is not None:
                    callback(f"attempted {attempted:,}/{attempt_limit:,}; success={counters['success']:,}")
                if should_flush:
                    await flush(force=True)
            finally:
                queue.task_done()

    started_clock = time.monotonic()
    try:
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await producer()
        await queue.join()
        await asyncio.gather(*workers)
        await flush(force=True)
    finally:
        if owned_client:
            await api_client.close()

    output_completed_after = _existing_successes(output, config_hash, population)
    completed_after = seeds | output_completed_after
    completed_at = datetime.now(UTC)
    manifest.update(
        {
            "status": "completed" if len(completed_after) == len(records) else "completed_partial",
            "completed_at": completed_at.isoformat(),
            "runtime_seconds": time.monotonic() - started_clock,
            "counts": {
                **manifest["counts"],
                "attempted_this_run": sum(counters.values()),
                "statuses_this_run": dict(counters),
                "successful_after_run": len(completed_after),
                "output_successful_after_run": len(output_completed_after),
                "remaining_after_run": len(records) - len(completed_after),
            },
            "usage_this_run": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reported_cost_usd": reported_cost,
            },
            "output_sha256": sha256_file(output) if output.is_file() else None,
        }
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_POPULATION",
    "LSEG_COLUMNS",
    "LICENCE_CONFIRMATION_DATE",
    "score_lseg_openrouter",
]
