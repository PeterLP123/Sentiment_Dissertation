"""Append-only, resumable LLM scoring for a frozen unique-headline population."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifact_io import atomic_write_text, sha256_file
from .budgets import resolve_max_completion_tokens
from .headline_value import HeadlineValueError, ScorableHeadline, collect_scorable_headline_records
from .models import BlindExample, PromptConfig
from .runtime_metadata import collect_run_environment

LABEL_SCORES = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_COLUMNS = (
    "headline_sha256",
    "headline",
    "matched_symbols",
    "first_timestamp",
    "explicit_target",
    "contextual",
    "market_price_technical",
    "model_id",
    "model_digest",
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
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_per_second",
    "completion_tokens_per_second",
)
_FLUSH_SIZE = 100


@dataclass(frozen=True)
class HeadlineScoringSummary:
    total_unique: int
    already_scored: int
    attempted: int
    succeeded: int
    failed: int
    output_path: Path
    manifest_path: Path


def _rate(tokens: int | None, duration_ns: Any) -> float | None:
    if tokens is None or isinstance(duration_ns, bool) or not isinstance(duration_ns, int | float) or duration_ns <= 0:
        return None
    return float(tokens) / (float(duration_ns) / 1_000_000_000)


def _input_headlines_path(collection_root: str | Path) -> Path:
    root = Path(collection_root)
    direct = root / "headlines.jsonl"
    if direct.exists():
        return direct
    matches = list(root.rglob("headlines.jsonl"))
    if len(matches) != 1:
        raise HeadlineValueError(f"expected exactly one headlines.jsonl under {root}, found {len(matches)}")
    return matches[0]


def _nvidia_device_metadata() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append({"name": parts[0], "driver_version": parts[1], "memory_total_mib": parts[2]})
    return devices


def _existing_success_shas(path: Path, model_id: str, model_digest: str | None) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("model_id") or "") != model_id or str(row.get("status") or "") != "success":
                continue
            stored_digest = str(row.get("model_digest") or "") or None
            if model_digest is not None and stored_digest != model_digest:
                raise HeadlineValueError(
                    f"cannot resume {path}: model tag {model_id!r} now resolves to digest {model_digest}, "
                    f"but successful rows contain {stored_digest or 'no digest'}"
                )
            done.add(str(row.get("headline_sha256") or ""))
    return done


async def _model_digest(client: Any, model_id: str, retries: int) -> str | None:
    resolver = getattr(client, "model_digest", None)
    if not callable(resolver):
        return None
    return await resolver(model_id, retries=retries)


async def score_headlines(
    client: Any,
    *,
    collection_root: str | Path,
    model_id: str,
    prompt: PromptConfig,
    output_path: str | Path,
    concurrency: int = 1,
    retries: int = 3,
    temperature: float = 0.0,
    max_completion_tokens: int = 64,
    limit: int | None = None,
    source_codes: tuple[str, ...] = (),
    direct_company_only: bool = False,
    max_population: int | None = None,
    ollama_think: bool | None = None,
    structured_output: bool = False,
    callback: Any = None,
) -> HeadlineScoringSummary:
    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    records = collect_scorable_headline_records(
        collection_root,
        source_codes=source_codes,
        direct_company_only=direct_company_only,
    )
    if max_population is not None and len(records) > max_population:
        raise HeadlineValueError(
            f"filtered scoring population has {len(records):,} headlines, exceeding "
            f"--max-population {max_population:,}; tighten the frozen filter or reduce the company universe"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    model_digest = await _model_digest(client, model_id, retries)
    done = _existing_success_shas(output, model_id, model_digest)
    pending = [record for record in records if record.headline_sha256 not in done]
    if limit is not None:
        pending = pending[:limit]

    soft_label = prompt.output_mode == "soft_label"
    # Headline classification intentionally honors the caller's small JSON
    # completion budget; the generic benchmark runner retains its larger floor.
    budget = resolve_max_completion_tokens(model_id, max_completion_tokens)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    buffer: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    latencies_ms: list[float] = []
    prompt_rates: list[float] = []
    completion_rates: list[float] = []

    def _append_rows(rows: list[dict[str, Any]]) -> None:
        write_header = not output.exists() or output.stat().st_size == 0
        with output.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    async def _flush(force: bool = False) -> None:
        async with write_lock:
            if not buffer or (not force and len(buffer) < _FLUSH_SIZE):
                return
            batch = list(buffer)
            buffer.clear()
            await asyncio.to_thread(_append_rows, batch)

    async def _score_one(index: int, item: ScorableHeadline) -> None:
        nonlocal succeeded, failed, total_prompt_tokens, total_completion_tokens
        probabilities: dict[str, float] = {}
        prompt_rate = None
        completion_rate = None
        try:
            kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_completion_tokens": budget,
                "retries": retries,
            }
            if ollama_think is not None:
                kwargs["ollama_think"] = ollama_think
            async with semaphore:
                response = await client.classify(
                    model_id=model_id,
                    prompt=prompt,
                    example=BlindExample(index, item.headline),
                    **kwargs,
                )
            label = response.normalized_label or ""
            if soft_label:
                probabilities = response.label_probabilities or {}
                score = (
                    float(probabilities["positive"]) - float(probabilities["negative"])
                    if set(probabilities) == {"positive", "negative", "neutral"}
                    else None
                )
            else:
                score = LABEL_SCORES.get(label)
            if score is not None and not -1.0 <= score <= 1.0:
                score = None
            if response.status == "success" and response.parse_status == "valid" and score is not None:
                status, error = "success", ""
                succeeded += 1
            else:
                status = response.status if response.status != "success" else "invalid_label"
                error = response.error or "invalid or incomplete model output"
                failed += 1
            latency_ms = response.latency_ms
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens
            total_tokens = response.total_tokens
            raw = response.raw_response_json or {}
            prompt_rate = _rate(prompt_tokens, raw.get("prompt_eval_duration"))
            completion_rate = _rate(completion_tokens, raw.get("eval_duration"))
            if prompt_tokens is not None:
                total_prompt_tokens += prompt_tokens
            if completion_tokens is not None:
                total_completion_tokens += completion_tokens
            if latency_ms is not None:
                latencies_ms.append(float(latency_ms))
            if prompt_rate is not None:
                prompt_rates.append(prompt_rate)
            if completion_rate is not None:
                completion_rates.append(completion_rate)
        except Exception as exc:  # one bad headline must not abort the sweep
            label, score, status, error = "", None, "client_error", f"{type(exc).__name__}: {exc}"
            latency_ms = prompt_tokens = completion_tokens = total_tokens = None
            failed += 1
        buffer.append(
            {
                "headline_sha256": item.headline_sha256,
                "headline": item.headline,
                "matched_symbols": "|".join(item.matched_symbols),
                "first_timestamp": item.first_timestamp,
                "explicit_target": item.explicit_target,
                "contextual": item.contextual,
                "market_price_technical": item.market_price_technical,
                "model_id": model_id,
                "model_digest": model_digest or "",
                "prompt_hash": prompt.prompt_hash,
                "label": label,
                "p_positive": probabilities.get("positive", ""),
                "p_negative": probabilities.get("negative", ""),
                "p_neutral": probabilities.get("neutral", ""),
                "score": "" if score is None else score,
                "score_100": "" if score is None else 100 * score,
                "status": status,
                "error": error[:500],
                "latency_ms": "" if latency_ms is None else latency_ms,
                "prompt_tokens": "" if prompt_tokens is None else prompt_tokens,
                "completion_tokens": "" if completion_tokens is None else completion_tokens,
                "total_tokens": "" if total_tokens is None else total_tokens,
                "prompt_tokens_per_second": "" if prompt_rate is None else prompt_rate,
                "completion_tokens_per_second": "" if completion_rate is None else completion_rate,
            }
        )
        if callback is not None and (succeeded + failed) % 250 == 0:
            callback(f"scored {succeeded + failed}/{len(pending)} headlines ({failed} failed)")
        await _flush()

    await asyncio.gather(*(_score_one(index, item) for index, item in enumerate(pending, start=1)))
    await _flush(force=True)

    completed_at = datetime.now(UTC)
    duration_seconds = time.monotonic() - started_clock
    headlines_path = _input_headlines_path(collection_root)
    source_manifest = headlines_path.parent / "manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "completed" if failed == 0 else "completed_with_failures",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "runtime_seconds": duration_seconds,
        "inputs": {
            "headlines_path": str(headlines_path),
            "headlines_sha256": sha256_file(headlines_path),
            "collection_manifest_path": str(source_manifest) if source_manifest.exists() else None,
            "collection_manifest_sha256": sha256_file(source_manifest) if source_manifest.exists() else None,
        },
        "model": {"tag": model_id, "digest": model_digest},
        "prompt": {"id": prompt.prompt_id, "hash": prompt.prompt_hash, "output_mode": prompt.output_mode},
        "inference": {
            "temperature": temperature,
            "max_completion_tokens": budget,
            "concurrency": concurrency,
            "retries": retries,
            "ollama_think": ollama_think,
            "structured_output": structured_output,
            "source_codes": list(source_codes),
            "direct_company_only": direct_company_only,
            "limit": limit,
            "environment": {
                name: os.getenv(name)
                for name in ("OLLAMA_NUM_PARALLEL", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_KEEP_ALIVE")
            },
        },
        "counts": {
            "population": len(records),
            "explicit_target": sum(item.explicit_target for item in records),
            "contextual": sum(item.contextual for item in records),
            "market_price_technical": sum(item.market_price_technical for item in records),
            "already_scored": len(done),
            "attempted": len(pending),
            "succeeded": succeeded,
            "failed": failed,
        },
        "performance": {
            "valid_headlines_per_second": succeeded / duration_seconds if duration_seconds > 0 else None,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "mean_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
            "mean_prompt_tokens_per_second": sum(prompt_rates) / len(prompt_rates) if prompt_rates else None,
            "mean_completion_tokens_per_second": sum(completion_rates) / len(completion_rates) if completion_rates else None,
        },
        "device": {"nvidia_gpus": _nvidia_device_metadata()},
        "environment": collect_run_environment(),
        "outputs": {
            "scores_path": str(output),
            "scores_sha256": sha256_file(output) if output.exists() else None,
        },
        "sharing": {
            "contains_licensed_headline_text": True,
            "redistribute": False,
            "source_control": False,
        },
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return HeadlineScoringSummary(
        total_unique=len(records),
        already_scored=len(done),
        attempted=len(pending),
        succeeded=succeeded,
        failed=failed,
        output_path=output,
        manifest_path=manifest_path,
    )
