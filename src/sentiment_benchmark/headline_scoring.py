"""Resumable LLM scoring of a collection's unique headlines.

Scores exactly the population that ``analyze_headline_value`` aggregates
(unique in-window company-matched headlines) and writes an append-only CSV
keyed by normalized-headline hash. Feed the CSV back via
``analyze-headline-value --llm-scores`` to backtest ``llm/<model>`` scorers
next to the lexicon ones — same corpus, same aggregation, only the scorer
swaps.
"""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budgets import resolve_max_completion_tokens
from .constants import SOFT_LABEL_MIN_COMPLETION_TOKENS
from .headline_value import HeadlineValueError, collect_scorable_headlines
from .models import BlindExample, PromptConfig

LABEL_SCORES = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_COLUMNS = ("headline_sha256", "headline", "model_id", "label", "score", "status", "error")
_FLUSH_SIZE = 100


@dataclass(frozen=True)
class HeadlineScoringSummary:
    total_unique: int
    already_scored: int
    attempted: int
    succeeded: int
    failed: int
    output_path: Path


def _existing_success_shas(path: Path, model_id: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row.get("headline_sha256") or "")
            for row in csv.DictReader(handle)
            if str(row.get("model_id") or "") == model_id and str(row.get("status") or "") == "success"
        }


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
    callback: Any = None,
) -> HeadlineScoringSummary:
    headlines = collect_scorable_headlines(
        collection_root,
        source_codes=source_codes,
        direct_company_only=direct_company_only,
    )
    if max_population is not None and len(headlines) > max_population:
        raise HeadlineValueError(
            f"filtered scoring population has {len(headlines):,} headlines, exceeding "
            f"--max-population {max_population:,}; tighten the frozen filter or reduce the company universe"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_success_shas(output, model_id)
    pending = [(sha, text) for sha, text in headlines if sha not in done]
    if limit is not None:
        pending = pending[:limit]

    # label_only prompts map labels to +1/0/-1; soft_label prompts score the
    # elicited distribution as P(positive) - P(negative) in [-1, 1].
    soft_label = prompt.output_mode == "soft_label"
    budget = resolve_max_completion_tokens(model_id, max_completion_tokens)
    if soft_label:
        budget = max(budget, SOFT_LABEL_MIN_COMPLETION_TOKENS)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    buffer: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0

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

    async def _score_one(index: int, sha: str, text: str) -> None:
        nonlocal succeeded, failed
        try:
            async with semaphore:
                record = await client.classify(
                    model_id=model_id,
                    prompt=prompt,
                    example=BlindExample(index, text),
                    temperature=temperature,
                    max_completion_tokens=budget,
                    retries=retries,
                )
            label = record.normalized_label or ""
            if soft_label:
                probabilities = record.label_probabilities or {}
                score = (
                    float(probabilities.get("positive", 0.0)) - float(probabilities.get("negative", 0.0))
                    if probabilities
                    else None
                )
            else:
                score = LABEL_SCORES.get(label)
            if record.status == "success" and score is not None:
                status, error = "success", ""
                succeeded += 1
            else:
                status = record.status if record.status != "success" else "invalid_label"
                error = record.error or ""
                failed += 1
        except Exception as exc:  # one bad headline must not abort the sweep
            label, score, status, error = "", None, "client_error", f"{type(exc).__name__}: {exc}"
            failed += 1
        buffer.append(
            {
                "headline_sha256": sha,
                "headline": text,
                "model_id": model_id,
                "label": label,
                "score": "" if score is None else score,
                "status": status,
                "error": error[:300],
            }
        )
        if callback is not None and (succeeded + failed) % 250 == 0:
            callback(f"scored {succeeded + failed}/{len(pending)} headlines ({failed} failed)")
        await _flush()

    await asyncio.gather(*(_score_one(index, sha, text) for index, (sha, text) in enumerate(pending, start=1)))
    await _flush(force=True)

    return HeadlineScoringSummary(
        total_unique=len(headlines),
        already_scored=len(done),
        attempted=len(pending),
        succeeded=succeeded,
        failed=failed,
        output_path=output,
    )
