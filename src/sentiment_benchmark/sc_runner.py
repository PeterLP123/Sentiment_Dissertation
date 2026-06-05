"""Self-consistency runner: run a model N times at temperature > 0 per row.

This produces the raw data needed to compute per-row label entropy as a
proxy for LLM sentiment ambiguity.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .dataset import load_dataset, select_rows
from .models import PromptConfig
from .openrouter import OpenRouterClient
from .self_consistency import SelfConsistencyResult
from .storage import BenchmarkStore

ProgressCallback = Callable[[str], None | Awaitable[None]]


class SelfConsistencyRunner:
    """Orchestrate multi-sample LLM classification for self-consistency analysis.

    For each row, the model is called ``num_samples`` times at ``temperature > 0``
    (or the user's chosen temperature). All responses are stored and aggregate
    consistency metrics are computed per-row and overall.
    """

    def __init__(self, client: OpenRouterClient, store: BenchmarkStore) -> None:
        self.client = client
        self.store = store

    async def run(
        self,
        model_id: str,
        prompt: PromptConfig,
        *,
        temperature: float = 0.7,
        num_samples: int = 5,
        mode: str = "pilot",
        dataset_path: str,
        max_completion_tokens: int = 64,
        concurrency: int = 1,
        retries: int = 3,
        seed: int = 42,
        sample_per_class: int = 30,
        callback: ProgressCallback | None = None,
    ) -> SelfConsistencyResult:
        self.store.initialize()

        # Load and select rows
        rows = load_dataset(dataset_path)
        # For self-consistency we always evaluate on ALL rows (not just primary) to
        # get the full picture including conflicting duplicates.
        selected = select_rows(
            rows, mode=mode, sample_per_class=sample_per_class, seed=seed  # type: ignore[arg-type]
        )

        # Create the self-consistency run
        sc_run_id = self.store.create_sc_run(
            model_id=model_id,
            temperature=temperature,
            num_samples=num_samples,
            mode=mode,
            dataset_path=dataset_path,
            prompt_hash=prompt.prompt_hash,
            scope="all",
        )

        row_map = {r.row_number: r for r in rows}

        await self._notify(
            callback,
            f"SC run {sc_run_id}: {model_id} × {num_samples} samples × {len(selected)} rows "
            f"at t={temperature}",
        )

        semaphore = asyncio.Semaphore(max(1, concurrency))
        total_cost = 0.0

        async def sample_one(row_number: int, sample_index: int) -> None:
            nonlocal total_cost
            ds_row = row_map[row_number]
            example = ds_row.blind()
            async with semaphore:
                record = await self.client.classify(
                    model_id=model_id,
                    prompt=prompt,
                    example=example,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    retries=retries,
                )
                # Record cost from generation metadata if available. A metadata
                # lookup failure must not abort the sample, mirroring BenchmarkRunner.
                if record.status == "success" and record.generation_id:
                    try:
                        meta = await self.client.get_generation_metadata(
                            record.generation_id, retries=retries
                        )
                    except Exception as exc:  # noqa: BLE001 - best-effort cost tracking
                        meta = None
                        await self._notify(
                            callback,
                            f"  metadata lookup failed for row {row_number} "
                            f"sample {sample_index + 1}: {exc}",
                        )
                    if meta and meta.get("total_cost"):
                        total_cost += float(meta["total_cost"])

            self.store.save_sc_sample(
                sc_run_id,
                row_number=row_number,
                sample_index=sample_index,
                normalized_label=record.normalized_label,
                parse_status=record.parse_status,
                status=record.status,
                latency_ms=record.latency_ms,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                total_tokens=record.total_tokens,
                generation_id=record.generation_id,
                error=record.error,
            )
            await self._notify(
                callback,
                f"  sample {sample_index + 1}/{num_samples} row {row_number}: "
                f"{record.normalized_label or record.status}",
            )

        async def sample_row(row_number: int) -> None:
            """Sample one row ``num_samples`` times (concurrency bounded by the semaphore)."""
            tasks = [
                sample_one(row_number, sample_index)
                for sample_index in range(num_samples)
            ]
            await asyncio.gather(*tasks)

        for row in selected:
            await sample_row(row.row_number)

        # Mark run complete and persist the accumulated cost so it is available to
        # build_sc_row_consistency (below) and to later read-only analysis commands.
        self.store.mark_sc_run_complete(sc_run_id, total_cost=total_cost)

        # Build results from the stored samples (cost is read back from sc_runs).
        result = self.store.build_sc_row_consistency(
            sc_run_id,
            model_id=model_id,
            temperature=temperature,
            num_samples=num_samples,
            scope="all",
        )

        self.store.save_sc_result(sc_run_id, result)
        await self._notify(
            callback,
            f"SC run {sc_run_id} complete: mean entropy={result.mean_entropy:.4f}, "
            f"majority vote acc={result.majority_vote_accuracy:.4f}, "
            f"cost=${total_cost:.4f}",
        )
        return result

    async def _notify(self, callback: ProgressCallback | None, message: str) -> None:
        if callback is None:
            return
        result = callback(message)
        if result is not None:
            await result