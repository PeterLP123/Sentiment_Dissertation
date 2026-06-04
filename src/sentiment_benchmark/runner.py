from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .dataset import load_dataset, select_rows
from .metrics import evaluate_responses
from .models import BlindExample, RunConfig
from .openrouter import OpenRouterClient
from .storage import BenchmarkStore

ProgressCallback = Callable[[str], None | Awaitable[None]]
RunEvent = dict[str, Any]
EventCallback = Callable[[RunEvent], None | Awaitable[None]]


@dataclass(frozen=True)
class RunSummary:
    run_id: int
    selected_row_count: int
    model_count: int
    status: str = "completed"


class BenchmarkRunner:
    def __init__(self, client: OpenRouterClient, store: BenchmarkStore) -> None:
        self.client = client
        self.store = store

    async def _notify(self, callback: ProgressCallback | None, message: str) -> None:
        if callback is None:
            return
        result = callback(message)
        if result is not None:
            await result

    async def _emit(self, event_callback: EventCallback | None, event: RunEvent) -> None:
        if event_callback is None:
            return
        result = event_callback(event)
        if result is not None:
            await result

    async def run(
        self,
        config: RunConfig,
        resume_run_id: int | None = None,
        callback: ProgressCallback | None = None,
        event_callback: EventCallback | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> RunSummary:
        self.store.initialize()
        rows = load_dataset(config.dataset_path)
        self.store.upsert_dataset(rows, config.dataset_path)
        self.store.save_prompt(config.prompt)

        if resume_run_id is None:
            selected_rows = select_rows(rows, config.mode, config.sample_per_class, config.seed)
            run_id = self.store.create_run(config, [row.row_number for row in selected_rows])
        else:
            selected_numbers = set(self.store.get_run_selected_rows(resume_run_id))
            selected_rows = [row for row in rows if row.row_number in selected_numbers]
            run_id = resume_run_id

        selected_by_number = {row.row_number: row for row in selected_rows}
        semaphore = asyncio.Semaphore(max(1, config.concurrency))

        await self._emit(
            event_callback,
            {
                "type": "run_started",
                "run_id": run_id,
                "models": list(config.models),
                "rows_per_model": len(selected_rows),
            },
        )

        async def classify_one(model_id: str, example: BlindExample) -> None:
            if self.store.response_exists(run_id, model_id, example.row_number, config.prompt.prompt_hash):
                await self._notify(callback, f"Skipped existing response: run={run_id} model={model_id} row={example.row_number}")
                await self._emit(
                    event_callback,
                    {
                        "type": "row_completed",
                        "model_id": model_id,
                        "row_number": example.row_number,
                        "status": "skipped",
                        "latency_ms": None,
                    },
                )
                return
            async with semaphore:
                record = await self.client.classify(
                    model_id=model_id,
                    prompt=config.prompt,
                    example=example,
                    temperature=config.temperature,
                    max_completion_tokens=config.max_completion_tokens,
                    retries=config.retries,
                )
                self.store.save_response(run_id, record)
                if record.status == "success" and record.generation_id:
                    metadata = await self.client.get_generation_metadata(record.generation_id, retries=config.retries)
                    if metadata:
                        self.store.save_generation_metadata(
                            run_id=run_id,
                            row_number=record.row_number,
                            model_id=model_id,
                            generation_id=record.generation_id,
                            metadata=metadata,
                        )
                await self._notify(
                    callback,
                    f"Saved response: run={run_id} model={model_id} row={example.row_number} status={record.status} label={record.normalized_label or '-'}",
                )
                await self._emit(
                    event_callback,
                    {
                        "type": "row_completed",
                        "model_id": model_id,
                        "row_number": example.row_number,
                        "status": record.status,
                        "latency_ms": record.latency_ms,
                    },
                )

        final_status = "completed"
        for model_id in config.models:
            if cancel_event is not None and cancel_event.is_set():
                final_status = "cancelled"
                break
            self.store.upsert_run_model(run_id, model_id, "running")
            await self._notify(callback, f"Starting model {model_id} on {len(selected_rows)} rows")
            await self._emit(
                event_callback,
                {"type": "model_started", "model_id": model_id, "total_rows": len(selected_rows)},
            )
            await asyncio.gather(*(classify_one(model_id, row.blind()) for row in selected_rows))
            responses = self.store.fetch_responses(run_id, model_id)
            all_rows_for_metrics = [row for row in rows if row.row_number in selected_by_number]
            primary = evaluate_responses(all_rows_for_metrics, responses, model_id=model_id, scope="primary")
            audit = evaluate_responses(all_rows_for_metrics, responses, model_id=model_id, scope="all")
            self.store.save_metrics(run_id, primary)
            self.store.save_metrics(run_id, audit)
            self.store.upsert_run_model(run_id, model_id, "completed")
            await self._notify(callback, f"Completed model {model_id}: primary accuracy={primary.accuracy:.4f}")
            await self._emit(
                event_callback,
                {"type": "model_completed", "model_id": model_id, "accuracy": primary.accuracy},
            )

        self.store.mark_run_complete(run_id, status=final_status)
        await self._emit(
            event_callback,
            {"type": "run_completed", "run_id": run_id, "status": final_status},
        )
        return RunSummary(
            run_id=run_id,
            selected_row_count=len(selected_rows),
            model_count=len(config.models),
            status=final_status,
        )

