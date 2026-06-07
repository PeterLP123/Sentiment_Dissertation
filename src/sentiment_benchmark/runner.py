from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from .budgets import resolve_max_completion_tokens
from .dataset import load_dataset, select_rows
from .demonstrations import demonstration_pool, select_demonstrations
from .metrics import evaluate_responses
from .models import BlindExample, DatasetRow, PromptConfig, RunConfig, RunResumeSettings
from .prompts import with_demonstrations
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
    def __init__(self, client: Any, store: BenchmarkStore) -> None:
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

    @staticmethod
    def _build_prompt(config: RunConfig, rows: list[DatasetRow], selected_rows: list[DatasetRow]) -> PromptConfig:
        if config.few_shot_k <= 0:
            return config.prompt
        seed = config.few_shot_seed if config.few_shot_seed is not None else config.seed
        pool = demonstration_pool(rows, [row.row_number for row in selected_rows])
        demonstrations = select_demonstrations(pool, config.few_shot_k, seed)
        return with_demonstrations(config.prompt, demonstrations, seed)

    def _resolve_selected_rows(
        self,
        rows: list[DatasetRow],
        config: RunConfig,
        resume_run_id: int | None,
    ) -> tuple[RunConfig, list[DatasetRow], RunResumeSettings | None]:
        if resume_run_id is None:
            selected = select_rows(rows, config.mode, config.sample_per_class, config.seed)
            return config, selected, None

        resume_settings = self.store.get_run_resume_settings(resume_run_id)
        config = replace(
            config,
            few_shot_k=resume_settings.few_shot_k,
            few_shot_seed=resume_settings.few_shot_seed,
        )
        selected_numbers = set(self.store.get_run_selected_rows(resume_run_id))
        selected = [row for row in rows if row.row_number in selected_numbers]
        return config, selected, resume_settings

    def _apply_prompt(
        self,
        config: RunConfig,
        rows: list[DatasetRow],
        selected_rows: list[DatasetRow],
        resume_run_id: int | None,
        resume_settings: RunResumeSettings | None,
    ) -> RunConfig:
        config = replace(config, prompt=self._build_prompt(config, rows, selected_rows))
        if resume_settings is not None and config.prompt.prompt_hash != resume_settings.prompt_hash:
            raise ValueError(
                f"Rebuilt prompt hash {config.prompt.prompt_hash!r} does not match run {resume_run_id} "
                f"stored hash {resume_settings.prompt_hash!r}. Use the same dataset, seed, and row selection."
            )
        self.store.save_prompt(config.prompt)
        return config

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

        config, selected_rows, resume_settings = self._resolve_selected_rows(rows, config, resume_run_id)
        config = self._apply_prompt(config, rows, selected_rows, resume_run_id, resume_settings)

        if resume_run_id is None:
            run_id = self.store.create_run(config, [row.row_number for row in selected_rows])
        else:
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
            if self.store.successful_response_exists(run_id, model_id, example.row_number, config.prompt.prompt_hash):
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
            max_completion_tokens = resolve_max_completion_tokens(
                model_id,
                config.max_completion_tokens,
                config.model_max_completion_tokens,
                config.reasoning_max_completion_tokens,
            )
            async with semaphore:
                record = await self.client.classify(
                    model_id=model_id,
                    prompt=config.prompt,
                    example=example,
                    temperature=config.temperature,
                    max_completion_tokens=max_completion_tokens,
                    retries=config.retries,
                )
                self.store.save_response(run_id, record)
                if record.status == "success" and record.generation_id:
                    try:
                        metadata = await self.client.get_generation_metadata(record.generation_id, retries=config.retries)
                    except Exception as exc:
                        metadata = None
                        await self._notify(
                            callback,
                            "Metadata lookup failed: "
                            f"run={run_id} model={model_id} row={example.row_number} "
                            f"generation={record.generation_id} error={exc}",
                        )
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
                    f"Saved response: run={run_id} model={model_id} row={example.row_number} "
                    f"status={record.status} label={record.normalized_label or '-'}",
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
            model_cancelled = False
            concurrency = max(1, config.concurrency)
            for start in range(0, len(selected_rows), concurrency):
                if cancel_event is not None and cancel_event.is_set():
                    model_cancelled = True
                    final_status = "cancelled"
                    break
                batch = selected_rows[start : start + concurrency]
                await asyncio.gather(*(classify_one(model_id, row.blind()) for row in batch))
                if cancel_event is not None and cancel_event.is_set():
                    model_cancelled = True
                    final_status = "cancelled"
                    break
            responses = self.store.fetch_responses(run_id, model_id)
            all_rows_for_metrics = [row for row in rows if row.row_number in selected_by_number]
            primary = evaluate_responses(all_rows_for_metrics, responses, model_id=model_id, scope="primary")
            audit = evaluate_responses(all_rows_for_metrics, responses, model_id=model_id, scope="all")
            self.store.save_metrics(run_id, primary)
            self.store.save_metrics(run_id, audit)
            model_status = "cancelled" if model_cancelled else "completed"
            self.store.upsert_run_model(run_id, model_id, model_status)
            await self._notify(callback, f"{model_status.title()} model {model_id}: primary accuracy={primary.accuracy:.4f}")
            await self._emit(
                event_callback,
                {
                    "type": "model_completed",
                    "model_id": model_id,
                    "accuracy": primary.accuracy,
                    "status": model_status,
                },
            )
            if model_cancelled:
                break

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
