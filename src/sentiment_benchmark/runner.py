from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .budgets import resolve_max_completion_tokens
from .constants import SOFT_LABEL_MIN_COMPLETION_TOKENS
from .dataset import load_dataset, select_rows
from .demonstrations import demonstration_pool, select_demonstrations
from .metrics import evaluate_responses
from .models import BlindExample, DatasetRow, LLMResponseRecord, PromptConfig, RunConfig, RunResumeSettings
from .prompts import with_demonstrations
from .storage import BenchmarkStore

ProgressCallback = Callable[[str], None | Awaitable[None]]
RunEvent = dict[str, Any]
EventCallback = Callable[[RunEvent], None | Awaitable[None]]


class CancelEvent(Protocol):
    """Anything that can report cancellation via ``is_set()``.

    The runner is driven from the TUI in a worker thread, so the concrete type
    is a thread-safe ``threading.Event`` rather than ``asyncio.Event``; only
    ``is_set()`` is ever read here.
    """

    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class RunSummary:
    run_id: int
    selected_row_count: int
    model_count: int
    status: str = "completed"


@dataclass(frozen=True)
class _RunContext:
    """Run-stable state shared by the per-model and per-row workers."""

    run_id: int
    config: RunConfig
    rows: list[DatasetRow]
    selected_rows: list[DatasetRow]
    selected_by_number: dict[int, DatasetRow]
    semaphore: asyncio.Semaphore
    cancel_event: CancelEvent | None
    callback: ProgressCallback | None
    event_callback: EventCallback | None


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
        restored: dict[str, Any] = {
            "few_shot_k": resume_settings.few_shot_k,
            "few_shot_seed": resume_settings.few_shot_seed,
        }
        # Generation settings are restored from the stored run so resumed rows are
        # produced exactly like the originals, regardless of the flags passed now.
        request = resume_settings.request
        for key in ("temperature", "max_completion_tokens", "reasoning_max_completion_tokens", "ollama_think"):
            if key in request:
                restored[key] = request[key]
        overrides = request.get("model_max_completion_tokens")
        if isinstance(overrides, dict):
            restored["model_max_completion_tokens"] = {str(k): int(v) for k, v in overrides.items()}
        config = replace(config, **restored)
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

    async def _classify_row(self, ctx: _RunContext, model_id: str, example: BlindExample, completed_rows: set[int]) -> None:
        if example.row_number in completed_rows:
            await self._notify(ctx.callback, f"Skipped existing response: run={ctx.run_id} model={model_id} row={example.row_number}")
            await self._emit(
                ctx.event_callback,
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
            ctx.config.max_completion_tokens,
            ctx.config.model_max_completion_tokens,
            ctx.config.reasoning_max_completion_tokens,
        )
        if ctx.config.prompt.output_mode == "soft_label":
            max_completion_tokens = max(max_completion_tokens, SOFT_LABEL_MIN_COMPLETION_TOKENS)
        async with ctx.semaphore:
            # Re-check after waiting for a slot: rows queued behind the
            # semaphore when the run is cancelled must not issue requests.
            if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                return
            classify_kwargs = {
                "model_id": model_id,
                "prompt": ctx.config.prompt,
                "example": example,
                "temperature": ctx.config.temperature,
                "max_completion_tokens": max_completion_tokens,
                "retries": ctx.config.retries,
            }
            if ctx.config.provider == "ollama":
                classify_kwargs["ollama_think"] = ctx.config.ollama_think
            record = await self.client.classify(**classify_kwargs)
            # Store writes run in a worker thread: on the libsql backend each
            # write is a network round trip that would otherwise block every
            # in-flight request coroutine.
            await asyncio.to_thread(self.store.save_response, ctx.run_id, record)
            if record.status == "success" and record.generation_id:
                try:
                    metadata = await self.client.get_generation_metadata(record.generation_id, retries=ctx.config.retries)
                except Exception as exc:
                    metadata = None
                    await self._notify(
                        ctx.callback,
                        "Metadata lookup failed: "
                        f"run={ctx.run_id} model={model_id} row={example.row_number} "
                        f"generation={record.generation_id} error={exc}",
                    )
                if metadata:
                    await asyncio.to_thread(
                        self.store.save_generation_metadata,
                        run_id=ctx.run_id,
                        row_number=record.row_number,
                        model_id=model_id,
                        generation_id=record.generation_id,
                        metadata=metadata,
                    )
            failure_detail = f" error={record.error[:160]}" if record.status != "success" and record.error else ""
            await self._notify(
                ctx.callback,
                f"Saved response: run={ctx.run_id} model={model_id} row={example.row_number} "
                f"status={record.status} label={record.normalized_label or '-'}{failure_detail}",
            )
            await self._emit(
                ctx.event_callback,
                {
                    "type": "row_completed",
                    "model_id": model_id,
                    "row_number": example.row_number,
                    "status": record.status,
                    "latency_ms": record.latency_ms,
                },
            )

    async def _classify_row_guarded(self, ctx: _RunContext, model_id: str, example: BlindExample, completed_rows: set[int]) -> None:
        # Unattempted rows must leave no record so a resume can pick them up.
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            return
        try:
            await self._classify_row(ctx, model_id, example, completed_rows)
        except Exception as exc:
            # Record the failure instead of letting it propagate: one bad row
            # must not abort the batch loop or leave the run unfinalized.
            record = LLMResponseRecord(
                row_number=example.row_number,
                model_id=model_id,
                prompt_hash=ctx.config.prompt.prompt_hash,
                raw_content=None,
                normalized_label=None,
                parse_status="error",
                status="client_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            try:
                await asyncio.to_thread(self.store.save_response, ctx.run_id, record)
            except Exception as save_exc:
                await self._notify(
                    ctx.callback,
                    f"Could not record row failure: run={ctx.run_id} model={model_id} "
                    f"row={example.row_number} error={save_exc}",
                )
            await self._notify(
                ctx.callback,
                f"Row failed: run={ctx.run_id} model={model_id} row={example.row_number} error={exc}",
            )
            await self._emit(
                ctx.event_callback,
                {
                    "type": "row_completed",
                    "model_id": model_id,
                    "row_number": example.row_number,
                    "status": "client_error",
                    "latency_ms": None,
                },
            )

    async def _run_model(self, ctx: _RunContext, model_id: str) -> str:
        self.store.upsert_run_model(ctx.run_id, model_id, "running")
        await self._notify(ctx.callback, f"Starting model {model_id} on {len(ctx.selected_rows)} rows")
        await self._emit(
            ctx.event_callback,
            {"type": "model_started", "model_id": model_id, "total_rows": len(ctx.selected_rows)},
        )
        model_cancelled = False
        model_failed = False
        # One query up front instead of one existence check per row: per-row
        # checks open a backend connection each, which stalls full-dataset
        # runs for minutes before the first request on the libsql backend.
        completed_rows = self.store.successful_row_numbers(ctx.run_id, model_id, ctx.config.prompt.prompt_hash)
        # Dispatch every row at once and let ctx.semaphore bound the in-flight
        # requests. Fixed-size gather batches would make all slots wait for the
        # slowest request in each batch (e.g. one row sleeping on Retry-After),
        # which starves throughput on rate-paced providers.
        results = await asyncio.gather(
            *(self._classify_row_guarded(ctx, model_id, row.blind(), completed_rows) for row in ctx.selected_rows),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            model_cancelled = True
        primary = None
        try:
            responses = self.store.fetch_responses(ctx.run_id, model_id)
            all_rows_for_metrics = [row for row in ctx.rows if row.row_number in ctx.selected_by_number]
            primary = evaluate_responses(all_rows_for_metrics, responses, model_id=model_id, scope="primary")
            audit = evaluate_responses(all_rows_for_metrics, responses, model_id=model_id, scope="all")
            self.store.save_metrics(ctx.run_id, primary)
            self.store.save_metrics(ctx.run_id, audit)
        except Exception as exc:
            model_failed = True
            await self._notify(ctx.callback, f"Metrics computation failed: run={ctx.run_id} model={model_id} error={exc}")
        if model_cancelled:
            model_status = "cancelled"
        elif model_failed:
            model_status = "failed"
        else:
            model_status = "completed"
        self.store.upsert_run_model(ctx.run_id, model_id, model_status)
        if primary is not None:
            await self._notify(ctx.callback, f"{model_status.title()} model {model_id}: primary accuracy={primary.accuracy:.4f}")
        await self._emit(
            ctx.event_callback,
            {
                "type": "model_completed",
                "model_id": model_id,
                "accuracy": primary.accuracy if primary is not None else None,
                "status": model_status,
            },
        )
        return model_status

    async def run(
        self,
        config: RunConfig,
        resume_run_id: int | None = None,
        callback: ProgressCallback | None = None,
        event_callback: EventCallback | None = None,
        cancel_event: CancelEvent | None = None,
    ) -> RunSummary:
        await self._notify(callback, "Preparing run database...")
        self.store.initialize()
        await self._notify(callback, f"Loading dataset: {config.dataset_path}")
        rows = load_dataset(config.dataset_path)
        if self.store.dataset_snapshot_exists(config.dataset_path, len(rows)):
            await self._notify(callback, f"Dataset already recorded: {len(rows)} row(s).")
        else:
            await self._notify(callback, f"Recording {len(rows)} dataset row(s) in the run database...")
            self.store.upsert_dataset(rows, config.dataset_path)

        config, selected_rows, resume_settings = self._resolve_selected_rows(rows, config, resume_run_id)
        config = self._apply_prompt(config, rows, selected_rows, resume_run_id, resume_settings)

        if resume_run_id is None:
            run_id = self.store.create_run(config, [row.row_number for row in selected_rows])
            await self._notify(callback, f"Created run {run_id}; selected {len(selected_rows)} row(s) per model.")
        else:
            run_id = resume_run_id
            await self._notify(
                callback,
                f"Resuming run {run_id}; restored stored generation settings "
                f"(temperature={config.temperature}, max_completion_tokens={config.max_completion_tokens}, "
                f"ollama_think={config.ollama_think}).",
            )

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

        ctx = _RunContext(
            run_id=run_id,
            config=config,
            rows=rows,
            selected_rows=selected_rows,
            selected_by_number=selected_by_number,
            semaphore=semaphore,
            cancel_event=cancel_event,
            callback=callback,
            event_callback=event_callback,
        )
        final_status = "completed"
        try:
            for model_id in config.models:
                if cancel_event is not None and cancel_event.is_set():
                    final_status = "cancelled"
                    break
                model_status = await self._run_model(ctx, model_id)
                if model_status == "failed":
                    final_status = "failed"
                elif model_status == "cancelled":
                    final_status = "cancelled"
                    break
        except asyncio.CancelledError:
            final_status = "cancelled"
            raise
        except BaseException:
            final_status = "failed"
            raise
        finally:
            # A run must never be left in 'running': finalize even when the loop
            # above aborts, so resumes and the runs table stay trustworthy.
            try:
                self.store.mark_run_complete(run_id, status=final_status)
            except Exception as exc:
                await self._notify(callback, f"Could not mark run {run_id} as {final_status}: {exc}")
            await self._emit(
                event_callback,
                {"type": "run_completed", "run_id": run_id, "status": final_status},
            )
        try:
            if self.store.sync_backend():
                await self._notify(callback, "Synced run database to hosted libSQL.")
        except Exception as exc:
            await self._notify(callback, f"Hosted libSQL sync failed: {exc}")
        return RunSummary(
            run_id=run_id,
            selected_row_count=len(selected_rows),
            model_count=len(config.models),
            status=final_status,
        )
