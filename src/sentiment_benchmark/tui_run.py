"""Run configuration, session persistence, and benchmark-execution logic for the TUI."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from threading import Event as ThreadEvent

from textual.coordinate import Coordinate
from textual.widgets import Button, DataTable, Input, ProgressBar, RadioButton, Select, Static, TextArea
from textual.worker import WorkerFailed

from .dataset import compute_stats, load_dataset
from .models import RunConfig
from .news_source import NEWS_TIME_RANGES, NEWS_TOPICS
from .prompts import make_prompt
from .providers import make_llm_client, normalize_provider
from .runner import BenchmarkRunner
from .storage import BenchmarkStore
from .tui_base import _CONFIRM_THRESHOLD, _VALIDATED_INPUTS, AppMixin
from .tui_format import _count_text, _status_text
from .tui_screens import ConfirmScreen

# (cache key, widget id, caster, low, high) for run settings persisted across sessions.
_RUN_SETTING_FIELDS = (
    ("sample_per_class", "sample-per-class", int, 1, 10000),
    # Seed widget validator only enforces >= 0 (no upper bound), so don't reject large seeds on reload.
    ("seed", "seed", int, 0, 2**63 - 1),
    ("concurrency", "concurrency", int, 1, 64),
    ("temperature", "temperature", float, 0.0, 2.0),
    ("max_completion_tokens", "max-tokens", int, 16, 8192),
)


class RunMixin(AppMixin):
    def _load_session(self) -> None:
        try:
            raw = self._session_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        selected = data.get("selected_models")
        if isinstance(selected, list):
            self.selected_models = [str(value) for value in selected if isinstance(value, str)]
        names = data.get("model_names")
        if isinstance(names, dict):
            self._model_names = {
                str(key): str(value) for key, value in names.items() if isinstance(key, str) and isinstance(value, str)
            }
        provider = data.get("provider")
        if isinstance(provider, str):
            with suppress(ValueError):
                self.provider = normalize_provider(provider)
        self._normalize_selected_ollama_cloud_models()
        prompt_id = data.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id in self.prompts:
            self.prompt = self.prompts[prompt_id]
            self._default_prompt_id = prompt_id
        base_url = data.get("base_url")
        if isinstance(base_url, str) and base_url:
            self.base_url = base_url
        ollama_host = data.get("ollama_host")
        if isinstance(ollama_host, str) and ollama_host:
            self.ollama_host = ollama_host
        news_query = data.get("news_query")
        if isinstance(news_query, str) and news_query:
            self.news_query = news_query
        news_topic = data.get("news_topic")
        if isinstance(news_topic, str) and news_topic in NEWS_TOPICS:
            self.news_topic = news_topic
        news_time_range = data.get("news_time_range")
        if isinstance(news_time_range, str) and news_time_range in NEWS_TIME_RANGES:
            self.news_time_range = news_time_range
        news_max_results = data.get("news_max_results")
        if isinstance(news_max_results, int) and 1 <= news_max_results <= 20:
            self.news_max_results = news_max_results
        news_extract = data.get("news_extract")
        if isinstance(news_extract, bool):
            self.news_extract = news_extract
        disable_ollama_thinking = data.get("disable_ollama_thinking")
        if isinstance(disable_ollama_thinking, bool):
            self.disable_ollama_thinking = disable_ollama_thinking
        news_output_dir = data.get("news_output_dir")
        if isinstance(news_output_dir, str) and news_output_dir:
            self.news_output_dir = Path(news_output_dir)
        theme = data.get("theme")
        if isinstance(theme, str) and theme in self.available_themes:
            self._theme_name = theme
        run_mode = data.get("run_mode")
        if run_mode in {"pilot", "full"}:
            self._run_settings["run_mode"] = run_mode
            self.run_mode = run_mode
        for key, _widget_id, caster, low, high in _RUN_SETTING_FIELDS:
            value = data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high:
                self._run_settings[key] = caster(value)

    def _refresh_run_settings_cache(self) -> None:
        """Pull current run-setting widget values into the persisted cache (skips invalid/missing)."""
        self._run_settings["run_mode"] = self.run_mode
        for key, widget_id, caster, _low, _high in _RUN_SETTING_FIELDS:
            with suppress(Exception):
                self._run_settings[key] = caster(self.query_one(f"#{widget_id}", Input).value)

    def _apply_loaded_run_settings(self) -> None:
        """Write loaded run settings onto the Run-tab widgets (called once after mount).

        ``self.run_mode`` is already the source of truth (set by ``_load_session``),
        so the radio reflects it rather than re-deriving from the settings cache.
        """
        try:
            target_id = "mode-full" if self.run_mode == "full" else "mode-pilot"
            self.query_one(f"#{target_id}", RadioButton).value = True
        except Exception:
            pass
        for key, widget_id, _caster, _low, _high in _RUN_SETTING_FIELDS:
            with suppress(Exception):
                self.query_one(f"#{widget_id}", Input).value = str(self._run_settings[key])
        with suppress(Exception):
            self.query_one("#sample-per-class", Input).disabled = self.run_mode != "pilot"

    def _save_session(self) -> None:
        self._refresh_run_settings_cache()
        payload = {
            "selected_models": self.selected_models,
            "model_names": self._model_names,
            "prompt_id": self.prompt.prompt_id,
            "provider": self.provider,
            "base_url": self.base_url,
            "ollama_host": self.ollama_host,
            "news_query": self.news_query,
            "news_topic": self.news_topic,
            "news_time_range": self.news_time_range,
            "news_max_results": self.news_max_results,
            "news_extract": self.news_extract,
            "disable_ollama_thinking": self.disable_ollama_thinking,
            "news_output_dir": str(self.news_output_dir),
            "theme": self.theme,
            **self._run_settings,
        }
        try:
            self._session_path.parent.mkdir(parents=True, exist_ok=True)
            self._session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _refresh_prompt_preview(self) -> None:
        self.query_one("#prompt-preview", Static).update(
            "\n".join(
                [
                    f"Active prompt id: {self.prompt.prompt_id}",
                    f"Prompt hash: {self.prompt.prompt_hash}",
                    f"Output mode: {self.prompt.output_mode}",
                    "Parser rule: valid outputs must normalize exactly to positive, negative, or neutral.",
                ]
            )
        )
        self._refresh_status_bar()

    def _refresh_run_estimate(self) -> None:
        mode = self.run_mode
        try:
            sample_text = self.query_one("#sample-per-class", Input).value.strip() or "30"
        except Exception:
            sample_text = "30"
        model_count = len(self.selected_models)
        try:
            sample_per_class = int(sample_text)
            rows_per_model = (
                sample_per_class * 3
                if mode == "pilot"
                else compute_stats(load_dataset(self.dataset_path)).row_count
            )
            total_calls = rows_per_model * model_count
            if model_count == 0:
                detail = f"{mode} is configured, but no models are selected yet."
            else:
                detail = f"{mode} will send {rows_per_model} request(s) per model, {total_calls} total request(s)."
        except ValueError:
            detail = "Sample per class must be a whole number."
        try:
            run_estimate = self.query_one("#run-estimate", Static)
        except Exception:
            return
        run_estimate.update(
            "Run estimate\n"
            f"{detail}\n"
            "Primary accuracy excludes rows whose duplicate sentence has conflicting labels."
        )
        self._refresh_status_bar()

    def _refresh_results_help(self) -> None:
        try:
            runs = BenchmarkStore(self.db_path).list_runs()
        except Exception:
            runs = []
        if not runs:
            text = "No benchmark runs are stored yet. Start a pilot run, then return here with its run id."
        else:
            latest = runs[0]
            machine = latest["machine_label"] or latest["machine_id"] or "unknown machine"
            text = (
                f"Latest run id: {latest['id']} | mode: {latest['mode']} | "
                f"status: {latest['status']} | machine: {machine} | created: {latest['created_at']}"
            )
        self.query_one("#results-help", Static).update(text)

    def _confirm_message(
        self,
        mode: str,
        rows_per_model: int,
        models: list[str],
        max_completion_tokens: int | None = None,
        provider_name: str | None = None,
    ) -> str | None:
        model_count = len(models)
        total_calls = rows_per_model * model_count
        if mode != "full" and total_calls <= _CONFIRM_THRESHOLD:
            return None
        cost_text = self._cost_estimate_text(rows_per_model, models, max_completion_tokens)
        cost_line = f"\n\n{cost_text}" if cost_text else ""
        destination = provider_name or self._provider_title()
        return (
            f"You are about to send {total_calls} request(s) to {destination} "
            f"({rows_per_model} per model x {model_count} models, mode={mode})."
            f"{cost_line}\n\n"
            "This may take time and incur cost. Continue?"
        )

    def _cost_estimate_text(
        self,
        rows_per_model: int,
        models: list[str],
        max_completion_tokens: int | None,
    ) -> str | None:
        if self.provider == "ollama":
            return None
        if max_completion_tokens is None:
            return None
        pricing_by_model = {model.model_id: model.pricing for model in self._all_models}
        total_completion_cost = 0.0
        priced_models = 0
        for model_id in models:
            raw_price = pricing_by_model.get(model_id, {}).get("completion")
            if not isinstance(raw_price, (int, float, str)):
                continue
            try:
                completion_price = float(raw_price)
            except (TypeError, ValueError):
                continue
            total_completion_cost += rows_per_model * max_completion_tokens * completion_price
            priced_models += 1
        if priced_models == 0:
            return None
        suffix = (
            ""
            if priced_models == len(models)
            else f" Pricing unavailable for {len(models) - priced_models} model(s)."
        )
        return f"Estimated completion-token cost ceiling: ${total_completion_cost:.4f}.{suffix}"

    def _settings_valid(self) -> bool:
        try:
            for input_id in _VALIDATED_INPUTS:
                widget = self.query_one(f"#{input_id}", Input)
                if widget.disabled:
                    continue
                if not widget.is_valid:
                    return False
            return True
        except Exception:
            return True

    def _refresh_stepper(self) -> None:
        try:
            stepper = self.query_one("#run-stepper", Static)
        except Exception:
            return
        models_ok = len(self.selected_models) > 0
        prompt_ok = bool(self.prompt and self.prompt.prompt_hash)
        settings_ok = self._settings_valid()
        ready = models_ok and prompt_ok and settings_ok
        mark = lambda flag: "[OK]" if flag else "[  ]"  # noqa: E731
        parts = [
            f"{mark(models_ok)} 1 Models",
            f"{mark(prompt_ok)} 2 Prompt",
            f"{mark(settings_ok)} 3 Settings",
            f"{mark(ready)} 4 Start",
        ]
        stepper.update("  ".join(parts))
        busy = self._run_in_progress or self._baseline_in_progress or self._confirmation_pending
        with suppress(Exception):
            self.query_one("#start-run", Button).disabled = busy or not ready
        with suppress(Exception):
            self.query_one("#run-baselines", Button).disabled = busy
        has_pending = any(not self._queue_item_done(item) for item in self._experiment_queue)
        # Editing the queue is safe except while it is draining; starting it needs a fully idle app.
        queue_editable = not self._queue_running
        has_items = bool(self._experiment_queue)
        for button_id, enabled in (
            ("#add-to-queue", ready and queue_editable),
            ("#run-queue", has_pending and not busy),
            ("#clear-queue", has_items and queue_editable),
            ("#queue-move-up", has_items and queue_editable),
            ("#queue-move-down", has_items and queue_editable),
            ("#queue-clone", has_items and queue_editable),
        ):
            with suppress(Exception):
                self.query_one(button_id, Button).disabled = not enabled

    def _reset_progress(self, models: list[str], rows_per_model: int) -> None:
        self._progress = {
            model_id: {"done": 0, "errors": 0, "latency_total": 0.0, "latency_count": 0, "status": "queued"}
            for model_id in models
        }
        self._rows_per_model = rows_per_model
        try:
            table = self.query_one("#run-progress", DataTable)
        except Exception:
            return
        table.clear()
        for model_id in models:
            table.add_row(model_id, f"0/{rows_per_model}", _count_text(0), "-", _status_text("queued"), key=model_id)
        try:
            bar = self.query_one("#run-progress-bar", ProgressBar)
            bar.update(total=max(1, rows_per_model * len(models)), progress=0)
        except Exception:
            pass

    def _render_progress_row(self, model_id: str) -> None:
        state = self._progress.get(model_id)
        if state is None:
            return
        try:
            table = self.query_one("#run-progress", DataTable)
        except Exception:
            return
        try:
            row_index = table.get_row_index(model_id)
        except Exception:
            return
        avg = (
            f"{state['latency_total'] / state['latency_count']:.0f} ms"
            if state["latency_count"] > 0
            else "-"
        )
        cells = [
            model_id,
            f"{state['done']}/{self._rows_per_model}",
            _count_text(state["errors"]),
            avg,
            _status_text(state["status"]),
        ]
        for column_index, value in enumerate(cells):
            with suppress(Exception):
                table.update_cell_at(Coordinate(row_index, column_index), value)

    def _handle_run_event(self, event: dict) -> None:
        self._handle_run_event_inner(event)
        # Keep the global status-bar run indicator live from any tab.
        self._refresh_status_bar()

    def _handle_run_event_inner(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "run_started":
            self._reset_progress(list(event.get("models", [])), int(event.get("rows_per_model", 0)))
        elif event_type == "model_started":
            model_id = str(event.get("model_id", ""))
            state = self._progress.get(model_id)
            if state is not None:
                state["status"] = "running"
                self._render_progress_row(model_id)
        elif event_type == "row_completed":
            model_id = str(event.get("model_id", ""))
            state = self._progress.get(model_id)
            if state is None:
                return
            state["done"] += 1
            status = event.get("status")
            if status not in {"success", "skipped"}:
                state["errors"] += 1
            latency = event.get("latency_ms")
            if isinstance(latency, (int, float)):
                state["latency_total"] += float(latency)
                state["latency_count"] += 1
            self._render_progress_row(model_id)
            try:
                bar = self.query_one("#run-progress-bar", ProgressBar)
                bar.advance(1)
            except Exception:
                pass
        elif event_type == "model_completed":
            model_id = str(event.get("model_id", ""))
            state = self._progress.get(model_id)
            if state is not None:
                state["status"] = "done"
                self._render_progress_row(model_id)
        elif event_type == "run_completed":
            status = event.get("status", "completed")
            # During a queue drain the loop owns these flags across experiments;
            # only a standalone run resets them here.
            if not self._queue_running:
                with suppress(Exception):
                    self.query_one("#cancel-run", Button).disabled = True
                self._run_in_progress = False
                self._refresh_stepper()
            self._set_monitor(f"Run finished with status: {status}")

    def _save_prompt_from_ui(self) -> None:
        try:
            output_mode_value = self.query_one("#output-mode", Select).value
            if output_mode_value is Select.BLANK:
                raise ValueError("Choose an output mode")
            self.prompt = make_prompt(
                prompt_id="tui_custom",
                system_prompt=self.query_one("#system-prompt", TextArea).text,
                user_template=self.query_one("#user-template", TextArea).text,
                output_mode=str(output_mode_value),
            )
        except Exception as exc:
            self._notify_error(f"Prompt error: {exc}", title="Prompt invalid")
            return
        self._refresh_prompt_preview()
        self._save_session()
        self._notify_info(
            "Saved prompt for this session. Press 4 to configure and start the run.",
            title="Prompt saved",
        )

    async def _start_run(self) -> None:
        if self._run_in_progress:
            self._notify_error("A run is already in progress.", title="Cannot start")
            return
        if self._baseline_in_progress:
            self._notify_error("Baselines are still running. Wait for them to finish.", title="Cannot start")
            return
        if self._confirmation_pending:
            self._notify_error("A run confirmation is already open.", title="Cannot start")
            return
        if not self.selected_models:
            self._notify_error("Add at least one model before starting a run.", title="Cannot start")
            return
        if not self._settings_valid():
            self._notify_error("Fix the highlighted run settings before starting.", title="Cannot start")
            return
        try:
            config = self._build_run_config_from_ui()
        except Exception as exc:
            self._notify_error(f"Run setup error: {exc}", title="Cannot start")
            return

        rows_per_model = (
            config.sample_per_class * 3
            if config.mode == "pilot"
            else compute_stats(load_dataset(config.dataset_path)).row_count
        )
        confirm = self._confirm_message(config.mode, rows_per_model, config.models, config.max_completion_tokens)
        if confirm is not None:
            self._confirmation_pending = True
            self._refresh_stepper()
            self.push_screen(
                ConfirmScreen(confirm),
                callback=lambda proceed: self._handle_run_confirmation(proceed, config),
            )
            return

        self._begin_run(config)

    def _handle_run_confirmation(self, proceed: bool, config: RunConfig) -> None:
        self._confirmation_pending = False
        self._refresh_stepper()
        if not proceed:
            self._set_monitor("Run cancelled before it started.")
            return
        self._begin_run(config)

    def _begin_run(self, config: RunConfig) -> None:
        self._cancel_event = ThreadEvent()
        self._run_in_progress = True
        with suppress(Exception):
            self.query_one("#cancel-run", Button).disabled = False
        self._refresh_stepper()
        self._set_monitor("Starting benchmark run...")
        self._run_task = asyncio.create_task(self._run_benchmark(config))

    def _execute_config_in_thread(self, config: RunConfig):
        return asyncio.run(self._execute_config_async(config))

    async def _execute_config_async(self, config: RunConfig):
        store = BenchmarkStore(config.db_path)

        def update_monitor(message: str) -> None:
            self.call_from_thread(self._set_monitor, message)

        def handle_event(event: dict) -> None:
            self.call_from_thread(self._handle_run_event, event)

        async with make_llm_client(config.provider, base_url=config.base_url, ollama_host=config.base_url) as client:
            runner = BenchmarkRunner(client=client, store=store)
            return await runner.run(
                config,
                callback=update_monitor,
                event_callback=handle_event,
                cancel_event=self._cancel_event,
            )

    async def _execute_config(self, config: RunConfig):
        """Run one RunConfig off the TUI loop and return its RunSummary.

        Shared by the single Start Run path and the experiment queue. Benchmark
        execution performs synchronous dataset, SQLite, parsing, and metrics work
        between awaits, so keeping it in a thread worker prevents UI stalls.
        """
        worker = self.run_worker(
            lambda: self._execute_config_in_thread(config),
            thread=True,
            group="benchmarks",
            exclusive=False,
            exit_on_error=False,
        )
        try:
            return await worker.wait()
        except WorkerFailed as exc:
            raise exc.error from exc

    async def _run_benchmark(self, config: RunConfig) -> None:
        try:
            summary = await self._execute_config(config)
            self._set_monitor(
                f"Run {summary.run_id} {summary.status} ({summary.selected_row_count} rows x {summary.model_count} models)."
            )
        except Exception as exc:
            self._notify_error(f"Run failed: {exc}", title="Run failed")
        finally:
            self._run_in_progress = False
            self._cancel_event = None
            self._run_task = None
            with suppress(Exception):
                self.query_one("#cancel-run", Button).disabled = True
            self._refresh_stepper()
            self._refresh_status_bar()
            self._refresh_results_help()
            self._refresh_runs_table()

    def _cancel_run(self) -> None:
        if not self._run_in_progress:
            self._notify_error("No active run to cancel.", title="Nothing to cancel")
            return
        # Stop the current run, and if a queue is draining, stop it advancing too.
        self._queue_cancel = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        message = (
            "Cancel requested. The current experiment will stop and the queue will not advance."
            if self._queue_running
            else "Cancel requested. The run will stop after in-flight requests finish."
        )
        self._notify_info(message, title="Cancelling")

    def _build_run_config_from_ui(self) -> RunConfig:
        """Build a RunConfig from the current Run-tab widget state (raises on error)."""
        config = RunConfig(
            models=list(self.selected_models),
            prompt=self.prompt,
            mode=self.run_mode,  # type: ignore[arg-type]
            dataset_path=str(self.dataset_path),
            db_path=str(self.db_path),
            base_url=self._active_endpoint(),
            provider=self.provider,
            sample_per_class=int(self.query_one("#sample-per-class", Input).value),
            seed=int(self.query_one("#seed", Input).value),
            concurrency=int(self.query_one("#concurrency", Input).value),
            temperature=float(self.query_one("#temperature", Input).value),
            max_completion_tokens=int(self.query_one("#max-tokens", Input).value),
            ollama_think=False if self.provider == "ollama" and self.disable_ollama_thinking else None,
        )
        if config.mode not in {"pilot", "full"}:
            raise ValueError("Mode must be pilot or full")
        return config
