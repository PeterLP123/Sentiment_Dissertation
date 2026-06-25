"""Experiment-queue and sweep feature logic for the TUI."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from contextlib import suppress
from dataclasses import replace
from threading import Event as ThreadEvent

from textual.widgets import Button, DataTable, Input, Select, Static

from .dataset import compute_stats, load_dataset
from .models import RunConfig
from .prompts import make_prompt
from .tui_base import _CONFIRM_THRESHOLD, AppMixin
from .tui_format import _status_text
from .tui_screens import ConfirmScreen, QueueResumeScreen

logger = logging.getLogger(__name__)


class QueueMixin(AppMixin):
    def _maybe_prompt_queue_resume(self) -> None:
        pending = sum(1 for item in self._experiment_queue if not self._queue_item_done(item))
        # Nothing to resume if the saved queue is empty or every experiment already ran.
        if pending == 0:
            return
        message = (
            f"Found a saved experiment queue with {len(self._experiment_queue)} experiment(s) "
            f"({pending} still pending).\n\nResume it, or start a new (empty) queue?"
        )
        self.push_screen(QueueResumeScreen(message), callback=self._handle_queue_resume)

    def _handle_queue_resume(self, resume: bool) -> None:
        if resume:
            self._set_monitor(f"Resumed saved experiment queue ({len(self._experiment_queue)} experiment(s)).")
            return
        self._experiment_queue = []
        self._render_queue_table()
        self._save_queue()
        self._set_monitor("Started a new (empty) experiment queue.")

    @staticmethod
    def _queue_item_done(item: dict) -> bool:
        return str(item.get("status", "")).startswith("done")

    def _rows_per_model_for(self, item: dict) -> int:
        if item.get("mode") == "pilot":
            return int(item.get("sample_per_class", 30)) * 3
        return compute_stats(load_dataset(self.dataset_path)).row_count

    def _queue_item_from_config(self, config: RunConfig) -> dict:
        self._queue_uid_counter += 1
        return {
            "uid": f"q{self._queue_uid_counter}",
            "status": "queued",
            "models": list(config.models),
            "model_names": {model_id: self._model_names.get(model_id, "") for model_id in config.models},
            "provider": config.provider,
            "base_url": config.base_url,
            "mode": config.mode,
            "sample_per_class": config.sample_per_class,
            "seed": config.seed,
            "concurrency": config.concurrency,
            "temperature": config.temperature,
            "max_completion_tokens": config.max_completion_tokens,
            "ollama_think": config.ollama_think,
            "prompt": {
                "prompt_id": config.prompt.prompt_id,
                "system_prompt": config.prompt.system_prompt,
                "user_template": config.prompt.user_template,
                "output_mode": config.prompt.output_mode,
                "demonstrations": [list(pair) for pair in config.prompt.demonstrations],
                "few_shot_seed": config.prompt.few_shot_seed,
            },
        }

    def _config_from_queue_item(self, item: dict) -> RunConfig:
        prompt_data = item["prompt"]
        prompt = make_prompt(
            prompt_id=prompt_data["prompt_id"],
            system_prompt=prompt_data["system_prompt"],
            user_template=prompt_data["user_template"],
            output_mode=prompt_data["output_mode"],
            demonstrations=[tuple(pair) for pair in prompt_data.get("demonstrations", [])],
            few_shot_seed=prompt_data.get("few_shot_seed"),
        )
        return RunConfig(
            models=list(item["models"]),
            prompt=prompt,
            mode=item["mode"],  # type: ignore[arg-type]
            dataset_path=str(self.dataset_path),
            db_path=str(self.db_path),
            base_url=item["base_url"],
            provider=item["provider"],
            sample_per_class=int(item["sample_per_class"]),
            seed=int(item["seed"]),
            concurrency=int(item["concurrency"]),
            temperature=float(item["temperature"]),
            max_completion_tokens=int(item["max_completion_tokens"]),
            ollama_think=item.get("ollama_think"),
        )

    def _add_current_to_queue(self) -> None:
        if self._queue_running:
            self._notify_error("Wait for the queue to finish before adding experiments.", title="Queue running")
            return
        if not self.selected_models:
            self._notify_error("Add at least one model before queueing an experiment.", title="Cannot queue")
            return
        if not self._settings_valid():
            self._notify_error("Fix the highlighted run settings before queueing.", title="Cannot queue")
            return
        try:
            config = self._build_run_config_from_ui()
        except Exception as exc:
            self._notify_error(f"Queue setup error: {exc}", title="Cannot queue")
            return
        self._experiment_queue.append(self._queue_item_from_config(config))
        self._render_queue_table()
        self._save_queue()
        self._set_monitor(
            f"Queued experiment #{len(self._experiment_queue)}: {len(config.models)} model(s), "
            f"prompt {config.prompt.prompt_id}, mode {config.mode}."
        )
        self._notify_info(
            f"Added experiment to queue ({len(self._experiment_queue)} total).",
            title="Queued",
        )

    @staticmethod
    def _parse_float_list(raw: str) -> list[float]:
        values: list[float] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError as exc:
                raise ValueError(f"'{token}' is not a number.") from exc
        return values

    def _build_sweep_items(self, base: RunConfig, axis: str, raw_values: str) -> list[dict]:
        items: list[dict] = []
        if axis == "temperature":
            temps = self._parse_float_list(raw_values)
            if not temps:
                raise ValueError("Enter one or more temperatures, e.g. 0,0.3,0.7.")
            for temp in temps:
                if not 0.0 <= temp <= 2.0:
                    raise ValueError(f"Temperature {temp} is outside the 0.0-2.0 range.")
                items.append(self._queue_item_from_config(replace(base, temperature=temp)))
        elif axis == "prompt":
            ids = [pid.strip() for pid in raw_values.split(",") if pid.strip()] or list(self.prompts)
            unknown = [pid for pid in ids if pid not in self.prompts]
            if unknown:
                raise ValueError(f"Unknown prompt preset(s): {', '.join(unknown)}.")
            for pid in ids:
                items.append(self._queue_item_from_config(replace(base, prompt=self.prompts[pid])))
        elif axis == "model":
            for model_id in base.models:
                items.append(self._queue_item_from_config(replace(base, models=[model_id])))
        else:
            raise ValueError(f"Unknown sweep axis: {axis}")
        return items

    def _add_sweep_to_queue(self) -> None:
        if self._queue_running:
            self._notify_error("Wait for the queue to finish before adding experiments.", title="Queue running")
            return
        if not self.selected_models:
            self._notify_error("Add at least one model before building a sweep.", title="Cannot sweep")
            return
        if not self._settings_valid():
            self._notify_error("Fix the highlighted run settings before building a sweep.", title="Cannot sweep")
            return
        try:
            axis = str(self.query_one("#sweep-axis", Select).value)
            raw_values = self.query_one("#sweep-values", Input).value.strip()
            base = self._build_run_config_from_ui()
            items = self._build_sweep_items(base, axis, raw_values)
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot sweep")
            return
        except Exception as exc:
            self._notify_error(f"Sweep setup error: {exc}", title="Cannot sweep")
            return
        if not items:
            self._notify_error("The sweep produced no experiments. Check the values.", title="Cannot sweep")
            return
        self._experiment_queue.extend(items)
        self._render_queue_table()
        self._save_queue()
        self._set_monitor(f"Added {len(items)} experiment(s) from a {axis} sweep.")
        self._notify_info(
            f"Queued {len(items)} sweep experiment(s) ({len(self._experiment_queue)} total).",
            title="Sweep queued",
        )

    def _remove_queue_item(self, uid: str) -> None:
        if self._queue_running:
            self._notify_error("Cannot edit the queue while it is running.", title="Queue busy")
            return
        before = len(self._experiment_queue)
        self._experiment_queue = [item for item in self._experiment_queue if str(item.get("uid")) != uid]
        if len(self._experiment_queue) != before:
            self._render_queue_table()
            self._save_queue()
            self._set_monitor("Removed an experiment from the queue.")

    def _clear_queue(self) -> None:
        if self._queue_running:
            self._notify_error("Cannot clear the queue while it is running.", title="Queue busy")
            return
        if not self._experiment_queue:
            return
        count = len(self._experiment_queue)
        self._experiment_queue = []
        self._render_queue_table()
        self._save_queue()
        self._set_monitor(f"Cleared {count} experiment(s) from the queue.")

    def _highlighted_queue_index(self) -> int | None:
        try:
            index = self.query_one("#queue-table", DataTable).cursor_row
        except Exception:
            return None
        if index is None or not (0 <= index < len(self._experiment_queue)):
            return None
        return index

    def _move_queue_item(self, delta: int) -> None:
        if self._queue_running:
            self._notify_error("Cannot reorder the queue while it is running.", title="Queue busy")
            return
        index = self._highlighted_queue_index()
        if index is None:
            self._notify_error("Highlight a queue row to move it.", title="No selection")
            return
        target = index + delta
        if not (0 <= target < len(self._experiment_queue)):
            return
        queue = self._experiment_queue
        queue[index], queue[target] = queue[target], queue[index]
        self._render_queue_table()
        self._save_queue()
        with suppress(Exception):
            self.query_one("#queue-table", DataTable).move_cursor(row=target)
        self._set_monitor(f"Moved experiment to position {target + 1}.")

    def _clone_queue_item(self) -> None:
        if self._queue_running:
            self._notify_error("Cannot edit the queue while it is running.", title="Queue busy")
            return
        index = self._highlighted_queue_index()
        if index is None:
            self._notify_error("Highlight a queue row to clone it.", title="No selection")
            return
        clone = copy.deepcopy(self._experiment_queue[index])
        self._queue_uid_counter += 1
        clone["uid"] = f"q{self._queue_uid_counter}"
        clone["status"] = "queued"
        self._experiment_queue.insert(index + 1, clone)
        self._render_queue_table()
        self._save_queue()
        self._set_monitor(f"Cloned experiment #{index + 1} into position {index + 2}.")

    def _render_queue_table(self) -> None:
        """Re-render the queue table and the UI that depends on queue state.

        Owns the full "queue changed" refresh: the queue summary, the status-bar
        depth, and (via ``_refresh_status_bar``) the stepper/button states. Callers
        that mutate the queue only need to call this plus ``_save_queue``.
        """
        try:
            table = self.query_one("#queue-table", DataTable)
        except Exception:
            return
        table.clear()
        for index, item in enumerate(self._experiment_queue, start=1):
            models = item.get("models", [])
            preview = ", ".join(models[:2])
            if len(models) > 2:
                preview += f" (+{len(models) - 2})"
            provider = "Ollama" if item.get("provider") == "ollama" else "OpenRouter"
            mode = item.get("mode", "pilot")
            sample = str(item.get("sample_per_class", "")) if mode == "pilot" else "full"
            prompt_id = (item.get("prompt") or {}).get("prompt_id", "-")
            table.add_row(
                str(index),
                provider,
                mode,
                preview or "(none)",
                prompt_id,
                sample,
                _status_text(str(item.get("status", "queued"))),
                key=str(item.get("uid")),
            )
        self._refresh_queue_summary()
        self._refresh_status_bar()

    def _refresh_queue_summary(self) -> None:
        try:
            summary = self.query_one("#queue-summary", Static)
        except Exception:
            return
        total = len(self._experiment_queue)
        if total == 0:
            summary.update("Queue is empty. Configure a run above and press Add to Queue.")
            return
        pending = sum(1 for item in self._experiment_queue if not self._queue_item_done(item))
        summary.update(f"{total} experiment(s) queued | {pending} pending | {total - pending} done.")

    def _save_queue(self) -> None:
        try:
            self._queue_path.parent.mkdir(parents=True, exist_ok=True)
            self._queue_path.write_text(json.dumps(self._experiment_queue, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_queue(self) -> None:
        try:
            data = json.loads(self._queue_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, list):
            return
        queue: list[dict] = []
        max_uid = 0
        for entry in data:
            if not isinstance(entry, dict) or "prompt" not in entry or "models" not in entry:
                continue
            status = str(entry.get("status", "queued"))
            # A "running" status means the TUI was closed mid-experiment; let it re-run.
            entry["status"] = "queued" if status == "running" else status
            uid = str(entry.get("uid") or "")
            if not uid:
                self._queue_uid_counter += 1
                uid = f"q{self._queue_uid_counter}"
                entry["uid"] = uid
            if uid.startswith("q") and uid[1:].isdigit():
                max_uid = max(max_uid, int(uid[1:]))
            queue.append(entry)
        self._experiment_queue = queue
        self._queue_uid_counter = max(self._queue_uid_counter, max_uid)

    async def _start_queue(self) -> None:
        if self._run_in_progress:
            self._notify_error("A run is already in progress.", title="Cannot start")
            return
        if self._baseline_in_progress:
            self._notify_error("Baselines are still running. Wait for them to finish.", title="Cannot start")
            return
        if self._confirmation_pending:
            self._notify_error("A run confirmation is already open.", title="Cannot start")
            return
        pending = [item for item in self._experiment_queue if not self._queue_item_done(item)]
        if not pending:
            self._notify_error("The experiment queue has no pending experiments.", title="Queue empty")
            return
        total_calls = sum(self._rows_per_model_for(item) * len(item.get("models", [])) for item in pending)
        if total_calls > _CONFIRM_THRESHOLD:
            self._confirmation_pending = True
            self._refresh_stepper()
            message = (
                f"You are about to run {len(pending)} queued experiment(s) totalling about "
                f"{total_calls} request(s). This may take time and incur cost. Continue?"
            )
            self.push_screen(ConfirmScreen(message), callback=self._handle_queue_confirmation)
            return
        self._begin_queue()

    def _handle_queue_confirmation(self, proceed: bool) -> None:
        self._confirmation_pending = False
        self._refresh_stepper()
        if not proceed:
            self._set_monitor("Queue run cancelled before it started.")
            return
        self._begin_queue()

    def _begin_queue(self) -> None:
        self._queue_cancel = False
        self._queue_running = True
        self._run_in_progress = True
        with suppress(Exception):
            self.query_one("#cancel-run", Button).disabled = False
        self._refresh_stepper()
        self._run_task = asyncio.create_task(self._run_queue())

    def _update_queue_progress(self, position: int, total: int, started: float) -> None:
        try:
            summary = self.query_one("#queue-summary", Static)
        except Exception:
            return
        minutes, seconds = divmod(int(time.monotonic() - started), 60)
        summary.update(f"Running experiment {position}/{total} · elapsed {minutes:02d}:{seconds:02d}")

    async def _run_queue(self) -> None:
        pending = [item for item in self._experiment_queue if not self._queue_item_done(item)]
        total = len(pending)
        started = time.monotonic()
        self._set_monitor(f"Running experiment queue: {total} experiment(s).")
        completed = 0
        cancelled = False
        try:
            for position, item in enumerate(pending, start=1):
                if self._queue_cancel:
                    item["status"] = "cancelled"
                    cancelled = True
                    self._render_queue_table()
                    self._save_queue()
                    break
                try:
                    config = self._config_from_queue_item(item)
                except Exception as exc:
                    item["status"] = "failed"
                    item["error"] = f"setup failed: {exc}"
                    logger.exception("Queue experiment %d/%d setup failed", position, total)
                    self._notify_error(f"Experiment {position} setup failed: {exc}", title="Queue")
                    self._render_queue_table()
                    self._save_queue()
                    continue
                item["status"] = "running"
                self._render_queue_table()
                self._save_queue()
                self._set_monitor(
                    f"Queue {position}/{total}: {len(config.models)} model(s), "
                    f"prompt {config.prompt.prompt_id}, mode {config.mode}."
                )
                self._update_queue_progress(position, total, started)
                self._cancel_event = ThreadEvent()
                try:
                    summary = await self._execute_config(config)
                    if summary.status == "completed":
                        item["status"] = f"done (run {summary.run_id})"
                        completed += 1
                    else:
                        item["status"] = f"{summary.status} (run {summary.run_id})"
                        cancelled = cancelled or summary.status == "cancelled"
                        if summary.status == "failed":
                            logger.warning("Queue experiment %d/%d failed (run %d)", position, total, summary.run_id)
                except Exception as exc:
                    item["status"] = "failed"
                    item["error"] = str(exc)
                    logger.exception("Queue experiment %d/%d failed", position, total)
                    self._notify_error(f"Experiment {position} failed: {exc}", title="Queue")
                finally:
                    self._cancel_event = None
                self._render_queue_table()
                self._save_queue()
                self._refresh_runs_table()
                self._refresh_results_help()
        finally:
            self._queue_running = False
            self._run_in_progress = False
            self._run_task = None
            self._queue_cancel = False
            with suppress(Exception):
                self.query_one("#cancel-run", Button).disabled = True
            self._render_queue_table()  # restores the queue summary and refreshes status bar + stepper
        outcome = "cancelled" if cancelled else "finished"
        self._set_monitor(f"Experiment queue {outcome}: {completed}/{total} experiment(s) completed.")
        self._notify_info(
            f"Queue {outcome}: {completed}/{total} experiment(s) completed.",
            title="Queue",
        )
        with suppress(Exception):
            self.bell()  # audible cue that an unattended queue has finished
