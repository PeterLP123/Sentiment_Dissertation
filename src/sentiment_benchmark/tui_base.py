"""Shared base for the SentimentBenchmarkApp feature mixins.

The TUI's behaviour lives across several feature mixins (baselines, news,
models, queue, results, run execution). Each mixin contributes methods to the
final :class:`~sentiment_benchmark.tui.SentimentBenchmarkApp`, which assembles
them alongside Textual's :class:`App`.

``AppMixin`` is a plain ``object`` at runtime, so the mixins only add methods and
never introduce a second ``App`` into the MRO. For type-checkers it is an ``App``
and additionally declares the shared instance state set in
``SentimentBenchmarkApp.__init__`` plus the cross-feature methods the mixins call
on ``self`` (their real implementations live on other mixins or the app itself).
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Event as ThreadEvent
from typing import TYPE_CHECKING, Any

from .models import ModelConfig, Provider

if TYPE_CHECKING:
    from textual.app import App as _AppBase
else:
    _AppBase = object


class AppMixin(_AppBase):
    if TYPE_CHECKING:
        # --- shared instance state (assigned in SentimentBenchmarkApp.__init__) ---
        dataset_path: Path
        db_path: Path
        provider: Provider
        base_url: str
        ollama_host: str
        news_query: str
        news_topic: str
        news_time_range: str
        news_max_results: int
        news_extract: bool
        news_output_dir: Path
        disable_ollama_thinking: bool
        selected_models: list[str]
        prompts: dict
        prompt: Any
        run_mode: str
        monitor_lines: list[str]
        news_lines: list[str]
        notifications: list[tuple[str, str]]
        _default_prompt_id: str
        _all_models: list[ModelConfig]
        _model_names: dict[str, str]
        _session_path: Path
        _queue_path: Path
        _active_run_id: int | None
        _metric_rows: dict[str, dict]
        _cancel_event: ThreadEvent | None
        _run_task: asyncio.Task | None
        _model_fetch_task: asyncio.Task | None
        _run_in_progress: bool
        _baseline_in_progress: bool
        _news_in_progress: bool
        _confirmation_pending: bool
        _progress: dict[str, dict]
        _rows_per_model: int
        _active_metric_model: str | None
        _active_metric_scope: str
        _misclassification_rows: dict[str, dict]
        _leaderboard_best_run: dict[str, int]
        _gpu_lines: list[str]
        _run_settings: dict
        _experiment_queue: list[dict]
        _queue_uid_counter: int
        _queue_running: bool
        _queue_cancel: bool
        _theme_name: str
        _button_labels: dict[str, object]
        _runs_sort: tuple[int, bool] | None
        _leaderboard_sort: tuple[int, bool] | None

        # --- cross-feature methods the mixins call on self (defined elsewhere) ---
        def _notify_error(self, message: str, *, title: str = ...) -> None: ...
        def _notify_info(self, message: str, *, title: str = ...) -> None: ...
        def _set_monitor(self, message: str) -> None: ...
        def _active_endpoint(self) -> str: ...
        def _provider_title(self) -> str: ...
        def _refresh_run_estimate(self) -> None: ...
        def _save_session(self) -> None: ...
        def _busy(
            self, button_id: str | None = ..., *, label: str | None = ..., spinner_widget: str | None = ...
        ) -> AbstractContextManager[None]: ...
        def _refresh_status_bar(self) -> None: ...
        def _refresh_stepper(self) -> None: ...
        def _refresh_runs_table(self) -> None: ...
        def _refresh_results_help(self) -> None: ...
