"""Baseline-run feature logic for the TUI."""

from __future__ import annotations

from textual.widgets import Checkbox, Input

from .baseline_runner import run_baselines
from .baselines import BASELINE_SPECS
from .tui_base import AppMixin


class BaselinesMixin(AppMixin):
    def _selected_baselines(self) -> list[str]:
        names: list[str] = []
        for name in BASELINE_SPECS:
            try:
                checkbox = self.query_one(f"#baseline-{name}", Checkbox)
            except Exception:
                continue
            if checkbox.value and not checkbox.disabled:
                names.append(name)
        return names

    def _start_baselines(self) -> None:
        if self._run_in_progress or self._baseline_in_progress:
            self._notify_error("A run is already in progress.", title="Busy")
            return
        names = self._selected_baselines()
        if not names:
            self._notify_error("Select at least one available baseline.", title="No baselines")
            return
        try:
            mode = self.run_mode
            seed = int(self.query_one("#seed", Input).value)
            sample_per_class = int(self.query_one("#sample-per-class", Input).value)
        except Exception as exc:
            self._notify_error(f"Baseline setup error: {exc}", title="Cannot start")
            return
        if mode not in {"pilot", "full"}:
            self._notify_error("Mode must be pilot or full", title="Cannot start")
            return
        self._baseline_in_progress = True
        self._refresh_stepper()
        self._refresh_status_bar()
        self._set_monitor(f"Starting baselines: {', '.join(names)}")
        self.run_worker(
            lambda: self._run_baselines_blocking(names, mode, sample_per_class, seed),
            thread=True,
            group="baselines",
            exclusive=False,
        )

    def _run_baselines_blocking(self, names: list[str], mode: str, sample_per_class: int, seed: int) -> None:
        try:
            summary = run_baselines(
                names,
                mode=mode,  # type: ignore[arg-type]
                dataset_path=str(self.dataset_path),
                db_path=str(self.db_path),
                sample_per_class=sample_per_class,
                seed=seed,
                callback=lambda message: self.call_from_thread(self._set_monitor, message),
            )
            self.call_from_thread(
                self._set_monitor,
                f"Baseline run {summary.run_id} completed: {summary.baseline_count} baseline(s), {summary.selected_row_count} rows.",
            )
        except Exception as exc:
            self.call_from_thread(self._notify_error, f"Baselines failed: {exc}")
        finally:
            self._baseline_in_progress = False
            self.call_from_thread(self._refresh_stepper)
            self.call_from_thread(self._refresh_status_bar)
            self.call_from_thread(self._refresh_runs_table)
            self.call_from_thread(self._refresh_results_help)
