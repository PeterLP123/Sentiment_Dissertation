"""Results, leaderboard, comparison, metrics, and export feature logic for the TUI."""

from __future__ import annotations

import json
import subprocess
from contextlib import suppress
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual.widgets import DataTable, Select, Static

from .comparison import ComparisonResult, ModelTarget, compare_models
from .constants import ALLOWED_LABELS
from .exporter import export_run
from .storage import BenchmarkStore
from .tui_base import AppMixin
from .tui_format import (
    _LEADERBOARD_HELP_BASE,
    _LEADERBOARD_SORT_COLUMNS,
    _RUNS_HELP_BASE,
    _RUNS_SORT_COLUMNS,
    _accuracy_text,
    _cost_text,
    _count_text,
    _latency_text,
    _num,
    _status_text,
    _tokens_text,
)


class ResultsMixin(AppMixin):
    def _refresh_runs_table(self) -> None:
        try:
            table = self.query_one("#runs-table", DataTable)
        except Exception:
            return
        table.clear()
        try:
            runs = list(BenchmarkStore(self.db_path).list_runs())
        except Exception:
            runs = []
        if self._runs_sort is not None:
            index, ascending = self._runs_sort
            meta = _RUNS_SORT_COLUMNS.get(index)
            if meta is not None:
                with suppress(Exception):
                    runs.sort(key=meta[2], reverse=not ascending)
        self._update_sort_help("runs-help", _RUNS_HELP_BASE, self._runs_sort, _RUNS_SORT_COLUMNS)
        for run in runs:
            models = json.loads(run["models_json"]) if run["models_json"] else []
            preview = ", ".join(models[:3])
            if len(models) > 3:
                preview += f" (+{len(models) - 3})"
            created = (run["created_at"] or "")[:19]
            machine = run["machine_label"] or run["machine_id"] or "-"
            table.add_row(
                str(run["id"]),
                created,
                run["mode"],
                _status_text(run["status"]),
                machine,
                preview,
                key=str(run["id"]),
            )
        if not runs:
            table.add_row(
                Text("No runs yet — configure and start one on the Run tab (press 4).", style="grey58"),
                "", "", "", "", "",
                key="__placeholder__",
            )
        self._refresh_compare_targets(runs)
        self._refresh_leaderboard()

    def _leaderboard_rows(self, scope: str) -> list[dict]:
        try:
            metric_rows = BenchmarkStore(self.db_path).fetch_all_metrics()
        except Exception:
            return []
        best: dict[str, dict] = {}
        counts: dict[str, int] = {}
        for row in metric_rows:
            if row["scope"] != scope:
                continue
            model_id = row["model_id"]
            try:
                metric = json.loads(row["metrics_json"])
            except (ValueError, TypeError):
                continue
            accuracy = float(metric.get("accuracy", 0.0))
            counts[model_id] = counts.get(model_id, 0) + 1
            current = best.get(model_id)
            if current is None or accuracy > current["accuracy"]:
                best[model_id] = {
                    "model_id": model_id,
                    "accuracy": accuracy,
                    "macro_f1": float(metric.get("macro_f1", 0.0)),
                    "run_id": int(row["run_id"]),
                    "row_count": int(metric.get("row_count", 0)),
                }
        rows = []
        for model_id, data in best.items():
            entry = dict(data)
            entry["runs"] = counts.get(model_id, 0)
            rows.append(entry)
        rows.sort(key=lambda item: (-item["accuracy"], item["model_id"]))
        return rows

    def _refresh_leaderboard(self) -> None:
        try:
            table = self.query_one("#leaderboard-table", DataTable)
        except Exception:
            return
        try:
            scope = str(self.query_one("#leaderboard-scope", Select).value)
        except Exception:
            scope = "primary"
        table.clear()
        self._leaderboard_best_run = {}
        rows = list(self._leaderboard_rows(scope))
        # Best accuracy drives the bold highlight; compute it before any re-sort.
        best_accuracy = max((data["accuracy"] for data in rows), default=None)
        if self._leaderboard_sort is not None:
            index, ascending = self._leaderboard_sort
            meta = _LEADERBOARD_SORT_COLUMNS.get(index)
            if meta is not None:
                sort_field = meta[2]
                with suppress(Exception):
                    rows.sort(key=lambda data: data[sort_field], reverse=not ascending)
        self._update_sort_help("leaderboard-help", _LEADERBOARD_HELP_BASE, self._leaderboard_sort, _LEADERBOARD_SORT_COLUMNS)
        for data in rows:
            self._leaderboard_best_run[data["model_id"]] = data["run_id"]
            is_best = best_accuracy is not None and data["accuracy"] >= best_accuracy
            table.add_row(
                Text(data["model_id"], style="bold") if is_best else data["model_id"],
                _accuracy_text(data["accuracy"], bold=is_best),
                _accuracy_text(data["macro_f1"]),
                _num(data["run_id"]),
                _num(data["runs"]),
                _num(data["row_count"]),
                key=data["model_id"],
            )
        if not rows:
            table.add_row(
                Text("No scored runs yet — metrics appear here once a run completes.", style="grey58"),
                "", "", "", "", "",
                key="__placeholder__",
            )

    def _compare_targets_from_runs(self, runs) -> list[tuple[str, str]]:
        options: list[tuple[str, str]] = []
        for run in runs:
            try:
                models = json.loads(run["models_json"]) if run["models_json"] else []
            except (ValueError, TypeError):
                models = []
            for model_id in models:
                options.append((f"run {run['id']} · {model_id}", f"{run['id']}|{model_id}"))
        return options

    def _refresh_compare_targets(self, runs=None) -> None:
        if runs is None:
            try:
                runs = BenchmarkStore(self.db_path).list_runs()
            except Exception:
                runs = []
        options = self._compare_targets_from_runs(runs)
        valid = {value for _, value in options}
        for select_id in ("#compare-a", "#compare-b"):
            try:
                select = self.query_one(select_id, Select)
            except Exception:
                continue
            current = select.value
            select.set_options(options)
            if isinstance(current, str) and current in valid:
                select.value = current

    def _set_compare_status(self, text: str) -> None:
        with suppress(Exception):
            self.query_one("#compare-result", Static).update(text)

    @staticmethod
    def _parse_compare_target(value: str) -> ModelTarget:
        run_str, model_id = value.split("|", 1)
        return ModelTarget(run_id=int(run_str), model_id=model_id)

    def _compare_models_action(self) -> None:
        try:
            a_value = self.query_one("#compare-a", Select).value
            b_value = self.query_one("#compare-b", Select).value
            scope = str(self.query_one("#compare-scope", Select).value)
        except Exception as exc:
            self._notify_error(f"Compare setup error: {exc}", title="Cannot compare")
            return
        if not isinstance(a_value, str) or not isinstance(b_value, str):
            self._notify_error("Pick a model for both A and B.", title="Cannot compare")
            return
        if a_value == b_value:
            self._notify_error("Pick two different run/model targets.", title="Cannot compare")
            return
        target_a = self._parse_compare_target(str(a_value))
        target_b = self._parse_compare_target(str(b_value))
        self._set_compare_status("Comparing... (bootstrapping confidence intervals)")
        self._set_busy("compare-run", True, label="Comparing…")
        self.run_worker(
            lambda: self._run_comparison_blocking(target_a, target_b, scope),
            thread=True,
            group="compare",
            exclusive=False,
        )

    def _run_comparison_blocking(self, target_a: ModelTarget, target_b: ModelTarget, scope: str) -> None:
        try:
            store = BenchmarkStore(self.db_path)
            result = compare_models(store, target_a, target_b, scope=scope)
        except Exception as exc:
            self.call_from_thread(self._notify_error, f"Comparison failed: {exc}", title="Compare failed")
            self.call_from_thread(self._set_compare_status, f"Comparison failed: {exc}")
            self.call_from_thread(self._set_busy, "compare-run", False, label="Comparing…")
            return
        self.call_from_thread(self._render_comparison, result)

    def _render_comparison(self, result: ComparisonResult) -> None:
        self._set_busy("compare-run", False, label="Comparing…")
        if result.n_paired == 0:
            self._set_compare_status(
                "No shared rows between these targets — they were evaluated on different rows, so they cannot be paired."
            )
            return
        verdict = (
            "SIGNIFICANT difference (p < 0.05)"
            if result.is_significant()
            else "no significant difference (p >= 0.05)"
        )
        lines = [
            f"Metric: {result.metric} | scope: {result.scope} | paired rows: {result.n_paired}",
            f"A  {result.target_a.label()}: {result.point_a:.4f}  "
            f"CI [{float(result.ci_a['lower']):.4f}, {float(result.ci_a['upper']):.4f}]",
            f"B  {result.target_b.label()}: {result.point_b:.4f}  "
            f"CI [{float(result.ci_b['lower']):.4f}, {float(result.ci_b['upper']):.4f}]",
            f"McNemar: statistic={float(result.mcnemar['statistic']):.4f}, "
            f"p={float(result.mcnemar['p_value']):.4g} ({result.mcnemar.get('method')})",
            f"Discordant pairs: {result.mcnemar.get('n_discordant')}",
            f"Verdict: {verdict}",
        ]
        self._set_compare_status("\n".join(lines))

    def _load_metrics_for(self, run_id: int) -> None:
        self._active_run_id = run_id
        self._metric_rows = {}
        self._active_metric_model = None
        self._active_metric_scope = "all"
        metrics_table = self.query_one("#metrics-table", DataTable)
        metrics_table.clear()
        self.query_one("#perclass-table", DataTable).clear()
        self.query_one("#confusion-table", DataTable).clear()
        self.query_one("#misclassified-table", DataTable).clear()
        self._misclassification_rows = {}
        self.query_one("#misclassified-detail", Static).update(
            "Select a misclassified row to inspect the sentence and raw model output."
        )
        try:
            store = BenchmarkStore(self.db_path)
            metric_rows = store.fetch_metrics(run_id)
            cost_by_model = store.run_cost_by_model(run_id)
        except Exception as exc:
            self._notify_error(f"Could not load metrics for run {run_id}: {exc}", title="Load failed")
            return

        parsed = [(row["model_id"], row["scope"], json.loads(row["metrics_json"])) for row in metric_rows]
        # Highlight the strongest primary-scope model so comparisons are obvious.
        best_primary = max(
            (metric["accuracy"] for _, scope, metric in parsed if scope == "primary"),
            default=None,
        )
        # Lead with primary scope, then best accuracy first, so the leaderboard reads top-down.
        scope_order = {"primary": 0, "all": 1}
        parsed.sort(key=lambda item: (scope_order.get(item[1], 2), -item[2]["accuracy"], item[0]))

        for model_id, scope, metric in parsed:
            row_key = f"{model_id}|{scope}"
            self._metric_rows[row_key] = metric
            is_best = scope == "primary" and best_primary is not None and metric["accuracy"] >= best_primary
            metrics_table.add_row(
                Text(model_id, style="bold") if is_best else model_id,
                scope,
                _num(metric["row_count"]),
                _accuracy_text(metric["accuracy"], bold=is_best),
                _accuracy_text(metric["macro_f1"]),
                _latency_text(metric.get("mean_latency_ms")),
                _tokens_text(metric.get("total_tokens")),
                _cost_text(cost_by_model.get(model_id)),
                _count_text(metric["invalid_output_count"]),
                _count_text(metric["api_error_count"]),
                key=row_key,
            )
        if not parsed:
            metrics_table.add_row(
                Text("No metrics for this run — it may still be running or failed before scoring.", style="grey58"),
                "", "", "", "", "", "", "", "", "",
                key="__placeholder__",
            )
            self._set_monitor(
                f"No metrics found for run {run_id}. The run may still be running or may have failed before metrics were saved."
            )
        else:
            self._set_monitor(f"Loaded {len(parsed)} metric rows for run {run_id}.")
            self._refresh_misclassifications()

    def _show_per_class(self, metric: dict) -> None:
        table = self.query_one("#perclass-table", DataTable)
        table.clear()
        per_class = metric.get("per_class") or {}
        for label, scores in per_class.items():
            table.add_row(
                str(label),
                _num(f"{float(scores.get('precision', 0.0)):.4f}"),
                _num(f"{float(scores.get('recall', 0.0)):.4f}"),
                _num(f"{float(scores.get('f1', 0.0)):.4f}"),
                _num(int(float(scores.get('support', 0.0)))),
            )

    def _show_confusion(self, metric: dict) -> None:
        table = self.query_one("#confusion-table", DataTable)
        table.clear()
        matrix = metric.get("confusion_matrix") or {}
        if not matrix:
            return
        pred_labels = (*ALLOWED_LABELS, "__invalid__", "__error__")
        for actual in ALLOWED_LABELS:
            counts = matrix.get(actual, {})
            cells: list[Text] = [Text(actual, style="bold")]
            for predicted in pred_labels:
                count = int(counts.get(predicted, 0) or 0)
                if predicted == actual:
                    # Diagonal = correct predictions.
                    style = "green" if count > 0 else "grey58"
                elif count > 0:
                    style = "red"
                else:
                    style = "grey58"
                cells.append(Text(str(count), style=style, justify="right"))
            table.add_row(*cells)

    def _refresh_misclassifications(self) -> None:
        table = self.query_one("#misclassified-table", DataTable)
        table.clear()
        self._misclassification_rows = {}
        if self._active_run_id is None:
            return
        try:
            rows = BenchmarkStore(self.db_path).fetch_misclassifications(
                self._active_run_id,
                model_id=self._active_metric_model,
                scope=self._active_metric_scope,
            )
        except Exception as exc:
            self._notify_error(f"Could not load misclassified rows: {exc}", title="Load failed")
            return
        for index, row in enumerate(rows):
            record = dict(row)
            row_key = f"{record['model_id']}|{record['row_number']}|{index}"
            self._misclassification_rows[row_key] = record
            predicted = record.get("normalized_label") or "__invalid__"
            sentence = str(record.get("sentence") or "").replace("\n", " ")
            if len(sentence) > 96:
                sentence = sentence[:93] + "..."
            table.add_row(
                str(record.get("row_number")),
                str(record.get("model_id")),
                Text(str(record.get("hidden_label")), style="green"),
                Text(str(predicted), style="red"),
                _status_text(str(record.get("status"))),
                sentence,
                key=row_key,
            )
        detail = self.query_one("#misclassified-detail", Static)
        model = self._active_metric_model or "all models"
        scope = self._active_metric_scope
        if rows:
            detail.update(
                f"Loaded {len(rows)} misclassified/failed row(s) for {escape(model)} ({escape(scope)} scope)."
            )
        else:
            detail.update(f"No misclassified or failed rows found for {escape(model)} ({escape(scope)} scope).")

    def _show_misclassification_detail(self, row: dict) -> None:
        raw_content = str(row.get("raw_content") or "")
        if len(raw_content) > 1200:
            raw_content = raw_content[:1200] + "\n..."
        error = str(row.get("error") or "")
        if len(error) > 500:
            error = error[:500] + "\n..."
        detail_lines = [
            f"Row: {row.get('row_number')} | Model: {row.get('model_id')}",
            f"Actual: {row.get('hidden_label')} | Predicted: {row.get('normalized_label') or '__invalid__'}",
            f"Status: {row.get('status')} | Parse: {row.get('parse_status')} | Latency: {row.get('latency_ms') or '-'} ms",
            "Conflicting duplicate: "
            f"{'yes' if row.get('has_conflicting_duplicate') else 'no'} | "
            f"Duplicate group: {row.get('duplicate_group_size')}",
            "",
            "Sentence:",
            str(row.get("sentence") or ""),
            "",
            "Raw model output:",
            raw_content or "(empty)",
        ]
        if error:
            detail_lines.extend(["", "Error:", error])
        self.query_one("#misclassified-detail", Static).update(escape("\n".join(detail_lines)))

    def _export_run(self) -> None:
        if self._active_run_id is None:
            self._notify_error("Select a run in the table above before exporting.", title="No run selected")
            return
        self._start_export_worker(self._active_run_id, open_figures=False, button_id="export-run", label="Exporting…")

    def _start_export_worker(self, run_id: int, *, open_figures: bool, button_id: str, label: str) -> None:
        # Export (and especially matplotlib figure rendering) is slow enough to
        # freeze the UI, so run it off the event loop with a busy button.
        self._set_busy(button_id, True, label=label)
        self.run_worker(
            lambda: self._export_blocking(run_id, open_figures, button_id, label),
            thread=True,
            group="export",
            exclusive=True,
        )

    def _export_blocking(self, run_id: int, open_figures: bool, button_id: str, label: str) -> None:
        try:
            paths = list(export_run(self.db_path, run_id))
        except Exception as exc:
            self.call_from_thread(self._finish_export_error, exc, run_id, button_id, label)
            return
        self.call_from_thread(self._finish_export, paths, run_id, open_figures, button_id, label)

    def _finish_export_error(self, exc: Exception, run_id: int, button_id: str, label: str) -> None:
        self._set_busy(button_id, False, label=label)
        self._notify_error(f"Could not export run {run_id}: {exc}", title="Export failed")

    def _finish_export(self, paths: list[Path], run_id: int, open_figures: bool, button_id: str, label: str) -> None:
        self._set_busy(button_id, False, label=label)
        if open_figures:
            figures = [path for path in paths if path.suffix == ".png"]
            if not figures:
                self._notify_error(
                    "No figures were generated. Install plotting support with: pip install '.[figures]'",
                    title="No figures",
                )
                return
            figures_dir = figures[0].parent
            self._set_monitor(f"Generated {len(figures)} figure(s) for run {run_id} in {figures_dir}")
            if self._open_path(figures_dir):
                self._notify_info(
                    f"Opened {len(figures)} figure(s) for run {run_id} in your viewer.",
                    title="Figures",
                )
            return
        figure_count = sum(1 for path in paths if path.suffix == ".png")
        figure_note = f", incl. {figure_count} figure(s)" if figure_count else " (install '.[figures]' for plots)"
        self._notify_info(
            f"Exported run {run_id} ({len(paths)} files{figure_note}).",
            title="Export complete",
        )
        self._set_monitor(f"Exported run {run_id}:\n" + "\n".join(str(path) for path in paths))

    def _open_path(self, path: Path) -> bool:
        """Open a file or folder in the OS file manager / viewer. Returns success."""
        import shutil
        import sys

        if sys.platform == "darwin":
            opener = "open"
        elif sys.platform.startswith("win"):
            opener = "explorer"
        else:
            opener = "xdg-open"
        if shutil.which(opener) is None:
            self._notify_error(f"Cannot open {path}: {opener} is not on PATH.", title="Open failed")
            return False
        try:
            subprocess.Popen([opener, str(path)])
            return True
        except OSError as exc:
            self._notify_error(f"Could not open {path}: {exc}", title="Open failed")
            return False

    def _view_figures(self) -> None:
        if self._active_run_id is None:
            self._notify_error("Select a run in the table above before viewing figures.", title="No run selected")
            return
        self._start_export_worker(self._active_run_id, open_figures=True, button_id="view-figures", label="Generating…")

    def _open_exports(self) -> None:
        folder = Path("results/exports")
        folder.mkdir(parents=True, exist_ok=True)
        if self._open_path(folder):
            self._notify_info(f"Opened {folder} in your file manager.")
