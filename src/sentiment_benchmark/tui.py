from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.coordinate import Coordinate
from textual.screen import ModalScreen
from textual.validation import Integer, Number, ValidationResult
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from .baseline_runner import run_baselines
from .baselines import BASELINE_SPECS, DEFAULT_BASELINES, BaselineSpec
from .constants import (
    DEFAULT_BASE_URL,
    DEFAULT_DATASET_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_PROMPTS_PATH,
)
from .dataset import compute_stats, load_dataset
from .env import load_env_file
from .exporter import export_run
from .models import ModelConfig, RunConfig
from .openrouter import OpenRouterClient
from .prompts import load_prompts, make_prompt
from .runner import BenchmarkRunner
from .storage import BenchmarkStore


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _baseline_available(spec: BaselineSpec) -> bool:
    return all(_module_available(dependency) for dependency in spec.requires)


def _baseline_install_hint(name: str) -> str:
    extra = "finbert" if name == "finbert" else "baselines"
    return f"pip install '.[{extra}]'"


_VALIDATED_INPUTS = ("sample-per-class", "seed", "concurrency", "temperature", "max-tokens")
_SESSION_PATH = Path("results/tui_session.json")
_SELECTED_MARK = "[x]"
_UNSELECTED_MARK = "[ ]"
_CONFIRM_THRESHOLD = 1000

# Status keyword -> colour, used to tint table cells so runs can be scanned at a glance.
_STATUS_COLORS = {
    "completed": "green",
    "done": "green",
    "success": "green",
    "running": "yellow",
    "queued": "grey58",
    "cancelled": "red",
    "failed": "red",
    "error": "red",
}


def _accuracy_text(value: float, *, bold: bool = False) -> Text:
    """Colour an accuracy/F1 score on a traffic-light scale."""
    if value >= 0.8:
        color = "green"
    elif value >= 0.6:
        color = "yellow"
    else:
        color = "red"
    return Text(f"{value:.4f}", style=f"bold {color}" if bold else color)


def _status_text(status: str) -> Text:
    """Colour a status label (first word drives the colour, e.g. 'done (acc ...)')."""
    key = status.split(" ", 1)[0].lower()
    return Text(status, style=_STATUS_COLORS.get(key, "white"))


def _count_text(value: int) -> Text:
    """Red when there is something to worry about (errors/invalids), dim otherwise."""
    return Text(str(value), style="red" if value > 0 else "grey58")


def _latency_text(value: float | None) -> Text:
    if not isinstance(value, (int, float)):
        return Text("-", style="grey58")
    return Text(f"{value:.0f} ms")


def _tokens_text(value: int | None) -> Text:
    if not isinstance(value, (int, float)) or value == 0:
        return Text("-", style="grey58")
    return Text(f"{int(value):,}")


def _cost_text(value: float | None) -> Text:
    if not isinstance(value, (int, float)):
        return Text("-", style="grey58")
    return Text(f"${value:.4f}", style="cyan")


class ConfirmScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-box {
        background: $surface;
        border: thick $accent;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    #confirm-message {
        margin-bottom: 1;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm-box"):
            yield Static(self._message, id="confirm-message")
            with Horizontal():
                yield Button("Cancel", id="confirm-cancel")
                yield Button("Start run", id="confirm-ok", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-ok")


class HelpScreen(ModalScreen[None]):
    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-box {
        background: $surface;
        border: thick $accent;
        padding: 1 2;
        width: 64;
        height: auto;
    }
    #help-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #help-body {
        margin-bottom: 1;
    }
    """

    BINDINGS = [("escape", "dismiss(None)", "Close"), ("?", "dismiss(None)", "Close")]

    _HELP_LINES = [
        "1-5    Switch tabs (Dashboard, Models, Prompt, Run, Results)",
        "r      Refresh the dashboard",
        "s      Start the benchmark run",
        "c      Cancel an in-progress run",
        "?      Toggle this help",
        "q      Quit",
    ]

    def compose(self) -> ComposeResult:
        with Container(id="help-box"):
            yield Static("Keyboard shortcuts", id="help-title")
            yield Static("\n".join(self._HELP_LINES), id="help-body")
            yield Button("Close", id="help-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss(None)


class SentimentBenchmarkApp(App):
    TITLE = "Sentiment Benchmark"
    SUB_TITLE = "OpenRouter LLM sentiment evaluation"
    CSS = """
    Screen {
        layout: vertical;
    }
    #status-bar {
        background: $boost;
        color: $text;
        padding: 0 2;
        height: 1;
    }
    TabPane {
        padding: 1 2;
    }
    Input, TextArea {
        margin-bottom: 0;
    }
    Button {
        margin-right: 1;
    }
    Select, RadioSet {
        margin-bottom: 1;
    }
    .panel {
        border: solid $accent;
        padding: 1;
        margin-bottom: 1;
    }
    .help {
        color: $text-muted;
        margin-bottom: 1;
    }
    .field-label {
        text-style: bold;
        margin-top: 1;
    }
    .section-title {
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }
    .validation-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    .validation-hint.-error {
        color: $error;
    }
    #dashboard, #prompt-preview, #run-estimate, #results-help, #selected-summary {
        border: solid $accent;
        padding: 1;
        margin-bottom: 1;
    }
    #model-search {
        margin-bottom: 1;
    }
    #selected-table {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }
    #model-table {
        height: auto;
        max-height: 20;
    }
    #run-stepper {
        background: $boost;
        padding: 0 1;
        margin-bottom: 1;
        height: 1;
    }
    #run-progress-bar {
        margin-bottom: 1;
    }
    #run-progress {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }
    #monitor {
        height: 12;
        margin-bottom: 1;
        border: solid $accent;
    }
    #runs-table {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }
    #metrics-table {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }
    #perclass-table {
        height: auto;
        max-height: 8;
        margin-bottom: 1;
    }
    #misclassified-table {
        height: auto;
        max-height: 12;
        margin-bottom: 1;
    }
    #misclassified-detail {
        border: solid $accent;
        padding: 1;
        min-height: 8;
        margin-bottom: 1;
    }
    #run-controls {
        height: auto;
        margin-bottom: 1;
    }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("?", "show_help", "Help"),
        ("1", "show_tab('dashboard-tab')", "Dashboard"),
        ("2", "show_tab('models-tab')", "Models"),
        ("3", "show_tab('prompt-tab')", "Prompt"),
        ("4", "show_tab('run-tab')", "Run"),
        ("5", "show_tab('results-tab')", "Results"),
        ("r", "refresh", "Refresh"),
        ("s", "start_run", "Start"),
        ("b", "run_baselines", "Baselines"),
        ("c", "cancel_run", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        load_env_file()
        self.dataset_path = DEFAULT_DATASET_PATH
        self.db_path = DEFAULT_DB_PATH
        self.base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
        self.selected_models: list[str] = []
        self.prompts = load_prompts(DEFAULT_PROMPTS_PATH)
        self._default_prompt_id = (
            "default_label_only" if "default_label_only" in self.prompts else next(iter(self.prompts))
        )
        self.prompt = self.prompts[self._default_prompt_id]
        self.run_mode: str = "pilot"
        self.monitor_lines: list[str] = []
        self._all_models: list[ModelConfig] = []
        self._model_names: dict[str, str] = {}
        self._session_path = _SESSION_PATH
        self._active_run_id: int | None = None
        self._metric_rows: dict[str, dict] = {}
        self._cancel_event: asyncio.Event | None = None
        self._run_task: asyncio.Task | None = None
        self._run_in_progress: bool = False
        self._baseline_in_progress: bool = False
        self._confirmation_pending: bool = False
        self._progress: dict[str, dict] = {}
        self._rows_per_model: int = 0
        self._active_metric_model: str | None = None
        self._active_metric_scope: str = "all"
        self._misclassification_rows: dict[str, dict] = {}
        self.notifications: list[tuple[str, str]] = []
        self._load_session()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="status-bar")
        with TabbedContent():
            with TabPane("Dashboard", id="dashboard-tab"):
                yield Static(
                    "Use this app to run OpenRouter models against the dissertation sentiment dataset. "
                    "Only Sentence text is sent to models; hidden Sentiment labels stay in the evaluator.",
                    classes="help",
                )
                yield Static(id="dashboard")
                yield Button("Refresh Dashboard", id="refresh-dashboard")
            with TabPane("Models", id="models-tab"):
                yield Static(
                    "Step 1: choose the OpenRouter models to test. Fetch models to browse available IDs, "
                    "press Enter on a row to toggle it, or type a model ID manually.",
                    classes="help",
                )
                yield Static("Manual model ID", classes="field-label")
                yield Static(
                    "Example: openai/gpt-4o-mini. Repeat Add Model for each model you want in the same benchmark run.",
                    classes="help",
                )
                yield Input(placeholder="openai/gpt-4o-mini", id="manual-model")
                yield Button("Add Model", id="add-model")
                yield Button("Fetch Models", id="fetch-models")
                yield Static("Selected models", classes="section-title")
                yield Static(id="selected-summary")
                yield Static(
                    "Press Enter on a row below to remove it from the run.",
                    classes="help",
                )
                yield DataTable(id="selected-table")
                yield Static("Fetched OpenRouter models", classes="section-title")
                yield Static(
                    "Type to filter. Press Enter on a row to toggle selection. [x] = currently selected.",
                    classes="help",
                )
                yield Input(placeholder="Search model id or name...", id="model-search")
                yield DataTable(id="model-table")
            with TabPane("Prompt", id="prompt-tab"):
                yield Static(
                    "Step 2: configure how the model is asked for sentiment. The default strict benchmark prompt "
                    "requests only positive, negative, or neutral.",
                    classes="help",
                )
                yield Static("Preset", classes="field-label")
                yield Static(
                    "Pick a preset from configs/default_prompts.toml. Use Save Prompt to keep custom edits as a tui_custom prompt.",
                    classes="help",
                )
                yield Select(
                    [(prompt_id, prompt_id) for prompt_id in self.prompts],
                    id="prompt-preset",
                    value=self._default_prompt_id,
                    allow_blank=False,
                )
                yield Static("System prompt", classes="field-label")
                yield Static(
                    "Instructions sent as the system message. Do not include labels, examples with answers, or dataset statistics here.",
                    classes="help",
                )
                yield TextArea(self.prompt.system_prompt, id="system-prompt")
                yield Static("User template", classes="field-label")
                yield Static(
                    "Template for each row. It must include {sentence}; the app fills that with only the sentence text.",
                    classes="help",
                )
                yield TextArea(self.prompt.user_template, id="user-template")
                yield Static("Output mode", classes="field-label")
                yield Static(
                    "Use Label only for comparable accuracy tests. Use Explanation only for qualitative review runs.",
                    classes="help",
                )
                yield Select(
                    [("Label only (recommended)", "label_only"), ("Explanation", "explanation")],
                    id="output-mode",
                    value=self.prompt.output_mode,
                    allow_blank=False,
                )
                yield Button("Save Prompt", id="save-prompt")
                yield Static(id="prompt-preview")
            with TabPane("Run", id="run-tab"):
                yield Static(id="run-stepper")
                yield Static(
                    "Step 3: run the benchmark. Pilot is the safe default: 30 rows from each class, 90 calls per selected model. "
                    "Full uses all 5,842 rows per model and may cost more.",
                    classes="help",
                )
                yield Static(id="run-estimate")
                yield Static("Run mode", classes="field-label")
                yield Static(
                    "Pilot is a small stratified test. Full processes every row in Data/data.csv.",
                    classes="help",
                )
                with RadioSet(id="run-mode"):
                    yield RadioButton("Pilot", value=True, id="mode-pilot")
                    yield RadioButton("Full", id="mode-full")
                yield Static("Sample per class", classes="field-label")
                yield Static(
                    "Pilot only. 30 means 30 positive, 30 negative, and 30 neutral rows.",
                    classes="help",
                )
                yield Input(
                    value="30",
                    placeholder="sample per class",
                    id="sample-per-class",
                    type="integer",
                    validators=[Integer(minimum=1, maximum=10000)],
                )
                yield Static(
                    "Whole number ≥ 1.",
                    id="hint-sample-per-class",
                    classes="validation-hint",
                )
                with Collapsible(
                    title="Advanced settings — seed, concurrency, temperature, tokens",
                    collapsed=True,
                    id="advanced-settings",
                ):
                    yield Static("Random seed", classes="field-label")
                    yield Static(
                        "Controls which rows are selected for pilot runs so experiments are reproducible.",
                        classes="help",
                    )
                    yield Input(
                        value="42",
                        placeholder="seed",
                        id="seed",
                        type="integer",
                        validators=[Integer(minimum=0)],
                    )
                    yield Static(
                        "Non-negative whole number.",
                        id="hint-seed",
                        classes="validation-hint",
                    )
                    yield Static("Concurrency", classes="field-label")
                    yield Static(
                        "Number of simultaneous API calls. Keep this at 1 unless you are comfortable with rate limits.",
                        classes="help",
                    )
                    yield Input(
                        value="1",
                        placeholder="concurrency",
                        id="concurrency",
                        type="integer",
                        validators=[Integer(minimum=1, maximum=64)],
                    )
                    yield Static(
                        "Whole number between 1 and 64.",
                        id="hint-concurrency",
                        classes="validation-hint",
                    )
                    yield Static("Temperature", classes="field-label")
                    yield Static("0 is recommended for deterministic classification.", classes="help")
                    yield Input(
                        value="0",
                        placeholder="temperature",
                        id="temperature",
                        type="number",
                        validators=[Number(minimum=0.0, maximum=2.0)],
                    )
                    yield Static(
                        "Number between 0.0 and 2.0.",
                        id="hint-temperature",
                        classes="validation-hint",
                    )
                    yield Static("Max completion tokens", classes="field-label")
                    yield Static(
                        "64 is recommended. Some reasoning-capable models reject tiny limits or spend them before emitting a label.",
                        classes="help",
                    )
                    yield Input(
                        value=str(DEFAULT_MAX_COMPLETION_TOKENS),
                        placeholder="max completion tokens",
                        id="max-tokens",
                        type="integer",
                        validators=[Integer(minimum=16, maximum=8192)],
                    )
                    yield Static(
                        "Whole number between 16 and 8192.",
                        id="hint-max-tokens",
                        classes="validation-hint",
                    )
                yield Static("Baselines", classes="section-title")
                yield Static(
                    "Non-LLM comparators evaluated on the same rows (uses the run mode, seed, and sample-per-class above). "
                    "Fitted baselines are scored out-of-fold; vader and finbert need optional dependencies.",
                    classes="help",
                )
                for _name, _spec in BASELINE_SPECS.items():
                    _available = _baseline_available(_spec)
                    _label = f"{_name} \u2014 {_spec.description}"
                    if not _available:
                        _label += f"  ({_baseline_install_hint(_name)})"
                    yield Checkbox(
                        _label,
                        value=_name in DEFAULT_BASELINES and _available,
                        id=f"baseline-{_name}",
                        disabled=not _available,
                    )
                with Horizontal(id="run-controls"):
                    yield Button("Start Run", id="start-run", variant="primary")
                    yield Button("Run Baselines", id="run-baselines", variant="success")
                    yield Button("Cancel Run", id="cancel-run", disabled=True, variant="error")
                yield Static("Progress", classes="section-title")
                yield ProgressBar(id="run-progress-bar", total=100, show_percentage=True, show_eta=True)
                yield DataTable(id="run-progress")
                yield Static("Live log", classes="section-title")
                yield RichLog(id="monitor", highlight=True, markup=False, wrap=True)
            with TabPane("Results", id="results-tab"):
                yield Static(
                    "Step 4: review completed runs. Pick a run from the list to load its metrics.",
                    classes="help",
                )
                yield Static(id="results-help")
                yield Static("Recent runs", classes="section-title")
                yield Static(
                    "Press Enter on a row to load metrics for that run.",
                    classes="help",
                )
                yield DataTable(id="runs-table")
                yield Button("Refresh Runs", id="refresh-runs")
                yield Button("Export Selected Run", id="export-run")
                yield Button("Open Exports Folder", id="open-exports")
                yield Static("Metrics", classes="section-title")
                yield Static(
                    "Press Enter on a metric row to see its per-class precision, recall, F1, and support.",
                    classes="help",
                )
                yield DataTable(id="metrics-table")
                yield Static("Per-class breakdown", classes="section-title")
                yield DataTable(id="perclass-table")
                yield Static("Misclassified and failed rows", classes="section-title")
                yield Static(
                    "Load a run to see mismatches. Select a metric row to filter by model/scope; press Enter on a row for details.",
                    classes="help",
                )
                yield DataTable(id="misclassified-table")
                yield Static(
                    "Select a misclassified row to inspect the sentence and raw model output.",
                    id="misclassified-detail",
                )
        yield Footer()

    def on_mount(self) -> None:
        model_table = self.query_one("#model-table", DataTable)
        model_table.add_columns("Sel", "Model ID", "Name", "Context")
        model_table.cursor_type = "row"
        selected_table = self.query_one("#selected-table", DataTable)
        selected_table.add_columns("Model ID", "Name")
        selected_table.cursor_type = "row"
        metrics = self.query_one("#metrics-table", DataTable)
        metrics.add_columns("Model", "Scope", "Rows", "Accuracy", "Macro F1", "Latency", "Tokens", "Cost", "Invalid", "Errors")
        metrics.cursor_type = "row"
        runs = self.query_one("#runs-table", DataTable)
        runs.add_columns("ID", "Created", "Mode", "Status", "Models")
        runs.cursor_type = "row"
        perclass = self.query_one("#perclass-table", DataTable)
        perclass.add_columns("Class", "Precision", "Recall", "F1", "Support")
        misclassified = self.query_one("#misclassified-table", DataTable)
        misclassified.add_columns("Row", "Model", "Actual", "Predicted", "Status", "Sentence")
        misclassified.cursor_type = "row"
        progress = self.query_one("#run-progress", DataTable)
        progress.add_columns("Model", "Done/Total", "Errors", "Avg latency", "Status")

        # Label the free-standing bordered panels so they read as titled cards.
        for panel_id, title in (
            ("#dashboard", "Dataset & environment"),
            ("#prompt-preview", "Active prompt"),
            ("#misclassified-detail", "Row detail"),
        ):
            try:
                self.query_one(panel_id, Static).border_title = title
            except Exception:
                pass

        self._refresh_dashboard()
        self._render_selected_table()
        self._render_model_table()
        self._refresh_prompt_preview()
        self._refresh_run_estimate()
        self._refresh_results_help()
        self._refresh_runs_table()
        self._refresh_status_bar()
        self._refresh_stepper()

    def _set_monitor(self, message: str) -> None:
        self.monitor_lines.append(message)
        self.monitor_lines = self.monitor_lines[-500:]
        try:
            log = self.query_one("#monitor", RichLog)
        except Exception:
            return
        should_follow = bool(log.is_vertical_scroll_end)
        log.write(message, scroll_end=should_follow)

    def _notify_error(self, message: str, *, title: str = "Error") -> None:
        self.notifications.append(("error", message))
        try:
            self.notify(message, title=title, severity="error")
        except Exception:
            pass

    def _notify_info(self, message: str, *, title: str = "") -> None:
        self.notifications.append(("information", message))
        try:
            self.notify(message, title=title, severity="information")
        except Exception:
            pass

    def action_show_tab(self, tab_id: str) -> None:
        try:
            self.query_one(TabbedContent).active = tab_id
        except Exception:
            pass

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_refresh(self) -> None:
        self._refresh_dashboard()
        self._refresh_runs_table()
        self._refresh_results_help()
        self._notify_info("Refreshed dashboard and runs.")

    async def action_start_run(self) -> None:
        await self._start_run()

    def action_cancel_run(self) -> None:
        self._cancel_run()

    def _refresh_status_bar(self) -> None:
        try:
            bar = self.query_one("#status-bar", Static)
        except Exception:
            return
        key_state = "OK" if os.getenv("OPENROUTER_API_KEY") else "missing"
        bar.update(
            f" API key: {key_state}  |  models: {len(self.selected_models)}  |  "
            f"prompt: {self.prompt.prompt_id}  |  mode: {self.run_mode} "
        )
        self._refresh_stepper()

    def _refresh_dashboard(self) -> None:
        rows = load_dataset(self.dataset_path)
        stats = compute_stats(rows)
        key_status = "present" if os.getenv("OPENROUTER_API_KEY") else "missing"
        self.query_one("#dashboard", Static).update(
            "\n".join(
                [
                    f"Dataset: {self.dataset_path}",
                    f"Rows: {stats.row_count}",
                    f"Labels: {stats.label_counts}",
                    f"Conflicting duplicate rows excluded from primary metrics: {stats.conflicting_duplicate_rows}",
                    f"Primary scoring rows: {stats.primary_row_count}",
                    f"Result DB: {self.db_path}",
                    f"OpenRouter API key: {key_status}",
                ]
            )
        )
        self._refresh_status_bar()

    def _render_selected_table(self) -> None:
        try:
            table = self.query_one("#selected-table", DataTable)
        except Exception:
            return
        table.clear()
        for model_id in self.selected_models:
            name = self._model_names.get(model_id) or "(manual)"
            table.add_row(model_id, name, key=model_id)
        self._refresh_selected_summary()
        self._refresh_status_bar()

    def _refresh_selected_summary(self) -> None:
        try:
            summary = self.query_one("#selected-summary", Static)
        except Exception:
            return
        count = len(self.selected_models)
        if count == 0:
            summary.update("No models selected yet.")
        else:
            summary.update(f"{count} model(s) selected.")

    def _render_model_table(self) -> None:
        try:
            table = self.query_one("#model-table", DataTable)
        except Exception:
            return
        table.clear()
        try:
            query = self.query_one("#model-search", Input).value.strip().lower()
        except Exception:
            query = ""
        selected = set(self.selected_models)
        for model in self._all_models:
            if query:
                haystack = f"{model.model_id} {model.name or ''}".lower()
                if query not in haystack:
                    continue
            mark = (
                Text(_SELECTED_MARK, style="bold green")
                if model.model_id in selected
                else Text(_UNSELECTED_MARK, style="grey58")
            )
            table.add_row(
                mark,
                model.model_id,
                model.name or "",
                str(model.context_length or ""),
                key=model.model_id,
            )

    def _toggle_model(self, model_id: str, name: str | None = None) -> None:
        if not model_id:
            return
        if model_id in self.selected_models:
            self.selected_models.remove(model_id)
        else:
            self.selected_models.append(model_id)
            if name:
                self._model_names[model_id] = name
        self._render_model_table()
        self._render_selected_table()
        self._refresh_run_estimate()
        self._save_session()

    def _select_model(self, model_id: str, name: str | None = None) -> None:
        if not model_id or model_id in self.selected_models:
            return
        self.selected_models.append(model_id)
        if name:
            self._model_names[model_id] = name
        self._render_model_table()
        self._render_selected_table()
        self._refresh_run_estimate()
        self._save_session()

    def _deselect_model(self, model_id: str) -> None:
        if model_id not in self.selected_models:
            return
        self.selected_models.remove(model_id)
        self._render_model_table()
        self._render_selected_table()
        self._refresh_run_estimate()
        self._save_session()

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
        prompt_id = data.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id in self.prompts:
            self.prompt = self.prompts[prompt_id]
            self._default_prompt_id = prompt_id
        base_url = data.get("base_url")
        if isinstance(base_url, str) and base_url:
            self.base_url = base_url

    def _save_session(self) -> None:
        payload = {
            "selected_models": self.selected_models,
            "model_names": self._model_names,
            "prompt_id": self.prompt.prompt_id,
            "base_url": self.base_url,
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
        self.query_one("#run-estimate", Static).update(
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
            text = (
                f"Latest run id: {latest['id']} | mode: {latest['mode']} | "
                f"status: {latest['status']} | created: {latest['created_at']}"
            )
        self.query_one("#results-help", Static).update(text)

    def _confirm_message(
        self,
        mode: str,
        rows_per_model: int,
        models: list[str],
        max_completion_tokens: int | None = None,
    ) -> str | None:
        model_count = len(models)
        total_calls = rows_per_model * model_count
        if mode != "full" and total_calls <= _CONFIRM_THRESHOLD:
            return None
        cost_text = self._cost_estimate_text(rows_per_model, models, max_completion_tokens)
        cost_line = f"\n\n{cost_text}" if cost_text else ""
        return (
            f"You are about to send {total_calls} request(s) to OpenRouter "
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
        try:
            self.query_one("#start-run", Button).disabled = busy or not ready
        except Exception:
            pass
        try:
            self.query_one("#run-baselines", Button).disabled = busy
        except Exception:
            pass

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
        except KeyError:
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
            try:
                table.update_cell_at(Coordinate(row_index, column_index), value)
            except Exception:
                pass

    def _handle_run_event(self, event: dict) -> None:
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
                accuracy = event.get("accuracy")
                state["status"] = (
                    f"done (acc {float(accuracy):.3f})" if isinstance(accuracy, (int, float)) else "done"
                )
                self._render_progress_row(model_id)
        elif event_type == "run_completed":
            status = event.get("status", "completed")
            try:
                self.query_one("#cancel-run", Button).disabled = True
            except Exception:
                pass
            self._run_in_progress = False
            self._refresh_stepper()
            self._set_monitor(f"Run finished with status: {status}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "refresh-dashboard":
            self._refresh_dashboard()
        elif button_id == "add-model":
            manual = self.query_one("#manual-model", Input)
            model = manual.value.strip()
            if model:
                self._select_model(model)
                manual.value = ""
        elif button_id == "fetch-models":
            await self._fetch_models()
        elif button_id == "save-prompt":
            self._save_prompt_from_ui()
        elif button_id == "start-run":
            await self._start_run()
        elif button_id == "run-baselines":
            self._start_baselines()
        elif button_id == "cancel-run":
            self._cancel_run()
        elif button_id == "refresh-runs":
            self._refresh_runs_table()
            self._refresh_results_help()
        elif button_id == "export-run":
            self._export_run()
        elif button_id == "open-exports":
            self._open_exports()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "sample-per-class":
            self._refresh_run_estimate()
        if event.input.id == "model-search":
            self._render_model_table()
        if event.input.id in _VALIDATED_INPUTS:
            self._update_validation_hint(event.input.id, event.validation_result)
            self._refresh_stepper()
            # Surface validation problems hidden inside the collapsed advanced panel.
            advanced_ids = {"seed", "concurrency", "temperature", "max-tokens"}
            result = event.validation_result
            if event.input.id in advanced_ids and result is not None and not result.is_valid:
                try:
                    self.query_one("#advanced-settings", Collapsible).collapsed = False
                except Exception:
                    pass

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id != "run-mode":
            return
        pressed_id = event.pressed.id or "mode-pilot"
        self.run_mode = pressed_id.removeprefix("mode-")
        try:
            self.query_one("#sample-per-class", Input).disabled = self.run_mode != "pilot"
        except Exception:
            pass
        self._refresh_run_estimate()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "prompt-preset":
            return
        if event.value is Select.BLANK:
            return
        preset = self.prompts.get(str(event.value))
        if preset is None:
            return
        self.query_one("#system-prompt", TextArea).text = preset.system_prompt
        self.query_one("#user-template", TextArea).text = preset.user_template
        self.query_one("#output-mode", Select).value = preset.output_mode
        self.prompt = preset
        self._refresh_prompt_preview()
        self._save_session()

    def _update_validation_hint(self, input_id: str, result: ValidationResult | None) -> None:
        try:
            hint = self.query_one(f"#hint-{input_id}", Static)
        except Exception:
            return
        if result is None or result.is_valid:
            hint.remove_class("-error")
            return
        hint.add_class("-error")
        message = "; ".join(result.failure_descriptions) if result.failure_descriptions else "Invalid value."
        hint.update(message)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table_id = event.data_table.id
        if table_id == "model-table":
            selected_row = event.data_table.get_row(event.row_key)
            if not selected_row:
                return
            model_id = str(selected_row[1]) if len(selected_row) >= 2 else ""
            name = str(selected_row[2]) if len(selected_row) >= 3 else ""
            was_selected = model_id in self.selected_models
            self._toggle_model(model_id, name or None)
            action = "Removed" if was_selected else "Added"
            self._set_monitor(f"{action} model: {model_id}")
        elif table_id == "selected-table":
            selected_row = event.data_table.get_row(event.row_key)
            if not selected_row:
                return
            model_id = str(selected_row[0])
            self._deselect_model(model_id)
            self._set_monitor(f"Removed model: {model_id}")
        elif table_id == "runs-table":
            key = event.row_key.value
            if key is None:
                return
            try:
                self._load_metrics_for(int(key))
            except ValueError:
                pass
        elif table_id == "metrics-table":
            key = event.row_key.value
            if key is None:
                return
            metric = self._metric_rows.get(str(key))
            if metric is not None:
                self._show_per_class(metric)
                self._active_metric_model = str(metric.get("model_id") or "") or None
                self._active_metric_scope = str(metric.get("scope") or "all")
                self._refresh_misclassifications()
        elif table_id == "misclassified-table":
            key = event.row_key.value
            if key is None:
                return
            row = self._misclassification_rows.get(str(key))
            if row is not None:
                self._show_misclassification_detail(row)

    async def _fetch_models(self) -> None:
        try:
            async with OpenRouterClient(base_url=self.base_url) as client:
                models = await client.list_models()
        except Exception as exc:
            self._notify_error(f"Could not fetch models: {exc}", title="Fetch failed")
            return
        self._all_models = list(models[:200])
        for model in self._all_models:
            if model.name:
                self._model_names[model.model_id] = model.name
        self._render_model_table()
        self._render_selected_table()
        self._set_monitor(
            f"Fetched {len(models)} models. Type in the search box to filter, press Enter on a row to toggle."
        )

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
            config = RunConfig(
                models=self.selected_models,
                prompt=self.prompt,
                mode=self.run_mode,  # type: ignore[arg-type]
                dataset_path=str(self.dataset_path),
                db_path=str(self.db_path),
                base_url=self.base_url,
                sample_per_class=int(self.query_one("#sample-per-class", Input).value),
                seed=int(self.query_one("#seed", Input).value),
                concurrency=int(self.query_one("#concurrency", Input).value),
                temperature=float(self.query_one("#temperature", Input).value),
                max_completion_tokens=int(self.query_one("#max-tokens", Input).value),
            )
            if config.mode not in {"pilot", "full"}:
                raise ValueError("Mode must be pilot or full")
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
        self._cancel_event = asyncio.Event()
        self._run_in_progress = True
        try:
            self.query_one("#cancel-run", Button).disabled = False
        except Exception:
            pass
        self._refresh_stepper()
        self._set_monitor("Starting benchmark run...")
        self._run_task = asyncio.create_task(self._run_benchmark(config))

    async def _run_benchmark(self, config: RunConfig) -> None:
        store = BenchmarkStore(self.db_path)
        try:
            async with OpenRouterClient(base_url=self.base_url) as client:
                runner = BenchmarkRunner(client=client, store=store)
                summary = await runner.run(
                    config,
                    callback=lambda message: self._set_monitor(message),
                    event_callback=self._handle_run_event,
                    cancel_event=self._cancel_event,
                )
            self._set_monitor(
                f"Run {summary.run_id} {summary.status} ({summary.selected_row_count} rows x {summary.model_count} models)."
            )
        except Exception as exc:
            self._notify_error(f"Run failed: {exc}", title="Run failed")
        finally:
            self._run_in_progress = False
            self._cancel_event = None
            self._run_task = None
            try:
                self.query_one("#cancel-run", Button).disabled = True
            except Exception:
                pass
            self._refresh_stepper()
            self._refresh_results_help()
            self._refresh_runs_table()

    def _cancel_run(self) -> None:
        if self._cancel_event is None or not self._run_in_progress:
            self._notify_error("No active run to cancel.", title="Nothing to cancel")
            return
        self._cancel_event.set()
        self._notify_info(
            "Cancel requested. The run will stop after in-flight requests finish.",
            title="Cancelling",
        )

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

    async def action_run_baselines(self) -> None:
        self._start_baselines()

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
            self.call_from_thread(self._refresh_runs_table)
            self.call_from_thread(self._refresh_results_help)

    def _refresh_runs_table(self) -> None:
        try:
            table = self.query_one("#runs-table", DataTable)
        except Exception:
            return
        table.clear()
        try:
            runs = BenchmarkStore(self.db_path).list_runs()
        except Exception:
            runs = []
        for run in runs:
            models = json.loads(run["models_json"]) if run["models_json"] else []
            preview = ", ".join(models[:3])
            if len(models) > 3:
                preview += f" (+{len(models) - 3})"
            created = (run["created_at"] or "")[:19]
            table.add_row(
                str(run["id"]),
                created,
                run["mode"],
                _status_text(run["status"]),
                preview,
                key=str(run["id"]),
            )

    def _load_metrics_for(self, run_id: int) -> None:
        self._active_run_id = run_id
        self._metric_rows = {}
        self._active_metric_model = None
        self._active_metric_scope = "all"
        metrics_table = self.query_one("#metrics-table", DataTable)
        metrics_table.clear()
        self.query_one("#perclass-table", DataTable).clear()
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
                str(metric["row_count"]),
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
                f"{float(scores.get('precision', 0.0)):.4f}",
                f"{float(scores.get('recall', 0.0)):.4f}",
                f"{float(scores.get('f1', 0.0)):.4f}",
                str(int(float(scores.get('support', 0.0)))),
            )

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
        paths = export_run(self.db_path, self._active_run_id)
        self._notify_info(
            f"Exported run {self._active_run_id} ({len(paths)} files).",
            title="Export complete",
        )
        self._set_monitor(
            f"Exported run {self._active_run_id}:\n" + "\n".join(str(path) for path in paths)
        )

    def _open_exports(self) -> None:
        import shutil
        import subprocess
        import sys

        folder = Path("results/exports")
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            opener = "open"
        elif sys.platform.startswith("win"):
            opener = "explorer"
        else:
            opener = "xdg-open"
        if shutil.which(opener) is None:
            self._notify_error(f"Cannot open {folder}: {opener} is not on PATH.", title="Open failed")
            return
        try:
            subprocess.Popen([opener, str(folder)])
            self._notify_info(f"Opened {folder} in your file manager.")
        except OSError as exc:
            self._notify_error(f"Could not open {folder}: {exc}", title="Open failed")


def main() -> None:
    SentimentBenchmarkApp().run()
