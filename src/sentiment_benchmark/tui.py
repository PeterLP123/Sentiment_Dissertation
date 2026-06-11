from __future__ import annotations

import asyncio
import copy
import faulthandler
import importlib.util
import json
import logging
import os
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Event as ThreadEvent

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
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
from textual.worker import WorkerFailed

from .baseline_runner import run_baselines
from .baselines import BASELINE_SPECS, DEFAULT_BASELINES, BaselineSpec
from .comparison import ComparisonResult, ModelTarget, compare_models
from .constants import (
    ALLOWED_LABELS,
    DEFAULT_BASE_URL,
    DEFAULT_DATASET_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_PROVIDER,
)
from .dataset import compute_stats, load_dataset
from .env import load_env_file
from .exporter import export_run
from .models import ModelConfig, RunConfig
from .news_source import (
    DEFAULT_NEWS_MAX_RESULTS,
    DEFAULT_NEWS_OUTPUT_DIR,
    DEFAULT_NEWS_TIME_RANGE,
    DEFAULT_NEWS_TOPIC,
    NEWS_TIME_RANGES,
    NEWS_TOPICS,
    TavilyNewsClient,
    make_news_fetch_config,
    write_news_corpus,
)
from .ollama_cloud import _to_cloud_tag, cloud_catalog_overridden, fetch_ollama_cloud_models, ollama_cloud_catalog
from .prompts import load_prompts, make_prompt
from .providers import endpoint_for_provider, make_llm_client, normalize_provider
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
# (cache key, widget id, caster, low, high) for run settings persisted across sessions.
_RUN_SETTING_FIELDS = (
    ("sample_per_class", "sample-per-class", int, 1, 10000),
    # Seed widget validator only enforces >= 0 (no upper bound), so don't reject large seeds on reload.
    ("seed", "seed", int, 0, 2**63 - 1),
    ("concurrency", "concurrency", int, 1, 64),
    ("temperature", "temperature", float, 0.0, 2.0),
    ("max_completion_tokens", "max-tokens", int, 16, 8192),
)
_SESSION_PATH = Path("results/tui_session.json")
_QUEUE_PATH = Path("results/tui_queue.json")
_LOG_PATH = Path("results/tui.log")
_CRASH_LOG_PATH = Path("results/tui_crash.log")

logger = logging.getLogger(__name__)
_SELECTED_MARK = "[x]"
_UNSELECTED_MARK = "[ ]"
_CONFIRM_THRESHOLD = 1000
_MONITOR_FIELD = re.compile(r"\b(run|model|row|status|label)=([^\s]+)")

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
    "interrupted": "orange1",
}


def _num(value: object, style: str | None = None) -> Text:
    """A right-aligned numeric cell so magnitudes line up down a column."""
    return Text(str(value), style=style or "", justify="right")


def _accuracy_text(value: float, *, bold: bool = False) -> Text:
    """Colour an accuracy/F1 score on a traffic-light scale."""
    if value >= 0.8:
        color = "green"
    elif value >= 0.6:
        color = "yellow"
    else:
        color = "red"
    return Text(f"{value:.4f}", style=f"bold {color}" if bold else color, justify="right")


def _status_text(status: str) -> Text:
    """Colour a status label (first word drives the colour, e.g. 'done (acc ...)')."""
    key = status.split(" ", 1)[0].lower()
    return Text(status, style=_STATUS_COLORS.get(key, "white"))


def _count_text(value: int) -> Text:
    """Red when there is something to worry about (errors/invalids), dim otherwise."""
    return Text(str(value), style="red" if value > 0 else "grey58", justify="right")


def _latency_text(value: float | None) -> Text:
    if not isinstance(value, (int, float)):
        return Text("-", style="grey58", justify="right")
    return Text(f"{value:.0f} ms", justify="right")


def _tokens_text(value: int | None) -> Text:
    if not isinstance(value, (int, float)) or value == 0:
        return Text("-", style="grey58", justify="right")
    return Text(f"{int(value):,}", justify="right")


def _cost_text(value: float | None) -> Text:
    if not isinstance(value, (int, float)):
        return Text("-", style="grey58", justify="right")
    return Text(f"${value:.4f}", style="cyan", justify="right")


def _monitor_field_style(key: str, value: str) -> str:
    if key == "model":
        return "bold cyan"
    if key in {"run", "row"}:
        return "cyan"
    if key == "status":
        return _STATUS_COLORS.get(value.lower(), "white")
    if key == "label":
        return {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow",
            "-": "grey58",
        }.get(value.lower(), "white")
    return "white"


def _monitor_text(message: str) -> Text:
    text = Text()
    cursor = 0
    for match in _MONITOR_FIELD.finditer(message):
        text.append(message[cursor : match.start()])
        key, value = match.groups()
        text.append(f"{key}=", style="yellow")
        text.append(value, style=_monitor_field_style(key, value))
        cursor = match.end()
    text.append(message[cursor:])
    return text


# Sort metadata for the clickable Results tables. Each column index maps to
# (header label, ascending-on-first-click, key extractor). Numeric/date columns
# default to descending (most recent / highest first); text columns to ascending.
_RUNS_SORT_COLUMNS: dict[int, tuple] = {
    0: ("ID", False, lambda run: int(run["id"])),
    1: ("Created", False, lambda run: run["created_at"] or ""),
    2: ("Mode", True, lambda run: run["mode"] or ""),
    3: ("Status", True, lambda run: run["status"] or ""),
    4: ("Machine", True, lambda run: (run["machine_label"] or run["machine_id"] or "").lower()),
    5: ("Models", False, lambda run: len(json.loads(run["models_json"]) if run["models_json"] else [])),
}
_LEADERBOARD_SORT_COLUMNS: dict[int, tuple] = {
    0: ("Model", True, "model_id"),
    1: ("Best Acc", False, "accuracy"),
    2: ("Macro F1", False, "macro_f1"),
    3: ("Best run", False, "run_id"),
    4: ("Runs", False, "runs"),
    5: ("Rows", False, "row_count"),
}
_RUNS_HELP_BASE = "Press Enter on a row to load metrics for that run. Click a column header to sort."
_LEADERBOARD_HELP_BASE = (
    "Best accuracy each model has reached across all stored runs at the chosen scope. "
    "Macro F1, best run id, and run count are for that best run. Click a column header to sort."
)


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


class QueueResumeScreen(ModalScreen[bool]):
    """Ask whether to resume a saved experiment queue or start fresh."""

    DEFAULT_CSS = """
    QueueResumeScreen {
        align: center middle;
    }
    #queue-resume-box {
        background: $surface;
        border: thick $accent;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    #queue-resume-message {
        margin-bottom: 1;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="queue-resume-box"):
            yield Static(self._message, id="queue-resume-message")
            with Horizontal():
                yield Button("Resume queue", id="queue-resume-yes", variant="primary")
                yield Button("Start new", id="queue-resume-no", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "queue-resume-yes")


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
        "1-6    Switch tabs (Dashboard, Models, Prompt, Run, Results, News)",
        "r      Refresh the dashboard",
        "s      Start the benchmark run",
        "a      Add the current config to the experiment queue",
        "g      Run the queued experiments in order",
        "c      Cancel an in-progress run or queue",
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
    SUB_TITLE = "LLM sentiment evaluation"
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
    #dashboard, #dashboard-resource-monitor, #run-resource-monitor, #prompt-preview,
    #run-estimate, #results-help, #selected-summary, #news-summary, #compare-result, #ollama-loaded {
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
    #run-monitor {
        height: 16;
        margin-bottom: 1;
    }
    #run-monitor-left {
        width: 1fr;
        height: 100%;
        padding-right: 1;
    }
    #run-monitor-right {
        width: 1fr;
        height: 100%;
    }
    #monitor {
        height: 1fr;
        border: solid $accent;
    }
    #news-log {
        height: 12;
        margin-bottom: 1;
        border: solid $accent;
    }
    #results-scroll, #run-scroll {
        height: 1fr;
    }
    .card {
        height: auto;
        border: round $accent;
        padding: 0 1 1 1;
        margin-bottom: 1;
    }
    .card > .section-title {
        margin-top: 0;
    }
    .toolbar {
        height: auto;
        margin-top: 1;
    }
    #runs-table {
        height: auto;
        max-height: 10;
        margin-bottom: 0;
    }
    #leaderboard-table {
        height: auto;
        max-height: 12;
        margin-bottom: 1;
    }
    #leaderboard-controls {
        height: auto;
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
    #confusion-table {
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
    #queue-controls, #queue-row-controls {
        height: auto;
        margin-bottom: 1;
    }
    #queue-table {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }
    #queue-summary {
        color: $text-muted;
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
        ("6", "show_tab('news-tab')", "News"),
        ("r", "refresh", "Refresh"),
        ("s", "start_run", "Start"),
        ("a", "queue_add", "Queue"),
        ("g", "run_queue", "Run queue"),
        ("b", "run_baselines", "Baselines"),
        ("c", "cancel_run", "Cancel"),
        ("ctrl+t", "cycle_theme", "Theme"),
    ]

    # A small curated rotation for the Theme keybinding; the full set is still
    # reachable through the command palette (ctrl+p).
    THEME_CYCLE = [
        "textual-dark",
        "nord",
        "gruvbox",
        "tokyo-night",
        "catppuccin-mocha",
        "textual-light",
        "solarized-light",
    ]

    def __init__(
        self,
        *,
        session_path: Path | None = None,
        queue_path: Path | None = None,
        db_path: Path | None = None,
    ) -> None:
        super().__init__()
        load_env_file()
        self.dataset_path = DEFAULT_DATASET_PATH
        self.db_path = db_path if db_path is not None else DEFAULT_DB_PATH
        try:
            self.provider = normalize_provider(os.getenv("SENTIMENT_BENCH_PROVIDER", DEFAULT_PROVIDER))
        except ValueError:
            self.provider = normalize_provider(DEFAULT_PROVIDER)
        self.base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
        self.ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.news_query = "financial markets"
        self.news_topic = DEFAULT_NEWS_TOPIC
        self.news_time_range = DEFAULT_NEWS_TIME_RANGE
        self.news_max_results = DEFAULT_NEWS_MAX_RESULTS
        self.news_extract = True
        self.news_output_dir = DEFAULT_NEWS_OUTPUT_DIR
        self.disable_ollama_thinking = True
        self.selected_models: list[str] = []
        self.prompts = load_prompts(DEFAULT_PROMPTS_PATH)
        self._default_prompt_id = (
            "default_label_only" if "default_label_only" in self.prompts else next(iter(self.prompts))
        )
        self.prompt = self.prompts[self._default_prompt_id]
        self.run_mode: str = "pilot"
        self.monitor_lines: list[str] = []
        self.news_lines: list[str] = []
        self._all_models: list[ModelConfig] = []
        self._model_names: dict[str, str] = {}
        self._session_path = session_path if session_path is not None else _SESSION_PATH
        self._active_run_id: int | None = None
        self._metric_rows: dict[str, dict] = {}
        self._cancel_event: ThreadEvent | None = None
        self._run_task: asyncio.Task | None = None
        self._model_fetch_task: asyncio.Task | None = None
        self._run_in_progress: bool = False
        self._baseline_in_progress: bool = False
        self._news_in_progress: bool = False
        self._confirmation_pending: bool = False
        self._progress: dict[str, dict] = {}
        self._rows_per_model: int = 0
        self._active_metric_model: str | None = None
        self._active_metric_scope: str = "all"
        self._misclassification_rows: dict[str, dict] = {}
        self._leaderboard_best_run: dict[str, int] = {}
        self.notifications: list[tuple[str, str]] = []
        self._gpu_lines: list[str] = ["GPU: gathering data..."]
        self._run_settings: dict = {
            "run_mode": "pilot",
            "sample_per_class": 30,
            "seed": 42,
            "concurrency": 1,
            "temperature": 0.0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        }
        self._experiment_queue: list[dict] = []
        self._queue_path = queue_path if queue_path is not None else _QUEUE_PATH
        self._queue_uid_counter: int = 0
        self._queue_running: bool = False
        self._queue_cancel: bool = False
        self._theme_name = "textual-dark"
        self._button_labels: dict[str, object] = {}
        self._runs_sort: tuple[int, bool] | None = None
        self._leaderboard_sort: tuple[int, bool] | None = None
        self._load_session()
        self._load_queue()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="status-bar")
        with TabbedContent():
            with TabPane("Dashboard", id="dashboard-tab"):
                yield Static(
                    "Use this app to run OpenRouter or Ollama models against the dissertation sentiment dataset. "
                    "Only Sentence text is sent to models; hidden Sentiment labels stay in the evaluator.",
                    classes="help",
                )
                yield Static(id="dashboard")
                yield Static(id="dashboard-resource-monitor")
                with Horizontal(classes="toolbar"):
                    yield Button("Refresh Dashboard", id="refresh-dashboard")
            with TabPane("Models", id="models-tab"):
                yield Static(
                    "Step 1: choose the models to test. Fetch models to browse available IDs, "
                    "press Enter on a row to toggle it, or type a model ID manually.",
                    classes="help",
                )
                yield Static("Provider", classes="field-label")
                yield Select(
                    [("OpenRouter", "openrouter"), ("Ollama", "ollama")],
                    id="provider",
                    value=self.provider,
                    allow_blank=False,
                )
                yield Static("Provider endpoint", classes="field-label")
                yield Input(
                    value=self._active_endpoint(),
                    placeholder=self._endpoint_placeholder(),
                    id="provider-endpoint",
                )
                yield Static("Manual model ID", classes="field-label")
                yield Static(
                    "Examples: openai/gpt-4o-mini or gemma3. Ollama Cloud models work through a signed-in "
                    "local daemon — add them by their cloud tag, e.g. gpt-oss:120b-cloud or minimax-m3:cloud, "
                    "or press Cloud Catalog to browse known cloud models without pulling them. "
                    "Repeat Add Model for each model you want in the same benchmark run.",
                    classes="help",
                )
                yield Input(placeholder="gpt-oss:120b-cloud, minimax-m3:cloud, or openai/gpt-4o-mini", id="manual-model")
                with Horizontal(classes="toolbar"):
                    yield Button("Add Model", id="add-model", variant="primary")
                    yield Button("Fetch Models", id="fetch-models")
                    yield Button("Cloud Catalog", id="show-cloud-catalog")
                    yield Button("Loaded in Ollama", id="fetch-loaded")
                yield Static(id="ollama-loaded")
                yield Static("Selected models", classes="section-title")
                yield Static(id="selected-summary")
                yield Static(
                    "Press Enter on a row below to remove it from the run.",
                    classes="help",
                )
                yield DataTable(id="selected-table")
                yield Static("Fetched provider models", classes="section-title")
                yield Static(
                    "Type to filter. Press Enter on a row to toggle selection. [x] = currently selected. "
                    "The Cloud column marks Ollama Cloud models; type 'cloud' to filter to them.",
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
                with VerticalScroll(id="run-scroll"):
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
                        yield Static("Ollama thinking", classes="field-label")
                        yield Static(
                            "Disable thinking for short classification runs so reasoning tokens do not consume the answer budget.",
                            classes="help",
                        )
                        yield Checkbox(
                            "Disable Ollama thinking",
                            value=self.disable_ollama_thinking,
                            id="disable-ollama-thinking",
                        )
                        yield Static(
                            "",
                            id="ollama-thinking-recommendation",
                            classes="validation-hint",
                        )
                    with Collapsible(
                        title="Baselines \u2014 non-LLM comparators (optional)",
                        collapsed=True,
                        id="baselines-section",
                    ):
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
                    yield Static("Experiment queue", classes="section-title")
                    yield Static(
                        "Snapshot the current models, prompt, and run settings as a queued experiment, then run several "
                        "back to back. Each experiment is stored as its own run. The queue is saved to results/tui_queue.json. "
                        "Press Enter on a row to remove it; Move/Clone act on the highlighted row.",
                        classes="help",
                    )
                    yield Static(id="queue-summary")
                    with Horizontal(id="queue-controls"):
                        yield Button("Add to Queue", id="add-to-queue", variant="primary")
                        yield Button("Run Queue", id="run-queue", variant="success")
                        yield Button("Clear Queue", id="clear-queue", variant="error")
                    with Horizontal(id="queue-row-controls"):
                        yield Button("Move Up", id="queue-move-up")
                        yield Button("Move Down", id="queue-move-down")
                        yield Button("Clone", id="queue-clone")
                    yield DataTable(id="queue-table")
                    with Collapsible(
                        title="Sweep builder — queue many experiments at once",
                        collapsed=True,
                        id="sweep-builder",
                    ):
                        yield Static(
                            "Expand the current models, prompt, and settings into several queued experiments that "
                            "vary one axis. Everything else stays fixed.",
                            classes="help",
                        )
                        yield Static("Axis", classes="field-label")
                        yield Select(
                            [
                                ("Temperatures", "temperature"),
                                ("Prompt presets", "prompt"),
                                ("One experiment per model", "model"),
                            ],
                            id="sweep-axis",
                            value="temperature",
                            allow_blank=False,
                        )
                        yield Static("Values", classes="field-label")
                        yield Static(
                            "Temperatures: comma list, e.g. 0,0.3,0.7. "
                            "Prompt presets: comma list of preset ids (blank = every preset). "
                            "Per model: this box is ignored; each selected model becomes its own experiment.",
                            classes="help",
                        )
                        yield Input(placeholder="0,0.3,0.7", id="sweep-values")
                        yield Button("Add Sweep to Queue", id="add-sweep", variant="primary")
                    yield Static("Live monitor", classes="section-title")
                    with Horizontal(id="run-monitor"):
                        with Vertical(id="run-monitor-left"):
                            yield Static("Progress", classes="field-label")
                            yield ProgressBar(id="run-progress-bar", total=100, show_percentage=True, show_eta=True)
                            yield DataTable(id="run-progress")
                            yield Static(id="run-resource-monitor")
                        with Vertical(id="run-monitor-right"):
                            yield Static("Live log", classes="field-label")
                            yield RichLog(id="monitor", highlight=False, markup=False, wrap=True)
            with TabPane("Results", id="results-tab"):
                with VerticalScroll(id="results-scroll"):
                    yield Static(
                        "Step 4: review completed runs. Pick a run from the list to load its metrics.",
                        classes="help",
                    )
                    yield Static(id="results-help")

                    with Container(classes="card"):
                        yield Static("Recent runs", classes="section-title")
                        yield Static(_RUNS_HELP_BASE, id="runs-help", classes="help")
                        yield DataTable(id="runs-table")
                        with Horizontal(classes="toolbar"):
                            yield Button("Refresh Runs", id="refresh-runs")
                            yield Button("Export Selected Run", id="export-run")
                            yield Button("View Figures", id="view-figures", variant="primary")
                            yield Button("Open Exports Folder", id="open-exports")

                    with Container(classes="card"):
                        yield Static("Metrics", classes="section-title")
                        yield Static(
                            "Press Enter on a run above to load its metrics, then on a metric row "
                            "to drill into per-class scores and the confusion matrix.",
                            classes="help",
                        )
                        yield DataTable(id="metrics-table")
                        with Collapsible(title="Per-class breakdown", collapsed=True, id="perclass-section"):
                            yield DataTable(id="perclass-table")
                        with Collapsible(title="Confusion matrix", collapsed=True, id="confusion-section"):
                            yield Static(
                                "Rows = actual label, columns = predicted (incl. invalid/error). "
                                "Select a metric row above to populate.",
                                classes="help",
                            )
                            yield DataTable(id="confusion-table")

                    with Collapsible(title="Misclassified and failed rows", collapsed=True, id="misclassified-section"):
                        yield Static(
                            "Load a run to see mismatches. Select a metric row to filter by model/scope; "
                            "press Enter on a row for details.",
                            classes="help",
                        )
                        yield DataTable(id="misclassified-table")
                        yield Static(
                            "Select a misclassified row to inspect the sentence and raw model output.",
                            id="misclassified-detail",
                        )

                    with Collapsible(title="Leaderboard — best run per model", collapsed=True):
                        yield Static(_LEADERBOARD_HELP_BASE, id="leaderboard-help", classes="help")
                        with Horizontal(id="leaderboard-controls"):
                            yield Select(
                                [("Primary (excludes conflicting duplicates)", "primary"), ("All scored rows", "all")],
                                id="leaderboard-scope",
                                value="primary",
                                allow_blank=False,
                            )
                            yield Button("Refresh Leaderboard", id="refresh-leaderboard")
                        yield DataTable(id="leaderboard-table")

                    with Collapsible(title="Compare two runs", collapsed=True):
                        yield Static(
                            "Pick two run/model targets to test whether their accuracy differs on the rows they share "
                            "(paired McNemar test + bootstrap confidence intervals).",
                            classes="help",
                        )
                        yield Static("Target A", classes="field-label")
                        yield Select([], id="compare-a", prompt="Pick run · model")
                        yield Static("Target B", classes="field-label")
                        yield Select([], id="compare-b", prompt="Pick run · model")
                        yield Static("Scope", classes="field-label")
                        yield Select(
                            [("Primary (excludes conflicting duplicates)", "primary"), ("All scored rows", "all")],
                            id="compare-scope",
                            value="primary",
                            allow_blank=False,
                        )
                        yield Button("Compare", id="compare-run", variant="primary")
                        yield Static("Pick two targets and press Compare.", id="compare-result")
            with TabPane("News", id="news-tab"):
                yield Static(
                    "Source unlabeled news articles from Tavily into derived files. This does not modify Data/data.csv.",
                    classes="help",
                )
                yield Static("Search query", classes="field-label")
                yield Input(value=self.news_query, placeholder="bank earnings sentiment", id="news-query")
                yield Static("Topic", classes="field-label")
                yield Select(
                    [(topic.title(), topic) for topic in NEWS_TOPICS],
                    id="news-topic",
                    value=self.news_topic,
                    allow_blank=False,
                )
                yield Static("Time range", classes="field-label")
                yield Select(
                    [(value.title(), value) for value in NEWS_TIME_RANGES],
                    id="news-time-range",
                    value=self.news_time_range,
                    allow_blank=False,
                )
                yield Static("Max results", classes="field-label")
                yield Input(
                    value=str(self.news_max_results),
                    placeholder="10",
                    id="news-max-results",
                    type="integer",
                    validators=[Integer(minimum=1, maximum=20)],
                )
                yield Checkbox("Extract full article text", value=self.news_extract, id="news-extract")
                yield Static("Output directory", classes="field-label")
                yield Input(value=str(self.news_output_dir), placeholder="Data/news", id="news-output-dir")
                yield Static(id="news-summary")
                with Horizontal(id="news-controls"):
                    yield Button("Check Tavily", id="news-check")
                    yield Button("Fetch News", id="news-fetch", variant="primary")
                yield Static("News log", classes="section-title")
                yield RichLog(id="news-log", highlight=True, markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        if self._theme_name in self.available_themes:
            self.theme = self._theme_name
        model_table = self.query_one("#model-table", DataTable)
        model_table.add_columns("Sel", "Model ID", "Name", "Context", "Cloud")
        model_table.cursor_type = "row"
        selected_table = self.query_one("#selected-table", DataTable)
        selected_table.add_columns("Model ID", "Name")
        selected_table.cursor_type = "row"
        metrics = self.query_one("#metrics-table", DataTable)
        metrics.add_columns("Model", "Scope", "Rows", "Accuracy", "Macro F1", "Latency", "Tokens", "Cost", "Invalid", "Errors")
        metrics.cursor_type = "row"
        runs = self.query_one("#runs-table", DataTable)
        runs.add_columns("ID", "Created", "Mode", "Status", "Machine", "Models")
        runs.cursor_type = "row"
        leaderboard = self.query_one("#leaderboard-table", DataTable)
        leaderboard.add_columns("Model", "Best Acc", "Macro F1", "Best run", "Runs", "Rows")
        leaderboard.cursor_type = "row"
        perclass = self.query_one("#perclass-table", DataTable)
        perclass.add_columns("Class", "Precision", "Recall", "F1", "Support")
        confusion = self.query_one("#confusion-table", DataTable)
        confusion.add_columns("Actual \\ Pred", "positive", "negative", "neutral", "invalid", "error")
        misclassified = self.query_one("#misclassified-table", DataTable)
        misclassified.add_columns("Row", "Model", "Actual", "Predicted", "Status", "Sentence")
        misclassified.cursor_type = "row"
        progress = self.query_one("#run-progress", DataTable)
        progress.add_column("Model")
        progress.add_column("Done/Total", width=10)
        progress.add_column("Errors", width=6)
        progress.add_column("Avg latency", width=11)
        progress.add_column("Status", width=9)
        queue = self.query_one("#queue-table", DataTable)
        queue.add_columns("#", "Provider", "Mode", "Models", "Prompt", "Sample", "Status")
        queue.cursor_type = "row"

        # Zebra striping makes wide multi-column tables far easier to read across.
        for table in (
            model_table,
            metrics,
            runs,
            leaderboard,
            perclass,
            confusion,
            misclassified,
            progress,
            queue,
        ):
            table.zebra_stripes = True

        # Label the free-standing bordered panels so they read as titled cards.
        for panel_id, title in (
            ("#dashboard", "Dataset & environment"),
            ("#dashboard-resource-monitor", "Local resource monitor"),
            ("#run-resource-monitor", "Local resource monitor"),
            ("#prompt-preview", "Active prompt"),
            ("#misclassified-detail", "Row detail"),
            ("#news-summary", "Tavily sourcing"),
            ("#compare-result", "Statistical comparison"),
            ("#ollama-loaded", "Loaded in Ollama (VRAM)"),
        ):
            try:
                self.query_one(panel_id, Static).border_title = title
            except Exception:
                pass

        self._reconcile_orphaned_runs()
        self._apply_loaded_run_settings()
        self._refresh_dashboard()
        self._render_selected_table()
        self._render_model_table()
        self._refresh_prompt_preview()
        self._refresh_run_estimate()
        self._refresh_results_help()
        self._refresh_runs_table()  # also refreshes the compare targets and leaderboard
        self._refresh_news_summary()
        self._refresh_status_bar()
        self._refresh_stepper()
        self._refresh_resource_monitor()
        self._render_queue_table()
        self._reset_ollama_loaded_hint()
        # Gather GPU stats off the event loop so the blocking nvidia-smi call does
        # not stall in-flight API requests during a run or queue drain.
        self.set_interval(2.0, self._refresh_gpu_lines)
        self._schedule_auto_fetch_models()
        self._maybe_prompt_queue_resume()

    def _reconcile_orphaned_runs(self) -> None:
        """Flip runs stranded in 'running' by a dead session to 'interrupted'."""
        try:
            orphaned = BenchmarkStore(self.db_path).reconcile_orphaned_runs()
        except Exception:
            logger.exception("Could not reconcile orphaned runs")
            return
        if not orphaned:
            return
        ids = ", ".join(str(run_id) for run_id in orphaned)
        logger.warning("Marked %d orphaned run(s) as interrupted: %s", len(orphaned), ids)
        self._notify_info(
            f"Marked {len(orphaned)} unfinished run(s) from a previous session as interrupted: {ids}. "
            "Finish them with: sentiment-bench run --resume-run-id <id> ...",
            title="Recovered runs",
        )

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

    def _schedule_auto_fetch_models(self) -> None:
        if not self._should_auto_fetch_models():
            return
        if self._model_fetch_task is not None and not self._model_fetch_task.done():
            return
        self._model_fetch_task = asyncio.create_task(self._auto_fetch_models())

    def _should_auto_fetch_models(self) -> bool:
        value = os.getenv("SENTIMENT_BENCH_AUTO_FETCH_MODELS", "1").strip().lower()
        enabled = value not in {"0", "false", "no", "off"}
        return enabled and self.provider == "ollama" and self.ollama_host.startswith("http://localhost")

    async def _auto_fetch_models(self) -> None:
        self._set_monitor("Auto-detecting local Ollama models...")
        await self._fetch_models()

    def _set_monitor(self, message: str) -> None:
        self.monitor_lines.append(message)
        self.monitor_lines = self.monitor_lines[-500:]
        try:
            log = self.query_one("#monitor", RichLog)
        except Exception:
            return
        should_follow = bool(log.is_vertical_scroll_end)
        log.write(_monitor_text(message), scroll_end=should_follow)

    def _set_news_log(self, message: str) -> None:
        self.news_lines.append(message)
        self.news_lines = self.news_lines[-500:]
        try:
            log = self.query_one("#news-log", RichLog)
        except Exception:
            return
        should_follow = bool(log.is_vertical_scroll_end)
        log.write(message, scroll_end=should_follow)

    def _make_news_client(self) -> TavilyNewsClient:
        return TavilyNewsClient()

    def _refresh_news_summary(self) -> None:
        try:
            summary = self.query_one("#news-summary", Static)
        except Exception:
            return
        api_state = "present" if os.getenv("TAVILY_API_KEY") else "missing"
        summary.update(
            "\n".join(
                [
                    f"Tavily API key: {api_state}",
                    f"Output directory: {self.news_output_dir}",
                    "Generated corpora are unlabeled source material.",
                ]
            )
        )

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

    def _active_endpoint(self) -> str:
        return endpoint_for_provider(self.provider, base_url=self.base_url, ollama_host=self.ollama_host)

    def _endpoint_placeholder(self) -> str:
        return "http://desktop-pc:11434" if self.provider == "ollama" else DEFAULT_BASE_URL

    def _provider_title(self) -> str:
        return "Ollama" if self.provider == "ollama" else "OpenRouter"

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

    def action_queue_add(self) -> None:
        self._add_current_to_queue()

    async def action_run_queue(self) -> None:
        await self._start_queue()

    def action_cancel_run(self) -> None:
        self._cancel_run()

    def action_cycle_theme(self) -> None:
        """Rotate through the curated theme list and remember the choice."""
        try:
            index = self.THEME_CYCLE.index(self.theme)
        except ValueError:
            index = -1
        self._theme_name = self.THEME_CYCLE[(index + 1) % len(self.THEME_CYCLE)]
        self.theme = self._theme_name
        self._notify_info(f"Theme: {self._theme_name}  (ctrl+t to cycle, ctrl+p for all)", title="Theme")
        self._save_session()

    def _refresh_status_bar(self) -> None:
        try:
            bar = self.query_one("#status-bar", Static)
        except Exception:
            return
        provider_status = (
            f"OpenRouter key: {'OK' if os.getenv('OPENROUTER_API_KEY') else 'missing'}"
            if self.provider == "openrouter"
            else f"Ollama: {self.ollama_host}"
        )
        queue_pending = sum(1 for item in self._experiment_queue if not self._queue_item_done(item))
        queue_status = f"queue: {queue_pending} pending" if self._experiment_queue else "queue: empty"
        base = (
            f" {provider_status}  |  models: {len(self.selected_models)}  |  "
            f"prompt: {self.prompt.prompt_id}  |  mode: {self.run_mode}  |  {queue_status} "
        )
        run_segment = self._run_status_segment()
        if run_segment is None:
            bar.update(base)
        else:
            text = Text(base)
            text.append(" |  ")
            text.append_text(run_segment)
            bar.update(text)
        self._refresh_stepper()

    def _run_status_segment(self) -> Text | None:
        """A live 'run in progress' indicator for the global status bar, or None when idle."""
        if self._baseline_in_progress and not self._run_in_progress:
            return Text("● running baselines", style="bold yellow")
        if not self._run_in_progress or not self._progress:
            return None
        states = list(self._progress.values())
        total_models = len(states)
        done_models = sum(1 for state in states if state["status"] == "done")
        running = [model_id for model_id, state in self._progress.items() if state["status"] == "running"]
        rows_per_model = self._rows_per_model or 0
        rows_done = sum(state["done"] for state in states)
        rows_total = rows_per_model * total_models
        errors = sum(state["errors"] for state in states)
        current_index = min(total_models, done_models + (1 if running else 0))
        parts = [f"● running · model {current_index}/{total_models}"]
        if running:
            current = running[0]
            short_name = current.split("/")[-1]
            parts.append(f"{short_name} {self._progress[current]['done']}/{rows_per_model}")
        if rows_total:
            parts.append(f"{rows_done}/{rows_total} rows")
        if errors:
            parts.append(f"{errors} err")
        return Text(" · ".join(parts), style="bold red" if errors else "bold green")

    def _refresh_dashboard(self) -> None:
        rows = load_dataset(self.dataset_path)
        stats = compute_stats(rows)
        key_status = "present" if os.getenv("OPENROUTER_API_KEY") else "missing"
        endpoint = self._active_endpoint()
        storage_label = BenchmarkStore(self.db_path).storage_label()
        self.query_one("#dashboard", Static).update(
            "\n".join(
                [
                    f"Dataset: {self.dataset_path}",
                    f"Rows: {stats.row_count}",
                    f"Labels: {stats.label_counts}",
                    f"Conflicting duplicate rows excluded from primary metrics: {stats.conflicting_duplicate_rows}",
                    f"Primary scoring rows: {stats.primary_row_count}",
                    f"Result DB: {storage_label}",
                    f"Provider: {self._provider_title()}",
                    f"Endpoint: {endpoint}",
                    f"OpenRouter API key: {key_status}",
                ]
            )
        )
        self._refresh_status_bar()

    @staticmethod
    def _format_gpu_value(value: str, suffix: str = "") -> str:
        value = value.strip()
        if not value or "not supported" in value.lower():
            return "-"
        try:
            number = float(value)
        except ValueError:
            return value
        if number.is_integer():
            return f"{int(number)}{suffix}"
        return f"{number:.1f}{suffix}"

    def _gpu_monitor_lines(self) -> list[str]:
        command = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
        except FileNotFoundError:
            return ["GPU: nvidia-smi not found."]
        except subprocess.TimeoutExpired:
            return ["GPU: nvidia-smi timed out."]
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or exc.stdout or str(exc)).strip()
            return [f"GPU: nvidia-smi failed: {error}"]

        lines: list[str] = []
        for index, raw_line in enumerate(result.stdout.splitlines()):
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 7:
                continue
            name, util, mem_used, mem_total, power_draw, power_limit, temp = parts[:7]
            used = self._format_gpu_value(mem_used)
            total = self._format_gpu_value(mem_total)
            memory_percent = "-"
            try:
                memory_percent = f"{(float(mem_used) / float(mem_total)) * 100:.0f}%"
            except (ValueError, ZeroDivisionError):
                pass
            lines.append(
                "GPU "
                f"{index}: {name} | util {self._format_gpu_value(util, '%')} | "
                f"VRAM {used}/{total} MB ({memory_percent}) | "
                f"power {self._format_gpu_value(power_draw)}/{self._format_gpu_value(power_limit)} W | "
                f"temp {self._format_gpu_value(temp)} C"
            )
        return lines or ["GPU: no NVIDIA devices reported."]

    def _refresh_resource_monitor(self) -> None:
        try:
            concurrency = self.query_one("#concurrency", Input).value.strip() or "?"
        except Exception:
            concurrency = "?"
        try:
            max_tokens = self.query_one("#max-tokens", Input).value.strip() or "?"
        except Exception:
            max_tokens = "?"

        lines = [
            f"Provider: {self._provider_title()} | Endpoint: {self._active_endpoint()}",
            f"Selected models: {len(self.selected_models)} | Fetched provider models: {len(self._all_models)}",
            f"Run settings: TUI concurrency {concurrency} | max completion tokens {max_tokens}",
            f"Ollama NUM_PARALLEL env visible to TUI: {os.getenv('OLLAMA_NUM_PARALLEL') or 'default'}",
            f"Ollama thinking: {'disabled' if self.disable_ollama_thinking else 'provider default'}",
            *self._gpu_lines,
        ]
        content = "\n".join(lines)
        for widget_id in ("#dashboard-resource-monitor", "#run-resource-monitor"):
            try:
                self.query_one(widget_id, Static).update(content)
            except Exception:
                pass

    async def _refresh_gpu_lines(self) -> None:
        """Refresh cached GPU stats in a worker thread, then re-render the monitor."""
        try:
            self._gpu_lines = await asyncio.to_thread(self._gpu_monitor_lines)
        except Exception:  # pragma: no cover - defensive
            self._gpu_lines = ["GPU: monitor unavailable."]
        self._refresh_resource_monitor()

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
            cloud = self._selected_cloud_count()
            cloud_note = f" ({cloud} cloud)" if cloud else ""
            summary.update(f"{count} model(s) selected{cloud_note}.")

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
            cloud_cell = (
                Text("cloud", style="cyan") if self._is_cloud_model(model.model_id, model) else Text("", style="grey58")
            )
            table.add_row(
                mark,
                model.model_id,
                model.name or "",
                str(model.context_length or ""),
                cloud_cell,
                key=model.model_id,
            )
        self._refresh_ollama_thinking_recommendation()

    def _selected_model_configs(self) -> list[ModelConfig]:
        by_id = {model.model_id: model for model in self._all_models}
        return [by_id[model_id] for model_id in self.selected_models if model_id in by_id]

    @staticmethod
    def _is_cloud_model(model_id: str, model: ModelConfig | None = None) -> bool:
        """Heuristic for Ollama Cloud models, which carry a ``-cloud``/``:cloud`` tag.

        These run on Ollama's hosted infrastructure (proxied through a signed-in
        local daemon) rather than local VRAM, so they are flagged for the user.
        """
        lowered = (model_id or "").lower()
        if lowered.endswith("-cloud") or lowered.endswith(":cloud") or "-cloud" in lowered:
            return True
        raw = model.raw_metadata if model is not None else {}
        if isinstance(raw, dict) and (raw.get("remote") or raw.get("cloud")):
            return True
        return False

    def _normalize_selected_ollama_cloud_models(self) -> None:
        """Migrate old cached cloud IDs to the current runnable Ollama tag."""
        if self.provider != "ollama":
            return
        normalized: list[str] = []
        for model_id in self.selected_models:
            next_id = model_id
            name = self._model_names.get(model_id)
            if name and self._is_cloud_model(model_id):
                next_id = _to_cloud_tag(name)
                self._model_names.setdefault(next_id, name)
            if next_id not in normalized:
                normalized.append(next_id)
        self.selected_models = normalized

    def _selected_cloud_count(self) -> int:
        by_id = {model.model_id: model for model in self._all_models}
        return sum(1 for model_id in self.selected_models if self._is_cloud_model(model_id, by_id.get(model_id)))

    @staticmethod
    def _model_recommends_disabled_thinking(model_id: str, model: ModelConfig | None = None) -> bool:
        lowered = model_id.lower()
        if "gemma4" in lowered:
            return True
        raw = model.raw_metadata if model is not None else {}
        capabilities = raw.get("capabilities") if isinstance(raw, dict) else None
        if isinstance(capabilities, list) and any(str(item).lower() == "thinking" for item in capabilities):
            return True
        details = raw.get("details") if isinstance(raw, dict) else None
        family = details.get("family") if isinstance(details, dict) else None
        return isinstance(family, str) and "gemma4" in family.lower()

    def _refresh_ollama_thinking_recommendation(self) -> None:
        try:
            target = self.query_one("#ollama-thinking-recommendation", Static)
        except Exception:
            return
        by_id = {model.model_id: model for model in self._all_models}
        recommended_models = [
            model_id
            for model_id in self.selected_models
            if self._model_recommends_disabled_thinking(model_id, by_id.get(model_id))
        ]
        if self.provider != "ollama":
            target.update("Used only for Ollama runs; OpenRouter runs ignore this setting.")
            return
        if recommended_models:
            state = "ON" if self.disable_ollama_thinking else "OFF"
            target.update(
                Text(
                    "Recommendation: keep Disable Ollama thinking ON for "
                    f"{', '.join(recommended_models[:3])}"
                    f"{'...' if len(recommended_models) > 3 else ''}. Current setting: {state}.",
                    style="bold yellow",
                )
            )
            return
        target.update("Recommended for Gemma 4 and other thinking-capable Ollama models.")

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
        provider = data.get("provider")
        if isinstance(provider, str):
            try:
                self.provider = normalize_provider(provider)
            except ValueError:
                pass
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
            try:
                self._run_settings[key] = caster(self.query_one(f"#{widget_id}", Input).value)
            except Exception:
                pass

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
            try:
                self.query_one(f"#{widget_id}", Input).value = str(self._run_settings[key])
            except Exception:
                pass
        try:
            self.query_one("#sample-per-class", Input).disabled = self.run_mode != "pilot"
        except Exception:
            pass

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
        try:
            self.query_one("#start-run", Button).disabled = busy or not ready
        except Exception:
            pass
        try:
            self.query_one("#run-baselines", Button).disabled = busy
        except Exception:
            pass
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
            try:
                self.query_one(button_id, Button).disabled = not enabled
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
            try:
                table.update_cell_at(Coordinate(row_index, column_index), value)
            except Exception:
                pass

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
        elif button_id == "show-cloud-catalog":
            await self._show_cloud_catalog()
        elif button_id == "fetch-loaded":
            await self._fetch_loaded_models()
        elif button_id == "save-prompt":
            self._save_prompt_from_ui()
        elif button_id == "start-run":
            await self._start_run()
        elif button_id == "run-baselines":
            self._start_baselines()
        elif button_id == "add-to-queue":
            self._add_current_to_queue()
        elif button_id == "run-queue":
            await self._start_queue()
        elif button_id == "clear-queue":
            self._clear_queue()
        elif button_id == "queue-move-up":
            self._move_queue_item(-1)
        elif button_id == "queue-move-down":
            self._move_queue_item(1)
        elif button_id == "queue-clone":
            self._clone_queue_item()
        elif button_id == "add-sweep":
            self._add_sweep_to_queue()
        elif button_id == "cancel-run":
            self._cancel_run()
        elif button_id == "refresh-runs":
            self._refresh_runs_table()  # also refreshes compare targets and leaderboard
            self._refresh_results_help()
        elif button_id == "refresh-leaderboard":
            self._refresh_leaderboard()
        elif button_id == "compare-run":
            self._compare_models_action()
        elif button_id == "export-run":
            self._export_run()
        elif button_id == "view-figures":
            self._view_figures()
        elif button_id == "open-exports":
            self._open_exports()
        elif button_id == "news-check":
            await self._check_news()
        elif button_id == "news-fetch":
            await self._fetch_news()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "sample-per-class":
            self._refresh_run_estimate()
        if event.input.id == "model-search":
            self._render_model_table()
        if event.input.id == "provider-endpoint":
            value = event.input.value.strip()
            if self.provider == "ollama":
                self.ollama_host = value or DEFAULT_OLLAMA_HOST
            else:
                self.base_url = value or DEFAULT_BASE_URL
            self._refresh_dashboard()
            self._refresh_status_bar()
            self._save_session()
        if event.input.id == "news-query":
            self.news_query = event.input.value.strip()
            self._save_session()
        if event.input.id == "news-output-dir":
            value = event.input.value.strip() or str(DEFAULT_NEWS_OUTPUT_DIR)
            self.news_output_dir = Path(value)
            self._refresh_news_summary()
            self._save_session()
        if event.input.id == "news-max-results":
            try:
                self.news_max_results = int(event.input.value.strip() or DEFAULT_NEWS_MAX_RESULTS)
            except ValueError:
                pass
            self._save_session()
        if event.input.id in _VALIDATED_INPUTS:
            self._update_validation_hint(event.input.id, event.validation_result)
            self._refresh_stepper()
            # Persist valid run settings so they survive a restart.
            if event.validation_result is None or event.validation_result.is_valid:
                self._save_session()
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
        self._save_session()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        if event.select.id == "provider":
            try:
                self.provider = normalize_provider(str(event.value))
            except ValueError as exc:
                self._notify_error(str(exc), title="Provider invalid")
                return
            try:
                endpoint = self.query_one("#provider-endpoint", Input)
                endpoint.value = self._active_endpoint()
                endpoint.placeholder = self._endpoint_placeholder()
            except Exception:
                pass
            self._all_models = []
            self._render_model_table()
            self._refresh_dashboard()
            self._refresh_run_estimate()
            self._refresh_ollama_thinking_recommendation()
            self._reset_ollama_loaded_hint()
            self._save_session()
            self._schedule_auto_fetch_models()
            return
        if event.select.id == "prompt-preset":
            preset = self.prompts.get(str(event.value))
            if preset is None:
                return
            self.query_one("#system-prompt", TextArea).text = preset.system_prompt
            self.query_one("#user-template", TextArea).text = preset.user_template
            self.query_one("#output-mode", Select).value = preset.output_mode
            self.prompt = preset
            self._refresh_prompt_preview()
            self._save_session()
        elif event.select.id == "news-topic":
            self.news_topic = str(event.value)
            self._save_session()
        elif event.select.id == "news-time-range":
            self.news_time_range = str(event.value)
            self._save_session()
        elif event.select.id == "leaderboard-scope":
            self._refresh_leaderboard()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "news-extract":
            self.news_extract = bool(event.value)
            self._save_session()
        if event.checkbox.id == "disable-ollama-thinking":
            self.disable_ollama_thinking = bool(event.value)
            self._refresh_ollama_thinking_recommendation()
            self._refresh_resource_monitor()
            self._save_session()

    def _set_news_busy(self, busy: bool) -> None:
        self._news_in_progress = busy
        for button_id in ("#news-check", "#news-fetch"):
            try:
                self.query_one(button_id, Button).disabled = busy
            except Exception:
                pass

    def _news_config_from_ui(self, *, check_only: bool = False):
        try:
            query = self.query_one("#news-query", Input).value.strip()
            max_results = 1 if check_only else int(self.query_one("#news-max-results", Input).value.strip())
            topic_value = self.query_one("#news-topic", Select).value
            time_range_value = self.query_one("#news-time-range", Select).value
            if topic_value is Select.BLANK or time_range_value is Select.BLANK:
                raise ValueError("Choose a Tavily topic and time range")
            extract = False if check_only else bool(self.query_one("#news-extract", Checkbox).value)
            return make_news_fetch_config(
                query=query,
                max_results=max_results,
                topic=str(topic_value),
                time_range=str(time_range_value),
                extract=extract,
            )
        except Exception as exc:
            raise ValueError(f"News setup error: {exc}") from exc

    async def _check_news(self) -> None:
        if self._news_in_progress:
            self._notify_error("A Tavily request is already in progress.", title="Busy")
            return
        try:
            config = self._news_config_from_ui(check_only=True)
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot check")
            return
        self._set_news_busy(True)
        self._set_news_log(f"Checking Tavily with query: {config.query}")
        try:
            async with self._make_news_client() as client:
                result = await client.fetch(config)
            credits = result.search_usage.get("credits") if isinstance(result.search_usage, dict) else None
            credit_text = f", credits={credits}" if credits is not None else ""
            self._set_news_log(
                f"Tavily check OK: {len(result.records)} result(s), request_id={result.search_request_id or '-'}{credit_text}"
            )
            self._notify_info("Tavily news API check succeeded.", title="Tavily")
        except Exception as exc:
            self._notify_error(f"Tavily check failed: {exc}", title="Tavily failed")
            self._set_news_log(f"Tavily check failed: {exc}")
        finally:
            self._set_news_busy(False)

    async def _fetch_news(self) -> None:
        if self._news_in_progress:
            self._notify_error("A Tavily request is already in progress.", title="Busy")
            return
        try:
            config = self._news_config_from_ui()
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot fetch")
            return
        self._set_news_busy(True)
        self._set_news_log(f"Fetching Tavily news: {config.query}")
        try:
            async with self._make_news_client() as client:
                result = await client.fetch(config)
            paths = write_news_corpus(result, self.news_output_dir)
            failed = sum(1 for record in result.records if record.extraction_status == "failed")
            self._set_news_log(
                f"Saved {len(result.records)} article record(s), failed extractions={failed}: {paths.output_dir}"
            )
            self._set_news_log(f"JSONL: {paths.articles_jsonl}")
            self._set_news_log(f"CSV: {paths.articles_csv}")
            self._set_news_log(f"Manifest: {paths.manifest_json}")
            self._notify_info(f"Saved Tavily article corpus to {paths.output_dir}", title="News saved")
        except Exception as exc:
            self._notify_error(f"Tavily fetch failed: {exc}", title="Fetch failed")
            self._set_news_log(f"Tavily fetch failed: {exc}")
        finally:
            self._set_news_busy(False)

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

    @staticmethod
    def _current_event_row(event: DataTable.RowSelected) -> list | None:
        if event.row_key not in event.data_table.rows:
            return None
        return event.data_table.get_row(event.row_key)

    def _set_busy(self, button_id: str | None, busy: bool, *, label: str | None = None, spinner_widget: str | None = None) -> None:
        """Toggle transient busy feedback: disable+relabel a button and spin a widget."""
        if button_id is not None:
            try:
                button = self.query_one(f"#{button_id}", Button)
                button.disabled = busy
                if label is not None:
                    if busy:
                        self._button_labels.setdefault(button_id, button.label)
                        button.label = label
                    else:
                        button.label = self._button_labels.pop(button_id, button.label)
            except Exception:
                pass
        if spinner_widget is not None:
            try:
                self.query_one(f"#{spinner_widget}").loading = busy
            except Exception:
                pass

    @contextmanager
    def _busy(self, button_id: str | None = None, *, label: str | None = None, spinner_widget: str | None = None):
        """Show busy feedback for the duration of a block (for awaited/async work)."""
        self._set_busy(button_id, True, label=label, spinner_widget=spinner_widget)
        try:
            yield
        finally:
            self._set_busy(button_id, False, label=label, spinner_widget=spinner_widget)

    def _expand_sections(self, *section_ids: str) -> None:
        """Open the given Collapsible sections, ignoring any not currently mounted."""
        for section_id in section_ids:
            try:
                self.query_one(f"#{section_id}", Collapsible).collapsed = False
            except Exception:
                pass

    @staticmethod
    def _cycle_sort(current: tuple[int, bool] | None, index: int, ascending_first: bool) -> tuple[int, bool]:
        """Pick the next sort state: toggle direction on the same column, else start fresh."""
        if current is not None and current[0] == index:
            return (index, not current[1])
        return (index, ascending_first)

    def _update_sort_help(self, help_id: str, base_text: str, sort_state: tuple[int, bool] | None, columns: dict) -> None:
        try:
            widget = self.query_one(f"#{help_id}", Static)
        except Exception:
            return
        if sort_state is None:
            widget.update(base_text)
            return
        index, ascending = sort_state
        meta = columns.get(index)
        if meta is None:
            widget.update(base_text)
            return
        arrow = "▲" if ascending else "▼"
        widget.update(f"{base_text}  ·  Sorted by {meta[0]} {arrow}")

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        table_id = event.data_table.id
        index = event.column_index
        if table_id == "runs-table":
            meta = _RUNS_SORT_COLUMNS.get(index)
            if meta is None:
                return
            self._runs_sort = self._cycle_sort(self._runs_sort, index, meta[1])
            self._refresh_runs_table()
        elif table_id == "leaderboard-table":
            meta = _LEADERBOARD_SORT_COLUMNS.get(index)
            if meta is None:
                return
            self._leaderboard_sort = self._cycle_sort(self._leaderboard_sort, index, meta[1])
            self._refresh_leaderboard()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Textual can deliver a selection event after a table has been redrawn.
        if event.row_key not in event.data_table.rows:
            return
        table_id = event.data_table.id
        if table_id == "model-table":
            selected_row = self._current_event_row(event)
            if not selected_row:
                return
            model_id = str(selected_row[1]) if len(selected_row) >= 2 else ""
            name = str(selected_row[2]) if len(selected_row) >= 3 else ""
            was_selected = model_id in self.selected_models
            self._toggle_model(model_id, name or None)
            action = "Removed" if was_selected else "Added"
            self._set_monitor(f"{action} model: {model_id}")
        elif table_id == "selected-table":
            selected_row = self._current_event_row(event)
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
                self._show_confusion(metric)
                self._active_metric_model = str(metric.get("model_id") or "") or None
                self._active_metric_scope = str(metric.get("scope") or "all")
                self._refresh_misclassifications()
                # Reveal the drill-downs the selected metric row just populated.
                self._expand_sections("perclass-section", "confusion-section", "misclassified-section")
        elif table_id == "misclassified-table":
            key = event.row_key.value
            if key is None:
                return
            row = self._misclassification_rows.get(str(key))
            if row is not None:
                self._show_misclassification_detail(row)
        elif table_id == "queue-table":
            key = event.row_key.value
            if key is not None:
                self._remove_queue_item(str(key))
        elif table_id == "leaderboard-table":
            key = event.row_key.value
            run_id = self._leaderboard_best_run.get(str(key)) if key is not None else None
            if run_id is not None:
                self._load_metrics_for(run_id)
                self._set_monitor(f"Loaded best run {run_id} for {key} from the leaderboard.")

    async def _fetch_models(self) -> None:
        with self._busy("fetch-models", label="Fetching…", spinner_widget="model-table"):
            try:
                async with make_llm_client(self.provider, base_url=self.base_url, ollama_host=self.ollama_host) as client:
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
                f"Fetched {len(models)} {self._provider_title()} models. Type in the search box to filter, press Enter on a row to toggle."
            )
        if self.provider == "ollama":
            await self._fetch_loaded_models()

    async def _show_cloud_catalog(self) -> None:
        """List Ollama Cloud models so they can be browsed and selected even when
        none are pulled locally. Fetches the live catalogue from ollama.com and
        falls back to the env override / cache / built-in list when offline."""
        if self.provider != "ollama":
            self._notify_error(
                "Ollama Cloud models run through the Ollama provider. Switch Provider to Ollama first.",
                title="Cloud catalog",
            )
            return
        with self._busy("show-cloud-catalog", label="Loading…", spinner_widget="model-table"):
            if cloud_catalog_overridden():
                catalog = ollama_cloud_catalog()
                source = "OLLAMA_CLOUD_MODELS override"
            else:
                try:
                    catalog = await fetch_ollama_cloud_models()
                    source = "ollama.com (live)"
                except Exception as exc:
                    catalog = ollama_cloud_catalog()
                    source = f"offline fallback — fetch failed: {exc}"

            existing = {model.model_id for model in self._all_models}
            added = 0
            for model in catalog:
                if model.model_id not in existing:
                    self._all_models.append(model)
                    existing.add(model.model_id)
                    added += 1
                if model.name:
                    self._model_names.setdefault(model.model_id, model.name)
            # Filter the table to the cloud entries so they are visible immediately.
            try:
                self.query_one("#model-search", Input).value = "cloud"
            except Exception:
                pass
            self._render_model_table()
            self._render_selected_table()
        self._set_monitor(
            f"Loaded {len(catalog)} Ollama Cloud model(s) from {source} ({added} new). These route through a "
            "signed-in daemon (run 'ollama signin'); no local pull needed. Press Enter on a row to select. "
            "Set OLLAMA_CLOUD_MODELS to pin a custom list."
        )

    def _set_ollama_loaded(self, text: str) -> None:
        try:
            self.query_one("#ollama-loaded", Static).update(text)
        except Exception:
            pass

    def _reset_ollama_loaded_hint(self) -> None:
        self._set_ollama_loaded(
            "Press 'Loaded in Ollama' to query /api/ps for models held in VRAM."
            if self.provider == "ollama"
            else "Ollama only — switch the provider to Ollama to see models loaded in VRAM."
        )

    @staticmethod
    def _format_vram(size_bytes: int | None) -> str:
        if not isinstance(size_bytes, (int, float)) or size_bytes <= 0:
            return "-"
        return f"{size_bytes / 1e9:.1f} GB"

    async def _fetch_loaded_models(self) -> None:
        if self.provider != "ollama":
            self._set_ollama_loaded("Ollama only — switch the provider to Ollama to see models loaded in VRAM.")
            return
        with self._busy("fetch-loaded", label="Querying…", spinner_widget="ollama-loaded"):
            try:
                async with make_llm_client("ollama", base_url=self.base_url, ollama_host=self.ollama_host) as client:
                    loaded = await client.list_loaded_models()
            except Exception as exc:
                self._set_ollama_loaded(f"Could not query Ollama /api/ps: {exc}")
                return
            if not loaded:
                self._set_ollama_loaded("No models are currently loaded in Ollama.")
                return
            lines = []
            for entry in loaded:
                vram = self._format_vram(entry.get("size_vram") or entry.get("size"))
                lines.append(f"{entry['model']} | VRAM {vram}")
            self._set_ollama_loaded("\n".join(lines))

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
        try:
            self.query_one("#cancel-run", Button).disabled = False
        except Exception:
            pass
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
            try:
                self.query_one("#cancel-run", Button).disabled = True
            except Exception:
                pass
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

    # --- Experiment queue ------------------------------------------------

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
        try:
            self.query_one("#queue-table", DataTable).move_cursor(row=target)
        except Exception:
            pass
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
        try:
            self.query_one("#cancel-run", Button).disabled = False
        except Exception:
            pass
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
            try:
                self.query_one("#cancel-run", Button).disabled = True
            except Exception:
                pass
            self._render_queue_table()  # restores the queue summary and refreshes status bar + stepper
        outcome = "cancelled" if cancelled else "finished"
        self._set_monitor(f"Experiment queue {outcome}: {completed}/{total} experiment(s) completed.")
        self._notify_info(
            f"Queue {outcome}: {completed}/{total} experiment(s) completed.",
            title="Queue",
        )
        try:
            self.bell()  # audible cue that an unattended queue has finished
        except Exception:
            pass

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
                try:
                    runs.sort(key=meta[2], reverse=not ascending)
                except Exception:
                    pass
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
                try:
                    rows.sort(key=lambda data, field=meta[2]: data[field], reverse=not ascending)
                except Exception:
                    pass
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
        try:
            self.query_one("#compare-result", Static).update(text)
        except Exception:
            pass

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
        import subprocess
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


def _setup_crash_logging() -> None:
    """Leave a trace on disk when the TUI dies overnight: a rotating app log for
    handled failures plus faulthandler output for hard crashes."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(_LOG_PATH, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level in (logging.NOTSET, logging.WARNING):
        root.setLevel(logging.WARNING)
    logging.getLogger("sentiment_benchmark").setLevel(logging.INFO)
    # Kept open for the process lifetime so faulthandler can write during a crash.
    crash_file = open(_CRASH_LOG_PATH, "a", encoding="utf-8")  # noqa: SIM115
    faulthandler.enable(file=crash_file)


def main() -> None:
    _setup_crash_logging()
    logger.info("TUI starting")
    try:
        SentimentBenchmarkApp().run()
    except Exception:
        logger.exception("TUI exited with an unhandled exception")
        raise
    logger.info("TUI exited normally")
