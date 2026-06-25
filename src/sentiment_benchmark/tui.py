from __future__ import annotations

import asyncio
import faulthandler
import importlib.util
import logging
import os
from contextlib import contextmanager, suppress
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Event as ThreadEvent

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
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

from .baselines import BASELINE_SPECS, DEFAULT_BASELINES, BaselineSpec
from .constants import (
    DEFAULT_BASE_URL,
    DEFAULT_DATASET_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_PROVIDER,
)
from .env import load_env_file
from .models import ModelConfig
from .news_source import (
    DEFAULT_NEWS_MAX_RESULTS,
    DEFAULT_NEWS_OUTPUT_DIR,
    DEFAULT_NEWS_TIME_RANGE,
    DEFAULT_NEWS_TOPIC,
    NEWS_TIME_RANGES,
    NEWS_TOPICS,
)
from .prompts import load_prompts
from .providers import endpoint_for_provider, normalize_provider
from .storage import BenchmarkStore
from .tui_base import _VALIDATED_INPUTS
from .tui_baselines import BaselinesMixin
from .tui_format import (
    _LEADERBOARD_HELP_BASE,
    _LEADERBOARD_SORT_COLUMNS,
    _RUNS_HELP_BASE,
    _RUNS_SORT_COLUMNS,
    _monitor_text,
)
from .tui_models import ModelsMixin
from .tui_monitor import MonitorMixin
from .tui_news import NewsMixin
from .tui_queue import QueueMixin
from .tui_results import ResultsMixin
from .tui_run import RunMixin
from .tui_screens import HelpScreen


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


_SESSION_PATH = Path("results/tui_session.json")
_QUEUE_PATH = Path("results/tui_queue.json")
_LOG_PATH = Path("results/tui.log")
_CRASH_LOG_PATH = Path("results/tui_crash.log")

logger = logging.getLogger(__name__)



class SentimentBenchmarkApp(
    BaselinesMixin, ModelsMixin, MonitorMixin, NewsMixin, QueueMixin, ResultsMixin, RunMixin, App
):
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
                        "Full uses every row in the selected dataset and may cost more.",
                        classes="help",
                    )
                    yield Static(id="run-estimate")
                    yield Static("Run mode", classes="field-label")
                    yield Static(
                        "Pilot is a small stratified test. Full processes every row in the selected dataset.",
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
                    "Source unlabeled news articles from Tavily into derived files. This does not modify benchmark datasets.",
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
            with suppress(Exception):
                self.query_one(panel_id, Static).border_title = title

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

    def _notify_error(self, message: str, *, title: str = "Error") -> None:
        self.notifications.append(("error", message))
        with suppress(Exception):
            self.notify(message, title=title, severity="error")

    def _notify_info(self, message: str, *, title: str = "") -> None:
        self.notifications.append(("information", message))
        with suppress(Exception):
            self.notify(message, title=title, severity="information")

    def _active_endpoint(self) -> str:
        return endpoint_for_provider(self.provider, base_url=self.base_url, ollama_host=self.ollama_host)

    def _endpoint_placeholder(self) -> str:
        return "http://desktop-pc:11434" if self.provider == "ollama" else DEFAULT_BASE_URL

    def _provider_title(self) -> str:
        return "Ollama" if self.provider == "ollama" else "OpenRouter"

    def action_show_tab(self, tab_id: str) -> None:
        with suppress(Exception):
            self.query_one(TabbedContent).active = tab_id

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
            with suppress(ValueError):
                self.news_max_results = int(event.input.value.strip() or DEFAULT_NEWS_MAX_RESULTS)
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
                with suppress(Exception):
                    self.query_one("#advanced-settings", Collapsible).collapsed = False

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id != "run-mode":
            return
        pressed_id = event.pressed.id or "mode-pilot"
        self.run_mode = pressed_id.removeprefix("mode-")
        with suppress(Exception):
            self.query_one("#sample-per-class", Input).disabled = self.run_mode != "pilot"
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
            with suppress(Exception):
                self.query_one(f"#{spinner_widget}").loading = busy

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
            with suppress(Exception):
                self.query_one(f"#{section_id}", Collapsible).collapsed = False

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
            with suppress(ValueError):
                self._load_metrics_for(int(key))
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

    async def action_run_baselines(self) -> None:
        self._start_baselines()


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
