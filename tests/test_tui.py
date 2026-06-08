import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from textual.widgets import Button, Checkbox, DataTable, Input, ProgressBar, Static

from sentiment_benchmark.baseline_runner import BaselineRunSummary
from sentiment_benchmark.models import DatasetRow, EvaluationResult, LLMResponseRecord, ModelConfig, PromptConfig
from sentiment_benchmark.news_source import NewsArticleRecord, NewsFetchResult, article_record_id, make_news_fetch_config, normalize_url
from sentiment_benchmark.storage import BenchmarkStore
from sentiment_benchmark.tui import ConfirmScreen, SentimentBenchmarkApp


def _make_app(tmp_path: Path) -> SentimentBenchmarkApp:
    app = SentimentBenchmarkApp()
    app._session_path = tmp_path / "tui_session.json"
    app.provider = "openrouter"
    app.selected_models = []
    app._model_names = {}
    return app


def _seed_run_with_metrics(db_path: Path, model_id: str = "openai/test-model") -> int:
    store = BenchmarkStore(db_path)
    store.initialize()
    prompt = PromptConfig(
        prompt_id="default_label_only",
        system_prompt="sys",
        user_template="{sentence}",
        output_mode="label_only",
        prompt_hash="hash123456",
    )
    store.save_prompt(prompt)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs (
                created_at, mode, prompt_hash, output_mode, models_json, selected_rows_json,
                dataset_path, base_url, request_json, status, completed_at
            ) VALUES (
                '2026-06-04T12:00:00+00:00', 'pilot', ?, 'label_only', ?, '[]',
                'Data/data.csv', 'https://openrouter.ai/api/v1', '{}', 'completed', '2026-06-04T12:01:00+00:00'
            )
            """,
            (prompt.prompt_hash, json.dumps([model_id])),
        )
        run_id = int(cursor.lastrowid)
        connection.commit()
    result = EvaluationResult(
        model_id=model_id,
        scope="primary",
        row_count=90,
        accuracy=0.8,
        balanced_accuracy=0.78,
        mcc=0.7,
        macro_f1=0.79,
        weighted_f1=0.81,
        per_class={
            "positive": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 30.0},
            "negative": {"precision": 0.7, "recall": 0.75, "f1": 0.72, "support": 30.0},
            "neutral": {"precision": 0.75, "recall": 0.7, "f1": 0.72, "support": 30.0},
        },
        confusion_matrix={},
        invalid_output_count=2,
        api_error_count=1,
        mean_latency_ms=120.0,
        total_prompt_tokens=900,
        total_completion_tokens=90,
        total_tokens=990,
    )
    store.save_metrics(run_id, result)
    return run_id


def _seed_run_with_misclassifications(db_path: Path, model_id: str = "openai/test-model") -> int:
    run_id = _seed_run_with_metrics(db_path, model_id=model_id)
    store = BenchmarkStore(db_path)
    store.upsert_dataset(
        [
            DatasetRow(1, "Share price rose after earnings.", "positive", False, False, 1),
            DatasetRow(2, "Guidance was cut sharply.", "negative", False, False, 1),
            DatasetRow(3, "The company opened a new office.", "neutral", False, False, 1),
        ],
        "Data/data.csv",
    )
    store.save_response(
        run_id,
        LLMResponseRecord(
            row_number=1,
            model_id=model_id,
            prompt_hash="hash123456",
            raw_content="positive",
            normalized_label="positive",
            parse_status="valid",
            status="success",
        ),
    )
    store.save_response(
        run_id,
        LLMResponseRecord(
            row_number=2,
            model_id=model_id,
            prompt_hash="hash123456",
            raw_content="positive",
            normalized_label="positive",
            parse_status="valid",
            status="success",
        ),
    )
    store.save_response(
        run_id,
        LLMResponseRecord(
            row_number=3,
            model_id=model_id,
            prompt_hash="hash123456",
            raw_content="I think this is mixed.",
            normalized_label=None,
            parse_status="invalid",
            status="success",
        ),
    )
    return run_id


def test_tui_explains_run_fields_and_updates_estimate(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            run_estimate = app.query_one("#run-estimate", Static)
            dashboard = app.query_one("#dashboard", Static)

            assert "Run estimate" in str(run_estimate.content)
            assert "no models are selected" in str(run_estimate.content)
            statics = "\n".join(str(w.content) for w in app.query(Static) if hasattr(w, "content"))
            assert "Only Sentence text is sent" in statics
            assert "OpenRouter API key" in str(dashboard.content)

            app.selected_models.append("test/model")
            app.query_one("#sample-per-class", Input).value = "5"
            app._refresh_run_estimate()
            assert "15 request(s) per model" in str(run_estimate.content)

    asyncio.run(scenario())


def test_tui_model_table_toggles_selection(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._all_models = [
                ModelConfig(model_id="provider/model", name="Readable Name", context_length=4096)
            ]
            app._render_model_table()

            table = app.query_one("#model-table", DataTable)
            row_key = next(iter(table.rows))
            event = DataTable.RowSelected(table, 0, row_key)
            app.on_data_table_row_selected(event)

            assert app.selected_models == ["provider/model"]
            selected_table = app.query_one("#selected-table", DataTable)
            assert selected_table.row_count == 1
            summary = app.query_one("#selected-summary", Static)
            assert "1 model(s) selected" in str(summary.content)

            # Toggling again removes
            row_key = next(iter(table.rows))
            event = DataTable.RowSelected(table, 0, row_key)
            app.on_data_table_row_selected(event)
            assert app.selected_models == []
            assert app.query_one("#selected-table", DataTable).row_count == 0

    asyncio.run(scenario())


def test_tui_model_search_filters_table(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._all_models = [
                ModelConfig(model_id="openai/gpt-4o-mini", name="GPT-4o mini"),
                ModelConfig(model_id="anthropic/claude-sonnet-4", name="Claude Sonnet 4"),
            ]
            app._render_model_table()
            assert app.query_one("#model-table", DataTable).row_count == 2

            app.query_one("#model-search", Input).value = "claude"
            await pilot.pause(0.05)
            assert app.query_one("#model-table", DataTable).row_count == 1

            app.query_one("#model-search", Input).value = ""
            await pilot.pause(0.05)
            assert app.query_one("#model-table", DataTable).row_count == 2

    asyncio.run(scenario())


def test_tui_persists_session_between_launches(tmp_path: Path) -> None:
    session_path = tmp_path / "tui_session.json"

    async def first_launch() -> None:
        app = SentimentBenchmarkApp()
        app._session_path = session_path
        app.selected_models = []
        app._model_names = {}
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._select_model("openai/gpt-4o-mini", name="GPT-4o mini")

    asyncio.run(first_launch())

    saved = json.loads(session_path.read_text())
    assert saved["selected_models"] == ["openai/gpt-4o-mini"]
    assert saved["model_names"]["openai/gpt-4o-mini"] == "GPT-4o mini"

    async def second_launch() -> None:
        app = SentimentBenchmarkApp()
        app._session_path = session_path
        app._load_session()
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            assert app.selected_models == ["openai/gpt-4o-mini"]
            selected_table = app.query_one("#selected-table", DataTable)
            assert selected_table.row_count == 1

    asyncio.run(second_launch())


def test_tui_runs_table_loads_metrics_and_per_class(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.sqlite"
    run_id = _seed_run_with_metrics(db_path)

    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.db_path = db_path
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._refresh_runs_table()

            runs_table = app.query_one("#runs-table", DataTable)
            assert runs_table.row_count == 1

            run_key = next(iter(runs_table.rows))
            app.on_data_table_row_selected(DataTable.RowSelected(runs_table, 0, run_key))
            assert app._active_run_id == run_id

            metrics_table = app.query_one("#metrics-table", DataTable)
            assert metrics_table.row_count == 1

            metric_key = next(iter(metrics_table.rows))
            app.on_data_table_row_selected(DataTable.RowSelected(metrics_table, 0, metric_key))

            perclass = app.query_one("#perclass-table", DataTable)
            assert perclass.row_count == 3

    asyncio.run(scenario())


def test_tui_metrics_table_shows_latency_tokens_and_cost(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.sqlite"
    run_id = _seed_run_with_metrics(db_path)
    store = BenchmarkStore(db_path)
    store.save_generation_metadata(run_id, 1, "openai/test-model", "gen-1", {"total_cost": 0.0025})

    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.db_path = db_path
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._load_metrics_for(run_id)

            metrics_table = app.query_one("#metrics-table", DataTable)
            assert len(metrics_table.columns) == 10
            assert metrics_table.row_count == 1

            row = metrics_table.get_row("openai/test-model|primary")
            rendered = [str(cell) for cell in row]
            assert "990" in rendered  # total tokens
            assert "120 ms" in rendered  # mean latency
            assert "$0.0025" in rendered  # observed cost

    asyncio.run(scenario())


def test_tui_metric_selection_populates_confusion_matrix(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.sqlite"
    run_id = _seed_run_with_metrics(db_path)
    # Replace the seeded (empty-matrix) primary metric with one carrying a confusion matrix.
    store = BenchmarkStore(db_path)
    store.save_metrics(
        run_id,
        EvaluationResult(
            model_id="openai/test-model",
            scope="primary",
            row_count=4,
            accuracy=0.75,
            balanced_accuracy=0.66,
            mcc=0.6,
            macro_f1=0.7,
            weighted_f1=0.72,
            per_class={"positive": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2.0}},
            confusion_matrix={
                "positive": {"positive": 2, "negative": 0, "neutral": 0, "__invalid__": 0, "__error__": 0},
                "negative": {"positive": 1, "negative": 1, "neutral": 0, "__invalid__": 0, "__error__": 0},
                "neutral": {"positive": 0, "negative": 0, "neutral": 0, "__invalid__": 1, "__error__": 0},
            },
            invalid_output_count=1,
            api_error_count=0,
            mean_latency_ms=100.0,
            total_prompt_tokens=4,
            total_completion_tokens=4,
            total_tokens=8,
        ),
    )

    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.db_path = db_path
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._load_metrics_for(run_id)

            metrics_table = app.query_one("#metrics-table", DataTable)
            metric_key = next(iter(metrics_table.rows))
            app.on_data_table_row_selected(DataTable.RowSelected(metrics_table, 0, metric_key))

            confusion = app.query_one("#confusion-table", DataTable)
            assert confusion.row_count == 3
            first_row = [str(cell) for cell in confusion.get_row_at(0)]
            assert first_row[0] == "positive"
            assert first_row[1] == "2"  # positive predicted as positive

    asyncio.run(scenario())


def test_tui_loads_misclassified_rows_and_details(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.sqlite"
    run_id = _seed_run_with_misclassifications(db_path)

    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.db_path = db_path
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._load_metrics_for(run_id)

            misclassified = app.query_one("#misclassified-table", DataTable)
            assert misclassified.row_count == 2

            row_key = next(iter(misclassified.rows))
            app.on_data_table_row_selected(DataTable.RowSelected(misclassified, 0, row_key))

            detail = app.query_one("#misclassified-detail", Static)
            assert "Actual: negative" in str(detail.content)
            assert "Guidance was cut sharply." in str(detail.content)

    asyncio.run(scenario())


def test_tui_stepper_blocks_start_until_models_selected(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            stepper = app.query_one("#run-stepper", Static)
            start = app.query_one("#start-run", Button)
            assert "[  ] 1 Models" in str(stepper.content)
            assert "[  ] 4 Start" in str(stepper.content)
            assert start.disabled is True

            app._select_model("openai/gpt-4o-mini")
            await pilot.pause(0.05)
            stepper = app.query_one("#run-stepper", Static)
            assert "[OK] 1 Models" in str(stepper.content)
            assert "[OK] 4 Start" in str(stepper.content)
            assert app.query_one("#start-run", Button).disabled is False

    asyncio.run(scenario())


def test_tui_settings_validation_disables_start(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._select_model("openai/gpt-4o-mini")
            await pilot.pause(0.05)
            assert app.query_one("#start-run", Button).disabled is False

            app.query_one("#concurrency", Input).value = "0"
            await pilot.pause(0.05)
            stepper = app.query_one("#run-stepper", Static)
            assert "[  ] 3 Settings" in str(stepper.content)
            assert app.query_one("#start-run", Button).disabled is True

            app.query_one("#concurrency", Input).value = "2"
            await pilot.pause(0.05)
            stepper = app.query_one("#run-stepper", Static)
            assert "[OK] 3 Settings" in str(stepper.content)
            assert app.query_one("#start-run", Button).disabled is False

    asyncio.run(scenario())


def test_tui_confirm_threshold(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            assert app._confirm_message("pilot", 90, ["m1"]) is None
            assert app._confirm_message("full", 5842, ["m1"]) is not None
            assert app._confirm_message("pilot", 501, ["m1", "m2", "m3"]) is not None
            message = app._confirm_message("full", 5842, ["m1", "m2"])
            assert message is not None
            assert "11684" in message

    asyncio.run(scenario())


def test_tui_confirm_message_includes_known_completion_cost(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app._all_models = [
        ModelConfig(model_id="priced/model", pricing={"completion": "0.000001"}),
        ModelConfig(model_id="missing/model", pricing={}),
    ]

    message = app._confirm_message("full", 10, ["priced/model", "missing/model"], max_completion_tokens=100)

    assert message is not None
    assert "Estimated completion-token cost ceiling: $0.0010" in message
    assert "Pricing unavailable for 1 model(s)" in message


def test_tui_full_run_confirmation_uses_callback(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.selected_models = ["openai/gpt-4o-mini"]
        app.run_mode = "full"
        started_modes: list[str] = []

        def fake_begin_run(config) -> None:
            started_modes.append(config.mode)

        app._begin_run = fake_begin_run  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._start_run()
            await pilot.pause(0.05)

            assert isinstance(app.screen, ConfirmScreen)
            assert app._confirmation_pending is True
            app.screen.dismiss(True)
            await pilot.pause(0.05)

            assert started_modes == ["full"]
            assert app._confirmation_pending is False

    asyncio.run(scenario())


def test_tui_blocks_duplicate_start_while_confirmation_pending(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.selected_models = ["openai/gpt-4o-mini"]
        app.run_mode = "full"
        started_modes: list[str] = []

        def fake_begin_run(config) -> None:
            started_modes.append(config.mode)

        app._begin_run = fake_begin_run  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._start_run()
            await pilot.pause(0.05)
            first_screen = app.screen

            await app._start_run()
            await pilot.pause(0.05)

            assert app.screen is first_screen
            assert app._confirmation_pending is True
            assert app.query_one("#start-run", Button).disabled is True
            assert started_modes == []
            assert any("confirmation is already open" in message for _, message in app.notifications)

    asyncio.run(scenario())


def test_tui_cancel_confirmation_clears_pending_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.selected_models = ["openai/gpt-4o-mini"]
        app.run_mode = "full"
        started_modes: list[str] = []

        def fake_begin_run(config) -> None:
            started_modes.append(config.mode)

        app._begin_run = fake_begin_run  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._start_run()
            await pilot.pause(0.05)

            assert isinstance(app.screen, ConfirmScreen)
            app.screen.dismiss(False)
            await pilot.pause(0.05)

            assert app._confirmation_pending is False
            assert app.query_one("#start-run", Button).disabled is False
            assert started_modes == []
            assert app.monitor_lines[-1] == "Run cancelled before it started."

    asyncio.run(scenario())


def test_tui_handles_run_events_into_progress_table(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._handle_run_event({"type": "run_started", "run_id": 1, "models": ["m1", "m2"], "rows_per_model": 2})
            progress_table = app.query_one("#run-progress", DataTable)
            assert progress_table.row_count == 2
            bar = app.query_one("#run-progress-bar", ProgressBar)
            assert bar.total == 4

            app._handle_run_event({"type": "model_started", "model_id": "m1", "total_rows": 2})
            app._handle_run_event({
                "type": "row_completed",
                "model_id": "m1",
                "row_number": 1,
                "status": "success",
                "latency_ms": 50.0,
            })
            app._handle_run_event({
                "type": "row_completed",
                "model_id": "m1",
                "row_number": 2,
                "status": "api_error",
                "latency_ms": 100.0,
            })
            assert app._progress["m1"]["done"] == 2
            assert app._progress["m1"]["errors"] == 1
            assert app._progress["m1"]["latency_count"] == 2

            app._handle_run_event({"type": "model_completed", "model_id": "m1", "accuracy": 0.5})
            assert "done" in app._progress["m1"]["status"]

    asyncio.run(scenario())


def test_tui_export_run_requires_active_run(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.sqlite"
    _seed_run_with_metrics(db_path)

    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.db_path = db_path
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._active_run_id = None
            app._export_run()
            assert any(
                severity == "error" and "Select a run" in message
                for severity, message in app.notifications
            )

    asyncio.run(scenario())


def test_tui_view_figures_requires_active_run(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._active_run_id = None
            app._view_figures()
            assert any(
                severity == "error" and "Select a run" in message
                for severity, message in app.notifications
            )

    asyncio.run(scenario())


def test_tui_view_figures_opens_generated_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    figures_dir = tmp_path / "run_1" / "figures"
    figures_dir.mkdir(parents=True)
    leaderboard = figures_dir / "leaderboard.png"
    leaderboard.write_bytes(b"fake-png")

    def fake_export(db_path: object, run_id: int, output_dir: object = None) -> list[Path]:
        return [tmp_path / "run_1" / "summary.md", leaderboard]

    monkeypatch.setattr("sentiment_benchmark.tui.export_run", fake_export)
    opened: dict[str, Path] = {}

    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._active_run_id = 1

            def fake_open(path: Path) -> bool:
                opened["path"] = path
                return True

            monkeypatch.setattr(app, "_open_path", fake_open)
            app._view_figures()

            assert opened.get("path") == figures_dir
            assert any(
                severity == "information" and "figure" in message.lower()
                for severity, message in app.notifications
            )

    asyncio.run(scenario())


def test_tui_view_figures_without_plots_hints_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_export(db_path: object, run_id: int, output_dir: object = None) -> list[Path]:
        return [tmp_path / "run_1" / "summary.md"]  # no .png figures

    monkeypatch.setattr("sentiment_benchmark.tui.export_run", fake_export)

    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._active_run_id = 1
            app._view_figures()
            assert any(
                severity == "error" and ".[figures]" in message
                for severity, message in app.notifications
            )

    asyncio.run(scenario())


def test_tui_keyboard_switches_tabs(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            from textual.widgets import TabbedContent

            tabs = app.query_one(TabbedContent)
            assert tabs.active == "dashboard-tab"

            await pilot.press("4")
            await pilot.pause(0.05)
            assert tabs.active == "run-tab"

            await pilot.press("2")
            await pilot.pause(0.05)
            assert tabs.active == "models-tab"

    asyncio.run(scenario())


def test_tui_help_action_pushes_help_screen(tmp_path: Path) -> None:
    from sentiment_benchmark.tui import HelpScreen

    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app.action_show_help()
            await pilot.pause(0.05)
            assert isinstance(app.screen, HelpScreen)

    asyncio.run(scenario())


def test_tui_refresh_action_notifies(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app.action_refresh()
            assert any(
                severity == "information" and "Refreshed" in message
                for severity, message in app.notifications
            )

    asyncio.run(scenario())


def test_tui_baseline_checkboxes_default_to_sklearn(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            assert app.query_one("#baseline-majority", Checkbox).value is True
            assert app.query_one("#baseline-tfidf_logreg", Checkbox).value is True
            assert set(app._selected_baselines()) == {"majority", "tfidf_logreg"}

    asyncio.run(scenario())


def test_tui_run_baselines_button_invokes_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_run_baselines(names: list[str], **kwargs: object) -> BaselineRunSummary:
        calls["names"] = list(names)
        calls["kwargs"] = kwargs
        callback = kwargs.get("callback")
        if callable(callback):
            callback("baseline progress")
        return BaselineRunSummary(run_id=7, selected_row_count=90, baseline_count=len(names))

    monkeypatch.setattr("sentiment_benchmark.tui.run_baselines", fake_run_baselines)

    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.db_path = tmp_path / "bench.sqlite"
        app.dataset_path = Path("Data/data.csv")
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._start_baselines()
            for _ in range(80):
                if "names" in calls and not app._baseline_in_progress:
                    break
                await pilot.pause(0.05)
            assert calls.get("names") == ["majority", "tfidf_logreg"]
            kwargs = calls["kwargs"]
            assert isinstance(kwargs, dict)
            assert kwargs["mode"] == "pilot"
            assert str(kwargs["db_path"]) == str(app.db_path)

    asyncio.run(scenario())


def _tui_news_result(query: str = "market news", *, failed: bool = False) -> NewsFetchResult:
    normalized = normalize_url("https://example.com/article")
    return NewsFetchResult(
        config=make_news_fetch_config(query=query, max_results=1),
        fetched_at="2026-06-07T12:00:00+00:00",
        records=[
            NewsArticleRecord(
                record_id=article_record_id(normalized),
                url="https://example.com/article",
                normalized_url=normalized,
                title="Market article",
                snippet="Snippet",
                article_text=None if failed else "Full article text",
                extraction_status="failed" if failed else "success",
                extract_error="blocked" if failed else None,
            )
        ],
        search_request_id="search-1",
        search_usage={"credits": 1},
    )


class FakeTuiNewsClient:
    def __init__(self, *, failed: bool = False) -> None:
        self.failed = failed
        self.configs = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def fetch(self, config):
        self.configs.append(config)
        return _tui_news_result(config.query, failed=self.failed)


def test_tui_news_tab_renders_controls(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            from textual.widgets import TabbedContent

            tabs = app.query_one(TabbedContent)
            await pilot.press("6")
            await pilot.pause(0.05)

            assert tabs.active == "news-tab"
            assert app.query_one("#news-query", Input).value == "financial markets"
            assert app.query_one("#news-extract", Checkbox).value is True
            assert "Tavily API key" in str(app.query_one("#news-summary", Static).content)

    asyncio.run(scenario())


def test_tui_news_check_uses_client_and_logs_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        fake = FakeTuiNewsClient()
        monkeypatch.setattr(app, "_make_news_client", lambda: fake)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._check_news()

            assert fake.configs[0].extract is False
            assert any("Tavily check OK" in line for line in app.news_lines)
            assert any("succeeded" in message for _, message in app.notifications)

    asyncio.run(scenario())


def test_tui_news_fetch_writes_outputs_and_logs_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        app.news_output_dir = tmp_path / "news"
        fake = FakeTuiNewsClient(failed=True)
        monkeypatch.setattr(app, "_make_news_client", lambda: fake)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._fetch_news()

            assert fake.configs[0].extract is True
            assert any("failed extractions=1" in line for line in app.news_lines)
            assert any("articles.jsonl" in line for line in app.news_lines)
            exported = list((tmp_path / "news").glob("tavily_news_*"))
            assert len(exported) == 1
            assert (exported[0] / "manifest.json").exists()

    asyncio.run(scenario())
