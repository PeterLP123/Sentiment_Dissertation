from pathlib import Path

from typer.testing import CliRunner

from sentiment_benchmark.cli import app
from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord, RunConfig
from sentiment_benchmark.news_source import NewsArticleRecord, NewsFetchResult, article_record_id, make_news_fetch_config, normalize_url
from sentiment_benchmark.prompts import make_prompt
from sentiment_benchmark.storage import BenchmarkStore

runner = CliRunner()


def _record(prompt_hash: str, row_number: int, model_id: str, label: str) -> LLMResponseRecord:
    return LLMResponseRecord(
        row_number=row_number,
        model_id=model_id,
        prompt_hash=prompt_hash,
        raw_content=label,
        normalized_label=label,
        parse_status="valid",
        status="success",
        latency_ms=10,
        total_tokens=2,
    )


def _seed(tmp_path: Path) -> tuple[Path, int]:
    db_path = tmp_path / "cli.sqlite"
    store = BenchmarkStore(db_path)
    store.initialize()
    rows = [
        DatasetRow(1, "a", "positive", False, False, 1),
        DatasetRow(2, "b", "negative", False, False, 1),
        DatasetRow(3, "c", "neutral", False, False, 1),
    ]
    store.upsert_dataset(rows, str(tmp_path / "data.csv"))
    prompt = make_prompt("t", "sys", "{sentence}", "label_only")
    store.save_prompt(prompt)
    config = RunConfig(
        models=["model-a", "model-b"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(tmp_path / "data.csv"),
        db_path=str(db_path),
        base_url="x",
        sample_per_class=1,
    )
    run_id = store.create_run(config, [1, 2, 3])
    for row_number, label in {1: "positive", 2: "negative", 3: "neutral"}.items():
        store.save_response(run_id, _record(prompt.prompt_hash, row_number, "model-a", label))
    for row_number, label in {1: "positive", 2: "positive", 3: "neutral"}.items():
        store.save_response(run_id, _record(prompt.prompt_hash, row_number, "model-b", label))
    store.mark_run_complete(run_id)
    for model_id in ("model-a", "model-b"):
        store.save_metrics(run_id, evaluate_responses(rows, store.fetch_responses(run_id, model_id), model_id, scope="primary"))
    return db_path, run_id


def test_cli_runs_lists_seeded_run(tmp_path: Path) -> None:
    db_path, run_id = _seed(tmp_path)
    result = runner.invoke(app, ["runs", "--db-path", str(db_path)])
    assert result.exit_code == 0
    assert "Benchmark Runs" in result.output
    assert "model-a, model-b" in result.output


def test_cli_runs_handles_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.sqlite"
    BenchmarkStore(db_path).initialize()
    result = runner.invoke(app, ["runs", "--db-path", str(db_path)])
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_cli_results_shows_metrics_and_confusion(tmp_path: Path) -> None:
    db_path, run_id = _seed(tmp_path)
    result = runner.invoke(app, ["results", "--run-id", str(run_id), "--db-path", str(db_path), "--confusion"])
    assert result.exit_code == 0
    assert f"Run {run_id} Metrics" in result.output
    assert "Confusion matrix" in result.output


def test_cli_results_missing_run_exits_nonzero(tmp_path: Path) -> None:
    db_path, _ = _seed(tmp_path)
    result = runner.invoke(app, ["results", "--run-id", "999", "--db-path", str(db_path)])
    assert result.exit_code == 1
    assert "No metrics found" in result.output


def test_cli_compare_reports_mcnemar_and_cis(tmp_path: Path) -> None:
    db_path, run_id = _seed(tmp_path)
    result = runner.invoke(
        app,
        [
            "compare",
            "--run-a", str(run_id),
            "--model-a", "model-a",
            "--model-b", "model-b",
            "--db-path", str(db_path),
            "--n-resamples", "200",
        ],
    )
    assert result.exit_code == 0
    assert "Comparison" in result.output
    assert "McNemar p =" in result.output


def test_cli_compare_no_overlap_exits_nonzero(tmp_path: Path) -> None:
    db_path, run_id = _seed(tmp_path)
    result = runner.invoke(
        app,
        ["compare", "--run-a", str(run_id), "--model-a", "model-a", "--model-b", "ghost", "--db-path", str(db_path)],
    )
    assert result.exit_code == 1
    assert "No overlapping" in result.output


def test_cli_compare_rejects_bad_metric(tmp_path: Path) -> None:
    db_path, run_id = _seed(tmp_path)
    result = runner.invoke(
        app,
        ["compare", "--run-a", str(run_id), "--model-a", "model-a", "--model-b", "model-b",
         "--metric", "bogus", "--db-path", str(db_path)],
    )
    assert result.exit_code != 0


def _news_result(query: str = "market news") -> NewsFetchResult:
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
                article_text="Full article text",
                extraction_status="success",
            )
        ],
        search_request_id="search-1",
        search_usage={"credits": 1},
    )


class FakeNewsClient:
    def __init__(self) -> None:
        self.configs = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def fetch(self, config):
        self.configs.append(config)
        return _news_result(config.query)


def test_cli_news_check_uses_tavily_client(monkeypatch) -> None:
    fake = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: fake)

    result = runner.invoke(app, ["news-check", "--query", "financial markets", "--max-results", "1"])

    assert result.exit_code == 0
    assert "Tavily News Check" in result.output
    assert "search-1" in result.output
    assert fake.configs[0].extract is False


def test_cli_fetch_news_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    fake = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: fake)
    output_dir = tmp_path / "news"

    result = runner.invoke(
        app,
        [
            "fetch-news",
            "--query", "bank earnings sentiment",
            "--max-results", "1",
            "--output-dir", str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Tavily News Fetch" in result.output
    exported = list(output_dir.glob("tavily_news_*"))
    assert len(exported) == 1
    assert (exported[0] / "articles.jsonl").exists()
    assert (exported[0] / "articles.csv").exists()
    assert (exported[0] / "manifest.json").exists()


def test_cli_fetch_news_rejects_invalid_topic(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "fetch-news",
            "--query", "bank earnings sentiment",
            "--topic", "bogus",
            "--output-dir", str(tmp_path / "news"),
        ],
    )

    assert result.exit_code != 0
    assert "topic must be one of" in result.output
