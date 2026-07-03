import json
from pathlib import Path

from typer.testing import CliRunner

from sentiment_benchmark.cli import app
from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord, RunConfig
from sentiment_benchmark.news_source import NewsArticleRecord, NewsFetchResult, article_record_id, make_news_fetch_config, normalize_url
from sentiment_benchmark.newsapi_source import NewsApiConfigurationError
from sentiment_benchmark.prompts import make_prompt
from sentiment_benchmark.storage import BenchmarkStore

runner = CliRunner()


def test_cli_concurrency_default_is_provider_aware() -> None:
    from sentiment_benchmark.cli import _resolve_concurrency
    from sentiment_benchmark.constants import DEFAULT_CEREBRAS_CONCURRENCY

    assert _resolve_concurrency("cerebras", None) == DEFAULT_CEREBRAS_CONCURRENCY
    assert _resolve_concurrency("openrouter", None) == 1
    assert _resolve_concurrency("ollama", None) == 1
    # An explicit flag always wins over the provider default.
    assert _resolve_concurrency("cerebras", 8) == 8


def test_cli_run_shows_progress_and_metrics_summary(tmp_path: Path, monkeypatch) -> None:
    from contextlib import asynccontextmanager

    dataset = tmp_path / "data.csv"
    dataset.write_text(
        "Sentence,Sentiment\npositive example,positive\nnegative example,negative\nneutral example,neutral\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "run.sqlite"

    class StubClient:
        async def classify(self, model_id, prompt, example, temperature=0.0, max_completion_tokens=8, retries=3):
            label = example.sentence.split()[0]
            return LLMResponseRecord(
                row_number=example.row_number,
                model_id=model_id,
                prompt_hash=prompt.prompt_hash,
                raw_content=label,
                normalized_label=label,
                parse_status="valid",
                status="success",
                latency_ms=10,
            )

        async def get_generation_metadata(self, generation_id, retries=3):
            return None

    @asynccontextmanager
    async def stub_make_llm_client(*args, **kwargs):
        yield StubClient()

    monkeypatch.setattr("sentiment_benchmark.cli.make_llm_client", stub_make_llm_client)
    result = runner.invoke(
        app,
        [
            "run",
            "--models",
            "stub/model",
            "--mode",
            "pilot",
            "--sample-per-class",
            "1",
            "--dataset-path",
            str(dataset),
            "--db-path",
            str(db_path),
        ],
    )

    assert result.exit_code == 0, result.output
    # Per-row success lines are replaced by the progress display.
    assert "Saved response" not in result.output
    # The closing summary shows status, throughput, and the metrics table.
    assert "completed" in result.output
    assert "rows/min" in result.output
    assert "Metrics" in result.output
    assert "stub/model" in result.output
    assert "results --run-id" in result.output


def test_cli_run_surfaces_row_errors_and_resume_hint(tmp_path: Path, monkeypatch) -> None:
    from contextlib import asynccontextmanager

    dataset = tmp_path / "data.csv"
    dataset.write_text(
        "Sentence,Sentiment\npositive example,positive\nnegative example,negative\nneutral example,neutral\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "run.sqlite"

    class FailingStubClient:
        async def classify(self, model_id, prompt, example, temperature=0.0, max_completion_tokens=8, retries=3):
            return LLMResponseRecord(
                row_number=example.row_number,
                model_id=model_id,
                prompt_hash=prompt.prompt_hash,
                raw_content=None,
                normalized_label=None,
                parse_status="error",
                status="api_error",
                error="HTTP 404: model_not_found",
                latency_ms=5,
            )

        async def get_generation_metadata(self, generation_id, retries=3):
            return None

    @asynccontextmanager
    async def stub_make_llm_client(*args, **kwargs):
        yield FailingStubClient()

    monkeypatch.setattr("sentiment_benchmark.cli.make_llm_client", stub_make_llm_client)
    result = runner.invoke(
        app,
        [
            "run",
            "--models",
            "stub/model",
            "--mode",
            "pilot",
            "--sample-per-class",
            "1",
            "--dataset-path",
            str(dataset),
            "--db-path",
            str(db_path),
        ],
    )

    assert result.exit_code == 0, result.output
    # Rich may soft-wrap the long line; assert on the unbreakable error token.
    assert "model_not_found" in result.output
    assert "3 row(s) failed" in result.output
    assert "--resume-run-id" in result.output


def test_cli_sweep_accepts_prices_override_and_news_date_cutoff(tmp_path: Path) -> None:
    import csv as _csv

    run_dir = tmp_path / "analysis"
    run_dir.mkdir()
    dates = ["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-15", "2026-06-16", "2026-06-17"]
    with (run_dir / "daily_signals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(
            handle,
            fieldnames=[
                "symbol", "news_date", "scorer_id", "article_count", "valid_count",
                "mean_score", "signal", "signal_value", "availability_timestamp",
            ],
        )
        writer.writeheader()
        for day in dates:
            writer.writerow({
                "symbol": "AAPL", "news_date": day, "scorer_id": "llm/test", "article_count": "3",
                "valid_count": "3", "mean_score": "0.5", "signal": "positive", "signal_value": "1",
                "availability_timestamp": "",
            })
        # A signal past the price history: --max-news-date must drop it.
        writer.writerow({
            "symbol": "AAPL", "news_date": "2026-06-30", "scorer_id": "llm/test", "article_count": "1",
            "valid_count": "1", "mean_score": "0.5", "signal": "positive", "signal_value": "1",
            "availability_timestamp": "",
        })
    prices_csv = tmp_path / "external_prices.csv"
    sessions = dates + ["2026-06-18", "2026-06-19", "2026-06-22"]
    with prices_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(
            handle,
            fieldnames=["symbol", "session_date", "open", "high", "low", "close", "volume", "repaired"],
        )
        writer.writeheader()
        for day in sessions:
            writer.writerow({
                "symbol": "AAPL", "session_date": day, "open": "100.0", "high": "105.0",
                "low": "95.0", "close": "101.0", "volume": "1000.0", "repaired": "False",
            })

    result = runner.invoke(
        app,
        [
            "sweep-trading-strategy",
            "--run-dir", str(run_dir),
            "--scorer", "llm/test",
            "--prices", str(prices_csv),
            "--max-news-date", "2026-06-17",
            "--thresholds", "0.0",
            "--horizons", "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (run_dir / "sweep.csv").exists()


def test_cli_trading_strategy_dry_run_has_no_credential_dependency() -> None:
    result = runner.invoke(app, ["run-trading-strategy", "--dry-run"])
    assert result.exit_code == 0
    assert "Trading Strategy Dry Run" in result.output
    assert "AAPL, AMZN, TSLA" in result.output
    assert "Entry Rule" in result.output


def test_cli_newsapi_check_reports_missing_key(monkeypatch) -> None:
    def missing():
        raise NewsApiConfigurationError("NEWSAPI_API_KEY is required for NewsAPI sourcing")

    monkeypatch.setattr("sentiment_benchmark.cli._make_newsapi_client", missing)
    result = runner.invoke(app, ["newsapi-check"])
    assert result.exit_code != 0
    assert "NEWSAPI_API_KEY is required" in result.output


def test_cli_lseg_init_config_writes_preset(tmp_path: Path) -> None:
    output = tmp_path / "lseg.toml"

    result = runner.invoke(
        app,
        [
            "lseg-init-config",
            "--collection-id",
            "cli_lseg",
            "--start",
            "2025-06-26T00:00:00Z",
            "--end",
            "2026-06-26T00:00:00Z",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "LSEG Config Created" in result.output
    text = output.read_text()
    assert 'symbol = "GOOGL"' in text
    assert 'ric = "JPM.N"' in text
    assert "window_days = 1" in text


def test_cli_lseg_catalog_writes_metadata(tmp_path: Path) -> None:
    corpus = tmp_path / "lseg" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "config": {
                    "collection": {"id": "corpus", "start": "2025-06-26T00:00:00Z", "end": "2026-06-26T00:00:00Z"},
                    "companies": [{"symbol": "AAPL"}],
                },
                "counts": {"articles": 3, "eligible": 2, "ineligible": 1, "text_quality": {"ok": 2}},
                "sharing": {"redistribute": False, "licensed_full_text": True},
            }
        )
    )

    result = runner.invoke(app, ["lseg-catalog", "--derived-root", str(tmp_path / "lseg")])

    assert result.exit_code == 0
    assert "LSEG Corpus Catalog" in result.output
    assert (tmp_path / "lseg" / "catalog.json").exists()
    assert (tmp_path / "lseg" / "catalog.csv").exists()


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
            "--run-a",
            str(run_id),
            "--model-a",
            "model-a",
            "--model-b",
            "model-b",
            "--db-path",
            str(db_path),
            "--n-resamples",
            "200",
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
        ["compare", "--run-a", str(run_id), "--model-a", "model-a", "--model-b", "model-b", "--metric", "bogus", "--db-path", str(db_path)],
    )
    assert result.exit_code != 0


def _news_result(query: str = "market news", config=None) -> NewsFetchResult:
    normalized = normalize_url("https://example.com/article")
    return NewsFetchResult(
        config=config or make_news_fetch_config(query=query, max_results=1),
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


def _write_package_source(path: Path, *, query: str = "bank earnings sentiment", url: str = "https://example.com/article") -> Path:
    path.mkdir(parents=True)
    normalized = normalize_url(url)
    record = {
        "record_id": article_record_id(normalized),
        "url": url,
        "normalized_url": normalized,
        "title": "Market article",
        "snippet": "Snippet",
        "article_text": "Full article text",
        "extraction_status": "success",
    }
    (path / "articles.jsonl").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "tavily",
                "fetched_at": "2026-06-07T12:00:00+00:00",
                "query": query,
                "record_count": 1,
                "search_request_id": "search-1",
                "extract_request_id": "extract-1",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_batch_matrix(path: Path) -> Path:
    path.write_text(
        """
version = 1
description = "Reusable Tavily query matrix for CLI tests."

[[queries]]
id = "bank_earnings"
family = "Bank earnings"
query = "bank earnings sentiment"
topic = "news"
time_range = "month"
max_results = 1
search_depth = "basic"
coverage_target = "Company performance and earnings commentary."
include_domains = []
exclude_domains = []
notes = "Use for source collection; labels are intentionally absent."

[[queries]]
id = "market_volatility"
family = "Market volatility"
query = "market volatility stocks bonds currency"
topic = "news"
time_range = "month"
max_results = 1
search_depth = "basic"
coverage_target = "Market movement and event-driven reporting."
include_domains = []
exclude_domains = []
notes = "Use for source collection; labels are intentionally absent."
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


class FakeNewsClient:
    def __init__(self) -> None:
        self.configs = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def fetch(self, config):
        self.configs.append(config)
        return _news_result(config.query, config=config)


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
            "--query",
            "bank earnings sentiment",
            "--max-results",
            "1",
            "--output-dir",
            str(output_dir),
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
            "--query",
            "bank earnings sentiment",
            "--topic",
            "bogus",
            "--output-dir",
            str(tmp_path / "news"),
        ],
    )

    assert result.exit_code != 0
    assert "topic must be one of" in result.output


