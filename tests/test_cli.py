import json
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


def test_cli_package_news_writes_share_package(tmp_path: Path) -> None:
    source = _write_package_source(tmp_path / "tavily_news_one")
    output_dir = tmp_path / "derived"

    result = runner.invoke(
        app,
        [
            "package-news",
            "--source",
            str(source),
            "--package-id",
            "shared",
            "--output-dir",
            str(output_dir),
            "--query-matrix",
            str(tmp_path / "missing.toml"),
        ],
    )

    assert result.exit_code == 0
    assert "Tavily News Package" in result.output
    package_dir = output_dir / "shared"
    assert (package_dir / "sources.csv").exists()
    assert (package_dir / "screening_index.csv").exists()
    assert (package_dir / "package_manifest.json").exists()
    assert (package_dir / "README.md").exists()
    assert not (package_dir / "extracts.jsonl").exists()


def test_cli_package_news_accepts_repeated_sources(tmp_path: Path) -> None:
    first = _write_package_source(tmp_path / "tavily_news_one", url="https://example.com/a")
    second = _write_package_source(tmp_path / "tavily_news_two", query="market volatility", url="https://example.com/b")

    result = runner.invoke(
        app,
        [
            "package-news",
            "--source",
            str(first),
            "--source",
            str(second),
            "--package-id",
            "shared",
            "--output-dir",
            str(tmp_path / "derived"),
            "--query-matrix",
            str(tmp_path / "missing.toml"),
        ],
    )

    assert result.exit_code == 0
    assert "Unique sources" in result.output
    manifest = json.loads((tmp_path / "derived" / "shared" / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_record_count"] == 2
    assert manifest["unique_source_count"] == 2


def test_cli_package_news_rejects_existing_output(tmp_path: Path) -> None:
    source = _write_package_source(tmp_path / "tavily_news_one")
    existing = tmp_path / "derived" / "shared"
    existing.mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "package-news",
            "--source",
            str(source),
            "--package-id",
            "shared",
            "--output-dir",
            str(tmp_path / "derived"),
        ],
    )

    assert result.exit_code != 0
    assert "already exists" in result.output


def test_cli_package_news_overwrite_package_rebuilds_existing_output(tmp_path: Path) -> None:
    source = _write_package_source(tmp_path / "tavily_news_one")
    existing = tmp_path / "derived" / "shared"
    existing.mkdir(parents=True)
    (existing / "stale.txt").write_text("leftover", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "package-news",
            "--source",
            str(source),
            "--package-id",
            "shared",
            "--output-dir",
            str(tmp_path / "derived"),
            "--overwrite-package",
        ],
    )

    assert result.exit_code == 0
    assert (existing / "sources.csv").exists()
    assert not (existing / "stale.txt").exists()


def test_cli_package_news_rejects_invalid_text_policy_and_source(tmp_path: Path) -> None:
    source = _write_package_source(tmp_path / "tavily_news_one")
    bad_policy = runner.invoke(
        app,
        [
            "package-news",
            "--source",
            str(source),
            "--package-id",
            "shared",
            "--output-dir",
            str(tmp_path / "derived"),
            "--text-policy",
            "full",
        ],
    )
    assert bad_policy.exit_code != 0
    assert "text_policy must be one of" in bad_policy.output

    bad_source = runner.invoke(
        app,
        [
            "package-news",
            "--source",
            str(tmp_path / "missing"),
            "--package-id",
            "shared",
            "--output-dir",
            str(tmp_path / "derived2"),
        ],
    )
    assert bad_source.exit_code != 0
    assert "source corpus directory does not exist" in bad_source.output


def test_cli_fetch_news_batch_dry_run_prints_matrix_plan(tmp_path: Path) -> None:
    matrix = _write_batch_matrix(tmp_path / "matrix.toml")

    result = runner.invoke(
        app,
        [
            "fetch-news-batch",
            "--query-matrix",
            str(matrix),
            "--date-window",
            "2026-05-01:2026-05-07",
            "--date-window",
            "2026-05-08:2026-05-14",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Tavily News Batch Dry Run" in result.output
    assert "Planned fetches: 4" in result.output
    assert "bank_earnings" in result.output


def test_cli_fetch_news_batch_weeks_generates_windows(tmp_path: Path) -> None:
    matrix = _write_batch_matrix(tmp_path / "matrix.toml")

    result = runner.invoke(
        app,
        [
            "fetch-news-batch",
            "--query-matrix",
            str(matrix),
            "--weeks",
            "3",
            "--end-date",
            "2026-06-11",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Planned fetches: 6" in result.output

    missing_weeks = runner.invoke(
        app,
        ["fetch-news-batch", "--query-matrix", str(matrix), "--end-date", "2026-06-11", "--dry-run"],
    )
    assert missing_weeks.exit_code != 0
    assert "--end-date requires --weeks" in missing_weeks.output


class FlakyNewsClient(FakeNewsClient):
    def __init__(self, fail_times: int, fail_query: str | None = None) -> None:
        super().__init__()
        self.fail_times = fail_times
        self.fail_query = fail_query
        self.failures = 0

    async def fetch(self, config):
        if (self.fail_query is None or config.query == self.fail_query) and self.failures < self.fail_times:
            self.failures += 1
            raise TimeoutError("Request timed out after 30 seconds.")
        return await super().fetch(config)


def test_cli_fetch_news_batch_retries_transient_failures(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sentiment_benchmark.cli.FETCH_RETRY_BASE_DELAY_SECONDS", 0.0)
    fake = FlakyNewsClient(fail_times=1)
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: fake)
    matrix = _write_batch_matrix(tmp_path / "matrix.toml")

    result = runner.invoke(
        app,
        [
            "fetch-news-batch",
            "--query-matrix",
            str(matrix),
            "--date-window",
            "2026-05-01:2026-05-07",
            "--output-dir",
            str(tmp_path / "news"),
            "--package-id",
            "retried",
            "--package-output-dir",
            str(tmp_path / "derived"),
        ],
    )

    assert result.exit_code == 0
    assert "retrying" in result.output
    assert (tmp_path / "derived" / "retried" / "sources.csv").exists()


def test_cli_fetch_news_batch_continues_past_persistent_failures(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sentiment_benchmark.cli.FETCH_RETRY_BASE_DELAY_SECONDS", 0.0)
    fake = FlakyNewsClient(fail_times=99, fail_query="bank earnings sentiment")
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: fake)
    matrix = _write_batch_matrix(tmp_path / "matrix.toml")

    result = runner.invoke(
        app,
        [
            "fetch-news-batch",
            "--query-matrix",
            str(matrix),
            "--date-window",
            "2026-05-01:2026-05-07",
            "--output-dir",
            str(tmp_path / "news"),
            "--package-id",
            "partial",
            "--package-output-dir",
            str(tmp_path / "derived"),
        ],
    )

    assert result.exit_code == 1
    assert "failed after 3 attempts" in result.output
    assert "Skipping packaging" in result.output
    assert not (tmp_path / "derived" / "partial").exists()
    # The healthy query still produced a corpus that a re-run will reuse.
    exported = list((tmp_path / "news").glob("tavily_news_*"))
    assert len(exported) == 1


def test_cli_fetch_news_batch_skips_already_fetched(tmp_path: Path, monkeypatch) -> None:
    matrix = _write_batch_matrix(tmp_path / "matrix.toml")
    base_args = [
        "fetch-news-batch",
        "--query-matrix",
        str(matrix),
        "--date-window",
        "2026-05-01:2026-05-07",
        "--output-dir",
        str(tmp_path / "news"),
        "--package-output-dir",
        str(tmp_path / "derived"),
    ]

    first = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: first)
    result = runner.invoke(app, [*base_args, "--package-id", "first"])
    assert result.exit_code == 0
    assert len(first.configs) == 2

    second = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: second)
    result = runner.invoke(app, [*base_args, "--package-id", "second"])
    assert result.exit_code == 0
    assert second.configs == []
    assert "already fetched, skipped" in result.output
    # The skipped corpora still flow into the new package.
    manifest = json.loads((tmp_path / "derived" / "second" / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_record_count"] == 2

    third = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: third)
    result = runner.invoke(app, [*base_args, "--package-id", "third", "--refetch"])
    assert result.exit_code == 0
    assert len(third.configs) == 2


def test_cli_fetch_news_batch_overwrite_package_allows_recurring_runs(tmp_path: Path, monkeypatch) -> None:
    matrix = _write_batch_matrix(tmp_path / "matrix.toml")
    base_args = [
        "fetch-news-batch",
        "--query-matrix",
        str(matrix),
        "--date-window",
        "2026-05-01:2026-05-07",
        "--output-dir",
        str(tmp_path / "news"),
        "--package-id",
        "weekly",
        "--package-output-dir",
        str(tmp_path / "derived"),
    ]

    first = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: first)
    result = runner.invoke(app, base_args)
    assert result.exit_code == 0
    package_dir = tmp_path / "derived" / "weekly"
    assert (package_dir / "package_manifest.json").exists()

    # Without --overwrite-package the second run refuses before fetching anything.
    second = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: second)
    result = runner.invoke(app, base_args)
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert second.configs == []

    third = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: third)
    result = runner.invoke(app, [*base_args, "--overwrite-package", "--date-window", "2026-05-08:2026-05-14"])
    assert result.exit_code == 0
    # Only the new window is fetched; the package is rebuilt from both corpora.
    assert len(third.configs) == 2
    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_record_count"] == 4


def test_cli_news_quality_reports_corpora(tmp_path: Path) -> None:
    news_dir = tmp_path / "news"
    _write_package_source(news_dir / "tavily_news_one")

    result = runner.invoke(app, ["news-quality", "--news-dir", str(news_dir)])

    assert result.exit_code == 0
    assert "Tavily News Quality" in result.output
    assert "Usable unique URLs" in result.output
    assert "Top 10 Source Domains" in result.output

    empty = runner.invoke(app, ["news-quality", "--news-dir", str(tmp_path / "missing")])
    assert empty.exit_code != 0
    assert "no Tavily corpus directories" in empty.output


def test_cli_fetch_news_batch_executes_and_packages(tmp_path: Path, monkeypatch) -> None:
    fake = FakeNewsClient()
    monkeypatch.setattr("sentiment_benchmark.cli._make_tavily_news_client", lambda: fake)
    matrix = _write_batch_matrix(tmp_path / "matrix.toml")

    result = runner.invoke(
        app,
        [
            "fetch-news-batch",
            "--query-matrix",
            str(matrix),
            "--query-id",
            "bank_earnings",
            "--date-window",
            "2026-05-01:2026-05-07",
            "--date-window",
            "2026-05-08:2026-05-14",
            "--output-dir",
            str(tmp_path / "news"),
            "--package-id",
            "batch_shared",
            "--package-output-dir",
            str(tmp_path / "derived"),
        ],
    )

    assert result.exit_code == 0
    assert "Tavily News Batch" in result.output
    assert "Fetched records" in result.output
    assert "Failed extractions" in result.output
    assert len(fake.configs) == 2
    assert fake.configs[0].query == "bank earnings sentiment"
    assert fake.configs[0].start_date == "2026-05-01"
    assert fake.configs[0].end_date == "2026-05-07"
    assert fake.configs[1].start_date == "2026-05-08"
    exported = list((tmp_path / "news").glob("tavily_news_*"))
    assert len(exported) == 2
    assert (tmp_path / "derived" / "batch_shared" / "sources.csv").exists()
    manifest = json.loads((tmp_path / "derived" / "batch_shared" / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_record_count"] == 2
