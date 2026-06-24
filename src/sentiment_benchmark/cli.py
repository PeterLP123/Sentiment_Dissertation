from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .baseline_runner import run_baselines
from .baselines import BASELINE_SPECS
from .comparison import ModelTarget, compare_models
from .constants import (
    ALLOWED_LABELS,
    CONFUSION_PREDICTION_LABELS,
    DEFAULT_BASE_URL,
    DEFAULT_CONCURRENCY,
    DEFAULT_DATASET_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_PILOT_PER_CLASS,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_PROVIDER,
    DEFAULT_REASONING_MAX_COMPLETION_TOKENS,
    DEFAULT_RETRIES,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
)
from .dataset import compute_stats, load_dataset
from .env import load_env_file
from .exporter import export_run
from .latex_tables import sensitivity_table_latex
from .lseg_corpus import build_lseg_corpus
from .lseg_source import (
    LsegNewsClient,
    LsegNewsError,
    check_lseg_news,
    fetch_lseg_news,
    load_lseg_collection_config,
)
from .lseg_validation import create_lseg_validation_sample, evaluate_lseg_annotations
from .models import PromptConfig, RunConfig
from .news_batch import (
    NewsBatchError,
    build_news_batch_plan,
    build_weekly_date_windows,
    find_fetched_corpus,
    load_fetched_corpora,
)
from .news_cleaning import NewsCleaningDependencyError
from .news_package import (
    DEFAULT_NEWS_PACKAGE_ID,
    DEFAULT_NEWS_PACKAGE_OUTPUT_DIR,
    DEFAULT_QUERY_MATRIX_PATH,
    NEWS_TEXT_POLICIES,
    NewsPackageError,
    package_news_sources,
    sanitize_package_id,
)
from .news_quality import NewsQualityError, discover_news_corpora, summarize_news_quality
from .news_source import (
    DEFAULT_MIN_ARTICLE_TEXT_CHARS,
    DEFAULT_NEWS_EXTRACT_DEPTH,
    DEFAULT_NEWS_MAX_RESULTS,
    DEFAULT_NEWS_OUTPUT_DIR,
    DEFAULT_NEWS_SEARCH_DEPTH,
    DEFAULT_NEWS_TIME_RANGE,
    DEFAULT_NEWS_TOPIC,
    NEWS_EXTRACT_DEPTHS,
    NEWS_SEARCH_DEPTHS,
    NEWS_TIME_RANGES,
    NEWS_TOPICS,
    TEXT_QUALITY_OK,
    TavilyNewsClient,
    make_news_fetch_config,
    write_news_corpus,
)
from .newsapi_source import (
    NEWSAPI_MAX_PAGES,
    NewsApiClient,
    NewsApiError,
    make_newsapi_fetch_config,
    write_newsapi_corpus,
)
from .perturbations import generate_prompt_suite
from .prompt_sensitivity import SENSITIVITY_METRICS, prompt_sensitivity
from .prompts import load_prompts
from .providers import endpoint_for_provider, make_llm_client, normalize_provider
from .reliability import run_agreement
from .runner import BenchmarkRunner
from .sc_runner import SelfConsistencyRunner
from .self_consistency import SelfConsistencyResult
from .storage import BenchmarkStore
from .trading_analysis import TradingAnalysisError, analyze_trading_run
from .trading_strategy import (
    TradingStrategyError,
    describe_trading_plan,
    load_trading_config,
)
from .trading_strategy import (
    run_trading_strategy as execute_trading_strategy,
)

console = Console()
app = typer.Typer(help="Benchmark OpenRouter and Ollama LLMs on dissertation sentiment data.")
load_env_file()

# Tavily batch fetches retry transient failures (timeouts, dropped connections)
# before recording a fetch as failed and moving on.
FETCH_ATTEMPTS = 3
FETCH_RETRY_BASE_DELAY_SECONDS = 5.0


def _make_tavily_news_client() -> TavilyNewsClient:
    return TavilyNewsClient()


def _make_newsapi_client() -> NewsApiClient:
    return NewsApiClient()


def _make_lseg_news_client() -> LsegNewsClient:
    return LsegNewsClient()


def _resolve_prompt(prompt_id: str, prompts_path: Path):
    prompts = load_prompts(prompts_path)
    if prompt_id not in prompts:
        available = ", ".join(sorted(prompts))
        raise typer.BadParameter(f"Unknown prompt id {prompt_id!r}. Available: {available}")
    return prompts[prompt_id]


def _resolve_provider(provider: str):
    try:
        return normalize_provider(provider)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _make_run_config(
    *,
    models: list[str],
    prompt: PromptConfig,
    mode: str,
    dataset_path: Path,
    db_path: Path,
    base_url: str,
    provider: str,
    sample_per_class: int,
    seed: int,
    temperature: float,
    max_completion_tokens: int,
    reasoning_max_tokens: int,
    concurrency: int,
    retries: int,
    model_max_completion_tokens: dict[str, int] | None = None,
    few_shot_k: int = 0,
    few_shot_seed: int | None = None,
    ollama_think: bool | None = None,
) -> RunConfig:
    return RunConfig(
        models=models,
        prompt=prompt,
        mode=mode,  # type: ignore[arg-type]
        dataset_path=str(dataset_path),
        db_path=str(db_path),
        base_url=base_url,
        provider=_resolve_provider(provider),
        sample_per_class=sample_per_class,
        seed=seed,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        reasoning_max_completion_tokens=reasoning_max_tokens,
        model_max_completion_tokens=model_max_completion_tokens or {},
        concurrency=concurrency,
        retries=retries,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        ollama_think=ollama_think,
    )


def _parse_model_max_tokens(values: list[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in values or []:
        if "=" not in item:
            raise typer.BadParameter(f"--model-max-tokens must be 'model_id=N', got {item!r}")
        model_id, _, raw = item.partition("=")
        model_id = model_id.strip()
        try:
            tokens = int(raw.strip())
        except ValueError as exc:
            raise typer.BadParameter(f"--model-max-tokens value must be an integer, got {raw!r}") from exc
        if not model_id or tokens <= 0:
            raise typer.BadParameter(f"--model-max-tokens needs a model id and positive integer, got {item!r}")
        overrides[model_id] = tokens
    return overrides


@app.command("validate-data")
def validate_data(
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset-path", help="CSV containing Sentence and Sentiment columns."),
    ] = DEFAULT_DATASET_PATH,
) -> None:
    rows = load_dataset(dataset_path)
    stats = compute_stats(rows)
    table = Table(title=f"Dataset Validation: {dataset_path}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Rows", str(stats.row_count))
    table.add_row("Label counts", str(stats.label_counts))
    table.add_row("Duplicate sentence groups", str(stats.duplicate_sentence_groups))
    table.add_row("Duplicate extra rows", str(stats.duplicate_extra_rows))
    table.add_row("Conflicting duplicate groups", str(stats.conflicting_duplicate_groups))
    table.add_row("Conflicting duplicate rows", str(stats.conflicting_duplicate_rows))
    table.add_row("Primary scoring rows", str(stats.primary_row_count))
    table.add_row("Primary label counts", str(stats.primary_label_counts))
    console.print(table)


@app.command("news-check")
def news_check(
    query: Annotated[str, typer.Option("--query", help="Small Tavily news query for the connectivity check.")] = "financial markets",
    max_results: Annotated[
        int,
        typer.Option("--max-results", help="Maximum Tavily results for the check."),
    ] = 1,
) -> None:
    """Run a small Tavily news search to verify API connectivity."""
    try:
        config = make_news_fetch_config(query=query, max_results=max_results, extract=False)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def main() -> None:
        async with _make_tavily_news_client() as client:
            result = await client.fetch(config)
        table = Table(title="Tavily News Check")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Query", result.config.query)
        table.add_row("Records", str(len(result.records)))
        table.add_row("Request ID", result.search_request_id or "-")
        credits = result.search_usage.get("credits") if isinstance(result.search_usage, dict) else None
        table.add_row("Credits", str(credits) if credits is not None else "-")
        console.print(table)

    asyncio.run(main())


@app.command("fetch-news")
def fetch_news(
    query: Annotated[str, typer.Option("--query", help="Tavily news search query.")],
    max_results: Annotated[
        int,
        typer.Option("--max-results", help="Maximum Tavily results to save (1-20)."),
    ] = DEFAULT_NEWS_MAX_RESULTS,
    topic: Annotated[
        str,
        typer.Option("--topic", help=f"Search topic: {', '.join(NEWS_TOPICS)}."),
    ] = DEFAULT_NEWS_TOPIC,
    time_range: Annotated[
        str | None,
        typer.Option("--time-range", help=f"Publish/update time range: {', '.join(NEWS_TIME_RANGES)}."),
    ] = DEFAULT_NEWS_TIME_RANGE,
    search_depth: Annotated[
        str,
        typer.Option("--search-depth", help=f"Tavily search depth: {', '.join(NEWS_SEARCH_DEPTHS)}."),
    ] = DEFAULT_NEWS_SEARCH_DEPTH,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="Only return results after this date (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="Only return results before this date (YYYY-MM-DD)."),
    ] = None,
    include_domain: Annotated[
        list[str] | None,
        typer.Option("--include-domain", help="Domain to include. Repeat for multiple domains."),
    ] = None,
    exclude_domain: Annotated[
        list[str] | None,
        typer.Option("--exclude-domain", help="Domain to exclude. Repeat for multiple domains."),
    ] = None,
    extract: Annotated[
        bool,
        typer.Option("--extract/--no-extract", help="Run Tavily Extract for full article text."),
    ] = True,
    extract_depth: Annotated[
        str,
        typer.Option("--extract-depth", help=f"Tavily extract depth: {', '.join(NEWS_EXTRACT_DEPTHS)}."),
    ] = DEFAULT_NEWS_EXTRACT_DEPTH,
    min_text_chars: Annotated[
        int,
        typer.Option("--min-text-chars", help="Minimum extracted characters before text counts as usable (0 disables)."),
    ] = DEFAULT_MIN_ARTICLE_TEXT_CHARS,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory where timestamped Tavily article exports are written."),
    ] = DEFAULT_NEWS_OUTPUT_DIR,
) -> None:
    """Fetch Tavily-sourced news articles into a derived article corpus."""
    try:
        config = make_news_fetch_config(
            query=query,
            max_results=max_results,
            topic=topic,
            time_range=time_range,
            search_depth=search_depth,
            extract=extract,
            extract_depth=extract_depth,
            start_date=start_date,
            end_date=end_date,
            include_domains=include_domain,
            exclude_domains=exclude_domain,
            min_text_chars=min_text_chars,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def main() -> None:
        async with _make_tavily_news_client() as client:
            result = await client.fetch(config)
        paths = write_news_corpus(result, output_dir)
        failed = sum(1 for record in result.records if record.extraction_status == "failed")
        usable = sum(1 for record in result.records if record.text_quality == TEXT_QUALITY_OK)
        table = Table(title="Tavily News Fetch")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Records", str(len(result.records)))
        table.add_row("Usable articles", str(usable))
        table.add_row("Failed extractions", str(failed))
        table.add_row("Output directory", str(paths.output_dir))
        table.add_row("JSONL", str(paths.articles_jsonl))
        table.add_row("CSV", str(paths.articles_csv))
        table.add_row("Manifest", str(paths.manifest_json))
        console.print(table)

    asyncio.run(main())


@app.command("newsapi-check")
def newsapi_check(
    query: Annotated[str, typer.Option("--query", help="Small NewsAPI query used for connectivity checking.")] = "financial markets",
) -> None:
    """Run a one-result NewsAPI Everything request to verify credentials."""

    async def main() -> None:
        async with _make_newsapi_client() as client:
            result = await client.check(query)
        table = Table(title="NewsAPI Check")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Query", result.config.query)
        table.add_row("Records", str(len(result.records)))
        table.add_row("Total available", str(result.total_results))
        table.add_row("Pages", str(result.pages_fetched))
        console.print(table)

    try:
        asyncio.run(main())
    except NewsApiError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("lseg-news-check")
def lseg_news_check(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="TOML LSEG collection definition used for the entitlement check."),
    ] = Path("configs/lseg_workspace_example.toml"),
) -> None:
    """Verify Workspace headline and story access without writing data."""
    try:
        config = load_lseg_collection_config(config_path)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def main() -> None:
        async with _make_lseg_news_client() as client:
            result = await check_lseg_news(config, client)
        table = Table(title="LSEG Workspace News Check")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Query", str(result["query"]))
        table.add_row("Headlines", str(result["headline_count"]))
        table.add_row("Story ID", str(result["story_id"] or "-"))
        table.add_row("Story status", str(result["story_status"]))
        console.print(table)

    try:
        asyncio.run(main())
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("fetch-lseg-news")
def fetch_lseg_news_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="TOML LSEG collection definition."),
    ] = Path("configs/lseg_workspace_example.toml"),
) -> None:
    """Fetch an immutable, resumable LSEG Workspace headline/story collection."""
    try:
        config = load_lseg_collection_config(config_path)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def main() -> None:
        async with _make_lseg_news_client() as client:
            result = await fetch_lseg_news(config, client)
        table = Table(title="LSEG Workspace News Fetch")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Collection ID", config.collection_id)
        table.add_row("Headlines", str(result.headline_count))
        table.add_row("Stories", str(result.story_count))
        table.add_row("Unavailable/failed stories", str(result.failed_story_count))
        table.add_row("Resumed", str(result.resumed))
        table.add_row("Raw collection", str(result.raw_dir))
        table.add_row("Manifest", str(result.manifest_path))
        console.print(table)

    try:
        asyncio.run(main())
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("build-lseg-corpus")
def build_lseg_corpus_command(
    source: Annotated[
        Path,
        typer.Option("--source", help="Completed raw LSEG collection directory under Data/news."),
    ],
) -> None:
    """Build deterministic clean text from a completed raw LSEG collection."""
    try:
        result = build_lseg_corpus(source)
    except (LsegNewsError, NewsCleaningDependencyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="LSEG Clean Corpus")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Articles", str(result.article_count))
    table.add_row("Eligible", str(result.eligible_count))
    table.add_row("Resumed", str(result.resumed))
    table.add_row("Derived corpus", str(result.derived_dir))
    table.add_row("Articles JSONL", str(result.articles_path))
    table.add_row("Screening index", str(result.screening_path))
    table.add_row("Manifest", str(result.manifest_path))
    console.print(table)


@app.command("sample-lseg-validation")
def sample_lseg_validation_command(
    corpus_manifest: Annotated[Path, typer.Option("--corpus-manifest", help="Verified derived LSEG corpus manifest.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="New local-only annotation directory.")],
    sample_size: Annotated[int, typer.Option("--sample-size")] = 150,
    double_code_size: Annotated[int, typer.Option("--double-code-size")] = 30,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    double_code_seed: Annotated[int, typer.Option("--double-code-seed")] = 43,
) -> None:
    """Create deterministic primary and double-code LSEG annotation sheets."""
    try:
        result = create_lseg_validation_sample(
            corpus_manifest,
            output_dir,
            sample_size=sample_size,
            double_code_size=double_code_size,
            seed=seed,
            double_code_seed=double_code_seed,
        )
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="LSEG Validation Sample")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Stories", str(result.sample_count))
    table.add_row("Double-coded", str(result.double_code_count))
    table.add_row("Primary sheet", str(result.primary_path))
    table.add_row("Secondary sheet", str(result.secondary_path))
    table.add_row("Manifest", str(result.manifest_path))
    console.print(table)


@app.command("evaluate-lseg-annotations")
def evaluate_lseg_annotations_command(
    primary: Annotated[Path, typer.Option("--primary", help="Completed primary annotation CSV.")],
    secondary: Annotated[Path, typer.Option("--secondary", help="Completed secondary annotation CSV.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="New local-only evaluation directory.")],
) -> None:
    """Validate annotations, report agreement, and export the adjudicated benchmark."""
    try:
        result = evaluate_lseg_annotations(primary, secondary, output_dir)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="LSEG Annotation Agreement")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Double-coded", str(result.double_code_count))
    table.add_row("Percent agreement", f"{result.percent_agreement:.1%}")
    table.add_row("Cohen's kappa", f"{result.cohen_kappa:.4f}")
    table.add_row("Adjudicated dataset", str(result.labeled_dataset_path))
    table.add_row("Metrics", str(result.metrics_path))
    console.print(table)


@app.command("fetch-newsapi")
def fetch_newsapi(
    query: Annotated[str, typer.Option("--query", help="NewsAPI Everything search query.")],
    from_time: Annotated[
        str | None,
        typer.Option("--from", help="Oldest publication date/time, in ISO 8601 format."),
    ] = None,
    to_time: Annotated[
        str | None,
        typer.Option("--to", help="Newest publication date/time, in ISO 8601 format."),
    ] = None,
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", help="Maximum 100-result pages to request."),
    ] = NEWSAPI_MAX_PAGES,
    domain: Annotated[
        list[str] | None,
        typer.Option("--domain", help="Publisher domain to include. Repeat for multiple domains."),
    ] = None,
    exclude_domain: Annotated[
        list[str] | None,
        typer.Option("--exclude-domain", help="Publisher domain to exclude. Repeat for multiple domains."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory where the timestamped NewsAPI corpus is written."),
    ] = DEFAULT_NEWS_OUTPUT_DIR,
) -> None:
    """Fetch NewsAPI titles and descriptions into a derived article corpus."""
    try:
        config = make_newsapi_fetch_config(
            query=query,
            from_time=from_time,
            to_time=to_time,
            max_pages=max_pages,
            domains=domain,
            exclude_domains=exclude_domain,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def main() -> None:
        async with _make_newsapi_client() as client:
            result = await client.fetch(config)
        paths = write_newsapi_corpus(result, output_dir)
        table = Table(title="NewsAPI Fetch")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Records", str(len(result.records)))
        table.add_row("Total available", str(result.total_results))
        table.add_row("Pages", str(result.pages_fetched))
        table.add_row("Output directory", str(paths.output_dir))
        table.add_row("JSONL", str(paths.articles_jsonl))
        table.add_row("CSV", str(paths.articles_csv))
        table.add_row("Manifest", str(paths.manifest_json))
        console.print(table)

    try:
        asyncio.run(main())
    except NewsApiError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("run-trading-strategy")
def run_trading_strategy_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="TOML trading strategy configuration."),
    ] = Path("configs/week3_trading_pilot.toml"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the fixed plan without API calls or output writes."),
    ] = False,
) -> None:
    """Run the reproducible news-sentiment trading pilot."""
    try:
        config = load_trading_config(config_path)
    except TradingStrategyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if dry_run:
        plan = describe_trading_plan(config)
        table = Table(title="Trading Strategy Dry Run")
        table.add_column("Setting")
        table.add_column("Value")
        for key, value in plan.items():
            table.add_row(key.replace("_", " ").title(), ", ".join(map(str, value)) if isinstance(value, list) else str(value))
        console.print(table)
        return

    async def main() -> None:
        async with AsyncExitStack() as stack:
            newsapi_client = (
                await stack.enter_async_context(_make_newsapi_client()) if config.newsapi_enabled else None
            )
            tavily_client = (
                await stack.enter_async_context(_make_tavily_news_client()) if config.tavily_gap_fetch else None
            )
            llm_client = await stack.enter_async_context(
                make_llm_client(
                    _resolve_provider(config.provider),
                    base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
                    ollama_host=config.ollama_host,
                    ollama_keep_alive=config.ollama_keep_alive,
                    ollama_think=config.ollama_think,
                    structured_label_output=config.structured_output,
                )
            )
            result = await execute_trading_strategy(
                config_path,
                newsapi_client=newsapi_client,
                tavily_client=tavily_client,
                llm_client=llm_client,
            )
        table = Table(title="Trading Strategy Complete")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Run ID", result.run_id)
        table.add_row("Accepted articles", str(result.accepted_article_count))
        table.add_row("Sentiment scores", str(result.sentiment_score_count))
        table.add_row("Return rows", str(result.return_count))
        table.add_row("Derived data", str(result.derived_dir))
        table.add_row("Meeting report", str(result.results_dir / "summary.md"))
        console.print(table)

    try:
        asyncio.run(main())
    except (NewsApiError, TradingStrategyError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("analyze-trading-run")
def analyze_trading_run_command(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Completed trading run directory containing run_manifest.json."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="New non-overwriting directory for analysis tables, plots, and report."),
    ],
    comparison_run_dir: Annotated[
        Path | None,
        typer.Option("--comparison-run-dir", help="Optional earlier run used for descriptive panel-sensitivity comparison."),
    ] = None,
    bootstrap_resamples: Annotated[
        int,
        typer.Option("--bootstrap-resamples", min=1, help="Number of deterministic percentile-bootstrap resamples."),
    ] = 10_000,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed used for bootstrap resampling."),
    ] = 42,
) -> None:
    """Create robustness tables, plots, and a technical report for a completed trading run."""
    try:
        result = analyze_trading_run(
            run_dir,
            output_dir,
            comparison_run_dir=comparison_run_dir,
            seed=seed,
            resamples=bootstrap_resamples,
        )
    except TradingAnalysisError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="Trading Analysis Complete")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("Technical report", str(result.summary_path))
    table.add_row("Analysis manifest", str(result.manifest_path))
    table.add_row("Generated files", str(len(result.generated_files)))
    console.print(table)


@app.command("package-news")
def package_news(
    source: Annotated[
        list[Path] | None,
        typer.Option("--source", help="Existing Tavily corpus directory. Repeat for multiple corpora."),
    ] = None,
    package_id: Annotated[
        str,
        typer.Option("--package-id", help="Share package id. Spaces are converted to hyphens."),
    ] = DEFAULT_NEWS_PACKAGE_ID,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Root directory where the share package directory is written."),
    ] = DEFAULT_NEWS_PACKAGE_OUTPUT_DIR,
    query_matrix: Annotated[
        Path,
        typer.Option("--query-matrix", help="Optional TOML query matrix used to enrich package provenance."),
    ] = DEFAULT_QUERY_MATRIX_PATH,
    text_policy: Annotated[
        str,
        typer.Option("--text-policy", help=f"Text sharing policy: {', '.join(NEWS_TEXT_POLICIES)}."),
    ] = "metadata",
    overwrite_package: Annotated[
        bool,
        typer.Option(
            "--overwrite-package/--no-overwrite-package",
            help="Remove and rebuild the package directory if it already exists.",
        ),
    ] = False,
) -> None:
    """Package existing Tavily corpora into a colleague-shareable source dataset."""
    if not source:
        raise typer.BadParameter("--source is required")
    try:
        result = package_news_sources(
            list(source),
            package_id=package_id,
            output_root=output_dir,
            query_matrix_path=query_matrix,
            text_policy=text_policy,
            overwrite=overwrite_package,
        )
    except NewsPackageError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Tavily News Package")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Package ID", result.package_id)
    table.add_row("Text policy", result.text_policy)
    table.add_row("Input records", str(result.input_record_count))
    table.add_row("Unique sources", str(result.unique_source_count))
    table.add_row("Duplicate URLs", str(result.duplicate_url_count))
    for relevance, count in result.entity_relevance_counts.items():
        table.add_row(f"Entity relevance: {relevance}", str(count))
    table.add_row("Output directory", str(result.paths.output_dir))
    table.add_row("Sources CSV", str(result.paths.sources_csv))
    table.add_row("Screening CSV", str(result.paths.screening_index_csv))
    table.add_row("Manifest", str(result.paths.package_manifest_json))
    table.add_row("README", str(result.paths.readme_md))
    if result.paths.extracts_jsonl is not None:
        table.add_row("Extracts JSONL", str(result.paths.extracts_jsonl))
    console.print(table)


@app.command("news-quality")
def news_quality(
    source: Annotated[
        list[Path] | None,
        typer.Option("--source", help="Tavily corpus directory to inspect. Repeat for multiple; defaults to all under --news-dir."),
    ] = None,
    news_dir: Annotated[
        Path,
        typer.Option("--news-dir", help="Root directory scanned for Tavily corpora when --source is not given."),
    ] = DEFAULT_NEWS_OUTPUT_DIR,
    query_matrix: Annotated[
        Path,
        typer.Option("--query-matrix", help="Optional TOML query matrix used to group corpora by query family."),
    ] = DEFAULT_QUERY_MATRIX_PATH,
    top_domains: Annotated[
        int,
        typer.Option("--top-domains", help="Number of source domains to list."),
    ] = 10,
) -> None:
    """Summarize text quality across fetched Tavily corpora without calling Tavily."""
    sources: list[str | Path] = []
    if source:
        sources.extend(Path(item) for item in source)
    else:
        sources.extend(discover_news_corpora(news_dir))
    try:
        report = summarize_news_quality(sources, query_matrix_path=query_matrix)
    except NewsQualityError as exc:
        raise typer.BadParameter(str(exc)) from exc

    overview = Table(title="Tavily News Quality")
    overview.add_column("Metric")
    overview.add_column("Value", justify="right")
    overview.add_row("Corpus directories", str(report.corpus_count))
    overview.add_row("Records", str(report.record_count))
    overview.add_row("Unique URLs", str(report.unique_url_count))
    overview.add_row("Usable unique URLs", str(report.usable_unique_url_count))
    for quality, count in report.quality_counts.items():
        overview.add_row(f"Text quality: {quality}", str(count))
    for source_name, count in report.published_date_counts.items():
        overview.add_row(f"Published date from {source_name}", str(count))
    for relevance, count in report.entity_relevance_counts.items():
        overview.add_row(f"Entity relevance: {relevance}", str(count))
    console.print(overview)

    families = Table(title="By Query Family")
    families.add_column("Family")
    families.add_column("Records", justify="right")
    families.add_column("Usable", justify="right")
    families.add_column("On-topic", justify="right")
    for family in report.families:
        if family.screened_count:
            share = family.on_topic_count / family.screened_count
            on_topic = f"{family.on_topic_count}/{family.screened_count} ({share:.0%})"
            if share < 0.7:
                on_topic = f"[yellow]{on_topic}[/yellow]"
        else:
            on_topic = "-"
        families.add_row(family.family, str(family.record_count), str(family.usable_count), on_topic)
    console.print(families)
    if any(family.screened_count for family in report.families):
        console.print(
            "[dim]On-topic = usable articles whose extracted text mentions a company alias for the query family in the "
            "title or lead and clears the body mention-count threshold. Families without a 'TICKER — Company Name' "
            "family field are not screened ('-').[/dim]"
        )

    domains = Table(title=f"Top {top_domains} Source Domains")
    domains.add_column("Domain")
    domains.add_column("Records", justify="right")
    domains.add_column("Usable", justify="right")
    domains.add_column("Shared prefix", justify="right")
    for domain in report.domains[:top_domains]:
        prefix = str(domain.shared_prefix_chars) if domain.shared_prefix_chars else "-"
        if domain.shared_prefix_chars >= 200:
            prefix = f"[yellow]{prefix}[/yellow]"
        domains.add_row(domain.domain, str(domain.record_count), str(domain.usable_count), prefix)
    console.print(domains)
    if any(domain.shared_prefix_chars >= 200 for domain in report.domains[:top_domains]):
        console.print(
            "[dim]A large shared prefix means at least two distinct usable articles from that domain open with the "
            "same text — site boilerplate in the extraction or duplicated coverage. Consider refetching with "
            "--extract-depth advanced and reviewing those articles before labeling.[/dim]"
        )


@app.command("fetch-news-batch")
def fetch_news_batch(
    date_window: Annotated[
        list[str] | None,
        typer.Option("--date-window", help="Date window as YYYY-MM-DD:YYYY-MM-DD. Repeat for multiple windows."),
    ] = None,
    weeks: Annotated[
        int | None,
        typer.Option("--weeks", help="Generate this many contiguous 7-day windows ending at --end-date (default today)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="Last day covered by --weeks as YYYY-MM-DD; defaults to today."),
    ] = None,
    query_matrix: Annotated[
        Path,
        typer.Option("--query-matrix", help="TOML query matrix defining reusable Tavily queries."),
    ] = DEFAULT_QUERY_MATRIX_PATH,
    query_id: Annotated[
        list[str] | None,
        typer.Option("--query-id", help="Query id from the matrix to include. Repeat to select a subset."),
    ] = None,
    extract: Annotated[
        bool,
        typer.Option("--extract/--no-extract", help="Run Tavily Extract for full article text."),
    ] = True,
    extract_depth: Annotated[
        str | None,
        typer.Option(
            "--extract-depth",
            help=f"Override matrix extract depth for every query: {', '.join(NEWS_EXTRACT_DEPTHS)}.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory where timestamped Tavily article exports are written."),
    ] = DEFAULT_NEWS_OUTPUT_DIR,
    package: Annotated[
        bool,
        typer.Option("--package/--no-package", help="Package fetched corpora after the batch completes."),
    ] = True,
    package_id: Annotated[
        str,
        typer.Option("--package-id", help="Share package id used when --package is enabled."),
    ] = DEFAULT_NEWS_PACKAGE_ID,
    package_output_dir: Annotated[
        Path,
        typer.Option("--package-output-dir", help="Root directory where the share package directory is written."),
    ] = DEFAULT_NEWS_PACKAGE_OUTPUT_DIR,
    text_policy: Annotated[
        str,
        typer.Option("--text-policy", help=f"Text sharing policy for the optional package: {', '.join(NEWS_TEXT_POLICIES)}."),
    ] = "metadata",
    overwrite_package: Annotated[
        bool,
        typer.Option(
            "--overwrite-package/--no-overwrite-package",
            help="Remove and rebuild the package directory if it already exists; needed for recurring runs with a fixed --package-id.",
        ),
    ] = False,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--refetch",
            help="Skip fetches whose exact parameters already exist under --output-dir; existing corpora still join the package.",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print planned fetches without calling Tavily or writing files."),
    ] = False,
) -> None:
    """Fetch a query matrix across repeated date windows and optionally package the results."""
    if end_date is not None and weeks is None:
        raise typer.BadParameter("--end-date requires --weeks")
    try:
        windows = list(date_window or [])
        if weeks is not None:
            windows.extend(build_weekly_date_windows(weeks, end_date=end_date))
        plans = build_news_batch_plan(
            query_matrix_path=query_matrix,
            date_windows=windows,
            query_ids=query_id,
            extract=extract,
            extract_depth=extract_depth,
        )
    except NewsBatchError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if dry_run:
        table = Table(title="Tavily News Batch Dry Run")
        table.add_column("Query ID")
        table.add_column("Family")
        table.add_column("Date Window")
        table.add_column("Max", justify="right")
        table.add_column("Topic")
        table.add_column("Depth (search/extract)")
        for plan in plans:
            table.add_row(
                plan.query_entry.id,
                plan.query_entry.family,
                f"{plan.date_window.start_date}:{plan.date_window.end_date}",
                str(plan.config.max_results),
                plan.config.topic,
                f"{plan.config.search_depth}/{plan.config.extract_depth}",
            )
        console.print(table)
        console.print(f"Planned fetches: {len(plans)}")
        if skip_existing:
            fetched = load_fetched_corpora(output_dir)
            already = sum(1 for plan in plans if find_fetched_corpus(plan, fetched) is not None)
            console.print(f"Already fetched (will be skipped): {already}")
        return

    if package:
        try:
            resolved_package_id = sanitize_package_id(package_id)
        except NewsPackageError as exc:
            raise typer.BadParameter(str(exc)) from exc
        package_target = package_output_dir / resolved_package_id
        if package_target.exists() and not overwrite_package:
            raise typer.BadParameter(
                f"output directory already exists: {package_target} (pass --overwrite-package to rebuild it)"
            )

    async def main() -> None:
        corpus_dirs: list[Path] = []
        fetch_count = 0
        skipped_count = 0
        failed_fetch_count = 0
        total_records = 0
        total_failed = 0
        total_usable = 0
        package_result = None
        fetched_corpora = load_fetched_corpora(output_dir) if skip_existing else []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        with progress:
            fetch_task = progress.add_task("Fetching Tavily corpora", total=len(plans))
            async with _make_tavily_news_client() as client:
                for index, plan in enumerate(plans, start=1):
                    window = f"{plan.date_window.start_date}:{plan.date_window.end_date}"
                    progress.update(fetch_task, description=f"[{index}/{len(plans)}] {plan.query_entry.id} | {window}")
                    existing_dir = find_fetched_corpus(plan, fetched_corpora) if skip_existing else None
                    if existing_dir is not None:
                        corpus_dirs.append(existing_dir)
                        skipped_count += 1
                        progress.console.print(
                            "[yellow]↷[/yellow] "
                            f"{plan.query_entry.id} [dim]{window}[/dim] "
                            f"-> already fetched, skipped [dim]{existing_dir}[/dim]"
                        )
                        progress.advance(fetch_task)
                        continue
                    result = None
                    for attempt in range(1, FETCH_ATTEMPTS + 1):
                        try:
                            result = await client.fetch(plan.config)
                            break
                        except Exception as exc:
                            if attempt == FETCH_ATTEMPTS:
                                failed_fetch_count += 1
                                progress.console.print(
                                    "[red]✗[/red] "
                                    f"{plan.query_entry.id} [dim]{window}[/dim] "
                                    f"-> failed after {FETCH_ATTEMPTS} attempts: {exc}"
                                )
                            else:
                                delay = FETCH_RETRY_BASE_DELAY_SECONDS * attempt
                                progress.console.print(
                                    "[yellow]![/yellow] "
                                    f"{plan.query_entry.id} [dim]{window}[/dim] "
                                    f"-> attempt {attempt} failed ({exc}); retrying in {delay:.0f}s"
                                )
                                await asyncio.sleep(delay)
                    if result is None:
                        progress.advance(fetch_task)
                        continue
                    paths = write_news_corpus(result, output_dir)
                    corpus_dirs.append(paths.output_dir)
                    fetch_count += 1
                    failed = sum(1 for record in result.records if record.extraction_status == "failed")
                    usable = sum(1 for record in result.records if record.text_quality == TEXT_QUALITY_OK)
                    total_records += len(result.records)
                    total_failed += failed
                    total_usable += usable
                    progress.console.print(
                        "[green]✓[/green] "
                        f"{plan.query_entry.id} [dim]{window}[/dim] "
                        f"-> {len(result.records)} records, {usable} usable, {failed} failed extractions "
                        f"[dim]{paths.output_dir}[/dim]"
                    )
                    progress.advance(fetch_task)

            if package and failed_fetch_count:
                progress.console.print(
                    "[yellow]![/yellow] "
                    f"Skipping packaging because {failed_fetch_count} fetch(es) failed. "
                    "Re-run the same command to retry only the failed fetches and build the package."
                )
            if package and not failed_fetch_count:
                package_dirs: list[str | Path] = list(dict.fromkeys(corpus_dirs))
                package_task = progress.add_task("Packaging share dataset", total=1)
                progress.update(
                    package_task,
                    description=f"Packaging {len(package_dirs)} corpora -> {package_output_dir / resolved_package_id}",
                )
                try:
                    package_result = package_news_sources(
                        package_dirs,
                        package_id=package_id,
                        output_root=package_output_dir,
                        query_matrix_path=query_matrix,
                        text_policy=text_policy,
                        overwrite=overwrite_package,
                    )
                except NewsPackageError as exc:
                    raise typer.BadParameter(str(exc)) from exc
                progress.console.print(
                    "[green]✓[/green] "
                    f"package {package_result.package_id} "
                    f"-> {package_result.unique_source_count} unique sources, "
                    f"{package_result.duplicate_url_count} duplicate URLs removed "
                    f"[dim]{package_result.paths.output_dir}[/dim]"
                )
                progress.advance(package_task)

        table = Table(title="Tavily News Batch")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Planned fetches", str(len(plans)))
        table.add_row("Fetched", str(fetch_count))
        table.add_row("Skipped (already fetched)", str(skipped_count))
        table.add_row("Failed fetches", str(failed_fetch_count))
        table.add_row("Corpus directories", str(len(dict.fromkeys(corpus_dirs))))
        table.add_row("Fetched records", str(total_records))
        table.add_row("Usable articles", str(total_usable))
        table.add_row("Failed extractions", str(total_failed))
        table.add_row("Output directory", str(output_dir))
        if package_result is not None:
            table.add_row("Package ID", package_result.package_id)
            table.add_row("Package directory", str(package_result.paths.output_dir))
            table.add_row("Unique package sources", str(package_result.unique_source_count))
        console.print(table)
        if failed_fetch_count:
            console.print(
                f"[red]{failed_fetch_count} fetch(es) failed.[/red] Re-run the same command: completed fetches are skipped automatically."
            )
            raise typer.Exit(1)

    asyncio.run(main())


@app.command("list-models")
def list_models(
    provider: Annotated[
        str,
        typer.Option("--provider", help="Model provider: openrouter or ollama."),
    ] = os.getenv("SENTIMENT_BENCH_PROVIDER", DEFAULT_PROVIDER),
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="OpenRouter-compatible base URL."),
    ] = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama host URL, e.g. http://desktop-pc:11434."),
    ] = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
    limit: Annotated[int, typer.Option("--limit", help="Maximum rows to show.")] = 50,
) -> None:
    resolved_provider = _resolve_provider(provider)

    async def main() -> None:
        async with make_llm_client(resolved_provider, base_url=base_url, ollama_host=ollama_host) as client:
            models = await client.list_models()
        table = Table(title=f"{resolved_provider.title()} Models")
        table.add_column("Model ID")
        table.add_column("Name")
        table.add_column("Context", justify="right")
        for model in models[:limit]:
            table.add_row(model.model_id, model.name or "", str(model.context_length or ""))
        console.print(table)
        console.print(f"Showing {min(limit, len(models))} of {len(models)} models")

    asyncio.run(main())


@app.command("run")
def run_benchmark(
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model id. Repeat for multiple models.")],
    mode: Annotated[str, typer.Option("--mode", help="pilot or full.")] = "pilot",
    prompt_id: Annotated[str, typer.Option("--prompt-id", help="Prompt id from configs/default_prompts.toml.")] = "default_label_only",
    dataset_path: Annotated[Path, typer.Option("--dataset-path")] = DEFAULT_DATASET_PATH,
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    prompts_path: Annotated[Path, typer.Option("--prompts-path")] = DEFAULT_PROMPTS_PATH,
    provider: Annotated[
        str,
        typer.Option("--provider", help="Model provider: openrouter or ollama."),
    ] = os.getenv("SENTIMENT_BENCH_PROVIDER", DEFAULT_PROVIDER),
    base_url: Annotated[str, typer.Option("--base-url")] = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama host URL, e.g. http://desktop-pc:11434."),
    ] = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
    sample_per_class: Annotated[int, typer.Option("--sample-per-class")] = DEFAULT_PILOT_PER_CLASS,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    temperature: Annotated[float, typer.Option("--temperature")] = DEFAULT_TEMPERATURE,
    max_completion_tokens: Annotated[int, typer.Option("--max-completion-tokens")] = DEFAULT_MAX_COMPLETION_TOKENS,
    reasoning_max_tokens: Annotated[
        int,
        typer.Option(
            "--reasoning-max-tokens",
            help="Completion-token budget applied to reasoning models when larger than --max-completion-tokens.",
        ),
    ] = DEFAULT_REASONING_MAX_COMPLETION_TOKENS,
    model_max_tokens: Annotated[
        list[str] | None,
        typer.Option(
            "--model-max-tokens",
            help="Per-model completion-token override as 'model_id=N'. Repeat for multiple models.",
        ),
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency")] = DEFAULT_CONCURRENCY,
    retries: Annotated[int, typer.Option("--retries")] = DEFAULT_RETRIES,
    few_shot_k: Annotated[
        int,
        typer.Option("--few-shot-k", help="In-context demonstrations per class (0 = zero-shot). Demos are drawn from non-evaluation rows."),
    ] = 0,
    few_shot_seed: Annotated[
        int | None,
        typer.Option("--few-shot-seed", help="Seed for demonstration sampling (defaults to --seed)."),
    ] = None,
    ollama_think: Annotated[
        bool | None,
        typer.Option(
            "--ollama-think/--no-ollama-think",
            help=(
                "Enable or disable Ollama model thinking. Default: provider default. "
                "Thinking models given a small completion budget can return empty output unless disabled."
            ),
        ),
    ] = None,
    resume_run_id: Annotated[
        int | None,
        typer.Option(
            "--resume-run-id",
            help=(
                "Resume an existing run without duplicating completed responses. Few-shot and generation settings "
                "(temperature, token budgets, Ollama thinking) are restored from the stored run; the matching CLI "
                "flags are ignored."
            ),
        ),
    ] = None,
) -> None:
    if mode not in {"pilot", "full"}:
        raise typer.BadParameter("mode must be pilot or full")
    if not models:
        raise typer.BadParameter("At least one --models value is required")
    if few_shot_k < 0:
        raise typer.BadParameter("--few-shot-k must be >= 0")
    resolved_provider = _resolve_provider(provider)
    endpoint = endpoint_for_provider(resolved_provider, base_url=base_url, ollama_host=ollama_host)
    model_token_overrides = _parse_model_max_tokens(model_max_tokens)
    prompt = _resolve_prompt(prompt_id, prompts_path)
    config = _make_run_config(
        models=models,
        prompt=prompt,
        mode=mode,
        dataset_path=dataset_path,
        db_path=db_path,
        base_url=endpoint,
        provider=resolved_provider,
        sample_per_class=sample_per_class,
        seed=seed,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        reasoning_max_tokens=reasoning_max_tokens,
        model_max_completion_tokens=model_token_overrides,
        concurrency=concurrency,
        retries=retries,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        ollama_think=ollama_think,
    )

    async def main() -> None:
        store = BenchmarkStore(db_path)
        async with make_llm_client(resolved_provider, base_url=base_url, ollama_host=ollama_host) as client:
            runner = BenchmarkRunner(client=client, store=store)
            summary = await runner.run(config, resume_run_id=resume_run_id, callback=lambda message: console.print(message))
        console.print(f"Run {summary.run_id} complete: {summary.model_count} model(s), {summary.selected_row_count} row(s)")

    asyncio.run(main())


@app.command("run-baselines")
def run_baselines_command(
    baselines: Annotated[
        list[str] | None,
        typer.Option("--baselines", "-b", help="Baseline name. Repeat for multiple. Default: majority, tfidf_logreg."),
    ] = None,
    mode: Annotated[str, typer.Option("--mode", help="pilot or full.")] = "pilot",
    dataset_path: Annotated[Path, typer.Option("--dataset-path")] = DEFAULT_DATASET_PATH,
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    sample_per_class: Annotated[int, typer.Option("--sample-per-class")] = DEFAULT_PILOT_PER_CLASS,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    folds: Annotated[int, typer.Option("--folds", help="Stratified CV folds for fitted baselines.")] = 5,
    match_run_id: Annotated[
        int | None,
        typer.Option("--match-run-id", help="Evaluate on the exact rows selected by an existing run (for fair comparison)."),
    ] = None,
) -> None:
    if mode not in {"pilot", "full"}:
        raise typer.BadParameter("mode must be pilot or full")
    unknown = [name for name in (baselines or []) if name not in BASELINE_SPECS]
    if unknown:
        available = ", ".join(sorted(BASELINE_SPECS))
        raise typer.BadParameter(f"Unknown baseline(s): {', '.join(unknown)}. Available: {available}")
    summary = run_baselines(
        baselines,
        mode=mode,  # type: ignore[arg-type]
        dataset_path=str(dataset_path),
        db_path=str(db_path),
        sample_per_class=sample_per_class,
        seed=seed,
        folds=folds,
        match_run_id=match_run_id,
        callback=lambda message: console.print(message),
    )
    console.print(f"Baseline run {summary.run_id} complete: {summary.baseline_count} baseline(s), {summary.selected_row_count} row(s)")


@app.command("export")
def export(
    run_id: Annotated[int, typer.Option("--run-id", help="Run id to export.")],
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    paths = export_run(db_path, run_id, output_dir=output_dir)
    for path in paths:
        console.print(path)


@app.command("runs")
def list_runs_command(
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
) -> None:
    """List stored benchmark runs (most recent first)."""
    store = BenchmarkStore(db_path)
    runs = store.list_runs()
    if not runs:
        console.print(f"No runs found in {store.storage_label()}.")
        return
    table = Table(title=f"Benchmark Runs: {store.storage_label()}")
    table.add_column("ID", justify="right")
    table.add_column("Created")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Machine")
    table.add_column("Models")
    for run in runs:
        models = json.loads(run["models_json"]) if run["models_json"] else []
        preview = ", ".join(models[:3]) + (f" (+{len(models) - 3})" if len(models) > 3 else "")
        machine = run["machine_label"] or run["machine_id"] or "-"
        table.add_row(str(run["id"]), (run["created_at"] or "")[:19], run["mode"], run["status"], machine, preview)
    console.print(table)


def _confusion_table(model_id: str, scope: str, matrix: dict[str, dict[str, int]]) -> Table:
    table = Table(title=f"Confusion matrix — {model_id} ({scope})")
    table.add_column("actual \\ pred")
    for predicted in CONFUSION_PREDICTION_LABELS:
        table.add_column(predicted.strip("_"), justify="right")
    for actual in ALLOWED_LABELS:
        counts = matrix.get(actual, {})
        table.add_row(actual, *[str(int(counts.get(predicted, 0) or 0)) for predicted in CONFUSION_PREDICTION_LABELS])
    return table


@app.command("results")
def results_command(
    run_id: Annotated[int, typer.Option("--run-id", help="Run id to summarise.")],
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    confusion: Annotated[bool, typer.Option("--confusion", help="Also print per-model confusion matrices.")] = False,
) -> None:
    """Show stored metrics (and optionally confusion matrices) for a run."""
    store = BenchmarkStore(db_path)
    metric_rows = store.fetch_metrics(run_id)
    if not metric_rows:
        console.print(f"No metrics found for run {run_id}. It may still be running or have failed before scoring.")
        raise typer.Exit(code=1)
    cost_by_model = store.run_cost_by_model(run_id)

    table = Table(title=f"Run {run_id} Metrics")
    for column in ("Model", "Scope", "Rows", "Accuracy", "Bal Acc", "MCC", "Macro F1", "Latency", "Tokens", "Cost", "Invalid", "Errors"):
        table.add_column(column, justify="right" if column not in {"Model", "Scope"} else "left")
    parsed = [(row["model_id"], row["scope"], json.loads(row["metrics_json"])) for row in metric_rows]
    scope_order = {"primary": 0, "all": 1}
    parsed.sort(key=lambda item: (scope_order.get(item[1], 2), -item[2]["accuracy"], item[0]))
    for model_id, scope, metric in parsed:
        latency = metric.get("mean_latency_ms")
        tokens = metric.get("total_tokens") or 0
        cost = cost_by_model.get(model_id)
        table.add_row(
            model_id,
            scope,
            str(metric["row_count"]),
            f"{metric['accuracy']:.4f}",
            f"{metric.get('balanced_accuracy', 0.0):.4f}",
            f"{metric.get('mcc', 0.0):.4f}",
            f"{metric['macro_f1']:.4f}",
            f"{latency:.0f} ms" if isinstance(latency, (int, float)) else "-",
            f"{int(tokens):,}" if tokens else "-",
            f"${cost:.4f}" if isinstance(cost, (int, float)) else "-",
            str(metric["invalid_output_count"]),
            str(metric["api_error_count"]),
        )
    console.print(table)

    if confusion:
        for model_id, scope, metric in parsed:
            matrix = metric.get("confusion_matrix") or {}
            if matrix:
                console.print(_confusion_table(model_id, scope, matrix))


@app.command("compare")
def compare_command(
    run_a: Annotated[int, typer.Option("--run-a", help="Run id for model A.")],
    model_a: Annotated[str, typer.Option("--model-a", help="Model id for side A.")],
    model_b: Annotated[str, typer.Option("--model-b", help="Model id for side B.")],
    run_b: Annotated[int | None, typer.Option("--run-b", help="Run id for model B (defaults to --run-a).")] = None,
    scope: Annotated[str, typer.Option("--scope", help="primary or all.")] = "primary",
    metric: Annotated[str, typer.Option("--metric", help="accuracy or macro_f1.")] = "accuracy",
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    n_resamples: Annotated[int, typer.Option("--n-resamples", help="Bootstrap resamples for the CIs.")] = 1000,
    confidence: Annotated[float, typer.Option("--confidence", help="CI confidence level.")] = 0.95,
    seed: Annotated[int, typer.Option("--seed", help="Bootstrap seed for reproducibility.")] = DEFAULT_SEED,
    alpha: Annotated[float, typer.Option("--alpha", help="Significance threshold for McNemar.")] = 0.05,
) -> None:
    """Compare two models with paired McNemar's test and bootstrap CIs."""
    if scope not in {"primary", "all"}:
        raise typer.BadParameter("scope must be primary or all")
    if metric not in {"accuracy", "macro_f1"}:
        raise typer.BadParameter("metric must be accuracy or macro_f1")

    target_a = ModelTarget(run_a, model_a)
    target_b = ModelTarget(run_b if run_b is not None else run_a, model_b)
    result = compare_models(
        BenchmarkStore(db_path),
        target_a,
        target_b,
        scope=scope,
        metric=metric,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    if result.n_paired == 0:
        console.print(
            f"No overlapping {scope}-scope rows between {target_a.label()} and {target_b.label()}. "
            "Compare models evaluated on the same dataset selection."
        )
        raise typer.Exit(code=1)

    confidence_pct = f"{confidence * 100:.0f}"
    table = Table(title=f"Comparison ({metric}, {scope} scope, n={result.n_paired})")
    table.add_column("Target")
    table.add_column(metric.replace("_", " ").title(), justify="right")
    table.add_column(f"{confidence_pct}% CI", justify="right")
    for target, ci in ((target_a, result.ci_a), (target_b, result.ci_b)):
        table.add_row(target.label(), f"{float(ci['point']):.4f}", f"[{float(ci['lower']):.4f}, {float(ci['upper']):.4f}]")
    console.print(table)

    mcnemar = result.mcnemar
    leader = target_a if result.point_a >= result.point_b else target_b
    significant = result.is_significant(alpha)
    verdict = (
        f"{leader.model_id} is higher and the difference is statistically significant"
        if significant
        else "the difference is not statistically significant"
    )
    console.print(
        f"McNemar p = {float(mcnemar['p_value']):.4f} ({mcnemar['method']}, "
        f"discordant b={mcnemar['b_a_correct_b_wrong']} / c={mcnemar['c_a_wrong_b_correct']}). "
        f"At α={alpha}, {verdict}."
    )


@app.command("run-prompt-suite")
def run_prompt_suite_command(
    models: Annotated[list[str], typer.Option("--models", "-m", help="Model id. Repeat for multiple models.")],
    base_prompt_id: Annotated[
        str,
        typer.Option("--base-prompt-id", help="Prompt id to perturb into a variant family."),
    ] = "default_label_only",
    include: Annotated[
        list[str] | None,
        typer.Option("--include", help="Perturbation families: label_order and/or paraphrase. Repeat to combine."),
    ] = None,
    mode: Annotated[str, typer.Option("--mode", help="pilot or full.")] = "pilot",
    dataset_path: Annotated[Path, typer.Option("--dataset-path")] = DEFAULT_DATASET_PATH,
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    prompts_path: Annotated[Path, typer.Option("--prompts-path")] = DEFAULT_PROMPTS_PATH,
    provider: Annotated[
        str,
        typer.Option("--provider", help="Model provider: openrouter or ollama."),
    ] = os.getenv("SENTIMENT_BENCH_PROVIDER", DEFAULT_PROVIDER),
    base_url: Annotated[str, typer.Option("--base-url")] = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama host URL, e.g. http://desktop-pc:11434."),
    ] = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
    sample_per_class: Annotated[int, typer.Option("--sample-per-class")] = DEFAULT_PILOT_PER_CLASS,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    temperature: Annotated[float, typer.Option("--temperature")] = DEFAULT_TEMPERATURE,
    max_completion_tokens: Annotated[int, typer.Option("--max-completion-tokens")] = DEFAULT_MAX_COMPLETION_TOKENS,
    reasoning_max_tokens: Annotated[int, typer.Option("--reasoning-max-tokens")] = DEFAULT_REASONING_MAX_COMPLETION_TOKENS,
    concurrency: Annotated[int, typer.Option("--concurrency")] = DEFAULT_CONCURRENCY,
    retries: Annotated[int, typer.Option("--retries")] = DEFAULT_RETRIES,
) -> None:
    """Run a model across a family of prompt perturbations (one run per variant).

    Every variant evaluates the same seeded row selection, so the resulting runs
    can be fed straight into `prompt-sensitivity` to quantify prompt robustness.
    """
    if mode not in {"pilot", "full"}:
        raise typer.BadParameter("mode must be pilot or full")
    if not models:
        raise typer.BadParameter("At least one --models value is required")
    resolved_provider = _resolve_provider(provider)
    endpoint = endpoint_for_provider(resolved_provider, base_url=base_url, ollama_host=ollama_host)
    base_prompt = _resolve_prompt(base_prompt_id, prompts_path)
    families = tuple(include) if include else ("label_order", "paraphrase")
    try:
        variants = generate_prompt_suite(base_prompt, include=families)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(f"Generated {len(variants)} prompt variant(s) from {base_prompt_id!r}: {', '.join(families)}")

    async def main() -> list[tuple[str, int]]:
        produced: list[tuple[str, int]] = []
        store = BenchmarkStore(db_path)
        async with make_llm_client(resolved_provider, base_url=base_url, ollama_host=ollama_host) as client:
            runner = BenchmarkRunner(client=client, store=store)
            for variant in variants:
                config = _make_run_config(
                    models=models,
                    prompt=variant,
                    mode=mode,
                    dataset_path=dataset_path,
                    db_path=db_path,
                    base_url=endpoint,
                    provider=resolved_provider,
                    sample_per_class=sample_per_class,
                    seed=seed,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_max_tokens=reasoning_max_tokens,
                    concurrency=concurrency,
                    retries=retries,
                )
                console.print(f"Running variant {variant.prompt_id!r}...")
                summary = await runner.run(config, callback=lambda message: console.print(message))
                produced.append((variant.prompt_id, summary.run_id))
        return produced

    produced = asyncio.run(main())
    table = Table(title="Prompt suite runs")
    table.add_column("Variant prompt id")
    table.add_column("Run id", justify="right")
    for prompt_id, run_id in produced:
        table.add_row(prompt_id, str(run_id))
    console.print(table)
    run_id_list = " ".join(f"--run-id {run_id}" for _, run_id in produced)
    console.print(f"Analyse with: sentiment-bench prompt-sensitivity {run_id_list} --model {models[0]}")


@app.command("prompt-sensitivity")
def prompt_sensitivity_command(
    run_ids: Annotated[list[int], typer.Option("--run-id", help="Run id of a prompt variant. Repeat for each variant.")],
    model: Annotated[str, typer.Option("--model", help="Model id to analyse across the runs.")],
    scope: Annotated[str, typer.Option("--scope", help="primary or all.")] = "primary",
    metric: Annotated[str, typer.Option("--metric", help=f"One of: {', '.join(SENSITIVITY_METRICS)}.")] = "accuracy",
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    latex_output: Annotated[
        Path | None,
        typer.Option("--latex-output", help="Also write a dissertation-ready booktabs LaTeX table to this path."),
    ] = None,
) -> None:
    """Quantify how a model's metric varies across a family of prompt variants."""
    if scope not in {"primary", "all"}:
        raise typer.BadParameter("scope must be primary or all")
    if metric not in SENSITIVITY_METRICS:
        raise typer.BadParameter(f"metric must be one of {', '.join(SENSITIVITY_METRICS)}")
    if len(run_ids) < 2:
        raise typer.BadParameter("Provide at least two --run-id values to measure sensitivity")
    try:
        result = prompt_sensitivity(BenchmarkStore(db_path), run_ids, model, scope=scope, metric=metric)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Prompt sensitivity — {model} ({metric}, {scope} scope, n={result.n})")
    table.add_column("Variant prompt id")
    table.add_column("Run", justify="right")
    table.add_column(metric.replace("_", " ").title(), justify="right")
    for variant in sorted(result.variants, key=lambda item: item.value, reverse=True):
        table.add_row(variant.prompt_id, str(variant.run_id), f"{variant.value:.4f}")
    console.print(table)
    console.print(
        f"mean={result.mean:.4f}  std={result.std:.4f}  min={result.minimum:.4f}  "
        f"max={result.maximum:.4f}  spread={result.spread:.4f}  cv={result.cv:.4f}"
    )
    if latex_output is not None:
        latex_output.parent.mkdir(parents=True, exist_ok=True)
        latex_output.write_text(sensitivity_table_latex(result), encoding="utf-8")
        console.print(f"LaTeX table written to {latex_output}")


@app.command("agreement")
def agreement_command(
    run_id: Annotated[int, typer.Option("--run-id", help="Run id to analyse.")],
    scope: Annotated[str, typer.Option("--scope", help="primary or all.")] = "primary",
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
) -> None:
    """Inter-model agreement (Cohen's, Fleiss' kappa, Krippendorff's alpha) for a run."""
    if scope not in {"primary", "all"}:
        raise typer.BadParameter("scope must be primary or all")
    result = run_agreement(BenchmarkStore(db_path), run_id, scope)
    if result is None:
        console.print(f"Need at least two models with valid {scope}-scope predictions in run {run_id} to measure agreement.")
        raise typer.Exit(code=1)

    def _fmt(value: float | None) -> str:
        return f"{value:.4f}" if isinstance(value, (int, float)) else "-"

    summary = Table(title=f"Inter-model agreement — run {run_id} ({scope} scope)")
    summary.add_column("Statistic")
    summary.add_column("Value", justify="right")
    summary.add_row("Raters (models)", str(result.n_raters))
    summary.add_row("Items rated by all", str(result.n_units_all_raters))
    summary.add_row("Items rated by ≥2", str(result.n_units))
    summary.add_row("Observed agreement", _fmt(result.observed_agreement))
    summary.add_row("Fleiss' kappa", _fmt(result.fleiss_kappa))
    summary.add_row("Krippendorff's alpha", _fmt(result.krippendorff_alpha))
    console.print(summary)

    if result.pairwise_cohen_kappa:
        pairwise = Table(title="Pairwise Cohen's kappa")
        pairwise.add_column("Model A")
        pairwise.add_column("Model B")
        pairwise.add_column("Kappa", justify="right")
        pairwise.add_column("n", justify="right")
        for pair in result.pairwise_cohen_kappa:
            pairwise.add_row(pair["rater_a"], pair["rater_b"], f"{pair['kappa']:.4f}", str(pair["n"]))
        console.print(pairwise)


@app.command("tui")
def tui() -> None:
    from .tui import SentimentBenchmarkApp

    SentimentBenchmarkApp().run()


# ------------------------------------------------------------------
# Self-consistency CLI commands
# ------------------------------------------------------------------


@app.command("run-self-consistency")
def run_self_consistency(
    model: Annotated[str, typer.Option("--model", "-m", help="Model id to sample.")],
    mode: Annotated[str, typer.Option("--mode", help="pilot or full.")] = "pilot",
    prompt_id: Annotated[str, typer.Option("--prompt-id", help="Prompt id from configs/default_prompts.toml.")] = "default_label_only",
    dataset_path: Annotated[Path, typer.Option("--dataset-path")] = DEFAULT_DATASET_PATH,
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    prompts_path: Annotated[Path, typer.Option("--prompts-path")] = DEFAULT_PROMPTS_PATH,
    provider: Annotated[
        str,
        typer.Option("--provider", help="Model provider: openrouter or ollama."),
    ] = os.getenv("SENTIMENT_BENCH_PROVIDER", DEFAULT_PROVIDER),
    base_url: Annotated[str, typer.Option("--base-url")] = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama host URL, e.g. http://desktop-pc:11434."),
    ] = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
    sample_per_class: Annotated[int, typer.Option("--sample-per-class")] = DEFAULT_PILOT_PER_CLASS,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    temperature: Annotated[
        float,
        typer.Option("--temperature", "-t", help="Sampling temperature (default 0.7). Use > 0 for diversity."),
    ] = 0.7,
    num_samples: Annotated[
        int,
        typer.Option("--num-samples", "-n", help="Number of repeated samples per row."),
    ] = 5,
    max_completion_tokens: Annotated[int, typer.Option("--max-completion-tokens")] = DEFAULT_MAX_COMPLETION_TOKENS,
    concurrency: Annotated[int, typer.Option("--concurrency")] = DEFAULT_CONCURRENCY,
    retries: Annotated[int, typer.Option("--retries")] = DEFAULT_RETRIES,
) -> None:
    """Run a model multiple times at temperature > 0 to measure self-consistency.

    This is the core experimental data-collection tool for the dissertation.
    Each row is classified ``num_samples`` times at the given ``temperature``,
    producing a label distribution whose entropy quantifies LLM sentiment ambiguity.
    """
    if mode not in {"pilot", "full"}:
        raise typer.BadParameter("mode must be pilot or full")
    if num_samples < 2:
        raise typer.BadParameter("--num-samples must be >= 2")
    if temperature < 0.0:
        raise typer.BadParameter("--temperature must be >= 0.0")
    resolved_provider = _resolve_provider(provider)

    prompt = _resolve_prompt(prompt_id, prompts_path)

    async def main() -> None:
        store = BenchmarkStore(db_path)
        async with make_llm_client(resolved_provider, base_url=base_url, ollama_host=ollama_host) as client:
            runner = SelfConsistencyRunner(client=client, store=store)
            result = await runner.run(
                model_id=model,
                prompt=prompt,
                temperature=temperature,
                num_samples=num_samples,
                mode=mode,
                dataset_path=str(dataset_path),
                max_completion_tokens=max_completion_tokens,
                concurrency=concurrency,
                retries=retries,
                seed=seed,
                sample_per_class=sample_per_class,
                callback=lambda message: console.print(message),
            )
        _print_sc_result(result)

    asyncio.run(main())


def _print_sc_result(result: SelfConsistencyResult) -> None:
    """Pretty-print a SelfConsistencyResult to the console."""
    box = Table(title=f"Self-Consistency — {result.model_id}")
    box.add_column("Metric")
    box.add_column("Value", justify="right")
    box.add_row("Temperature", f"{result.temperature}")
    box.add_row("Samples/row", str(result.num_samples))
    box.add_row("Rows", str(result.n_rows))
    box.add_row("Scope", result.scope)
    box.add_row("Mean entropy", f"{result.mean_entropy:.4f}")
    box.add_row("Median entropy", f"{result.median_entropy:.4f}")
    box.add_row("Mean majority fraction", f"{result.mean_majority_fraction:.4f}")
    box.add_row("Consistency rate (entropy=0)", f"{result.consistency_rate:.4f}")
    box.add_row("Majority-vote accuracy", f"{result.majority_vote_accuracy:.4f}")
    box.add_row("Majority-vote balanced acc", f"{result.majority_vote_balanced_accuracy:.4f}")
    if result.conflicting_entropy is not None:
        box.add_row("Conflicting rows", str(result.conflicting_rows))
        box.add_row("Conflicting mean entropy", f"{result.conflicting_entropy:.4f}")
    if result.non_conflicting_entropy is not None:
        box.add_row("Non-conflicting rows", str(result.non_conflicting_rows))
        box.add_row("Non-conflicting mean entropy", f"{result.non_conflicting_entropy:.4f}")
    gap = result.entropy_gap
    if gap is not None:
        # gap = non_conflicting - conflicting, so gap < 0 means conflicting rows
        # have the higher entropy.
        direction = "higher" if gap < 0 else "lower"
        box.add_row("Entropy gap", f"{gap:.4f} ({direction} on conflicting rows)")
    if result.total_cost is not None:
        box.add_row("Total cost", f"${result.total_cost:.4f}")
    console.print(box)


@app.command("self-consistency")
def self_consistency_command(
    sc_run_id: Annotated[int, typer.Option("--sc-run-id", help="Self-consistency run id.")],
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    top_rows: Annotated[
        int | None,
        typer.Option("--top-rows", help="Show the N most ambiguous rows (highest entropy)."),
    ] = None,
) -> None:
    """Analyse an existing self-consistency run.

    Displays aggregate entropy statistics and, with --top-rows, the most
    ambiguous individual sentences.
    """
    store = BenchmarkStore(db_path)
    run_info = store.sc_run_by_id(sc_run_id)
    if run_info is None:
        console.print(f"No self-consistency run found with id {sc_run_id}.")
        raise typer.Exit(code=1)

    result = store.build_sc_row_consistency(
        sc_run_id,
        model_id=str(run_info["model_id"]),
        temperature=float(run_info["temperature"]),
        num_samples=int(run_info["num_samples"]),
        scope=str(run_info["scope"]),
    )

    _print_sc_result(result)

    if top_rows and result.rows:
        sorted_rows = sorted(result.rows, key=lambda r: (-r.entropy, -r.n_valid))
        top = sorted_rows[:top_rows]
        top_table = Table(title=f"Top {len(top)} Most Ambiguous Rows (highest entropy)")
        top_table.add_column("Row")
        top_table.add_column("Sentence")
        top_table.add_column("Label")
        top_table.add_column("Conflict")
        top_table.add_column("Entropy", justify="right")
        top_table.add_column("Distribution", justify="right")
        top_table.add_column("Valid", justify="right")
        for r in top:
            dist = ", ".join(f"{k}={v}" for k, v in sorted(r.label_counts.items(), key=lambda x: -x[1]))
            top_table.add_row(
                str(r.row_number),
                r.sentence[:60],
                r.hidden_label,
                "✓" if r.is_conflicting_duplicate else "",
                f"{r.entropy:.3f}",
                dist,
                str(r.n_valid),
            )
        console.print(top_table)


@app.command("self-consistency-list")
def self_consistency_list_command(
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
) -> None:
    """List existing self-consistency runs."""
    store = BenchmarkStore(db_path)
    runs = store.list_sc_runs()
    if not runs:
        console.print("No self-consistency runs found.")
        return
    table = Table(title="Self-Consistency Runs")
    table.add_column("ID", justify="right")
    table.add_column("Created")
    table.add_column("Model")
    table.add_column("T", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Cost", justify="right")
    for run in runs:
        table.add_row(
            str(run["id"]),
            (run["created_at"] or "")[:19],
            str(run["model_id"]),
            f"{run['temperature']:.1f}",
            str(run["num_samples"]),
            str(run["mode"]),
            str(run["status"]),
            f"${run['total_cost']:.4f}" if run["total_cost"] else "-",
        )
    console.print(table)


@app.command("sc-compare-by-conflict")
def sc_compare_by_conflict_command(
    sc_run_id: Annotated[int, typer.Option("--sc-run-id", help="Self-consistency run id.")],
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
) -> None:
    """Test the dissertation hypothesis: do conflicting-duplicate rows have higher entropy?

    Compares mean entropy between rows with conflicting human annotations and
    rows without. This is the primary experimental validation of LLM disagreement
    as a proxy for ground-truth uncertainty.
    """
    store = BenchmarkStore(db_path)
    run_info = store.sc_run_by_id(sc_run_id)
    if run_info is None:
        console.print(f"No self-consistency run found with id {sc_run_id}.")
        raise typer.Exit(code=1)

    result = store.build_sc_row_consistency(
        sc_run_id,
        model_id=str(run_info["model_id"]),
        temperature=float(run_info["temperature"]),
        num_samples=int(run_info["num_samples"]),
        scope="all",
    )

    conflicting = [r for r in result.rows if r.is_conflicting_duplicate]
    non_conflicting = [r for r in result.rows if not r.is_conflicting_duplicate]

    if not conflicting:
        console.print("No conflicting-duplicate rows found in this run.")
        raise typer.Exit(code=1)
    if not non_conflicting:
        console.print("No non-conflicting rows found.")
        raise typer.Exit(code=1)

    c_entropy = sum(r.entropy for r in conflicting) / len(conflicting)
    nc_entropy = sum(r.entropy for r in non_conflicting) / len(non_conflicting)
    gap = nc_entropy - c_entropy  # negative = supports hypothesis

    table = Table(
        title=f"Hypothesis Test — Conflicting vs Non-Conflicting Entropy\n"
        f"{run_info['model_id']} t={run_info['temperature']} n={run_info['num_samples']}×"
    )
    table.add_column("Group")
    table.add_column("Rows", justify="right")
    table.add_column("Mean Entropy", justify="right")
    table.add_column("Consistency Rate", justify="right")
    table.add_column("Majority Acc", justify="right")
    for group, rows in [("Conflicting", conflicting), ("Non-conflicting", non_conflicting)]:
        mean_e = sum(r.entropy for r in rows) / len(rows)
        cons_rate = sum(1 for r in rows if r.entropy == 0) / len(rows)
        maj_acc = sum(1 for r in rows if r.correct_majority) / len(rows)
        table.add_row(group, str(len(rows)), f"{mean_e:.4f}", f"{cons_rate:.4f}", f"{maj_acc:.4f}")
    console.print(table)

    from scipy.stats import mannwhitneyu

    c_vals = [r.entropy for r in conflicting]
    nc_vals = [r.entropy for r in non_conflicting]
    try:
        stat, p_value = mannwhitneyu(c_vals, nc_vals, alternative="greater")
    except ValueError as exc:
        # scipy raises when every entropy value is identical (e.g. all rows perfectly
        # consistent), so there is no rank variation to test.
        console.print(f"\nMann-Whitney U test could not be computed: {exc}")
    else:
        verdict = "SUPPORTS hypothesis" if p_value < 0.05 else "does NOT support hypothesis"
        console.print(f"\nMann-Whitney U test (conflicting > non-conflicting): U={stat:.1f}, p={p_value:.4f} → {verdict} at α=0.05")

    gap_direction = "Conflicting rows have HIGHER entropy" if gap < 0 else "Non-conflicting rows have higher entropy"
    console.print(f"\nEntropy gap (non-conflicting - conflicting) = {gap:.4f}\n({gap_direction})")


if __name__ == "__main__":
    app()
