from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import time
import tomllib
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .baseline_runner import run_baselines
from .baselines import BASELINE_SPECS
from .comparison import ModelTarget, compare_models
from .constants import (
    ALLOWED_LABELS,
    CONFUSION_PREDICTION_LABELS,
    DEFAULT_BASE_URL,
    DEFAULT_CEREBRAS_CONCURRENCY,
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
from .corpus_scoring import frozen_design_call_counts, load_matrix_config, load_matrix_items, matrix_plan, score_corpus_matrix
from .dataset import compute_stats, load_dataset
from .env import load_env_file
from .exporter import export_run
from .headline_scoring import score_headlines
from .headline_value import HeadlineValueError, analyze_headline_value, collect_scorable_headlines
from .l2_event_study import L2AnalysisError, analyze_l2
from .l3_reliability import L3AnalysisError, analyze_l3
from .latex_tables import sensitivity_table_latex
from .lseg_catalog import build_lseg_catalog
from .lseg_cohort import build_lseg_analysis_cohort
from .lseg_corpus import build_lseg_corpus
from .lseg_presets import (
    LSEG_PRESET_US_MEGA_EIGHT,
    LSEG_PRESETS,
    LSEG_US_MEGA_CAP_1Y_END,
    LSEG_US_MEGA_CAP_1Y_ID,
    LSEG_US_MEGA_CAP_1Y_START,
    lseg_config_from_preset,
    write_lseg_config,
)
from .lseg_source import (
    LsegNewsClient,
    LsegNewsError,
    LsegProgressUpdate,
    check_lseg_news,
    fetch_lseg_news,
    load_lseg_collection_config,
)
from .lseg_validation import create_lseg_validation_sample, evaluate_lseg_annotations
from .models import PromptConfig, RunConfig
from .news_cleaning import NewsCleaningDependencyError
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
from .price_export import PriceExportError, export_lseg_prices
from .prompt_sensitivity import SENSITIVITY_METRICS, prompt_sensitivity
from .prompts import load_prompts
from .providers import endpoint_for_provider, make_llm_client, normalize_provider
from .reliability import run_agreement
from .runner import BenchmarkRunner
from .sc_runner import SelfConsistencyRunner
from .self_consistency import SelfConsistencyResult
from .signal_portfolio import SignalPortfolioConfig, SignalPortfolioError, run_signal_portfolio
from .storage import BenchmarkStore
from .strategies import available as available_strategies
from .strategies import get as get_strategy
from .strategy_research.cli import app as strategy_research_app
from .strategy_sweep import (
    chronological_split_date,
    load_prices_csv,
    load_signals_csv,
    sweep_strategy,
    write_sweep_csv,
)
from .trading_analysis import TradingAnalysisError, analyze_trading_run
from .trading_plots import plot_sweep_heatmap
from .trading_strategy import (
    TradingStrategyError,
    describe_trading_plan,
    load_trading_config,
)
from .trading_strategy import (
    run_trading_strategy as execute_trading_strategy,
)
from .week6_model_comparison import (
    Week6ModelComparisonConfig,
    run_week6_model_comparison,
)
from .week6_pnl import Week6PnlConfig, Week6PnlError, run_week6_pnl

console = Console()
app = typer.Typer(help="Benchmark OpenRouter and Ollama LLMs on dissertation sentiment data.")
app.add_typer(strategy_research_app, name="strategy")
load_env_file()

# Tavily batch fetches retry transient failures (timeouts, dropped connections)
# before recording a fetch as failed and moving on.
FETCH_ATTEMPTS = 3
FETCH_RETRY_BASE_DELAY_SECONDS = 5.0


def _comma_separated_floats(value: str, *, option: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"{option} must be a comma-separated numeric list") from exc
    if not parsed:
        raise typer.BadParameter(f"{option} cannot be empty")
    return parsed


def _comma_separated_ints(value: str, *, option: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"{option} must be a comma-separated integer list") from exc
    if not parsed:
        raise typer.BadParameter(f"{option} cannot be empty")
    return parsed


@app.command("analyze-week6-pnl")
def analyze_week6_pnl_command(
    signals: Annotated[Path, typer.Option("--signals", help="Existing daily_signals.csv input.")],
    prices: Annotated[Path, typer.Option("--prices", help="Corresponding prices.csv input.")],
    run_id: Annotated[str, typer.Option("--run-id", help="New immutable run directory name.")],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Parent directory for Week 6 runs."),
    ] = Path("results/week6_pnl"),
    scorer: Annotated[str, typer.Option("--scorer", help="One stock-day sentiment signal to analyse.")] = "headline/sentiment_all",
    starting_capital: Annotated[float, typer.Option("--starting-capital", min=0.01)] = 100_000.0,
    transaction_cost_bps_per_side: Annotated[
        float,
        typer.Option("--transaction-cost-bps-per-side", min=0.0),
    ] = 10.0,
    development_fraction: Annotated[float, typer.Option("--development-fraction", min=0.01, max=0.99)] = 0.6,
    thresholds: Annotated[str, typer.Option("--thresholds", help="Small development-only threshold grid.")] = "0,0.05,0.10",
    holding_periods: Annotated[
        str,
        typer.Option("--holding-periods", help="Small development-only holding-session grid."),
    ] = "1,3,5,7",
    min_joint_active_dates: Annotated[
        int,
        typer.Option("--min-joint-active-dates", min=1, help="Correlation support threshold."),
    ] = 10,
    low_correlation_stock_count: Annotated[
        int | None,
        typer.Option("--low-correlation-stock-count", min=2, help="Subset size; default is ceil(sqrt(universe))."),
    ] = None,
    skip_low_correlation_portfolio: Annotated[
        bool,
        typer.Option(
            "--skip-low-correlation-portfolio",
            help="Run the all-stock acceptance portfolio without constructing the optional stock subset.",
        ),
    ] = False,
) -> None:
    """Run the leak-controlled exploratory Week 6 daily P&L analysis."""

    config = Week6PnlConfig(
        run_id=run_id,
        scorer_id=scorer,
        starting_capital=starting_capital,
        transaction_cost_bps_per_side=transaction_cost_bps_per_side,
        development_fraction=development_fraction,
        thresholds=_comma_separated_floats(thresholds, option="--thresholds"),
        holding_periods=_comma_separated_ints(holding_periods, option="--holding-periods"),
        min_joint_active_dates=min_joint_active_dates,
        low_correlation_stock_count=low_correlation_stock_count,
        include_low_correlation_portfolio=not skip_low_correlation_portfolio,
    )
    try:
        result = run_week6_pnl(
            signals,
            prices,
            output_root,
            config,
            command=shlex.join(sys.argv),
            repo_root=Path.cwd(),
        )
    except Week6PnlError as exc:
        console.print(f"[red]Week 6 P&L analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Week 6 P&L analysis complete:[/green] {result.output_dir}")
    console.print(
        f"Frozen rule: threshold={result.selected_threshold:g}, holding={result.selected_holding_period}; "
        f"evaluation begins {result.split_date}."
    )


@app.command("compare-week6-models")
def compare_week6_models_command(
    signals: Annotated[Path, typer.Option("--signals", help="Multi-scorer daily_signals.csv input.")],
    prices: Annotated[Path, typer.Option("--prices", help="Corresponding daily OHLCV prices CSV.")],
    run_id: Annotated[str, typer.Option("--run-id", help="New immutable comparison directory name.")],
    scorers: Annotated[str, typer.Option("--scorers", help="Comma-separated scorer ids compared on identical company-days.")],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Parent directory for Week 6 model comparisons."),
    ] = Path("results/week6_model_comparison"),
    starting_capital: Annotated[float, typer.Option("--starting-capital", min=0.01)] = 100_000.0,
    transaction_cost_bps_per_side: Annotated[
        float,
        typer.Option("--transaction-cost-bps-per-side", min=0.0),
    ] = 10.0,
    development_fraction: Annotated[float, typer.Option("--development-fraction", min=0.01, max=0.99)] = 0.7,
    threshold_quantiles: Annotated[
        str,
        typer.Option("--threshold-quantiles", help="Development-only absolute-score quantile gates."),
    ] = "0,0.5,0.7",
    holding_periods: Annotated[
        str,
        typer.Option("--holding-periods", help="Development-only holding-session grid."),
    ] = "1,3,5,7",
    min_joint_active_dates: Annotated[
        int,
        typer.Option("--min-joint-active-dates", min=1, help="Joint activity required for a supported correlation."),
    ] = 10,
    min_stock_active_days: Annotated[
        int,
        typer.Option("--min-stock-active-days", min=1, help="Development activity required for diversification."),
    ] = 10,
    negative_portfolio_stock_count: Annotated[
        int | None,
        typer.Option("--negative-portfolio-stock-count", min=2, help="Maximum negative-correlation clique size."),
    ] = None,
    maximum_stock_weight: Annotated[
        float,
        typer.Option("--maximum-stock-weight", min=0.01, max=1.0, help="Development-fitted portfolio weight cap."),
    ] = 0.6,
) -> None:
    """Compare frozen funded model strategies and a gated negative-correlation portfolio."""

    scorer_ids = tuple(value.strip() for value in scorers.split(",") if value.strip())
    config = Week6ModelComparisonConfig(
        run_id=run_id,
        scorer_ids=scorer_ids,
        starting_capital=starting_capital,
        transaction_cost_bps_per_side=transaction_cost_bps_per_side,
        development_fraction=development_fraction,
        threshold_quantiles=_comma_separated_floats(threshold_quantiles, option="--threshold-quantiles"),
        holding_periods=_comma_separated_ints(holding_periods, option="--holding-periods"),
        min_joint_active_dates=min_joint_active_dates,
        min_stock_active_days=min_stock_active_days,
        negative_portfolio_stock_count=negative_portfolio_stock_count,
        maximum_stock_weight=maximum_stock_weight,
    )
    try:
        result = run_week6_model_comparison(
            signals,
            prices,
            output_root,
            config,
            command=shlex.join(sys.argv),
            repo_root=Path.cwd(),
        )
    except Week6PnlError as exc:
        console.print(f"[red]Week 6 model comparison failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Week 6 model comparison complete:[/green] {result.output_dir}")
    console.print(f"Frozen evaluation begins {result.split_date}.")
    for scorer_id, quantile, threshold, holding in result.selected_rules:
        console.print(f"{scorer_id}: development gate q={quantile:.0%} -> {threshold:.4f}; holding={holding}.")


@app.command("optimize-signal-portfolio")
def optimize_signal_portfolio_command(
    signals: Annotated[Path, typer.Option("--signals", help="Multi-scorer daily_signals.csv input.")],
    prices: Annotated[Path, typer.Option("--prices", help="Corresponding daily OHLCV prices CSV.")],
    run_id: Annotated[str, typer.Option("--run-id", help="New immutable study directory name.")],
    scorers: Annotated[str, typer.Option("--scorers", help="Comma-separated scorer ids combined on identical company-days.")],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Parent directory for signal-portfolio studies."),
    ] = Path("results/signal_portfolio"),
    return_variant: Annotated[
        str,
        typer.Option("--return-variant", help="Selection surface: gross (before cost) or net (after cost)."),
    ] = "gross",
    threshold: Annotated[
        float,
        typer.Option("--threshold", min=0.0, help="Shared absolute score gate applied to every scorer."),
    ] = 0.0,
    holding_period: Annotated[
        int,
        typer.Option("--holding-period", min=1, help="Shared holding period in sessions."),
    ] = 1,
    transaction_cost_bps_per_side: Annotated[
        float,
        typer.Option("--transaction-cost-bps-per-side", min=0.0, help="Cost charged on absolute position change."),
    ] = 10.0,
    development_fraction: Annotated[
        float,
        typer.Option("--development-fraction", min=0.05, max=0.95, help="Fraction of news dates used to select."),
    ] = 0.7,
    bootstrap_replications: Annotated[
        int,
        typer.Option("--bootstrap-replications", min=1, help="Replications for the null and interval bootstraps."),
    ] = 2_000,
    random_seed: Annotated[int, typer.Option("--random-seed", help="Seed for both bootstraps.")] = 20260729,
) -> None:
    """Select Sharpe-maximising signal subsets and solve two-signal weights by Lagrange multipliers."""

    scorer_ids = tuple(value.strip() for value in scorers.split(",") if value.strip())
    try:
        config = SignalPortfolioConfig(
            run_id=run_id,
            scorer_ids=scorer_ids,
            return_variant=return_variant,
            threshold=threshold,
            holding_period=holding_period,
            transaction_cost_bps_per_side=transaction_cost_bps_per_side,
            development_fraction=development_fraction,
            bootstrap_replications=bootstrap_replications,
            random_seed=random_seed,
        )
        result = run_signal_portfolio(
            signals,
            prices,
            output_root,
            config,
            command=shlex.join(sys.argv),
            repo_root=Path.cwd(),
        )
    except (SignalPortfolioError, Week6PnlError) as exc:
        console.print(f"[red]Signal portfolio study failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Signal portfolio study complete:[/green] {result.output_dir}")
    console.print(f"Frozen evaluation begins {result.split_date}.")
    console.print(f"Best pair: {', '.join(result.best_pair)}")
    console.print(f"Best triple: {', '.join(result.best_triple)}")
    console.print(f"Add/drop-stable subsets: {len(result.stable_subsets)}; global best: {', '.join(result.global_best_subset)}")


@app.command("score-corpus-matrix")
def score_corpus_matrix_command(
    input_path: Annotated[Path, typer.Option("--input", help="Main cohort JSONL or benchmark CSV.")],
    subset_path: Annotated[Path, typer.Option("--subset", help="L3 prompt-facet subset in the same format.")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    kind: Annotated[str, typer.Option("--kind", help="lseg or benchmark.")],
    config: Annotated[Path, typer.Option("--config")] = Path("configs/crossed_scoring.toml"),
    prompts_path: Annotated[Path, typer.Option("--prompts-path")] = DEFAULT_PROMPTS_PATH,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show call counts without contacting providers.")] = False,
) -> None:
    """Run or size the resumable crossed-model corpus scoring matrix."""
    try:
        matrix_config = load_matrix_config(config)
        items = load_matrix_items(input_path, kind)
        subset_ids = {item.item_id for item in load_matrix_items(subset_path, kind)}
        plan = matrix_plan(matrix_config, items, subset_ids)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="Crossed Corpus Scoring Plan")
    table.add_column("Metric")
    table.add_column("Calls", justify="right")
    table.add_row("This run", f"{plan.total_calls:,}")
    table.add_row("Hosted", f"{plan.hosted_calls:,}")
    table.add_row("Local", f"{plan.local_calls:,}")
    frozen = frozen_design_call_counts()
    table.add_row("Reference LSEG + benchmark total", f"{frozen['total']:,}")
    console.print(table)
    if dry_run:
        return

    async def main() -> None:
        async with AsyncExitStack() as stack:
            clients = {
                "openrouter": await stack.enter_async_context(
                    make_llm_client("openrouter", base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL))
                ),
                "cerebras": await stack.enter_async_context(make_llm_client("cerebras")),
                "ollama": await stack.enter_async_context(
                    make_llm_client(
                        "ollama",
                        ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
                        structured_label_output=False,
                    )
                ),
            }
            await score_corpus_matrix(
                input_path=input_path,
                subset_path=subset_path,
                input_kind=kind,
                config_path=config,
                prompts_path=prompts_path,
                output_dir=output_dir,
                clients=clients,
            )

    asyncio.run(main())


@app.command("analyze-l2")
def analyze_l2_command(
    scores: Annotated[Path, typer.Option("--scores", help="Completed crossed-score JSONL.")],
    prices: Annotated[Path, typer.Option("--prices", help="Cached stock and ^GSPC daily prices CSV.")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Run the L2 clustered agreement/ambiguity event study."""
    try:
        result = analyze_l2(scores, prices, output_dir)
    except L2AnalysisError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"L2 event study written to {result.output_dir}")


@app.command("analyze-l3")
def analyze_l3_command(
    scores: Annotated[Path, typer.Option("--scores")],
    event_returns: Annotated[Path, typer.Option("--event-returns")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    benchmark_scores: Annotated[Path | None, typer.Option("--benchmark-scores")] = None,
    l2_hypotheses: Annotated[Path | None, typer.Option("--l2-hypotheses")] = None,
) -> None:
    """Run the crossed G-study and reliability-aware holdout comparison."""
    try:
        result = analyze_l3(scores, event_returns, output_dir, benchmark_scores, l2_hypotheses)
    except L3AnalysisError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"L3 reliability analysis written to {result.output_dir}")


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


def _resolve_concurrency(provider: str, concurrency: int | None) -> int:
    """Default concurrency per provider when --concurrency is not given.

    Cerebras is quota-paced by the adaptive limiter, so it needs enough
    in-flight requests to reach the model RPM; local/OpenRouter runs keep
    the conservative single-request default.
    """
    if concurrency is not None:
        if concurrency < 1:
            raise typer.BadParameter("--concurrency must be >= 1")
        return concurrency
    if provider == "cerebras":
        return DEFAULT_CEREBRAS_CONCURRENCY
    return DEFAULT_CONCURRENCY


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


@app.command("lseg-init-config")
def lseg_init_config(
    preset: Annotated[
        str,
        typer.Option("--preset", help=f"LSEG config preset: {', '.join(LSEG_PRESETS)}."),
    ] = LSEG_PRESET_US_MEGA_EIGHT,
    collection_id: Annotated[
        str,
        typer.Option("--collection-id", help="Stable collection id. Do not reuse for different settings."),
    ] = LSEG_US_MEGA_CAP_1Y_ID,
    start: Annotated[
        str,
        typer.Option("--start", help="UTC collection start timestamp."),
    ] = LSEG_US_MEGA_CAP_1Y_START,
    end: Annotated[
        str,
        typer.Option("--end", help="UTC collection end timestamp."),
    ] = LSEG_US_MEGA_CAP_1Y_END,
    output: Annotated[
        Path,
        typer.Option("--output", help="TOML config path to create."),
    ] = Path("configs/lseg_us_mega_cap_1y.toml"),
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing config file."),
    ] = False,
) -> None:
    """Create a reproducible LSEG Workspace collection config from a preset."""
    try:
        config = lseg_config_from_preset(preset=preset, collection_id=collection_id, start=start, end=end)
        path = write_lseg_config(config, output, overwrite=overwrite)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="LSEG Config Created")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Preset", preset)
    table.add_row("Collection ID", config.collection_id)
    table.add_row("Date range", f"{config.start} to {config.end}")
    table.add_row("Window days", str(config.window_days or "whole interval"))
    table.add_row("Companies", ", ".join(company.symbol for company in config.companies))
    table.add_row("Output", str(path))
    console.print(table)


@app.command("lseg-news-check")
def lseg_news_check(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="TOML LSEG collection definition used for the entitlement check."),
    ] = Path("configs/lseg_workspace_example.toml"),
    all_companies: Annotated[
        bool,
        typer.Option("--all-companies", help="Check every configured RIC over the complete interval."),
    ] = False,
) -> None:
    """Verify Workspace headline and story access without writing data."""
    try:
        config = load_lseg_collection_config(config_path)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def main() -> None:
        async with _make_lseg_news_client() as client:
            result = await check_lseg_news(config, client, all_companies=all_companies)
        table = Table(title="LSEG Workspace News Check")
        table.add_column("Symbol")
        table.add_column("RIC")
        table.add_column("Headlines", justify="right")
        table.add_column("Story status")
        for check in result["checks"]:
            table.add_row(
                str(check["symbol"]),
                str(check["ric"]),
                str(check["headline_count"]),
                str(check["story_status"]),
            )
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
    max_requests: Annotated[
        int | None,
        typer.Option(
            "--max-requests",
            min=1,
            help="Operational request cap for this invocation; does not change collection identity.",
        ),
    ] = None,
) -> None:
    """Fetch an immutable, resumable LSEG Workspace headline/story collection."""
    try:
        config = load_lseg_collection_config(config_path)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def main() -> None:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"[bold]{config.collection_id}[/bold]",
                        f"[cyan]{len(config.companies)} companies[/cyan]  •  {config.start[:10]} → {config.end[:10]}",
                        f"[cyan]API pace: {config.requests_per_second:g} requests/second[/cyan]",
                        f"[cyan]Request budget: {max_requests or config.max_requests_per_run or 'unbounded'}[/cyan]",
                        f"[dim]Output: {config.raw_dir}[/dim]",
                        "[dim]Saved checkpoints are reused. ETA stabilizes after a few live requests.[/dim]",
                    ]
                ),
                title="[bold cyan]LSEG Collection Monitor[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("[dim]ETA[/dim]"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            auto_refresh=False,
        )
        api_status_line = Text("API control: waiting for first request", style="dim")
        headline_task = progress.add_task(
            "[cyan]Headline windows[/cyan]",
            total=1,
            start=False,
        )
        story_task = progress.add_task(
            "[magenta]Full stories[/magenta]",
            total=1,
            start=False,
            visible=False,
        )
        initialized_phases: set[str] = set()

        def update_progress(update: LsegProgressUpdate) -> None:
            api_status_line.plain = (
                f"API {update.requests_per_second:g}/s cap  |  {update.requests_started:,} requests"
                f"  |  paced {update.paced_waits:,} / {update.paced_wait_seconds:.1f}s"
                f"  |  retries {update.retries:,}  |  safe cursors {update.pagination_anomalies:,}"
            )
            if update.phase == "headlines":
                task_id = headline_task
                description = (
                    f"[cyan]Headlines[/cyan] {update.current} [dim]• {update.unique_headlines:,} unique • ↻{update.checkpointed:,}[/dim]"
                )
            else:
                task_id = story_task
                progress.update(story_task, visible=True)
                progress.update(
                    headline_task,
                    description="[green]✓ Headlines collected[/green]",
                )
                detail = f"• {update.failed_stories:,} unavailable • ↻{update.checkpointed:,}"
                description = f"[magenta]Stories[/magenta] [dim]{detail}[/dim]"
            display_total = max(update.total, 1)
            display_completed = display_total if update.total == 0 else update.completed
            progress.update(
                task_id,
                total=display_total,
                completed=display_completed,
                description=description,
            )
            if update.phase not in initialized_phases:
                initialized_phases.add(update.phase)
                progress.start_task(task_id)

        with Live(
            Group(progress, api_status_line),
            console=console,
            refresh_per_second=4,
            transient=False,
        ):
            async with _make_lseg_news_client() as client:
                result = await fetch_lseg_news(
                    config,
                    client,
                    progress_callback=update_progress,
                    request_budget_override=max_requests,
                )
            if not initialized_phases:
                progress.update(headline_task, visible=False)
                progress.update(story_task, visible=False)
        table = Table(title="LSEG Workspace News Fetch")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Collection ID", config.collection_id)
        table.add_row("Headlines", str(result.headline_count))
        table.add_row("Stories", str(result.story_count))
        table.add_row("Unavailable/failed stories", str(result.failed_story_count))
        table.add_row("Requests this session", str(result.request_count))
        table.add_row("Retries this session", str(result.retry_count))
        table.add_row("Safe cursor terminations", str(result.pagination_anomaly_count))
        table.add_row("Resumed", str(result.resumed))
        table.add_row("Window days", str(config.window_days or "whole interval"))
        table.add_row("Raw collection", str(result.raw_dir))
        table.add_row("Manifest", str(result.manifest_path))
        console.print(table)

    try:
        asyncio.run(main())
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("fetch-lseg-prices")
def fetch_lseg_prices_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="LSEG collection config supplying the symbol-to-RIC universe."),
    ],
    start: Annotated[str, typer.Option("--start", help="First requested price date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option("--end", help="Last requested price date (YYYY-MM-DD).")],
    output: Annotated[Path, typer.Option("--output", help="Destination OHLCV CSV.")],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing CSV and manifest."),
    ] = False,
    ric_overrides_path: Annotated[
        Path | None,
        typer.Option(
            "--ric-overrides",
            help="Optional TOML file with a [rics] symbol-to-RIC table used only for prices.",
        ),
    ] = None,
) -> None:
    """Fetch a hash-manifested LSEG price panel for a configured company universe."""

    try:
        config = load_lseg_collection_config(config_path)
        ric_overrides: dict[str, str] | None = None
        if ric_overrides_path is not None:
            with ric_overrides_path.open("rb") as handle:
                override_payload = tomllib.load(handle)
            raw_rics = override_payload.get("rics")
            if not isinstance(raw_rics, dict) or not all(
                isinstance(symbol, str) and isinstance(ric, str) for symbol, ric in raw_rics.items()
            ):
                raise PriceExportError("price RIC override file must contain a [rics] string table")
            ric_overrides = dict(raw_rics)
        result = export_lseg_prices(
            config,
            start=start,
            end=end,
            output=output,
            overwrite=overwrite,
            ric_overrides=ric_overrides,
        )
    except (OSError, tomllib.TOMLDecodeError, LsegNewsError, PriceExportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="LSEG Price Export")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Symbols", str(result.symbol_count))
    table.add_row("Rows", f"{result.row_count:,}")
    table.add_row("CSV", str(result.output_path))
    table.add_row("Manifest", str(result.manifest_path))
    console.print(table)


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


@app.command("build-lseg-analysis-cohort")
def build_lseg_analysis_cohort_command(
    corpus_manifest: Annotated[Path, typer.Option("--corpus-manifest", help="Completed derived LSEG corpus manifest.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="Analysis cohort TOML configuration."),
    ] = Path("configs/lseg_us_sector_33_analysis.toml"),
) -> None:
    """Build a deterministic, relevance-screened, family-deduplicated LSEG cohort."""
    try:
        result = build_lseg_analysis_cohort(corpus_manifest, config)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="LSEG Analysis Cohort")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Cohort rows", str(result.cohort_count))
    table.add_row("Manifest", str(result.manifest_path))
    table.add_row("Screening index", str(result.screening_path))
    table.add_row("Coverage", str(result.coverage_path))
    console.print(table)


@app.command("lseg-catalog")
def lseg_catalog(
    derived_root: Annotated[
        Path,
        typer.Option("--derived-root", help="Root containing derived LSEG corpora."),
    ] = Path("Data/derived/lseg"),
) -> None:
    """Refresh the local metadata-only LSEG corpus catalog."""
    try:
        result = build_lseg_catalog(derived_root)
    except LsegNewsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="LSEG Corpus Catalog")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Corpora", str(result.corpus_count))
    table.add_row("Eligible articles", str(result.eligible_count))
    table.add_row("Catalog JSON", str(result.catalog_json))
    table.add_row("Catalog CSV", str(result.catalog_csv))
    console.print(table)


@app.command("sample-lseg-validation")
def sample_lseg_validation_command(
    output_dir: Annotated[Path, typer.Option("--output-dir", help="New local-only annotation directory.")],
    corpus_manifest: Annotated[
        Path | None,
        typer.Option("--corpus-manifest", help="Verified derived LSEG corpus manifest."),
    ] = None,
    cohort_manifest: Annotated[
        Path | None,
        typer.Option("--cohort-manifest", help="Verified LSEG analysis cohort manifest."),
    ] = None,
    sample_size: Annotated[int, typer.Option("--sample-size")] = 150,
    double_code_size: Annotated[int, typer.Option("--double-code-size")] = 30,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    double_code_seed: Annotated[int, typer.Option("--double-code-seed")] = 43,
) -> None:
    """Create deterministic primary and double-code LSEG annotation sheets."""
    if (corpus_manifest is None) == (cohort_manifest is None):
        raise typer.BadParameter("provide exactly one of --corpus-manifest or --cohort-manifest")
    try:
        result = create_lseg_validation_sample(
            corpus_manifest or cohort_manifest,  # type: ignore[arg-type]
            output_dir,
            sample_size=sample_size,
            double_code_size=double_code_size,
            seed=seed,
            double_code_seed=double_code_seed,
            cohort_manifest=cohort_manifest,
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
    table.add_row("Relevance agreement", f"{result.relevance_percent_agreement:.1%}")
    table.add_row("Relevance kappa", f"{result.relevance_cohen_kappa:.4f}")
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
    ] = Path("configs/trading_pilot_3co.toml"),
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
            newsapi_client = await stack.enter_async_context(_make_newsapi_client()) if config.newsapi_enabled else None
            tavily_client = await stack.enter_async_context(_make_tavily_news_client()) if config.tavily_gap_fetch else None
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
        table.add_row("Event return rows", str(result.return_count))
        if result.portfolio_summary_count:
            table.add_row("Portfolio trades", str(result.portfolio_trade_count))
            table.add_row("Portfolio days", str(result.portfolio_day_count))
            table.add_row("Funded cases", str(result.portfolio_summary_count))
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


@app.command("score-headlines")
def score_headlines_command(
    model: Annotated[str, typer.Option("--model", "-m", help="Model id to score with, e.g. gpt-oss-120b.")],
    collection_root: Annotated[
        Path,
        typer.Option("--collection-root", help="Collection folder or raw LSEG directory containing headlines.jsonl."),
    ] = Path("Data/collections/lseg_us_sector_33_6m"),
    provider: Annotated[
        str,
        typer.Option("--provider", help="Model provider: openrouter, cerebras, or ollama."),
    ] = os.getenv("SENTIMENT_BENCH_PROVIDER", DEFAULT_PROVIDER),
    base_url: Annotated[str, typer.Option("--base-url")] = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama host URL, e.g. http://desktop-pc:11434."),
    ] = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
    prompt_id: Annotated[
        str,
        typer.Option(
            "--prompt-id",
            help=(
                "Prompt id from configs/default_prompts.toml. label_only prompts score +1/0/-1; "
                "soft_label prompts (e.g. financial_soft_label_base) score P(positive)-P(negative) in [-1, 1]."
            ),
        ),
    ] = "finance_calibrated_label_only",
    prompts_path: Annotated[Path, typer.Option("--prompts-path")] = DEFAULT_PROMPTS_PATH,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Scores CSV (append-only, resumable). Default: <collection>/derived/headline_scores_<model>.csv"),
    ] = None,
    temperature: Annotated[float, typer.Option("--temperature")] = DEFAULT_TEMPERATURE,
    max_completion_tokens: Annotated[int, typer.Option("--max-completion-tokens")] = DEFAULT_MAX_COMPLETION_TOKENS,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Simultaneous API calls. Default: 64 for Cerebras, otherwise 1."),
    ] = None,
    retries: Annotated[int, typer.Option("--retries")] = DEFAULT_RETRIES,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Score at most this many unscored headlines (pilot runs)."),
    ] = None,
    source_code: Annotated[
        list[str] | None,
        typer.Option("--source-code", help="Only score this exact source code; repeat for multiple sources."),
    ] = None,
    direct_company_only: Annotated[
        bool,
        typer.Option(
            "--direct-company-only/--all-company-matched",
            help="Keep only single-company headlines classified as directly actionable.",
        ),
    ] = False,
    max_population: Annotated[
        int | None,
        typer.Option(
            "--max-population",
            min=1,
            help="Refuse to score when the complete filtered population exceeds this safety ceiling.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Count the frozen scoring population without opening a model client."),
    ] = False,
    ollama_think: Annotated[
        bool | None,
        typer.Option(
            "--ollama-think/--no-ollama-think",
            help="Enable or disable Ollama thinking. Default: model/provider default.",
        ),
    ] = None,
    ollama_keep_alive: Annotated[
        str,
        typer.Option("--ollama-keep-alive", help="Ollama keep-alive value, for example -1 or 30m."),
    ] = os.getenv("OLLAMA_KEEP_ALIVE", "-1"),
    structured_output: Annotated[
        bool,
        typer.Option("--structured-output/--no-structured-output", help="Request provider-supported JSON schema output."),
    ] = True,
) -> None:
    """Score the collection's unique headlines with an LLM for analyze-headline-value --llm-scores."""
    resolved_provider = _resolve_provider(provider)
    concurrency = _resolve_concurrency(resolved_provider, concurrency)
    prompt = _resolve_prompt(prompt_id, prompts_path)
    safe_model = model.replace("/", "_").replace(":", "_")
    output_path = output if output is not None else collection_root / "derived" / f"headline_scores_{safe_model}.csv"
    source_codes = tuple(source_code or ())

    if dry_run:
        try:
            population = collect_scorable_headlines(
                collection_root,
                source_codes=source_codes,
                direct_company_only=direct_company_only,
            )
        except HeadlineValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        table = Table(title="Headline Scoring Dry Run")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Filtered unique headlines", f"{len(population):,}")
        table.add_row("Source codes", ", ".join(source_codes) if source_codes else "all")
        table.add_row("Direct-company only", str(direct_company_only))
        table.add_row("Safety ceiling", f"{max_population:,}" if max_population is not None else "none")
        console.print(table)
        if max_population is not None and len(population) > max_population:
            raise typer.BadParameter(
                f"filtered scoring population has {len(population):,} headlines, exceeding --max-population {max_population:,}"
            )
        return

    async def main() -> None:
        async with make_llm_client(
            resolved_provider,
            base_url=base_url,
            ollama_host=ollama_host,
            ollama_keep_alive=ollama_keep_alive,
            ollama_think=ollama_think,
            structured_label_output=structured_output,
        ) as client:
            summary = await score_headlines(
                client,
                collection_root=collection_root,
                model_id=model,
                prompt=prompt,
                output_path=output_path,
                concurrency=concurrency,
                retries=retries,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                limit=limit,
                source_codes=source_codes,
                direct_company_only=direct_company_only,
                max_population=max_population,
                ollama_think=ollama_think,
                structured_output=structured_output,
                callback=lambda message: console.print(f"[dim]{message}[/dim]"),
            )
        table = Table(title=f"Headline Scoring — {model}")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Unique scorable headlines", f"{summary.total_unique:,}")
        table.add_row("Already scored (skipped)", f"{summary.already_scored:,}")
        table.add_row("Attempted", f"{summary.attempted:,}")
        table.add_row("Succeeded", f"{summary.succeeded:,}")
        table.add_row("Failed", f"{summary.failed:,}")
        table.add_row("Scores CSV", str(summary.output_path))
        table.add_row("Manifest", str(summary.manifest_path))
        console.print(table)
        console.print(f"[dim]Backtest with: sentiment-bench analyze-headline-value --llm-scores {summary.output_path}[/dim]")

    try:
        asyncio.run(main())
    except HeadlineValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("analyze-headline-value")
def analyze_headline_value_command(
    collection_root: Annotated[
        Path,
        typer.Option("--collection-root", help="Collection folder or raw LSEG directory containing headlines.jsonl."),
    ] = Path("Data/collections/lseg_us_sector_33_6m"),
    prices: Annotated[
        Path | None,
        typer.Option("--prices", help="Optional daily prices CSV. If omitted/missing, trading value is reported as blocked."),
    ] = Path("Data/derived/prices/lseg_us_sector_33.csv"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Output directory for local headline-value artifacts."),
    ] = Path("Data/collections/lseg_us_sector_33_6m/derived/headline_value_analysis"),
    sample_size: Annotated[
        int,
        typer.Option("--sample-size", min=0, help="Rows to include in the local raw-headline calibration template."),
    ] = 2_000,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic sample seed.")] = 42,
    timezone: Annotated[str, typer.Option("--timezone", help="Exchange timezone used by the backtest core.")] = "America/New_York",
    horizons: Annotated[str, typer.Option("--horizons", help="Comma-separated holding horizons in trading sessions.")] = "1,5,10",
    transaction_cost_bps_per_side: Annotated[
        float,
        typer.Option("--transaction-cost-bps-per-side", min=0.0, help="Per-side trading cost applied to headline signals."),
    ] = 10.0,
    notional_usd: Annotated[float, typer.Option("--notional-usd", min=1.0, help="Per-signal notional for P&L summaries.")] = 10_000.0,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace files in an existing non-empty output directory.")] = False,
    llm_scores: Annotated[
        list[Path] | None,
        typer.Option(
            "--llm-scores",
            help="Model/baseline score CSV adding llm/<model> scorers next to lexicon ones. Repeat for multiple files.",
        ),
    ] = None,
    source_code: Annotated[
        list[str] | None,
        typer.Option("--source-code", help="Only analyze this exact source code; repeat for multiple sources."),
    ] = None,
    direct_company_only: Annotated[
        bool,
        typer.Option(
            "--direct-company-only/--all-company-matched",
            help="Apply the same single-company actionable filter used by score-headlines.",
        ),
    ] = False,
) -> None:
    """Analyze headline-only coverage, taxonomy, and trading value without requiring story bodies."""
    try:
        parsed_horizons = tuple(int(part.strip()) for part in horizons.split(",") if part.strip())
        if not parsed_horizons or any(value < 1 for value in parsed_horizons):
            raise ValueError
    except ValueError as exc:
        raise typer.BadParameter("--horizons must contain positive integers, e.g. 1,5,10") from exc
    try:
        result = analyze_headline_value(
            collection_root,
            prices=prices,
            output_dir=output_dir,
            sample_size=sample_size,
            seed=seed,
            timezone=timezone,
            horizons=parsed_horizons,
            transaction_cost_bps_per_side=transaction_cost_bps_per_side,
            notional_usd=notional_usd,
            overwrite=overwrite,
            llm_scores=tuple(llm_scores or ()),
            source_codes=tuple(source_code or ()),
            direct_company_only=direct_company_only,
        )
    except HeadlineValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="Headline Value Analysis Complete")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("Summary", str(result.summary_path))
    table.add_row("Manifest", str(result.manifest_path))
    table.add_row("Trading status", result.trading_status)
    table.add_row("Generated files", str(len(result.generated_files)))
    console.print(table)


@app.command("sweep-trading-strategy")
def sweep_trading_strategy_command(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Completed trading run directory (reads daily_signals.csv + prices.csv)."),
    ],
    scorer: Annotated[
        str,
        typer.Option("--scorer", help="scorer_id to tune, e.g. consensus/majority or openai/gpt-4o-mini."),
    ] = "consensus/majority",
    strategy: Annotated[
        str,
        typer.Option("--strategy", help="Registered strategy id to tune (see list-strategies)."),
    ] = "sentiment_threshold_v1",
    thresholds: Annotated[
        str,
        typer.Option("--thresholds", help="Comma-separated decision thresholds."),
    ] = "0.0,0.1,0.2,0.3",
    horizons: Annotated[
        str,
        typer.Option("--horizons", help="Comma-separated holding horizons (trading sessions)."),
    ] = "1,3,5",
    metric: Annotated[
        str,
        typer.Option("--metric", help="Selection metric: mean_return, hit_rate, or sharpe."),
    ] = "sharpe",
    train_fraction: Annotated[
        float,
        typer.Option("--train-fraction", min=0.05, max=0.95, help="Fraction of distinct dates used to tune."),
    ] = 0.6,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Where to write sweep.csv (default: <run-dir>/sweep.csv)."),
    ] = None,
    prices_path: Annotated[
        Path | None,
        typer.Option("--prices", help="Prices CSV override for run dirs without one (e.g. headline-value output)."),
    ] = None,
    max_news_date: Annotated[
        str | None,
        typer.Option("--max-news-date", help="Drop signals after this date (YYYY-MM-DD), e.g. when prices lack future sessions."),
    ] = None,
) -> None:
    """Tune a strategy's parameters on a completed run, selecting on a training
    split only and reporting held-out test metrics (no look-ahead in tuning). The
    strategy's own parameter space (e.g. ``scale`` for the magnitude idea) is swept;
    ``--thresholds`` overrides the decision-threshold axis for any strategy."""
    try:
        strategy_obj = get_strategy(strategy)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        signals = [s for s in load_signals_csv(run_dir / "daily_signals.csv") if s.scorer_id == scorer]
        prices = load_prices_csv(prices_path if prices_path is not None else run_dir / "prices.csv")
    except OSError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if max_news_date is not None:
        signals = [s for s in signals if s.news_date <= max_news_date]
    if not signals:
        raise typer.BadParameter(f"no daily signals for scorer {scorer!r} in {run_dir}")
    param_space = dict(strategy_obj.param_space())
    param_space["threshold"] = tuple(float(value) for value in thresholds.split(",") if value.strip())
    horizons_tuple = tuple(int(value) for value in horizons.split(",") if value.strip())
    try:
        split_date = chronological_split_date([s.news_date for s in signals], train_fraction)
        results = sweep_strategy(
            signals,
            prices,
            strategy_obj,
            horizons=horizons_tuple,
            split_date=split_date,
            notional_usd=10_000.0,
            metric=metric,
            param_space=param_space,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    destination = output or (run_dir / "sweep.csv")
    write_sweep_csv(results, destination)
    heatmap_path = plot_sweep_heatmap(
        results,
        destination.with_name(f"{destination.stem}_heatmap.png"),
        attr="test_metric",
        title=f"Sweep test {metric} — {scorer}",
    )

    table = Table(title=f"Parameter sweep — {strategy}, {metric}, scorer={scorer} (train split before {split_date})")
    for column in ("threshold", "horizon", "params", "n_train", "n_test", "train", "test", "selected"):
        table.add_column(column)
    for result in results:
        table.add_row(
            f"{result.threshold:g}",
            str(result.horizon),
            result.params or "—",
            str(result.n_train),
            str(result.n_test),
            "—" if result.train_metric is None else f"{result.train_metric:.4f}",
            "—" if result.test_metric is None else f"{result.test_metric:.4f}",
            "★" if result.selected else "",
        )
    console.print(table)
    console.print(f"[green]Wrote[/green] {destination}")
    if heatmap_path is not None:
        console.print(f"[green]Wrote[/green] {heatmap_path}")


@app.command("list-strategies")
def list_strategies_command() -> None:
    """List registered trading strategies (ideas) and their sweepable parameters."""
    table = Table(title="Registered strategies")
    for column in ("id", "axes", "eval frame", "param space", "description"):
        table.add_column(column)
    for strat in available_strategies():
        space = "; ".join(f"{name}={list(values)}" for name, values in strat.param_space().items()) or "—"
        table.add_row(strat.id, ", ".join(strat.axes), strat.evaluator.frame, space, strat.description)
    console.print(table)
    console.print(
        "Evaluation frames: 'event_study' (independent event diagnostics) and "
        "'cross_sectional' (funded daily portfolios). The table shows each strategy's default frame."
    )


@app.command("list-models")
def list_models(
    provider: Annotated[
        str,
        typer.Option("--provider", help="Model provider: openrouter, cerebras, or ollama."),
    ] = os.getenv("SENTIMENT_BENCH_PROVIDER", DEFAULT_PROVIDER),
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="OpenRouter-compatible base URL (Cerebras uses CEREBRAS_BASE_URL)."),
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


class _RunProgress:
    """Live per-model progress bars driven by the runner's structured events.

    Successful rows advance the bars silently; failed rows print their error
    detail above the display so quick experiments surface problems immediately.
    """

    def __init__(self, console: Console) -> None:
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[ok]} ok[/green] [red]{task.fields[failed]} failed[/red]"),
            TextColumn("[dim]ETA[/dim]"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        self._tasks: dict[str, Any] = {}
        self._counts: dict[str, dict[str, int]] = {}
        self.failed_rows = 0

    def __enter__(self) -> _RunProgress:
        self.progress.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.progress.stop()

    def handle_event(self, event: dict[str, Any]) -> None:
        if event["type"] == "model_started":
            model_id = event["model_id"]
            self._counts[model_id] = {"ok": 0, "failed": 0}
            self._tasks[model_id] = self.progress.add_task(f"[cyan]{model_id}[/cyan]", total=event["total_rows"], ok=0, failed=0)
        elif event["type"] == "row_completed":
            model_id = event["model_id"]
            counts = self._counts.get(model_id)
            if counts is None or model_id not in self._tasks:
                return
            if event["status"] in {"success", "skipped"}:
                counts["ok"] += 1
            else:
                counts["failed"] += 1
                self.failed_rows += 1
            self.progress.update(self._tasks[model_id], advance=1, ok=counts["ok"], failed=counts["failed"])
        elif event["type"] == "model_completed":
            model_id = event["model_id"]
            if model_id not in self._tasks:
                return
            accuracy = event.get("accuracy")
            suffix = f" [dim]accuracy={accuracy:.4f}[/dim]" if isinstance(accuracy, (int, float)) else ""
            style = "green" if event.get("status") == "completed" else "yellow"
            self.progress.update(self._tasks[model_id], description=f"[{style}]{model_id}[/{style}]{suffix}")

    def handle_message(self, message: str) -> None:
        # Per-row success lines are covered by the bars; keep lifecycle
        # messages and anything carrying an error detail. Text() avoids rich
        # markup parsing of brackets inside API error payloads.
        if message.startswith(("Saved response:", "Skipped existing response:")) and " error=" not in message:
            return
        style = "red" if " error=" in message or "failed" in message.lower() else "dim"
        self.progress.console.print(Text(message, style=style))


def _metrics_summary(store: BenchmarkStore, run_id: int) -> tuple[Table, list[tuple[str, str, dict]]] | None:
    """Build the per-model metrics table for a run (shared by run/results)."""
    metric_rows = store.fetch_metrics(run_id)
    if not metric_rows:
        return None
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
    return table, parsed


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
        typer.Option("--provider", help="Model provider: openrouter, cerebras, or ollama."),
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
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Simultaneous API calls. Default: 64 for Cerebras, otherwise 1."),
    ] = None,
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
    concurrency = _resolve_concurrency(resolved_provider, concurrency)
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
        started = time.monotonic()
        with _RunProgress(console) as display:
            async with make_llm_client(resolved_provider, base_url=base_url, ollama_host=ollama_host) as client:
                runner = BenchmarkRunner(client=client, store=store)
                summary = await runner.run(
                    config,
                    resume_run_id=resume_run_id,
                    callback=display.handle_message,
                    event_callback=display.handle_event,
                )
        elapsed = time.monotonic() - started
        attempted = summary.selected_row_count * summary.model_count
        rate = f" ({attempted / elapsed * 60:.0f} rows/min)" if elapsed > 0 and attempted else ""
        status_style = {"completed": "green", "cancelled": "yellow"}.get(summary.status, "red")
        console.print(
            f"\n[{status_style}]Run {summary.run_id} {summary.status}[/{status_style}]: "
            f"{summary.model_count} model(s) x {summary.selected_row_count} row(s) in {elapsed:.1f}s{rate}"
        )
        rendered = _metrics_summary(store, summary.run_id)
        if rendered is not None:
            console.print(rendered[0])
        if display.failed_rows:
            console.print(
                f"[yellow]{display.failed_rows} row(s) failed.[/yellow] Re-attempt just those with: "
                f"sentiment-bench run --resume-run-id {summary.run_id} "
                + " ".join(f"--models {model}" for model in config.models)
                + f" --provider {resolved_provider}"
            )
        console.print(f"[dim]Full details: sentiment-bench results --run-id {summary.run_id} --confusion[/dim]")

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
    rendered = _metrics_summary(store, run_id)
    if rendered is None:
        console.print(f"No metrics found for run {run_id}. It may still be running or have failed before scoring.")
        raise typer.Exit(code=1)
    table, parsed = rendered
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
        typer.Option("--provider", help="Model provider: openrouter, cerebras, or ollama."),
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
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Simultaneous API calls. Default: 64 for Cerebras, otherwise 1."),
    ] = None,
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
    concurrency = _resolve_concurrency(resolved_provider, concurrency)
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
        typer.Option("--provider", help="Model provider: openrouter, cerebras, or ollama."),
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
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Simultaneous API calls. Default: 64 for Cerebras, otherwise 1."),
    ] = None,
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
    concurrency = _resolve_concurrency(resolved_provider, concurrency)

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
