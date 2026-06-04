from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .baseline_runner import run_baselines
from .baselines import BASELINE_SPECS
from .constants import (
    DEFAULT_BASE_URL,
    DEFAULT_CONCURRENCY,
    DEFAULT_DATASET_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_PILOT_PER_CLASS,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_REASONING_MAX_COMPLETION_TOKENS,
    DEFAULT_RETRIES,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
)
from .dataset import compute_stats, load_dataset
from .env import load_env_file
from .exporter import export_run
from .models import RunConfig
from .openrouter import OpenRouterClient
from .prompts import load_prompts
from .runner import BenchmarkRunner
from .storage import BenchmarkStore

console = Console()
app = typer.Typer(help="Benchmark OpenRouter LLMs on dissertation sentiment data.")
load_env_file()


def _resolve_prompt(prompt_id: str, prompts_path: Path):
    prompts = load_prompts(prompts_path)
    if prompt_id not in prompts:
        available = ", ".join(sorted(prompts))
        raise typer.BadParameter(f"Unknown prompt id {prompt_id!r}. Available: {available}")
    return prompts[prompt_id]


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


@app.command("list-models")
def list_models(
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="OpenRouter-compatible base URL."),
    ] = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    limit: Annotated[int, typer.Option("--limit", help="Maximum rows to show.")] = 50,
) -> None:
    async def main() -> None:
        async with OpenRouterClient(base_url=base_url) as client:
            models = await client.list_models()
        table = Table(title="OpenRouter Models")
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
    models: Annotated[list[str], typer.Option("--models", "-m", help="OpenRouter model id. Repeat for multiple models.")],
    mode: Annotated[str, typer.Option("--mode", help="pilot or full.")] = "pilot",
    prompt_id: Annotated[str, typer.Option("--prompt-id", help="Prompt id from configs/default_prompts.toml.")] = "default_label_only",
    dataset_path: Annotated[Path, typer.Option("--dataset-path")] = DEFAULT_DATASET_PATH,
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    prompts_path: Annotated[Path, typer.Option("--prompts-path")] = DEFAULT_PROMPTS_PATH,
    base_url: Annotated[str, typer.Option("--base-url")] = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
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
    resume_run_id: Annotated[
        int | None,
        typer.Option("--resume-run-id", help="Resume an existing run without duplicating completed responses."),
    ] = None,
) -> None:
    if mode not in {"pilot", "full"}:
        raise typer.BadParameter("mode must be pilot or full")
    if not models:
        raise typer.BadParameter("At least one --models value is required")
    model_token_overrides = _parse_model_max_tokens(model_max_tokens)
    prompt = _resolve_prompt(prompt_id, prompts_path)
    config = RunConfig(
        models=models,
        prompt=prompt,
        mode=mode,  # type: ignore[arg-type]
        dataset_path=str(dataset_path),
        db_path=str(db_path),
        base_url=base_url,
        sample_per_class=sample_per_class,
        seed=seed,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        reasoning_max_completion_tokens=reasoning_max_tokens,
        model_max_completion_tokens=model_token_overrides,
        concurrency=concurrency,
        retries=retries,
    )

    async def main() -> None:
        store = BenchmarkStore(db_path)
        async with OpenRouterClient(base_url=base_url) as client:
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
    console.print(
        f"Baseline run {summary.run_id} complete: {summary.baseline_count} baseline(s), {summary.selected_row_count} row(s)"
    )


@app.command("export")
def export(
    run_id: Annotated[int, typer.Option("--run-id", help="Run id to export.")],
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    paths = export_run(db_path, run_id, output_dir=output_dir)
    for path in paths:
        console.print(path)


@app.command("tui")
def tui() -> None:
    from .tui import SentimentBenchmarkApp

    SentimentBenchmarkApp().run()


if __name__ == "__main__":
    app()
