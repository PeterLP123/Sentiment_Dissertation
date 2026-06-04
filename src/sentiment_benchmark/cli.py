from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .baseline_runner import run_baselines
from .baselines import BASELINE_SPECS
from .comparison import ModelTarget, compare_models
from .constants import (
    ALLOWED_LABELS,
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

_CONFUSION_PRED_LABELS = (*ALLOWED_LABELS, "__invalid__", "__error__")

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


@app.command("runs")
def list_runs_command(
    db_path: Annotated[Path, typer.Option("--db-path")] = DEFAULT_DB_PATH,
) -> None:
    """List stored benchmark runs (most recent first)."""
    runs = BenchmarkStore(db_path).list_runs()
    if not runs:
        console.print(f"No runs found in {db_path}.")
        return
    table = Table(title=f"Benchmark Runs: {db_path}")
    table.add_column("ID", justify="right")
    table.add_column("Created")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Models")
    for run in runs:
        models = json.loads(run["models_json"]) if run["models_json"] else []
        preview = ", ".join(models[:3]) + (f" (+{len(models) - 3})" if len(models) > 3 else "")
        table.add_row(str(run["id"]), (run["created_at"] or "")[:19], run["mode"], run["status"], preview)
    console.print(table)


def _confusion_table(model_id: str, scope: str, matrix: dict[str, dict[str, int]]) -> Table:
    table = Table(title=f"Confusion matrix — {model_id} ({scope})")
    table.add_column("actual \\ pred")
    for predicted in _CONFUSION_PRED_LABELS:
        table.add_column(predicted.strip("_"), justify="right")
    for actual in ALLOWED_LABELS:
        counts = matrix.get(actual, {})
        table.add_row(actual, *[str(int(counts.get(predicted, 0) or 0)) for predicted in _CONFUSION_PRED_LABELS])
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
    for column in ("Model", "Scope", "Rows", "Accuracy", "Macro F1", "Latency", "Tokens", "Cost", "Invalid", "Errors"):
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


@app.command("tui")
def tui() -> None:
    from .tui import SentimentBenchmarkApp

    SentimentBenchmarkApp().run()


if __name__ == "__main__":
    app()
