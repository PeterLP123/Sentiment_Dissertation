"""Typer commands for the isolated strategy-research pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from .artifacts import StrategyArtifactError
from .config import StrategyConfigurationError, load_strategy_config
from .pipeline import DryRunReport, StrategyPipelineError, execute_pipeline, inspect_pipeline

console = Console()
app = typer.Typer(help="Run the separate point-in-time historical strategy-research pipeline.")


def _config(path: Path):
    try:
        return load_strategy_config(path)
    except StrategyConfigurationError as exc:
        console.print(f"[red]Invalid strategy configuration:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _execute(
    path: Path,
    through: str,
    *,
    allow_paid: bool = False,
    max_new_scores: int | None = None,
) -> None:
    config = _config(path)
    try:
        result = execute_pipeline(
            config,
            through=through,  # type: ignore[arg-type]
            allow_paid=allow_paid,
            max_new_scores=max_new_scores,
            repo_root=Path.cwd(),
            command=sys.argv,
        )
    except (StrategyConfigurationError, StrategyPipelineError, StrategyArtifactError, ValueError) as exc:
        console.print(f"[red]Strategy {through} failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    progress = result.score_progress
    if progress is not None and not progress.complete:
        console.print(f"[green]Local scoring canary paused safely:[/green] {result.paths.identity.resolved_run_id}")
        console.print(
            f"New calls: {progress.calls_made}; cached successes: {progress.successful_scores}; "
            f"remaining full-universe scores: {progress.missing_scores}"
        )
        console.print(f"Development-period canary population: {progress.canary_eligible_events}")
        console.print("The score stage remains in_progress; no scores.jsonl was finalized.")
    else:
        console.print(f"[green]Strategy {through} complete:[/green] {result.paths.identity.resolved_run_id}")
    console.print(f"Derived: {result.paths.derived_dir}")
    console.print(f"Results: {result.paths.results_dir}")
    if result.reused_stages:
        console.print(f"Validated/reused stages: {', '.join(result.reused_stages)}")


@app.command("build-events")
def build_events_command(
    config: Annotated[Path, typer.Option("--config", help="Strategy TOML configuration.")],
) -> None:
    """Build the full unsampled canonical event universe."""

    _execute(config, "events")


@app.command("score")
def score_command(
    config: Annotated[Path, typer.Option("--config", help="Strategy TOML configuration.")],
    max_new_scores: Annotated[
        int | None,
        typer.Option(
            "--max-new-scores",
            min=1,
            help="Bound new local Ollama calls for a resumable development-only canary.",
        ),
    ] = None,
) -> None:
    """Explicitly score missing events and materialize frozen scores."""

    _execute(config, "scores", allow_paid=True, max_new_scores=max_new_scores)


@app.command("tune")
def tune_command(
    config: Annotated[Path, typer.Option("--config", help="Strategy TOML configuration.")],
) -> None:
    """Select the reset rule using chronological development folds only."""

    _execute(config, "tuning")


@app.command("build-state")
def build_state_command(
    config: Annotated[Path, typer.Option("--config", help="Strategy TOML configuration.")],
) -> None:
    """Build the selected auditable state and action tables."""

    _execute(config, "state")


@app.command("backtest")
def backtest_command(
    config: Annotated[Path, typer.Option("--config", help="Strategy TOML configuration.")],
) -> None:
    """Run the selected rule and fixed baselines through one portfolio ledger."""

    _execute(config, "backtest")


@app.command("report")
def report_command(
    config: Annotated[Path, typer.Option("--config", help="Strategy TOML configuration.")],
) -> None:
    """Write the technical summary, diagnostics, figures, and master manifest."""

    _execute(config, "report")


@app.command("run")
def run_command(
    config: Annotated[Path, typer.Option("--config", help="Strategy TOML configuration.")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Inspect identities, coverage, calls, outputs, and blockers without writing."),
    ] = False,
    allow_paid: Annotated[
        bool,
        typer.Option("--allow-paid", help="Explicitly authorize missing hosted-model calls."),
    ] = False,
) -> None:
    """Inspect or execute the complete historical pipeline."""

    loaded = _config(config)
    if dry_run:
        try:
            report = inspect_pipeline(loaded)
        except (StrategyConfigurationError, StrategyPipelineError, StrategyArtifactError, ValueError) as exc:
            console.print(f"[red]Strategy dry-run failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        _print_dry_run(report)
        return
    _execute(config, "report", allow_paid=allow_paid)


@app.command("paper")
def paper_command(
) -> None:
    """Report the prospective-paper acceptance gate without writing."""

    console.print(
        "[yellow]Prospective paper mode is gated.[/yellow] Complete and accept the formal historical v1 run first; "
        "this command performs no journal writes."
    )
    raise typer.Exit(code=1)


def _print_dry_run(report: DryRunReport) -> None:
    table = Table(title="Historical Strategy Research Dry Run")
    table.add_column("Item")
    table.add_column("Resolved value")
    rows: list[tuple[str, Any]] = [
        ("Run identity", report.resolved_run_id),
        ("Source", report.source),
        ("Event count", report.event_count),
        ("Event date range", " to ".join(report.event_date_range) if report.event_date_range else None),
        ("Unique symbols", report.unique_symbols),
        ("Expected hosted calls", report.expected_score_calls),
        ("Cached/fixture scores", report.cached_scores),
        ("Missing scores", report.missing_scores),
        ("Score success rate", report.score_success_rate),
        ("Price rows", report.price_rows),
        ("Price symbols", report.price_symbols),
        ("Price date range", " to ".join(report.price_date_range) if report.price_date_range else None),
        ("Development sessions", report.development_sessions),
        ("Evaluation sessions", report.evaluation_sessions),
        ("Feasible tuning folds", report.feasible_folds),
        ("Scored executable events", report.executable_events),
        ("Supported event symbols", report.supported_event_symbols),
        ("Tuning candidates", report.tuning_candidates),
        ("Derived output", report.derived_dir),
        ("Results output", report.results_dir),
    ]
    for name, value in rows:
        table.add_row(name, "not available" if value is None else str(value))
    console.print(table)
    if report.blockers:
        console.print("[yellow]Blockers:[/yellow]")
        for blocker in report.blockers:
            console.print(f"- {blocker}")
    else:
        console.print("[green]No blockers detected. No files or provider clients were created.[/green]")
    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
