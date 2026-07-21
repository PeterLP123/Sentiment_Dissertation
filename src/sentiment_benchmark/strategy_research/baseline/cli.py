"""CLI for the frozen local sentiment trading baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ..artifacts import StrategyArtifactError
from .config import BaselineConfigurationError, load_baseline_config
from .pipeline import BaselineRunError, inspect_baseline, run_baseline

console = Console()
app = typer.Typer(help="Inspect or run the literature-grounded FinBERT/VADER-share-argmax trading baseline.")


def _load(path: Path):
    try:
        return load_baseline_config(path)
    except BaselineConfigurationError as exc:
        console.print(f"[red]Invalid baseline configuration:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("inspect")
def inspect_command(
    config: Annotated[Path, typer.Option("--config", help="Frozen baseline TOML configuration.")],
) -> None:
    """Validate manifests, joins, price coverage, and identities without writing."""

    try:
        report = inspect_baseline(_load(config), repo_root=Path.cwd())
    except (BaselineRunError, StrategyArtifactError, ValueError) as exc:
        console.print(f"[red]Baseline inspection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table(title="Sentiment Trading Baseline Inspection")
    table.add_column("Item")
    table.add_column("Validated value")
    for name, value in (
        ("Resolved run", report.resolved_run_id),
        ("Point-in-time events", report.event_count),
        ("Selected events", report.selected_event_count),
        ("Firm-session signals", report.firm_session_signal_count),
        ("Symbols", len(report.symbols)),
        ("Price sessions", report.price_sessions),
        ("Decision sessions", report.decision_sessions),
        ("Development sessions", report.development_sessions),
        ("Evaluation sessions", report.evaluation_sessions),
        ("Active sessions", report.active_sessions_by_scorer),
        ("Derived output", report.derived_dir),
        ("Results output", report.results_dir),
    ):
        table.add_row(name, str(value))
    console.print(table)
    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print("[green]All local inputs validated. No files or model clients were created.[/green]")


@app.command("run")
def run_command(
    config: Annotated[Path, typer.Option("--config", help="Frozen baseline TOML configuration.")],
) -> None:
    """Run the local, no-retuning baseline and write immutable aggregate evidence."""

    try:
        result = run_baseline(_load(config), repo_root=Path.cwd())
    except (BaselineRunError, StrategyArtifactError, ValueError) as exc:
        console.print(f"[red]Baseline run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed"
    console.print(f"[green]Sentiment trading baseline {status}:[/green] {result.resolved_run_id}")
    console.print(f"Derived: {result.derived_dir}")
    console.print(f"Results: {result.results_dir}")
    console.print(f"Report: {result.report_path}")
