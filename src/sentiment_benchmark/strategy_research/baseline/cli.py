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
from .v2.config import BaselineV2ConfigurationError, load_baseline_v2_config
from .v2.pipeline import inspect_baseline_v2, run_baseline_v2
from .v3.config import FinbertV3ConfigurationError, load_finbert_v3_config
from .v3.pipeline import inspect_finbert_v3, run_finbert_v3

console = Console()
app = typer.Typer(help="Inspect or run the literature-grounded FinBERT/VADER-share-argmax trading baseline.")
v2_app = typer.Typer(help="Inspect or run the frozen v2 baseline (canonical VADER, two-name legs, event-level inference).")
v3_app = typer.Typer(help="Inspect or run the frozen exploratory FinBERT rank-reversal strategy.")
app.add_typer(v2_app, name="v2")
app.add_typer(v3_app, name="v3")


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


def _load_v2(path: Path):
    try:
        return load_baseline_v2_config(path)
    except BaselineV2ConfigurationError as exc:
        console.print(f"[red]Invalid baseline v2 configuration:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@v2_app.command("inspect")
def inspect_v2_command(
    config: Annotated[Path, typer.Option("--config", help="Frozen v2 baseline TOML configuration.")],
) -> None:
    """Validate manifests, joins, price coverage, and identities without writing."""

    try:
        report = inspect_baseline_v2(_load_v2(config), repo_root=Path.cwd())
    except (BaselineRunError, StrategyArtifactError, ValueError) as exc:
        console.print(f"[red]Baseline v2 inspection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table(title="Sentiment Trading Baseline v2 Inspection")
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
        ("Active sessions by arm", report.active_sessions_by_arm),
        ("Derived output", report.derived_dir),
        ("Results output", report.results_dir),
    ):
        table.add_row(name, str(value))
    console.print(table)
    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print("[green]All local inputs validated. No files or model clients were created.[/green]")


@v2_app.command("run")
def run_v2_command(
    config: Annotated[Path, typer.Option("--config", help="Frozen v2 baseline TOML configuration.")],
) -> None:
    """Run the local, no-retuning v2 baseline and write immutable aggregate evidence."""

    try:
        result = run_baseline_v2(_load_v2(config), repo_root=Path.cwd())
    except (BaselineRunError, StrategyArtifactError, ValueError) as exc:
        console.print(f"[red]Baseline v2 run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed"
    console.print(f"[green]Sentiment trading baseline v2 {status}:[/green] {result.resolved_run_id}")
    console.print(f"Derived: {result.derived_dir}")
    console.print(f"Results: {result.results_dir}")
    console.print(f"Report: {result.report_path}")


def _load_v3(path: Path):
    try:
        return load_finbert_v3_config(path)
    except FinbertV3ConfigurationError as exc:
        console.print(f"[red]Invalid FinBERT v3 configuration:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@v3_app.command("inspect")
def inspect_v3_command(
    config: Annotated[Path, typer.Option("--config", help="Frozen FinBERT v3 TOML configuration.")],
) -> None:
    """Validate v3 inputs, provenance, breadth and identities without writing."""

    try:
        report = inspect_finbert_v3(_load_v3(config), repo_root=Path.cwd())
    except (BaselineRunError, StrategyArtifactError, ValueError) as exc:
        console.print(f"[red]FinBERT v3 inspection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table(title="FinBERT Rank-Reversal v3 Inspection")
    table.add_column("Item")
    table.add_column("Validated value")
    for name, value in (
        ("Resolved run", report.resolved_run_id),
        ("Point-in-time events", report.event_count),
        ("Selected events", report.selected_event_count),
        ("Firm-session signals", report.firm_session_signal_count),
        ("Symbols", len(report.symbols)),
        ("Decision sessions", report.decision_sessions),
        ("Evaluation sessions", report.evaluation_sessions),
        ("Median ranked firms", report.median_ranked_firms),
        ("Active sessions by arm", report.active_sessions_by_arm),
        ("Derived output", report.derived_dir),
        ("Results output", report.results_dir),
    ):
        table.add_row(name, str(value))
    console.print(table)
    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print("[green]All v3 inputs validated; no output files were written.[/green]")


@v3_app.command("run")
def run_v3_command(
    config: Annotated[Path, typer.Option("--config", help="Frozen FinBERT v3 TOML configuration.")],
) -> None:
    """Run the immutable post-selection diagnostic v3 strategy."""

    try:
        result = run_finbert_v3(_load_v3(config), repo_root=Path.cwd())
    except (BaselineRunError, StrategyArtifactError, ValueError) as exc:
        console.print(f"[red]FinBERT v3 run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed"
    console.print(f"[green]FinBERT rank-reversal v3 {status}:[/green] {result.resolved_run_id}")
    console.print(f"Derived: {result.derived_dir}")
    console.print(f"Results: {result.results_dir}")
    console.print(f"Report: {result.report_path}")
