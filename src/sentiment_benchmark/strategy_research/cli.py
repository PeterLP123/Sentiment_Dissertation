"""Typer commands for the isolated strategy-research pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from .artifacts import StrategyArtifactError
from .config import StrategyConfigurationError, load_strategy_config
from .development_controls import DevelopmentControlError, run_development_controls
from .development_tests import DevelopmentTestError, run_development_tests
from .pipeline import DryRunReport, StrategyPipelineError, execute_pipeline, inspect_pipeline
from .signal_quality import (
    SignalAuditSpec,
    SignalQualityError,
    prepare_signal_audit,
    score_signal_audit,
    validate_signal_audit,
)
from .single_stock import SingleStockExperimentError, run_single_stock_experiment

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
    score_cache_from: Path | None = None,
) -> None:
    config = _config(path)
    try:
        result = execute_pipeline(
            config,
            through=through,  # type: ignore[arg-type]
            allow_paid=allow_paid,
            max_new_scores=max_new_scores,
            score_cache_from=score_cache_from,
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
    cache_from: Annotated[
        Path | None,
        typer.Option(
            "--cache-from",
            help="Import a complete, identity-verified score cache from a prior pipeline run.",
        ),
    ] = None,
) -> None:
    """Explicitly score missing events and materialize frozen scores."""

    _execute(
        config,
        "scores",
        allow_paid=True,
        max_new_scores=max_new_scores,
        score_cache_from=cache_from,
    )


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
def paper_command() -> None:
    """Report the prospective-paper acceptance gate without writing."""

    console.print(
        "[yellow]Prospective paper mode is gated.[/yellow] Complete and accept the formal historical v1 run first; "
        "this command performs no journal writes."
    )
    raise typer.Exit(code=1)


@app.command("stock-test")
def stock_test_command(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Completed strategy-research results directory."),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Optional immutable experiment output directory."),
    ] = None,
    price_panel: Annotated[
        Path | None,
        typer.Option("--price-panel", help="Override the price panel recorded in the completed run."),
    ] = None,
) -> None:
    """Test one frozen strategy separately for every stock, without a portfolio."""

    try:
        result = run_single_stock_experiment(
            run_dir,
            output_root=output_root,
            price_panel=price_panel,
        )
    except (SingleStockExperimentError, ValueError) as exc:
        console.print(f"[red]Independent stock test failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed"
    console.print(f"[green]Independent stock test {status}:[/green] {result.experiment_id}")
    console.print(f"Results: {result.output_dir}")
    console.print(f"Report: {result.report_path}")


@app.command("development-tests")
def development_tests_command(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Completed strategy-research results directory."),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Optional immutable development-test output directory."),
    ] = None,
) -> None:
    """Run label, horizon, centering, threshold, and side tests on development only."""

    try:
        result = run_development_tests(run_dir, output_root=output_root)
    except (DevelopmentTestError, ValueError) as exc:
        console.print(f"[red]Development tests failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed"
    console.print(f"[green]Development tests {status}:[/green] {result.experiment_id}")
    console.print(f"Results: {result.output_dir}")
    console.print(f"Report: {result.report_path}")
    if result.selected_candidate is not None:
        selected = result.selected_candidate
        console.print(
            "Best valid development candidate: "
            f"{selected.signal_kind}, hold={selected.hold_sessions}, threshold={selected.threshold:g}, "
            f"objective={selected.objective:.6f}"
        )


@app.command("development-controls")
def development_controls_command(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Completed strategy-research results directory."),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Optional immutable control-experiment output directory."),
    ] = None,
) -> None:
    """Run non-refreshing v2 against timing, refresh, inversion, and shuffle controls."""

    try:
        result = run_development_controls(run_dir, output_root=output_root)
    except (DevelopmentControlError, ValueError) as exc:
        console.print(f"[red]Development controls failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed"
    passed = sum(check.passed for check in result.acceptance)
    console.print(f"[green]Development controls {status}:[/green] {result.experiment_id}")
    console.print(f"Results: {result.output_dir}")
    console.print(f"Report: {result.report_path}")
    console.print(f"Sentiment v2 objective={result.v2_summary.objective:.6f}; acceptance={passed}/{len(result.acceptance)}")


@app.command("prepare-signal-audit")
def prepare_signal_audit_command(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Completed strategy-research results directory."),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Optional immutable audit-sample output directory."),
    ] = None,
    sample_per_symbol: Annotated[
        int,
        typer.Option("--sample-per-symbol", min=1, help="Blind audit allocation per stock."),
    ] = 3,
) -> None:
    """Freeze a development-only human audit packet without model calls."""

    try:
        result = prepare_signal_audit(
            run_dir,
            output_root=output_root,
            spec=SignalAuditSpec(sample_per_symbol=sample_per_symbol),
        )
    except (SignalQualityError, ValueError) as exc:
        console.print(f"[red]Signal-quality audit preparation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed"
    console.print(f"[green]Signal-quality audit sample {status}:[/green] {result.experiment_id}")
    console.print(f"Items: {result.item_count}")
    console.print(f"Audit packet: {result.output_dir}")
    console.print(f"Human template: {result.human_template_path}")


@app.command("score-signal-audit")
def score_signal_audit_command(
    audit_dir: Annotated[
        Path,
        typer.Option("--audit-dir", help="Completed signal-quality audit-sample directory."),
    ],
    endpoint: Annotated[
        str,
        typer.Option("--endpoint", help="Loopback Ollama endpoint, normally an SSH local forward."),
    ] = "http://127.0.0.1:11435",
    max_new_calls: Annotated[
        int | None,
        typer.Option("--max-new-calls", min=1, help="Bound new local calls while preserving resumable cache entries."),
    ] = None,
) -> None:
    """Explicitly authorize resumable local-only scoring of the audit sample."""

    try:
        result = asyncio.run(
            score_signal_audit(
                audit_dir,
                endpoint=endpoint,
                max_new_calls=max_new_calls,
                allow_calls=True,
            )
        )
    except (SignalQualityError, ValueError) as exc:
        console.print(f"[red]Signal-quality audit scoring failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed" if result.complete else "paused safely"
    console.print(f"[green]Signal-quality audit scoring {status}:[/green] {result.score_id}")
    console.print(
        f"Successful joint event scores: {result.successful_scores}/{result.expected_calls}; "
        f"new calls: {result.calls_made}"
    )
    console.print(f"Scoring artifacts: {result.output_dir}")


@app.command("validate-signal-audit")
def validate_signal_audit_command(
    score_dir: Annotated[
        Path,
        typer.Option("--score-dir", help="Completed signal-quality scoring directory."),
    ],
    human_labels: Annotated[
        Path,
        typer.Option("--human-labels", help="Completed blind human_labels.csv copied from the frozen template."),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Optional immutable validation-results directory."),
    ] = None,
) -> None:
    """Apply the frozen human-agreement gate before any full rescoring."""

    try:
        result = validate_signal_audit(score_dir, human_labels, output_root=output_root)
    except (SignalQualityError, ValueError) as exc:
        console.print(f"[red]Signal-quality human validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "validated/reused" if result.reused else "completed"
    passed = sum(check.passed for check in result.acceptance)
    console.print(f"[green]Signal-quality human validation {status}:[/green] {result.validation_id}")
    console.print(f"Acceptance: {passed}/{len(result.acceptance)}")
    console.print(f"Report: {result.report_path}")


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
