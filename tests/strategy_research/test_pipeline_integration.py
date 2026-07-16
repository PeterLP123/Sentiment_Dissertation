from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.cli import app as root_app
from sentiment_benchmark.strategy_research import pipeline
from sentiment_benchmark.strategy_research.config import OutputSettings, load_strategy_config
from sentiment_benchmark.strategy_research.pipeline import (
    StrategyPipelineError,
    execute_pipeline,
    inspect_pipeline,
    load_verified_price_panel,
)

SMOKE_CONFIG = Path("configs/strategy_research/smoke.toml")


def isolated_smoke_config(tmp_path: Path):
    config = load_strategy_config(SMOKE_CONFIG)
    return replace(
        config,
        outputs=OutputSettings(
            derived_root=tmp_path / "derived",
            results_root=tmp_path / "results",
        ),
    )


def stable_output_hashes(*roots: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name.endswith(".lock") or "manifests" in path.parts:
                continue
            hashes[path.relative_to(root.parent).as_posix()] = sha256_file(path)
    return hashes


def test_dry_run_is_pure_and_resolves_exact_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = isolated_smoke_config(tmp_path)

    def provider_call_is_a_test_failure(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("dry-run created a provider client")

    monkeypatch.setattr(pipeline, "make_llm_client", provider_call_is_a_test_failure)
    report = inspect_pipeline(config)

    assert report.blockers == ()
    assert report.event_count == 12
    assert report.cached_scores == 11
    assert report.missing_scores == 1
    assert report.development_sessions == 19
    assert report.evaluation_sessions == 13
    assert report.feasible_folds == 2
    assert report.executable_events == 10
    assert not config.outputs.derived_root.exists()
    assert not config.outputs.results_root.exists()


def test_dry_run_reports_corrupt_score_identity_without_writing(tmp_path: Path) -> None:
    config = isolated_smoke_config(tmp_path)
    score_rows = [
        json.loads(line)
        for line in config.scoring.scores_path.read_text(encoding="utf-8").splitlines()  # type: ignore[union-attr]
    ]
    score_rows[0]["temperature"] = 0.25
    corrupt_scores = tmp_path / "corrupt-scores.jsonl"
    corrupt_scores.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in score_rows),
        encoding="utf-8",
    )
    config = replace(config, scoring=replace(config.scoring, scores_path=corrupt_scores))

    report = inspect_pipeline(config)

    assert any("score identity" in blocker for blocker in report.blockers)
    assert not config.outputs.derived_root.exists()
    assert not config.outputs.results_root.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adjustment_supported", False),
        ("split_adjusted", False),
        ("dividend_adjusted", True),
    ],
)
def test_price_manifest_adjustment_contract_fails_closed(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    config = isolated_smoke_config(tmp_path)
    manifest = json.loads(config.prices.manifest_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    manifest[field] = value
    manifest_path = tmp_path / "prices.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = replace(config, prices=replace(config.prices, manifest_path=manifest_path))

    with pytest.raises(StrategyPipelineError, match=field):
        load_verified_price_panel(config)


def test_price_panel_requires_complete_exchange_session_spine(tmp_path: Path) -> None:
    config = isolated_smoke_config(tmp_path)
    panel_lines = config.prices.panel_path.read_text(encoding="utf-8").splitlines()  # type: ignore[union-attr]
    panel_path = tmp_path / "prices.csv"
    panel_path.write_text(
        "\n".join(line for line in panel_lines if ",2025-12-08," not in line) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(config.prices.manifest_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    manifest["files"]["price_panel"] = {"path": panel_path.name, "sha256": sha256_file(panel_path)}
    manifest_path = tmp_path / "prices.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = replace(
        config,
        prices=replace(config.prices, panel_path=panel_path, manifest_path=manifest_path),
    )

    with pytest.raises(StrategyPipelineError, match="complete XNYS spine"):
        load_verified_price_panel(config)


def test_price_panel_rejects_a_single_symbol_session_gap(tmp_path: Path) -> None:
    config = isolated_smoke_config(tmp_path)
    panel_lines = config.prices.panel_path.read_text(encoding="utf-8").splitlines()  # type: ignore[union-attr]
    panel_path = tmp_path / "prices.csv"
    panel_path.write_text(
        "\n".join(
            line
            for line in panel_lines
            if not (line.startswith("AAA,") and ",2025-12-23," in line)
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(config.prices.manifest_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    manifest["files"]["price_panel"] = {"path": panel_path.name, "sha256": sha256_file(panel_path)}
    manifest_path = tmp_path / "prices.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = replace(
        config,
        prices=replace(config.prices, panel_path=panel_path, manifest_path=manifest_path),
    )

    with pytest.raises(StrategyPipelineError, match="complete symbol-by-XNYS-session panel"):
        load_verified_price_panel(config)


def test_dry_run_rejects_non_session_evaluation_boundary(tmp_path: Path) -> None:
    config = isolated_smoke_config(tmp_path)
    config = replace(config, run=replace(config.run, evaluation_start=date(2026, 1, 11)))

    report = inspect_pipeline(config)

    assert any("evaluation_start must be an executable XNYS" in blocker for blocker in report.blockers)
    assert not config.outputs.derived_root.exists()


def test_formal_dry_run_reports_invalid_canonical_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_strategy_config(Path("configs/strategy_research/lseg_llm_decay_v1.toml"))
    corpus_manifest = tmp_path / "corpus-manifest.json"
    corpus_manifest.write_text("{}\n", encoding="utf-8")
    config = replace(
        config,
        data=replace(config.data, corpus_manifest=corpus_manifest),
        outputs=OutputSettings(tmp_path / "derived", tmp_path / "results"),
    )
    monkeypatch.setattr(
        pipeline,
        "build_strategy_events",
        lambda *args, **kwargs: SimpleNamespace(events=(), attrition={"invalid_timestamp": 2}),
    )

    report = inspect_pipeline(config)

    assert any("invalid timestamps" in blocker and "2 rows" in blocker for blocker in report.blockers)
    assert not config.outputs.derived_root.exists()
    assert not config.outputs.results_root.exists()


def test_formal_dry_run_mirrors_predecision_warmup_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = isolated_smoke_config(tmp_path)
    panel = load_verified_price_panel(smoke)
    config = replace(
        smoke,
        run=replace(smoke.run, mode="formal"),
        prices=replace(smoke.prices, volatility_minimum_sessions=30),
    )
    monkeypatch.setattr(pipeline, "load_verified_price_panel", lambda _config: panel)

    report = inspect_pipeline(config)

    assert any("pre-decision volatility warm-up" in blocker for blocker in report.blockers)
    assert not config.outputs.derived_root.exists()
    assert not config.outputs.results_root.exists()


def test_two_clean_offline_runs_have_identical_stable_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = isolated_smoke_config(tmp_path)

    def provider_call_is_a_test_failure(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("fixture smoke run created a provider client")

    monkeypatch.setattr(pipeline, "make_llm_client", provider_call_is_a_test_failure)
    first = execute_pipeline(config, repo_root=Path.cwd(), command=("test", "strategy", "run"))
    first_hashes = stable_output_hashes(config.outputs.derived_root, config.outputs.results_root)
    first_manifest = json.loads((first.paths.results_dir / "manifest.json").read_text(encoding="utf-8"))
    first_metrics = json.loads(first.evaluation_metrics_path.read_text(encoding="utf-8"))

    assert first.summary_path is not None
    assert first_metrics["metrics"]["decay_with_reset"]["observations"] == 14
    assert {metrics["observations"] for metrics in first_metrics["metrics"].values()} == {14}
    assert first_manifest["event_build_counts"]["selected_events"] == 12
    assert first_manifest["downstream_attrition"] == {"missing_price_history": 1, "missing_score": 1}
    leakage = json.loads(
        (first.paths.results_dir / "report" / "leakage_checklist.json").read_text(encoding="utf-8")
    )
    model_check = next(
        check for check in leakage["checks"] if check["check"] == "model_selection_from_portfolio_pnl"
    )
    assert model_check["classification"] == "mechanically_checked"
    assert "synthetic smoke fixture" in model_check["evidence"]
    pnl_rows = [
        json.loads(line)
        for line in (first.paths.results_dir / "backtest" / "daily_pnl.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    evaluation_open_rows = [row for row in pnl_rows if row["session"] == "2026-01-12"]
    assert len(evaluation_open_rows) == 4
    assert {row["start_nav_usd"] for row in evaluation_open_rows} == {1_000_000.0}

    shutil.rmtree(config.outputs.derived_root)
    shutil.rmtree(config.outputs.results_root)

    second = execute_pipeline(config, repo_root=Path.cwd(), command=("test", "strategy", "run"))
    second_hashes = stable_output_hashes(config.outputs.derived_root, config.outputs.results_root)

    assert second.paths.identity == first.paths.identity
    assert second_hashes == first_hashes


def test_reused_smoke_run_validates_every_completed_stage(tmp_path: Path) -> None:
    config = isolated_smoke_config(tmp_path)
    first = execute_pipeline(config, repo_root=Path.cwd(), command=("test", "strategy", "run"))
    second = execute_pipeline(config, repo_root=Path.cwd(), command=("test", "strategy", "run"))
    assert second.paths.identity == first.paths.identity
    assert second.reused_stages == ("events", "scores", "tuning", "state", "backtest", "report")


def test_completed_output_mismatch_is_refused(tmp_path: Path) -> None:
    config = isolated_smoke_config(tmp_path)
    result = execute_pipeline(config, repo_root=Path.cwd(), command=("test", "strategy", "run"))
    result.scores_path.write_text(result.scores_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="output hash mismatch"):
        execute_pipeline(config, repo_root=Path.cwd(), command=("test", "strategy", "run"))


def test_ordinary_pipeline_run_never_crosses_paid_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = isolated_smoke_config(tmp_path)
    scoring = replace(
        config.scoring,
        provider="cerebras",
        model="gemma-4-31b",
        scores_path=None,
    )
    config = replace(config, scoring=scoring)

    def provider_call_is_a_test_failure(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("ordinary run created a provider client without authorization")

    monkeypatch.setattr(pipeline, "make_llm_client", provider_call_is_a_test_failure)
    with pytest.raises(StrategyPipelineError, match="score calls are missing"):
        execute_pipeline(
            config,
            through="scores",
            allow_paid=False,
            repo_root=Path.cwd(),
            command=("test", "strategy", "run"),
        )


def test_strategy_cli_is_registered_and_paper_phase_is_gated() -> None:
    runner = CliRunner()
    result = runner.invoke(root_app, ["strategy", "run", "--config", str(SMOKE_CONFIG), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Development sessions" in result.output
    assert "19" in result.output

    paper = runner.invoke(root_app, ["strategy", "paper"])
    assert paper.exit_code == 1
    assert "Prospective paper mode is gated" in paper.output
