from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sentiment_benchmark.strategy_research import config as strategy_config
from sentiment_benchmark.strategy_research.config import (
    OutputSettings,
    StrategyConfigurationError,
    compute_run_identity,
    load_strategy_config,
    resolve_run_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_ready_smoke_config(tmp_path: Path, *, output_suffix: str = "one") -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    for name, content in {
        "events.jsonl": "{}\n",
        "scores.jsonl": "{}\n",
        "prices.csv": "session,symbol,adjusted_open\n",
        "prices.manifest.json": "{}\n",
        "prompts.toml": "[prompts]\n",
    }.items():
        (inputs / name).write_text(content, encoding="utf-8")
    path = tmp_path / f"smoke-{output_suffix}.toml"
    path.write_text(
        f"""
[run]
id = "unit-smoke"
mode = "smoke"
evaluation_start = "2026-01-12"

[data]
source = "synthetic"
events_path = "{inputs / 'events.jsonl'}"

[scoring]
prompts_path = "{inputs / 'prompts.toml'}"
scores_path = "{inputs / 'scores.jsonl'}"

[prices]
panel_path = "{inputs / 'prices.csv'}"
manifest_path = "{inputs / 'prices.manifest.json'}"

[tuning]
folds = 2
minimum_training_sessions = 5
validation_sessions = 3
minimum_validation_sessions = 2
minimum_supported_stocks = 1
minimum_active_validation_days = 1

[outputs]
derived_root = "{tmp_path / ('derived-' + output_suffix)}"
results_root = "{tmp_path / ('results-' + output_suffix)}"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_repository_configs_parse_and_expose_deliberate_readiness_state() -> None:
    smoke = load_strategy_config(REPO_ROOT / "configs/strategy_research/smoke.toml")
    formal = load_strategy_config(REPO_ROOT / "configs/strategy_research/lseg_llm_decay_v1.toml")

    assert smoke.run.mode == "smoke"
    assert smoke.tuning.candidate_count == 4
    assert formal.run.mode == "formal"
    assert formal.tuning.candidate_count == 45
    assert formal.scoring.endpoint == "https://api.cerebras.ai/v1"
    assert formal.scoring.competence_experiment_id == "clean-llm-full-20260703"
    assert formal.scoring.competence_evidence_path is not None
    assert formal.scoring.competence_evidence_path.resolve() == REPO_ROOT / "experiments/manifest.toml"
    assert any("evaluation_start" in issue for issue in formal.readiness_issues(check_files=False))


def test_ready_smoke_config_hashes_inputs_and_resolves_split_paths_without_writes(tmp_path: Path) -> None:
    config = load_strategy_config(_write_ready_smoke_config(tmp_path))
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    identity = compute_run_identity(config)
    paths = resolve_run_paths(config, identity)

    assert config.readiness_issues() == ()
    assert identity.resolved_run_id.startswith("unit-smoke-")
    assert len(identity.identity_sha256) == 64
    assert set(identity.input_identities) == {
        "dependency:exchange-calendars",
        "events",
        "price_manifest",
        "price_panel",
        "prompts",
        "scores",
    }
    assert paths.derived_dir.parent.name == "derived-one"
    assert paths.results_dir.parent.name == "results-one"
    assert paths.stage_manifest("events") == paths.results_dir / "manifests/events.json"
    assert not paths.derived_dir.exists()
    assert not paths.results_dir.exists()
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_input_content_changes_run_identity(tmp_path: Path) -> None:
    config = load_strategy_config(_write_ready_smoke_config(tmp_path))
    first = compute_run_identity(config)
    assert config.data.events_path is not None
    config.data.events_path.write_text('{"changed":true}\n', encoding="utf-8")

    second = compute_run_identity(config)

    assert first.config_sha256 == second.config_sha256
    assert first.identity_sha256 != second.identity_sha256
    assert first.resolved_run_id != second.resolved_run_id


def test_calendar_dependency_version_changes_run_identity(tmp_path: Path, monkeypatch) -> None:
    config = load_strategy_config(_write_ready_smoke_config(tmp_path))
    monkeypatch.setattr(strategy_config.metadata, "version", lambda _package: "4.13.2")
    first = compute_run_identity(config)
    monkeypatch.setattr(strategy_config.metadata, "version", lambda _package: "4.14.0")

    second = compute_run_identity(config)

    assert first.identity_sha256 != second.identity_sha256


def test_output_location_changes_config_hash_but_not_research_identity(tmp_path: Path) -> None:
    first = load_strategy_config(_write_ready_smoke_config(tmp_path, output_suffix="one"))
    second = load_strategy_config(_write_ready_smoke_config(tmp_path, output_suffix="two"))

    first_identity = compute_run_identity(first)
    second_identity = compute_run_identity(second)

    assert first.config_sha256 != second.config_sha256
    assert first_identity.identity_sha256 == second_identity.identity_sha256


def test_actionable_validation_rejects_unknown_key(tmp_path: Path) -> None:
    path = _write_ready_smoke_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n[reporting]\ninclude_profit_claims = true\n", encoding="utf-8")

    with pytest.raises(StrategyConfigurationError, match="reporting.include_profit_claims"):
        load_strategy_config(path)


def test_formal_mode_rejects_prebuilt_fixture_overrides(tmp_path: Path) -> None:
    path = tmp_path / "formal.toml"
    path.write_text(
        """
[run]
id = "formal"
mode = "formal"

[data]
source = "lseg"
events_path = "fixture.jsonl"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StrategyConfigurationError, match="allowed only in smoke mode"):
        load_strategy_config(path)


def test_formal_mode_cannot_redirect_outputs_into_existing_research_trees(tmp_path: Path) -> None:
    path = tmp_path / "formal-output.toml"
    path.write_text(
        """
[run]
id = "formal"
mode = "formal"

[outputs]
derived_root = "results/week6_trade_explorer"
results_root = "results/event_study"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StrategyConfigurationError, match="formal strategy outputs must remain"):
        load_strategy_config(path)


def test_public_path_resolver_rejects_programmatically_redirected_formal_outputs(tmp_path: Path) -> None:
    config = load_strategy_config(REPO_ROOT / "configs/strategy_research/lseg_llm_decay_v1.toml")
    redirected = replace(
        config,
        outputs=OutputSettings(tmp_path / "derived", tmp_path / "results"),
    )

    assert any("formal strategy outputs must remain" in issue for issue in redirected.readiness_issues())
    with pytest.raises(StrategyConfigurationError, match="formal strategy outputs must remain"):
        resolve_run_paths(redirected, compute_run_identity(redirected))
    assert not redirected.outputs.derived_root.exists()
    assert not redirected.outputs.results_root.exists()


def test_v1_rejects_unimplemented_scoring_concurrency(tmp_path: Path) -> None:
    path = _write_ready_smoke_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "[scoring]\n",
        "[scoring]\nconcurrency = 2\n",
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(StrategyConfigurationError, match="requires concurrency = 1"):
        load_strategy_config(path)


def test_ollama_scoring_requires_a_frozen_model_digest(tmp_path: Path) -> None:
    path = _write_ready_smoke_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "[scoring]\n",
        '[scoring]\nprovider = "ollama"\nmodel = "gemma3:latest"\n',
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(StrategyConfigurationError, match="requires an explicit scoring.model_digest"):
        load_strategy_config(path)


def test_missing_inputs_are_part_of_blocked_identity(tmp_path: Path) -> None:
    path = tmp_path / "blocked.toml"
    path.write_text(
        """
[run]
id = "blocked"
mode = "formal"

[data]
source = "lseg"
corpus_manifest = "missing-corpus.json"

[prices]
panel_path = "missing-prices.csv"
manifest_path = "missing-prices.manifest.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_strategy_config(path)

    identity = compute_run_identity(config)

    assert set(identity.missing_inputs) >= {"corpus_manifest", "price_panel", "price_manifest"}
    with pytest.raises(StrategyConfigurationError, match="evaluation_start"):
        config.require_ready()


@pytest.mark.parametrize(
    ("section", "setting"),
    [
        ("signal", "half_life_sessions = nan"),
        ("prices", "volatility_floor = inf"),
        ("execution", "cost_bps_per_side = -inf"),
        ("tuning", "half_life_grid = [1, nan]"),
    ],
)
def test_non_finite_numeric_configuration_is_rejected(tmp_path: Path, section: str, setting: str) -> None:
    path = _write_ready_smoke_config(tmp_path)
    text = path.read_text(encoding="utf-8")
    marker = f"[{section}]"
    text = text.replace(marker, f"{marker}\n{setting}", 1) if marker in text else text + f"\n{marker}\n{setting}\n"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(StrategyConfigurationError, match="finite"):
        load_strategy_config(path)
