from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from sentiment_benchmark.strategy_research.artifacts import (
    StageManifestStore,
    StrategyArtifactError,
    write_immutable_json,
    write_immutable_jsonl,
    write_immutable_text,
)
from sentiment_benchmark.strategy_research.config import compute_run_identity, load_strategy_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def _race_immutable_write(path: str, value: str, start: Any, results: Any) -> None:
    start.wait()
    try:
        _, reused = write_immutable_text(path, value)
        results.put(("ok", value, reused))
    except StrategyArtifactError as exc:
        results.put(("error", value, str(exc)))


def _config():
    return load_strategy_config(REPO_ROOT / "configs/strategy_research/smoke.toml")


def test_stage_inspection_is_read_only_and_begin_is_resumable(tmp_path: Path) -> None:
    config = _config()
    identity = compute_run_identity(config)
    manifest_path = tmp_path / "results/manifests/events.json"
    store = StageManifestStore(manifest_path, identity=identity, stage="events", repo_root=tmp_path)

    inspection = store.inspect(config=config, input_identities={"corpus": "abc"})

    assert inspection.exists is False
    assert not tmp_path.joinpath("results").exists()

    started = store.begin(
        config=config,
        input_identities={"corpus": "abc"},
        command=["sentiment-bench", "strategy", "build-events"],
        runtime={"test": True},
        created_at_utc="2026-07-16T09:00:00Z",
    )
    resumed = store.begin(
        config=config,
        input_identities={"corpus": "abc"},
        command=["sentiment-bench", "strategy", "build-events"],
        runtime={"ignored": True},
    )

    assert started.action == "started"
    assert resumed.action == "resume"
    assert resumed.manifest == started.manifest


def test_completed_stage_reuses_only_matching_immutable_outputs(tmp_path: Path) -> None:
    config = _config()
    identity = compute_run_identity(config)
    output = tmp_path / "Data/derived/events/events.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text('{"event_id":"one"}\n', encoding="utf-8")
    store = StageManifestStore(
        tmp_path / "results/manifests/events.json",
        identity=identity,
        stage="events",
        repo_root=tmp_path,
    )
    store.begin(
        config=config,
        input_identities={"corpus": "abc"},
        command=["strategy", "build-events"],
        runtime={"test": True},
        created_at_utc="2026-07-16T09:00:00Z",
    )

    completed = store.complete(
        outputs={"events": output},
        row_counts={"events": 1},
        exclusions={"invalid_timestamp": 0},
        warnings=["synthetic fixture"],
        completed_at_utc="2026-07-16T09:01:00Z",
    )
    reused = store.begin(
        config=config,
        input_identities={"corpus": "abc"},
        command=["strategy", "run"],
    )

    assert completed.status == "completed"
    assert completed.outputs["events"].path == "Data/derived/events/events.jsonl"
    assert reused.action == "reuse"
    assert reused.manifest.command == ("strategy", "build-events")
    assert store.inspect(config=config, input_identities={"corpus": "abc"}, verify_outputs=True).reusable is True

    output.write_text('{"event_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(StrategyArtifactError, match="hash mismatch"):
        store.verify_outputs()


def test_stage_rejects_different_inputs_but_allows_wrapper_command(tmp_path: Path) -> None:
    config = _config()
    identity = compute_run_identity(config)
    store = StageManifestStore(tmp_path / "stage.json", identity=identity, stage="scores", repo_root=tmp_path)
    store.begin(
        config=config,
        input_identities={"events": "one"},
        command=["strategy", "score"],
        runtime={"test": True},
    )

    with pytest.raises(StrategyArtifactError, match="input identities differ"):
        store.begin(
            config=config,
            input_identities={"events": "two"},
            command=["strategy", "score"],
        )
    resumed = store.begin(
        config=config,
        input_identities={"events": "one"},
        command=["strategy", "run"],
    )
    assert resumed.action == "resume"
    assert resumed.manifest.command == ("strategy", "score")


def test_immutable_writers_reuse_identical_content_and_reject_changes(tmp_path: Path) -> None:
    text_path = tmp_path / "artifact.txt"
    json_path = tmp_path / "artifact.json"
    jsonl_path = tmp_path / "artifact.jsonl"

    assert write_immutable_text(text_path, "stable\n")[1] is False
    assert write_immutable_text(text_path, "stable\n")[1] is True
    assert write_immutable_json(json_path, {"b": 2, "a": 1})[1] is False
    assert write_immutable_json(json_path, {"a": 1, "b": 2})[1] is True
    assert write_immutable_jsonl(jsonl_path, [{"b": 2, "a": 1}])[1] is False
    assert write_immutable_jsonl(jsonl_path, [{"a": 1, "b": 2}])[1] is True

    with pytest.raises(StrategyArtifactError, match="refusing to overwrite"):
        write_immutable_text(text_path, "changed\n")


def test_completed_stage_manifest_itself_cannot_be_rewritten(tmp_path: Path) -> None:
    config = _config()
    identity = compute_run_identity(config)
    output = tmp_path / "out.json"
    output.write_text("{}\n", encoding="utf-8")
    store = StageManifestStore(tmp_path / "stage.json", identity=identity, stage="report", repo_root=tmp_path)
    store.begin(config=config, input_identities={}, command=["strategy", "report"], runtime={"test": True})
    store.complete(outputs={"report": output}, row_counts={"rows": 1})

    with pytest.raises(StrategyArtifactError, match="completed stage"):
        store.complete(outputs={"report": output}, row_counts={"rows": 2})


def test_stage_completion_cannot_precede_creation(tmp_path: Path) -> None:
    config = _config()
    identity = compute_run_identity(config)
    output = tmp_path / "out.json"
    output.write_text("{}\n", encoding="utf-8")
    store = StageManifestStore(tmp_path / "stage.json", identity=identity, stage="report", repo_root=tmp_path)
    store.begin(
        config=config,
        input_identities={},
        command=["strategy", "report"],
        runtime={"test": True},
        created_at_utc="2026-07-16T10:00:00Z",
    )

    with pytest.raises(StrategyArtifactError, match="cannot precede"):
        store.complete(outputs={"report": output}, completed_at_utc="2026-07-16T09:59:59Z")


def test_concurrent_immutable_writers_cannot_last_writer_win(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    target = tmp_path / "immutable.txt"
    processes = [
        context.Process(target=_race_immutable_write, args=(str(target), value, start, results))
        for value in ("first\n", "second\n")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=5), results.get(timeout=5)]
    results.close()

    assert sorted(outcome[0] for outcome in outcomes) == ["error", "ok"]
    winning_value = next(outcome[1] for outcome in outcomes if outcome[0] == "ok")
    assert target.read_text(encoding="utf-8") == winning_value
