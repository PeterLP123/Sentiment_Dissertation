from pathlib import Path

from sentiment_benchmark.comparison import ModelTarget, compare_models
from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord, RunConfig
from sentiment_benchmark.prompts import make_prompt
from sentiment_benchmark.storage import BenchmarkStore


def _record(prompt_hash: str, row_number: int, model_id: str, label: str) -> LLMResponseRecord:
    return LLMResponseRecord(
        row_number=row_number,
        model_id=model_id,
        prompt_hash=prompt_hash,
        raw_content=label,
        normalized_label=label,
        parse_status="valid",
        status="success",
        latency_ms=10,
    )


def _seed(tmp_path: Path) -> tuple[BenchmarkStore, int]:
    db_path = tmp_path / "compare.sqlite"
    store = BenchmarkStore(db_path)
    store.initialize()
    rows = [
        DatasetRow(1, "a", "positive", False, False, 1),
        DatasetRow(2, "b", "negative", False, False, 1),
        DatasetRow(3, "c", "neutral", False, False, 1),
        DatasetRow(4, "d", "positive", True, True, 2),  # conflicting duplicate -> excluded from primary
    ]
    store.upsert_dataset(rows, str(tmp_path / "data.csv"))
    prompt = make_prompt("t", "sys", "{sentence}", "label_only")
    store.save_prompt(prompt)
    config = RunConfig(
        models=["model-a", "model-b"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(tmp_path / "data.csv"),
        db_path=str(db_path),
        base_url="x",
        sample_per_class=1,
    )
    run_id = store.create_run(config, [1, 2, 3, 4])
    # model-a: all correct. model-b: row 2 wrong (predicts positive instead of negative).
    for row_number, label in {1: "positive", 2: "negative", 3: "neutral", 4: "positive"}.items():
        store.save_response(run_id, _record(prompt.prompt_hash, row_number, "model-a", label))
    for row_number, label in {1: "positive", 2: "positive", 3: "neutral", 4: "positive"}.items():
        store.save_response(run_id, _record(prompt.prompt_hash, row_number, "model-b", label))
    for model_id in ("model-a", "model-b"):
        store.save_metrics(run_id, evaluate_responses(rows, store.fetch_responses(run_id, model_id), model_id, scope="primary"))
    return store, run_id


def test_compare_models_primary_scope_excludes_conflicts_and_pairs_by_row(tmp_path: Path) -> None:
    store, run_id = _seed(tmp_path)

    result = compare_models(
        store,
        ModelTarget(run_id, "model-a"),
        ModelTarget(run_id, "model-b"),
        scope="primary",
        metric="accuracy",
        n_resamples=200,
        seed=1,
    )

    assert result.n_paired == 3  # conflicting row 4 excluded
    assert result.point_a == 1.0
    assert abs(result.point_b - (2 / 3)) < 1e-9
    assert result.mcnemar["b_a_correct_b_wrong"] == 1
    assert result.mcnemar["c_a_wrong_b_correct"] == 0
    assert result.mcnemar["n_discordant"] == 1


def test_compare_models_all_scope_includes_every_shared_row(tmp_path: Path) -> None:
    store, run_id = _seed(tmp_path)

    result = compare_models(store, ModelTarget(run_id, "model-a"), ModelTarget(run_id, "model-b"), scope="all")

    assert result.n_paired == 4
    assert result.point_a == 1.0
    assert result.point_b == 0.75


def test_compare_models_no_overlap_returns_zero_paired(tmp_path: Path) -> None:
    store, run_id = _seed(tmp_path)

    result = compare_models(store, ModelTarget(run_id, "model-a"), ModelTarget(run_id, "missing-model"))

    assert result.n_paired == 0
