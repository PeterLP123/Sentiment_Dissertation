import json
from pathlib import Path

from sentiment_benchmark.exporter import export_run
from sentiment_benchmark.metrics import bootstrap_metric_ci, evaluate_responses, mcnemar_test
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord, RunConfig
from sentiment_benchmark.prompts import make_prompt
from sentiment_benchmark.storage import BenchmarkStore

LABELS = ("positive", "negative", "neutral")


def _mixed_predictions(correct: int, total: int) -> tuple[list[str], list[str]]:
    y_true = [LABELS[index % 3] for index in range(total)]
    y_pred: list[str] = []
    for index in range(total):
        if index < correct:
            y_pred.append(y_true[index])
        else:
            y_pred.append(LABELS[(index + 1) % 3])
    return y_true, y_pred


def test_bootstrap_accuracy_point_within_interval_and_deterministic() -> None:
    y_true, y_pred = _mixed_predictions(correct=14, total=20)
    first = bootstrap_metric_ci(y_true, y_pred, "accuracy", n_resamples=500, seed=123)
    second = bootstrap_metric_ci(y_true, y_pred, "accuracy", n_resamples=500, seed=123)

    assert first == second
    assert first["n"] == 20
    assert first["n_resamples"] == 500
    assert first["confidence"] == 0.95
    assert first["lower"] <= first["point"] <= first["upper"]
    assert abs(first["point"] - 0.7) < 1e-9


def test_bootstrap_macro_f1_point_within_interval() -> None:
    y_true, y_pred = _mixed_predictions(correct=15, total=21)
    result = bootstrap_metric_ci(y_true, y_pred, "macro_f1", n_resamples=500, seed=7)

    assert result["n"] == 21
    assert 0.0 <= result["lower"] <= result["point"] <= result["upper"] <= 1.0


def test_bootstrap_empty_input_returns_zeros() -> None:
    result = bootstrap_metric_ci([], [], "accuracy")

    assert result["n"] == 0
    assert result["point"] == 0.0
    assert result["lower"] == 0.0
    assert result["upper"] == 0.0


def test_mcnemar_known_discordant_counts() -> None:
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred_a = ["positive", "negative", "positive", "positive"]
    y_pred_b = ["negative", "negative", "neutral", "positive"]

    result = mcnemar_test(y_true, y_pred_a, y_pred_b)

    assert result["b_a_correct_b_wrong"] == 1
    assert result["c_a_wrong_b_correct"] == 1
    assert result["n_discordant"] == 2
    assert result["n"] == 4
    assert result["method"] == "exact_binomial"
    assert 0.0 <= result["p_value"] <= 1.0


def test_mcnemar_no_discordant_pairs() -> None:
    y_true = ["positive", "negative", "neutral"]
    predictions = ["positive", "negative", "neutral"]

    result = mcnemar_test(y_true, predictions, predictions)

    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0
    assert result["statistic"] == 0.0
    assert result["method"] == "exact_binomial"


def test_mcnemar_exact_path_small_p_value() -> None:
    y_true = ["positive"] * 15
    y_pred_a = ["positive"] * 15
    y_pred_b = ["negative"] * 15

    result = mcnemar_test(y_true, y_pred_a, y_pred_b)

    assert result["b_a_correct_b_wrong"] == 15
    assert result["c_a_wrong_b_correct"] == 0
    assert result["method"] == "exact_binomial"
    assert result["p_value"] < 0.05


def test_mcnemar_chi2_path_for_large_discordant() -> None:
    y_true = ["positive"] * 40
    y_pred_a = ["positive"] * 40
    y_pred_b = ["negative"] * 40

    result = mcnemar_test(y_true, y_pred_a, y_pred_b)

    assert result["n_discordant"] == 40
    assert result["method"] == "chi2_continuity"
    assert result["statistic"] > 0.0
    assert result["p_value"] < 0.05


def _make_record(prompt_hash: str, row_number: int, model_id: str, label: str) -> LLMResponseRecord:
    return LLMResponseRecord(
        row_number=row_number,
        model_id=model_id,
        prompt_hash=prompt_hash,
        raw_content=label,
        normalized_label=label,
        parse_status="valid",
        status="success",
        latency_ms=10,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )


def _seed_run(tmp_path: Path, seed: int) -> tuple[Path, int]:
    db_path = tmp_path / "benchmark.sqlite"
    dataset_path = tmp_path / "data.csv"
    store = BenchmarkStore(db_path)
    store.initialize()
    rows = [
        DatasetRow(1, "positive one", "positive", False, False, 1),
        DatasetRow(2, "negative two", "negative", False, False, 1),
        DatasetRow(3, "neutral three", "neutral", False, False, 1),
        DatasetRow(4, "conflicting four", "positive", True, True, 2),
    ]
    store.upsert_dataset(rows, str(dataset_path))
    prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nLabel:", "label_only")
    store.save_prompt(prompt)
    config = RunConfig(
        models=["model-a", "model-b"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(dataset_path),
        db_path=str(db_path),
        base_url="https://openrouter.test/api/v1",
        sample_per_class=1,
        seed=seed,
    )
    run_id = store.create_run(config, [1, 2, 3, 4])
    model_a_labels = {1: "positive", 2: "negative", 3: "neutral", 4: "positive"}
    model_b_labels = {1: "positive", 2: "positive", 3: "neutral", 4: "positive"}
    for row_number, label in model_a_labels.items():
        store.save_response(run_id, _make_record(prompt.prompt_hash, row_number, "model-a", label))
    for row_number, label in model_b_labels.items():
        store.save_response(run_id, _make_record(prompt.prompt_hash, row_number, "model-b", label))
    store.mark_run_complete(run_id)
    for model_id in ("model-a", "model-b"):
        result = evaluate_responses(rows, store.fetch_responses(run_id, model_id), model_id, scope="primary")
        store.save_metrics(run_id, result)
    return db_path, run_id


def test_export_run_writes_statistics_json(tmp_path: Path) -> None:
    db_path, run_id = _seed_run(tmp_path, seed=7)

    paths = export_run(db_path, run_id, output_dir=tmp_path / "exports")
    exported_names = {Path(path).name for path in paths}
    assert "statistics.json" in exported_names

    statistics = json.loads((tmp_path / "exports" / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["scope"] == "primary"
    assert statistics["seed"] == 7
    assert set(statistics["per_model"]) == {"model-a", "model-b"}
    for model_statistics in statistics["per_model"].values():
        assert model_statistics["n"] == 3
        assert {"point", "lower", "upper", "confidence", "n", "n_resamples"} <= set(model_statistics["accuracy_ci"])
        assert {"point", "lower", "upper", "confidence", "n", "n_resamples"} <= set(model_statistics["macro_f1_ci"])

    assert len(statistics["pairwise_mcnemar"]) == 1
    pair = statistics["pairwise_mcnemar"][0]
    assert pair["model_a"] == "model-a"
    assert pair["model_b"] == "model-b"
    assert {"n", "n_discordant", "b_a_correct_b_wrong", "c_a_wrong_b_correct", "statistic", "p_value", "method"} <= set(pair)


def test_export_run_statistics_robust_without_responses(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.sqlite"
    dataset_path = tmp_path / "data.csv"
    store = BenchmarkStore(db_path)
    store.initialize()
    store.upsert_dataset([DatasetRow(1, "positive one", "positive", False, False, 1)], str(dataset_path))
    prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nLabel:", "label_only")
    store.save_prompt(prompt)
    config = RunConfig(
        models=["model-a"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(dataset_path),
        db_path=str(db_path),
        base_url="https://openrouter.test/api/v1",
        sample_per_class=1,
    )
    run_id = store.create_run(config, [1])

    paths = export_run(db_path, run_id, output_dir=tmp_path / "exports")
    exported_names = {Path(path).name for path in paths}
    assert "statistics.json" in exported_names

    statistics = json.loads((tmp_path / "exports" / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["per_model"] == {}
    assert statistics["pairwise_mcnemar"] == []
