from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, EvaluationResult, LLMResponseRecord
from sentiment_benchmark.storage import BenchmarkStore


def test_storage_insert_and_resume_check(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    record = LLMResponseRecord(
        row_number=2,
        model_id="test/model",
        prompt_hash="abc123",
        raw_content="positive",
        normalized_label="positive",
        parse_status="valid",
        status="success",
    )
    assert not store.response_exists(1, "test/model", 2, "abc123")
    store.save_response(1, record)
    assert store.response_exists(1, "test/model", 2, "abc123")
    store.save_response(1, record)
    responses = store.fetch_responses(1, "test/model")
    assert len(responses) == 1


def test_metrics_counts_invalid_as_wrong_and_excludes_conflicts() -> None:
    rows = [
        DatasetRow(2, "a", "positive", False, False, 1),
        DatasetRow(3, "b", "negative", False, False, 1),
        DatasetRow(4, "c", "neutral", True, True, 2),
    ]
    responses = [
        {
            "row_number": 2,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "positive",
            "latency_ms": 10,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 3,
            "status": "success",
            "parse_status": "invalid",
            "normalized_label": None,
            "latency_ms": 20,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 4,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "negative",
            "latency_ms": 30,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    ]
    primary = evaluate_responses(rows, responses, "test/model", scope="primary")
    audit = evaluate_responses(rows, responses, "test/model", scope="all")
    assert primary.row_count == 2
    assert primary.accuracy == 0.5
    assert primary.balanced_accuracy == 0.5
    assert primary.mcc == 0.0
    assert primary.invalid_output_count == 1
    assert audit.row_count == 3


def test_balanced_accuracy_and_mcc_count_invalid_as_wrong() -> None:
    rows = [
        DatasetRow(1, "a", "positive", False, False, 1),
        DatasetRow(2, "b", "negative", False, False, 1),
        DatasetRow(3, "c", "neutral", False, False, 1),
        DatasetRow(4, "d", "positive", False, False, 1),
    ]
    responses = [
        {
            "row_number": 1,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "positive",
            "latency_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 2,
            "status": "success",
            "parse_status": "invalid",
            "normalized_label": None,
            "latency_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 3,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "neutral",
            "latency_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 4,
            "status": "api_error",
            "parse_status": "error",
            "normalized_label": None,
            "latency_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    ]
    result = evaluate_responses(rows, responses, "test/model", scope="primary")
    assert result.accuracy == 0.5
    # Scored only over ALLOWED_LABELS: invalid/error map to a wrong class, not extra categories.
    assert result.balanced_accuracy == 0.5
    assert result.mcc == 0.2


def _eval_result(model_id: str, scope: str, accuracy: float) -> EvaluationResult:
    return EvaluationResult(
        model_id=model_id,
        scope=scope,
        row_count=10,
        accuracy=accuracy,
        balanced_accuracy=accuracy,
        mcc=accuracy,
        macro_f1=accuracy,
        weighted_f1=accuracy,
        per_class={},
        confusion_matrix={},
        invalid_output_count=0,
        api_error_count=0,
        mean_latency_ms=100.0,
        total_prompt_tokens=10,
        total_completion_tokens=10,
        total_tokens=20,
    )


def test_fetch_metrics_returns_all_scopes(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    store.save_metrics(1, _eval_result("a/model", "primary", 0.9))
    store.save_metrics(1, _eval_result("a/model", "all", 0.8))
    store.save_metrics(2, _eval_result("b/model", "primary", 0.5))

    rows = store.fetch_metrics(1)
    assert {(row["model_id"], row["scope"]) for row in rows} == {("a/model", "primary"), ("a/model", "all")}


def test_run_cost_by_model_sums_recorded_generation_cost(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    store.save_generation_metadata(1, 2, "a/model", "gen-1", {"total_cost": 0.001})
    store.save_generation_metadata(1, 3, "a/model", "gen-2", {"total_cost": 0.002})
    store.save_generation_metadata(1, 4, "b/model", "gen-3", {"total_cost": 0.005})
    # Missing cost should be ignored rather than counted as zero.
    store.save_generation_metadata(1, 5, "b/model", "gen-4", {})

    costs = store.run_cost_by_model(1)
    assert costs["a/model"] == 0.003
    assert costs["b/model"] == 0.005
    assert store.run_cost_by_model(99) == {}

