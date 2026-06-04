from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord
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
    assert primary.invalid_output_count == 1
    assert audit.row_count == 3

