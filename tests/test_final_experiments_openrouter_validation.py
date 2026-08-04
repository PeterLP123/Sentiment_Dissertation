import asyncio
import csv
import hashlib
import math
from pathlib import Path

import httpx
import pytest

import final_experiments.lib.openrouter_validation as validation
from final_experiments.lib.openrouter_validation import (
    MODEL_ID,
    PROMPT_HASH,
    PROVIDER_SLUG,
    _retry_delay_seconds,
    build_request,
    evaluate_checkpoint,
    paired_classification_comparison,
    prepare_manifest,
    run_validation,
    select_validation_rows,
)


def _write_dataset(path: Path, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Sentence", "Sentiment"])
        writer.writeheader()
        for index, label in enumerate(labels):
            writer.writerow({"Sentence": f"Headline {index}", "Sentiment": label})


def test_hash_order_sample_is_deterministic_and_label_blind(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_dataset(first, ["positive", "negative", "neutral", "positive"])
    _write_dataset(second, ["neutral", "positive", "negative", "neutral"])

    first_rows = select_validation_rows(first, 3)
    second_rows = select_validation_rows(second, 3)

    assert [row.sentence for row in first_rows] == [row.sentence for row in second_rows]


def test_request_freezes_model_provider_privacy_and_schema(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    _write_dataset(dataset, ["positive"])
    row = select_validation_rows(dataset, 1)[0]

    request = build_request(row)

    assert request["model"] == MODEL_ID
    assert request["provider"] == {
        "only": [PROVIDER_SLUG],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "max_price": {"prompt": 0.07, "completion": 0.34},
    }
    assert request["reasoning"] == {"effort": "none", "exclude": True}
    assert request["max_tokens"] == 128
    assert "max_completion_tokens" not in request
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert set(schema["schema"]["required"]) == {"positive", "negative", "neutral"}
    assert schema["schema"]["additionalProperties"] is False
    assert request["messages"][-1]["content"] == "Headline:\nHeadline 0"

    fp8_request = build_request(row, require_fp8=True)
    assert fp8_request["provider"]["quantizations"] == ["fp8"]


def test_retry_delay_respects_numeric_retry_after_header() -> None:
    response = httpx.Response(429, headers={"Retry-After": "12.5"})

    assert _retry_delay_seconds(response, attempt=2) == 12.5
    assert _retry_delay_seconds(httpx.Response(429), attempt=2) == 4.0
    assert _retry_delay_seconds(httpx.Response(429, headers={"Retry-After": "invalid"}), attempt=5) == 8.0


def test_prepare_manifest_records_public_only_boundary(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    _write_dataset(dataset, ["positive", "negative", "neutral"])

    manifest = prepare_manifest(dataset, tmp_path / "out", 2)

    assert manifest["status"] == "prepared"
    assert manifest["prompt"]["hash"] == PROMPT_HASH
    assert manifest["selection"]["label_blind"] is True
    assert manifest["input"]["licensed_lseg_text"] is False
    assert "Licensed LSEG headlines remain local" in manifest["licence_boundary"]


def test_log_loss_uses_probability_of_the_true_named_class(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    _write_dataset(dataset, ["positive"])
    selected = select_validation_rows(dataset, 1)
    responses = [
        {
            "row_number": selected[0].row_number,
            "true_label": "positive",
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "positive",
            "probabilities": {"positive": 0.7, "negative": 0.2, "neutral": 0.1},
            "latency_ms": 1.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reported_cost_usd": 0.001,
        }
    ]

    metrics = evaluate_checkpoint(selected, responses)

    assert metrics["multiclass_log_loss"] == pytest.approx(-math.log(0.7))


def test_paired_comparison_counts_discordant_predictions() -> None:
    result = paired_classification_comparison(
        ["positive", "negative", "neutral", "positive"],
        ["positive", "neutral", "neutral", "negative"],
        ["neutral", "negative", "neutral", "negative"],
        n_resamples=50,
        seed=7,
    )

    assert result["candidate_only_correct"] == 1
    assert result["reference_only_correct"] == 1
    assert result["accuracy_difference"] == pytest.approx(0.0)
    assert result["mcnemar_exact_p"] == pytest.approx(1.0)


class _FakeValidationClient:
    def __init__(self, *, fail_negative: bool) -> None:
        self.fail_negative = fail_negative

    async def close(self) -> None:
        return None

    async def classify(self, row, retries: int = 3):
        del retries
        base = {
            "row_number": row.row_number,
            "sentence_sha256": hashlib.sha256(row.sentence.strip().encode("utf-8")).hexdigest(),
            "true_label": row.hidden_label,
            "latency_ms": 1.0,
            "attempt_count": 1,
        }
        if self.fail_negative and row.hidden_label == "negative":
            return {**base, "status": "api_error", "parse_status": "error", "error": "temporary 429"}
        probabilities = {label: (0.9 if label == row.hidden_label else 0.05) for label in ("positive", "negative", "neutral")}
        return {
            **base,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": row.hidden_label,
            "probabilities": probabilities,
            "reported_cost_usd": 0.001,
        }


def test_run_is_incomplete_until_every_latest_attempt_succeeds(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    output = tmp_path / "output"
    _write_dataset(dataset, ["positive", "negative"])
    monkeypatch.setattr(
        validation,
        "PinnedOpenRouterClient",
        lambda: _FakeValidationClient(fail_negative=True),
    )

    first = asyncio.run(run_validation(dataset, output, limit=2, concurrency=2))

    assert first["status"] == "incomplete"
    assert first["counts"]["checkpoint_unique_rows"] == 2
    assert first["counts"]["checkpoint_successful_rows"] == 1
    assert first["results"]["coverage_success_share"] == pytest.approx(0.5)

    monkeypatch.setattr(
        validation,
        "PinnedOpenRouterClient",
        lambda: _FakeValidationClient(fail_negative=False),
    )
    second = asyncio.run(run_validation(dataset, output, limit=2, concurrency=1, retry_failures=True))

    assert second["status"] == "completed"
    assert second["counts"]["attempted_this_run"] == 1
    assert second["counts"]["checkpoint_successful_rows"] == 2
