import json
import sqlite3

import pytest

from sentiment_benchmark.metrics import (
    brier_score_multiclass,
    evaluate_responses,
    expected_calibration_error,
)
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord
from sentiment_benchmark.storage import BenchmarkStore


def _probs(positive: float, negative: float, neutral: float) -> dict[str, float]:
    return {"positive": positive, "negative": negative, "neutral": neutral}


def test_brier_score_hand_computed_values() -> None:
    # Perfectly confident and correct: 0.
    assert brier_score_multiclass(["positive"], [_probs(1.0, 0.0, 0.0)]) == 0.0
    # Perfectly confident and wrong: (0-1)^2 + 1^2 = 2 (the multiclass maximum).
    assert brier_score_multiclass(["positive"], [_probs(0.0, 1.0, 0.0)]) == 2.0
    # (0.5-1)^2 + 0.25^2 + 0.25^2 = 0.375.
    assert brier_score_multiclass(["positive"], [_probs(0.5, 0.25, 0.25)]) == pytest.approx(0.375)
    # Mean over items.
    score = brier_score_multiclass(
        ["positive", "positive"],
        [_probs(1.0, 0.0, 0.0), _probs(0.5, 0.25, 0.25)],
    )
    assert score == pytest.approx(0.1875)


def test_brier_score_requires_items() -> None:
    with pytest.raises(ValueError):
        brier_score_multiclass([], [])


def test_ece_perfectly_calibrated_confident_predictions() -> None:
    assert expected_calibration_error(["positive"], [_probs(1.0, 0.0, 0.0)]) == 0.0


def test_ece_hand_computed_value() -> None:
    # Both items land in the same confidence bin (0.6): mean confidence 0.6,
    # accuracy 0.5, so ECE = |0.6 - 0.5| = 0.1.
    y_true = ["positive", "negative"]
    probabilities = [_probs(0.6, 0.3, 0.1), _probs(0.6, 0.3, 0.1)]
    assert expected_calibration_error(y_true, probabilities) == pytest.approx(0.1)


def test_ece_weights_bins_by_support() -> None:
    # Three items: two in the 0.6 bin (one right, one wrong) and one in the
    # top bin (right). ECE = (2/3)*|0.6-0.5| + (1/3)*|1.0-1.0| = 0.0666...
    y_true = ["positive", "negative", "neutral"]
    probabilities = [
        _probs(0.6, 0.3, 0.1),
        _probs(0.6, 0.3, 0.1),
        _probs(0.0, 0.0, 1.0),
    ]
    assert expected_calibration_error(y_true, probabilities) == pytest.approx(2 / 3 * 0.1)


def test_ece_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([], [])
    with pytest.raises(ValueError):
        expected_calibration_error(["positive"], [_probs(1.0, 0.0, 0.0)], n_bins=0)


def _response(row_number: int, label: str, probabilities: dict[str, float] | None) -> dict:
    return {
        "row_number": row_number,
        "status": "success",
        "parse_status": "valid",
        "normalized_label": label,
        # Probabilities arrive as JSON text when responses come from storage.
        "label_probabilities": json.dumps(probabilities) if probabilities is not None else None,
        "latency_ms": 1,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }


def test_evaluate_responses_computes_calibration_from_soft_labels() -> None:
    rows = [
        DatasetRow(1, "a", "positive", False, False, 1),
        DatasetRow(2, "b", "negative", False, False, 1),
        DatasetRow(3, "c", "neutral", False, False, 1),
    ]
    responses = [
        _response(1, "positive", _probs(1.0, 0.0, 0.0)),
        _response(2, "positive", _probs(0.5, 0.25, 0.25)),
        # A row without probabilities (e.g. parse fell back) is excluded from
        # calibration but still scored for accuracy.
        _response(3, "neutral", None),
    ]
    result = evaluate_responses(rows, responses, "test/model", scope="primary")
    assert result.calibration is not None
    assert result.calibration["n_scored"] == 2
    assert result.calibration["n_bins"] == 10
    # Items: correct w/ Brier 0, wrong-true-negative w/ (0.5)^2+(0.25-1)^2+(0.25)^2 = 0.875.
    assert result.calibration["brier_score"] == pytest.approx((0.0 + 0.875) / 2)
    # Bins: conf 1.0 correct (gap 0) and conf 0.5 wrong (gap 0.5), equal weight.
    assert result.calibration["ece"] == pytest.approx(0.25)


def test_evaluate_responses_without_probabilities_has_no_calibration() -> None:
    rows = [DatasetRow(1, "a", "positive", False, False, 1)]
    responses = [_response(1, "positive", None)]
    result = evaluate_responses(rows, responses, "test/model", scope="primary")
    assert result.calibration is None


def test_storage_roundtrips_label_probabilities(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    record = LLMResponseRecord(
        row_number=1,
        model_id="test/model",
        prompt_hash="hash",
        raw_content='{"positive": 0.7, "negative": 0.1, "neutral": 0.2}',
        normalized_label="positive",
        parse_status="valid",
        status="success",
        label_probabilities={"positive": 0.7, "negative": 0.1, "neutral": 0.2},
    )
    store.save_response(1, record)
    fetched = store.fetch_responses(1, "test/model")
    assert len(fetched) == 1
    stored = json.loads(fetched[0]["label_probabilities"])
    assert stored == {"positive": 0.7, "negative": 0.1, "neutral": 0.2}

    rows = [DatasetRow(1, "a", "positive", False, False, 1)]
    result = evaluate_responses(rows, fetched, "test/model", scope="primary")
    assert result.calibration is not None
    assert result.calibration["n_scored"] == 1


def test_initialize_migrates_responses_table_for_label_probabilities(tmp_path) -> None:
    db_path = tmp_path / "old.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                row_number INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                raw_content TEXT,
                normalized_label TEXT,
                parse_status TEXT NOT NULL,
                explanation TEXT,
                raw_response_json TEXT,
                latency_ms REAL,
                status TEXT NOT NULL,
                error TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                generation_id TEXT,
                created_at TEXT NOT NULL,
                UNIQUE (run_id, row_number, model_id, prompt_hash)
            );
            """
        )
        connection.commit()

    store = BenchmarkStore(db_path)
    store.initialize()

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(responses)").fetchall()}
    assert "label_probabilities" in columns
