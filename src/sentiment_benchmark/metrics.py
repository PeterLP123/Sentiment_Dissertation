from __future__ import annotations

import json
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .constants import ALLOWED_LABELS
from .models import DatasetRow, EvaluationResult


def _response_prediction(response: dict[str, Any]) -> str:
    if response["status"] != "success":
        return "__error__"
    if response["parse_status"] != "valid" or not response["normalized_label"]:
        return "__invalid__"
    return str(response["normalized_label"])


def evaluate_responses(
    rows: list[DatasetRow],
    responses: list[Any],
    model_id: str,
    scope: str = "primary",
) -> EvaluationResult:
    rows_by_number = {row.row_number: row for row in rows}
    response_dicts = [dict(response) for response in responses]
    if scope == "primary":
        response_dicts = [
            response
            for response in response_dicts
            if response["row_number"] in rows_by_number
            and not rows_by_number[response["row_number"]].has_conflicting_duplicate
        ]

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for response in response_dicts:
        row = rows_by_number.get(response["row_number"])
        if row is None:
            continue
        y_true.append(row.hidden_label)
        y_pred.append(_response_prediction(response))
        if response["latency_ms"] is not None:
            latencies.append(float(response["latency_ms"]))
        total_prompt_tokens += int(response["prompt_tokens"] or 0)
        total_completion_tokens += int(response["completion_tokens"] or 0)
        total_tokens += int(response["total_tokens"] or 0)

    if not y_true:
        return EvaluationResult(
            model_id=model_id,
            scope=scope,
            row_count=0,
            accuracy=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            per_class={label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0} for label in ALLOWED_LABELS},
            confusion_matrix={
                label: {prediction: 0 for prediction in (*ALLOWED_LABELS, "__invalid__", "__error__")}
                for label in ALLOWED_LABELS
            },
            invalid_output_count=0,
            api_error_count=0,
            mean_latency_ms=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(ALLOWED_LABELS),
        zero_division=0,
    )
    _, _, macro_f1_values, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(ALLOWED_LABELS),
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1_values, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(ALLOWED_LABELS),
        average="weighted",
        zero_division=0,
    )
    prediction_labels = [*ALLOWED_LABELS, "__invalid__", "__error__"]
    matrix = confusion_matrix(y_true, y_pred, labels=prediction_labels)
    confusion = {
        actual: {predicted: int(matrix[i][j]) for j, predicted in enumerate(prediction_labels)}
        for i, actual in enumerate(prediction_labels)
        if actual in ALLOWED_LABELS
    }
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }
        for index, label in enumerate(ALLOWED_LABELS)
    }
    invalid_output_count = sum(1 for response in response_dicts if response["parse_status"] == "invalid")
    api_error_count = sum(1 for response in response_dicts if response["status"] != "success")
    mean_latency_ms = sum(latencies) / len(latencies) if latencies else None
    return EvaluationResult(
        model_id=model_id,
        scope=scope,
        row_count=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(macro_f1_values),
        weighted_f1=float(weighted_f1_values),
        per_class=per_class,
        confusion_matrix=confusion,
        invalid_output_count=invalid_output_count,
        api_error_count=api_error_count,
        mean_latency_ms=mean_latency_ms,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
    )


def load_metric_json(value: str) -> dict[str, Any]:
    return json.loads(value)
