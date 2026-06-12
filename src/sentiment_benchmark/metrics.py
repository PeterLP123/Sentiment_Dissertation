from __future__ import annotations

import json
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from .constants import ALLOWED_LABELS, CONFUSION_PREDICTION_LABELS, is_valid_label
from .models import DatasetRow, EvaluationResult

_LABELS = list(ALLOWED_LABELS)


def _response_prediction(response: dict[str, Any]) -> str:
    if response["status"] != "success":
        return "__error__"
    if response["parse_status"] != "valid" or not response["normalized_label"]:
        return "__invalid__"
    return str(response["normalized_label"])


def _response_probabilities(response: dict[str, Any]) -> dict[str, float] | None:
    """Label probabilities from a response record, or None when absent/unusable.

    Storage serializes the distribution as JSON text; in-memory records may
    carry it as a dict already. Anything that does not decode to a full
    distribution over ALLOWED_LABELS is treated as absent.
    """
    if response.get("status") != "success" or response.get("parse_status") != "valid":
        return None
    value = response.get("label_probabilities")
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    try:
        probabilities = {label: float(value[label]) for label in ALLOWED_LABELS}
    except (KeyError, TypeError, ValueError):
        return None
    return probabilities


def brier_score_multiclass(y_true: list[str], probabilities: list[dict[str, float]]) -> float:
    """Mean multiclass Brier score over ALLOWED_LABELS.

    Per item: sum over classes of (p_c - 1[y=c])^2. Range [0, 2]; lower is
    better; a perfectly confident correct prediction scores 0.
    """
    if not y_true:
        raise ValueError("brier_score_multiclass requires at least one item")
    total = 0.0
    for true_label, probs in zip(y_true, probabilities, strict=True):
        total += sum((probs[label] - (1.0 if label == true_label else 0.0)) ** 2 for label in ALLOWED_LABELS)
    return total / len(y_true)


def expected_calibration_error(
    y_true: list[str],
    probabilities: list[dict[str, float]],
    n_bins: int = 10,
) -> float:
    """Expected calibration error with equal-width confidence bins.

    Confidence is the maximum class probability; an item is accurate when the
    argmax class (ties broken in ALLOWED_LABELS order, matching the parser)
    equals the true label. ECE is the support-weighted mean absolute gap
    between per-bin confidence and per-bin accuracy.
    """
    if not y_true:
        raise ValueError("expected_calibration_error requires at least one item")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    bin_totals = [0] * n_bins
    bin_confidence = [0.0] * n_bins
    bin_accuracy = [0.0] * n_bins
    for true_label, probs in zip(y_true, probabilities, strict=True):
        predicted = max(ALLOWED_LABELS, key=lambda label: probs[label])
        confidence = probs[predicted]
        bin_index = min(int(confidence * n_bins), n_bins - 1)
        bin_totals[bin_index] += 1
        bin_confidence[bin_index] += confidence
        bin_accuracy[bin_index] += 1.0 if predicted == true_label else 0.0
    n = len(y_true)
    ece = 0.0
    for index in range(n_bins):
        if bin_totals[index] == 0:
            continue
        mean_confidence = bin_confidence[index] / bin_totals[index]
        mean_accuracy = bin_accuracy[index] / bin_totals[index]
        ece += (bin_totals[index] / n) * abs(mean_confidence - mean_accuracy)
    return ece


def compute_calibration(
    y_true: list[str],
    probabilities: list[dict[str, float]],
    n_bins: int = 10,
) -> dict[str, float | int]:
    return {
        "brier_score": brier_score_multiclass(y_true, probabilities),
        "ece": expected_calibration_error(y_true, probabilities, n_bins=n_bins),
        "n_scored": len(y_true),
        "n_bins": n_bins,
    }


def _prediction_for_multiclass_metrics(true_label: str, prediction: str) -> str:
    """Map invalid/error predictions to a wrong allowed label for MCC and balanced accuracy."""
    if is_valid_label(prediction):
        return prediction
    for label in ALLOWED_LABELS:
        if label != true_label:
            return label
    return ALLOWED_LABELS[0]


def _empty_per_class() -> dict[str, dict[str, float]]:
    zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}
    return {label: dict(zero) for label in ALLOWED_LABELS}


def _empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {
        label: {prediction: 0 for prediction in CONFUSION_PREDICTION_LABELS}
        for label in ALLOWED_LABELS
    }


def _empty_evaluation_result(model_id: str, scope: str) -> EvaluationResult:
    return EvaluationResult(
        model_id=model_id,
        scope=scope,
        row_count=0,
        accuracy=0.0,
        balanced_accuracy=0.0,
        mcc=0.0,
        macro_f1=0.0,
        weighted_f1=0.0,
        per_class=_empty_per_class(),
        confusion_matrix=_empty_confusion_matrix(),
        invalid_output_count=0,
        api_error_count=0,
        mean_latency_ms=None,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_tokens=0,
    )


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
    calibration_true: list[str] = []
    calibration_probs: list[dict[str, float]] = []
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
        probabilities = _response_probabilities(response)
        if probabilities is not None:
            calibration_true.append(row.hidden_label)
            calibration_probs.append(probabilities)
        if response["latency_ms"] is not None:
            latencies.append(float(response["latency_ms"]))
        total_prompt_tokens += int(response["prompt_tokens"] or 0)
        total_completion_tokens += int(response["completion_tokens"] or 0)
        total_tokens += int(response["total_tokens"] or 0)

    if not y_true:
        return _empty_evaluation_result(model_id, scope)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABELS, zero_division=0
    )
    _, _, macro_f1_values, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABELS, average="macro", zero_division=0
    )
    _, _, weighted_f1_values, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABELS, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CONFUSION_PREDICTION_LABELS)
    confusion = {
        actual: {predicted: int(matrix[i][j]) for j, predicted in enumerate(CONFUSION_PREDICTION_LABELS)}
        for i, actual in enumerate(CONFUSION_PREDICTION_LABELS)
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
    y_pred_scored = [_prediction_for_multiclass_metrics(true, pred) for true, pred in zip(y_true, y_pred, strict=True)]
    return EvaluationResult(
        model_id=model_id,
        scope=scope,
        row_count=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred_scored)),
        mcc=float(matthews_corrcoef(y_true, y_pred_scored)),
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
        calibration=compute_calibration(calibration_true, calibration_probs) if calibration_true else None,
    )


def load_metric_json(value: str) -> dict[str, Any]:
    return json.loads(value)


def _metric_value(y_true: list[str], y_pred: list[str], metric: str) -> float:
    from sklearn.metrics import f1_score

    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, labels=_LABELS, average="macro", zero_division=0))
    return float(accuracy_score(y_true, y_pred))


def bootstrap_metric_ci(
    y_true: list[str],
    y_pred: list[str],
    metric: str = "accuracy",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap percentile confidence interval for accuracy or macro-F1.

    Indices are resampled with replacement ``n_resamples`` times using a numpy
    Generator seeded with ``seed`` so the interval is reproducible.
    """
    n = len(y_true)
    if n == 0:
        return {
            "point": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "confidence": confidence,
            "n": 0,
            "n_resamples": n_resamples,
        }

    import numpy as np

    point = _metric_value(y_true, y_pred, metric)
    true_array = np.asarray(y_true, dtype=object)
    pred_array = np.asarray(y_pred, dtype=object)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        samples[index] = _metric_value(true_array[indices].tolist(), pred_array[indices].tolist(), metric)

    lower_pct = (1.0 - confidence) / 2.0 * 100.0
    upper_pct = (1.0 + confidence) / 2.0 * 100.0
    return {
        "point": point,
        "lower": float(np.percentile(samples, lower_pct)),
        "upper": float(np.percentile(samples, upper_pct)),
        "confidence": confidence,
        "n": n,
        "n_resamples": n_resamples,
    }


def mcnemar_test(y_true: list[str], y_pred_a: list[str], y_pred_b: list[str]) -> dict[str, float | int | str]:
    """McNemar's test comparing two models' correctness on the same items.

    Uses the exact two-sided binomial test for small discordant counts
    (``n_discordant <= 25``) and the continuity-corrected chi-square statistic
    otherwise. ``correct`` means the prediction equals the true label.
    """
    b = sum(1 for true, pred_a, pred_b in zip(y_true, y_pred_a, y_pred_b, strict=False) if pred_a == true and pred_b != true)
    c = sum(1 for true, pred_a, pred_b in zip(y_true, y_pred_a, y_pred_b, strict=False) if pred_a != true and pred_b == true)
    n_discordant = b + c
    base: dict[str, float | int | str] = {
        "n": len(y_true),
        "n_discordant": n_discordant,
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
    }
    if n_discordant == 0:
        return {**base, "statistic": 0.0, "p_value": 1.0, "method": "exact_binomial"}

    if n_discordant <= 25:
        from scipy.stats import binomtest

        p_value = float(binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue)
        return {**base, "statistic": 0.0, "p_value": p_value, "method": "exact_binomial"}

    from scipy.stats import chi2

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(chi2.sf(statistic, 1))
    return {**base, "statistic": float(statistic), "p_value": p_value, "method": "chi2_continuity"}
