from __future__ import annotations

from .agreement import AgreementResult, compute_agreement
from .constants import is_valid_label
from .metrics import _response_prediction
from .storage import BenchmarkStore


def collect_predictions(store: BenchmarkStore, run_id: int, scope: str = "primary") -> dict[str, dict[int, str]]:
    """Map each model to its valid per-row predictions for a run."""
    predictions: dict[str, dict[int, str]] = {}
    for model_id in store.run_model_ids(run_id):
        per_row: dict[int, str] = {}
        for row in store.fetch_predictions(run_id, model_id, scope):
            label = _response_prediction(dict(row))
            if is_valid_label(label):
                per_row[int(row["row_number"])] = label
        if per_row:
            predictions[model_id] = per_row
    return predictions


def run_agreement(store: BenchmarkStore, run_id: int, scope: str = "primary") -> AgreementResult | None:
    """Inter-model agreement for a run, or None if fewer than two models have predictions."""
    predictions = collect_predictions(store, run_id, scope)
    if len(predictions) < 2:
        return None
    return compute_agreement(predictions)
