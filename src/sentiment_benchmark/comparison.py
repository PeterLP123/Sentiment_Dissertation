from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import _response_prediction, bootstrap_metric_ci, mcnemar_test
from .storage import BenchmarkStore


@dataclass(frozen=True)
class ModelTarget:
    """One side of a comparison: a model as evaluated within a specific run."""

    run_id: int
    model_id: str

    def label(self) -> str:
        return f"run {self.run_id} · {self.model_id}"


@dataclass(frozen=True)
class ComparisonResult:
    target_a: ModelTarget
    target_b: ModelTarget
    scope: str
    metric: str
    n_paired: int
    ci_a: dict[str, float | int]
    ci_b: dict[str, float | int]
    mcnemar: dict[str, float | int | str]

    @property
    def point_a(self) -> float:
        return float(self.ci_a["point"])

    @property
    def point_b(self) -> float:
        return float(self.ci_b["point"])

    def is_significant(self, alpha: float = 0.05) -> bool:
        return float(self.mcnemar["p_value"]) < alpha


def _aligned_predictions(store: BenchmarkStore, target: ModelTarget, scope: str) -> dict[int, tuple[str, str]]:
    rows = store.fetch_predictions(target.run_id, target.model_id, scope)
    aligned: dict[int, tuple[str, str]] = {}
    for row in rows:
        record: dict[str, Any] = dict(row)
        aligned[int(record["row_number"])] = (str(record["hidden_label"]), _response_prediction(record))
    return aligned


def compare_models(
    store: BenchmarkStore,
    target_a: ModelTarget,
    target_b: ModelTarget,
    scope: str = "primary",
    metric: str = "accuracy",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ComparisonResult:
    """Compare two model/run targets on the rows they share.

    Predictions are paired by ``row_number`` so McNemar's test (which requires
    the same items for both models) is valid. Per-model bootstrap confidence
    intervals are computed on the same shared rows for a like-for-like reading.
    """
    aligned_a = _aligned_predictions(store, target_a, scope)
    aligned_b = _aligned_predictions(store, target_b, scope)
    shared = sorted(set(aligned_a) & set(aligned_b))

    y_true = [aligned_a[row_number][0] for row_number in shared]
    y_pred_a = [aligned_a[row_number][1] for row_number in shared]
    y_pred_b = [aligned_b[row_number][1] for row_number in shared]

    ci_a = bootstrap_metric_ci(y_true, y_pred_a, metric, n_resamples=n_resamples, confidence=confidence, seed=seed)
    ci_b = bootstrap_metric_ci(y_true, y_pred_b, metric, n_resamples=n_resamples, confidence=confidence, seed=seed)
    mcnemar = mcnemar_test(y_true, y_pred_a, y_pred_b)

    return ComparisonResult(
        target_a=target_a,
        target_b=target_b,
        scope=scope,
        metric=metric,
        n_paired=len(shared),
        ci_a=ci_a,
        ci_b=ci_b,
        mcnemar=mcnemar,
    )
