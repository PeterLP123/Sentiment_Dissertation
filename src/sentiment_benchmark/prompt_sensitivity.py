from __future__ import annotations

import statistics as stats
from dataclasses import dataclass

from .metrics import load_metric_json
from .storage import BenchmarkStore

SENSITIVITY_METRICS = ("accuracy", "balanced_accuracy", "mcc", "macro_f1", "weighted_f1")


@dataclass(frozen=True)
class VariantResult:
    run_id: int
    prompt_id: str
    value: float


@dataclass(frozen=True)
class SensitivityResult:
    model_id: str
    scope: str
    metric: str
    variants: list[VariantResult]
    mean: float
    std: float
    minimum: float
    maximum: float
    spread: float
    cv: float

    @property
    def n(self) -> int:
        return len(self.variants)


def _metric_value(store: BenchmarkStore, run_id: int, model_id: str, scope: str, metric: str) -> float | None:
    for row in store.fetch_metrics(run_id):
        if row["model_id"] != model_id or row["scope"] != scope:
            continue
        payload = load_metric_json(row["metrics_json"])
        value = payload.get(metric)
        return float(value) if isinstance(value, (int, float)) else None
    return None


def prompt_sensitivity(
    store: BenchmarkStore,
    run_ids: list[int],
    model_id: str,
    scope: str = "primary",
    metric: str = "accuracy",
) -> SensitivityResult:
    """Summarise how one model's metric varies across a family of prompt variants.

    Each run is assumed to evaluate the same model on the same row selection under
    a different prompt, so the spread of the metric quantifies prompt sensitivity.
    """
    if metric not in SENSITIVITY_METRICS:
        raise ValueError(f"metric must be one of {', '.join(SENSITIVITY_METRICS)}")

    variants: list[VariantResult] = []
    missing: list[int] = []
    for run_id in run_ids:
        value = _metric_value(store, run_id, model_id, scope, metric)
        if value is None:
            missing.append(run_id)
            continue
        prompt_id = store.run_prompt_id(run_id) or f"run-{run_id}"
        variants.append(VariantResult(run_id=run_id, prompt_id=prompt_id, value=value))

    if missing:
        raise ValueError(
            f"No {scope}-scope {metric} metrics for model {model_id!r} in run(s): {missing}"
        )

    values = [variant.value for variant in variants]
    mean = stats.fmean(values)
    std = stats.pstdev(values) if len(values) > 1 else 0.0
    return SensitivityResult(
        model_id=model_id,
        scope=scope,
        metric=metric,
        variants=variants,
        mean=mean,
        std=std,
        minimum=min(values),
        maximum=max(values),
        spread=max(values) - min(values),
        cv=(std / abs(mean)) if mean else 0.0,
    )
