from pathlib import Path

import pytest

from sentiment_benchmark.exporter import export_run
from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord, RunConfig
from sentiment_benchmark.prompts import make_prompt
from sentiment_benchmark.storage import BenchmarkStore

pytest.importorskip("matplotlib")


def _metric_record(model_id: str, accuracy: float) -> dict:
    return {
        "model_id": model_id,
        "scope": "primary",
        "row_count": 3,
        "accuracy": accuracy,
        "macro_f1": accuracy - 0.05,
        "weighted_f1": accuracy,
        "per_class": {
            "positive": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 1.0},
            "negative": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 1.0},
            "neutral": {"precision": 0.5, "recall": 0.4, "f1": 0.45, "support": 1.0},
        },
        "confusion_matrix": {
            "positive": {"positive": 1, "negative": 0, "neutral": 0, "__invalid__": 0, "__error__": 0},
            "negative": {"positive": 1, "negative": 0, "neutral": 0, "__invalid__": 0, "__error__": 0},
            "neutral": {"positive": 0, "negative": 0, "neutral": 0, "__invalid__": 1, "__error__": 0},
        },
        "invalid_output_count": 1,
        "api_error_count": 0,
    }


def _operational_entry(p95: float | None, usd_per_1k: float | None) -> dict:
    return {
        "n_rows": 3,
        "latency_ms": {"n": 3 if p95 is not None else 0, "mean": p95, "p50": p95, "p95": p95},
        "cost": {
            "n_rows_with_cost": 3 if usd_per_1k is not None else 0,
            "total_usd": usd_per_1k,
            "usd_per_1k_rows": usd_per_1k,
        },
        "tokens": {"total_prompt": 9, "total_completion": 3, "mean_total_per_row": 4.0},
        "invalid_count": 0,
        "invalid_rate": 0.0,
        "api_error_count": 0,
        "api_error_rate": 0.0,
    }


def _statistics() -> dict:
    return {
        "scope": "primary",
        "seed": 42,
        "per_model": {
            "model/a": {
                "accuracy_ci": {"point": 0.9, "lower": 0.7, "upper": 1.0, "confidence": 0.95},
                "macro_f1_ci": {"point": 0.85, "lower": 0.65, "upper": 0.95, "confidence": 0.95},
                "n": 3,
            },
            "model/b": {
                "accuracy_ci": {"point": 0.6, "lower": 0.4, "upper": 0.8, "confidence": 0.95},
                "macro_f1_ci": {"point": 0.55, "lower": 0.35, "upper": 0.75, "confidence": 0.95},
                "n": 3,
            },
        },
        "pairwise_mcnemar": [],
        "operational": {
            "note": "test",
            "per_model": {
                "model/a": _operational_entry(p95=900.0, usd_per_1k=0.5),
                "model/b": _operational_entry(p95=120.0, usd_per_1k=0.1),
            },
        },
    }


def test_generate_figures_writes_expected_pngs(tmp_path: Path) -> None:
    from sentiment_benchmark.plotting import generate_figures

    records = [_metric_record("model/a", 0.9), _metric_record("model/b", 0.6)]
    paths = generate_figures(records, _statistics(), tmp_path)

    names = {path.name for path in paths}
    assert "leaderboard.png" in names
    assert "accuracy_ci_forest.png" in names
    assert "confusion_model_a.png" in names
    assert "per_class_f1_model_b.png" in names
    assert "pareto_quality_cost.png" in names
    assert "pareto_quality_latency.png" in names
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_pareto_frontier_keeps_only_undominated_points() -> None:
    from sentiment_benchmark.plotting import pareto_frontier

    # (cost, quality): index 1 dominates index 0 (cheaper, better); index 2 is
    # the cheap/weak end of the frontier; index 3 is expensive but best.
    points = [(2.0, 0.6), (1.0, 0.7), (0.5, 0.5), (4.0, 0.9)]
    assert pareto_frontier(points) == [2, 1, 3]


def test_pareto_frontier_breaks_ties_toward_higher_quality() -> None:
    from sentiment_benchmark.plotting import pareto_frontier

    points = [(1.0, 0.5), (1.0, 0.8)]
    assert pareto_frontier(points) == [1]


def test_pareto_plots_skipped_with_single_model(tmp_path: Path) -> None:
    from sentiment_benchmark.plotting import generate_figures

    statistics = _statistics()
    statistics["operational"]["per_model"].pop("model/b")
    paths = generate_figures([_metric_record("model/a", 0.9)], statistics, tmp_path)
    names = {path.name for path in paths}
    assert "pareto_quality_cost.png" not in names
    assert "pareto_quality_latency.png" not in names


def test_generate_figures_empty_input_returns_empty(tmp_path: Path) -> None:
    from sentiment_benchmark.plotting import generate_figures

    assert generate_figures([], {"per_model": {}}, tmp_path) == []


def _record(prompt_hash: str, row_number: int, model_id: str, label: str) -> LLMResponseRecord:
    return LLMResponseRecord(
        row_number=row_number,
        model_id=model_id,
        prompt_hash=prompt_hash,
        raw_content=label,
        normalized_label=label,
        parse_status="valid",
        status="success",
        latency_ms=10,
    )


def test_export_run_includes_figures(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.sqlite"
    store = BenchmarkStore(db_path)
    store.initialize()
    rows = [
        DatasetRow(1, "a", "positive", False, False, 1),
        DatasetRow(2, "b", "negative", False, False, 1),
        DatasetRow(3, "c", "neutral", False, False, 1),
    ]
    store.upsert_dataset(rows, str(tmp_path / "data.csv"))
    prompt = make_prompt("t", "sys", "{sentence}", "label_only")
    store.save_prompt(prompt)
    config = RunConfig(
        models=["model-a"],
        prompt=prompt,
        mode="pilot",
        dataset_path=str(tmp_path / "data.csv"),
        db_path=str(db_path),
        base_url="x",
        sample_per_class=1,
    )
    run_id = store.create_run(config, [1, 2, 3])
    for row_number, label in {1: "positive", 2: "positive", 3: "neutral"}.items():
        store.save_response(run_id, _record(prompt.prompt_hash, row_number, "model-a", label))
    store.save_metrics(run_id, evaluate_responses(rows, store.fetch_responses(run_id, "model-a"), "model-a", scope="primary"))

    output_dir = tmp_path / "export"
    paths = export_run(db_path, run_id, output_dir=output_dir)

    figures = [path for path in paths if path.suffix == ".png"]
    assert figures, "export should include figures when matplotlib is installed"
    assert (output_dir / "figures").is_dir()
    assert all(path.exists() for path in figures)
