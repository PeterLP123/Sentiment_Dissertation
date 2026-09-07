from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sentiment_benchmark.baseline_runner import run_baselines
from sentiment_benchmark.baselines import (
    BASELINE_SPECS,
    _soft_sentiment_from_scores,
    iter_finbert_text_batches,
    predict_baseline,
    score_vader_text,
)
from sentiment_benchmark.constants import ALLOWED_LABELS
from sentiment_benchmark.exporter import export_run
from sentiment_benchmark.models import DatasetRow
from sentiment_benchmark.storage import BenchmarkStore

POSITIVE = "strong profit growth gain record revenue beat expectations"
NEGATIVE = "loss decline weak drop risk warning miss expectations"
NEUTRAL = "company report quarterly meeting statement update schedule release"


def _row(row_number: int, sentence: str, label: str) -> DatasetRow:
    return DatasetRow(
        row_number=row_number,
        sentence=sentence,
        hidden_label=label,
        is_duplicate=False,
        has_conflicting_duplicate=False,
        duplicate_group_size=1,
    )


def _synthetic_rows() -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    counts = {"neutral": 12, "positive": 7, "negative": 7}
    bodies = {"positive": POSITIVE, "negative": NEGATIVE, "neutral": NEUTRAL}
    n = 1
    for label, count in counts.items():
        for i in range(count):
            rows.append(_row(n, f"{bodies[label]} item {i}", label))
            n += 1
    return rows


def test_specs_cover_requested_baselines() -> None:
    for name in ("majority", "tfidf_logreg", "vader", "finbert"):
        assert name in BASELINE_SPECS


def test_majority_predicts_dominant_class_out_of_fold() -> None:
    rows = _synthetic_rows()
    predictions = predict_baseline("majority", rows, folds=5, seed=42)
    assert len(predictions) == len(rows)
    # neutral strictly dominates every training fold, so OOF majority is always neutral.
    assert set(predictions) == {"neutral"}


def test_tfidf_logreg_returns_valid_labels() -> None:
    rows = _synthetic_rows()
    predictions = predict_baseline("tfidf_logreg", rows, folds=5, seed=42)
    assert len(predictions) == len(rows)
    assert all(label in ALLOWED_LABELS for label in predictions)


def test_unknown_baseline_raises() -> None:
    with pytest.raises(ValueError):
        predict_baseline("does_not_exist", _synthetic_rows())


def test_finbert_distribution_uses_positive_minus_negative_score() -> None:
    result = _soft_sentiment_from_scores(
        [
            {"label": "negative", "score": 0.1},
            {"label": "neutral", "score": 0.2},
            {"label": "positive", "score": 0.7},
        ]
    )

    assert result.label == "positive"
    assert result.score == pytest.approx(0.6)
    assert result.p_positive + result.p_negative + result.p_neutral == pytest.approx(1.0)


def test_vader_distribution_is_normalized_after_lexicon_rounding(monkeypatch) -> None:
    class RoundedAnalyzer:
        def polarity_scores(self, _text: str) -> dict[str, float]:
            return {"pos": 0.333, "neg": 0.333, "neu": 0.333, "compound": 0.0}

    monkeypatch.setattr("sentiment_benchmark.baselines._vader_analyzer", lambda: RoundedAnalyzer())

    result = score_vader_text("synthetic")

    assert result.p_positive + result.p_negative + result.p_neutral == pytest.approx(1.0)


def test_vader_cache_only_scoring_disables_lexicon_download(monkeypatch) -> None:
    calls: list[bool] = []

    class Analyzer:
        def polarity_scores(self, _text: str) -> dict[str, float]:
            return {"pos": 0.7, "neg": 0.1, "neu": 0.2, "compound": 0.6}

    def fake_analyzer(allow_download: bool = True) -> Analyzer:
        calls.append(allow_download)
        return Analyzer()

    monkeypatch.setattr("sentiment_benchmark.baselines._vader_analyzer", fake_analyzer)

    score_vader_text("synthetic", local_files_only=True)

    assert calls == [False]


def test_finbert_cache_only_scoring_is_forwarded_to_transformers(monkeypatch) -> None:
    import sys
    import types

    pipeline_options: dict[str, object] = {}

    def fake_pipeline(_task: str, **kwargs):
        pipeline_options.update(kwargs)

        def classify(texts, **_inference_options):
            return [
                [
                    {"label": "positive", "score": 0.7},
                    {"label": "negative", "score": 0.1},
                    {"label": "neutral", "score": 0.2},
                ]
                for _text in texts
            ]

        return classify

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline  # type: ignore[attr-defined]
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    fake_torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    results = list(
        iter_finbert_text_batches(
            ["synthetic"],
            revision="4556d13015211d73dccd3fdd39d39232506f3e43",
            local_files_only=True,
        )
    )

    assert pipeline_options["local_files_only"] is True
    assert results[0][0].label == "positive"


def _write_csv(path: Path, rows: list[DatasetRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Sentence", "Sentiment"])
        for row in rows:
            writer.writerow([row.sentence, row.hidden_label])


def test_run_baselines_persists_and_exports(tmp_path: Path) -> None:
    rows = _synthetic_rows()
    dataset_path = tmp_path / "data.csv"
    db_path = tmp_path / "bench.sqlite"
    _write_csv(dataset_path, rows)

    summary = run_baselines(
        ["majority", "tfidf_logreg"],
        mode="full",
        dataset_path=str(dataset_path),
        db_path=str(db_path),
        folds=5,
        seed=42,
    )
    assert summary.baseline_count == 2
    assert summary.selected_row_count == len(rows)

    store = BenchmarkStore(db_path)
    saved = store.fetch_responses(summary.run_id, "baseline/majority")
    assert len(saved) == len(rows)
    assert all(dict(response)["status"] == "success" for response in saved)

    output_dir = tmp_path / "export"
    paths = export_run(db_path, summary.run_id, output_dir=output_dir)
    assert (output_dir / "metrics.json").exists()
    assert any(path.name == "summary.md" for path in paths)


def test_vader_baseline_if_available() -> None:
    pytest.importorskip("nltk")
    rows = [
        _row(1, "Profits surged and the outlook is excellent.", "positive"),
        _row(2, "Losses mounted as the company collapsed.", "negative"),
    ]
    try:
        predictions = [score_vader_text(row.sentence, local_files_only=True).label for row in rows]
    except (LookupError, RuntimeError) as exc:  # pragma: no cover - optional local lexicon unavailable
        pytest.skip(f"cached VADER lexicon unavailable: {exc}")
    assert len(predictions) == len(rows)
    assert all(label in ALLOWED_LABELS for label in predictions)


def test_disable_hf_progress_bars_avoids_multiprocessing_lock() -> None:
    """Regression: FinBERT crashed in the TUI worker thread because tqdm built a
    multiprocessing lock (fork_exec) from a non-main thread. The helper must run
    cleanly and leave tqdm on a non-multiprocessing lock."""
    import os
    import threading

    from sentiment_benchmark.baselines import _disable_hf_progress_bars

    result: dict[str, object] = {}

    def worker() -> None:
        try:
            _disable_hf_progress_bars()
            result["ok"] = True
        except Exception as exc:  # pragma: no cover - the bug would surface here
            result["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert result.get("error") is None
    assert result.get("ok") is True
    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"

    tqdm = pytest.importorskip("tqdm").tqdm
    # The lock must not be tqdm's multiprocessing default lock.
    assert "multiprocessing" not in type(tqdm.get_lock()).__module__
