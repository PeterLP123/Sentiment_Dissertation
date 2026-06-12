from pathlib import Path

from sentiment_benchmark.constants import DEFAULT_DATASET_PATH
from sentiment_benchmark.dataset import compute_stats, load_dataset, select_rows


def test_load_default_dataset_stats() -> None:
    rows = load_dataset(DEFAULT_DATASET_PATH)
    stats = compute_stats(rows)
    assert stats.row_count == 5947
    assert stats.label_counts == {"neutral": 2884, "positive": 2082, "negative": 981}
    assert stats.duplicate_sentence_groups == 0
    assert stats.duplicate_extra_rows == 0
    assert stats.conflicting_duplicate_rows == 0
    assert stats.primary_row_count == 5947
    assert stats.primary_label_counts == {"neutral": 2884, "positive": 2082, "negative": 981}


def test_load_legacy_kaggle_dataset_stats() -> None:
    rows = load_dataset(Path("Data/data.csv"))
    stats = compute_stats(rows)
    assert stats.row_count == 5842
    assert stats.label_counts == {"positive": 1852, "negative": 860, "neutral": 3130}
    assert stats.conflicting_duplicate_rows == 1028
    assert stats.primary_row_count == 4814
    assert stats.primary_label_counts == {"positive": 1852, "negative": 346, "neutral": 2616}


def test_pilot_sampler_returns_30_per_class() -> None:
    rows = load_dataset(DEFAULT_DATASET_PATH)
    selected = select_rows(rows, mode="pilot", sample_per_class=30, seed=42)
    assert len(selected) == 90
    assert all(not row.has_conflicting_duplicate for row in selected)
    counts = {}
    for row in selected:
        counts[row.hidden_label] = counts.get(row.hidden_label, 0) + 1
    assert counts == {"positive": 30, "negative": 30, "neutral": 30}
