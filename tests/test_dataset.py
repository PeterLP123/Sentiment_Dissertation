from pathlib import Path

from sentiment_benchmark.dataset import compute_stats, load_dataset, select_rows


def test_load_actual_dataset_stats() -> None:
    rows = load_dataset(Path("Data/data.csv"))
    stats = compute_stats(rows)
    assert stats.row_count == 5842
    assert stats.label_counts == {"positive": 1852, "negative": 860, "neutral": 3130}
    assert stats.conflicting_duplicate_rows == 1028
    assert stats.primary_row_count == 4814
    assert stats.primary_label_counts == {"positive": 1852, "negative": 346, "neutral": 2616}


def test_pilot_sampler_returns_30_per_class() -> None:
    rows = load_dataset(Path("Data/data.csv"))
    selected = select_rows(rows, mode="pilot", sample_per_class=30, seed=42)
    assert len(selected) == 90
    assert all(not row.has_conflicting_duplicate for row in selected)
    counts = {}
    for row in selected:
        counts[row.hidden_label] = counts.get(row.hidden_label, 0) + 1
    assert counts == {"positive": 30, "negative": 30, "neutral": 30}

