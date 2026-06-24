import csv
import json
from pathlib import Path

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.lseg_validation import (
    chronological_date_split,
    create_lseg_validation_sample,
    evaluate_lseg_annotations,
)


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [
        {
            "article_id": f"a{index}",
            "revision_id": f"r{index}",
            "story_id": f"story:{index}",
            "matched_symbols": ["AAA" if index % 2 else "BBB"],
            "version_created": f"2026-06-0{1 + index % 3}T10:00:00+00:00",
            "headline": f"Headline {index}",
            "clean_text": f"Story body {index}",
            "scoring_eligible": True,
        }
        for index in range(8)
    ]
    articles = corpus / "articles.jsonl"
    articles.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "files": {"articles_jsonl": {"path": "articles.jsonl", "sha256": sha256_file(articles)}},
    }
    path = corpus / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _rewrite_labels(path: Path, field: str, labels: list[str], adjudicated: list[str] | None = None) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for index, row in enumerate(rows):
        row[field] = labels[index]
        if adjudicated is not None:
            row["adjudicated_label"] = adjudicated[index]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_validation_sampling_and_annotation_agreement_are_reproducible(tmp_path: Path) -> None:
    result = create_lseg_validation_sample(
        _corpus(tmp_path),
        tmp_path / "sample",
        sample_size=6,
        double_code_size=2,
        seed=42,
        double_code_seed=43,
    )
    assert result.sample_count == 6
    assert len(list(csv.DictReader(result.secondary_path.open()))) == 2

    with result.primary_path.open(encoding="utf-8", newline="") as handle:
        primary_rows = list(csv.DictReader(handle))
    with result.secondary_path.open(encoding="utf-8", newline="") as handle:
        secondary_rows = list(csv.DictReader(handle))
    secondary_ids = {row["revision_id"] for row in secondary_rows}
    primary_labels = ["positive" if index % 2 else "neutral" for index in range(len(primary_rows))]
    adjudicated = [
        "negative" if row["revision_id"] in secondary_ids else ""
        for row in primary_rows
    ]
    _rewrite_labels(result.primary_path, "primary_label", primary_labels, adjudicated)
    _rewrite_labels(result.secondary_path, "secondary_label", ["negative", "negative"])

    evaluation = evaluate_lseg_annotations(result.primary_path, result.secondary_path, tmp_path / "evaluation")

    assert evaluation.double_code_count == 2
    assert 0 <= evaluation.percent_agreement <= 1
    assert evaluation.labeled_dataset_path.exists()


def test_chronological_split_keeps_dates_whole() -> None:
    development, holdout = chronological_date_split(["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-04"])

    assert development == ("2026-01-01", "2026-01-02")
    assert holdout == ("2026-01-03", "2026-01-04")
