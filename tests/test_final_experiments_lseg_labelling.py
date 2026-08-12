import csv
import json
from pathlib import Path

import pytest

from final_experiments.lib import lseg_labelling
from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.headline_return_study import BASELINE_COLUMNS
from sentiment_benchmark.headline_value import ScorableHeadline


def _score_row(headline_hash: str, scorer: str) -> dict[str, str]:
    return {
        "headline_sha256": headline_hash,
        "headline": f"Headline {headline_hash}",
        "matched_symbols": "AAA",
        "first_timestamp": "2026-01-01T00:00:00+00:00",
        "explicit_target": "True",
        "contextual": "False",
        "market_price_technical": "False",
        "baseline": scorer,
        "label": "neutral",
        "p_positive": "0.1",
        "p_negative": "0.1",
        "p_neutral": "0.8",
        "score": "0.0",
        "score_100": "0.0",
        "status": "success",
        "error": "",
    }


def _write_prior(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "completed",
        "models": {
            "finbert": {
                "model_id": lseg_labelling.FINBERT_MODEL_ID,
                "revision": lseg_labelling.FINBERT_REVISION,
                "revision_enforced": True,
            }
        },
        "output": {"sha256": sha256_file(path)},
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _record(headline_hash: str) -> ScorableHeadline:
    return ScorableHeadline(headline_hash, "Headline", ("AAA",), "2026-01-01", True, False, False)


def test_prepare_reuses_only_complete_target_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prior = tmp_path / "prior.csv"
    output = tmp_path / "seed.csv"
    _write_prior(
        prior,
        [
            _score_row("keep", "vader"),
            _score_row("drop", "vader"),
            _score_row("keep", "finbert"),
            _score_row("drop", "finbert"),
        ],
    )
    monkeypatch.setattr(lseg_labelling, "collect_scorable_headline_records", lambda _: [_record("keep"), _record("new")])

    summary = lseg_labelling.prepare_reusable_baseline_seed(tmp_path, prior, output)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary.target_unique_headlines == 2
    assert summary.reusable_unique_headlines == 1
    assert summary.missing_unique_headlines == 1
    assert {(row["headline_sha256"], row["baseline"]) for row in rows} == {
        ("keep", "finbert"),
        ("keep", "vader"),
    }
    assert {row["headline"] for row in rows} == {"Headline"}
    seed_manifest = json.loads(output.with_suffix(output.suffix + ".seed.json").read_text(encoding="utf-8"))
    assert seed_manifest["metadata_reconciliation"] == {
        "changed_headlines": 1,
        "changed_headlines_by_field": {"first_timestamp": 1, "headline": 1},
    }
    assert lseg_labelling.prepare_reusable_baseline_seed(tmp_path, prior, output) == summary


def test_prepare_rejects_partial_reusable_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prior = tmp_path / "prior.csv"
    _write_prior(prior, [_score_row("keep", "vader")])
    monkeypatch.setattr(lseg_labelling, "collect_scorable_headline_records", lambda _: [_record("keep")])

    with pytest.raises(ValueError, match="incomplete for 1 reusable headlines"):
        lseg_labelling.prepare_reusable_baseline_seed(tmp_path, prior, tmp_path / "seed.csv")


def test_export_finbert_only_scores_excludes_vader(tmp_path: Path) -> None:
    combined = tmp_path / "combined.csv"
    output = tmp_path / "finbert.csv"
    rows = [
        _score_row("one", "vader"),
        _score_row("one", "finbert"),
        _score_row("two", "vader"),
        _score_row("two", "finbert"),
    ]
    with combined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "completed",
        "input": {"unique_headlines": 2},
        "counts": {"finbert_successes": 2, "vader_successes": 2},
        "models": {
            "finbert": {
                "model_id": lseg_labelling.FINBERT_MODEL_ID,
                "revision": lseg_labelling.FINBERT_REVISION,
                "revision_enforced": True,
            }
        },
        "output": {"sha256": sha256_file(combined)},
    }
    combined.with_suffix(combined.suffix + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = lseg_labelling.export_finbert_only_scores(combined, output)

    with output.open(encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert summary.unique_headlines == 2
    assert [row["baseline"] for row in exported] == ["finbert", "finbert"]
    export_manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert export_manifest["summary"] == {"excluded_vader_rows": 2, "unique_headlines": 2}
    assert export_manifest["output"]["sha256"] == sha256_file(output)
