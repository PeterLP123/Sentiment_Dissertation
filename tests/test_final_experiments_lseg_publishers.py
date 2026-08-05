from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from final_experiments.lib.lseg_publishers import (
    aggregate_publisher_events,
    load_headline_source_features,
)
from final_experiments.lib.novelty import headline_norm_sha256
from sentiment_benchmark.artifact_io import sha256_file


def _write_corpus(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    path.with_name("manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "files": {"headlines_jsonl": {"sha256": sha256_file(path)}},
            }
        ),
        encoding="utf-8",
    )


def test_source_features_use_source_specific_earliest_timestamp_and_drop_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "headlines.jsonl"
    headline = "AAA reports higher profit"
    digest = headline_norm_sha256(headline)
    _write_corpus(
        path,
        [
            {
                "headline": headline,
                "source_code": "NS:OTHER",
                "version_created": "2026-01-05T13:00:00Z",
            },
            {
                "headline": headline,
                "source_code": "NS:RTRS",
                "version_created": "2026-01-05T14:00:00Z",
            },
            {
                "headline": headline,
                "source_code": "NS:RTRS",
                "version_created": "2026-01-05T13:30:00Z",
            },
        ],
    )

    features, audit = load_headline_source_features(
        path, expected_hashes={digest}
    )

    row = features.iloc[0]
    assert bool(row["has_reuters"])
    assert bool(row["has_non_reuters"])
    assert row["first_reuters_timestamp"] == pd.Timestamp(
        "2026-01-05T13:30:00Z"
    )
    assert row["first_non_reuters_timestamp"] == pd.Timestamp(
        "2026-01-05T13:00:00Z"
    )
    assert row["source_corpus_rows"] == 3
    assert "headline" not in features
    assert audit["distinct_source_codes"] == 2
    assert audit["hashes_with_both"] == 1


def test_source_loader_fails_closed_when_score_hash_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "headlines.jsonl"
    _write_corpus(
        path,
        [
            {
                "headline": "AAA reports higher profit",
                "source_code": "NS:RTRS",
                "version_created": "2026-01-05T14:00:00Z",
            }
        ],
    )

    with pytest.raises(ValueError, match="misses 1 frozen successful hashes"):
        load_headline_source_features(path, expected_hashes={"missing"})


def test_reuters_two_x_weighting_is_hash_level_and_deterministic() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "entry_session": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "headline_sha256": ["a", "b"],
            "score": [1.0, -1.0],
            "has_reuters": [True, False],
        }
    )

    unweighted = aggregate_publisher_events(events, weight_reuters=False)
    weighted = aggregate_publisher_events(events, weight_reuters=True)

    assert unweighted["publisher_signal"].iloc[0] == pytest.approx(0.0)
    assert weighted["publisher_signal"].iloc[0] == pytest.approx(1.0 / 3.0)
    assert weighted["unique_headlines"].iloc[0] == 2
    assert weighted["reuters_headlines"].iloc[0] == 1
