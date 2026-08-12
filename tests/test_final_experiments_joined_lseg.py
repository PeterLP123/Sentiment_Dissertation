from __future__ import annotations

import pandas as pd
import pytest

from final_experiments.lib.joined_lseg import (
    _deduplicate_join,
    _pair_scores,
    _restrict_associations,
    _validate_block_dates,
)


def _score_row(
    headline_hash: str,
    *,
    symbols: str,
    timestamp: str,
    score_gemma: float,
    score_finbert: float,
    source_block: str,
    explicit_target: bool = False,
    contextual: bool = True,
    market_price_technical: bool = False,
) -> dict[str, object]:
    return {
        "headline_sha256": headline_hash,
        "matched_symbols": symbols,
        "first_timestamp": pd.Timestamp(timestamp),
        "explicit_target": explicit_target,
        "contextual": contextual,
        "market_price_technical": market_price_technical,
        "label_gemma": "positive" if score_gemma > 0 else "negative",
        "score_gemma": score_gemma,
        "label_finbert": "positive" if score_finbert > 0 else "negative",
        "score_finbert": score_finbert,
        "source_block": source_block,
    }


def test_cross_block_join_keeps_earliest_scores_and_unions_associations() -> None:
    backward = pd.DataFrame(
        [
            _score_row(
                "shared",
                symbols="AAA",
                timestamp="2024-01-02T08:00:00Z",
                score_gemma=0.2,
                score_finbert=0.3,
                source_block="backward_2024_2025",
            )
        ]
    )
    recent = pd.DataFrame(
        [
            _score_row(
                "shared",
                symbols="BBB",
                timestamp="2025-11-01T08:00:00Z",
                score_gemma=-0.8,
                score_finbert=-0.7,
                source_block="opened_2025_2026",
                explicit_target=True,
                contextual=False,
                market_price_technical=True,
            ),
            _score_row(
                "recent-only",
                symbols="CCC",
                timestamp="2025-11-02T08:00:00Z",
                score_gemma=0.1,
                score_finbert=0.2,
                source_block="opened_2025_2026",
            ),
        ]
    )

    joined, audit = _deduplicate_join((backward, recent))

    shared = joined.set_index("headline_sha256").loc["shared"]
    assert shared["matched_symbols"] == "AAA|BBB"
    assert shared["first_timestamp"] == pd.Timestamp("2024-01-02T08:00:00Z")
    assert shared["score_gemma"] == pytest.approx(0.2)
    assert shared["score_finbert"] == pytest.approx(0.3)
    assert bool(shared["explicit_target"])
    assert not bool(shared["contextual"])
    assert bool(shared["market_price_technical"])
    assert audit["input_rows"] == 3
    assert audit["joined_unique_headlines"] == 2
    assert audit["cross_block_overlap_hashes"] == 1
    assert audit["cross_block_score_collision_hashes"] == {
        "gemma": 1,
        "finbert": 1,
    }


def test_association_restriction_removes_additional_only_headlines() -> None:
    frame = pd.DataFrame(
        {
            "headline_sha256": ["mixed", "added-only"],
            "matched_symbols": ["ZZZ|AAA|BBB", "ZZZ"],
        }
    )

    restricted, removed = _restrict_associations(frame, {"AAA", "BBB"})

    assert removed == 1
    assert restricted[["headline_sha256", "matched_symbols"]].to_dict("records") == [
        {"headline_sha256": "mixed", "matched_symbols": "AAA|BBB"}
    ]


def test_pair_scores_uses_gemma_metadata_and_audits_mismatch() -> None:
    metadata = {
        "headline_sha256": ["one", "two"],
        "matched_symbols": ["AAA", "BBB"],
        "first_timestamp": pd.to_datetime(
            ["2024-01-02T08:00:00Z", "2024-01-03T08:00:00Z"], utc=True
        ),
        "explicit_target": [True, False],
        "contextual": [False, True],
        "market_price_technical": [False, False],
    }
    gemma = pd.DataFrame(metadata).assign(
        label=["positive", "negative"], score=[0.7, -0.4]
    )
    finbert = pd.DataFrame(metadata).assign(
        label=["positive", "neutral"], score=[0.6, 0.0]
    )
    finbert.loc[1, "matched_symbols"] = "STALE"

    paired, audit = _pair_scores(
        finbert,
        gemma,
        expected_population=2,
        block="test",
    )

    assert paired.loc[paired["headline_sha256"].eq("two"), "matched_symbols"].item() == "BBB"
    assert audit["metadata_mismatch_counts"]["matched_symbols"] == 1
    assert set(paired) >= {"score_gemma", "score_finbert", "source_block"}


def test_block_date_validation_treats_end_as_exclusive() -> None:
    frame = pd.DataFrame(
        {"first_timestamp": pd.to_datetime(["2025-10-26T00:00:00Z"], utc=True)}
    )

    with pytest.raises(ValueError, match="outside its frozen interval"):
        _validate_block_dates(
            frame,
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2025-10-26", tz="UTC"),
            block="backward",
        )
