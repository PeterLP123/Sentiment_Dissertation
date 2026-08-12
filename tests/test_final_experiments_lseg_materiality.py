from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from final_experiments.lib.lseg_materiality import (
    MaterialityPilotError,
    parse_materiality_labels,
    score_materiality_sample,
    select_stratified_sample,
)


def _candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    symbols = [f"S{index:02d}" for index in range(33)]
    blocks = ("backward_2024_2025", "opened_2025_2026")
    labels = ("negative", "neutral", "positive")
    session = pd.Timestamp("2024-01-02")
    for symbol in symbols:
        offset = 0
        for block in blocks:
            for label in labels:
                for repeat in range(12):
                    rows.append(
                        {
                            "headline_sha256": f"{symbol}-{block}-{label}-{repeat}",
                            "symbol": symbol,
                            "entry_session": session + pd.Timedelta(offset, unit="D"),
                            "source_block": block,
                            "label_finbert": label,
                        }
                    )
                    offset += 1
    return pd.DataFrame(rows)


def test_materiality_sampling_is_exact_balanced_and_deterministic() -> None:
    candidates = _candidate_rows()
    first = select_stratified_sample(candidates, size=2_000)
    second = select_stratified_sample(candidates, size=2_000)

    assert first["headline_sha256"].tolist() == second["headline_sha256"].tolist()
    assert len(first) == 2_000
    assert not first["headline_sha256"].duplicated().any()
    assert not first.duplicated(["symbol", "entry_session"]).any()
    counts = first["symbol"].value_counts()
    assert set(counts.index) == set(candidates["symbol"])
    assert counts.min() == 60
    assert counts.max() == 61


def test_materiality_parser_enforces_cross_field_contract() -> None:
    valid = parse_materiality_labels(
        {
            "direction_severity": "neutral",
            "materiality": "none",
            "novelty": "high",
            "valuation_horizon": "unclear",
            "target_specific": False,
            "evidence_sufficient": True,
        }
    )
    assert valid["target_specific"] is False

    with pytest.raises(ValueError, match="target_specific=false"):
        parse_materiality_labels(
            {
                **valid,
                "direction_severity": "positive",
                "materiality": "high",
            }
        )
    with pytest.raises(ValueError, match="extreme direction"):
        parse_materiality_labels(
            {
                **valid,
                "target_specific": True,
                "direction_severity": "very_positive",
                "materiality": "moderate",
            }
        )


def test_hosted_scoring_requires_exact_transfer_confirmation(tmp_path) -> None:
    with pytest.raises(MaterialityPilotError, match="confirm-authorized-transfer"):
        asyncio.run(score_materiality_sample(output_root=tmp_path))
