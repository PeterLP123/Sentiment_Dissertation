from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from final_experiments.lib import lseg_expanded
from final_experiments.lib.lseg_expanded import (
    _charge_final_liquidation,
    assert_paired_population,
    build_expanded_aggregation_panel,
    load_success_scores,
    map_headlines_to_entry_sessions,
)
from sentiment_benchmark.artifact_io import sha256_file


def _write_score_artifact(path: Path, manifest: dict[str, object]) -> None:
    pd.DataFrame(
        {
            "headline_sha256": ["a"],
            "matched_symbols": ["AAA"],
            "first_timestamp": ["2026-01-05T14:00:00Z"],
            "explicit_target": [True],
            "contextual": [False],
            "market_price_technical": [False],
            "label": ["positive"],
            "p_positive": [0.8],
            "p_negative": [0.1],
            "p_neutral": [0.1],
            "score": [0.7],
            "status": ["success"],
            "reported_cost_usd": [0.001],
        }
    ).to_csv(path, index=False)
    manifest["status"] = "completed"
    manifest["output"] = {"sha256": sha256_file(path)}
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_precise_timestamp_mapping_respects_information_buffer() -> None:
    scores = pd.DataFrame(
        {
            "headline_sha256": ["a", "b", "c"],
            "matched_symbols": ["AAA", "AAA", "AAA|BBB"],
            "first_timestamp": pd.to_datetime(
                [
                    "2026-01-05T14:00:00Z",  # 09:00 ET -> same open after 15m buffer
                    "2026-01-05T14:20:00Z",  # 09:20 ET -> next open after buffer
                    "2026-01-05T22:00:00Z",  # after close -> next open for both symbols
                ],
                utc=True,
            ),
            "explicit_target": [True, True, False],
            "contextual": [False, False, True],
            "market_price_technical": [False, False, False],
            "label": ["positive", "negative", "neutral"],
            "score": [0.8, -0.8, 0.0],
            "scorer": ["gemma"] * 3,
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "session_date": pd.to_datetime(["2026-01-05", "2026-01-06"] * 2),
            "adjusted_open": [100.0, 101.0, 50.0, 51.0],
        }
    )

    mapped = map_headlines_to_entry_sessions(scores, prices)

    got = mapped.groupby("headline_sha256")["entry_session"].first().dt.strftime("%Y-%m-%d").to_dict()
    assert got == {"a": "2026-01-05", "b": "2026-01-06", "c": "2026-01-06"}
    assert len(mapped.loc[mapped["headline_sha256"].eq("c")]) == 2


def test_final_liquidation_charges_active_book() -> None:
    daily = pd.DataFrame(
        {
            "n_long": [2],
            "n_short": [2],
            "turnover": [0.5],
            "cost": [0.001],
            "net_return": [0.009],
        }
    )

    closed = _charge_final_liquidation(daily, cost_bps_per_side=10.0)

    assert closed.loc[0, "turnover"] == 1.0
    assert closed.loc[0, "cost"] == 0.002
    assert closed.loc[0, "net_return"] == 0.008


def test_paired_population_uses_current_metadata_and_hash_paired_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lseg_expanded, "EXPECTED_POPULATION", 2)
    shared = {
        "headline_sha256": ["a", "b"],
        "explicit_target": [True, False],
        "contextual": [False, True],
        "market_price_technical": [False, False],
    }
    finbert = pd.DataFrame(
        {
            **shared,
            "matched_symbols": ["OLD", "BBB"],
            "first_timestamp": pd.to_datetime(
                ["2026-01-01T12:00:00Z", "2026-01-02T12:00:00Z"], utc=True
            ),
            "label": ["negative", "positive"],
            "score": [-0.7, 0.8],
        }
    )
    gemma = pd.DataFrame(
        {
            **shared,
            "matched_symbols": ["AAA", "BBB"],
            "first_timestamp": pd.to_datetime(
                ["2026-01-03T12:00:00Z", "2026-01-02T12:00:00Z"], utc=True
            ),
            "label": ["neutral", "negative"],
            "score": [0.0, -0.6],
        }
    )

    paired = assert_paired_population(finbert, gemma)

    assert paired["matched_symbols"].tolist() == ["AAA", "BBB"]
    assert paired["score_finbert"].tolist() == [-0.7, 0.8]
    assert paired["score_gemma"].tolist() == [0.0, -0.6]
    assert paired.attrs["finbert_seed_metadata_mismatch_counts"]["matched_symbols"] == 1
    assert paired.attrs["finbert_seed_metadata_mismatch_counts"]["first_timestamp"] == 1

    with pytest.raises(ValueError, match="populations do not match exactly"):
        assert_paired_population(finbert, gemma.assign(headline_sha256=["a", "c"]))


def test_expanded_panel_uses_immediate_next_session_and_leaves_terminal_return_missing() -> None:
    sessions = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    events = pd.DataFrame(
        {
            "headline_sha256": ["a", "b", "c"],
            "symbol": ["AAA", "AAA", "AAA"],
            "entry_session": sessions,
            "session_date": sessions,
            "finbert_score": [0.8, -0.5, 0.2],
            "polarity": ["positive", "negative", "positive"],
            "is_recap": [False, False, False],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "session_date": sessions,
            "adjusted_open": [100.0, 110.0, 121.0],
        }
    )

    panel = build_expanded_aggregation_panel(events, prices).set_index("session_date")

    assert panel.loc[sessions[0], "raw_open_h1"] == pytest.approx(0.1)
    assert panel.loc[sessions[1], "raw_open_h1"] == pytest.approx(0.1)
    assert pd.isna(panel.loc[sessions[2], "raw_open_h1"])


@pytest.mark.parametrize(
    ("scorer", "manifest", "drift_path", "drift_value"),
    [
        (
            "finbert",
            {
                "contract": {
                    "scorer": "finbert",
                    "model_id": "ProsusAI/finbert",
                    "revision": lseg_expanded.FINBERT_REVISION,
                    "revision_enforced": True,
                }
            },
            ("contract", "revision"),
            "different-revision",
        ),
        (
            "gemma4_26b",
            {
                "model": lseg_expanded.GEMMA_MODEL,
                "provider": "DeepInfra",
                "quantization": "fp8",
                "prompt_id": lseg_expanded.GEMMA_PROMPT_ID,
                "prompt_hash": lseg_expanded.GEMMA_PROMPT_HASH,
                "request": {
                    "provider": {
                        "only": ["deepinfra"],
                        "quantizations": ["fp8"],
                        "allow_fallbacks": False,
                        "data_collection": "deny",
                        "require_parameters": True,
                        "zdr": True,
                    }
                },
            },
            ("request", "provider", "zdr"),
            False,
        ),
    ],
)
def test_score_loader_rejects_pinned_manifest_contract_drift(
    tmp_path: Path,
    scorer: str,
    manifest: dict[str, object],
    drift_path: tuple[str, ...],
    drift_value: object,
) -> None:
    path = tmp_path / "scores.csv"
    _write_score_artifact(path, manifest)
    scores, _ = load_success_scores(path, scorer=scorer, expected_population=1)
    assert len(scores) == 1

    target: object = manifest
    for part in drift_path[:-1]:
        assert isinstance(target, dict)
        target = target[part]
    assert isinstance(target, dict)
    target[drift_path[-1]] = drift_value
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="score manifest contract mismatch"):
        load_success_scores(path, scorer=scorer, expected_population=1)
