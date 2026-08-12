from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_experiments.lib.lseg_materiality_returns import (
    attach_forward_open_returns,
    benjamini_hochberg_qvalues,
    load_model_only_development_events,
    load_preconfirmation_lseg_prices,
)
from sentiment_benchmark.artifact_io import sha256_file


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_development_loader_filters_confirmation_before_return_join(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.jsonl"
    score_path = tmp_path / "scores.jsonl"
    sample_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for index in range(34):
        audit_id = f"row-{index:02d}"
        headline_hash = f"{index:064x}"
        sample_rows.append(
            {
                "audit_id": audit_id,
                "headline_sha256": headline_hash,
                "symbol": f"S{index:02d}" if index < 33 else "S00",
                "entry_session": "2025-12-30" if index < 33 else "2026-01-02",
                "score_gemma": -0.8 + index / 50,
                "score_finbert": 0.8 - index / 50,
            }
        )
        score_rows.append(
            {
                "audit_id": audit_id,
                "headline_sha256": headline_hash,
                "direction_severity": "positive" if index % 2 else "negative",
                "materiality": "moderate",
                "novelty": "high",
                "valuation_horizon": "short_2_5_sessions",
                "target_specific": True,
                "evidence_sufficient": True,
            }
        )
    _write_jsonl(sample_path, sample_rows)
    _write_jsonl(score_path, score_rows)

    result = load_model_only_development_events(
        sample_path,
        score_path,
        sample_sha256=sha256_file(sample_path),
        score_sha256=sha256_file(score_path),
        expected_development_rows=33,
    )

    assert len(result) == 33
    assert result["symbol"].nunique() == 33
    assert result["entry_session"].max() < pd.Timestamp("2026-01-01")
    assert result["measurement_status"].eq("model_only_unvalidated").all()


def test_price_loader_never_parses_numeric_confirmation_value(tmp_path: Path) -> None:
    price_path = tmp_path / "prices.csv"
    price_path.write_text(
        "symbol,session_date,open\n"
        "AAA,2025-12-30,100\n"
        "AAA,2025-12-31,101\n"
        "AAA,2026-01-02,SEALED_NOT_NUMERIC\n",
        encoding="utf-8",
    )
    digest = sha256_file(price_path)
    price_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "provider": "lseg",
                "return_convention": "split-adjusted price returns; dividends not back-adjusted",
                "file": {"sha256": digest},
            }
        ),
        encoding="utf-8",
    )

    prices, audit = load_preconfirmation_lseg_prices(
        price_path,
        expected_sha256=digest,
        expected_symbols=1,
    )

    assert len(prices) == 2
    assert prices["session_date"].max() == pd.Timestamp("2025-12-31")
    assert audit["numeric_confirmation_prices_parsed"] == 0
    assert audit["confirmation_returns_materialised"] == 0


def test_forward_return_attachment_cannot_cross_confirmation() -> None:
    prices = pd.DataFrame(
        [
            {
                "symbol": f"S{symbol:02d}",
                "session_date": pd.Timestamp(session),
                "adjusted_open": 100.0 + day,
            }
            for day, session in enumerate(("2025-12-29", "2025-12-30", "2025-12-31"))
            for symbol in range(33)
        ]
    )
    events = pd.DataFrame(
        {
            "audit_id": ["a", "b"],
            "symbol": ["S00", "S00"],
            "entry_session": pd.to_datetime(["2025-12-29", "2025-12-31"]),
        }
    )

    joined, _ = attach_forward_open_returns(events, prices, horizons=(1,))

    assert np.isclose(joined.loc[joined["audit_id"].eq("a"), "ret_open_h1"].iloc[0], 0.01)
    assert joined.loc[joined["audit_id"].eq("b"), "ret_open_h1"].isna().all()
    assert joined["return_end_h1"].dropna().max() < pd.Timestamp("2026-01-01")


def test_benjamini_hochberg_qvalues_are_monotone_by_rank() -> None:
    q = benjamini_hochberg_qvalues([0.01, 0.04, 0.03])
    np.testing.assert_allclose(q, [0.03, 0.04, 0.04])
