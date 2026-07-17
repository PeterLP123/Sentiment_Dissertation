from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_benchmark.sentiment_term_structure_pilot import (
    add_horizon_outcomes,
    audit_corpus_depth,
    classify_term_structure_decision,
    decay_curve_values,
    fit_decay_curve,
)


def _prices(symbol: str, dates: pd.DatetimeIndex, returns: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "session_date": dates,
            "close": 100 * np.exp(np.cumsum(returns)),
        }
    )


def test_horizon_outcomes_are_single_day_not_cumulative() -> None:
    dates = pd.bdate_range("2026-01-02", periods=5)
    market = _prices("^GSPC", dates, np.array([0.0, 0.01, 0.01, 0.01, 0.01]))
    stock = _prices("AAA", dates, np.array([0.0, 0.02, 0.03, 0.04, 0.05]))
    events = pd.DataFrame([{"symbol": "AAA", "session_date": dates[1], "level": 0.5}])
    result, _ = add_horizon_outcomes(
        events,
        stock,
        market,
        market_symbol="^GSPC",
        horizons=(0, 1, 2),
    )
    assert np.isclose(result.iloc[0]["ar_h0"], 0.01)
    assert np.isclose(result.iloc[0]["ar_h1"], 0.02)
    assert np.isclose(result.iloc[0]["ar_h2"], 0.03)


def test_decay_fits_recover_noise_free_curves() -> None:
    horizons = np.arange(11, dtype=float)
    se = np.full(11, 0.01)
    single_truth = {"lambda": -0.04, "rho": 0.35}
    single_curve = single_truth["lambda"] * np.exp(-single_truth["rho"] * horizons)
    single = fit_decay_curve(single_curve, se, model="single")
    assert np.isclose(single["lambda"], single_truth["lambda"], atol=1e-7)
    assert np.isclose(single["rho"], single_truth["rho"], atol=1e-6)
    assert np.isclose(single["half_life"], math.log(2) / single_truth["rho"], atol=1e-5)

    two_truth = {
        "lambda_fast": -0.05,
        "rho_fast": 1.2,
        "lambda_slow": 0.02,
        "rho_slow": 0.12,
    }
    two_curve = decay_curve_values("two_component", two_truth, horizons)
    two = fit_decay_curve(two_curve, se, model="two_component")
    assert two["rho_fast"] >= two["rho_slow"]
    assert np.allclose(decay_curve_values("two_component", two, horizons), two_curve, atol=1e-6)


def test_corpus_audit_flags_zero_to_nonzero_regime_break(tmp_path: Path) -> None:
    path = tmp_path / "headlines.jsonl"
    rows = [
        {
            "headline": "a",
            "matched_symbols": ["AAA"],
            "version_created": "2026-01-10T12:00:00Z",
        },
        {
            "headline": "b",
            "matched_symbols": ["AAA"],
            "version_created": "2026-03-10T12:00:00Z",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monthly, flags, summary = audit_corpus_depth(
        path,
        cohort="test",
        through_month="2026-03",
        break_ratio=3.0,
    )
    assert monthly["article_associations"].tolist() == [1, 0, 1]
    assert len(flags) == 2
    assert np.isinf(flags["max_fold_change"]).all()
    assert summary["symbols"] == 1


def test_precision_failure_forces_no_go() -> None:
    impulse = pd.DataFrame({"t_stat": [3.0, 0.0]})
    decision, reasons = classify_term_structure_decision(
        history_months=24,
        precision_pass=False,
        development_ir=impulse,
        history_gate_months=18,
    )
    assert decision == "NO-GO"
    assert "precision_gate_fails" in reasons
