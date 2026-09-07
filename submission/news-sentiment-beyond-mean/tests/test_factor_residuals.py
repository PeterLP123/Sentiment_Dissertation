"""Tests for trailing FF3 residuals: no lookahead and French-file parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.lib.factor_residuals import load_french_daily_ff3, trailing_ff3_residuals


def test_load_french_vendor_percent_file(tmp_path: Path) -> None:
    path = tmp_path / "ff_daily.CSV"
    path.write_text(
        "This file was created by Ken French.\n"
        "\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "20100104,    1.00,    0.50,   -0.20,   0.010\n"
        "20100105,   -0.50,    0.00,    0.10,   0.010\n"
        "\n"
        " Annual Factors: January-December\n"
        "2010, 10.0, 1.0, 2.0, 0.1\n",
        encoding="utf-8",
    )
    frame = load_french_daily_ff3(path)
    assert len(frame) == 2
    assert abs(float(frame.iloc[0]["mkt_rf"]) - 0.01) < 1e-12
    assert abs(float(frame.iloc[0]["rf"]) - 0.0001) < 1e-12
    assert frame.iloc[1]["date"] == pd.Timestamp("2010-01-05")


def test_load_french_cleaned_decimal_csv(tmp_path: Path) -> None:
    path = tmp_path / "french_ff3_daily.csv"
    pd.DataFrame(
        {
            "date": ["2010-01-04", "2010-01-05"],
            "mkt_rf": [0.01, -0.005],
            "smb": [0.0, 0.0],
            "hml": [0.0, 0.0],
            "rf": [0.0001, 0.0001],
        }
    ).to_csv(path, index=False)
    frame = load_french_daily_ff3(path)
    assert abs(float(frame.iloc[0]["mkt_rf"]) - 0.01) < 1e-12
    assert len(frame) == 2


def test_trailing_ff3_does_not_use_the_outcome_session() -> None:
    """A one-session shock at t+1 must not enter betas used to residualise t+1."""
    dates = pd.bdate_range("2020-01-06", periods=40)
    rng = np.random.default_rng(20260813)
    mkt = rng.normal(0.0, 0.01, len(dates))
    smb = rng.normal(0.0, 0.005, len(dates))
    hml = rng.normal(0.0, 0.005, len(dates))
    factors = pd.DataFrame(
        {"date": dates, "mkt_rf": mkt, "smb": smb, "hml": hml, "rf": 0.0}
    )
    ret = mkt.copy()
    ret[-1] = 1.0
    rows = [
        {
            "symbol": "AAA",
            "session_date": dates[i],
            "return_end_date": dates[i + 1],
            "ret_close_h1": float(ret[i + 1]),
        }
        for i in range(len(dates) - 1)
    ]
    out = trailing_ff3_residuals(pd.DataFrame(rows), factors, window=15, min_obs=15)
    last = out.iloc[-1]
    assert abs(float(last["beta_mkt"]) - 1.0) < 0.05
    assert float(last["ff3_residual"]) > 0.9


def test_trailing_ff3_shock_is_in_the_own_session_residual() -> None:
    dates = pd.bdate_range("2020-01-06", periods=30)
    rng = np.random.default_rng(7)
    mkt = rng.normal(0.0, 0.01, len(dates))
    factors = pd.DataFrame(
        {
            "date": dates,
            "mkt_rf": mkt,
            "smb": rng.normal(0.0, 0.005, len(dates)),
            "hml": rng.normal(0.0, 0.005, len(dates)),
            "rf": 0.0,
        }
    )
    ret = mkt.copy()
    ret[16] = 0.8
    rows = [
        {
            "symbol": "AAA",
            "session_date": dates[i],
            "return_end_date": dates[i + 1],
            "ret_close_h1": float(ret[i + 1]),
        }
        for i in range(len(dates) - 1)
    ]
    out = trailing_ff3_residuals(pd.DataFrame(rows), factors, window=12, min_obs=12)
    shocked = out.loc[out["session_date"] == dates[15]].iloc[0]
    assert float(shocked["ff3_residual"]) > 0.6
