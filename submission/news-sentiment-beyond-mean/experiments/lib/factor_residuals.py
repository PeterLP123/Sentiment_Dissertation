"""Trailing Fama--French three-factor residuals for close-to-close returns.

Betas at formation session ``t`` use only returns that have already completed
by the close of ``t``. The residual attached to session ``t`` is the next-session
close-to-close excess return minus the factor realisation on that next session
times those lagged betas. French daily factors are close-to-close, so they are
aligned to the return *end* date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FF3_COLUMNS = ("mkt_rf", "smb", "hml", "rf")
WINDOW = 252
MIN_OBS = 126


def load_french_daily_ff3(path: Path) -> pd.DataFrame:
    """Load Ken French daily FF3 factors.

    Accepts the vendor CSV (percent units, header/footer text) or a cleaned
    CSV with columns ``date,mkt_rf,smb,hml,rf``. Returned values are decimals.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first.lower().startswith("date"):
        frame = pd.read_csv(path)
        frame.columns = [str(c).strip().casefold().replace("-", "_") for c in frame.columns]
        if "date" not in frame.columns:
            raise ValueError(f"French factor file missing date: {path}")
        keep = ["date", *[col for col in FF3_COLUMNS if col in frame.columns]]
        frame = frame[keep].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in FF3_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame["mkt_rf"].abs().median() > 0.05:
            for col in FF3_COLUMNS:
                frame[col] = frame[col] / 100.0
        return (
            frame.dropna(subset=["date", *FF3_COLUMNS])
            .sort_values("date")
            .reset_index(drop=True)
        )

    rows: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if rows:
                break
            continue
        if not line[:8].isdigit():
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 5:
            continue
        rows.append(parts[:5])
    if not rows:
        raise ValueError(f"no daily FF3 rows parsed from {path}")
    parsed = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    parsed["date"] = pd.to_datetime(parsed["date"], format="%Y%m%d", errors="coerce")
    for col in FF3_COLUMNS:
        parsed[col] = pd.to_numeric(parsed[col], errors="coerce") / 100.0
    return parsed.dropna(subset=["date", *FF3_COLUMNS]).sort_values("date").reset_index(drop=True)


def _rolling_betas_exclude_current(
    y: np.ndarray,
    x: np.ndarray,
    *,
    window: int,
    min_obs: int,
) -> np.ndarray:
    """OLS betas at i use rows [i-window, i), never y[i] or x[i]."""
    n, k = x.shape
    betas = np.full((n, k), np.nan)
    for i in range(min_obs, n):
        start = 0 if i < window else i - window
        y_w = y[start:i]
        x_w = x[start:i]
        finite = np.isfinite(y_w) & np.isfinite(x_w).all(axis=1)
        if int(finite.sum()) < min_obs:
            continue
        y_use = y_w[finite]
        x_use = x_w[finite]
        if np.linalg.matrix_rank(x_use) < k:
            continue
        betas[i] = np.linalg.lstsq(x_use, y_use, rcond=None)[0]
    return betas


def trailing_ff3_residuals(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    window: int = WINDOW,
    min_obs: int = MIN_OBS,
    market_symbol: str = "SPY",
) -> pd.DataFrame:
    """Attach next-session FF3 residuals dated at the formation session.

    ``returns`` must contain ``symbol``, ``session_date``, ``ret_close_h1`` and
    ``return_end_date``. Factors are merged on ``return_end_date``.
    """
    required = {"symbol", "session_date", "ret_close_h1", "return_end_date"}
    if missing := required - set(returns.columns):
        raise ValueError(f"returns missing columns: {sorted(missing)}")
    if window < min_obs:
        raise ValueError("window must be at least min_obs")
    if min_obs < 8:
        raise ValueError("min_obs must be at least 8")

    stock = returns.loc[returns["symbol"].astype(str).str.upper() != market_symbol.upper()].copy()
    stock["symbol"] = stock["symbol"].astype(str).str.upper()
    stock["session_date"] = pd.to_datetime(stock["session_date"]).dt.normalize()
    stock["return_end_date"] = pd.to_datetime(stock["return_end_date"]).dt.normalize()

    ff = factors.copy()
    ff["date"] = pd.to_datetime(ff["date"]).dt.normalize()
    ff = ff.rename(columns={"date": "return_end_date"})
    merged = stock.merge(ff, on="return_end_date", how="left", validate="m:1")
    merged["excess"] = merged["ret_close_h1"] - merged["rf"]

    pieces: list[pd.DataFrame] = []
    for symbol, block in merged.groupby("symbol", sort=False):
        block = block.sort_values("session_date", kind="mergesort").reset_index(drop=True)
        y = block["excess"].to_numpy(dtype=float)
        x = np.column_stack(
            [
                np.ones(len(block), dtype=float),
                block["mkt_rf"].to_numpy(dtype=float),
                block["smb"].to_numpy(dtype=float),
                block["hml"].to_numpy(dtype=float),
            ]
        )
        betas = _rolling_betas_exclude_current(y, x, window=window, min_obs=min_obs)
        fitted = np.einsum("ij,ij->i", x, betas)
        residual = y - fitted
        n_obs = np.full(len(block), np.nan)
        for i in range(len(block)):
            if np.isfinite(betas[i, 0]):
                start = 0 if i < window else i - window
                n_obs[i] = float(i - start)
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "session_date": block["session_date"].to_numpy(),
                "ff3_residual": residual,
                "n_beta_obs": n_obs,
                "beta_mkt": betas[:, 1],
                "beta_smb": betas[:, 2],
                "beta_hml": betas[:, 3],
            }
        )
        pieces.append(out)
    if not pieces:
        return pd.DataFrame(
            columns=[
                "symbol",
                "session_date",
                "ff3_residual",
                "n_beta_obs",
                "beta_mkt",
                "beta_smb",
                "beta_hml",
            ]
        )
    return pd.concat(pieces, ignore_index=True)


__all__ = [
    "FF3_COLUMNS",
    "MIN_OBS",
    "WINDOW",
    "load_french_daily_ff3",
    "trailing_ff3_residuals",
]
