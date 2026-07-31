"""Non-pooled LSEG source-weighting and robustness helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from final_experiments.lib.aggregators import benjamini_hochberg, cross_sectional_daily_ic

SECTOR_ARMS = (
    "mean_sentiment_score",
    "mean_reuters_sentiment_score",
    "mean_non_reuters_sentiment_score",
    "reuters_2x_weighted_score",
    "mean_actionable_sentiment_score",
)
SECTOR_FAMILY = "five LSEG sector-33 source/actionability arms x h1; BH-FDR q=0.05"


def publisher_table(events: pd.DataFrame) -> pd.DataFrame:
    """Publisher/source inventory on unique LSEG headline events."""
    frame = events.copy()
    frame["news_date"] = pd.to_datetime(frame["news_date"]).dt.normalize()
    frame["_firm_day"] = list(zip(frame["symbol"].astype(str), frame["news_date"], strict=True))
    return (
        frame.groupby("source_codes", dropna=False)
        .agg(
            event_rows=("event_id", "size"),
            unique_firm_headline_pairs=("event_id", "nunique"),
            symbols=("symbol", "nunique"),
            firm_days=("_firm_day", "nunique"),
            mean_abs_score=("sentiment_score", lambda x: float(x.abs().mean())),
        )
        .reset_index()
        .sort_values("event_rows", ascending=False)
    )


def attach_weighted_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the sole ex-ante source-weight arm: Reuters receives weight 2."""
    out = panel.copy()
    numerator = (
        2.0 * out["reuters_count"] * out["mean_reuters_sentiment_score"]
        + out["non_reuters_count"] * out["mean_non_reuters_sentiment_score"]
    )
    denominator = 2.0 * out["reuters_count"] + out["non_reuters_count"]
    out["reuters_2x_weighted_score"] = np.where(denominator > 0, numerator / denominator, np.nan)
    return out


def attach_next_open_returns(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Map date-only news to the first strictly later open, then one open ahead."""
    price = prices.copy()
    price["session_date"] = pd.to_datetime(price["session_date"]).dt.normalize()
    price = price.sort_values(["symbol", "session_date"], kind="mergesort")
    price["next_adjusted_open"] = price.groupby("symbol", sort=False)["adjusted_open"].shift(-1)
    price["raw_open_h1"] = price["next_adjusted_open"] / price["adjusted_open"] - 1.0
    groups = {symbol: group.reset_index(drop=True) for symbol, group in price.groupby("symbol")}
    pieces = []
    source = panel.copy()
    source["news_date"] = pd.to_datetime(source["news_date"]).dt.normalize()
    for symbol, rows in source.groupby("symbol", sort=False):
        rows = rows.copy()
        stock = groups.get(symbol)
        if stock is None:
            rows["entry_session"] = pd.NaT
            rows["raw_open_h1"] = np.nan
        else:
            dates = pd.DatetimeIndex(stock["session_date"])
            pos = dates.searchsorted(rows["news_date"], side="right")
            valid = pos < len(stock)
            entry = np.full(len(rows), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
            returns = np.full(len(rows), np.nan)
            entry[valid] = stock.iloc[pos[valid]]["session_date"].to_numpy()
            returns[valid] = stock.iloc[pos[valid]]["raw_open_h1"].to_numpy(dtype=float)
            rows["entry_session"] = entry
            rows["raw_open_h1"] = returns
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def lseg_ic_table(
    frame: pd.DataFrame,
    arms: tuple[str, ...] = SECTOR_ARMS,
    *,
    family: str = SECTOR_FAMILY,
) -> pd.DataFrame:
    use = frame.loc[frame["headline_count"].gt(0)].copy() if "headline_count" in frame else frame
    rows = []
    for arm in arms:
        stats = cross_sectional_daily_ic(use[arm], use["raw_open_h1"], use["entry_session"])
        rows.append({"arm": arm, **stats})
    out = pd.DataFrame(rows)
    out["bh_reject_q05"] = benjamini_hochberg(out["p"].fillna(1.0).tolist())
    out["multiplicity_family"] = family
    out["target"] = "raw open-to-next-open return; LSEG robustness only"
    out["eligibility"] = "headline_count > 0 on identical firm-days"
    return out


__all__ = [
    "SECTOR_ARMS",
    "SECTOR_FAMILY",
    "attach_next_open_returns",
    "attach_weighted_signals",
    "lseg_ic_table",
    "publisher_table",
]
