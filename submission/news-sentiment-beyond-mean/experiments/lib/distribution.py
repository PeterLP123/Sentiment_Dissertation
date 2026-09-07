"""Within-firm-day score distribution helpers for Workstream 2 EDA.

No return regressions. Moments are computed from story-level FinBERT scores
restricted to panel ``(symbol, session_date)`` keys.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.lib.panel import (
    DEFAULT_EVENTS_DB,
    FROZEN_DEV_END,
    FROZEN_EVAL_START,
)

# Plan checklist bins for count-conditioned bias.
N_BINS: tuple[tuple[str, int | None, int | None], ...] = (
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-20", 6, 20),
    ("21-75", 21, 75),
    ("76+", 76, None),
)
N_BIN_ORDER = [label for label, _, _ in N_BINS]


def assign_n_bin(n: pd.Series) -> pd.Series:
    """Map article counts to the frozen plan n-bins."""
    out = pd.Series(index=n.index, dtype="object")
    for label, lo, hi in N_BINS:
        if hi is None:
            mask = n >= lo
        else:
            mask = (n >= lo) & (n <= hi)
        out.loc[mask] = label
    return pd.Categorical(out, categories=N_BIN_ORDER, ordered=True)


def polarity_label(frame: pd.DataFrame) -> pd.Series:
    """Hard polarity from FinBERT class probabilities (argmax)."""
    probs = frame[["p_negative", "p_neutral", "p_positive"]].to_numpy(dtype=float)
    argmax = probs.argmax(axis=1)
    return pd.Series(
        np.asarray(["negative", "neutral", "positive"], dtype=object)[argmax],
        index=frame.index,
        name="polarity",
    )


def load_story_scores(
    db_path: Path,
    panel_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Load checkpoint stories that land on panel firm-days.

    Returns one row per story with scores; headlines are dropped so licensed
    text does not leave the notebook workspace via helper outputs.
    """
    required = {"symbol", "session_date"}
    missing = required - set(panel_keys.columns)
    if missing:
        raise ValueError(f"panel_keys missing columns: {sorted(missing)}")

    keys = panel_keys[["symbol", "session_date"]].drop_duplicates().copy()
    keys["symbol"] = keys["symbol"].astype(str)
    keys["session_date"] = pd.to_datetime(keys["session_date"]).dt.normalize()

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        events = pd.read_sql_query(
            """
            SELECT symbol, session_date, is_recap,
                   vader_compound, p_positive, p_negative, p_neutral,
                   finbert_score, finbert_negative_dominant
            FROM events
            """,
            con,
        )
    finally:
        con.close()

    events["symbol"] = events["symbol"].astype(str)
    events["session_date"] = pd.to_datetime(events["session_date"]).dt.normalize()
    merged = events.merge(keys, on=["symbol", "session_date"], how="inner")
    merged["polarity"] = polarity_label(merged)
    return merged


def firm_day_moments(stories: pd.DataFrame) -> pd.DataFrame:
    """Collapse stories to one firm-day distribution summary."""
    if stories.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "session_date",
                "n",
                "mean",
                "median",
                "std",
                "skew",
                "min",
                "max",
                "share_negative",
                "share_neutral",
                "share_positive",
                "mean_minus_median",
                "recap_share",
                "n_bin",
            ]
        )

    g = stories.groupby(["symbol", "session_date"], sort=False)
    moments = g["finbert_score"].agg(
        n="size",
        mean="mean",
        median="median",
        std="std",
        skew="skew",
        min="min",
        max="max",
    )
    polarity = (
        stories.groupby(["symbol", "session_date"])["polarity"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=["negative", "neutral", "positive"], fill_value=0.0)
        .rename(
            columns={
                "negative": "share_negative",
                "neutral": "share_neutral",
                "positive": "share_positive",
            }
        )
    )
    recap = g["is_recap"].mean().rename("recap_share")
    out = moments.join(polarity).join(recap).reset_index()
    out["mean_minus_median"] = out["mean"] - out["median"]
    out["std"] = out["std"].fillna(0.0)
    out["skew"] = out["skew"].fillna(0.0)
    out["n_bin"] = assign_n_bin(out["n"])
    return out


def moments_by_n_bin(moments: pd.DataFrame) -> pd.DataFrame:
    """Mean of firm-day moments within each n-bin (equal firm-day weight)."""
    cols = [
        "n",
        "mean",
        "median",
        "std",
        "skew",
        "min",
        "max",
        "share_negative",
        "share_neutral",
        "share_positive",
        "mean_minus_median",
        "recap_share",
    ]
    summary = moments.groupby("n_bin", observed=True)[cols].agg(["count", "mean", "median"]).sort_index()
    return summary


def mass_by_n_bin(moments: pd.DataFrame) -> pd.DataFrame:
    """Firm-day share and article mass by n-bin."""
    frame = moments.copy()
    frame["articles"] = frame["n"].astype(float)
    grouped = frame.groupby("n_bin", observed=True).agg(
        firm_days=("n", "size"),
        articles=("articles", "sum"),
    )
    grouped["firm_day_share"] = grouped["firm_days"] / grouped["firm_days"].sum()
    grouped["article_share"] = grouped["articles"] / grouped["articles"].sum()
    return grouped.sort_index()


def attach_split(moments: pd.DataFrame) -> pd.DataFrame:
    out = moments.copy()
    out["split"] = np.where(
        out["session_date"] <= pd.Timestamp(FROZEN_DEV_END),
        "development",
        np.where(
            out["session_date"] >= pd.Timestamp(FROZEN_EVAL_START),
            "evaluation",
            "gap",
        ),
    )
    return out


def build_distribution_tables(
    panel: pd.DataFrame,
    *,
    events_db: Path = DEFAULT_EVENTS_DB,
) -> dict[str, Any]:
    """Story → firm-day moments → n-bin summaries for the executable panel."""
    stories = load_story_scores(events_db, panel)
    moments = attach_split(firm_day_moments(stories))
    # Guard: moments should align with panel grain.
    panel_n = len(panel.drop_duplicates(["symbol", "session_date"]))
    return {
        "n_stories": int(len(stories)),
        "n_firm_days": int(len(moments)),
        "n_panel_keys": int(panel_n),
        "moments": moments,
        "by_n_bin": moments_by_n_bin(moments),
        "mass": mass_by_n_bin(moments),
        "split_counts": moments["split"].value_counts().to_dict(),
    }
