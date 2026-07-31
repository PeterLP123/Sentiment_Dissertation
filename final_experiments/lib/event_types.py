"""Exploratory story-type and three-way story-class helpers.

The 12-type taxonomy is deterministic but not yet human-validated on FNSPID.
Outputs from this module are therefore candidate classifications, not ground
truth.  Publisher identity and story-family revision lineage remain unavailable
on the checkpoint grain.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from final_experiments.lib.aggregators import benjamini_hochberg
from src.sentiment_benchmark.headline_value import EVENT_PATTERNS, EVENT_TYPES

TYPE_FAMILY = "12 dominant event types x h1; BH-FDR q=0.05"
AUDIT_SEED = 20260731
AUDIT_N = 180

_ROUTINE_PATTERN = re.compile(
    r"\b(q[1-4]|quarterly|half[- ]year|full[- ]year|annual results?|"
    r"earnings report|reports? (first|second|third|fourth)[- ]quarter|"
    r"financial results?)\b",
    re.I,
)


def _classify_type(headlines: pd.Series) -> pd.Series:
    """Apply the repository's ordered regex taxonomy without row-wise Python."""
    text = headlines.fillna("").astype(str)
    out = pd.Series("other", index=text.index, dtype="object")
    unmatched = pd.Series(True, index=text.index)
    for event_type, pattern in EVENT_PATTERNS:
        hit = unmatched & text.str.contains(pattern, na=False)
        out.loc[hit] = event_type
        unmatched.loc[hit] = False
    return out


def build_event_type_artifacts(
    events_db: str | Path,
    novelty_path: str | Path,
    *,
    audit_n: int = AUDIT_N,
    seed: int = AUDIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return firm-day types, aggregate class mix, and a blinded audit sheet."""
    novelty = pd.read_parquet(
        novelty_path,
        columns=["event_index", "novelty_class", "is_market_recap"],
    )
    with sqlite3.connect(events_db) as connection:
        events = pd.read_sql_query(
            "SELECT event_index, symbol, session_date, headline, finbert_score FROM events",
            connection,
        )
    events = events.merge(novelty, on="event_index", how="inner", validate="one_to_one")
    events["session_date"] = pd.to_datetime(events["session_date"]).dt.normalize()
    events["event_type"] = _classify_type(events["headline"])
    routine = events["event_type"].eq("earnings_guidance") & events["headline"].fillna("").str.contains(_ROUTINE_PATTERN, na=False)
    events["story_class_candidate"] = np.select(
        [events["novelty_class"].astype(str).eq("repetition"), routine],
        ["repetition", "routine_candidate"],
        default="breaking_candidate",
    )

    type_counts = events.groupby(["symbol", "session_date", "event_type"], observed=True).size().rename("event_type_count").reset_index()
    type_order = {name: i for i, name in enumerate(EVENT_TYPES)}
    type_counts["_order"] = type_counts["event_type"].map(type_order)
    dominant = (
        type_counts.sort_values(
            ["symbol", "session_date", "event_type_count", "_order"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["symbol", "session_date"])
        .rename(columns={"event_type": "dominant_event_type"})[["symbol", "session_date", "dominant_event_type", "event_type_count"]]
    )
    totals = (
        events.groupby(["symbol", "session_date"], observed=True)
        .agg(
            typed_story_count=("event_index", "size"),
            earnings_guidance_count=(
                "event_type",
                lambda x: int((x == "earnings_guidance").sum()),
            ),
            repetition_count=(
                "story_class_candidate",
                lambda x: int((x == "repetition").sum()),
            ),
            routine_candidate_count=(
                "story_class_candidate",
                lambda x: int((x == "routine_candidate").sum()),
            ),
            breaking_candidate_count=(
                "story_class_candidate",
                lambda x: int((x == "breaking_candidate").sum()),
            ),
            market_recap_count=("is_market_recap", "sum"),
        )
        .reset_index()
    )
    firm_day = dominant.merge(totals, on=["symbol", "session_date"], validate="one_to_one")
    firm_day["earnings_guidance_share"] = firm_day["earnings_guidance_count"] / firm_day["typed_story_count"]

    mix = (
        events.groupby(["event_type", "story_class_candidate"], observed=True)
        .agg(
            stories=("event_index", "size"),
            symbols=("symbol", "nunique"),
            sessions=("session_date", "nunique"),
            mean_abs_score=("finbert_score", lambda x: float(x.abs().mean())),
            market_recap_share=("is_market_recap", "mean"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    sampled: list[pd.DataFrame] = []
    per_class = max(1, audit_n // 3)
    for label in ("breaking_candidate", "repetition", "routine_candidate"):
        pool = events.loc[events["story_class_candidate"] == label]
        take = min(per_class, len(pool))
        if take:
            sampled.append(pool.iloc[rng.choice(len(pool), size=take, replace=False)])
    audit = pd.concat(sampled, ignore_index=True)
    audit = audit.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    audit.insert(0, "audit_id", [f"FNSPID-TYPE-{i:03d}" for i in range(1, len(audit) + 1)])
    audit = audit[["audit_id", "symbol", "session_date", "headline"]].assign(
        human_event_type="",
        human_story_class="",
        human_confidence="",
        notes="",
    )
    return firm_day, mix, audit


def pooled_type_slopes(
    frame: pd.DataFrame,
    *,
    signal_col: str = "negative_share",
    outcome_col: str = "ar_open_h1",
    type_col: str = "dominant_event_type",
) -> pd.DataFrame:
    """One pooled model with type-specific intercepts and signal slopes.

    The model is ``y = alpha_type + beta_type * (-negative_share)``. Standard
    errors are clustered by session date, and BH is applied to all 12 slopes.
    """
    use = frame[[signal_col, outcome_col, type_col, "session_date"]].dropna().copy()
    use = use.loc[use[type_col].isin(EVENT_TYPES)]
    oriented = -use[signal_col].astype(float)
    dummies = pd.get_dummies(use[type_col], dtype=float).reindex(columns=EVENT_TYPES, fill_value=0.0)
    x = pd.concat(
        [
            dummies.add_prefix("intercept__"),
            dummies.mul(oriented.to_numpy(), axis=0).add_prefix("slope__"),
        ],
        axis=1,
    )
    model = sm.OLS(use[outcome_col].astype(float).to_numpy(), x.to_numpy()).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(use["session_date"])[0]},
    )
    rows = []
    for i, event_type in enumerate(EVENT_TYPES):
        j = len(EVENT_TYPES) + i
        rows.append(
            {
                "event_type": event_type,
                "slope": float(model.params[j]),
                "se": float(model.bse[j]),
                "t": float(model.tvalues[j]),
                "p": float(model.pvalues[j]),
                "n_firm_days": int((use[type_col] == event_type).sum()),
                "n_dates": int(use.loc[use[type_col] == event_type, "session_date"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out["bh_reject_q05"] = benjamini_hochberg(out["p"].fillna(1.0).tolist())
    out["multiplicity_family"] = TYPE_FAMILY
    out["estimand"] = "type-specific slope of oriented negative_share vs ar_open_h1"
    out["cluster"] = "session_date"
    return out


__all__ = [
    "AUDIT_N",
    "AUDIT_SEED",
    "TYPE_FAMILY",
    "build_event_type_artifacts",
    "pooled_type_slopes",
]
