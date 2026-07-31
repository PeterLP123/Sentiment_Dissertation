"""Story-level novelty and repetition features for Workstream 2.

Built for the FNSPID FinBERT checkpoint grain: ``headline`` and ``is_recap``
exist; ``story_family_id`` does not. Family-revision count is therefore always
null rather than proxied.

Classification is an **ex-ante repetition screen**: an exact or near-duplicate
against strictly earlier same-firm dates in a trailing window is repetition;
everything else is novel-or-unclassified. The checkpoint market-recap regex is
kept as a separate flag and is not treated as routine scheduled reporting.
The screen is uncalibrated until a blinded hand-labelled audit exists.

Headlines are loaded for feature construction only. Aggregate helpers and the
audit template writer keep licensed text out of tracked paths; local
``outputs/`` may hold an audit worksheet with headlines for labelling.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.panel import DEFAULT_EVENTS_DB

# Plan defaults: trailing firm window and Jaccard near-duplicate cut.
DEFAULT_WINDOW_DAYS = 30
DEFAULT_JACCARD_THRESHOLD = 0.80
NOVELTY_CLASSES: tuple[str, ...] = ("novel_or_unclassified", "repetition")
AUDIT_SEED_DEFAULT = 42
AUDIT_N_DEFAULT = 180

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def normalize_headline(value: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (matches pipeline norm)."""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", (value or "").lower())).strip()


def headline_norm_sha256(headline: str) -> str:
    normalized = normalize_headline(headline)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def token_set(normalized_headline: str) -> frozenset[str]:
    if not normalized_headline:
        return frozenset()
    return frozenset(normalized_headline.split(" "))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def classify_novelty(
    *,
    exact_repeat_in_window: bool,
    near_dup_in_window: bool,
) -> str:
    """Repetition screen; non-repeats remain novel-or-unclassified."""
    if exact_repeat_in_window or near_dup_in_window:
        return "repetition"
    return "novel_or_unclassified"


@dataclass(frozen=True)
class NoveltyConfig:
    window_days: int = DEFAULT_WINDOW_DAYS
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD


def load_story_headlines(
    db_path: Path,
    panel_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load checkpoint stories; optionally restrict to panel firm-days.

    Includes ``headline`` for in-memory feature work. Callers must not write
    headline text to tracked artifacts.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        events = pd.read_sql_query(
            """
            SELECT event_index, event_key, symbol, session_date, headline, is_recap
            FROM events
            ORDER BY symbol, session_date, event_index
            """,
            con,
        )
    finally:
        con.close()

    events["symbol"] = events["symbol"].astype(str)
    events["session_date"] = pd.to_datetime(events["session_date"]).dt.normalize()
    events["is_recap"] = events["is_recap"].astype(bool)
    events["headline"] = events["headline"].astype(str)

    if panel_keys is not None:
        required = {"symbol", "session_date"}
        missing = required - set(panel_keys.columns)
        if missing:
            raise ValueError(f"panel_keys missing columns: {sorted(missing)}")
        keys = panel_keys[["symbol", "session_date"]].drop_duplicates().copy()
        keys["symbol"] = keys["symbol"].astype(str)
        keys["session_date"] = pd.to_datetime(keys["session_date"]).dt.normalize()
        events = events.merge(keys, on=["symbol", "session_date"], how="inner")
        events = events.sort_values(["symbol", "session_date", "event_key"], kind="mergesort").reset_index(drop=True)

    return events


def annotate_novelty(
    stories: pd.DataFrame,
    *,
    config: NoveltyConfig | None = None,
) -> pd.DataFrame:
    """Add novelty features and heuristic class; drop headline from the result.

    Required columns: ``event_index``, ``event_key``, ``symbol``,
    ``session_date``, ``headline``, ``is_recap``.
    """
    cfg = config or NoveltyConfig()
    required = {
        "event_index",
        "event_key",
        "symbol",
        "session_date",
        "headline",
        "is_recap",
    }
    missing = required - set(stories.columns)
    if missing:
        raise ValueError(f"stories missing columns: {sorted(missing)}")

    if stories.empty:
        return _empty_annotated()

    if config is not None:
        if config.window_days < 1:
            raise ValueError("window_days must be positive")
        if not 0.0 <= config.jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold must be between 0 and 1")

    # event_index is archive ingestion order, not publication chronology. Use
    # it only as retained provenance; event_key gives deterministic row order.
    frame = stories.sort_values(["symbol", "session_date", "event_key"], kind="mergesort").reset_index(drop=True)

    n = len(frame)
    norm_hash = np.empty(n, dtype=object)
    is_first = np.zeros(n, dtype=bool)
    exact_rep = np.zeros(n, dtype=bool)
    near_dup = np.zeros(n, dtype=bool)
    max_jac = np.zeros(n, dtype=float)
    days_since = np.full(n, np.nan, dtype=float)
    novelty_class = np.empty(n, dtype=object)

    symbols = frame["symbol"].astype(str).to_numpy()
    dates = pd.to_datetime(frame["session_date"]).dt.normalize().to_numpy()
    headlines = frame["headline"].astype(str).tolist()
    window_days = int(cfg.window_days)

    start = 0
    while start < n:
        symbol = symbols[start]
        symbol_end = start + 1
        while symbol_end < n and symbols[symbol_end] == symbol:
            symbol_end += 1

        seen_hashes: set[str] = set()
        window: deque[tuple[pd.Timestamp, str, frozenset[str]]] = deque()
        prev_date: pd.Timestamp | None = None
        day_start = start

        while day_start < symbol_end:
            session = pd.Timestamp(dates[day_start]).normalize()
            day_end = day_start + 1
            while day_end < symbol_end and pd.Timestamp(dates[day_end]).normalize() == session:
                day_end += 1

            cutoff = session - timedelta(days=window_days)
            while window and window[0][0] < cutoff:
                window.popleft()

            if prev_date is not None:
                days_since[day_start:day_end] = float((session - prev_date).days)

            # Every comparison in this loop is against a strictly earlier
            # firm date. Current-day rows enter state only after all are scored.
            current_day: list[tuple[pd.Timestamp, str, frozenset[str]]] = []
            for i in range(day_start, day_end):
                normalized = normalize_headline(headlines[i])
                h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                tokens = token_set(normalized)
                norm_hash[i] = h
                is_first[i] = h not in seen_hashes

                exact = any(prev_h == h for _, prev_h, _ in window)
                exact_rep[i] = exact
                best = 1.0 if exact else 0.0
                if not exact and tokens:
                    for _, prev_h, prev_tok in window:
                        if prev_h == h or not prev_tok:
                            continue
                        best = max(best, jaccard(tokens, prev_tok))
                max_jac[i] = best
                is_near = (not exact) and best >= cfg.jaccard_threshold
                near_dup[i] = is_near
                novelty_class[i] = classify_novelty(
                    exact_repeat_in_window=bool(exact),
                    near_dup_in_window=bool(is_near),
                )
                current_day.append((session, h, tokens))

            window.extend(current_day)
            seen_hashes.update(item[1] for item in current_day)
            prev_date = session
            day_start = day_end

        start = symbol_end

    out = frame.drop(columns=["headline"]).copy()
    out["headline_norm_sha256"] = norm_hash
    out["is_first_mention"] = is_first
    out["exact_repeat_in_window"] = exact_rep
    out["near_dup_in_window"] = near_dup
    out["max_jaccard_in_window"] = max_jac
    out = out.rename(columns={"is_recap": "is_market_recap"})
    out["family_revision_count"] = pd.NA
    out["days_since_prior_story"] = days_since
    out["novelty_class"] = pd.Categorical(novelty_class, categories=list(NOVELTY_CLASSES), ordered=False)
    out["novelty_window_days"] = cfg.window_days
    out["novelty_jaccard_threshold"] = cfg.jaccard_threshold
    return out


def _empty_annotated() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_index",
            "event_key",
            "symbol",
            "session_date",
            "is_market_recap",
            "headline_norm_sha256",
            "is_first_mention",
            "exact_repeat_in_window",
            "near_dup_in_window",
            "max_jaccard_in_window",
            "family_revision_count",
            "days_since_prior_story",
            "novelty_class",
            "novelty_window_days",
            "novelty_jaccard_threshold",
        ]
    )


def mix_by_firm_month(annotated: pd.DataFrame) -> pd.DataFrame:
    """Story-class counts and shares by symbol-month."""
    if annotated.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "year_month",
                "n_stories",
                "n_novel_or_unclassified",
                "n_repetition",
                "n_market_recap",
                "share_novel_or_unclassified",
                "share_repetition",
                "share_market_recap",
            ]
        )

    frame = annotated.copy()
    frame["year_month"] = pd.to_datetime(frame["session_date"]).dt.to_period("M").astype(str)
    counts = (
        frame.groupby(["symbol", "year_month", "novelty_class"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(NOVELTY_CLASSES), fill_value=0)
        .rename(
            columns={
                "novel_or_unclassified": "n_novel_or_unclassified",
                "repetition": "n_repetition",
            }
        )
        .reset_index()
    )
    counts["n_stories"] = counts["n_novel_or_unclassified"] + counts["n_repetition"]
    recap = frame.groupby(["symbol", "year_month"], observed=True)["is_market_recap"].sum().rename("n_market_recap").reset_index()
    counts = counts.merge(recap, on=["symbol", "year_month"], how="left")
    for cls, col in (
        ("novel_or_unclassified", "share_novel_or_unclassified"),
        ("repetition", "share_repetition"),
    ):
        counts[col] = counts[f"n_{cls}"] / counts["n_stories"].clip(lower=1)
    counts["share_market_recap"] = counts["n_market_recap"] / counts["n_stories"].clip(lower=1)
    return counts.sort_values(["symbol", "year_month"]).reset_index(drop=True)


def mix_overall(annotated: pd.DataFrame) -> pd.DataFrame:
    """Panel-wide class counts and shares."""
    if annotated.empty:
        return pd.DataFrame({"novelty_class": list(NOVELTY_CLASSES), "n": 0, "share": 0.0})
    counts = annotated["novelty_class"].value_counts(dropna=False).reindex(list(NOVELTY_CLASSES), fill_value=0).rename("n").reset_index()
    counts.columns = ["novelty_class", "n"]
    total = int(counts["n"].sum())
    counts["share"] = counts["n"] / total if total else 0.0
    return counts


def draw_audit_sample(
    stories_with_headline: pd.DataFrame,
    annotated: pd.DataFrame,
    *,
    n: int = AUDIT_N_DEFAULT,
    seed: int = AUDIT_SEED_DEFAULT,
    classes: Iterable[str] = NOVELTY_CLASSES,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a blinded worksheet and separate prediction/weight key.

    Sampling is stratified by machine novelty class. The worksheet excludes
    predictions and features; the key records inclusion probabilities and
    inverse-probability weights for population precision/recall estimates.
    Both outputs contain licensed local data and belong under ``outputs/``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if window_days < 1:
        raise ValueError("window_days must be positive")
    class_list = list(classes)
    if not class_list:
        raise ValueError("classes must be non-empty")

    merged = stories_with_headline.merge(
        annotated[
            [
                "event_key",
                "novelty_class",
                "is_market_recap",
                "is_first_mention",
                "exact_repeat_in_window",
                "near_dup_in_window",
                "max_jaccard_in_window",
                "days_since_prior_story",
            ]
        ],
        on="event_key",
        how="inner",
    )
    if merged.empty:
        return merged, merged

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    base, remainder = divmod(n, len(class_list))
    for class_index, cls in enumerate(class_list):
        pool = merged.loc[merged["novelty_class"] == cls]
        if pool.empty:
            continue
        requested = base + (1 if class_index < remainder else 0)
        if requested == 0:
            continue
        take = min(requested, len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        part = pool.loc[idx].copy()
        part["population_count"] = len(pool)
        part["sampled_count"] = take
        part["inclusion_probability"] = take / len(pool)
        part["sample_weight"] = len(pool) / take
        parts.append(part)

    if not parts:
        empty = merged.iloc[0:0].copy()
        return empty, empty

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample.insert(0, "audit_seed", seed)

    context_by_key: dict[str, str] = {}
    stories = stories_with_headline.copy()
    stories["session_date"] = pd.to_datetime(stories["session_date"]).dt.normalize()
    stories["_symbol_key"] = stories["symbol"].astype(str)
    stories = stories.sort_values(["_symbol_key", "session_date", "event_key"], kind="mergesort")
    stories_by_symbol = stories.set_index("_symbol_key", drop=False)
    for symbol, selected in sample.groupby("symbol", sort=False):
        history = stories_by_symbol.loc[[str(symbol)]]
        for row in selected.itertuples(index=False):
            session = pd.Timestamp(row.session_date).normalize()
            prior = history.loc[
                history["session_date"].between(
                    session - timedelta(days=window_days),
                    session - timedelta(days=1),
                )
            ]
            records = [
                {
                    "session_date": pd.Timestamp(item.session_date).date().isoformat(),
                    "headline": str(item.headline),
                }
                for item in prior.itertuples(index=False)
            ]
            context_by_key[str(row.event_key)] = json.dumps(records, ensure_ascii=False)

    worksheet = sample[["audit_seed", "event_key", "symbol", "session_date", "headline"]].copy()
    worksheet["prior_30d_headlines_json"] = worksheet["event_key"].astype(str).map(context_by_key)
    worksheet["human_novelty_class"] = pd.NA
    worksheet["human_story_type"] = pd.NA
    worksheet["human_notes"] = pd.NA

    key_columns = [
        "audit_seed",
        "event_key",
        "novelty_class",
        "is_market_recap",
        "is_first_mention",
        "exact_repeat_in_window",
        "near_dup_in_window",
        "max_jaccard_in_window",
        "days_since_prior_story",
        "population_count",
        "sampled_count",
        "inclusion_probability",
        "sample_weight",
    ]
    key = sample[key_columns].copy()
    return worksheet.head(n), key.head(n)


def build_novelty_tables(
    panel: pd.DataFrame,
    *,
    events_db: Path = DEFAULT_EVENTS_DB,
    config: NoveltyConfig | None = None,
    audit_n: int = AUDIT_N_DEFAULT,
    audit_seed: int = AUDIT_SEED_DEFAULT,
) -> dict[str, Any]:
    """Load → annotate → aggregate. Headlines retained only for the audit draw."""
    cfg = config or NoveltyConfig()
    stories = load_story_headlines(events_db, panel)
    annotated = annotate_novelty(stories, config=cfg)
    overall = mix_overall(annotated)
    by_firm_month = mix_by_firm_month(annotated)
    audit, audit_key = draw_audit_sample(
        stories,
        annotated,
        n=audit_n,
        seed=audit_seed,
        window_days=cfg.window_days,
    )
    return {
        "n_stories": int(len(annotated)),
        "config": {
            "window_days": cfg.window_days,
            "jaccard_threshold": cfg.jaccard_threshold,
            "family_revision_count": "null — story_family_id absent from checkpoint",
            "classifier": "heuristic_uncalibrated",
            "comparison_timing": "strictly_earlier_firm_dates_only",
            "market_recap_role": "orthogonal_flag_not_routine_reporting",
            "routine_reporting_status": "unavailable_on_checkpoint_grain",
            "audit_seed": audit_seed,
            "audit_n": int(len(audit)),
            "audit_design": "blinded_stratified_with_separate_weight_key",
        },
        "annotated": annotated,
        "overall": overall,
        "by_firm_month": by_firm_month,
        "audit_sample": audit,
        "audit_key": audit_key,
    }
