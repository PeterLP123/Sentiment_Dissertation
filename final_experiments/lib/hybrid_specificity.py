"""Return-blind specificity filtering for the LSEG/Gemma hybrid experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.sparse_spread import build_exact_extrema_targets
from sentiment_benchmark.strategy_research.portfolio import (
    PositionTarget,
    TargetPortfolio,
)

FILTER_ID = "explicit_single_company_nontechnical_v1"


def _boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError(f"{column} must contain only true/false values")
    return mapped.astype(bool)


def apply_high_specificity_filter(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep explicit, single-company, non-technical unique headlines.

    The filter uses only deterministic metadata frozen with the completed score
    population. It never reads sentiment values or returns.
    """

    required = {
        "headline_sha256",
        "matched_symbols",
        "explicit_target",
        "contextual",
        "market_price_technical",
    }
    if missing := required - set(scores.columns):
        raise ValueError(f"score frame missing columns: {sorted(missing)}")
    if scores.empty:
        raise ValueError("score frame must not be empty")
    if scores["headline_sha256"].isna().any() or scores["headline_sha256"].duplicated().any():
        raise ValueError("score frame must contain one non-missing row per headline hash")

    matched = scores["matched_symbols"].astype(str).str.strip()
    if matched.eq("").any():
        raise ValueError("matched_symbols must not be empty")
    matched_count = matched.str.count(r"\|") + 1
    explicit = _boolean_column(scores, "explicit_target")
    contextual = _boolean_column(scores, "contextual")
    technical = _boolean_column(scores, "market_price_technical")
    if not contextual.eq(~explicit).all():
        raise ValueError("contextual must be the logical complement of explicit_target")

    keep = explicit & matched_count.eq(1) & ~technical
    filtered = scores.loc[keep].copy()
    filtered.attrs.update(scores.attrs)
    filtered.attrs["specificity_filter_id"] = FILTER_ID
    audit = {
        "filter_id": FILTER_ID,
        "input_unique_headlines": int(scores["headline_sha256"].nunique()),
        "retained_unique_headlines": int(filtered["headline_sha256"].nunique()),
        "input_company_associations": int(matched_count.sum()),
        "retained_company_associations": int(matched_count.loc[keep].sum()),
        "retained_share": float(keep.mean()),
        "dropped_contextual": int((~explicit).sum()),
        "dropped_multi_company": int(matched_count.gt(1).sum()),
        "dropped_market_price_technical": int(technical.sum()),
        "sentiment_or_return_used": False,
    }
    return filtered, audit


def audit_topic_metadata(path: str | Path) -> dict[str, int]:
    """Count non-empty subject/entity fields without retaining licensed text."""

    corpus_path = Path(path)
    rows = rows_with_subjects = rows_with_entities = 0
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            subjects = row.get("subjects") or []
            entities = row.get("entities") or []
            if not isinstance(subjects, list) or not isinstance(entities, list):
                raise ValueError("subjects and entities must be lists when present")
            rows += 1
            rows_with_subjects += bool(subjects)
            rows_with_entities += bool(entities)
    if rows == 0:
        raise ValueError("corpus must contain at least one row")
    return {
        "corpus_rows": rows,
        "rows_with_subjects": rows_with_subjects,
        "rows_with_entities": rows_with_entities,
    }


def build_capped_hybrid_targets(
    gemma_panel: pd.DataFrame,
    finbert_panel: pd.DataFrame,
    ordered_symbols: tuple[str, ...],
    *,
    single_name_cap: float = 0.25,
) -> tuple[tuple[TargetPortfolio, ...], pd.DataFrame]:
    """Use Gemma exact-extrema counts and FinBERT ranks for capped targets."""

    if not ordered_symbols or len(ordered_symbols) != len(set(ordered_symbols)):
        raise ValueError("ordered_symbols must be non-empty and unique")
    symbols = tuple(str(symbol).upper() for symbol in ordered_symbols)
    if len(symbols) != len(set(symbols)):
        raise ValueError("ordered_symbols collide after upper-casing")

    required = {"session_date", "symbol", "strongest_event"}
    prepared: list[pd.DataFrame] = []
    for name, panel in (("gemma", gemma_panel), ("finbert", finbert_panel)):
        if missing := required - set(panel.columns):
            raise ValueError(f"{name} panel missing columns: {sorted(missing)}")
        frame = panel.loc[:, sorted(required)].copy()
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame = frame.loc[frame["symbol"].isin(symbols)].sort_values(
            ["session_date", "symbol"], kind="mergesort", ignore_index=True
        )
        if frame.duplicated(["session_date", "symbol"]).any():
            raise ValueError(f"{name} panel contains duplicate symbol-sessions")
        prepared.append(frame)
    gemma, finbert = prepared
    keys = ["session_date", "symbol"]
    if not gemma[keys].equals(finbert[keys]):
        raise ValueError("Gemma and FinBERT panels are not aligned")
    if set(gemma["symbol"]) != set(symbols):
        raise ValueError("panels do not contain the requested symbol universe")
    if finbert["strongest_event"].isna().any():
        raise ValueError("FinBERT strongest-event ranks must be complete")

    _, audit = build_exact_extrema_targets(
        gemma,
        single_name_cap=single_name_cap,
    )
    ranks = {
        pd.Timestamp(session): tuple(
            day.sort_values(
                ["strongest_event", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            )["symbol"]
        )
        for session, day in finbert.groupby("session_date", sort=True)
    }
    expected = set(symbols)
    if any(set(row) != expected for row in ranks.values()):
        raise ValueError("every FinBERT session must contain the complete symbol universe")

    targets: list[TargetPortfolio] = []
    for row in audit.itertuples(index=False):
        n_long = int(row.positive_names)
        n_short = int(row.negative_names)
        eligible = bool(row.eligible)
        ranked = ranks[pd.Timestamp(row.session_date)]
        long_symbols = set(ranked[:n_long]) if eligible else set()
        short_symbols = set(ranked[-n_short:]) if eligible else set()
        if eligible and (not n_long or not n_short or long_symbols & short_symbols):
            raise RuntimeError("invalid hybrid rank selection")
        leg_exposure = (
            min(0.5, single_name_cap * n_long, single_name_cap * n_short)
            if eligible
            else 0.0
        )
        weights = {
            symbol: (
                leg_exposure / n_long
                if symbol in long_symbols
                else -leg_exposure / n_short
                if symbol in short_symbols
                else 0.0
            )
            for symbol in symbols
        }
        vector = np.asarray([weights[symbol] for symbol in symbols], dtype=float)
        if abs(vector.sum()) > 1e-12 or np.abs(vector).max() > single_name_cap + 1e-12:
            raise RuntimeError("hybrid target violates exposure constraints")
        positions = tuple(
            PositionTarget(
                symbol=symbol,
                action=(1.0 if symbol in long_symbols else -1.0 if symbol in short_symbols else 0.0),
                volatility=None,
                raw_weight=weights[symbol],
                target_weight=weights[symbol],
                exclusion_reason=None,
            )
            for symbol in symbols
        )
        targets.append(
            TargetPortfolio(
                session=str(pd.Timestamp(row.session_date).date()),
                positions=positions,
                gross_exposure=float(np.abs(vector).sum()),
                net_exposure=float(vector.sum()),
                long_exposure=float(vector[vector > 0].sum()),
                short_exposure=float(-vector[vector < 0].sum()),
                cash_weight=1.0 - float(vector.sum()),
            )
        )
    return tuple(targets), audit


__all__ = [
    "FILTER_ID",
    "apply_high_specificity_filter",
    "audit_topic_metadata",
    "build_capped_hybrid_targets",
]
