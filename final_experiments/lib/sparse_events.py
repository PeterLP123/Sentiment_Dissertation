"""Retrospective sparse-event strategy helpers for expanded LSEG headlines.

The experiment deliberately narrows the 44-company Gemma population before
portfolio construction: direct single-company material events, no technical
market commentary, and no near-duplicate against a strictly earlier 30-day
same-company window.  A second signal attenuates the selected event by an
unvalidated headline-only proxy for distance from realised cash flow.

Headline text is accepted for local feature construction and always removed
from returned frames.  These helpers do not make the proxy a validated event
taxonomy and do not turn the already-open 2025--2026 window into a holdout.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.aggregators import (
    benjamini_hochberg,
    cross_sectional_daily_ic,
)
from final_experiments.lib.evaluate import (
    TradeConfig,
    cross_sectional_rank_scores,
    summarize_daily_portfolio,
)
from final_experiments.lib.event_types import _classify_type
from final_experiments.lib.lseg_expanded import (
    EXPLORATORY_SPLIT,
    INFORMATION_BUFFER_MINUTES,
    map_headlines_to_entry_sessions,
)
from final_experiments.lib.novelty import normalize_headline, token_set

NOVELTY_WINDOW_DAYS = 30
NOVELTY_JACCARD_THRESHOLD = 0.80
HOLDING_HORIZONS: tuple[int, ...] = (1, 3, 5)
COST_BPS_PER_SIDE = 10.0
BOOTSTRAP_BLOCK_SESSIONS = 5
BOOTSTRAP_REPLICATIONS = 1_999
SEED = 20260805
MIN_PORTFOLIO_NAMES = 10
SPARSE_PRIMARY_FAMILY = (
    "two predeclared sparse signals x three state horizons; "
    "daily cross-sectional IC, BH-FDR q=0.05"
)

MATERIAL_EVENT_TYPES: tuple[str, ...] = (
    "earnings_guidance",
    "mna_strategy",
    "legal_regulatory",
    "product_technology",
    "management_governance",
    "labor_esg",
)

# Ordered, headline-only proxy.  Priority is intentionally frozen before any
# return is inspected: speculative > pending > realised > binding/intermediate.
_DISTANCE_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (
        3,
        re.compile(
            r"\b(?:considers?|considering|explores?|exploring|may|might|could|"
            r"potential|preliminary|non[- ]binding|memorandum|aims?|plans?|"
            r"planning|targets?|seeks?|seeking)\b",
            re.I,
        ),
    ),
    (
        2,
        re.compile(
            r"\b(?:subject to|pending|awaiting|trials?|phase [1-3]|applications?|"
            r"proposals?|expects?|expected|intends?|forecast|guidance)\b",
            re.I,
        ),
    ),
    (
        0,
        re.compile(
            r"\b(?:completes?|completed|closes?|closed|receives?|received|paid|"
            r"delivers?|delivered|reports?|reported|profits?|revenues?|earnings|"
            r"dividends?|buybacks?)\b",
            re.I,
        ),
    ),
    (
        1,
        re.compile(
            r"\b(?:signs?|signed|contracts?|orders?|awards?|approvals?|approved|"
            r"launches?|launched|starts?|started|opens?|opened|agrees?|agreed|"
            r"acquires?|acquired|mergers?|deals?)\b",
            re.I,
        ),
    ),
)


def classify_proxy_cash_flow_distance(headlines: pd.Series) -> pd.Series:
    """Return nullable integer distance 0..3 from the frozen phrase proxy."""

    text = headlines.fillna("").astype(str)
    out = pd.Series(pd.NA, index=text.index, dtype="Int64")
    for distance, pattern in _DISTANCE_PATTERNS:
        hit = out.isna() & text.str.contains(pattern, na=False)
        out.loc[hit] = distance
    return out


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return overlap / len(left | right) if overlap else 0.0


def annotate_strict_prior_novelty(
    stories: pd.DataFrame,
    *,
    window_days: int = NOVELTY_WINDOW_DAYS,
    jaccard_threshold: float = NOVELTY_JACCARD_THRESHOLD,
) -> pd.DataFrame:
    """Screen near-duplicates against strictly earlier same-symbol timestamps.

    All rows sharing an exact timestamp are classified before any of them enter
    the state.  The token posting index is only an exact acceleration: a pair
    with positive Jaccard must share at least one token, so no possible match is
    omitted.
    """

    required = {
        "headline_sha256",
        "symbol",
        "first_timestamp",
        "headline",
    }
    missing = required - set(stories.columns)
    if missing:
        raise ValueError(f"stories missing columns: {sorted(missing)}")
    if window_days < 1:
        raise ValueError("window_days must be positive")
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")

    frame = stories.copy()
    frame["first_timestamp"] = pd.to_datetime(
        frame["first_timestamp"], utc=True, errors="raise"
    )
    frame = frame.sort_values(
        ["symbol", "first_timestamp", "headline_sha256"], kind="mergesort"
    ).reset_index(drop=True)
    normalized = frame["headline"].map(normalize_headline)
    tokens = normalized.map(token_set).tolist()
    hashes = frame["headline_sha256"].astype(str).tolist()

    exact_repeat = np.zeros(len(frame), dtype=bool)
    near_duplicate = np.zeros(len(frame), dtype=bool)
    max_jaccard = np.zeros(len(frame), dtype=float)
    window_delta = np.timedelta64(int(window_days), "D")

    for _, positions in frame.groupby("symbol", sort=False).indices.items():
        ordered = list(positions)
        active: deque[tuple[pd.Timestamp, int]] = deque()
        active_tokens: dict[int, frozenset[str]] = {}
        active_hashes: dict[int, str] = {}
        token_postings: defaultdict[str, set[int]] = defaultdict(set)
        hash_postings: defaultdict[str, set[int]] = defaultdict(set)

        cursor = 0
        while cursor < len(ordered):
            start = cursor
            timestamp = pd.Timestamp(frame.loc[ordered[cursor], "first_timestamp"])
            while (
                cursor < len(ordered)
                and pd.Timestamp(frame.loc[ordered[cursor], "first_timestamp"])
                == timestamp
            ):
                cursor += 1
            current = ordered[start:cursor]
            cutoff = timestamp - window_delta

            while active and active[0][0] < cutoff:
                _, old = active.popleft()
                old_tokens = active_tokens.pop(old)
                old_hash = active_hashes.pop(old)
                for token in old_tokens:
                    posting = token_postings[token]
                    posting.discard(old)
                    if not posting:
                        del token_postings[token]
                hashes_for_value = hash_postings[old_hash]
                hashes_for_value.discard(old)
                if not hashes_for_value:
                    del hash_postings[old_hash]

            for row in current:
                row_tokens = tokens[row]
                row_hash = hashes[row]
                exact = bool(hash_postings.get(row_hash))
                candidates: set[int] = set()
                for token in row_tokens:
                    candidates.update(token_postings.get(token, ()))

                best = 1.0 if exact else 0.0
                if not exact:
                    n_tokens = len(row_tokens)
                    for prior in candidates:
                        prior_tokens = active_tokens[prior]
                        prior_n = len(prior_tokens)
                        if (
                            prior_n < jaccard_threshold * n_tokens
                            or prior_n > n_tokens / max(jaccard_threshold, 1e-12)
                        ):
                            continue
                        best = max(best, _jaccard(row_tokens, prior_tokens))
                exact_repeat[row] = exact
                max_jaccard[row] = best
                near_duplicate[row] = (not exact) and best >= jaccard_threshold

            # Strict-prior guarantee: add the whole timestamp group only now.
            for row in current:
                active.append((timestamp, row))
                active_tokens[row] = tokens[row]
                active_hashes[row] = hashes[row]
                for token in tokens[row]:
                    token_postings[token].add(row)
                hash_postings[hashes[row]].add(row)

    out = frame.drop(columns=["headline"])
    out["exact_repeat_in_window"] = exact_repeat
    out["near_duplicate_in_window"] = near_duplicate
    out["max_jaccard_in_window"] = max_jaccard
    out["is_novel_material_event"] = ~(exact_repeat | near_duplicate)
    out["novelty_window_days"] = window_days
    out["novelty_jaccard_threshold"] = jaccard_threshold
    return out


def prepare_sparse_event_candidates(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    information_buffer_minutes: int = INFORMATION_BUFFER_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen eligibility/novelty rules and map candidates to entry opens."""

    required = {
        "headline_sha256",
        "headline",
        "matched_symbols",
        "first_timestamp",
        "explicit_target",
        "market_price_technical",
        "label",
        "score",
        "scorer",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing columns: {sorted(missing)}")

    frame = scores.copy()
    frame["matched_symbol_count"] = frame["matched_symbols"].str.count(r"\|") + 1
    frame["event_type"] = _classify_type(frame["headline"])

    cumulative_masks: list[tuple[str, pd.Series]] = []
    mask = pd.Series(True, index=frame.index)
    cumulative_masks.append(("successful_unique_headlines", mask.copy()))
    mask &= frame["explicit_target"].astype(bool)
    cumulative_masks.append(("explicit_target", mask.copy()))
    mask &= frame["matched_symbol_count"].eq(1)
    cumulative_masks.append(("single_company", mask.copy()))
    mask &= ~frame["market_price_technical"].astype(bool)
    cumulative_masks.append(("exclude_market_price_technical", mask.copy()))
    mask &= frame["event_type"].isin(MATERIAL_EVENT_TYPES)
    cumulative_masks.append(("material_event_type", mask.copy()))

    eligible = frame.loc[mask].copy()
    eligible["symbol"] = eligible["matched_symbols"].astype(str)
    eligible["proxy_cash_flow_distance"] = classify_proxy_cash_flow_distance(
        eligible["headline"]
    )
    annotated = annotate_strict_prior_novelty(eligible)
    novel = annotated.loc[annotated["is_novel_material_event"]].copy()

    mapped_base = map_headlines_to_entry_sessions(
        novel,
        prices,
        information_buffer_minutes=information_buffer_minutes,
    )
    feature_columns = [
        "headline_sha256",
        "event_type",
        "proxy_cash_flow_distance",
        "exact_repeat_in_window",
        "near_duplicate_in_window",
        "max_jaccard_in_window",
        "is_novel_material_event",
        "novelty_window_days",
        "novelty_jaccard_threshold",
    ]
    mapped = mapped_base.merge(
        novel[feature_columns],
        on="headline_sha256",
        how="left",
        validate="many_to_one",
    )

    audit_rows = [
        {"stage": stage, "headlines": int(stage_mask.sum())}
        for stage, stage_mask in cumulative_masks
    ]
    audit_rows.extend(
        [
            {
                "stage": "novel_material_event",
                "headlines": int(annotated["is_novel_material_event"].sum()),
            },
            {
                "stage": "novel_with_distance_proxy",
                "headlines": int(
                    (
                        annotated["is_novel_material_event"]
                        & annotated["proxy_cash_flow_distance"].notna()
                    ).sum()
                ),
            },
            {
                "stage": "mapped_to_eligible_open",
                "headlines": int(mapped["headline_sha256"].nunique()),
            },
        ]
    )
    audit = pd.DataFrame(audit_rows)
    audit["share_of_successes"] = audit["headlines"] / len(frame)
    return mapped, audit


def select_strongest_sparse_events(mapped: pd.DataFrame) -> pd.DataFrame:
    """Select one strongest absolute-score event per company/entry session.

    Equal-magnitude opposite-direction ties are omitted.  The distance signal
    uses exactly the same selected event and is missing if that headline does
    not match the frozen proxy.
    """

    required = {
        "symbol",
        "entry_session",
        "headline_sha256",
        "score",
        "proxy_cash_flow_distance",
        "event_type",
    }
    missing = required - set(mapped.columns)
    if missing:
        raise ValueError(f"mapped events missing columns: {sorted(missing)}")
    if mapped.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "entry_session",
                "headline_sha256",
                "event_type",
                "proxy_cash_flow_distance",
                "material_signal",
                "distance_signal",
            ]
        )

    keys = ["symbol", "entry_session"]
    frame = mapped.copy()
    frame["_abs_score"] = frame["score"].astype(float).abs()
    frame["_max_abs"] = frame.groupby(keys, sort=False)["_abs_score"].transform("max")
    top = frame.loc[np.isclose(frame["_abs_score"], frame["_max_abs"])].copy()
    top["_sign"] = np.sign(top["score"].astype(float))
    conflict = (
        top.loc[top["_sign"].ne(0)]
        .groupby(keys, sort=False)["_sign"]
        .nunique()
        .gt(1)
        .rename("_conflict")
    )
    top = top.merge(conflict, on=keys, how="left")
    conflict_mask = top["_conflict"].astype("boolean").fillna(False)
    top = top.loc[~conflict_mask].copy()
    selected = (
        top.sort_values(keys + ["headline_sha256"], kind="mergesort")
        .drop_duplicates(keys, keep="first")
        .copy()
    )
    selected["material_signal"] = selected["score"].astype(float)
    distance = selected["proxy_cash_flow_distance"].astype("Float64")
    selected["distance_signal"] = selected["material_signal"] / (1.0 + distance)
    keep = [
        "symbol",
        "entry_session",
        "headline_sha256",
        "event_type",
        "proxy_cash_flow_distance",
        "material_signal",
        "distance_signal",
    ]
    return selected[keep].reset_index(drop=True)


def build_sparse_state_panels(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int] = HOLDING_HORIZONS,
) -> dict[int, pd.DataFrame]:
    """Carry selected signals for fixed exchange-session state horizons."""

    if any(int(horizon) < 1 for horizon in horizons):
        raise ValueError("state horizons must be positive")
    price = prices.sort_values(["symbol", "session_date"], kind="mergesort").copy()
    price["raw_open_h1"] = (
        price.groupby("symbol", sort=False)["adjusted_open"].shift(-1)
        / price["adjusted_open"]
        - 1.0
    )
    events = selected.rename(columns={"entry_session": "session_date"})[
        ["symbol", "session_date", "material_signal", "distance_signal"]
    ]
    base = price.merge(
        events,
        on=["symbol", "session_date"],
        how="left",
        validate="one_to_one",
    )
    base["split"] = EXPLORATORY_SPLIT
    # Any newly selected material event starts a fresh state segment.  This is
    # also binding for the distance arm: an unknown-distance new event clears
    # the older classified state instead of silently carrying it through.
    base["_event_trigger"] = base["material_signal"].notna()
    base["_state_id"] = base.groupby("symbol", sort=False)["_event_trigger"].cumsum()

    panels: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        panel = base.copy()
        for signal in ("material_signal", "distance_signal"):
            if int(horizon) > 1:
                panel[signal] = panel.groupby(
                    ["symbol", "_state_id"], sort=False
                )[signal].ffill(limit=int(horizon) - 1)
        panel["state_horizon_sessions"] = int(horizon)
        panels[int(horizon)] = panel.drop(columns=["_event_trigger", "_state_id"])
    return panels


def build_sparse_daily_portfolio(
    panel: pd.DataFrame,
    signal_col: str,
    *,
    cost_bps_per_side: float = COST_BPS_PER_SIDE,
    min_names: int = MIN_PORTFOLIO_NAMES,
) -> pd.DataFrame:
    """Dollar-neutral rank book that explicitly closes on inactive sessions."""

    required = {"session_date", "symbol", signal_col, "raw_open_h1"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    frame = panel.loc[panel["raw_open_h1"].notna()].sort_values(
        ["session_date", "symbol"], kind="mergesort"
    )
    previous: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    cost_rate = cost_bps_per_side / 10_000.0

    for session, full_day in frame.groupby("session_date", sort=True):
        day = full_day.dropna(subset=[signal_col])
        symbols = day["symbol"].astype(str).to_numpy()
        values = day[signal_col].to_numpy(dtype=float)
        returns = day["raw_open_h1"].to_numpy(dtype=float)
        if len(day) >= min_names:
            raw_weights = cross_sectional_rank_scores(values)
            gross = float(np.abs(raw_weights).sum())
            weights = raw_weights / gross if gross > 0 else np.zeros(len(day))
        else:
            weights = np.zeros(len(day))

        current = {
            symbol: float(weight)
            for symbol, weight in zip(symbols, weights, strict=True)
            if weight != 0.0
        }
        names = set(previous) | set(current)
        turnover = 0.5 * sum(
            abs(current.get(name, 0.0) - previous.get(name, 0.0))
            for name in names
        )
        gross_return = float(np.dot(weights, returns)) if len(day) else 0.0
        cost = float(2.0 * turnover * cost_rate)
        rows.append(
            {
                "session_date": session,
                "split": EXPLORATORY_SPLIT,
                "n_names": int(len(day)),
                "n_long": int((weights > 0).sum()),
                "n_short": int((weights < 0).sum()),
                "net_exposure": float(weights.sum()),
                "gross_return": gross_return,
                "turnover": float(turnover),
                "cost": cost,
                "net_return": gross_return - cost,
            }
        )
        previous = current

    daily = pd.DataFrame(rows)
    if not daily.empty and previous:
        last = daily.index[-1]
        daily.loc[last, "turnover"] += 0.5
        daily.loc[last, "cost"] += cost_rate
        daily.loc[last, "net_return"] -= cost_rate
    return daily


def _bh_adjusted_p(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def evaluate_sparse_arms(
    panels: dict[int, pd.DataFrame],
    *,
    cost_bps_per_side: float = COST_BPS_PER_SIDE,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    """Evaluate the frozen two-signal by three-horizon family."""

    cfg = TradeConfig(
        outcome="raw_open_h1",
        position_mode="cs_rank",
        threshold=0.0,
        min_names=MIN_PORTFOLIO_NAMES,
        cost_bps_per_side=cost_bps_per_side,
        bootstrap_block_length=BOOTSTRAP_BLOCK_SESSIONS,
        bootstrap_replications=bootstrap_replications,
        random_seed=seed,
    )
    rows: list[dict[str, Any]] = []
    daily_outputs: dict[tuple[str, int], pd.DataFrame] = {}
    for horizon in HOLDING_HORIZONS:
        panel = panels[horizon]
        for signal in ("material_signal", "distance_signal"):
            active = panel.dropna(subset=[signal, "raw_open_h1"])
            ic = cross_sectional_daily_ic(
                active[signal],
                active["raw_open_h1"],
                active["session_date"],
                min_names=3,
                hac_lags=5,
            )
            daily = build_sparse_daily_portfolio(
                panel,
                signal,
                cost_bps_per_side=cost_bps_per_side,
                min_names=MIN_PORTFOLIO_NAMES,
            )
            daily_outputs[(signal, horizon)] = daily
            summary = summarize_daily_portfolio(daily, config=cfg)
            midpoint = len(daily) // 2
            first = daily.iloc[:midpoint]
            second = daily.iloc[midpoint:]
            first_mean = float(first["gross_return"].mean()) if len(first) else math.nan
            second_mean = float(second["gross_return"].mean()) if len(second) else math.nan
            rows.append(
                {
                    "signal": signal,
                    "state_horizon_sessions": horizon,
                    "arm": f"{signal}_h{horizon}",
                    "active_firm_sessions": int(len(active)),
                    "active_sessions": int(active["session_date"].nunique()),
                    "mean_active_names": float(
                        active.groupby("session_date").size().mean()
                    ),
                    "ic": ic["ic"],
                    "ic_se": ic["se"],
                    "ic_t": ic["t"],
                    "ic_p": ic["p"],
                    "ic_n": ic["n"],
                    "ic_n_clusters": ic["n_clusters"],
                    "first_half_mean_gross": first_mean,
                    "second_half_mean_gross": second_mean,
                    "temporal_sign_stable_positive": bool(
                        first_mean > 0 and second_mean > 0
                    ),
                    **summary,
                }
            )

    results = pd.DataFrame(rows)
    p = results["ic_p"].fillna(1.0).to_numpy(dtype=float)
    results["ic_p_bh"] = _bh_adjusted_p(p)
    results["bh_reject_ic_q05"] = benjamini_hochberg(p.tolist(), q=0.05)
    results["economically_viable"] = (
        results["bh_reject_ic_q05"]
        & results["sharpe_net"].gt(0)
        & results["breakeven_bps_per_side"].ge(cost_bps_per_side)
        & results["bootstrap_net_ci_low"].gt(0)
        & results["temporal_sign_stable_positive"]
    )
    results["multiplicity_family"] = SPARSE_PRIMARY_FAMILY
    results["status"] = "retrospective_discovery_falsification_not_holdout"
    return results, daily_outputs


__all__ = [
    "BOOTSTRAP_BLOCK_SESSIONS",
    "BOOTSTRAP_REPLICATIONS",
    "COST_BPS_PER_SIDE",
    "HOLDING_HORIZONS",
    "MATERIAL_EVENT_TYPES",
    "MIN_PORTFOLIO_NAMES",
    "NOVELTY_JACCARD_THRESHOLD",
    "NOVELTY_WINDOW_DAYS",
    "SEED",
    "SPARSE_PRIMARY_FAMILY",
    "annotate_strict_prior_novelty",
    "build_sparse_daily_portfolio",
    "build_sparse_state_panels",
    "classify_proxy_cash_flow_distance",
    "evaluate_sparse_arms",
    "prepare_sparse_event_candidates",
    "select_strongest_sparse_events",
]
