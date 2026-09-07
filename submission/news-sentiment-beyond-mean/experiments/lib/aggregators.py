"""Firm-day aggregation rules for Workstream 3.

Nine rules, one firm-day grain. Same-day rules take story rows for a single
``(symbol, session_date)``; ``decayed_state`` is applied across a symbol's
news-bearing sessions after the within-day continuous mean is known.

Strongest-event ranking is adapted to the FNSPID checkpoint (no LLM
materiality/novelty labels): prefer non-recap, then ``|finbert_score|``;
opposite-direction ties at the top rank yield NaN.

Attention weighting uses ``mean_continuous * log(1 + n)``. Publisher-tier
weights are blocked on this spine; novelty-class weights are optional when
``novelty_class`` is present on story rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from experiments.lib.distribution import (
    N_BIN_ORDER,
    assign_n_bin,
    load_story_scores,
    polarity_label,
)
from experiments.lib.panel import DEFAULT_EVENTS_DB, FROZEN_DEV_END, FROZEN_EVAL_START

HARD_LABEL = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

AGGREGATOR_NAMES: tuple[str, ...] = (
    "mean_hard_label",
    "mean_continuous",
    "median_continuous",
    "trimmed_mean",
    "negative_share",
    "dispersion",
    "strongest_event",
    "attention_log_n",
    "decayed_state",
)

# Declared once for the comparison notebook / later evaluate harness.
PRIMARY_OUTCOME = "ar_open_h1"
PRIMARY_HORIZON = "h1_open_to_open"
SENSITIVITY_HORIZONS: tuple[str, ...] = ()  # only h1 on current panel
MULTIPLICITY_FAMILY = (
    "nine aggregators × primary horizon h1; "
    "n-bin strata are descriptive, not additional family members"
)
DECAY_HALF_LIFE_SESSIONS = 3.0
DECAY_SEVERITY_POWER = 1.5
DECAY_REVERSAL_RESET = 0.75
TRIM_PROPORTION = 0.10

AggregatorFn = Callable[[pd.DataFrame], float]


def _scores(story_rows: pd.DataFrame) -> np.ndarray:
    return story_rows["finbert_score"].to_numpy(dtype=float)


def _hard_labels(story_rows: pd.DataFrame) -> np.ndarray:
    if "polarity" not in story_rows.columns:
        polarity = polarity_label(story_rows)
    else:
        polarity = story_rows["polarity"]
    return polarity.map(HARD_LABEL).astype(float).to_numpy()


def mean_hard_label(story_rows: pd.DataFrame) -> float:
    """Mean of -1/0/+1 hard labels (incumbent production rule)."""
    if story_rows.empty:
        return float("nan")
    return float(np.mean(_hard_labels(story_rows)))


def mean_continuous(story_rows: pd.DataFrame) -> float:
    """Mean of continuous FinBERT score ``p_pos - p_neg``."""
    if story_rows.empty:
        return float("nan")
    return float(np.mean(_scores(story_rows)))


def median_continuous(story_rows: pd.DataFrame) -> float:
    if story_rows.empty:
        return float("nan")
    return float(np.median(_scores(story_rows)))


def trimmed_mean(story_rows: pd.DataFrame, proportion: float = TRIM_PROPORTION) -> float:
    """Symmetric trim; with small n the trim count is zero and this equals the mean."""
    if story_rows.empty:
        return float("nan")
    if not 0 <= proportion < 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    values = np.sort(_scores(story_rows))
    n = len(values)
    k = int(math.floor(proportion * n))
    if 2 * k >= n:
        return float(np.mean(values))
    return float(np.mean(values[k : n - k]))


def negative_share(story_rows: pd.DataFrame) -> float:
    if story_rows.empty:
        return float("nan")
    labels = _hard_labels(story_rows)
    return float(np.mean(labels < 0))


def soft_negative_mass(story_rows: pd.DataFrame) -> float:
    """Mean FinBERT negative-class probability; not a tenth family aggregator."""
    if story_rows.empty:
        return float("nan")
    if "p_negative" not in story_rows.columns:
        raise ValueError("soft_negative_mass requires p_negative")
    values = pd.to_numeric(story_rows["p_negative"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def firm_day_soft_negative_mass(stories: pd.DataFrame) -> pd.DataFrame:
    """Collapse story-level p_negative to one firm-day mean."""
    if stories.empty:
        return pd.DataFrame(columns=["symbol", "session_date", "soft_negative_mass"])
    required = {"symbol", "session_date", "p_negative"}
    if missing := required - set(stories.columns):
        raise ValueError(f"stories missing columns: {sorted(missing)}")
    frame = stories.loc[:, list(required)].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["p_negative"] = pd.to_numeric(frame["p_negative"], errors="coerce")
    grouped = (
        frame.groupby(["symbol", "session_date"], sort=False)["p_negative"]
        .mean()
        .rename("soft_negative_mass")
        .reset_index()
    )
    return grouped


def dispersion(story_rows: pd.DataFrame) -> float:
    """Same-day standard deviation (0 when n == 1)."""
    if story_rows.empty:
        return float("nan")
    values = _scores(story_rows)
    if len(values) == 1:
        return 0.0
    return float(np.std(values, ddof=1))


def strongest_event(story_rows: pd.DataFrame) -> float:
    """One story: prefer non-recap, then |score|; conflicting top-direction → NaN."""
    if story_rows.empty:
        return float("nan")
    frame = story_rows.copy()
    scores = frame["finbert_score"].astype(float)
    if "is_recap" in frame.columns:
        is_recap = frame["is_recap"].astype(bool)
    else:
        is_recap = pd.Series(False, index=frame.index)
    frame = frame.assign(
        _score=scores,
        _abs=scores.abs(),
        _non_recap=(~is_recap).astype(int),
        _sign=np.sign(scores.to_numpy()),
    )
    # Lexicographic: non-recap first, then |score|.
    best_non_recap = int(frame["_non_recap"].max())
    top = frame.loc[frame["_non_recap"] == best_non_recap]
    best_abs = float(top["_abs"].max())
    top = top.loc[np.isclose(top["_abs"], best_abs)]
    signs = set(int(s) for s in top["_sign"].tolist() if s != 0)
    if len(signs) > 1:
        return float("nan")
    # Stable pick among remaining ties.
    if "event_index" in top.columns:
        pick = top.sort_values(["event_index"]).iloc[0]
    else:
        pick = top.iloc[0]
    return float(pick["_score"])


def attention_log_n(story_rows: pd.DataFrame) -> float:
    """Volume attention: mean continuous × log(1 + n)."""
    if story_rows.empty:
        return float("nan")
    n = len(story_rows)
    return float(np.mean(_scores(story_rows)) * math.log1p(n))


def attention_novelty_weighted(story_rows: pd.DataFrame) -> float:
    """Optional: mean score weighted by novelty class (breaking > repetition > routine)."""
    if story_rows.empty:
        return float("nan")
    if "novelty_class" not in story_rows.columns:
        raise ValueError("novelty_class column required for attention_novelty_weighted")
    weights = {
        "breaking": 1.0,
        "repetition": 0.25,
        "routine": 0.0,
    }
    w = story_rows["novelty_class"].astype(str).map(weights).fillna(0.0).to_numpy(dtype=float)
    scores = _scores(story_rows)
    if float(w.sum()) <= 0:
        return float("nan")
    return float(np.average(scores, weights=w))


SAME_DAY_AGGREGATORS: dict[str, AggregatorFn] = {
    "mean_hard_label": mean_hard_label,
    "mean_continuous": mean_continuous,
    "median_continuous": median_continuous,
    "trimmed_mean": trimmed_mean,
    "negative_share": negative_share,
    "dispersion": dispersion,
    "strongest_event": strongest_event,
    "attention_log_n": attention_log_n,
}


def severity_impulse(score: float, severity_power: float = DECAY_SEVERITY_POWER) -> float:
    score = float(score)
    if not math.isfinite(score):
        return float("nan")
    score = max(-1.0, min(1.0, score))
    if score == 0.0:
        return 0.0
    return math.copysign(abs(score) ** severity_power, score)


def decayed_state_series(
    firm_day: pd.DataFrame,
    *,
    signal_col: str = "mean_continuous",
    half_life_sessions: float = DECAY_HALF_LIFE_SESSIONS,
    severity_power: float = DECAY_SEVERITY_POWER,
    reversal_reset: float = DECAY_REVERSAL_RESET,
    session_ordinals: Mapping[Any, int] | None = None,
) -> pd.Series:
    """Half-life decay with severity impulse and opposite-sign reset.

    Computed on news-bearing firm-days only. Elapsed sessions use
    ``session_ordinals`` when provided (preferred), else the per-symbol
    consecutive-news gap counted as 1.
    """
    if firm_day.empty:
        return pd.Series(dtype=float)

    frame = firm_day.sort_values(["symbol", "session_date"], kind="mergesort").copy()
    out = np.full(len(frame), np.nan, dtype=float)
    symbols = frame["symbol"].astype(str).tolist()
    dates = [pd.Timestamp(ts).normalize() for ts in pd.to_datetime(frame["session_date"])]
    impulses_raw = frame[signal_col].astype(float).tolist()

    prev_symbol: str | None = None
    state = 0.0
    prev_ordinal: int | None = None

    for i, (symbol, session, raw) in enumerate(zip(symbols, dates, impulses_raw, strict=True)):
        if symbol != prev_symbol:
            state = 0.0
            prev_ordinal = None
            prev_symbol = symbol

        if session_ordinals is not None:
            ordinal = int(session_ordinals[session])
            elapsed = 0 if prev_ordinal is None else ordinal - prev_ordinal
        else:
            elapsed = 0 if prev_ordinal is None else 1
            ordinal = (prev_ordinal or 0) + elapsed

        state = state * (2 ** (-elapsed / half_life_sessions))
        impulse = severity_impulse(raw, severity_power)
        if not math.isfinite(impulse):
            out[i] = state
            prev_ordinal = ordinal
            continue
        if state != 0.0 and impulse != 0.0 and math.copysign(1.0, state) != math.copysign(1.0, impulse):
            state *= 1.0 - reversal_reset
        state = state + impulse
        out[i] = state
        prev_ordinal = ordinal

    return pd.Series(out, index=frame.index, name="decayed_state")


def _vectorized_same_day(stories: pd.DataFrame) -> pd.DataFrame:
    """Fast path for all same-day rules via groupby."""
    frame = stories.copy()
    if "polarity" not in frame.columns:
        frame["polarity"] = polarity_label(frame)
    frame["hard_label"] = frame["polarity"].map(HARD_LABEL).astype(float)
    frame["is_recap"] = frame["is_recap"].astype(bool)

    g = frame.groupby(["symbol", "session_date"], sort=False)
    base = g.agg(
        n=("finbert_score", "size"),
        mean_continuous=("finbert_score", "mean"),
        median_continuous=("finbert_score", "median"),
        dispersion=("finbert_score", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
        mean_hard_label=("hard_label", "mean"),
        negative_share=("hard_label", lambda s: float((s < 0).mean())),
    )
    base["attention_log_n"] = base["mean_continuous"] * np.log1p(base["n"].astype(float))

    # Trimmed mean per group.
    def _trim(s: pd.Series) -> float:
        return trimmed_mean(pd.DataFrame({"finbert_score": s}))

    base["trimmed_mean"] = g["finbert_score"].apply(_trim)

    # Strongest event per group.
    def _strong(group: pd.DataFrame) -> float:
        return strongest_event(group)

    base["strongest_event"] = g.apply(_strong, include_groups=False)
    return base.reset_index()


def build_session_ordinals(session_dates: Sequence[Any]) -> dict[pd.Timestamp, int]:
    unique = sorted({pd.Timestamp(ts).normalize() for ts in session_dates})
    return {ts: i for i, ts in enumerate(unique)}


def build_aggregation_panel(
    panel: pd.DataFrame,
    *,
    events_db=DEFAULT_EVENTS_DB,
    stories: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join all nine aggregator signals onto the firm-day panel."""
    keys = panel[["symbol", "session_date"]].drop_duplicates()
    if stories is None:
        stories = load_story_scores(events_db, keys)
    same_day = _vectorized_same_day(stories)

    ordinals = build_session_ordinals(panel["session_date"])
    same_day = same_day.sort_values(["symbol", "session_date"], kind="mergesort")
    same_day["decayed_state"] = decayed_state_series(
        same_day, signal_col="mean_continuous", session_ordinals=ordinals
    ).to_numpy()

    same_day["n_bin"] = assign_n_bin(same_day["n"])
    keep_panel = [
        c
        for c in (
            "symbol",
            "session_date",
            "split",
            "ar_open_h1",
            "ret_open_h1",
            "spy_ret_open_h1",
            "article_count",
            "is_earnings_session",
        )
        if c in panel.columns
    ]
    out = panel[keep_panel].merge(same_day, on=["symbol", "session_date"], how="inner")
    if "split" not in out.columns:
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


def spearmans_ic(signal: pd.Series, outcome: pd.Series) -> float:
    frame = pd.DataFrame({"x": signal, "y": outcome}).dropna()
    if len(frame) < 3:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method="spearman"))


def cross_sectional_daily_ic(
    signal: pd.Series,
    outcome: pd.Series,
    dates: pd.Series,
    *,
    min_names: int = 3,
    hac_lags: int = 5,
) -> dict[str, float | int | str]:
    """Mean daily cross-sectional Spearman IC with HAC inference over dates.

    IC is first computed independently within each session. The estimand is the
    time-series mean of those daily correlations; Newey-West/HAC standard errors
    allow the daily IC series to be serially dependent. This avoids the market-
    regime confounding in a global pooled rank correlation.
    """
    import statsmodels.api as sm

    frame = pd.DataFrame(
        {
            "x": signal.to_numpy(),
            "y": outcome.to_numpy(),
            "date": pd.to_datetime(dates).to_numpy(),
        }
    ).dropna()
    n = len(frame)
    daily_rows: list[tuple[pd.Timestamp, float, int]] = []
    for session, day in frame.groupby("date", sort=True):
        if len(day) < min_names or day["x"].nunique() < 2 or day["y"].nunique() < 2:
            continue
        ic = float(day["x"].corr(day["y"], method="spearman"))
        if math.isfinite(ic):
            daily_rows.append((pd.Timestamp(session), ic, int(len(day))))

    daily = pd.DataFrame(daily_rows, columns=["date", "daily_ic", "n_names"])
    base: dict[str, float | int | str] = {
        "n": n,
        "n_clusters": int(len(daily)),
        "n_dates_total": int(frame["date"].nunique()) if n else 0,
        "min_names": min_names,
        "hac_lags": hac_lags,
        "inference": "mean_daily_cross_sectional_spearman_hac",
    }
    if len(daily) < 2:
        return {
            **base,
            "ic": float("nan"),
            "se": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
        }
    model = sm.OLS(daily["daily_ic"], np.ones((len(daily), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": min(hac_lags, len(daily) - 1)},
    )
    return {
        **base,
        "ic": float(model.params.iloc[0]),
        "se": float(model.bse.iloc[0]),
        "t": float(model.tvalues.iloc[0]),
        "p": float(model.pvalues.iloc[0]),
    }


def date_clustered_rank_ic(
    signal: pd.Series,
    outcome: pd.Series,
    dates: pd.Series,
) -> dict[str, float | int | str]:
    """Compatibility alias for the corrected daily cross-sectional IC."""
    return cross_sectional_daily_ic(signal, outcome, dates)


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> list[bool]:
    """Return reject flags for BH-FDR at level q."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    ranked = np.asarray(p_values, dtype=float)[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        max_i = int(np.max(np.where(below)[0]))
        reject[order[: max_i + 1]] = True
    return reject.tolist()


def benjamini_hochberg_q_values(p_values: Sequence[float]) -> list[float]:
    """BH q-values; monotonic in the sorted p-value order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    q_raw = ranked * m / np.arange(1, m + 1)
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_mono = np.clip(q_mono, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = q_mono
    return out.tolist()


def development_ic_table(
    agg_panel: pd.DataFrame,
    *,
    outcome: str = PRIMARY_OUTCOME,
    aggregators: Sequence[str] = AGGREGATOR_NAMES,
    by_n_bin: bool = True,
    fdr_q: float = 0.05,
) -> pd.DataFrame:
    """Mean daily cross-sectional Spearman IC on development only."""
    dev = agg_panel.loc[agg_panel["split"] == "development"].copy()
    rows: list[dict[str, Any]] = []

    strata: list[tuple[str, pd.DataFrame]] = [("all", dev)]
    if by_n_bin:
        for label in N_BIN_ORDER:
            strata.append((label, dev.loc[dev["n_bin"] == label]))

    for stratum, frame in strata:
        for name in aggregators:
            if name not in frame.columns:
                continue
            stats = date_clustered_rank_ic(
                frame[name], frame[outcome], frame["session_date"]
            )
            rows.append(
                {
                    "stratum": stratum,
                    "aggregator": name,
                    "outcome": outcome,
                    **stats,
                }
            )

    table = pd.DataFrame(rows)
    # BH only on the primary all-n family (nine rules × one horizon).
    primary = table["stratum"] == "all"
    reject = pd.Series(False, index=table.index)
    if primary.any():
        flags = benjamini_hochberg(table.loc[primary, "p"].tolist(), q=fdr_q)
        reject.loc[table.index[primary]] = flags
    table["bh_reject_q05"] = reject
    table["bh_family"] = np.where(primary, MULTIPLICITY_FAMILY, "descriptive_n_bin_stratum")
    return table
