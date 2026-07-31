"""Earnings-window mapping and interaction inference for Workstream 6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from final_experiments.lib.aggregators import benjamini_hochberg

WINDOW = 5
INTERACTION_FAMILY = "pre/event/post interactions vs outside x h1; BH-FDR q=0.05"


def map_earnings_sessions(
    calendar: pd.DataFrame,
    sessions: pd.Series | pd.Index,
) -> pd.DataFrame:
    """Map report dates to XNYS sessions using the calendar's frozen rule."""
    session_index = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize().unique().sort_values()
    cal = calendar.copy()
    cal["report_date"] = pd.to_datetime(cal["report_date"]).dt.normalize()
    mapped: list[pd.Timestamp | pd.NaT] = []
    for row in cal.itertuples(index=False):
        report_date = pd.Timestamp(row.report_date)
        side = "right" if str(row.session_rule) == "next_session" else "left"
        idx = int(session_index.searchsorted(report_date, side=side))
        mapped.append(session_index[idx] if idx < len(session_index) else pd.NaT)
    cal["earnings_session"] = mapped
    cal = cal.dropna(subset=["earnings_session"])
    cal = cal.sort_values(["symbol", "earnings_session", "report_date"], kind="mergesort")
    return cal.drop_duplicates(["symbol", "earnings_session"], keep="first")


def attach_earnings_distance(
    firm_day: pd.DataFrame,
    mapped_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Attach signed distance to the nearest earnings event in exchange sessions."""
    out = firm_day.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.normalize()
    sessions = pd.DatetimeIndex(out["session_date"].unique()).sort_values()
    ordinal = pd.Series(np.arange(len(sessions)), index=sessions)
    out["_session_ordinal"] = out["session_date"].map(ordinal)
    cal = mapped_calendar.copy()
    cal["_earnings_ordinal"] = pd.to_datetime(cal["earnings_session"]).map(ordinal)
    cal = cal.dropna(subset=["_earnings_ordinal"])

    pieces: list[pd.DataFrame] = []
    cal_groups = {symbol: group for symbol, group in cal.groupby("symbol", sort=False)}
    for symbol, rows in out.groupby("symbol", sort=False):
        rows = rows.copy()
        events = cal_groups.get(symbol)
        if events is None or events.empty:
            rows["sessions_to_earnings"] = np.nan
            rows["nearest_earnings_session"] = pd.NaT
            rows["nearest_earnings_timing"] = pd.NA
            pieces.append(rows)
            continue
        event_ord = events["_earnings_ordinal"].astype(int).to_numpy()
        row_ord = rows["_session_ordinal"].astype(int).to_numpy()
        pos = np.searchsorted(event_ord, row_ord)
        left = np.clip(pos - 1, 0, len(event_ord) - 1)
        right = np.clip(pos, 0, len(event_ord) - 1)
        left_dist = event_ord[left] - row_ord
        right_dist = event_ord[right] - row_ord
        # On ties choose the upcoming event, which keeps a symmetric pre-window.
        choose_right = np.abs(right_dist) <= np.abs(left_dist)
        chosen = np.where(choose_right, right, left)
        # Conventional signed event time: negative before, zero on, positive after.
        rows["sessions_to_earnings"] = row_ord - event_ord[chosen]
        rows["nearest_earnings_session"] = events.iloc[chosen]["earnings_session"].to_numpy()
        rows["nearest_earnings_timing"] = events.iloc[chosen]["timing_flag"].to_numpy()
        pieces.append(rows)

    result = pd.concat(pieces, ignore_index=True).drop(columns=["_session_ordinal"])
    distance = result["sessions_to_earnings"]
    result["earnings_window"] = np.select(
        [
            distance.between(-WINDOW, -1),
            distance.eq(0),
            distance.between(1, WINDOW),
        ],
        ["pre", "event", "post"],
        default="outside",
    )
    result.loc[distance.isna(), "earnings_window"] = "outside"
    result["inside_earnings_window"] = result["earnings_window"].ne("outside")
    return result


def earnings_interaction_table(
    frame: pd.DataFrame,
    *,
    signal_col: str = "negative_share",
    outcome_col: str = "ar_open_h1",
) -> pd.DataFrame:
    """Pooled interaction model, with outside-window slope as the reference."""
    use = frame[[signal_col, outcome_col, "earnings_window", "session_date"]].dropna().copy()
    signal = -use[signal_col].astype(float).to_numpy()
    x_parts = [np.ones(len(use)), signal]
    names = ["intercept", "slope_outside"]
    for window in ("pre", "event", "post"):
        flag = use["earnings_window"].eq(window).astype(float).to_numpy()
        x_parts.extend([flag, flag * signal])
        names.extend([f"intercept__{window}", f"slope_delta__{window}"])
    x = np.column_stack(x_parts)
    model = sm.OLS(use[outcome_col].astype(float).to_numpy(), x).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(use["session_date"])[0]},
    )
    rows = []
    for window in ("pre", "event", "post"):
        j = names.index(f"slope_delta__{window}")
        rows.append(
            {
                "window": window,
                "slope_delta_vs_outside": float(model.params[j]),
                "se": float(model.bse[j]),
                "t": float(model.tvalues[j]),
                "p": float(model.pvalues[j]),
                "n_firm_days": int((use["earnings_window"] == window).sum()),
                "n_dates": int(use.loc[use["earnings_window"] == window, "session_date"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out["bh_reject_q05"] = benjamini_hochberg(out["p"].fillna(1.0).tolist())
    out["multiplicity_family"] = INTERACTION_FAMILY
    out["cluster"] = "session_date"
    return out


def read_quarterly_calendar(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


__all__ = [
    "INTERACTION_FAMILY",
    "WINDOW",
    "attach_earnings_distance",
    "earnings_interaction_table",
    "map_earnings_sessions",
    "read_quarterly_calendar",
]
