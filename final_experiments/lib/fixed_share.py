"""Fixed-share long-basket accounting for sparse LSEG risk translations.

The base basket buys initially equal-dollar fixed shares. Sparse event rules
may temporarily cut selected holdings, leave the released notional in cash, or
transfer it to selected positive-event holdings. The simulator keeps cash and
shares explicitly, charges costs on traded notional, and liquidates after the
last open-to-open interval.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR = 252


def _validate_open_wide(open_wide: pd.DataFrame) -> pd.DataFrame:
    frame = open_wide.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    if frame.index.duplicated().any() or frame.columns.duplicated().any():
        raise ValueError("open prices must be unique by session and symbol")
    frame = frame.sort_index().sort_index(axis="columns")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("open prices must be finite and positive")
    return frame


def equal_dollar_base_shares(
    open_wide: pd.DataFrame,
    first_session: pd.Timestamp | str,
    *,
    entry_cost_bps: float,
) -> pd.Series:
    """Return fixed shares that reserve enough cash for initial transaction cost."""
    if entry_cost_bps < 0:
        raise ValueError("entry_cost_bps must be non-negative")
    frame = _validate_open_wide(open_wide)
    session = pd.Timestamp(first_session).normalize()
    if session not in frame.index:
        raise ValueError("first_session is absent from open prices")
    funded_notional = 1.0 / (1.0 + entry_cost_bps / 10_000.0)
    return (funded_notional / len(frame.columns) / frame.loc[session]).rename("base_shares")


def build_sparse_extreme_targets(
    open_wide: pd.DataFrame,
    sessions: pd.DatetimeIndex | list[pd.Timestamp],
    negative_flags: pd.DataFrame,
    positive_flags: pd.DataFrame,
    *,
    risk_fraction: float,
    entry_cost_bps: float,
    reallocate_to_positive: bool,
) -> pd.DataFrame:
    """Build fixed-share targets for a sparse negative brake or extreme rotation.

    Negative-event holdings are multiplied by ``risk_fraction``. When
    ``reallocate_to_positive`` is true and at least one positive event exists
    on the same session, the released current-open notional is divided equally
    across positive-event names. Otherwise the released value remains cash.
    """
    if not 0 <= risk_fraction <= 1:
        raise ValueError("risk_fraction must lie in [0, 1]")
    frame = _validate_open_wide(open_wide)
    index = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize()
    if index.empty or index.duplicated().any() or not index.is_monotonic_increasing:
        raise ValueError("sessions must be non-empty, unique, and increasing")
    missing_sessions = index.difference(frame.index)
    if len(missing_sessions):
        raise ValueError("target sessions are absent from open prices")

    def flags(value: pd.DataFrame, label: str) -> pd.DataFrame:
        use = value.copy()
        use.index = pd.DatetimeIndex(pd.to_datetime(use.index)).normalize()
        use = use.reindex(index=index, columns=frame.columns)
        if use.isna().any().any():
            raise ValueError(f"{label} flags must cover every target session and symbol")
        return use.astype(bool)

    negative = flags(negative_flags, "negative")
    positive = flags(positive_flags, "positive")
    if (negative & positive).any().any():
        raise ValueError("negative and positive flags may not overlap")

    base = equal_dollar_base_shares(
        frame,
        index[0],
        entry_cost_bps=entry_cost_bps,
    )
    target = pd.DataFrame(
        np.repeat(base.to_numpy(dtype=float)[None, :], len(index), axis=0),
        index=index,
        columns=frame.columns,
    )
    for session in index:
        neg = negative.loc[session].to_numpy(dtype=bool)
        if not neg.any():
            continue
        price = frame.loc[session].to_numpy(dtype=float)
        base_values = base.to_numpy(dtype=float) * price
        released = float(((1.0 - risk_fraction) * base_values[neg]).sum())
        target.loc[session, neg] = base.to_numpy(dtype=float)[neg] * risk_fraction
        pos = positive.loc[session].to_numpy(dtype=bool)
        if reallocate_to_positive and pos.any() and released > 0:
            added_notional = released / int(pos.sum())
            target.loc[session, pos] += added_notional / price[pos]
    return target


def simulate_fixed_share_targets(
    open_wide: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    """Simulate target shares with cash, costs, and final liquidation."""
    if cost_bps_per_side < 0:
        raise ValueError("cost_bps_per_side must be non-negative")
    frame = _validate_open_wide(open_wide)
    target = targets.copy()
    target.index = pd.DatetimeIndex(pd.to_datetime(target.index)).normalize()
    target = target.reindex(columns=frame.columns)
    values = target.to_numpy(dtype=float)
    if target.empty or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("targets must be non-empty, finite, and long-only")

    sessions = pd.DatetimeIndex(target.index)
    locations = frame.index.get_indexer(sessions)
    if (locations < 0).any() or (locations + 1 >= len(frame.index)).any():
        raise ValueError("every target session must have a next price session")
    if len(sessions) > 1 and not np.array_equal(locations[1:], locations[:-1] + 1):
        raise ValueError("target sessions must be consecutive price sessions")

    rate = cost_bps_per_side / 10_000.0
    cash = 1.0
    holdings = np.zeros(target.shape[1], dtype=float)
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for offset, (session, location) in enumerate(zip(sessions, locations, strict=True)):
        start_price = frame.loc[session].to_numpy(dtype=float)
        end_session = frame.index[location + 1]
        end_price = frame.loc[end_session].to_numpy(dtype=float)
        wealth_start = float(cash + np.dot(holdings, start_price))
        desired = target.loc[session].to_numpy(dtype=float)
        delta = desired - holdings
        traded_notional = float(np.abs(delta * start_price).sum())
        entry_cost = rate * traded_notional
        cash -= float(np.dot(delta, start_price)) + entry_cost
        holdings = desired
        post_trade_wealth = float(cash + np.dot(holdings, start_price))
        security_notional = float(np.abs(holdings * start_price).sum())

        liquidation_notional = 0.0
        liquidation_cost = 0.0
        if offset == len(sessions) - 1:
            liquidation_notional = float(np.abs(holdings * end_price).sum())
            liquidation_cost = rate * liquidation_notional
            cash += float(np.dot(holdings, end_price)) - liquidation_cost
            holdings = np.zeros(target.shape[1], dtype=float)
            wealth_end = float(cash)
        else:
            wealth_end = float(cash + np.dot(holdings, end_price))

        total_notional = traded_notional + liquidation_notional
        total_cost = entry_cost + liquidation_cost
        rows.append(
            {
                "session_date": session,
                "return_end_date": end_session,
                "net_return": wealth_end / wealth_start - 1.0,
                "wealth_start": wealth_start,
                "wealth_end": wealth_end,
                "turnover": total_notional / wealth_start,
                "cost_return": total_cost / wealth_start,
                "gross_exposure": security_notional / post_trade_wealth,
                "active_positions": int(np.sum(np.abs(desired) > 1e-15)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_fixed_share_targets(
    open_wide: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    """Return one daily frame containing zero-cost gross and charged-cost net paths."""
    gross = simulate_fixed_share_targets(open_wide, targets, cost_bps_per_side=0.0)
    net = simulate_fixed_share_targets(
        open_wide,
        targets,
        cost_bps_per_side=cost_bps_per_side,
    )
    out = net.copy()
    out["gross_return"] = gross["net_return"].to_numpy(dtype=float)
    out["downside_sq"] = np.minimum(out["net_return"], 0.0) ** 2
    return out


def summarize_fixed_share_path(daily: pd.DataFrame) -> dict[str, Any]:
    """Summarise one evaluated fixed-share path."""
    need = {"gross_return", "net_return", "turnover", "gross_exposure"}
    if missing := need - set(daily.columns):
        raise ValueError(f"daily path missing columns: {sorted(missing)}")
    if daily.empty:
        return {"n_sessions": 0}
    gross = daily["gross_return"].to_numpy(dtype=float)
    net = daily["net_return"].to_numpy(dtype=float)
    gross_equity = np.cumprod(1.0 + gross)
    net_equity = np.cumprod(1.0 + net)
    with_start = np.r_[1.0, net_equity]
    peak = np.maximum.accumulate(with_start)

    def sharpe(values: np.ndarray) -> float:
        std = float(np.std(values, ddof=1))
        return (
            float(math.sqrt(TRADING_SESSIONS_PER_YEAR) * np.mean(values) / std)
            if std > 0
            else float("nan")
        )

    mean_turnover = float(daily["turnover"].mean())
    return {
        "n_sessions": int(len(daily)),
        "mean_gross": float(np.mean(gross)),
        "mean_net": float(np.mean(net)),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "ann_vol_net": float(np.std(net, ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)),
        "total_return_gross": float(gross_equity[-1] - 1.0),
        "total_return_net": float(net_equity[-1] - 1.0),
        "max_drawdown": float(np.min(with_start / peak - 1.0)),
        "mean_turnover": mean_turnover,
        "total_turnover": float(daily["turnover"].sum()),
        "mean_gross_exposure": float(daily["gross_exposure"].mean()),
        "breakeven_bps_per_side": (
            float(10_000.0 * np.mean(gross) / mean_turnover)
            if mean_turnover > 0
            else None
        ),
    }
