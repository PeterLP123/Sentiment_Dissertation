"""Trading-strategy backtest for firm-day sentiment signals (Workstream 3b).

This is the step past ``evaluate.py``. ``evaluate.py`` answers "does the signal
rank next-session returns"; this answers "what happens to an account that trades
it" — equity curve, drawdown, per-year behaviour, and the cost at which the whole
thing dies.

Design decisions that matter, all made once here:

**Overlapping tranches.** A horizon-h strategy runs h books side by side, each
holding 1/h of gross exposure, one opened per session. The weight actually held
on session t is the mean of the last h formation books. With h=1 this reduces to
the one-session construction in ``evaluate.py``; with h=5 it trades roughly a
fifth as much, which is the entire point of testing longer holds. A name with
news on consecutive days appears in several tranches and its weight adds — that
is intended, not a bug: repeated news is repeated conviction.

**Dense daily returns.** Positions are held through sessions on which a name has
no news, so returns come from a dense (symbol × session) market-adjusted panel
rather than from the news-bearing panel. Held weight whose return is missing that
session is reported as ``missing_weight_share`` instead of being silently
treated as zero.

**Selection discipline.** ``sweep_development`` only ever sees development rows.
``run_frozen_spec`` is the single evaluation call and takes a spec dict that was
written to disk first. Nothing here picks a winner using evaluation outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.evaluate import block_bootstrap_mean
from final_experiments.lib.panel import FROZEN_DEV_END, FROZEN_EVAL_START

#: Holding horizons in sessions. h1 reproduces the existing one-session arm.
DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10)

#: Breadth bands on the within-day centred rank. 0.0 trades every name with news;
#: 0.5 keeps roughly the top and bottom quartile; 0.8 roughly the top and bottom
#: decile. Concentrating breadth is the other lever besides horizon.
DEFAULT_BREADTHS: tuple[float, ...] = (0.0, 0.5, 0.8)

TRADING_SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class StrategyConfig:
    """One fully specified strategy. Everything needed to re-run it."""

    signal: str
    horizon: int = 1
    breadth: float = 0.0
    orient: float = 1.0
    cost_bps_per_side: float = 10.0
    min_names: int = 5
    bootstrap_block_length: int = 10
    bootstrap_replications: int = 999
    random_seed: int = 20260731

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least one session")
        if not 0.0 <= self.breadth < 1.0:
            raise ValueError("breadth must be in [0, 1)")
        if self.cost_bps_per_side < 0:
            raise ValueError("cost_bps_per_side must be non-negative")
        if self.min_names < 1:
            raise ValueError("min_names must be positive")

    def label(self) -> str:
        return f"{self.signal}|h{self.horizon}|b{self.breadth:g}"


@dataclass
class BacktestResult:
    config: StrategyConfig
    daily: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


def build_daily_ar_panel(
    prices: pd.DataFrame,
    *,
    market_symbol: str = "SPY",
) -> pd.DataFrame:
    """Dense market-adjusted open-to-open return for every symbol-session.

    ``ar = (open_{t+1}/open_t - 1) - (mkt_{t+1}/mkt_t - 1)``, indexed at t so it
    is the return earned by a position opened at session t's open. Same
    convention as the panel's ``ar_open_h1``, extended to sessions with no news.
    """
    frame = prices.sort_values(["symbol", "session_date"], kind="mergesort").copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    calendar = (
        frame.loc[frame["symbol"] == market_symbol.upper(), ["session_date"]]
        .drop_duplicates()
        .sort_values("session_date")
    )
    calendar["return_end_date"] = calendar["session_date"].shift(-1)
    dense = frame.merge(calendar, on="session_date", how="left", validate="m:1")
    lead = frame.rename(
        columns={"session_date": "return_end_date", "adjusted_open": "next_open"}
    )
    dense = dense.merge(
        lead[["symbol", "return_end_date", "next_open"]],
        on=["symbol", "return_end_date"],
        how="left",
        validate="m:1",
    )
    dense["fwd_ret"] = dense["next_open"] / dense["adjusted_open"] - 1.0

    market = dense.loc[
        dense["symbol"] == market_symbol.upper(),
        ["session_date", "return_end_date", "fwd_ret"],
    ]
    market = market.rename(columns={"fwd_ret": "fwd_mkt"}).drop_duplicates("session_date")

    stock = dense.loc[
        dense["symbol"] != market_symbol.upper(),
        ["symbol", "session_date", "return_end_date", "fwd_ret"],
    ]
    out = stock.merge(
        market,
        on=["session_date", "return_end_date"],
        how="inner",
        validate="m:1",
    )
    out["ar"] = out["fwd_ret"] - out["fwd_mkt"]
    return out.dropna(subset=["ar"])[
        ["symbol", "session_date", "return_end_date", "ar"]
    ]


def _formation_matrix(
    firm_day: pd.DataFrame,
    config: StrategyConfig,
    sessions: pd.DatetimeIndex,
    symbol_index: dict[str, int],
) -> np.ndarray:
    """(T x N) target book per formation session, gross exposure 1 (or 0).

    Vectorised: the sweep builds this 27 times (signal x breadth) rather than
    once per grid cell, and a Python loop over 2,264 sessions x ~225 names would
    dominate the whole run.
    """
    session_index = {ts: i for i, ts in enumerate(sessions)}
    book = np.zeros((len(sessions), len(symbol_index)), dtype=float)

    frame = firm_day.dropna(subset=[config.signal]).copy()
    frame["_row"] = frame["session_date"].map(session_index)
    frame["_col"] = frame["symbol"].astype(str).str.upper().map(symbol_index)
    frame = frame.dropna(subset=["_row", "_col"])
    if frame.empty:
        return book

    oriented = config.orient * frame[config.signal].to_numpy(dtype=float)
    frame["_oriented"] = oriented
    grouped = frame.groupby("_row", sort=False)["_oriented"]
    rank = grouped.rank(method="average").to_numpy(dtype=float)
    size = grouped.transform("size").to_numpy(dtype=float)
    # Same centred rank as cross_sectional_rank_scores; a one-name session is 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(size > 1, 2.0 * (rank - 1.0) / (size - 1.0) - 1.0, 0.0)
    z = np.where(np.abs(z) <= config.breadth, 0.0, z)
    frame["_z"] = z

    # Normalise each leg to 0.5 gross *separately* rather than scaling the whole
    # book by its total gross. Signals with a large tie block — dispersion is
    # exactly zero on every n=1 firm-day, half the panel — otherwise push the
    # entire tie to one side of the centred rank, and a breadth cut then keeps
    # only the other side. That yields a one-sided book masquerading as a
    # cross-sectional strategy, which is the same defect the `sign` position
    # mode had. With per-leg normalisation a day whose surviving names are all
    # long or all short has an empty leg and is simply not traded.
    frame["_leg"] = np.sign(frame["_z"].to_numpy())
    leg_gross = frame.groupby(["_row", "_leg"], sort=False)["_z"].transform(
        lambda s: s.abs().sum()
    )
    n_long = frame.groupby("_row", sort=False)["_leg"].transform(lambda s: (s > 0).sum())
    n_short = frame.groupby("_row", sort=False)["_leg"].transform(lambda s: (s < 0).sum())
    active = n_long + n_short

    keep = (
        (active.to_numpy() >= config.min_names)
        & (n_long.to_numpy() > 0)
        & (n_short.to_numpy() > 0)
        & (leg_gross.to_numpy() > 0)
        & (frame["_leg"].to_numpy() != 0)
    )
    if not keep.any():
        return book

    weights = np.zeros(len(frame), dtype=float)
    weights[keep] = 0.5 * frame["_z"].to_numpy()[keep] / leg_gross.to_numpy()[keep]
    rows = frame["_row"].to_numpy(dtype=int)
    cols = frame["_col"].to_numpy(dtype=int)
    # `add.at` so a symbol appearing twice on a session accumulates rather than
    # overwriting.
    np.add.at(book, (rows, cols), weights)
    return book


def _tranche_weights(book: np.ndarray, horizon: int) -> np.ndarray:
    """Mean of the last ``horizon`` formation books, per session."""
    if horizon == 1:
        return book
    padded = np.vstack([np.zeros((horizon - 1, book.shape[1])), book])
    cumulative = np.cumsum(padded, axis=0)
    cumulative = np.vstack([np.zeros((1, book.shape[1])), cumulative])
    # Rolling sum of `horizon` rows ending at each original row.
    rolled = cumulative[horizon:] - cumulative[:-horizon]
    return rolled / float(horizon)


@dataclass
class BacktestGrid:
    """Shared session x symbol scaffolding, built once and reused by the sweep."""

    frame: pd.DataFrame
    sessions: pd.DatetimeIndex
    symbol_index: dict[str, int]
    returns: np.ndarray


def build_grid(
    firm_day: pd.DataFrame,
    daily_ar: pd.DataFrame,
    *,
    split: str | None = None,
    tail_horizon: int = 1,
) -> BacktestGrid | None:
    """Align signal rows and dense returns onto one tradable calendar."""
    frame = firm_day.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if split is not None:
        if "split" not in frame.columns:
            raise ValueError("split column required when split filter is set")
        frame = frame.loc[frame["split"] == split]
    if frame.empty:
        return None

    ar = daily_ar.copy()
    ar["session_date"] = pd.to_datetime(ar["session_date"]).dt.normalize()
    if "return_end_date" not in ar.columns:
        raise ValueError("daily_ar must carry return_end_date for split-boundary validation")
    ar["return_end_date"] = pd.to_datetime(ar["return_end_date"]).dt.normalize()
    ar["symbol"] = ar["symbol"].astype(str).str.upper()

    if split == "development":
        boundary = pd.Timestamp(FROZEN_DEV_END)
        ar = ar.loc[
            (ar["session_date"] <= boundary) & (ar["return_end_date"] <= boundary)
        ]
    elif split == "evaluation":
        ar = ar.loc[ar["session_date"] >= pd.Timestamp(FROZEN_EVAL_START)]

    # A session is tradable only if a forward return exists for it, so the
    # calendar comes from the return panel. The last price session has no
    # forward return by construction and is therefore not a trading day: a book
    # formed there could never be marked, and counting it would let the final
    # formation contribute weight with no P&L.
    lo = frame["session_date"].min()
    in_window = pd.DatetimeIndex(sorted(set(ar["session_date"])))
    in_window = in_window[in_window >= lo]
    if len(in_window) < tail_horizon:
        return None
    # Keep enough returns after the last formation to mark every tranche fully.
    # For T return sessions, an h-session formation may start no later than T-h.
    last_formation = in_window[len(in_window) - tail_horizon]
    frame = frame.loc[frame["session_date"] <= last_formation]
    sessions = in_window
    # Formation outside the tradable calendar cannot be acted on.
    keep = set(sessions)
    frame = frame.loc[frame["session_date"].isin(keep)]
    if frame.empty:
        return None
    ar = ar.loc[ar["session_date"].isin(keep)]

    symbols = sorted(set(ar["symbol"]) | set(frame["symbol"]))
    symbol_index = {s: i for i, s in enumerate(symbols)}

    returns = np.full((len(sessions), len(symbols)), np.nan, dtype=float)
    session_pos = {ts: i for i, ts in enumerate(sessions)}
    rows = ar["session_date"].map(session_pos).to_numpy()
    cols = ar["symbol"].map(symbol_index).to_numpy()
    returns[rows.astype(int), cols.astype(int)] = ar["ar"].to_numpy(dtype=float)
    return BacktestGrid(frame=frame, sessions=sessions, symbol_index=symbol_index, returns=returns)


def backtest_on_grid(
    grid: BacktestGrid,
    config: StrategyConfig,
    *,
    book: np.ndarray | None = None,
) -> BacktestResult:
    """Backtest one config against a prebuilt grid, optionally reusing a book."""
    if book is None:
        book = _formation_matrix(grid.frame, config, grid.sessions, grid.symbol_index)
    sessions = grid.sessions
    returns = grid.returns
    weights = _tranche_weights(book, config.horizon)

    present = np.isfinite(returns)
    filled = np.where(present, returns, 0.0)
    gross = (weights * filled).sum(axis=1)
    held = np.abs(weights).sum(axis=1)
    missing_weight = np.abs(np.where(present, 0.0, weights)).sum(axis=1)

    previous = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    turnover = 0.5 * np.abs(weights - previous).sum(axis=1)
    # Liquidate the final held book at the last interval's ending open. Without
    # this, the reported backtest receives its last return but never pays to
    # close the position.
    if len(turnover):
        turnover[-1] += 0.5 * np.abs(weights[-1]).sum()
    cost = 2.0 * turnover * (config.cost_bps_per_side / 10_000.0)

    daily = pd.DataFrame(
        {
            "session_date": sessions,
            "gross_exposure": held,
            "net_exposure": weights.sum(axis=1),
            "n_positions": (np.abs(weights) > 1e-12).sum(axis=1),
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
            "missing_weight_share": np.divide(
                missing_weight, held, out=np.zeros_like(held), where=held > 0
            ),
        }
    )
    # Drop the flat lead-in before the first position is ever opened.
    active = daily["gross_exposure"] > 0
    if active.any():
        daily = daily.loc[active.idxmax() :].reset_index(drop=True)
    result = BacktestResult(config=config, daily=daily)
    result.summary = summarize_backtest(daily, config)
    return result


def run_backtest(
    firm_day: pd.DataFrame,
    daily_ar: pd.DataFrame,
    config: StrategyConfig,
    *,
    split: str | None = None,
) -> BacktestResult:
    """Backtest one strategy on one split and return the daily series.

    ``firm_day`` supplies the signal; ``daily_ar`` supplies dense returns. When
    ``split`` is given, formation is restricted to that split's sessions.
    """
    if config.signal not in firm_day.columns:
        raise ValueError(f"signal column missing: {config.signal}")
    grid = build_grid(firm_day, daily_ar, split=split, tail_horizon=config.horizon)
    if grid is None:
        return BacktestResult(config=config, daily=pd.DataFrame(), summary={"n_sessions": 0})
    return backtest_on_grid(grid, config)


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough fall of a compounded equity curve."""
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def summarize_backtest(daily: pd.DataFrame, config: StrategyConfig) -> dict[str, Any]:
    """Headline strategy metrics plus the break-even cost that kills it."""
    if daily.empty:
        return {"n_sessions": 0, "sharpe_net": float("nan"), "breakeven_bps_per_side": None}

    net = daily["net_return"].to_numpy(dtype=float)
    gross_r = daily["gross_return"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + net)
    years = len(net) / TRADING_SESSIONS_PER_YEAR
    mean_turnover = float(daily["turnover"].mean())

    def _sharpe(x: np.ndarray) -> float:
        vol = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        if vol <= 0 or not math.isfinite(vol):
            return float("nan")
        return float(math.sqrt(TRADING_SESSIONS_PER_YEAR) * np.mean(x) / vol)

    boot = block_bootstrap_mean(
        net,
        block_length=config.bootstrap_block_length,
        replications=config.bootstrap_replications,
        seed=config.random_seed,
    )
    breakeven = (
        float(10_000.0 * np.mean(gross_r) / (2.0 * mean_turnover))
        if mean_turnover > 0
        else None
    )
    return {
        "signal": config.signal,
        "horizon": config.horizon,
        "breadth": config.breadth,
        "n_sessions": int(len(net)),
        "years": float(years),
        "mean_gross": float(np.mean(gross_r)),
        "mean_net": float(np.mean(net)),
        "cagr_net": float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan"),
        "total_return_net": float(equity[-1] - 1.0),
        "ann_vol": float(np.std(net, ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)),
        "sharpe_gross": _sharpe(gross_r),
        "sharpe_net": _sharpe(net),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float(np.mean(net > 0)),
        "mean_turnover": mean_turnover,
        "annualized_turnover": mean_turnover * TRADING_SESSIONS_PER_YEAR,
        "mean_positions": float(daily["n_positions"].mean()),
        "mean_net_exposure": float(daily["net_exposure"].mean()),
        "max_missing_weight_share": float(daily["missing_weight_share"].max()),
        "breakeven_bps_per_side": breakeven,
        "cost_bps_per_side_charged": config.cost_bps_per_side,
        "bootstrap_net_mean": boot["mean"],
        "bootstrap_net_ci_low": boot["ci_low"],
        "bootstrap_net_ci_high": boot["ci_high"],
        "bootstrap_block_length": config.bootstrap_block_length,
        "bootstrap_replications": config.bootstrap_replications,
        "bootstrap_seed": config.random_seed,
    }


def sweep_development(
    firm_day: pd.DataFrame,
    daily_ar: pd.DataFrame,
    *,
    signals: tuple[str, ...],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    breadths: tuple[float, ...] = DEFAULT_BREADTHS,
    orients: dict[str, float] | None = None,
    cost_bps_per_side: float = 10.0,
    min_names: int = 5,
) -> pd.DataFrame:
    """Grid over signal x horizon x breadth, **development rows only**.

    Returns one summary row per cell. This is the selection surface; the
    evaluation block is not touched.
    """
    orient_map = orients or {}
    grid = build_grid(
        firm_day, daily_ar, split="development", tail_horizon=max(horizons)
    )
    if grid is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for signal in signals:
        if signal not in firm_day.columns:
            continue
        for breadth in breadths:
            # The book depends on (signal, breadth) only; horizon just changes
            # how many books are averaged. Build it once per pair.
            book_config = StrategyConfig(
                signal=signal,
                horizon=1,
                breadth=breadth,
                orient=float(orient_map.get(signal, 1.0)),
                cost_bps_per_side=cost_bps_per_side,
                min_names=min_names,
            )
            book = _formation_matrix(grid.frame, book_config, grid.sessions, grid.symbol_index)
            for horizon in horizons:
                config = StrategyConfig(
                    signal=signal,
                    horizon=horizon,
                    breadth=breadth,
                    orient=float(orient_map.get(signal, 1.0)),
                    cost_bps_per_side=cost_bps_per_side,
                    min_names=min_names,
                )
                rows.append(backtest_on_grid(grid, config, book=book).summary)
    return pd.DataFrame(rows)


def select_frozen_spec(
    sweep: pd.DataFrame,
    *,
    criterion: str = "sharpe_net",
    min_sessions: int = 250,
    min_positions: float = 5.0,
) -> dict[str, Any]:
    """Apply the pre-declared selection rule to the development sweep.

    The rule is declared before the sweep is read: highest development net
    Sharpe after costs, among cells with enough sessions and enough names to be
    a real portfolio. Ties break on lower turnover.
    """
    eligible = sweep.loc[
        (sweep["n_sessions"] >= min_sessions) & (sweep["mean_positions"] >= min_positions)
    ].copy()
    if eligible.empty:
        raise ValueError("no sweep cell satisfies the eligibility filters")
    eligible = eligible.sort_values(
        [criterion, "annualized_turnover"], ascending=[False, True]
    )
    best = eligible.iloc[0]
    return {
        "signal": str(best["signal"]),
        "horizon": int(best["horizon"]),
        "breadth": float(best["breadth"]),
        "selection_criterion": criterion,
        "selection_split": "development",
        "eligibility": {"min_sessions": min_sessions, "min_positions": min_positions},
        "development_sharpe_net": float(best["sharpe_net"]),
        "development_mean_net": float(best["mean_net"]),
        "development_breakeven_bps_per_side": (
            float(best["breakeven_bps_per_side"])
            if pd.notna(best["breakeven_bps_per_side"])
            else None
        ),
        "n_cells_considered": int(len(sweep)),
        "n_cells_eligible": int(len(eligible)),
        "development_end": FROZEN_DEV_END,
        "evaluation_start": FROZEN_EVAL_START,
    }


def run_frozen_spec(
    firm_day: pd.DataFrame,
    daily_ar: pd.DataFrame,
    spec: dict[str, Any],
    *,
    orients: dict[str, float] | None = None,
    cost_bps_per_side: float = 10.0,
    min_names: int = 5,
    split: str = "evaluation",
) -> BacktestResult:
    """Run a spec that was frozen to disk. The single evaluation call."""
    orient_map = orients or {}
    config = StrategyConfig(
        signal=str(spec["signal"]),
        horizon=int(spec["horizon"]),
        breadth=float(spec["breadth"]),
        orient=float(orient_map.get(str(spec["signal"]), 1.0)),
        cost_bps_per_side=cost_bps_per_side,
        min_names=min_names,
    )
    return run_backtest(firm_day, daily_ar, config, split=split)


def yearly_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-year net return, volatility, Sharpe and worst drawdown."""
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["session_date"]).dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        net = group["net_return"].to_numpy(dtype=float)
        equity = np.cumprod(1.0 + net)
        vol = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
        rows.append(
            {
                "year": int(year),
                "sessions": int(len(net)),
                "net_return": float(equity[-1] - 1.0),
                "ann_vol": vol * math.sqrt(TRADING_SESSIONS_PER_YEAR) if vol == vol else float("nan"),
                "sharpe_net": (
                    float(math.sqrt(TRADING_SESSIONS_PER_YEAR) * np.mean(net) / vol)
                    if vol and vol > 0
                    else float("nan")
                ),
                "max_drawdown": max_drawdown(equity),
                "mean_turnover": float(group["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


def event_time_car(
    firm_day: pd.DataFrame,
    daily_ar: pd.DataFrame,
    *,
    signal: str,
    orient: float = 1.0,
    n_quantiles: int = 5,
    max_lag: int = 20,
    split: str | None = "development",
) -> pd.DataFrame:
    """Cumulative abnormal return in event time, by within-day signal quantile.

    For every formation session names are sorted into ``n_quantiles`` groups by
    the oriented signal and equally weighted inside each group. Lag *k* is the
    k-th holding session after formation, so lag 1 is the return the one-session
    arm in ``03`` captures and later lags show whether anything continues.

    This is the diagnostic that decides whether a null is "no signal" or "signal
    at a horizon we did not trade": a flat CAR in every quantile means there is
    nothing to find, while a fanned CAR that the portfolio still fails to
    monetise means the problem is cost or breadth.
    """
    if signal not in firm_day.columns:
        raise ValueError(f"signal column missing: {signal}")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")

    grid = build_grid(firm_day, daily_ar, split=split, tail_horizon=max_lag)
    if grid is None:
        return pd.DataFrame()

    frame = grid.frame.dropna(subset=[signal]).copy()
    session_index = {ts: i for i, ts in enumerate(grid.sessions)}
    frame["_row"] = frame["session_date"].map(session_index)
    frame["_col"] = frame["symbol"].map(grid.symbol_index)
    frame = frame.dropna(subset=["_row", "_col"])
    if frame.empty:
        return pd.DataFrame()

    frame["_oriented"] = orient * frame[signal].astype(float)
    # Rank within the session, then cut into equal-count groups. `rank(pct=True)`
    # keeps this well defined when a day has fewer names than quantiles.
    pct = frame.groupby("_row", sort=False)["_oriented"].rank(method="first", pct=True)
    frame["_q"] = np.clip(np.ceil(pct.to_numpy() * n_quantiles), 1, n_quantiles).astype(int)

    n_sessions = len(grid.sessions)
    filled = np.where(np.isfinite(grid.returns), grid.returns, 0.0)
    observed = np.isfinite(grid.returns).astype(float)

    rows: list[dict[str, Any]] = []
    for q in range(1, n_quantiles + 1):
        member = frame.loc[frame["_q"] == q]
        weight = np.zeros((n_sessions, len(grid.symbol_index)), dtype=float)
        counts = np.zeros(n_sessions, dtype=float)
        r = member["_row"].to_numpy(dtype=int)
        c = member["_col"].to_numpy(dtype=int)
        np.add.at(weight, (r, c), 1.0)
        np.add.at(counts, r, 1.0)
        active = counts > 0
        weight[active] /= counts[active, None]

        cumulative = 0.0
        for lag in range(1, max_lag + 1):
            # Position formed at t earns returns[t] on its first held session,
            # so the k-th session is returns[t + k - 1].
            usable = n_sessions - lag + 1
            if usable <= 0:
                break
            w = weight[:usable]
            rows_active = active[:usable]
            if not rows_active.any():
                break
            per_day = (w * filled[lag - 1 :]).sum(axis=1)[rows_active]
            coverage = (w * observed[lag - 1 :]).sum(axis=1)[rows_active]
            mean_ar = float(np.mean(per_day))
            cumulative += mean_ar
            rows.append(
                {
                    "signal": signal,
                    "quantile": q,
                    "lag": lag,
                    "mean_ar": mean_ar,
                    "cum_ar": cumulative,
                    "se_ar": float(np.std(per_day, ddof=1) / math.sqrt(len(per_day))),
                    "n_events": int(rows_active.sum()),
                    "mean_coverage": float(np.mean(coverage)),
                }
            )
    return pd.DataFrame(rows)


def event_time_spread_inference(
    firm_day: pd.DataFrame,
    daily_ar: pd.DataFrame,
    *,
    signal: str,
    orient: float = 1.0,
    n_quantiles: int = 5,
    max_lag: int = 20,
    split: str | None = "development",
    block_length: int = 20,
    replications: int = 999,
    seed: int = 20260731,
) -> pd.DataFrame:
    """Q-high minus Q-low event-time spread with date-block inference.

    The observational unit is a formation session. The same circular block
    resample of formation dates is used at every lag, preserving both calendar
    dependence and the covariance of the cumulative path. P-values use a
    centred-null block bootstrap; callers should still adjust across the
    declared signal-by-lag family.
    """
    if signal not in firm_day.columns:
        raise ValueError(f"signal column missing: {signal}")
    if n_quantiles < 2 or max_lag < 1 or block_length < 1 or replications < 1:
        raise ValueError("invalid quantile, lag, block, or replication setting")

    grid = build_grid(firm_day, daily_ar, split=split, tail_horizon=max_lag)
    if grid is None:
        return pd.DataFrame()
    frame = grid.frame.dropna(subset=[signal]).copy()
    session_index = {ts: i for i, ts in enumerate(grid.sessions)}
    frame["_row"] = frame["session_date"].map(session_index)
    frame["_col"] = frame["symbol"].astype(str).str.upper().map(grid.symbol_index)
    frame = frame.dropna(subset=["_row", "_col"])
    frame["_oriented"] = orient * frame[signal].astype(float)
    pct = frame.groupby("_row", sort=False)["_oriented"].rank(method="first", pct=True)
    frame["_q"] = np.clip(
        np.ceil(pct.to_numpy() * n_quantiles), 1, n_quantiles
    ).astype(int)

    formation_rows = np.array(sorted(frame["_row"].astype(int).unique()), dtype=int)
    per_date = np.full((len(formation_rows), max_lag), np.nan, dtype=float)
    groups = {int(row): day for row, day in frame.groupby("_row", sort=False)}
    for i, row in enumerate(formation_rows):
        day = groups[row]
        low = day.loc[day["_q"] == 1, "_col"].to_numpy(dtype=int)
        high = day.loc[day["_q"] == n_quantiles, "_col"].to_numpy(dtype=int)
        if len(low) == 0 or len(high) == 0:
            continue
        for lag in range(max_lag):
            target = row + lag
            if target >= len(grid.sessions):
                break
            low_ret = grid.returns[target, low]
            high_ret = grid.returns[target, high]
            low_ret = low_ret[np.isfinite(low_ret)]
            high_ret = high_ret[np.isfinite(high_ret)]
            if len(low_ret) and len(high_ret):
                per_date[i, lag] = float(high_ret.mean() - low_ret.mean())

    valid = np.isfinite(per_date)
    counts = valid.sum(axis=0)
    sums = np.nansum(per_date, axis=0)
    point_increment = np.divide(
        sums,
        counts,
        out=np.full(max_lag, np.nan, dtype=float),
        where=counts > 0,
    )
    point_cumulative = np.cumsum(np.nan_to_num(point_increment, nan=0.0))

    n_dates = len(per_date)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n_dates / block_length))
    draw_counts = np.zeros((replications, n_dates), dtype=np.int32)
    for b in range(replications):
        starts = rng.integers(0, n_dates, size=n_blocks)
        idx = np.concatenate(
            [np.arange(s, s + block_length) % n_dates for s in starts]
        )[:n_dates]
        draw_counts[b] = np.bincount(idx, minlength=n_dates)

    filled = np.nan_to_num(per_date, nan=0.0)
    observed = valid.astype(float)
    boot_n = draw_counts @ observed
    boot_increment = np.divide(
        draw_counts @ filled,
        boot_n,
        out=np.full((replications, max_lag), np.nan, dtype=float),
        where=boot_n > 0,
    )
    boot_cumulative = np.cumsum(np.nan_to_num(boot_increment, nan=0.0), axis=1)

    centred = np.where(valid, per_date - point_increment[None, :], 0.0)
    null_increment = np.divide(
        draw_counts @ centred,
        boot_n,
        out=np.full((replications, max_lag), np.nan, dtype=float),
        where=boot_n > 0,
    )
    null_cumulative = np.cumsum(np.nan_to_num(null_increment, nan=0.0), axis=1)

    rows: list[dict[str, Any]] = []
    for lag in range(max_lag):
        null_abs = np.abs(null_cumulative[:, lag])
        p = max(
            float(np.mean(null_abs >= abs(point_cumulative[lag]))),
            1.0 / replications,
        )
        rows.append(
            {
                "signal": signal,
                "lag": lag + 1,
                "spread_mean_ar": point_increment[lag],
                "spread_cum_ar": point_cumulative[lag],
                "ci_low": float(np.quantile(boot_cumulative[:, lag], 0.025)),
                "ci_high": float(np.quantile(boot_cumulative[:, lag], 0.975)),
                "p_centered_block": p,
                "n_formation_dates": int(counts[lag]),
                "block_length": block_length,
                "replications": replications,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def rolling_annualized_sharpe(daily: pd.DataFrame, *, window: int = 252) -> pd.Series:
    """Trailing annualised net Sharpe. Shows regime dependence a single number hides."""
    net = daily["net_return"].astype(float)
    mean = net.rolling(window).mean()
    vol = net.rolling(window).std(ddof=1)
    return (math.sqrt(TRADING_SESSIONS_PER_YEAR) * mean / vol).rename(f"sharpe_{window}d")


def monthly_return_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    """Year × month compounded net return, for the calendar heatmap."""
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame["year"] = frame["session_date"].dt.year
    frame["month"] = frame["session_date"].dt.month
    monthly = (
        frame.groupby(["year", "month"])["net_return"]
        .apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0))
        .rename("net_return")
        .reset_index()
    )
    return monthly.pivot(index="year", columns="month", values="net_return")


def cost_curve(
    firm_day: pd.DataFrame,
    daily_ar: pd.DataFrame,
    config: StrategyConfig,
    *,
    cost_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0),
    split: str = "development",
) -> pd.DataFrame:
    """Net Sharpe as a function of the cost charged. Gross series is reused."""
    base = run_backtest(firm_day, daily_ar, config, split=split)
    if base.daily.empty:
        return pd.DataFrame()
    gross = base.daily["gross_return"].to_numpy(dtype=float)
    turnover = base.daily["turnover"].to_numpy(dtype=float)
    rows = []
    for bps in cost_grid:
        net = gross - 2.0 * turnover * (bps / 10_000.0)
        vol = float(np.std(net, ddof=1))
        rows.append(
            {
                "cost_bps_per_side": bps,
                "mean_net": float(np.mean(net)),
                "sharpe_net": (
                    float(math.sqrt(TRADING_SESSIONS_PER_YEAR) * np.mean(net) / vol)
                    if vol > 0
                    else float("nan")
                ),
                "total_return_net": float(np.cumprod(1.0 + net)[-1] - 1.0),
            }
        )
    return pd.DataFrame(rows)
