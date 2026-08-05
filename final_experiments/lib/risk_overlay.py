"""Price-momentum portfolio with sparse sentiment risk brakes.

This module supports the Notebook 19 experiment.  The return signal is a
standard, slow 12-minus-1-month cross-sectional momentum rank.  Sentiment is
not asked to forecast direction on its own; a high-confidence negative-news
flag may instead reduce an existing book's exposure for one session.

The important invariants are explicit:

* momentum uses prices ending ``skip_sessions`` before formation;
* books form monthly and are held on a dense exchange-session calendar;
* a brake can move capital to cash but never lever the surviving book up;
* both legs are scaled to the smaller surviving gross, preserving dollar
  neutrality; and
* half-L1 turnover pays a quoted per-side cost on both sides, including final
  liquidation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from final_experiments.lib.evaluate import block_bootstrap_mean

TRADING_SESSIONS_PER_YEAR = 252
BrakeMode = Literal["symmetric", "long_only"]
ExposureMode = Literal["one_day", "hysteresis"]


@dataclass(frozen=True)
class MomentumOverlayConfig:
    """Frozen design inputs for the momentum-plus-risk-overlay experiment."""

    lookback_sessions: int = 252
    skip_sessions: int = 21
    tail_fraction: float = 0.10
    min_universe: int = 100
    cost_bps_per_side: float = 10.0
    bootstrap_block_length: int = 20
    bootstrap_replications: int = 4_999
    random_seed: int = 20260805

    def __post_init__(self) -> None:
        if self.lookback_sessions <= self.skip_sessions:
            raise ValueError("lookback_sessions must exceed skip_sessions")
        if self.skip_sessions < 1:
            raise ValueError("skip_sessions must be positive")
        if not 0 < self.tail_fraction < 0.5:
            raise ValueError("tail_fraction must be in (0, 0.5)")
        if self.min_universe < 2:
            raise ValueError("min_universe must be at least two")
        if self.cost_bps_per_side < 0:
            raise ValueError("cost_bps_per_side must be non-negative")
        if self.bootstrap_block_length < 1 or self.bootstrap_replications < 1:
            raise ValueError("bootstrap settings must be positive")


def _normalise_prices(prices: pd.DataFrame) -> pd.DataFrame:
    need = {"symbol", "session_date", "adjusted_open"}
    missing = need - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {sorted(missing)}")
    frame = prices.loc[:, ["symbol", "session_date", "adjusted_open"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    if frame.duplicated(["symbol", "session_date"]).any():
        raise ValueError("prices must be unique by symbol/session_date")
    return frame.sort_values(["session_date", "symbol"], kind="mergesort")


def build_adjusted_open_returns(
    prices: pd.DataFrame,
    *,
    symbol: str,
    return_col: str = "return",
) -> pd.DataFrame:
    """Immediate next-session adjusted-open returns for one dense price series."""
    frame = _normalise_prices(prices)
    target = frame.loc[frame["symbol"].eq(symbol.upper())].copy()
    if target.empty:
        raise ValueError(f"symbol {symbol.upper()} is absent from prices")
    target["return_end_date"] = target["session_date"].shift(-1)
    target["next_adjusted_open"] = target["adjusted_open"].shift(-1)
    target[return_col] = target["next_adjusted_open"] / target["adjusted_open"] - 1.0
    return target.dropna(subset=["return_end_date", return_col])[
        ["symbol", "session_date", "return_end_date", return_col]
    ].reset_index(drop=True)


def build_monthly_momentum_weights(
    prices: pd.DataFrame,
    tradable_sessions: pd.DatetimeIndex | list[pd.Timestamp],
    *,
    config: MomentumOverlayConfig | None = None,
    market_symbol: str = "SPY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build dense monthly 12-minus-1 momentum weights.

    ``tradable_sessions`` must already respect the requested split boundary and
    contain only sessions with a forward return.  The first tradable session of
    each calendar month is the formation date.  At formation ``t``, momentum is
    ``open[t-skip] / open[t-lookback] - 1``; the current open is used only to
    establish that the name is tradable.

    Returns a daily long weight panel and one formation-diagnostic row per
    month.  Each active base book has gross exposure one and net exposure zero.
    """
    cfg = config or MomentumOverlayConfig()
    frame = _normalise_prices(prices)
    market = market_symbol.upper()
    market_sessions = pd.DatetimeIndex(
        frame.loc[frame["symbol"].eq(market), "session_date"].drop_duplicates().sort_values()
    )
    if market_sessions.empty:
        raise ValueError(f"market symbol {market} is absent from prices")

    sessions = pd.DatetimeIndex(pd.to_datetime(tradable_sessions)).normalize().sort_values().unique()
    if sessions.empty:
        return (
            pd.DataFrame(columns=["session_date", "symbol", "weight", "formation_date"]),
            pd.DataFrame(),
        )
    absent = sessions.difference(market_sessions)
    if len(absent):
        raise ValueError(f"tradable sessions absent from market calendar: {list(absent[:5])}")

    stock = frame.loc[~frame["symbol"].eq(market)]
    open_wide = stock.pivot(index="session_date", columns="symbol", values="adjusted_open").reindex(
        market_sessions
    )
    session_pos = {session: i for i, session in enumerate(market_sessions)}
    formation_dates = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).first().tolist()

    daily_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for formation_number, formation_date in enumerate(formation_dates):
        position = session_pos[pd.Timestamp(formation_date)]
        if position < cfg.lookback_sessions:
            continue
        start_date = market_sessions[position - cfg.lookback_sessions]
        end_date = market_sessions[position - cfg.skip_sessions]
        current = open_wide.loc[formation_date]
        start = open_wide.loc[start_date]
        end = open_wide.loc[end_date]
        valid = current.notna() & start.gt(0) & end.gt(0)
        momentum = (end / start - 1.0).loc[valid].sort_index()
        n_universe = int(momentum.notna().sum())
        if n_universe < cfg.min_universe:
            diagnostics.append(
                {
                    "formation_date": formation_date,
                    "lookback_start": start_date,
                    "lookback_end": end_date,
                    "n_universe": n_universe,
                    "n_per_leg": 0,
                    "status": "below_min_universe",
                }
            )
            continue

        ranked = (
            momentum.dropna()
            .rename("momentum")
            .reset_index()
            .sort_values(["momentum", "symbol"], kind="mergesort")
        )
        n_per_leg = max(1, int(math.floor(len(ranked) * cfg.tail_fraction)))
        if 2 * n_per_leg > len(ranked):
            raise RuntimeError("tail selection overlaps")
        short_symbols = ranked.iloc[:n_per_leg]["symbol"].astype(str).tolist()
        long_symbols = ranked.iloc[-n_per_leg:]["symbol"].astype(str).tolist()
        book = pd.DataFrame(
            {
                "symbol": short_symbols + long_symbols,
                "weight": ([-0.5 / n_per_leg] * n_per_leg) + ([0.5 / n_per_leg] * n_per_leg),
            }
        )

        next_formation = (
            pd.Timestamp(formation_dates[formation_number + 1])
            if formation_number + 1 < len(formation_dates)
            else None
        )
        held_sessions = sessions[sessions >= formation_date]
        if next_formation is not None:
            held_sessions = held_sessions[held_sessions < next_formation]
        if len(held_sessions):
            repeated = book.merge(
                pd.DataFrame({"session_date": held_sessions}),
                how="cross",
            )
            repeated["formation_date"] = pd.Timestamp(formation_date)
            daily_parts.append(repeated[["session_date", "symbol", "weight", "formation_date"]])

        diagnostics.append(
            {
                "formation_date": formation_date,
                "lookback_start": start_date,
                "lookback_end": end_date,
                "n_universe": n_universe,
                "n_per_leg": n_per_leg,
                "status": "formed",
            }
        )

    weights = (
        pd.concat(daily_parts, ignore_index=True)
        if daily_parts
        else pd.DataFrame(columns=["session_date", "symbol", "weight", "formation_date"])
    )
    if not weights.empty:
        check = weights.groupby("session_date")["weight"].agg(net="sum", gross=lambda s: s.abs().sum())
        if not np.allclose(check["net"], 0.0, atol=1e-12):
            raise RuntimeError("momentum base weights are not dollar-neutral")
        if not np.allclose(check["gross"], 1.0, atol=1e-12):
            raise RuntimeError("momentum base weights do not have gross exposure one")
    return weights, pd.DataFrame(diagnostics)


def build_negative_risk_flags(
    firm_day: pd.DataFrame,
    *,
    negative_share_threshold: float = 1.0,
    min_articles: int = 2,
) -> pd.DataFrame:
    """Return one high-confidence negative-news flag per firm-session."""
    need = {"symbol", "session_date", "negative_share", "n"}
    missing = need - set(firm_day.columns)
    if missing:
        raise ValueError(f"firm_day missing columns: {sorted(missing)}")
    if not 0 <= negative_share_threshold <= 1:
        raise ValueError("negative_share_threshold must be in [0, 1]")
    if min_articles < 1:
        raise ValueError("min_articles must be positive")
    frame = firm_day.loc[:, list(need)].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    if frame.duplicated(["session_date", "symbol"]).any():
        raise ValueError("firm_day must be unique by symbol/session_date")
    flag = frame["negative_share"].ge(negative_share_threshold) & frame["n"].ge(min_articles)
    return frame.loc[flag, ["session_date", "symbol"]].assign(risk_flag=True).reset_index(drop=True)


def build_market_negative_pressure(
    firm_day: pd.DataFrame,
    sessions: pd.DatetimeIndex | list[pd.Timestamp],
    *,
    trailing_window: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    """Aggregate headline-weighted negative share and its lagged trailing z-score.

    The rolling mean and standard deviation are shifted by one session.  The
    current session's mapped news enters the numerator but can never enter its
    own normalisation baseline.
    """
    need = {"session_date", "negative_share", "n"}
    if missing := need - set(firm_day.columns):
        raise ValueError(f"firm_day missing columns: {sorted(missing)}")
    if trailing_window < 2:
        raise ValueError("trailing_window must be at least two")
    if not 2 <= min_periods <= trailing_window:
        raise ValueError("min_periods must be in [2, trailing_window]")

    frame = firm_day.loc[:, list(need)].dropna(subset=["negative_share", "n"]).copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["negative_headlines"] = frame["negative_share"].astype(float) * frame["n"].astype(float)
    daily = frame.groupby("session_date", sort=True).agg(
        negative_headlines=("negative_headlines", "sum"),
        headline_count=("n", "sum"),
        firms_with_news=("n", "size"),
    )
    daily["negative_pressure"] = daily["negative_headlines"] / daily["headline_count"]

    index = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize().sort_values().unique()
    daily = daily.reindex(index)
    prior = daily["negative_pressure"].shift(1)
    daily["trailing_mean"] = prior.rolling(trailing_window, min_periods=min_periods).mean()
    daily["trailing_std"] = prior.rolling(trailing_window, min_periods=min_periods).std(ddof=1)
    daily["negative_pressure_z"] = (
        (daily["negative_pressure"] - daily["trailing_mean"]) / daily["trailing_std"]
    ).where(daily["trailing_std"].gt(0))
    return daily.rename_axis("session_date").reset_index()


def build_market_risk_exposure(
    pressure: pd.DataFrame,
    *,
    mode: ExposureMode,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    risk_exposure: float = 0.25,
) -> pd.DataFrame:
    """Map aggregate negative-news surprise to an SPY exposure path.

    ``one_day`` cuts exposure only on sessions at or above ``entry_z``.
    ``hysteresis`` remains risk-off until the score reaches ``exit_z``.  Before
    the lagged z-score exists, exposure is one.  Both rules are long-or-cash and
    never exceed the baseline exposure.
    """
    if mode not in {"one_day", "hysteresis"}:
        raise ValueError("mode must be 'one_day' or 'hysteresis'")
    if exit_z >= entry_z:
        raise ValueError("exit_z must be below entry_z")
    if not 0 <= risk_exposure <= 1:
        raise ValueError("risk_exposure must be in [0, 1]")
    need = {"session_date", "negative_pressure_z"}
    if missing := need - set(pressure.columns):
        raise ValueError(f"pressure missing columns: {sorted(missing)}")

    out = pressure.copy().sort_values("session_date", kind="mergesort").reset_index(drop=True)
    z = out["negative_pressure_z"].to_numpy(dtype=float)
    risk_off = np.zeros(len(out), dtype=bool)
    if mode == "one_day":
        risk_off = np.isfinite(z) & (z >= entry_z)
    else:
        state = False
        for i, value in enumerate(z):
            if math.isfinite(value):
                if not state and value >= entry_z:
                    state = True
                elif state and value <= exit_z:
                    state = False
            risk_off[i] = state
    out["risk_off"] = risk_off
    out["exposure"] = np.where(risk_off, risk_exposure, 1.0)
    out["exposure_mode"] = mode
    return out


def apply_sentiment_brake(
    base_weights: pd.DataFrame,
    risk_flags: pd.DataFrame,
    *,
    mode: BrakeMode,
) -> pd.DataFrame:
    """Apply a one-session risk brake while preserving dollar neutrality.

    Flagged positions are set to zero under ``symmetric``; under ``long_only``
    only flagged long positions are removed.  The surviving long and short
    books are then scaled *down* to the smaller leg.  No weight is ever scaled
    above its base magnitude, so the brake can only move capital to cash.
    """
    if mode not in {"symmetric", "long_only"}:
        raise ValueError("mode must be 'symmetric' or 'long_only'")
    need_weights = {"session_date", "symbol", "weight"}
    need_flags = {"session_date", "symbol", "risk_flag"}
    if missing := need_weights - set(base_weights.columns):
        raise ValueError(f"base_weights missing columns: {sorted(missing)}")
    if missing := need_flags - set(risk_flags.columns):
        raise ValueError(f"risk_flags missing columns: {sorted(missing)}")
    if base_weights.duplicated(["session_date", "symbol"]).any():
        raise ValueError("base_weights must be unique by symbol/session_date")
    if risk_flags.duplicated(["session_date", "symbol"]).any():
        raise ValueError("risk_flags must be unique by symbol/session_date")

    frame = base_weights.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    flags = risk_flags.copy()
    flags["session_date"] = pd.to_datetime(flags["session_date"]).dt.normalize()
    flags["symbol"] = flags["symbol"].astype(str).str.upper()
    frame = frame.merge(flags, on=["session_date", "symbol"], how="left", validate="1:1")
    frame["risk_flag"] = frame["risk_flag"].eq(True)
    frame["base_weight"] = frame["weight"].astype(float)
    brake = frame["risk_flag"] if mode == "symmetric" else frame["risk_flag"] & frame["weight"].gt(0)
    frame["brake_applied"] = brake
    frame.loc[brake, "weight"] = 0.0

    long_gross = frame["weight"].clip(lower=0).groupby(frame["session_date"]).transform("sum")
    short_gross = (-frame["weight"].clip(upper=0)).groupby(frame["session_date"]).transform("sum")
    target = np.minimum(long_gross, short_gross)
    positive = frame["weight"].gt(0)
    negative = frame["weight"].lt(0)
    long_scale = np.divide(target, long_gross, out=np.zeros(len(frame)), where=long_gross.gt(0))
    short_scale = np.divide(target, short_gross, out=np.zeros(len(frame)), where=short_gross.gt(0))
    frame.loc[positive, "weight"] *= long_scale[positive]
    frame.loc[negative, "weight"] *= short_scale[negative]

    check = frame.groupby("session_date")["weight"].agg(net="sum", gross=lambda s: s.abs().sum())
    base_gross = frame.groupby("session_date")["base_weight"].apply(lambda s: s.abs().sum())
    if not np.allclose(check["net"], 0.0, atol=1e-12):
        raise RuntimeError("sentiment brake broke dollar neutrality")
    if (check["gross"] - base_gross > 1e-12).any():
        raise RuntimeError("sentiment brake increased gross exposure")
    frame["overlay_mode"] = mode
    return frame


def backtest_weight_panel(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    cost_bps_per_side: float,
    return_col: str = "ar",
) -> pd.DataFrame:
    """Backtest daily weights with explicit entry, transition and exit costs."""
    if cost_bps_per_side < 0:
        raise ValueError("cost_bps_per_side must be non-negative")
    need_weights = {"session_date", "symbol", "weight"}
    need_returns = {"session_date", "symbol", return_col}
    if missing := need_weights - set(weights.columns):
        raise ValueError(f"weights missing columns: {sorted(missing)}")
    if missing := need_returns - set(daily_returns.columns):
        raise ValueError(f"daily_returns missing columns: {sorted(missing)}")
    if weights.empty:
        return pd.DataFrame()

    w = weights.loc[:, ["session_date", "symbol", "weight"]].copy()
    r = daily_returns.loc[:, ["session_date", "symbol", return_col]].copy()
    for frame in (w, r):
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if w.duplicated(["session_date", "symbol"]).any():
        raise ValueError("weights must be unique by symbol/session_date")
    if r.duplicated(["session_date", "symbol"]).any():
        raise ValueError("daily_returns must be unique by symbol/session_date")

    sessions = pd.DatetimeIndex(sorted(w["session_date"].unique()))
    symbols = sorted(w["symbol"].unique())
    weight_matrix = (
        w.pivot(index="session_date", columns="symbol", values="weight")
        .reindex(index=sessions, columns=symbols)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return_frame = r.loc[r["session_date"].isin(sessions) & r["symbol"].isin(symbols)]
    return_matrix = (
        return_frame.pivot(index="session_date", columns="symbol", values=return_col)
        .reindex(index=sessions, columns=symbols)
        .to_numpy(dtype=float)
    )
    present = np.isfinite(return_matrix)
    gross_return = (weight_matrix * np.where(present, return_matrix, 0.0)).sum(axis=1)
    gross_exposure = np.abs(weight_matrix).sum(axis=1)
    missing_weight = np.abs(np.where(present, 0.0, weight_matrix)).sum(axis=1)
    previous = np.vstack([np.zeros((1, weight_matrix.shape[1])), weight_matrix[:-1]])
    turnover = 0.5 * np.abs(weight_matrix - previous).sum(axis=1)
    if len(turnover):
        turnover[-1] += 0.5 * np.abs(weight_matrix[-1]).sum()
    cost = 2.0 * turnover * (cost_bps_per_side / 10_000.0)
    return pd.DataFrame(
        {
            "session_date": sessions,
            "gross_exposure": gross_exposure,
            "net_exposure": weight_matrix.sum(axis=1),
            "n_positions": (np.abs(weight_matrix) > 1e-12).sum(axis=1),
            "gross_return": gross_return,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross_return - cost,
            "missing_weight_share": np.divide(
                missing_weight,
                gross_exposure,
                out=np.zeros_like(missing_weight),
                where=gross_exposure > 0,
            ),
        }
    )


def reprice_daily(daily: pd.DataFrame, *, cost_bps_per_side: float) -> pd.DataFrame:
    """Reprice a fixed weight path at another per-side transaction cost."""
    if cost_bps_per_side < 0:
        raise ValueError("cost_bps_per_side must be non-negative")
    out = daily.copy()
    out["cost"] = 2.0 * out["turnover"] * (cost_bps_per_side / 10_000.0)
    out["net_return"] = out["gross_return"] - out["cost"]
    return out


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def summarize_weight_backtest(
    daily: pd.DataFrame,
    *,
    config: MomentumOverlayConfig | None = None,
) -> dict[str, Any]:
    """Summarize a fixed daily path under the experiment accounting rules."""
    cfg = config or MomentumOverlayConfig()
    if daily.empty:
        return {"n_sessions": 0}
    gross = daily["gross_return"].to_numpy(dtype=float)
    net = daily["net_return"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + net)
    mean_turnover = float(daily["turnover"].mean())

    def sharpe(values: np.ndarray) -> float:
        vol = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return (
            float(math.sqrt(TRADING_SESSIONS_PER_YEAR) * np.mean(values) / vol)
            if vol > 0 and math.isfinite(vol)
            else float("nan")
        )

    boot = block_bootstrap_mean(
        net,
        block_length=cfg.bootstrap_block_length,
        replications=cfg.bootstrap_replications,
        seed=cfg.random_seed,
    )
    years = len(net) / TRADING_SESSIONS_PER_YEAR
    return {
        "n_sessions": int(len(daily)),
        "active_sessions": int(daily["gross_exposure"].gt(0).sum()),
        "mean_gross": float(np.mean(gross)),
        "mean_net": float(np.mean(net)),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "ann_vol_net": float(np.std(net, ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)),
        "total_return_net": float(equity[-1] - 1.0),
        "cagr_net": float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan"),
        "max_drawdown": max_drawdown(equity),
        "mean_turnover": mean_turnover,
        "annualized_turnover": mean_turnover * TRADING_SESSIONS_PER_YEAR,
        "mean_gross_exposure": float(daily["gross_exposure"].mean()),
        "mean_positions": float(daily["n_positions"].mean()),
        "max_abs_net_exposure": float(daily["net_exposure"].abs().max()),
        "max_missing_weight_share": float(daily["missing_weight_share"].max()),
        "breakeven_bps_per_side": (
            float(10_000.0 * np.mean(gross) / (2.0 * mean_turnover))
            if mean_turnover > 0
            else None
        ),
        "bootstrap_net_mean": boot["mean"],
        "bootstrap_net_ci_low": boot["ci_low"],
        "bootstrap_net_ci_high": boot["ci_high"],
        "bootstrap_block_length": cfg.bootstrap_block_length,
        "bootstrap_replications": cfg.bootstrap_replications,
        "bootstrap_seed": cfg.random_seed,
    }


def paired_block_bootstrap_difference(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    value_col: str = "net_return",
    block_length: int = 20,
    replications: int = 4_999,
    seed: int = 20260805,
) -> dict[str, float | int]:
    """Paired circular-block inference for challenger minus baseline mean."""
    if block_length < 1 or replications < 1:
        raise ValueError("bootstrap settings must be positive")
    left = challenger.loc[:, ["session_date", value_col]].rename(columns={value_col: "challenger"})
    right = baseline.loc[:, ["session_date", value_col]].rename(columns={value_col: "baseline"})
    paired = left.merge(right, on="session_date", how="inner", validate="1:1").sort_values("session_date")
    diff = (paired["challenger"] - paired["baseline"]).to_numpy(dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        return {
            "n_sessions": n,
            "mean_difference": float(np.mean(diff)) if n else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_two_sided": float("nan"),
        }
    if block_length >= n:
        raise ValueError("block_length must be smaller than the paired sample")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_length))
    means = np.empty(replications, dtype=float)
    for replication in range(replications):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate(
            [(np.arange(start, start + block_length) % n) for start in starts]
        )[:n]
        means[replication] = float(diff[indices].mean())
    point = float(diff.mean())
    centred = means - point
    p_value = float((1 + np.sum(np.abs(centred) >= abs(point))) / (replications + 1))
    return {
        "n_sessions": n,
        "mean_difference": point,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_two_sided": p_value,
        "block_length": int(block_length),
        "replications": int(replications),
        "seed": int(seed),
    }
