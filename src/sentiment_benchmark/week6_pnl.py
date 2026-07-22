"""Leak-controlled Week 6 daily P&L analysis.

This module deliberately sits beside the event-study backtest.  The existing
backtest answers an event-level fixed-notional question; Week 6 needs a funded,
daily stock/portfolio series with position-change costs.  Untimestamped daily
signals are conservatively actioned at the *next* trading session close and can
therefore earn only the following close-to-close return.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .strategy_sweep import chronological_split_date


class Week6PnlError(RuntimeError):
    """Raised when Week 6 inputs or accounting invariants are invalid."""


@dataclass(frozen=True)
class Week6PnlConfig:
    """Frozen assumptions for one Week 6 run."""

    run_id: str
    scorer_id: str = "headline/sentiment_all"
    starting_capital: float = 100_000.0
    transaction_cost_bps_per_side: float = 10.0
    development_fraction: float = 0.6
    thresholds: tuple[float, ...] = (0.0, 0.05, 0.10)
    holding_periods: tuple[int, ...] = (1, 3, 5, 7)
    annualization_periods: int = 252
    annual_risk_free_rate: float = 0.0
    exchange_timezone: str = "America/New_York"
    min_joint_active_dates: int = 10
    low_correlation_stock_count: int | None = None
    include_low_correlation_portfolio: bool = True


@dataclass(frozen=True)
class Week6RunResult:
    """Paths and frozen choices returned by :func:`run_week6_pnl`."""

    output_dir: Path
    split_date: str
    selected_threshold: float
    selected_holding_period: int
    selected_stocks: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentCounts:
    input_signals: int
    aligned_signals: int
    missing_score_signals: int
    no_future_price_signals: int
    superseded_same_entry_signals: int
    boundary_attrited_signals: int


REQUIRED_SIGNAL_COLUMNS = {
    "symbol",
    "news_date",
    "scorer_id",
    "article_count",
    "valid_count",
    "mean_score",
}
REQUIRED_PRICE_COLUMNS = {"symbol", "session_date", "open", "high", "low", "close", "volume"}
STOCK_DAILY_COLUMNS = [
    "date",
    "symbol",
    "signal_date",
    "signal_timestamp",
    "signal",
    "signal_direction",
    "position",
    "previous_position",
    "holding_period",
    "capital_allocation",
    "price_entry_date",
    "price_exit_date",
    "entry_price",
    "exit_price",
    "realised_forward_return",
    "gross_return",
    "gross_pnl",
    "turnover",
    "transaction_cost",
    "net_return",
    "net_pnl",
    "split",
    "active_trade",
    "trade_executed",
    "timing_attrition_status",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_iso_dates(values: pd.Series, *, field: str) -> None:
    try:
        parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise Week6PnlError(f"{field} contains an invalid ISO date: {exc}") from exc
    canonical = parsed.dt.strftime("%Y-%m-%d")
    if not canonical.equals(values.astype(str).reset_index(drop=True)):
        raise Week6PnlError(f"{field} must use canonical YYYY-MM-DD dates")


def load_and_validate_inputs(
    signals_path: str | Path,
    prices_path: str | Path,
    scorer_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the selected signal and price panels and enforce input invariants."""

    signal_file = Path(signals_path).expanduser().resolve()
    price_file = Path(prices_path).expanduser().resolve()
    if not signal_file.is_file():
        raise Week6PnlError(f"signals file does not exist: {signal_file}")
    if not price_file.is_file():
        raise Week6PnlError(f"prices file does not exist: {price_file}")
    signals = pd.read_csv(signal_file)
    prices = pd.read_csv(price_file)
    missing_signal_columns = REQUIRED_SIGNAL_COLUMNS - set(signals.columns)
    missing_price_columns = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing_signal_columns:
        raise Week6PnlError(f"signals file lacks columns: {sorted(missing_signal_columns)}")
    if missing_price_columns:
        raise Week6PnlError(f"prices file lacks columns: {sorted(missing_price_columns)}")

    signals = signals.loc[signals["scorer_id"].astype(str) == scorer_id].copy()
    if signals.empty:
        available = sorted(pd.read_csv(signal_file, usecols=["scorer_id"])["scorer_id"].dropna().astype(str).unique())
        raise Week6PnlError(f"scorer {scorer_id!r} is absent; available scorers: {available}")
    signals["symbol"] = signals["symbol"].astype(str).str.strip()
    signals["news_date"] = signals["news_date"].astype(str)
    prices["symbol"] = prices["symbol"].astype(str).str.strip()
    prices["session_date"] = prices["session_date"].astype(str)
    signals = signals.reset_index(drop=True)
    prices = prices.reset_index(drop=True)
    _validate_iso_dates(signals["news_date"], field="news_date")
    _validate_iso_dates(prices["session_date"], field="session_date")

    signal_duplicates = signals.duplicated(["symbol", "news_date", "scorer_id"], keep=False)
    if signal_duplicates.any():
        sample = signals.loc[signal_duplicates, ["symbol", "news_date", "scorer_id"]].head(5).to_dict("records")
        raise Week6PnlError(f"duplicate signal rows detected: {sample}")
    price_duplicates = prices.duplicated(["symbol", "session_date"], keep=False)
    if price_duplicates.any():
        sample = prices.loc[price_duplicates, ["symbol", "session_date"]].head(5).to_dict("records")
        raise Week6PnlError(f"duplicate price rows detected: {sample}")
    if (signals["symbol"] == "").any() or (prices["symbol"] == "").any():
        raise Week6PnlError("blank symbols are not allowed")
    for column in ("open", "high", "low", "close"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
        if prices[column].isna().any() or (prices[column] <= 0).any():
            raise Week6PnlError(f"prices contain missing or non-positive {column} values")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    if prices["volume"].isna().any() or (prices["volume"] < 0).any():
        raise Week6PnlError("prices contain missing or negative volume values")
    signals["mean_score"] = pd.to_numeric(signals["mean_score"], errors="coerce")
    signals["article_count"] = pd.to_numeric(signals["article_count"], errors="raise").astype(int)
    signals["valid_count"] = pd.to_numeric(signals["valid_count"], errors="raise").astype(int)
    if (signals[["article_count", "valid_count"]] < 0).any().any():
        raise Week6PnlError("article and valid counts must be non-negative")
    if (signals["valid_count"] > signals["article_count"]).any():
        raise Week6PnlError("valid_count cannot exceed article_count")
    missing_symbols = sorted(set(signals["symbol"]) - set(prices["symbol"]))
    if missing_symbols:
        raise Week6PnlError(f"signals have no price panel for symbols: {missing_symbols}")
    if "availability_timestamp" not in signals:
        signals["availability_timestamp"] = ""
    signals["availability_timestamp"] = signals["availability_timestamp"].fillna("").astype(str)
    prices = prices.sort_values(["symbol", "session_date"], kind="stable").reset_index(drop=True)
    signals = signals.sort_values(["symbol", "news_date"], kind="stable").reset_index(drop=True)
    return signals, prices


def signal_position(score: float | None, threshold: float) -> int:
    """Apply the symmetric, strict Week 6 threshold rule."""

    if score is None or not _finite(score):
        return 0
    if float(score) > threshold:
        return 1
    if float(score) < -threshold:
        return -1
    return 0


def _entry_session(
    signal: pd.Series,
    session_dates: list[str],
    *,
    exchange_timezone: str,
) -> tuple[str | None, str]:
    timestamp = str(signal.get("availability_timestamp") or "").strip()
    if not timestamp:
        candidates = [value for value in session_dates if value > str(signal["news_date"])]
        return (candidates[0], "aligned_next_session_close") if candidates else (None, "no_future_price")
    try:
        available_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Week6PnlError(f"invalid availability timestamp {timestamp!r}") from exc
    if available_at.tzinfo is None:
        raise Week6PnlError(f"availability timestamp lacks timezone offset: {timestamp!r}")
    zone = ZoneInfo(exchange_timezone)
    for session_date in session_dates:
        session_close = datetime.combine(date.fromisoformat(session_date), time(16, 0), zone)
        if session_close > available_at:
            return session_date, "aligned_after_timestamp_close"
    return None, "no_future_price"


def align_signals_to_sessions(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    threshold: float,
    exchange_timezone: str,
) -> tuple[pd.DataFrame, AlignmentCounts]:
    """Map each signal to the first tradable close strictly after availability.

    Multiple non-trading news dates can map to the same entry session.  The most
    recent available signal deterministically supersedes older signals before the
    session close; this is recorded as attrition rather than silently averaged.
    """

    sessions = {symbol: group["session_date"].astype(str).tolist() for symbol, group in prices.groupby("symbol", sort=True)}
    rows: list[dict[str, Any]] = []
    no_future = 0
    missing_score = 0
    for _, signal in signals.iterrows():
        entry_date, status = _entry_session(signal, sessions[str(signal["symbol"])], exchange_timezone=exchange_timezone)
        if entry_date is None:
            no_future += 1
            continue
        score = None if pd.isna(signal["mean_score"]) else float(signal["mean_score"])
        if score is None:
            missing_score += 1
        rows.append(
            {
                "symbol": str(signal["symbol"]),
                "signal_date": str(signal["news_date"]),
                "signal_timestamp": str(signal.get("availability_timestamp") or ""),
                "entry_date": entry_date,
                "signal": score,
                "position": signal_position(score, threshold),
                "alignment_status": status if score is not None else "aligned_missing_score_flat",
            }
        )
    aligned = pd.DataFrame(rows)
    if aligned.empty:
        raise Week6PnlError("no signals align to a future price session")
    aligned = aligned.sort_values(["symbol", "entry_date", "signal_date", "signal_timestamp"], kind="stable")
    duplicate_entry = aligned.duplicated(["symbol", "entry_date"], keep="last")
    superseded = int(duplicate_entry.sum())
    aligned = aligned.loc[~duplicate_entry].reset_index(drop=True)
    return aligned, AlignmentCounts(
        input_signals=len(signals),
        aligned_signals=len(aligned),
        missing_score_signals=missing_score,
        no_future_price_signals=no_future,
        superseded_same_entry_signals=superseded,
        boundary_attrited_signals=0,
    )


def _split_for_date(session_date: str, split_date: str) -> str:
    return "development" if session_date < split_date else "evaluation"


def build_stock_daily_pnl(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    threshold: float,
    holding_period: int,
    split_date: str,
    starting_capital: float,
    transaction_cost_bps_per_side: float,
    exchange_timezone: str = "America/New_York",
) -> tuple[pd.DataFrame, AlignmentCounts]:
    """Construct daily stock positions, fixed-notional P&L, and costs.

    Position is set at a session close and earns that close-to-next-close return.
    A newer signal supersedes an overlapping holding.  Signals without enough
    same-split future sessions to realise the full holding period are attrited, so
    development holdings never leak into evaluation and evaluation never uses
    unavailable trailing returns.
    """

    if threshold < 0:
        raise Week6PnlError("threshold must be non-negative")
    if holding_period < 1:
        raise Week6PnlError("holding_period must be at least one session")
    if starting_capital <= 0:
        raise Week6PnlError("starting_capital must be positive")
    if transaction_cost_bps_per_side < 0:
        raise Week6PnlError("transaction cost cannot be negative")
    aligned, counts = align_signals_to_sessions(
        signals,
        prices,
        threshold=threshold,
        exchange_timezone=exchange_timezone,
    )
    symbols = sorted(set(signals["symbol"]))
    allocation = starting_capital / len(symbols)
    cost_rate = transaction_cost_bps_per_side / 10_000.0
    output: list[dict[str, Any]] = []
    boundary_attrited = 0

    for symbol in symbols:
        symbol_prices = prices.loc[prices["symbol"] == symbol].sort_values("session_date", kind="stable").reset_index(drop=True)
        dates = symbol_prices["session_date"].astype(str).tolist()
        close_by_date = dict(zip(dates, symbol_prices["close"].astype(float), strict=True))
        next_date = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
        event_rows = aligned.loc[aligned["symbol"] == symbol].to_dict("records")
        eligible_events: dict[str, dict[str, Any]] = {}
        for event in event_rows:
            entry_date = str(event["entry_date"])
            entry_index = dates.index(entry_date)
            exit_index = entry_index + holding_period
            if exit_index >= len(dates) or _split_for_date(dates[exit_index], split_date) != _split_for_date(entry_date, split_date):
                boundary_attrited += 1
                continue
            eligible_events[entry_date] = event

        for split in ("development", "evaluation"):
            split_dates = [value for value in dates if _split_for_date(value, split_date) == split]
            previous_position = 0
            active_position = 0
            active_signal: dict[str, Any] | None = None
            remaining = 0
            for session_date in split_dates:
                event = eligible_events.get(session_date)
                if event is not None:
                    active_position = int(event["position"])
                    active_signal = event
                    remaining = holding_period
                    status = str(event["alignment_status"])
                elif remaining > 0:
                    status = "carried_existing_signal"
                else:
                    active_position = 0
                    active_signal = None
                    status = "holding_expired" if previous_position != 0 else "no_eligible_signal"

                exit_date = next_date.get(session_date)
                if exit_date is None or _split_for_date(exit_date, split_date) != split:
                    position = 0
                    realised_return = 0.0
                    exit_date_value = ""
                    exit_price = float(close_by_date[session_date])
                    if previous_position != 0:
                        status = "split_boundary_or_sample_end_exit"
                else:
                    position = active_position
                    entry_price = float(close_by_date[session_date])
                    exit_price = float(close_by_date[exit_date])
                    realised_return = exit_price / entry_price - 1.0
                    exit_date_value = exit_date
                entry_price = float(close_by_date[session_date])
                turnover = abs(position - previous_position)
                gross_return = position * realised_return
                gross_pnl = allocation * gross_return
                transaction_cost = allocation * turnover * cost_rate
                net_pnl = gross_pnl - transaction_cost
                net_return = net_pnl / allocation
                signal_value = "" if active_signal is None or active_signal["signal"] is None else float(active_signal["signal"])
                signal_date = "" if active_signal is None else str(active_signal["signal_date"])
                signal_timestamp = "" if active_signal is None else str(active_signal["signal_timestamp"])
                output.append(
                    {
                        "date": session_date,
                        "symbol": symbol,
                        "signal_date": signal_date,
                        "signal_timestamp": signal_timestamp,
                        "signal": signal_value,
                        "signal_direction": "long" if position > 0 else "short" if position < 0 else "flat",
                        "position": position,
                        "previous_position": previous_position,
                        "holding_period": holding_period,
                        "capital_allocation": allocation,
                        "price_entry_date": session_date,
                        "price_exit_date": exit_date_value,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "realised_forward_return": realised_return,
                        "gross_return": gross_return,
                        "gross_pnl": gross_pnl,
                        "turnover": turnover,
                        "transaction_cost": transaction_cost,
                        "net_return": net_return,
                        "net_pnl": net_pnl,
                        "split": split,
                        "active_trade": position != 0,
                        "trade_executed": turnover > 0,
                        "timing_attrition_status": status,
                    }
                )
                previous_position = position
                if event is not None:
                    remaining = holding_period - 1
                elif remaining > 0:
                    remaining -= 1

    frame = pd.DataFrame(output, columns=STOCK_DAILY_COLUMNS)
    frame = frame.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    counts = AlignmentCounts(**{**asdict(counts), "boundary_attrited_signals": boundary_attrited})
    return frame, counts


def annualized_sharpe(
    returns: list[float] | np.ndarray | pd.Series,
    *,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> float | None:
    """Annualised arithmetic Sharpe from daily returns (sample volatility)."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    daily_rf = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    spread = float(np.std(values, ddof=1))
    if spread == 0.0:
        return None
    return float(np.mean(values - daily_rf) / spread * math.sqrt(periods_per_year))


def compounded_return(returns: list[float] | np.ndarray | pd.Series) -> float:
    """Compound a daily return sequence, preserving losses."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.prod(1.0 + values) - 1.0) if len(values) else 0.0


def maximum_drawdown(returns: list[float] | np.ndarray | pd.Series) -> float:
    """Maximum drawdown of a compounded wealth path, as a negative fraction."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def performance_metrics(
    returns: pd.Series,
    *,
    turnover: pd.Series,
    active: pd.Series,
    periods_per_year: int,
    annual_risk_free_rate: float,
) -> dict[str, Any]:
    """Return the common stock/portfolio metric set."""

    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    n = len(values)
    total_return = compounded_return(values)
    annual_return = (1.0 + total_return) ** (periods_per_year / n) - 1.0 if n and total_return > -1.0 else None
    volatility = float(values.std(ddof=1) * math.sqrt(periods_per_year)) if n >= 2 else None
    return {
        "observations": n,
        "sample_return": total_return,
        "annualized_return": annual_return,
        "annualized_sharpe": annualized_sharpe(
            values,
            periods_per_year=periods_per_year,
            annual_risk_free_rate=annual_risk_free_rate,
        ),
        "annualized_volatility": volatility,
        "maximum_drawdown": maximum_drawdown(values),
        "total_turnover": float(pd.to_numeric(turnover, errors="coerce").fillna(0.0).sum()),
        "average_daily_turnover": float(pd.to_numeric(turnover, errors="coerce").fillna(0.0).mean()) if n else 0.0,
        "active_days": int(pd.Series(active).fillna(False).astype(bool).sum()),
        "winning_days": int((values > 0).sum()),
        "losing_days": int((values < 0).sum()),
        "worst_daily_return": float(values.min()) if n else 0.0,
    }


def _development_subperiods(frame: pd.DataFrame, count: int = 3) -> list[pd.DataFrame]:
    dates = sorted(frame["date"].unique())
    chunks = np.array_split(np.asarray(dates, dtype=object), count)
    return [frame.loc[frame["date"].isin(chunk.tolist())] for chunk in chunks if len(chunk)]


def build_strategy_grid(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    config: Week6PnlConfig,
    split_date: str,
) -> tuple[pd.DataFrame, dict[tuple[float, int], pd.DataFrame], dict[tuple[float, int], AlignmentCounts]]:
    """Evaluate the small candidate grid on development data only."""

    rows: list[dict[str, Any]] = []
    frames: dict[tuple[float, int], pd.DataFrame] = {}
    counts: dict[tuple[float, int], AlignmentCounts] = {}
    for threshold in config.thresholds:
        for holding in config.holding_periods:
            frame, alignment = build_stock_daily_pnl(
                signals,
                prices,
                threshold=threshold,
                holding_period=holding,
                split_date=split_date,
                starting_capital=config.starting_capital,
                transaction_cost_bps_per_side=config.transaction_cost_bps_per_side,
                exchange_timezone=config.exchange_timezone,
            )
            frames[(threshold, holding)] = frame
            counts[(threshold, holding)] = alignment
            development = frame.loc[frame["split"] == "development"]
            stock_count = development["symbol"].nunique()
            daily = (
                development.groupby("date", sort=True)
                .agg(
                    net_return=("net_return", lambda values, n=stock_count: float(values.sum()) / n),
                    gross_return=("gross_return", lambda values, n=stock_count: float(values.sum()) / n),
                    turnover=("turnover", lambda values, n=stock_count: float(values.sum()) / n),
                    active=("active_trade", "any"),
                )
                .reset_index()
            )
            metrics = performance_metrics(
                daily["net_return"],
                turnover=daily["turnover"],
                active=daily["active"],
                periods_per_year=config.annualization_periods,
                annual_risk_free_rate=config.annual_risk_free_rate,
            )
            subperiod_sharpes: list[float | None] = []
            for subperiod in _development_subperiods(daily):
                subperiod_sharpes.append(
                    annualized_sharpe(
                        subperiod["net_return"],
                        periods_per_year=config.annualization_periods,
                        annual_risk_free_rate=config.annual_risk_free_rate,
                    )
                )
            finite_subperiods = [value for value in subperiod_sharpes if value is not None and math.isfinite(value)]
            rows.append(
                {
                    "threshold": threshold,
                    "holding_period": holding,
                    "weighting": "equal_position",
                    "development_observations": metrics["observations"],
                    "development_net_sample_return": metrics["sample_return"],
                    "development_net_annualized_return": metrics["annualized_return"],
                    "development_net_sharpe": metrics["annualized_sharpe"],
                    "development_net_volatility": metrics["annualized_volatility"],
                    "development_max_drawdown": metrics["maximum_drawdown"],
                    "development_turnover": metrics["total_turnover"],
                    "development_active_days": metrics["active_days"],
                    "subperiod_1_sharpe": subperiod_sharpes[0] if len(subperiod_sharpes) > 0 else None,
                    "subperiod_2_sharpe": subperiod_sharpes[1] if len(subperiod_sharpes) > 1 else None,
                    "subperiod_3_sharpe": subperiod_sharpes[2] if len(subperiod_sharpes) > 2 else None,
                    "median_subperiod_sharpe": float(np.median(finite_subperiods)) if finite_subperiods else None,
                    "worst_subperiod_sharpe": min(finite_subperiods) if finite_subperiods else None,
                    "selected": False,
                    "selection_reason": "",
                }
            )
    return pd.DataFrame(rows), frames, counts


def select_stable_strategy(grid: pd.DataFrame) -> tuple[float, int, pd.DataFrame]:
    """Freeze a candidate using development stability, never evaluation data."""

    required = {"threshold", "holding_period", "median_subperiod_sharpe", "worst_subperiod_sharpe", "development_net_sharpe"}
    if missing := required - set(grid.columns):
        raise Week6PnlError(f"strategy grid lacks selection fields: {sorted(missing)}")

    def rank(row: pd.Series) -> tuple[float, float, float, float, int, float]:
        def score(name: str) -> float:
            value = row[name]
            return float(value) if pd.notna(value) and math.isfinite(float(value)) else float("-inf")

        return (
            score("median_subperiod_sharpe"),
            score("worst_subperiod_sharpe"),
            score("development_net_sharpe"),
            -float(row.get("development_turnover", 0.0)),
            -int(row["holding_period"]),
            -float(row["threshold"]),
        )

    selected_index = max(grid.index, key=lambda index: rank(grid.loc[index]))
    selected = grid.copy()
    selected.loc[:, "selected"] = False
    selected.loc[:, "selection_reason"] = "development candidate; evaluation not consulted"
    reason = (
        "selected by highest median development-subperiod net Sharpe, then worst-subperiod Sharpe, "
        "full-development net Sharpe, lower turnover, shorter holding period, and lower threshold"
    )
    selected.loc[selected_index, "selected"] = True
    selected.loc[selected_index, "selection_reason"] = reason
    row = selected.loc[selected_index]
    return float(row["threshold"]), int(row["holding_period"]), selected


def development_correlations(
    stock_daily: pd.DataFrame,
    *,
    min_joint_active_dates: int,
) -> pd.DataFrame:
    """Pairwise development correlations with overlap and activity support."""

    development = stock_daily.loc[stock_daily["split"] == "development"]
    symbols = sorted(development["symbol"].unique())
    returns = development.pivot(index="date", columns="symbol", values="net_return")
    active_source = development[["date", "symbol", "active_trade"]].copy()
    active_source["active_trade"] = active_source["active_trade"].astype(int)
    active = active_source.pivot(index="date", columns="symbol", values="active_trade").fillna(0).astype(bool)
    active_counts = active.sum().to_dict()
    rows: list[dict[str, Any]] = []
    for index, symbol_a in enumerate(symbols):
        for symbol_b in symbols[index + 1 :]:
            pair = returns[[symbol_a, symbol_b]].dropna()
            jointly_active = int((active.loc[pair.index, symbol_a] & active.loc[pair.index, symbol_b]).sum())
            correlation = (
                float(pair[symbol_a].corr(pair[symbol_b]))
                if len(pair) >= 2 and pair[symbol_a].nunique() > 1 and pair[symbol_b].nunique() > 1
                else float("nan")
            )
            if not math.isfinite(correlation):
                status = "zero_variance_or_insufficient_overlap"
            elif jointly_active < min_joint_active_dates:
                status = "insufficient_joint_activity"
            else:
                status = "supported"
            rows.append(
                {
                    "symbol_a": symbol_a,
                    "symbol_b": symbol_b,
                    "pairwise_correlation": correlation if math.isfinite(correlation) else None,
                    "overlapping_dates": len(pair),
                    "jointly_active_dates": jointly_active,
                    "symbol_a_active_days": int(active_counts.get(symbol_a, 0)),
                    "symbol_b_active_days": int(active_counts.get(symbol_b, 0)),
                    "support_status": status,
                }
            )
    return pd.DataFrame(rows)


def select_low_correlation_stocks(
    correlations: pd.DataFrame,
    stock_daily: pd.DataFrame,
    *,
    requested_count: int | None = None,
    min_active_days: int = 20,
) -> pd.DataFrame:
    """Greedily choose a reproducible, development-only low-correlation subset."""

    development = stock_daily.loc[stock_daily["split"] == "development"]
    active_counts = development.groupby("symbol")["active_trade"].sum().astype(int).to_dict()
    supported = correlations.loc[correlations["support_status"] == "supported"]
    supported_symbols = set(supported["symbol_a"]) | set(supported["symbol_b"])
    eligible = sorted(symbol for symbol, count in active_counts.items() if count >= min_active_days and symbol in supported_symbols)
    if len(eligible) < 2:
        raise Week6PnlError("fewer than two stocks have enough development active days for diversification")
    target = requested_count if requested_count is not None else math.ceil(math.sqrt(len(eligible)))
    target = max(2, min(int(target), len(eligible)))
    lookup: dict[tuple[str, str], float] = {}
    for row in supported.to_dict("records"):
        a, b = sorted((str(row["symbol_a"]), str(row["symbol_b"])))
        lookup[(a, b)] = float(row["pairwise_correlation"])

    def corr(a: str, b: str) -> float | None:
        key = tuple(sorted((a, b)))
        return lookup.get((key[0], key[1]))

    universe_abs_mean: dict[str, float] = {}
    for symbol in eligible:
        values = [abs(value) for other in eligible if other != symbol for value in [corr(symbol, other)] if value is not None]
        universe_abs_mean[symbol] = fmean(values) if values else 1.0
    first = min(eligible, key=lambda symbol: (universe_abs_mean[symbol], -active_counts[symbol], symbol))
    selected = [first]
    evidence: list[dict[str, Any]] = []

    def evidence_row(symbol: str, order: int) -> dict[str, Any]:
        values = [abs(value) for other in selected if other != symbol and (value := corr(symbol, other)) is not None]
        return {
            "selection_order": order,
            "symbol": symbol,
            "development_active_days": active_counts[symbol],
            "mean_absolute_correlation_to_prior_selected": fmean(values) if values else 0.0,
            "maximum_absolute_correlation_to_prior_selected": max(values) if values else 0.0,
            "universe_mean_absolute_supported_correlation": universe_abs_mean[symbol],
            "target_stock_count": target,
            "selection_rule": (
                "development-only greedy minimum mean absolute supported correlation; "
                "unsupported pairs rejected; ties prefer more active days then symbol"
            ),
        }

    evidence.append(evidence_row(first, 1))
    while len(selected) < target:
        remaining = [symbol for symbol in eligible if symbol not in selected and all(corr(symbol, other) is not None for other in selected)]
        if not remaining:
            break

        def candidate_rank(symbol: str) -> tuple[float, float, int, str]:
            values = [abs(value) if (value := corr(symbol, other)) is not None else 1.0 for other in selected]
            return (fmean(values), max(values), -active_counts[symbol], symbol)

        chosen = min(remaining, key=candidate_rank)
        selected.append(chosen)
        evidence.append(evidence_row(chosen, len(selected)))
    if len(selected) < 2:
        raise Week6PnlError("development correlations do not support a low-correlation stock pair")
    result = pd.DataFrame(evidence)
    result["actual_stock_count"] = len(selected)
    return result


def build_portfolio_daily(stock_daily: pd.DataFrame, stock_sets: dict[str, list[str]], *, starting_capital: float) -> pd.DataFrame:
    """Build equal-weight fixed-notional and compounded funded portfolio series."""

    rows: list[dict[str, Any]] = []
    for portfolio_name, symbols in stock_sets.items():
        subset = stock_daily.loc[stock_daily["symbol"].isin(symbols)].copy()
        weight = 1.0 / len(symbols)
        daily = (
            subset.groupby(["date", "split"], sort=True)
            .agg(
                gross_return=("gross_return", "sum"),
                net_return=("net_return", "sum"),
                turnover=("turnover", "sum"),
                active_days=("active_trade", "sum"),
            )
            .reset_index()
        )
        daily["gross_return"] *= weight
        daily["net_return"] *= weight
        daily["turnover"] *= weight
        daily["fixed_notional_gross_pnl"] = starting_capital * daily["gross_return"]
        daily["fixed_notional_net_pnl"] = starting_capital * daily["net_return"]
        gross_value = starting_capital
        net_value = starting_capital
        running_peak = starting_capital
        cumulative_fixed_gross = 0.0
        cumulative_fixed_net = 0.0
        for item in daily.to_dict("records"):
            start_gross = gross_value
            start_net = net_value
            gross_wealth_pnl = start_gross * float(item["gross_return"])
            gross_pnl = start_net * float(item["gross_return"])
            net_pnl = start_net * float(item["net_return"])
            transaction_cost_rate = max(0.0, float(item["gross_return"] - item["net_return"]))
            transaction_cost = start_net * transaction_cost_rate
            gross_value += gross_wealth_pnl
            net_value += net_pnl
            running_peak = max(running_peak, net_value)
            cumulative_fixed_gross += float(item["fixed_notional_gross_pnl"])
            cumulative_fixed_net += float(item["fixed_notional_net_pnl"])
            rows.append(
                {
                    "date": item["date"],
                    "split": item["split"],
                    "portfolio": portfolio_name,
                    "selected_stock_count": len(symbols),
                    "weight_per_stock": weight,
                    "starting_capital": starting_capital,
                    "start_portfolio_value": start_net,
                    "daily_gross_return": item["gross_return"],
                    "daily_net_return": item["net_return"],
                    "daily_gross_pnl": gross_pnl,
                    "daily_net_pnl": net_pnl,
                    "transaction_cost": transaction_cost,
                    "gross_wealth_pnl": gross_wealth_pnl,
                    "fixed_notional_gross_pnl": item["fixed_notional_gross_pnl"],
                    "fixed_notional_net_pnl": item["fixed_notional_net_pnl"],
                    "fixed_notional_transaction_cost": (float(item["fixed_notional_gross_pnl"]) - float(item["fixed_notional_net_pnl"])),
                    "cumulative_fixed_notional_gross_pnl": cumulative_fixed_gross,
                    "cumulative_fixed_notional_net_pnl": cumulative_fixed_net,
                    "cumulative_gross_pnl": gross_value - starting_capital,
                    "cumulative_net_pnl": net_value - starting_capital,
                    "gross_portfolio_value": gross_value,
                    "portfolio_value": net_value,
                    "running_peak": running_peak,
                    "drawdown": net_value / running_peak - 1.0,
                    "turnover": item["turnover"],
                    "transaction_cost_rate": transaction_cost_rate,
                    "active_stock_count": int(item["active_days"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "portfolio"], kind="stable").reset_index(drop=True)


def build_stock_performance(stock_daily: pd.DataFrame, config: Week6PnlConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in sorted(stock_daily["symbol"].unique()):
        symbol_rows = stock_daily.loc[stock_daily["symbol"] == symbol]
        allocation = float(symbol_rows["capital_allocation"].iloc[0])
        for period in ("development", "evaluation"):
            period_rows = symbol_rows.loc[symbol_rows["split"] == period]
            for variant in ("gross", "net"):
                metrics = performance_metrics(
                    period_rows[f"{variant}_return"],
                    turnover=period_rows["turnover"],
                    active=period_rows["active_trade"],
                    periods_per_year=config.annualization_periods,
                    annual_risk_free_rate=config.annual_risk_free_rate,
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "period": period,
                        "return_variant": variant,
                        "capital_allocation": allocation,
                        "sample_total_profit": float(period_rows[f"{variant}_pnl"].sum()),
                        "actual_sample_percentage_return": metrics["sample_return"] * 100,
                        "annualized_percentage_return": (
                            metrics["annualized_return"] * 100 if metrics["annualized_return"] is not None else None
                        ),
                        "annualized_sharpe": metrics["annualized_sharpe"],
                        "annualized_volatility": (
                            metrics["annualized_volatility"] * 100 if metrics["annualized_volatility"] is not None else None
                        ),
                        "maximum_drawdown": metrics["maximum_drawdown"] * 100,
                        "turnover": metrics["total_turnover"],
                        "active_days": metrics["active_days"],
                        "winning_days": metrics["winning_days"],
                        "losing_days": metrics["losing_days"],
                        "observations": metrics["observations"],
                    }
                )
    return pd.DataFrame(rows)


def build_portfolio_performance(portfolio_daily: pd.DataFrame, config: Week6PnlConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for portfolio in sorted(portfolio_daily["portfolio"].unique()):
        portfolio_rows = portfolio_daily.loc[portfolio_daily["portfolio"] == portfolio]
        for period in ("development", "evaluation"):
            period_rows = portfolio_rows.loc[portfolio_rows["split"] == period]
            for variant in ("gross", "net"):
                returns = period_rows[f"daily_{variant}_return"]
                metrics = performance_metrics(
                    returns,
                    turnover=period_rows["turnover"],
                    active=period_rows["active_stock_count"] > 0,
                    periods_per_year=config.annualization_periods,
                    annual_risk_free_rate=config.annual_risk_free_rate,
                )
                rows.append(
                    {
                        "portfolio": portfolio,
                        "period": period,
                        "return_variant": variant,
                        "stock_count": int(period_rows["selected_stock_count"].iloc[0]),
                        "sample_total_profit": config.starting_capital * metrics["sample_return"],
                        "actual_sample_percentage_return": metrics["sample_return"] * 100,
                        "annualized_percentage_return": (
                            metrics["annualized_return"] * 100 if metrics["annualized_return"] is not None else None
                        ),
                        "annualized_sharpe": metrics["annualized_sharpe"],
                        "annualized_volatility": (
                            metrics["annualized_volatility"] * 100 if metrics["annualized_volatility"] is not None else None
                        ),
                        "maximum_drawdown": metrics["maximum_drawdown"] * 100,
                        "turnover": metrics["total_turnover"],
                        "average_daily_turnover": metrics["average_daily_turnover"],
                        "active_days": metrics["active_days"],
                        "winning_days": metrics["winning_days"],
                        "losing_days": metrics["losing_days"],
                        "worst_daily_loss": float(period_rows[f"daily_{variant}_pnl"].min()),
                        "worst_daily_return_percentage": metrics["worst_daily_return"] * 100,
                        "observations": metrics["observations"],
                    }
                )
    return pd.DataFrame(rows)


def build_manual_timing_check(stock_daily: pd.DataFrame, *, cost_rate: float, observations: int = 5) -> pd.DataFrame:
    """Create and verify a five-row arithmetic timing audit for one stock."""

    evaluation = stock_daily.loc[stock_daily["split"] == "evaluation"]
    candidates = []
    for symbol in sorted(evaluation["symbol"].unique()):
        symbol_rows = evaluation.loc[evaluation["symbol"] == symbol].reset_index(drop=True)
        active_indices = symbol_rows.index[symbol_rows["active_trade"]].tolist()
        if active_indices and len(symbol_rows) >= observations:
            start = min(max(0, active_indices[0] - 1), len(symbol_rows) - observations)
            candidates.append((symbol, start))
    if not candidates:
        raise Week6PnlError("cannot find five consecutive evaluation observations for the manual timing check")
    symbol, start = candidates[0]
    sample = evaluation.loc[evaluation["symbol"] == symbol].reset_index(drop=True).iloc[start : start + observations]
    audit_rows: list[dict[str, Any]] = []
    failed = False
    for item in sample.to_dict("records"):
        manual_return = float(item["exit_price"]) / float(item["entry_price"]) - 1.0 if item["price_exit_date"] else 0.0
        manual_gross_pnl = float(item["capital_allocation"]) * float(item["position"]) * manual_return
        manual_cost = float(item["capital_allocation"]) * float(item["turnover"]) * cost_rate
        manual_net_pnl = manual_gross_pnl - manual_cost
        passed = (
            math.isclose(manual_return, float(item["realised_forward_return"]), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(manual_gross_pnl, float(item["gross_pnl"]), rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(manual_cost, float(item["transaction_cost"]), rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(manual_net_pnl, float(item["net_pnl"]), rel_tol=0.0, abs_tol=1e-8)
        )
        failed = failed or not passed
        audit_rows.append(
            {
                "symbol": symbol,
                "signal_date": item["signal_date"],
                "signal_timestamp": item["signal_timestamp"],
                "price_entry_date": item["price_entry_date"],
                "price_exit_date": item["price_exit_date"],
                "entry_price": item["entry_price"],
                "exit_price": item["exit_price"],
                "manually_recomputed_return": manual_return,
                "program_return": item["realised_forward_return"],
                "position": item["position"],
                "gross_pnl": item["gross_pnl"],
                "turnover": item["turnover"],
                "cost": item["transaction_cost"],
                "net_pnl": item["net_pnl"],
                "pass_fail": "pass" if passed else "fail",
            }
        )
    if failed:
        raise Week6PnlError(f"manual timing check failed for {symbol}")
    return pd.DataFrame(audit_rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _plot_outputs(portfolio_daily: pd.DataFrame, correlations: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependency failure
        raise Week6PnlError("matplotlib is required for Week 6 figures; install the figures extra") from exc

    def save_line(filename: str, y: str, title: str, ylabel: str, data: pd.DataFrame) -> None:
        fig, axis = plt.subplots(figsize=(10, 5.5))
        for name, group in data.groupby("portfolio", sort=True):
            axis.plot(pd.to_datetime(group["date"]), group[y], label=name, linewidth=1.7)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set(title=title, xlabel="Date", ylabel=ylabel)
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    save_line("cumulative_pnl.png", "cumulative_net_pnl", "Week 6 cumulative net P&L", "Net P&L (£)", portfolio_daily)
    evaluation = portfolio_daily.loc[portfolio_daily["split"] == "evaluation"].copy()
    starting_capital = float(portfolio_daily["starting_capital"].iloc[0])
    evaluation["evaluation_cumulative_net_pnl"] = evaluation.groupby("portfolio")["daily_net_return"].transform(
        lambda values: (1.0 + values).cumprod() * starting_capital - starting_capital
    )
    save_line(
        "cumulative_pnl_evaluation.png",
        "evaluation_cumulative_net_pnl",
        "Frozen-rule evaluation cumulative net P&L (rebased)",
        "Net P&L (£)",
        evaluation,
    )
    save_line("drawdown.png", "drawdown", "Week 6 net portfolio drawdown", "Drawdown", portfolio_daily)

    symbols = sorted(set(correlations["symbol_a"]) | set(correlations["symbol_b"]))
    matrix = pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
    for row in correlations.to_dict("records"):
        value = row["pairwise_correlation"]
        if value is not None and _finite(value):
            matrix.loc[row["symbol_a"], row["symbol_b"]] = float(value)
            matrix.loc[row["symbol_b"], row["symbol_a"]] = float(value)
    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(symbols)), labels=symbols, rotation=90, fontsize=7)
    axis.set_yticks(range(len(symbols)), labels=symbols, fontsize=7)
    axis.set_title("Development daily stock net-return correlations")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "pnl_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def _git_metadata(cwd: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd, check=True, capture_output=True, text=True).stdout)
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
        return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "branch": None, "dirty": None}


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None or not _finite(value):
        return "n/a"
    return f"{float(value):.3f}{'%' if percent else ''}"


def _write_summary(
    path: Path,
    *,
    config: Week6PnlConfig,
    signals_path: Path,
    prices_path: Path,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    split_date: str,
    threshold: float,
    holding: int,
    selected_stocks: list[str],
    alignment: AlignmentCounts,
    portfolio_performance: pd.DataFrame,
) -> None:
    net = portfolio_performance.loc[portfolio_performance["return_variant"] == "net"]

    def metric(portfolio: str, period: str, name: str) -> Any:
        row = net.loc[(net["portfolio"] == portfolio) & (net["period"] == period)]
        return None if row.empty else row.iloc[0][name]

    dev_sharpe = metric("all_stock_equal_weight", "development", "annualized_sharpe")
    eval_sharpe = metric("all_stock_equal_weight", "evaluation", "annualized_sharpe")
    reproduces = bool(_finite(dev_sharpe) and _finite(eval_sharpe) and float(dev_sharpe) > 0 and float(eval_sharpe) < 0)
    rows = []
    for portfolio in sorted(net["portfolio"].unique()):
        for period in ("development", "evaluation"):
            rows.append(
                "| "
                + " | ".join(
                    [
                        portfolio,
                        period,
                        _fmt(metric(portfolio, period, "actual_sample_percentage_return"), percent=True),
                        _fmt(metric(portfolio, period, "annualized_sharpe")),
                        _fmt(metric(portfolio, period, "annualized_volatility"), percent=True),
                        _fmt(metric(portfolio, period, "maximum_drawdown"), percent=True),
                        _fmt(metric(portfolio, period, "turnover")),
                        _fmt(metric(portfolio, period, "worst_daily_return_percentage"), percent=True),
                    ]
                )
                + " |"
            )
    text = f"""# Week 6 exploratory daily P&L analysis

## Scope and data

This is exploratory supervisor-assigned backtest evidence, not causal evidence or
deployable alpha. It does not alter the frozen dissertation event study and performs
no external scoring or model calls.

- Signals: `{signals_path}` (`{config.scorer_id}`), {len(signals):,} selected rows,
  {signals["symbol"].nunique()} stocks, {signals["news_date"].min()} to {signals["news_date"].max()}.
- Prices: `{prices_path}`, {len(prices):,} daily bars,
  {prices["session_date"].min()} to {prices["session_date"].max()}.
- Price convention: daily close prices from the supplied panel; this LSEG panel uses
  price returns rather than dividend-adjusted total returns.
- Starting capital: £{config.starting_capital:,.0f}; equal fixed allocation across each portfolio's frozen stock set.
- Costs: {config.transaction_cost_bps_per_side:g} bps per side on `abs(position_t - position_t-1)`; direct long-to-short turnover is 2.
- Risk-free rate: {config.annual_risk_free_rate:g}; annualisation: {config.annualization_periods} trading days.

## Timing and split

The source is a day-level aggregate and has no usable per-signal timestamp for this
scorer. Each signal is therefore actioned conservatively at the next observed
trading-session close and earns only later close-to-close returns. A newer mapped
signal supersedes an overlapping holding; an unchanged position has zero turnover.
Positions are forced flat inside each split, and signals unable to complete their
holding within that split are attrited.

The chronological development/evaluation boundary is `{split_date}` (first evaluation
date). Every threshold/holding candidate was evaluated only on dates before that
boundary. The frozen rule was run once on evaluation dates; evaluation results were
not written into the candidate grid and were not used for stock selection.

## Frozen strategy

- Candidate thresholds: {", ".join(f"{value:g}" for value in config.thresholds)} (scores lie in [-1, 1]).
- Candidate holding periods: {", ".join(str(value) for value in config.holding_periods)} sessions.
- Weighting: equal signed positions only; magnitude weighting was intentionally excluded from this transparent baseline.
- Selected threshold: `{threshold:g}`; selected holding period: `{holding}` sessions.
- Selection rule: highest median of three chronological development-subperiod net
  Sharpes, then worst-subperiod Sharpe, full-development net Sharpe, lower turnover,
  shorter holding, and lower threshold.
{
        f"- Low-correlation stocks frozen from development only ({len(selected_stocks)}): {', '.join(selected_stocks)}."
        if config.include_low_correlation_portfolio
        else "- The optional low-correlation portfolio was disabled; the all-stock book is the only acceptance portfolio."
    }

## Gross and net portfolio evidence

| Portfolio | Period | Net sample return | Net Sharpe | Annual volatility | Max drawdown | Turnover | Worst day |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{os.linesep.join(rows)}

Gross and net variants are both retained in `portfolio_performance.csv`; the table
above focuses on the cost-aware net result. `portfolio_daily_pnl.csv` contains both
fixed-notional monetary P&L and fixed-capital compounded wealth.

The previously observed development-positive/evaluation-negative sign pattern is
**{"reproduced" if reproduces else "not reproduced exactly"}** for the all-stock net
annualised Sharpe. This is reported mechanically, without reselection.

## Attrition and missingness

- Input selected-scorer signals: {alignment.input_signals:,}.
- Signals aligned after availability: {alignment.aligned_signals:,}.
- Missing-score signals mapped flat: {alignment.missing_score_signals:,}.
- Signals without a future price session: {alignment.no_future_price_signals:,}.
- Same-entry signals superseded by a more recent signal: {alignment.superseded_same_entry_signals:,}.
- Signals attrited because the full holding would cross the split/sample boundary: {alignment.boundary_attrited_signals:,}.

Correlations use development daily stock net returns, not cumulative P&L curves.
`pnl_correlations.csv` reports overlapping and jointly active observations; pairs
below {config.min_joint_active_dates} jointly active dates are flagged and excluded
from low-correlation selection.

## Limitations

- Day-level aggregate signals do not preserve the underlying news timestamp;
  next-session-close entry is conservative but cannot recover intraday reaction timing.
- Close prices omit dividends in this LSEG panel, short borrow, taxes, market impact,
  liquidity limits, and execution slippage beyond the stated cost.
{
        (
            "- The low-correlation subset is a simple development-only greedy screen, not an optimiser. "
            "Sparse zero-position days can depress apparent correlations."
        )
        if config.include_low_correlation_portfolio
        else "- No stock subset was selected; every stock remains in the all-stock portfolio."
    }
- This period and headline lexicon signal were previously explored. Evaluation is
  leakage-controlled within this command, but it is not a pristine confirmatory
  holdout for the dissertation.
"""
    path.write_text(text, encoding="utf-8")


def run_week6_pnl(
    signals_path: str | Path,
    prices_path: str | Path,
    output_root: str | Path,
    config: Week6PnlConfig,
    *,
    command: str | None = None,
    repo_root: str | Path | None = None,
    generated_at: str | None = None,
) -> Week6RunResult:
    """Execute the complete Week 6 analysis into a new, immutable run directory."""

    output_base = Path(output_root).expanduser().resolve()
    final_dir = output_base / config.run_id
    temporary_dir = output_base / f".{config.run_id}.tmp"
    if final_dir.exists() or temporary_dir.exists():
        raise Week6PnlError(f"refusing to overwrite existing Week 6 output: {final_dir}")
    output_base.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()

    try:
        signal_file = Path(signals_path).expanduser().resolve()
        price_file = Path(prices_path).expanduser().resolve()
        signals, prices = load_and_validate_inputs(signal_file, price_file, config.scorer_id)
        split_date = chronological_split_date(signals["news_date"].astype(str).tolist(), config.development_fraction)
        grid, candidate_frames, candidate_counts = build_strategy_grid(signals, prices, config=config, split_date=split_date)
        threshold, holding, grid = select_stable_strategy(grid)
        stock_daily = candidate_frames[(threshold, holding)]
        alignment = candidate_counts[(threshold, holding)]
        correlations = development_correlations(stock_daily, min_joint_active_dates=config.min_joint_active_dates)
        if config.include_low_correlation_portfolio:
            selected_frame = select_low_correlation_stocks(
                correlations,
                stock_daily,
                requested_count=config.low_correlation_stock_count,
                min_active_days=config.min_joint_active_dates,
            )
            selected_stocks = selected_frame["symbol"].astype(str).tolist()
        else:
            selected_frame = pd.DataFrame(
                columns=[
                    "selection_order",
                    "symbol",
                    "development_active_days",
                    "mean_absolute_correlation_to_prior_selected",
                    "maximum_absolute_correlation_to_prior_selected",
                    "universe_mean_absolute_supported_correlation",
                    "target_stock_count",
                    "selection_rule",
                    "actual_stock_count",
                ]
            )
            selected_stocks = []
        all_stocks = sorted(stock_daily["symbol"].unique())
        stock_sets = {"all_stock_equal_weight": all_stocks}
        if config.include_low_correlation_portfolio:
            stock_sets["development_low_correlation"] = selected_stocks
        portfolio_daily = build_portfolio_daily(
            stock_daily,
            stock_sets,
            starting_capital=config.starting_capital,
        )
        stock_performance = build_stock_performance(stock_daily, config)
        portfolio_performance = build_portfolio_performance(portfolio_daily, config)
        manual = build_manual_timing_check(
            stock_daily,
            cost_rate=config.transaction_cost_bps_per_side / 10_000.0,
        )

        _write_csv(stock_daily, temporary_dir / "stock_daily_pnl.csv")
        _write_csv(portfolio_daily, temporary_dir / "portfolio_daily_pnl.csv")
        _write_csv(stock_performance, temporary_dir / "stock_performance.csv")
        _write_csv(portfolio_performance, temporary_dir / "portfolio_performance.csv")
        _write_csv(grid, temporary_dir / "strategy_grid.csv")
        _write_csv(correlations, temporary_dir / "pnl_correlations.csv")
        _write_csv(selected_frame, temporary_dir / "selected_low_correlation_stocks.csv")
        _write_csv(manual, temporary_dir / "manual_timing_check.csv")
        _plot_outputs(portfolio_daily, correlations, temporary_dir)
        _write_summary(
            temporary_dir / "summary.md",
            config=config,
            signals_path=signal_file,
            prices_path=price_file,
            signals=signals,
            prices=prices,
            split_date=split_date,
            threshold=threshold,
            holding=holding,
            selected_stocks=selected_stocks,
            alignment=alignment,
            portfolio_performance=portfolio_performance,
        )
        root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
        output_hashes = {
            path.name: _sha256(path)
            for path in sorted(temporary_dir.iterdir(), key=lambda value: value.name)
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "schema_version": 1,
            "run_id": config.run_id,
            "generated_at": generated_at or datetime.now(UTC).isoformat(),
            "git": _git_metadata(root),
            "command": command or shlex.join(["sentiment-bench", "analyze-week6-pnl"]),
            "inputs": {
                "daily_signals": {"path": str(signal_file), "sha256": _sha256(signal_file)},
                "prices": {"path": str(price_file), "sha256": _sha256(price_file)},
            },
            "configuration": asdict(config),
            "development_evaluation": {
                "split_date": split_date,
                "development_start": str(stock_daily.loc[stock_daily["split"] == "development", "date"].min()),
                "development_end": str(stock_daily.loc[stock_daily["split"] == "development", "date"].max()),
                "evaluation_start": str(stock_daily.loc[stock_daily["split"] == "evaluation", "date"].min()),
                "evaluation_end": str(stock_daily.loc[stock_daily["split"] == "evaluation", "date"].max()),
            },
            "transaction_cost_assumption": {
                "bps_per_side": config.transaction_cost_bps_per_side,
                "formula": "capital_allocation * abs(position_t - position_t_minus_1) * bps_per_side / 10000",
            },
            "return_timing": (
                "untimestamped daily signal -> next observed session close; position earns only that close to next close; "
                "full holding must remain inside one chronological split"
            ),
            "selected_rule": {
                "threshold": threshold,
                "holding_period": holding,
                "weighting": "equal_position",
                "selection_data": "development_only",
            },
            "selected_stock_set": selected_stocks,
            "alignment_counts": asdict(alignment),
            "output_hashes": output_hashes,
        }
        (temporary_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_dir.rename(final_dir)
        return Week6RunResult(final_dir, split_date, threshold, holding, tuple(selected_stocks))
    except Exception:
        # Remove only the fresh staging directory created by this invocation.
        if temporary_dir.exists():
            for path in temporary_dir.iterdir():
                if path.is_file():
                    path.unlink()
            temporary_dir.rmdir()
        raise
