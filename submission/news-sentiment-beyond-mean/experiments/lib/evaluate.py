"""Shared evaluation helpers for experiments arms.

Provides (1) daily cross-sectional rank IC with HAC inference, (2) a
simple cross-sectional long/short translation of a firm-day signal into daily
gross/net returns with turnover and break-even cost, and (3) a date-block
bootstrap of mean daily net return.

Designed for the FNSPID news-bearing panel: trade only names with a signal on
that session; names absent that day have weight 0. Holding is one session
(``ar_open_h1``). Evaluation-block metrics may be computed for disclosure but
must not be used to select aggregators or thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from experiments.lib.aggregators import (
    AGGREGATOR_NAMES,
    MULTIPLICITY_FAMILY,
    PRIMARY_OUTCOME,
    benjamini_hochberg,
    date_clustered_rank_ic,
    development_ic_table,
)

# Economic orientation before signing: +1 means higher signal → long.
# Negative share / dispersion have negative development IC vs AR by construction
# of the estimand (more negativity / disagreement ranks with worse AR).
DEFAULT_SIGNAL_ORIENT: dict[str, float] = {
    "mean_hard_label": 1.0,
    "mean_continuous": 1.0,
    "median_continuous": 1.0,
    "trimmed_mean": 1.0,
    "negative_share": -1.0,
    "dispersion": -1.0,
    "strongest_event": 1.0,
    "attention_log_n": 1.0,
    "decayed_state": 1.0,
}


#: ``sign`` reproduces the first W3 pass: position = sign(oriented signal) after
#: a symmetric band. It is only meaningful for signals that straddle zero, and it
#: discards magnitude, so any rule that is a sign-preserving transform of another
#: (``attention_log_n`` vs ``mean_continuous``) collapses onto the same book.
#: ``cs_rank`` ranks the oriented signal *within the session*, centres the rank on
#: [-1, 1] and weights proportionally. That is dollar-neutral by construction,
#: keeps magnitude ordering, and is defined for one-sided signals such as
#: ``negative_share`` and ``dispersion``.
POSITION_MODES: tuple[str, ...] = ("sign", "cs_rank")


class PortfolioInsolvencyError(ValueError):
    """Raised when a leveraged book has non-positive end-of-period wealth."""

    def __init__(
        self,
        wealth_growth: float,
        *,
        session: object | None = None,
    ) -> None:
        self.wealth_growth = float(wealth_growth)
        self.session = session
        where = f" on {session}" if session is not None else ""
        super().__init__(
            f"portfolio wealth is non-positive{where}: "
            f"growth={self.wealth_growth:.12g}"
        )


@dataclass(frozen=True)
class TradeConfig:
    """Fixed one-session long/short rule used for every aggregator arm."""

    threshold: float = 0.0
    cost_bps_per_side: float = 10.0
    min_names: int = 2
    annualization_periods: int = 252
    bootstrap_block_length: int = 5
    bootstrap_replications: int = 999
    random_seed: int = 20260731
    outcome: str = PRIMARY_OUTCOME
    asset_return: str = "ret_open_h1"
    position_mode: str = "cs_rank"

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.cost_bps_per_side < 0:
            raise ValueError("cost_bps_per_side must be non-negative")
        if self.min_names < 1:
            raise ValueError("min_names must be positive")
        if self.bootstrap_block_length < 1:
            raise ValueError("bootstrap_block_length must be positive")
        if self.bootstrap_replications < 1:
            raise ValueError("bootstrap_replications must be positive")
        if self.position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode must be one of {POSITION_MODES}")


def signal_to_position(signal: float, *, threshold: float = 0.0, orient: float = 1.0) -> int:
    """Symmetric no-trade band after orientation."""
    if signal is None or not math.isfinite(float(signal)):
        return 0
    signed = float(orient) * float(signal)
    if signed > threshold:
        return 1
    if signed < -threshold:
        return -1
    return 0


def assign_positions(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    threshold: float = 0.0,
    orient: float = 1.0,
) -> pd.Series:
    return frame[signal_col].map(
        lambda x: signal_to_position(x, threshold=threshold, orient=orient)
    )


def _normalize_gross(raw: np.ndarray) -> np.ndarray:
    """Scale to gross exposure 1: ``w_i = raw_i / sum(|raw|)``."""
    abs_sum = float(np.abs(raw).sum())
    if abs_sum <= 0:
        return np.zeros_like(raw, dtype=float)
    return raw.astype(float) / abs_sum


def _day_weights(positions: np.ndarray) -> np.ndarray:
    """Gross-exposure-1 weights from -1/0/+1 positions."""
    return _normalize_gross(np.asarray(positions, dtype=float))


def drifted_weight_map(
    symbols: Sequence[str],
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> dict[str, float]:
    """Return end-of-period weights after asset returns move the target book.

    Weights are normalised by portfolio wealth growth, so the next rebalance is
    measured against the holdings that actually arrive at the following open.
    Transaction costs remain a separate return deduction and do not alter this
    standard pre-cost drift calculation.
    """
    w = np.asarray(weights, dtype=float)
    r = np.asarray(asset_returns, dtype=float)
    names = np.asarray(symbols, dtype=str)
    if not (len(names) == len(w) == len(r)):
        raise ValueError("symbols, weights and asset_returns must have equal length")
    if not np.isfinite(w).all() or not np.isfinite(r).all():
        raise ValueError("weights and asset_returns must be finite")
    if (r <= -1.0).any():
        raise ValueError("asset_returns must be greater than -1")

    wealth_growth = 1.0 + float(np.dot(w, r))
    if not math.isfinite(wealth_growth) or wealth_growth <= 0.0:
        raise PortfolioInsolvencyError(wealth_growth)
    drifted = w * (1.0 + r) / wealth_growth
    return {
        symbol: float(weight)
        for symbol, weight in zip(names, drifted, strict=True)
        if weight != 0.0
    }


def cross_sectional_rank_scores(values: np.ndarray) -> np.ndarray:
    """Centre a session's signal on [-1, 1] by within-day average rank.

    The median name gets 0, the best gets +1, the worst -1. Ties share a rank, so
    a signal that is constant across the session's names produces all-zero scores
    and therefore no trade. This is what makes one-sided signals (``dispersion``,
    ``negative_share``) tradable: what matters is being *more* dispersed than the
    other names with news today, not being dispersed at all.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return x
    if n == 1:
        return np.zeros(1, dtype=float)
    ranks = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    return 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0


def _cs_rank_weights(
    values: np.ndarray, *, threshold: float = 0.0, orient: float = 1.0
) -> np.ndarray:
    """Dollar-neutral, magnitude-preserving weights from within-day ranks."""
    z = cross_sectional_rank_scores(float(orient) * np.asarray(values, dtype=float))
    z = np.where(np.abs(z) <= threshold, 0.0, z)
    return _normalize_gross(z)


def build_daily_portfolio(
    firm_day: pd.DataFrame,
    signal_col: str,
    *,
    config: TradeConfig | None = None,
    orient: float = 1.0,
    split: str | None = "development",
) -> pd.DataFrame:
    """Cross-sectional daily long/short portfolio from firm-day signals.

    Returns one row per session with gross/net return and turnover. Rebalancing
    compares each target with the prior book after raw asset returns have moved
    its weights. The final row also includes liquidation at the ending open.
    """
    cfg = config or TradeConfig()
    need = {"session_date", "symbol", signal_col, cfg.outcome, cfg.asset_return}
    missing = need - set(firm_day.columns)
    if missing:
        raise ValueError(f"firm_day missing columns: {sorted(missing)}")

    frame = firm_day
    if split is not None:
        if "split" not in frame.columns:
            raise ValueError("split column required when split filter is set")
        frame = frame.loc[frame["split"] == split]
    frame = frame.dropna(subset=[signal_col, cfg.outcome, cfg.asset_return]).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "split",
                "n_names",
                "n_long",
                "n_short",
                "net_exposure",
                "gross_return",
                "turnover",
                "cost",
                "net_return",
            ]
        )

    frame = frame.sort_values(["session_date", "symbol"], kind="mergesort")

    prev_drifted_weights: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    cost_rate = cfg.cost_bps_per_side / 10_000.0

    for session, day in frame.groupby("session_date", sort=True):
        symbols = day["symbol"].astype(str).to_numpy()
        rets = day[cfg.outcome].to_numpy(dtype=float)
        asset_rets = day[cfg.asset_return].to_numpy(dtype=float)
        values = day[signal_col].to_numpy(dtype=float)

        if cfg.position_mode == "cs_rank":
            w = _cs_rank_weights(values, threshold=cfg.threshold, orient=orient)
        else:
            pos = np.array(
                [
                    signal_to_position(v, threshold=cfg.threshold, orient=orient)
                    for v in values
                ],
                dtype=int,
            )
            w = _day_weights(pos)

        if int((w != 0).sum()) < cfg.min_names:
            # Too few tradable names: go flat, but still pay to exit yesterday.
            w = np.zeros_like(w)

        weight_map = {s: float(wi) for s, wi in zip(symbols, w, strict=True) if wi != 0.0}
        # One-sided turnover against yesterday's return-drifted book. Names
        # absent from either side count once via ``.get(s, 0.0)``.
        names = set(prev_drifted_weights) | set(weight_map)
        turnover = 0.5 * sum(
            abs(weight_map.get(s, 0.0) - prev_drifted_weights.get(s, 0.0)) for s in names
        )
        gross = float(np.dot(w, rets))
        # ``turnover`` is half-L1. A quoted per-side cost is paid on both the
        # buy and sell notional, hence the factor 2. Opening or closing a
        # gross-1 dollar-neutral book has turnover 0.5 and costs exactly one
        # gross-notional × the per-side rate.
        cost = float(2.0 * turnover * cost_rate)
        rows.append(
            {
                "session_date": session,
                "split": day["split"].iloc[0] if "split" in day.columns else split,
                "n_names": int(len(day)),
                "n_long": int((w > 0).sum()),
                "n_short": int((w < 0).sum()),
                "net_exposure": float(w.sum()),
                "gross_return": gross,
                "turnover": float(turnover),
                "cost": cost,
                "net_return": gross - cost,
            }
        )
        try:
            prev_drifted_weights = drifted_weight_map(symbols, w, asset_rets)
        except PortfolioInsolvencyError as exc:
            raise PortfolioInsolvencyError(exc.wealth_growth, session=session) from exc

    # Close the final book at the last interval's ending open. Without this
    # charge, a backtest receives its final return while leaving the holdings
    # open, which understates both turnover and costs.
    if rows and prev_drifted_weights:
        liquidation_turnover = 0.5 * sum(
            abs(weight) for weight in prev_drifted_weights.values()
        )
        liquidation_cost = 2.0 * liquidation_turnover * cost_rate
        rows[-1]["turnover"] += liquidation_turnover
        rows[-1]["cost"] += liquidation_cost
        rows[-1]["net_return"] -= liquidation_cost

    return pd.DataFrame(rows)


def breakeven_bps_per_side(daily: pd.DataFrame) -> float | None:
    """Per-side bps making net mean zero under half-L1 turnover."""
    if daily.empty:
        return None
    gross = float(daily["gross_return"].mean())
    turnover = float(daily["turnover"].mean())
    if not math.isfinite(gross) or not math.isfinite(turnover) or turnover <= 0:
        return None
    return float(10_000.0 * gross / (2.0 * turnover))


def annualized_sharpe(
    daily_returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> float:
    r = daily_returns.dropna().astype(float)
    if len(r) < 2:
        return float("nan")
    vol = float(r.std(ddof=1))
    if vol <= 0 or not math.isfinite(vol):
        return float("nan")
    return float(math.sqrt(periods_per_year) * r.mean() / vol)


def block_bootstrap_mean(
    values: np.ndarray,
    *,
    block_length: int,
    replications: int,
    seed: int,
) -> dict[str, float]:
    """Circular block bootstrap of the sample mean."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {
            "mean": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    if n == 1 or block_length >= n:
        return {
            "mean": float(x.mean()),
            "se": float("nan"),
            "ci_low": float(x.mean()),
            "ci_high": float(x.mean()),
        }

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_length))
    means = np.empty(replications, dtype=float)
    for b in range(replications):
        starts = rng.integers(0, n, size=n_blocks)
        wrapped = []
        for s in starts:
            if s + block_length <= n:
                wrapped.append(x[s : s + block_length])
            else:
                wrapped.append(np.concatenate([x[s:], x[: (s + block_length) % n]]))
        sample = np.concatenate(wrapped)[:n]
        means[b] = float(sample.mean())
    point = float(x.mean())
    return {
        "mean": point,
        "se": float(means.std(ddof=1)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def summarize_daily_portfolio(
    daily: pd.DataFrame,
    *,
    config: TradeConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TradeConfig()
    if daily.empty:
        return {
            "n_sessions": 0,
            "position_mode": cfg.position_mode,
            "mean_net_exposure": float("nan"),
            "mean_gross": float("nan"),
            "mean_net": float("nan"),
            "mean_turnover": float("nan"),
            "annualized_turnover": float("nan"),
            "sharpe_gross": float("nan"),
            "sharpe_net": float("nan"),
            "breakeven_bps_per_side": None,
            "cost_bps_per_side_charged": cfg.cost_bps_per_side,
            "bootstrap_net_mean": float("nan"),
            "bootstrap_net_se": float("nan"),
            "bootstrap_net_ci_low": float("nan"),
            "bootstrap_net_ci_high": float("nan"),
        }

    boot = block_bootstrap_mean(
        daily["net_return"].to_numpy(dtype=float),
        block_length=cfg.bootstrap_block_length,
        replications=cfg.bootstrap_replications,
        seed=cfg.random_seed,
    )
    mean_turn = float(daily["turnover"].mean())
    return {
        "n_sessions": int(len(daily)),
        "position_mode": cfg.position_mode,
        "mean_net_exposure": (
            float(daily["net_exposure"].mean()) if "net_exposure" in daily.columns else float("nan")
        ),
        "mean_gross": float(daily["gross_return"].mean()),
        "mean_net": float(daily["net_return"].mean()),
        "mean_turnover": mean_turn,
        "annualized_turnover": float(mean_turn * cfg.annualization_periods),
        "sharpe_gross": annualized_sharpe(
            daily["gross_return"], periods_per_year=cfg.annualization_periods
        ),
        "sharpe_net": annualized_sharpe(
            daily["net_return"], periods_per_year=cfg.annualization_periods
        ),
        "breakeven_bps_per_side": breakeven_bps_per_side(daily),
        "cost_bps_per_side_charged": cfg.cost_bps_per_side,
        "bootstrap_net_mean": boot["mean"],
        "bootstrap_net_se": boot["se"],
        "bootstrap_net_ci_low": boot["ci_low"],
        "bootstrap_net_ci_high": boot["ci_high"],
        "bootstrap_block_length": cfg.bootstrap_block_length,
        "bootstrap_replications": cfg.bootstrap_replications,
        "bootstrap_seed": cfg.random_seed,
    }


def evaluate_signal_arm(
    firm_day: pd.DataFrame,
    signal_col: str,
    *,
    config: TradeConfig | None = None,
    orient: float | None = None,
    split: str = "development",
) -> dict[str, Any]:
    """IC + portfolio translation for one signal column on one split."""
    cfg = config or TradeConfig()
    orient_val = (
        float(orient)
        if orient is not None
        else float(DEFAULT_SIGNAL_ORIENT.get(signal_col, 1.0))
    )
    frame = firm_day
    if split is not None:
        frame = firm_day.loc[firm_day["split"] == split]
    ic = date_clustered_rank_ic(
        frame[signal_col], frame[cfg.outcome], frame["session_date"]
    )
    daily = build_daily_portfolio(
        firm_day, signal_col, config=cfg, orient=orient_val, split=split
    )
    summary = summarize_daily_portfolio(daily, config=cfg)
    return {
        "signal": signal_col,
        "split": split,
        "orient": orient_val,
        "threshold": cfg.threshold,
        "outcome": cfg.outcome,
        "ic": ic["ic"],
        "ic_se": ic["se"],
        "ic_t": ic["t"],
        "ic_p": ic["p"],
        "ic_n": ic["n"],
        "ic_n_clusters": ic["n_clusters"],
        **summary,
        "config": asdict(cfg),
    }


def evaluate_aggregator_arms(
    firm_day: pd.DataFrame,
    *,
    aggregators: Sequence[str] = AGGREGATOR_NAMES,
    config: TradeConfig | None = None,
    split: str = "development",
    orients: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One summary row per aggregator arm."""
    cfg = config or TradeConfig()
    orient_map = orients or DEFAULT_SIGNAL_ORIENT
    rows = [
        evaluate_signal_arm(
            firm_day,
            name,
            config=cfg,
            orient=orient_map.get(name, 1.0),
            split=split,
        )
        for name in aggregators
        if name in firm_day.columns
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # BH on portfolio? Keep BH on IC p-values for the primary family.
    table["bh_reject_ic_q05"] = benjamini_hochberg(table["ic_p"].fillna(1.0).tolist())
    table["multiplicity_family"] = MULTIPLICITY_FAMILY
    return table


# Re-exports so arms can import a single harness module.
__all__ = [
    "DEFAULT_SIGNAL_ORIENT",
    "POSITION_MODES",
    "PortfolioInsolvencyError",
    "TradeConfig",
    "cross_sectional_rank_scores",
    "annualized_sharpe",
    "assign_positions",
    "benjamini_hochberg",
    "block_bootstrap_mean",
    "breakeven_bps_per_side",
    "build_daily_portfolio",
    "date_clustered_rank_ic",
    "development_ic_table",
    "drifted_weight_map",
    "evaluate_aggregator_arms",
    "evaluate_signal_arm",
    "signal_to_position",
    "summarize_daily_portfolio",
]
