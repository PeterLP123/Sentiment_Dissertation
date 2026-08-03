"""Development-only learned trade/no-trade gates for the FNSPID spine."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from final_experiments.lib.evaluate import (
    annualized_sharpe,
    block_bootstrap_mean,
    cross_sectional_rank_scores,
)

SEED = 20260731
TRAIN_END = pd.Timestamp("2017-12-31")
VALIDATION_START = pd.Timestamp("2018-01-01")
VALIDATION_END = pd.Timestamp("2019-12-31")
FIXED_BANDS = (0.0, 0.2, 0.4, 0.6, 0.8)
PROBABILITY_CUTOFFS = (0.50, 0.525, 0.55, 0.575, 0.60)
MIN_MEAN_ACTIVE = 5.0
MIN_ACTIVE_SESSION_SHARE = 0.50
FEATURES = (
    "oriented_rank",
    "abs_oriented_rank",
    "log_article_count",
    "dispersion",
    "mean_continuous",
    "firm_baseline",
    "cs_mean",
    "ret_lag1",
)


@dataclass(frozen=True)
class GateRun:
    name: str
    cutoff: float
    model: object | None


def prepare_gate_frame(aggregators: pd.DataFrame, surprise: pd.DataFrame) -> pd.DataFrame:
    """Create split-safe features and a direction-correctness classification target."""
    extra = surprise[["symbol", "session_date", "firm_baseline", "cs_mean", "ret_lag1"]].copy()
    frame = aggregators.merge(extra, on=["symbol", "session_date"], how="left")
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["log_article_count"] = np.log1p(frame["article_count"].astype(float))
    frame = frame.sort_values(["session_date", "symbol"], kind="mergesort")
    frame["oriented_rank"] = frame.groupby("session_date", sort=False)["negative_share"].transform(
        lambda x: cross_sectional_rank_scores(-x.to_numpy(dtype=float))
    )
    frame["abs_oriented_rank"] = frame["oriented_rank"].abs()
    frame["direction"] = np.sign(frame["oriented_rank"])
    frame["aligned_payoff"] = frame["direction"] * frame["ar_open_h1"]
    frame["target_correct_direction"] = (frame["aligned_payoff"] > 0).astype(int)
    return frame.loc[frame["direction"].ne(0) & frame["ar_open_h1"].notna()].copy()


def make_models(seed: int = SEED) -> dict[str, Pipeline]:
    numeric = ColumnTransformer(
        [("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(FEATURES))],
        remainder="drop",
    )
    return {
        "logistic": Pipeline([("features", numeric), ("model", LogisticRegression(max_iter=300, random_state=seed))]),
        "gradient_boosted": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        max_leaf_nodes=15,
                        learning_rate=0.06,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("features", numeric),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16,),
                        alpha=0.01,
                        batch_size=2048,
                        early_stopping=True,
                        max_iter=60,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def gate_daily_portfolio(
    frame: pd.DataFrame,
    active: np.ndarray | pd.Series,
    *,
    cost_bps_per_side: float = 10.0,
) -> pd.DataFrame:
    """Dollar-neutral rank portfolio after a row-level trade/no-trade gate.

    A gate can retain an unbalanced subset of the original cross-section. Merely
    zeroing inactive ranks and normalising total gross would therefore turn the
    result into a directional long- or short-only book. Preserve the original
    rank direction and normalise the surviving long and short legs separately
    to 0.5 gross each. If either leg is empty, the session is not traded.
    """
    use = frame[["session_date", "symbol", "oriented_rank", "ar_open_h1"]].copy()
    use["active"] = np.asarray(active, dtype=bool)
    use = use.sort_values(["session_date", "symbol"], kind="mergesort")
    previous: dict[str, float] = {}
    rows = []
    cost_rate = cost_bps_per_side / 10_000.0
    for session, day in use.groupby("session_date", sort=True):
        raw = np.where(day["active"], day["oriented_rank"], 0.0).astype(float)
        long_gross = float(raw[raw > 0].sum())
        short_gross = float(-raw[raw < 0].sum())
        weights = np.zeros(len(day), dtype=float)
        if long_gross > 0 and short_gross > 0:
            weights[raw > 0] = 0.5 * raw[raw > 0] / long_gross
            weights[raw < 0] = 0.5 * raw[raw < 0] / short_gross
        if np.count_nonzero(weights) < 2:
            weights = np.zeros(len(day), dtype=float)
        current = {symbol: float(weight) for symbol, weight in zip(day["symbol"].astype(str), weights, strict=True) if weight != 0.0}
        names = set(previous) | set(current)
        turnover = 0.5 * sum(abs(current.get(s, 0.0) - previous.get(s, 0.0)) for s in names)
        gross = float(np.dot(weights, day["ar_open_h1"].to_numpy(dtype=float)))
        cost = 2.0 * turnover * cost_rate
        rows.append(
            {
                "session_date": session,
                "n_active": int(np.count_nonzero(weights)),
                "net_exposure": float(weights.sum()),
                "gross_return": gross,
                "turnover": turnover,
                "cost": cost,
                "net_return": gross - cost,
            }
        )
        previous = current
    # The last held book must be closed at the final interval's ending open.
    # Otherwise the backtest receives its last return without paying the exit
    # cost, which makes sparse gates look slightly better than they are.
    if rows and previous:
        liquidation_turnover = 0.5 * sum(abs(weight) for weight in previous.values())
        liquidation_cost = 2.0 * liquidation_turnover * cost_rate
        rows[-1]["turnover"] += liquidation_turnover
        rows[-1]["cost"] += liquidation_cost
        rows[-1]["net_return"] -= liquidation_cost
    return pd.DataFrame(rows)


def portfolio_summary(daily: pd.DataFrame) -> dict[str, float | int]:
    mean_turnover = float(daily["turnover"].mean())
    mean_gross = float(daily["gross_return"].mean())
    breakeven = 10_000.0 * mean_gross / (2.0 * mean_turnover) if mean_turnover > 0 else float("nan")
    bootstrap = block_bootstrap_mean(
        daily["net_return"].to_numpy(dtype=float),
        block_length=20,
        replications=999,
        seed=SEED,
    )
    return {
        "n_sessions": int(len(daily)),
        "mean_active": float(daily["n_active"].mean()),
        "active_session_share": float(daily["n_active"].gt(0).mean()),
        "mean_net_exposure": float(daily["net_exposure"].mean()),
        "mean_gross": mean_gross,
        "mean_net": float(daily["net_return"].mean()),
        "mean_turnover": mean_turnover,
        "gross_sharpe": annualized_sharpe(daily["gross_return"]),
        "net_sharpe": annualized_sharpe(daily["net_return"]),
        "breakeven_bps_per_side": float(breakeven),
        "net_mean_ci_low": float(bootstrap["ci_low"]),
        "net_mean_ci_high": float(bootstrap["ci_high"]),
        "bootstrap_block_length": 20,
        "bootstrap_replications": 999,
        "bootstrap_seed": SEED,
    }


def _activity_eligibility(table: pd.DataFrame) -> pd.Series:
    return (
        table["mean_active"].ge(MIN_MEAN_ACTIVE) & table["active_session_share"].ge(MIN_ACTIVE_SESSION_SHARE) & table["net_sharpe"].notna()
    )


def choose_fixed_band(
    validation: pd.DataFrame,
    *,
    require_activity: bool = False,
) -> tuple[float, pd.DataFrame]:
    """Choose a fixed band, optionally under the learned-gate activity floor.

    ``require_activity=False`` preserves the historical exploratory selection.
    New comparisons should set it to true so a nearly inactive fixed band
    cannot appear superior merely by sitting in cash while learned gates are
    required to trade on at least half of validation sessions.
    """
    rows = []
    for band in FIXED_BANDS:
        daily = gate_daily_portfolio(validation, validation["abs_oriented_rank"] > band)
        rows.append({"band": band, **portfolio_summary(daily)})
    table = pd.DataFrame(rows)
    table["selection_eligible"] = _activity_eligibility(table)
    candidates = table.loc[table["selection_eligible"]] if require_activity else table
    candidates = candidates.loc[candidates["net_sharpe"].notna()]
    if candidates.empty:
        raise ValueError("no fixed band meets the selection constraints")
    best = candidates.sort_values(["net_sharpe", "band"], ascending=[False, True]).iloc[0]
    return float(best["band"]), table


def choose_probability_cutoff(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for cutoff in PROBABILITY_CUTOFFS:
        daily = gate_daily_portfolio(validation, probabilities >= cutoff)
        rows.append({"cutoff": cutoff, **portfolio_summary(daily)})
    table = pd.DataFrame(rows)
    table["selection_eligible"] = _activity_eligibility(table)
    eligible = table.loc[table["selection_eligible"]]
    if eligible.empty:
        raise ValueError("no probability cutoff meets the minimum activity constraint")
    best = eligible.sort_values(["net_sharpe", "cutoff"], ascending=[False, False]).iloc[0]
    return float(best["cutoff"]), table


def split_gate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["session_date"] <= TRAIN_END].copy()
    validation = frame.loc[frame["session_date"].between(VALIDATION_START, VALIDATION_END)].copy()
    evaluation = frame.loc[frame["session_date"] >= pd.Timestamp("2020-01-01")].copy()
    return train, validation, evaluation


def finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


__all__ = [
    "FEATURES",
    "FIXED_BANDS",
    "MIN_ACTIVE_SESSION_SHARE",
    "MIN_MEAN_ACTIVE",
    "PROBABILITY_CUTOFFS",
    "SEED",
    "choose_fixed_band",
    "choose_probability_cutoff",
    "gate_daily_portfolio",
    "make_models",
    "portfolio_summary",
    "prepare_gate_frame",
    "split_gate_frame",
]
