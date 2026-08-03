"""Sentiment-surprise demeaning stack and nested OOS horse race (Workstream 4).

Redesign relative to the 2026-07 sentiment-surprise pilot (NO-GO):

- Wide FNSPID firm-day panel (thousands of date clusters, not ~28).
- Fitted trailing market model for abnormal returns (not raw stock−market).
- Firm trailing baseline **and** cross-sectional daily mean removed.
- Sector-day layer left switchable but inactive until a sector table exists.
- Nested comparison primary statistic: out-of-sample R² (loss vs development
  mean benchmark), not a coefficient t-stat alone.

July pilot reference (do not bury): level OOS R² +0.003411 vs surprise
OOS R² −0.005515 on that small sample.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from final_experiments.lib.panel import (
    DEFAULT_PRICE_ZIP,
    FROZEN_DEV_END,
    FROZEN_EVAL_START,
    load_adjusted_open_prices,
)

# Trailing market-model defaults aligned with l2_event_study's 120 / ~21 gap.
MARKET_MODEL_WINDOW = 120
MARKET_MODEL_GAP = 21
MARKET_MODEL_MIN_OBS = 80
FIRM_BASELINE_NEWS_DAYS = 20
FIRM_BASELINE_MIN_PERIODS = 5

JULY_PILOT_NOTE = (
    "July 2026 surprise pilot NO-GO: level beat surprise on OOS R² "
    "(+0.003411 vs -0.005515); only ~28 evaluation date clusters; "
    "raw market difference; no cross-sectional demeaning."
)


@dataclass(frozen=True)
class SurpriseConfig:
    level_col: str = "mean_continuous"
    outcome_col: str = "ar_mm_h1"  # market-model AR; falls back handled by caller
    firm_baseline_news_days: int = FIRM_BASELINE_NEWS_DAYS
    firm_baseline_min_periods: int = FIRM_BASELINE_MIN_PERIODS
    market_window: int = MARKET_MODEL_WINDOW
    market_gap: int = MARKET_MODEL_GAP
    market_min_obs: int = MARKET_MODEL_MIN_OBS
    use_firm_baseline: bool = True
    use_cross_section: bool = True
    use_sector: bool = False  # blocked: no sector table
    development_end: str = FROZEN_DEV_END
    evaluation_start: str = FROZEN_EVAL_START


def trailing_firm_baseline(
    frame: pd.DataFrame,
    *,
    level_col: str,
    news_days: int = FIRM_BASELINE_NEWS_DAYS,
    min_periods: int = FIRM_BASELINE_MIN_PERIODS,
) -> pd.Series:
    """Trailing mean of ``level_col`` over prior news-days only (excludes today)."""
    ordered = frame.sort_values(["symbol", "session_date"], kind="mergesort")
    baseline = (
        ordered.groupby("symbol", sort=False)[level_col]
        .transform(
            lambda s: s.shift(1).rolling(news_days, min_periods=min_periods).mean()
        )
    )
    return baseline.reindex(frame.index)


def cross_section_day_mean(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Same-session cross-sectional mean (includes the firm; peel after firm layer)."""
    return values.groupby(dates).transform("mean")


def apply_demeaning_stack(
    frame: pd.DataFrame,
    *,
    config: SurpriseConfig | None = None,
) -> pd.DataFrame:
    """Add level components and nested surprise layers.

    Columns
    -------
    level
        Raw firm-day sentiment.
    firm_baseline
        Trailing news-day mean (NaN until min periods).
    surprise_firm
        level − firm_baseline (NaN where baseline missing).
    cs_mean
        Cross-sectional mean of surprise_firm (or level if firm layer off).
    surprise_cs
        After firm layer, minus cs_mean.
    surprise_full
        Alias of the deepest enabled layer.
    sector_mean / surprise_sector
        Always null when ``use_sector`` is False.
    """
    cfg = config or SurpriseConfig()
    if cfg.level_col not in frame.columns:
        raise ValueError(f"missing level column: {cfg.level_col}")

    out = frame.copy()
    out["level"] = out[cfg.level_col].astype(float)

    if cfg.use_firm_baseline:
        out["firm_baseline"] = trailing_firm_baseline(
            out,
            level_col="level",
            news_days=cfg.firm_baseline_news_days,
            min_periods=cfg.firm_baseline_min_periods,
        )
        out["surprise_firm"] = out["level"] - out["firm_baseline"]
        peel = out["surprise_firm"]
    else:
        out["firm_baseline"] = np.nan
        out["surprise_firm"] = out["level"]
        peel = out["level"]

    if cfg.use_cross_section:
        out["cs_mean"] = cross_section_day_mean(peel, out["session_date"])
        out["surprise_cs"] = peel - out["cs_mean"]
    else:
        out["cs_mean"] = np.nan
        out["surprise_cs"] = peel

    if cfg.use_sector:
        if "sector" not in out.columns:
            raise ValueError("use_sector=True requires a sector column")
        key = out["session_date"].astype(str) + "|" + out["sector"].astype(str)
        out["sector_mean"] = out["surprise_cs"].groupby(key).transform("mean")
        out["surprise_sector"] = out["surprise_cs"] - out["sector_mean"]
        out["surprise_full"] = out["surprise_sector"]
    else:
        out["sector_mean"] = pd.NA
        out["surprise_sector"] = pd.NA
        out["surprise_full"] = out["surprise_cs"]

    # Decomposition shares of |level| for plotting (descriptive).
    out["comp_baseline"] = out["firm_baseline"]
    out["comp_market_wide"] = out["cs_mean"]
    out["comp_idiosyncratic"] = out["surprise_full"]
    return out


def _session_returns_from_prices(
    prices: pd.DataFrame,
    *,
    market_symbol: str = "SPY",
) -> pd.DataFrame:
    """Exact next-exchange-session open returns, indexed by formation date.

    A row dated ``t`` is the forward return from open(t) to open(t+1).  A
    missing stock price on the exchange's next session stays missing; it is not
    replaced by the stock's next observed price.
    """
    frame = prices[["symbol", "session_date", "adjusted_open"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame = frame.sort_values(["symbol", "session_date"], kind="mergesort")

    calendar = (
        frame.loc[frame["symbol"] == market_symbol.upper(), ["session_date"]]
        .drop_duplicates()
        .sort_values("session_date")
    )
    calendar["return_end_date"] = calendar["session_date"].shift(-1)
    out = frame.merge(calendar, on="session_date", how="left", validate="m:1")
    next_open = frame.rename(
        columns={
            "session_date": "return_end_date",
            "adjusted_open": "next_adjusted_open",
        }
    )
    out = out.merge(
        next_open[["symbol", "return_end_date", "next_adjusted_open"]],
        on=["symbol", "return_end_date"],
        how="left",
        validate="m:1",
    )
    out["ret"] = out["next_adjusted_open"] / out["adjusted_open"] - 1.0
    return out.dropna(subset=["ret", "return_end_date"])


def trailing_market_model_ar(
    news_panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    market_symbol: str = "SPY",
    window: int = MARKET_MODEL_WINDOW,
    gap: int = MARKET_MODEL_GAP,
    min_obs: int = MARKET_MODEL_MIN_OBS,
) -> pd.DataFrame:
    """Attach trailing market-model AR for each news firm-day.

    For event session t, estimate α, β on prior formation sessions in
    ``[t - gap - window, t - gap)`` (dense calendar), then evaluate the
    post-news return ``AR[t,t+1] = ret[t,t+1] − α − β·mkt[t,t+1]``.
    """
    rets = _session_returns_from_prices(prices, market_symbol=market_symbol)
    market = (
        rets.loc[
            rets["symbol"] == market_symbol.upper(),
            ["session_date", "return_end_date", "ret"],
        ]
        .rename(columns={"ret": "mkt_ret"})
        .drop_duplicates("session_date")
    )
    # Normalise market symbol casing from zip stems.
    stock = rets.loc[
        rets["symbol"] != market_symbol.upper(),
        ["symbol", "session_date", "return_end_date", "ret"],
    ]
    dense = stock.merge(
        market,
        on=["session_date", "return_end_date"],
        how="inner",
    ).sort_values(
        ["symbol", "session_date"], kind="mergesort"
    )

    keys = news_panel[["symbol", "session_date"]].drop_duplicates().copy()
    keys["symbol"] = keys["symbol"].astype(str).str.upper()
    keys["session_date"] = pd.to_datetime(keys["session_date"]).dt.normalize()
    want_by_symbol = {sym: set(g["session_date"]) for sym, g in keys.groupby("symbol", sort=False)}

    rows: list[dict[str, Any]] = []
    for symbol, group in dense.groupby("symbol", sort=False):
        wanted = want_by_symbol.get(str(symbol))
        if not wanted:
            continue
        g = group.reset_index(drop=True)
        dates = pd.to_datetime(g["session_date"]).dt.normalize()
        y = g["ret"].to_numpy(dtype=float)
        x = g["mkt_ret"].to_numpy(dtype=float)
        n = len(g)
        # Rolling sums for window ending at index j (exclusive): [j-window, j)
        # We need estimation end = i - gap, so j = i - gap.
        c1 = np.cumsum(np.insert(np.ones(n), 0, 0.0))
        cy = np.cumsum(np.insert(y, 0, 0.0))
        cx = np.cumsum(np.insert(x, 0, 0.0))
        cxx = np.cumsum(np.insert(x * x, 0, 0.0))
        cxy = np.cumsum(np.insert(x * y, 0, 0.0))

        date_list = list(dates)
        for i, session in enumerate(date_list):
            if session not in wanted:
                continue
            end = i - gap
            start = end - window
            if end < min_obs or start < 0:
                continue
            n_est = c1[end] - c1[start]
            if n_est < min_obs:
                continue
            sum_y = cy[end] - cy[start]
            sum_x = cx[end] - cx[start]
            sum_xx = cxx[end] - cxx[start]
            sum_xy = cxy[end] - cxy[start]
            # Population/sample moments with ddof=1
            var_x = (sum_xx - (sum_x * sum_x) / n_est) / (n_est - 1)
            if not math.isfinite(var_x) or var_x <= 0:
                continue
            cov_xy = (sum_xy - (sum_x * sum_y) / n_est) / (n_est - 1)
            beta = float(cov_xy / var_x)
            alpha = float(sum_y / n_est - beta * sum_x / n_est)
            ar = float(y[i] - (alpha + beta * x[i]))
            rows.append(
                {
                    "symbol": str(symbol),
                    "session_date": pd.Timestamp(session).normalize(),
                    "return_end_date": pd.Timestamp(g.loc[i, "return_end_date"]).normalize(),
                    "ret_h1": float(y[i]),
                    "mkt_ret_h1": float(x[i]),
                    "market_alpha": alpha,
                    "market_beta": beta,
                    "ar_mm_h1": ar,
                    "mm_window": window,
                    "mm_gap": gap,
                    "mm_n_est": int(n_est),
                }
            )

    mm = pd.DataFrame(rows)
    out = news_panel.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    if mm.empty:
        out["ar_mm_h1"] = np.nan
        return out
    return out.merge(mm, on=["symbol", "session_date"], how="left")


def prior_session_return(
    news_panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    market_symbol: str = "SPY",
) -> pd.Series:
    """Return from the immediately prior exchange open to the event open.

    This is the initial-reaction control ending at event session ``t``.  It is
    deliberately distinct from the outcome, which starts at ``t`` and ends at
    the next session.
    """
    rets = _session_returns_from_prices(prices, market_symbol=market_symbol)
    stock = rets.loc[
        rets["symbol"] != market_symbol.upper(),
        ["symbol", "return_end_date", "ret"],
    ]
    stock = stock.rename(
        columns={"return_end_date": "session_date", "ret": "ret_lag1"}
    )
    keyed = news_panel[["symbol", "session_date"]].copy()
    keyed["symbol"] = keyed["symbol"].astype(str).str.upper()
    keyed["session_date"] = pd.to_datetime(keyed["session_date"]).dt.normalize()
    # validate guards row alignment: a duplicated (symbol, session_date) in the
    # price frame would silently fan the panel out and misalign every lag.
    merged = keyed.merge(stock, on=["symbol", "session_date"], how="left", validate="m:1")
    return pd.Series(merged["ret_lag1"].to_numpy(), index=news_panel.index, name="ret_lag1")


MODEL_SPECS: dict[str, tuple[str, ...]] = {
    # The null nest. Every other model adds sentiment on top of this, so no
    # sentiment claim is readable without it: an OOS R² of +0.003 means nothing
    # until you know what the control alone scores.
    "M_control_only": ("ret_lag1",),
    "M_level": ("ret_lag1", "level"),
    "M_surprise": ("ret_lag1", "surprise_full"),
    "M_both": ("ret_lag1", "level", "surprise_full"),
    "M_firm_only": ("ret_lag1", "surprise_firm"),
}
BASELINE_MODEL = "M_control_only"


def nested_oos_horse_race(
    frame: pd.DataFrame,
    *,
    outcome_col: str = "ar_mm_h1",
    development_end: str = FROZEN_DEV_END,
    evaluation_start: str = FROZEN_EVAL_START,
    model_specs: dict[str, tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on development and score OOS R² on evaluation.

    Primary statistic: evaluation OOS R² vs predicting the development mean of
    ``outcome_col``. Secondary coefficient estimates and date-clustered standard
    errors come from the development fit; the evaluation block is used only for
    out-of-sample loss.
    """
    specs = model_specs or MODEL_SPECS
    # ``split`` is optional: derived from the frozen dates when absent.
    data = frame.copy()
    data["session_date"] = pd.to_datetime(data["session_date"]).dt.normalize()
    if "split" not in data.columns:
        data["split"] = np.where(
            data["session_date"] <= pd.Timestamp(development_end),
            "development",
            np.where(
                data["session_date"] >= pd.Timestamp(evaluation_start),
                "evaluation",
                "gap",
            ),
        )

    predictor_pool = sorted({p for preds in specs.values() for p in preds})
    cols = [outcome_col, *predictor_pool]
    common = data.dropna(subset=cols).copy()
    common = common[np.isfinite(common[cols]).all(axis=1)].copy()
    development = common.loc[common["split"] == "development"]
    evaluation = common.loc[common["split"] == "evaluation"]

    meta = {
        "n_development": int(len(development)),
        "n_evaluation": int(len(evaluation)),
        "development_date_clusters": int(development["session_date"].nunique()),
        "evaluation_date_clusters": int(evaluation["session_date"].nunique()),
        "outcome": outcome_col,
        "july_pilot_note": JULY_PILOT_NOTE,
        "min_eval_clusters_gate": 280,  # 10× the failed pilot's 28
        "eval_clusters_pass": int(evaluation["session_date"].nunique()) >= 280,
    }
    if development.empty or evaluation.empty:
        raise ValueError("empty development or evaluation after dropna")
    if meta["evaluation_date_clusters"] < 2:
        raise ValueError("need at least two evaluation date clusters")

    y_bar_dev = float(development[outcome_col].mean())
    benchmark_sse = float(np.square(evaluation[outcome_col] - y_bar_dev).sum())
    if benchmark_sse <= 0:
        raise ValueError("benchmark SSE is zero")

    rows: list[dict[str, Any]] = []
    for model_name, predictors in specs.items():
        dev_x = sm.add_constant(development[list(predictors)], has_constant="add")
        ev_x = sm.add_constant(evaluation[list(predictors)], has_constant="add")
        dev_fit = sm.OLS(development[outcome_col], dev_x).fit(
            cov_type="cluster",
            cov_kwds={"groups": development["session_date"]},
        )
        pred = dev_fit.predict(ev_x)
        model_sse = float(np.square(evaluation[outcome_col] - pred).sum())
        oos_r2 = 1.0 - model_sse / benchmark_sse

        for term in dev_fit.params.index:
            rows.append(
                {
                    "model": model_name,
                    "term": str(term),
                    "coefficient": float(dev_fit.params[term]),
                    "std_error": float(dev_fit.bse[term]),
                    "t_stat": float(dev_fit.tvalues[term]),
                    "p_value": float(dev_fit.pvalues[term]),
                    "coefficient_sample": "development",
                    "oos_r2": oos_r2,
                    "oos_sse": model_sse,
                    "benchmark_sse": benchmark_sse,
                    "development_n": len(development),
                    "evaluation_n": len(evaluation),
                    "evaluation_date_clusters": meta["evaluation_date_clusters"],
                    "predictors": "|".join(predictors),
                }
            )

    result = pd.DataFrame(rows)
    # Model-level summary for the primary comparison.
    model_summary = (
        result.groupby("model", sort=False)
        .agg(
            oos_r2=("oos_r2", "first"),
            evaluation_date_clusters=("evaluation_date_clusters", "first"),
            predictors=("predictors", "first"),
        )
        .reset_index()
        .sort_values("oos_r2", ascending=False)
    )
    meta["model_oos_r2"] = model_summary.to_dict(orient="records")
    meta["best_model"] = str(model_summary.iloc[0]["model"])
    meta["best_oos_r2"] = float(model_summary.iloc[0]["oos_r2"])
    return result, meta


def oos_r2_comparison(
    frame: pd.DataFrame,
    *,
    outcome_col: str = "ar_mm_h1",
    development_end: str = FROZEN_DEV_END,
    evaluation_start: str = FROZEN_EVAL_START,
    model_specs: dict[str, tuple[str, ...]] | None = None,
    baseline_model: str = BASELINE_MODEL,
    block_length: int = 5,
    replications: int = 999,
    seed: int = 20260731,
) -> pd.DataFrame:
    """OOS R² per model plus a date-block bootstrap CI on Δ vs ``baseline_model``.

    A ranking of OOS R² values with no uncertainty attached is not a result:
    0.0037 vs 0.0027 could be a finding or a rounding of noise. Blocks of
    consecutive evaluation *sessions* are resampled, and every model is scored on
    the same resampled sessions, so the CI is on the loss differential rather
    than on each model separately.
    """
    specs = model_specs or MODEL_SPECS
    if baseline_model not in specs:
        raise ValueError(f"baseline_model {baseline_model!r} not in model_specs")

    data = frame.copy()
    data["session_date"] = pd.to_datetime(data["session_date"]).dt.normalize()
    if "split" not in data.columns:
        data["split"] = np.where(
            data["session_date"] <= pd.Timestamp(development_end),
            "development",
            np.where(data["session_date"] >= pd.Timestamp(evaluation_start), "evaluation", "gap"),
        )

    predictor_pool = sorted({p for preds in specs.values() for p in preds})
    cols = [outcome_col, *predictor_pool]
    common = data.dropna(subset=cols)
    common = common[np.isfinite(common[cols]).all(axis=1)]
    development = common.loc[common["split"] == "development"]
    evaluation = common.loc[common["split"] == "evaluation"]
    if development.empty or evaluation.empty:
        raise ValueError("empty development or evaluation after dropna")

    y_bar_dev = float(development[outcome_col].mean())
    y_eval = evaluation[outcome_col].to_numpy(dtype=float)
    losses = {"__benchmark__": np.square(y_eval - y_bar_dev)}
    for model_name, predictors in specs.items():
        dev_x = sm.add_constant(development[list(predictors)], has_constant="add")
        ev_x = sm.add_constant(evaluation[list(predictors)], has_constant="add")
        fit = sm.OLS(development[outcome_col], dev_x).fit()
        losses[model_name] = np.square(y_eval - fit.predict(ev_x).to_numpy(dtype=float))

    # Collapse to one SSE per session, then bootstrap whole sessions in blocks.
    session = evaluation["session_date"].to_numpy()
    daily = pd.DataFrame(losses).groupby(session).sum()
    sessions = daily.index.to_numpy()
    n_sessions = len(sessions)
    matrix = daily.to_numpy(dtype=float)
    names = list(daily.columns)
    bench_col = names.index("__benchmark__")

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n_sessions / block_length))
    draws: dict[str, list[float]] = {m: [] for m in specs}
    for _ in range(replications):
        starts = rng.integers(0, n_sessions, size=n_blocks)
        idx = np.concatenate(
            [np.arange(s, s + block_length) % n_sessions for s in starts]
        )[:n_sessions]
        totals = matrix[idx].sum(axis=0)
        bench = totals[bench_col]
        if bench <= 0:
            continue
        for model_name in specs:
            draws[model_name].append(1.0 - totals[names.index(model_name)] / bench)

    point_bench = float(matrix[:, bench_col].sum())
    point = {m: 1.0 - float(matrix[:, names.index(m)].sum()) / point_bench for m in specs}
    base_draws = np.asarray(draws[baseline_model], dtype=float)

    rows: list[dict[str, Any]] = []
    for model_name, predictors in specs.items():
        delta = np.asarray(draws[model_name], dtype=float) - base_draws
        point_delta = point[model_name] - point[baseline_model]
        rows.append(
            {
                "model": model_name,
                "predictors": "|".join(predictors),
                "oos_r2": point[model_name],
                "delta_vs_baseline": point_delta,
                "delta_ci_low": float(np.quantile(delta, 0.025)) if len(delta) else float("nan"),
                "delta_ci_high": float(np.quantile(delta, 0.975)) if len(delta) else float("nan"),
                # Two-sided bootstrap p: how often the resampled differential
                # lands on the opposite side of zero from the point estimate.
                # Floored at 1/replications — the bootstrap cannot resolve below
                # its own resolution, and reporting an exact 0 would overstate it.
                "delta_p_two_sided": (
                    max(
                        float(2.0 * min((delta <= 0).mean(), (delta >= 0).mean())),
                        1.0 / len(delta),
                    )
                    if len(delta) and model_name != baseline_model
                    else float("nan")
                ),
                "baseline_model": baseline_model,
                "evaluation_sessions": n_sessions,
                "evaluation_n": int(len(evaluation)),
                "block_length": block_length,
                "replications": int(len(delta)),
                "seed": seed,
            }
        )
    return pd.DataFrame(rows).sort_values("oos_r2", ascending=False).reset_index(drop=True)


def decomposition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute contribution of baseline / market-wide / idiosyncratic."""
    cols = ["comp_baseline", "comp_market_wide", "comp_idiosyncratic"]
    sub = frame.dropna(subset=cols).copy()
    if sub.empty:
        return pd.DataFrame(columns=["component", "mean", "mean_abs", "share_of_abs_sum"])
    abs_sum = sub[cols].abs().sum(axis=1).replace(0, np.nan)
    rows = []
    for col, name in zip(
        cols,
        ["firm_baseline", "market_wide_cs", "idiosyncratic"],
        strict=True,
    ):
        rows.append(
            {
                "component": name,
                "mean": float(sub[col].mean()),
                "mean_abs": float(sub[col].abs().mean()),
                "share_of_abs_sum": float((sub[col].abs() / abs_sum).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_surprise_panel(
    panel: pd.DataFrame,
    *,
    price_zip: Path = DEFAULT_PRICE_ZIP,
    config: SurpriseConfig | None = None,
    market_symbol: str = "SPY",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full W4 panel: demeaning stack + market-model AR + lag control."""
    cfg = config or SurpriseConfig()
    symbols = set(panel["symbol"].astype(str)) | {market_symbol}
    prices, missing = load_adjusted_open_prices(price_zip, symbols, market_symbol=market_symbol)
    base = apply_demeaning_stack(panel, config=cfg)
    base["ret_lag1"] = prior_session_return(base, prices, market_symbol=market_symbol)
    with_mm = trailing_market_model_ar(
        base,
        prices,
        market_symbol=market_symbol,
        window=cfg.market_window,
        gap=cfg.market_gap,
        min_obs=cfg.market_min_obs,
    )
    meta = {
        "config": asdict(cfg),
        "price_zip": str(price_zip),
        "missing_price_symbols": missing,
        "n_rows": int(len(with_mm)),
        "n_with_market_model_ar": int(with_mm["ar_mm_h1"].notna().sum())
        if "ar_mm_h1" in with_mm.columns
        else 0,
        "july_pilot_note": JULY_PILOT_NOTE,
        "sector_layer": "inactive_no_sector_table",
    }
    return with_mm, meta
