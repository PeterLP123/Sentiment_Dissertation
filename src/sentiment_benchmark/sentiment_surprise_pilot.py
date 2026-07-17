"""Leak-free pilot comparing sentiment level with point-in-time surprise."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .artifact_io import atomic_write_json, sha256_file
from .runtime_metadata import collect_run_environment
from .strategy_sweep import chronological_split_date


class SentimentSurprisePilotError(RuntimeError):
    """Raised when the frozen pilot cannot be executed without a workaround."""


@dataclass(frozen=True)
class PilotConfig:
    scorer: str = "finbert"
    market_symbol: str = "^GSPC"
    exchange_timezone: str = "America/New_York"
    development_fraction: float = 0.70
    minimum_symbol_news_days: int = 60
    minimum_prior_news_days: int = 30
    dev20_window: int = 20
    bootstrap_samples: int = 10_000
    random_seed: int = 20260717
    evaluation_run_number: int = 1
    rerun_reason: str | None = None


MODEL_SPECS: dict[str, tuple[str, ...]] = {
    "M1": ("ar0",),
    "M2": ("ar0", "level"),
    "M3": ("ar0", "surprise"),
    "M4": ("ar0", "level", "surprise"),
    "M5": ("ar0", "delta"),
    "M6": ("ar0", "dev20"),
}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise SentimentSurprisePilotError(f"{label} lacks required columns: {sorted(missing)}")


def _load_prices(path: str | Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, {"symbol", "session_date", "close"}, label)
    frame = frame[["symbol", "session_date", "close"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame["session_date"] = pd.to_datetime(frame["session_date"], format="%Y-%m-%d", errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame["close"].isna().any() or (frame["close"] <= 0).any():
        raise SentimentSurprisePilotError(f"{label} contains missing or non-positive closes")
    if frame.duplicated(["symbol", "session_date"]).any():
        raise SentimentSurprisePilotError(f"{label} contains duplicate symbol-session rows")
    return frame.sort_values(["symbol", "session_date"], kind="stable").reset_index(drop=True)


def _session_closes(prices: pd.DataFrame, timezone: str) -> pd.DataFrame:
    local_dates = pd.to_datetime(prices["session_date"].dt.strftime("%Y-%m-%d") + " 16:00:00")
    closes = local_dates.dt.tz_localize(ZoneInfo(timezone)).dt.tz_convert("UTC")
    return prices[["symbol", "session_date"]].assign(session_close_utc=closes)


def build_daily_sentiment(
    scores_path: str | Path,
    stock_prices: pd.DataFrame,
    config: PilotConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Explode target associations and aggregate scores at the first close after availability."""

    usecols = ["headline_sha256", "matched_symbols", "first_timestamp", "baseline", "score", "status"]
    scores = pd.read_csv(scores_path, usecols=usecols)
    input_rows = len(scores)
    scores = scores[(scores["baseline"] == config.scorer) & (scores["status"] == "success")].copy()
    scorer_rows = len(scores)
    scores["timestamp"] = pd.to_datetime(scores["first_timestamp"], utc=True, errors="coerce")
    scores["level"] = pd.to_numeric(scores["score"], errors="coerce")
    scores["symbol"] = scores["matched_symbols"].fillna("").astype(str).str.split("|")
    scores = scores.explode("symbol", ignore_index=True)
    scores["symbol"] = scores["symbol"].astype(str).str.strip()
    scores = scores[scores["timestamp"].notna() & scores["level"].notna() & np.isfinite(scores["level"]) & (scores["symbol"] != "")].copy()
    valid_associations = len(scores)

    sessions = _session_closes(stock_prices, config.exchange_timezone)
    known_symbols = set(sessions["symbol"])
    unknown = sorted(set(scores["symbol"]) - known_symbols)
    if unknown:
        raise SentimentSurprisePilotError(f"scored associations lack stock prices for symbols: {unknown}")

    aligned_parts: list[pd.DataFrame] = []
    for symbol, group in scores.groupby("symbol", sort=True):
        right = sessions[sessions["symbol"] == symbol][["session_date", "session_close_utc"]]
        aligned = pd.merge_asof(
            group.sort_values("timestamp", kind="stable"),
            right.sort_values("session_close_utc", kind="stable"),
            left_on="timestamp",
            right_on="session_close_utc",
            direction="forward",
            allow_exact_matches=False,
        )
        aligned["symbol"] = symbol
        aligned_parts.append(aligned)
    aligned_scores = pd.concat(aligned_parts, ignore_index=True)
    aligned_scores = aligned_scores[aligned_scores["session_date"].notna()].copy()

    daily = (
        aligned_scores.groupby(["symbol", "session_date"], sort=True)
        .agg(level=("level", "mean"), article_count=("headline_sha256", "nunique"))
        .reset_index()
    )
    daily["total_symbol_news_days"] = daily.groupby("symbol")["session_date"].transform("size")
    symbols_before = daily["symbol"].nunique()
    daily = daily[daily["total_symbol_news_days"] >= config.minimum_symbol_news_days].copy()
    symbols_after = daily["symbol"].nunique()
    daily = daily.sort_values(["symbol", "session_date"], kind="stable").reset_index(drop=True)
    daily["prior_news_days"] = daily.groupby("symbol").cumcount()
    daily["delta"] = daily.groupby("symbol", sort=False)["level"].diff()
    prior_mean = daily.groupby("symbol", sort=False)["level"].transform(
        lambda values: values.shift(1).rolling(config.dev20_window, min_periods=config.dev20_window).mean()
    )
    daily["dev20"] = daily["level"] - prior_mean
    daily["eligible_baseline_history"] = daily["prior_news_days"] >= config.minimum_prior_news_days

    attrition = [
        {"step": "score_file_rows_all_scorers", "unit": "score rows", "surviving": input_rows},
        {"step": f"successful_{config.scorer}_headline_rows", "unit": "headline rows", "surviving": scorer_rows},
        {"step": "valid_symbol_headline_associations", "unit": "associations", "surviving": valid_associations},
        {"step": "associations_mapped_to_stock_session", "unit": "associations", "surviving": len(aligned_scores)},
        {"step": "aggregated_symbol_session_news_days", "unit": "events", "surviving": len(daily)},
        {"step": "symbols_before_60_news_day_filter", "unit": "symbols", "surviving": symbols_before},
        {"step": "symbols_after_60_news_day_filter", "unit": "symbols", "surviving": symbols_after},
        {
            "step": "events_with_30_prior_news_days",
            "unit": "events",
            "surviving": int(daily["eligible_baseline_history"].sum()),
        },
    ]
    return daily.reset_index(drop=True), attrition


def add_market_adjusted_outcomes(
    events: pd.DataFrame,
    stock_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    market_symbol: str,
) -> pd.DataFrame:
    """Add AR(0) and strictly post-event additive log-return CARs."""

    market = market_prices[market_prices["symbol"] == market_symbol].copy()
    if market.empty:
        available = sorted(market_prices["symbol"].unique())
        raise SentimentSurprisePilotError(f"market price input lacks {market_symbol!r}; available symbols: {available}")
    market = market.sort_values("session_date", kind="stable")
    market["market_return"] = np.log(market["close"]).diff()
    stocks = stock_prices.copy()
    stocks["stock_return"] = stocks.groupby("symbol", sort=False)["close"].transform(lambda values: np.log(values).diff())
    abnormal = stocks.merge(market[["session_date", "market_return"]], on="session_date", how="inner", validate="many_to_one")
    abnormal["abnormal_return"] = abnormal["stock_return"] - abnormal["market_return"]
    abnormal = abnormal.dropna(subset=["abnormal_return"])

    lookup: dict[tuple[str, pd.Timestamp], tuple[float, float, float, float]] = {}
    for symbol, group in abnormal.groupby("symbol", sort=True):
        ordered = group.sort_values("session_date", kind="stable").reset_index(drop=True)
        values = ordered["abnormal_return"].to_numpy(float)
        dates = ordered["session_date"].tolist()
        for index, event_date in enumerate(dates):
            cars: list[float] = []
            for horizon in (1, 5, 10):
                window = values[index + 1 : index + horizon + 1]
                cars.append(float(window.sum()) if len(window) == horizon else float("nan"))
            lookup[(symbol, event_date)] = (
                float(values[index]),
                cars[0],
                cars[1],
                cars[2],
            )

    result = events.copy()
    outcomes = [lookup.get((str(row.symbol), pd.Timestamp(row.session_date)), (math.nan,) * 4) for row in result.itertuples(index=False)]
    result[["ar0", "car_p1_p1", "car_p1_p5", "car_p1_p10"]] = pd.DataFrame(outcomes, index=result.index)
    return result


def _local_level_one_step(
    values: np.ndarray,
    development_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if development_count < 10:
        raise SentimentSurprisePilotError("local-level model needs at least 10 development observations")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning,
        )
        development_model = UnobservedComponents(values[:development_count], level="llevel")
        fitted = development_model.fit(method="powell", maxiter=1_000, disp=False)
        if not bool(fitted.mle_retvals.get("converged", False)):
            raise SentimentSurprisePilotError("Powell local-level maximum-likelihood fit did not converge")
        full_model = UnobservedComponents(values, level="llevel")
        filtered = full_model.filter(fitted.params)
    prediction = np.asarray(filtered.forecasts[0], dtype=float)
    prediction_std = np.sqrt(np.asarray(filtered.forecasts_error_cov[0, 0], dtype=float))
    parameters = {name: float(value) for name, value in zip(full_model.param_names, fitted.params, strict=True)}
    return prediction, prediction_std, parameters


def add_surprise_signals(
    events: pd.DataFrame,
    split_date: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Fit parameters on development only, then obtain one-sided filter forecasts."""

    parts: list[pd.DataFrame] = []
    parameters: dict[str, dict[str, float]] = {}
    boundary = pd.Timestamp(split_date)
    for symbol, group in events.groupby("symbol", sort=True):
        ordered = group.sort_values("session_date", kind="stable").copy()
        values = ordered["level"].to_numpy(float)
        development_count = int((ordered["session_date"] < boundary).sum())
        prediction, prediction_std, fitted_parameters = _local_level_one_step(values, development_count)
        if np.any(~np.isfinite(prediction_std)) or np.any(prediction_std <= 0):
            raise SentimentSurprisePilotError(f"invalid local-level prediction variance for {symbol}")
        ordered["surprise_prediction"] = prediction
        ordered["surprise_prediction_std"] = prediction_std
        ordered["surprise"] = (ordered["level"].to_numpy(float) - prediction) / prediction_std
        parameters[str(symbol)] = fitted_parameters
        parts.append(ordered)
    return pd.concat(parts, ignore_index=True), parameters


def _clustered_evaluation_fit(data: pd.DataFrame, predictors: tuple[str, ...]) -> Any:
    design = sm.add_constant(data[list(predictors)], has_constant="add")
    fitted = sm.OLS(data["car_p1_p5"], design).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["session_date"], "use_correction": True, "df_correction": True},
        use_t=True,
    )
    return fitted


def run_horse_race(
    events: pd.DataFrame,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit evaluation inference and development-trained out-of-sample forecasts."""

    required = ["car_p1_p5", "ar0", "level", "surprise", "delta", "dev20"]
    common = events[events["eligible_baseline_history"]].dropna(subset=required).copy()
    common = common[np.isfinite(common[required]).all(axis=1)].copy()
    boundary = pd.Timestamp(split_date)
    development = common[common["session_date"] < boundary].copy()
    evaluation = common[common["session_date"] >= boundary].copy()
    if development.empty or evaluation.empty:
        raise SentimentSurprisePilotError("chronological split produced an empty development or evaluation sample")
    if evaluation["session_date"].nunique() < 2:
        raise SentimentSurprisePilotError("clustered inference needs at least two evaluation dates")

    rows: list[dict[str, Any]] = []
    development_mean = float(development["car_p1_p5"].mean())
    benchmark_sse = float(np.square(evaluation["car_p1_p5"] - development_mean).sum())
    if benchmark_sse <= 0:
        raise SentimentSurprisePilotError("out-of-sample R-squared benchmark has zero error")
    for model_name, predictors in MODEL_SPECS.items():
        evaluation_fit = _clustered_evaluation_fit(evaluation, predictors)
        development_design = sm.add_constant(development[list(predictors)], has_constant="add")
        development_fit = sm.OLS(development["car_p1_p5"], development_design).fit()
        evaluation_design = sm.add_constant(evaluation[list(predictors)], has_constant="add")
        prediction = development_fit.predict(evaluation_design)
        model_sse = float(np.square(evaluation["car_p1_p5"] - prediction).sum())
        oos_r2 = 1.0 - model_sse / benchmark_sse
        for term in evaluation_fit.params.index:
            rows.append(
                {
                    "model": model_name,
                    "term": str(term),
                    "coefficient": float(evaluation_fit.params[term]),
                    "standard_error": float(evaluation_fit.bse[term]),
                    "t_stat": float(evaluation_fit.tvalues[term]),
                    "p_value": float(evaluation_fit.pvalues[term]),
                    "oos_r2": oos_r2,
                    "development_events": len(development),
                    "evaluation_events": len(evaluation),
                    "evaluation_date_clusters": evaluation["session_date"].nunique(),
                }
            )
    return pd.DataFrame(rows), development, evaluation


def quintile_spread(
    evaluation: pd.DataFrame,
    signal: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    ranked = evaluation.copy()
    ranked["quintile"] = pd.qcut(ranked[signal].rank(method="first"), 5, labels=False) + 1
    tails = ranked[ranked["quintile"].isin([1, 5])].copy()
    bottom = float(tails.loc[tails["quintile"] == 1, "car_p1_p5"].mean())
    top = float(tails.loc[tails["quintile"] == 5, "car_p1_p5"].mean())

    dates = sorted(tails["session_date"].unique())
    date_index = {date: index for index, date in enumerate(dates)}
    aggregates = np.zeros((len(dates), 4), dtype=float)
    for date, group in tails.groupby("session_date", sort=True):
        index = date_index[date]
        low = group[group["quintile"] == 1]["car_p1_p5"]
        high = group[group["quintile"] == 5]["car_p1_p5"]
        aggregates[index] = (low.sum(), len(low), high.sum(), len(high))
    rng = np.random.default_rng(seed)
    bootstraps = np.empty(samples, dtype=float)
    for draw in range(samples):
        selected = rng.integers(0, len(dates), size=len(dates))
        sample = aggregates[selected].sum(axis=0)
        bootstraps[draw] = sample[2] / sample[3] - sample[0] / sample[1]
    low, high = np.quantile(bootstraps, [0.025, 0.975])
    return {
        "signal": signal,
        "evaluation_events": len(ranked),
        "evaluation_date_clusters": len(dates),
        "bottom_quintile_mean_car": bottom,
        "top_quintile_mean_car": top,
        "top_minus_bottom": top - bottom,
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "bootstrap_samples": samples,
        "random_seed": seed,
    }


def classify_decision(regressions: pd.DataFrame) -> tuple[str, list[str]]:
    model_r2 = regressions.groupby("model", sort=False)["oos_r2"].first().to_dict()
    m4_surprise = regressions[(regressions["model"] == "M4") & (regressions["term"] == "surprise")].iloc[0]
    go_cells = {
        "m4_surprise_abs_t_gt_2": abs(float(m4_surprise["t_stat"])) > 2,
        "m3_oos_r2_gt_m2": model_r2["M3"] > model_r2["M2"],
        "m3_oos_r2_gt_m5": model_r2["M3"] > model_r2["M5"],
        "m3_oos_r2_gt_m6": model_r2["M3"] > model_r2["M6"],
    }
    if all(go_cells.values()):
        return "GO", [name for name, passed in go_cells.items() if passed]
    if max(model_r2[name] for name in ("M2", "M3", "M4", "M5", "M6")) <= model_r2["M1"]:
        return "NO-GO", ["nothing_beats_m1_oos"]
    change_best = max(model_r2[name] for name in ("M3", "M5", "M6"))
    if change_best > model_r2["M1"] and change_best > model_r2["M2"]:
        return "WEAK-GO", ["change_or_deviation_signal_beats_m1_and_level_but_go_cells_fail"]
    return "NO-GO", ["go_and_weak_go_orderings_not_met"]


def _format_cell(regressions: pd.DataFrame, model: str, term: str) -> str:
    match = regressions[(regressions["model"] == model) & (regressions["term"] == term)]
    if match.empty:
        return "—"
    row = match.iloc[0]
    return f"{row['coefficient']:.6f} ({row['t_stat']:.2f})"


def render_report(
    regressions: pd.DataFrame,
    spreads: pd.DataFrame,
    attrition: pd.DataFrame,
    decision: str,
    decision_reasons: list[str],
    split_date: str,
    methodology: str,
    config: PilotConfig,
) -> str:
    lines = [
        "# Sentiment surprise vs level: leak-free pilot",
        "",
        f"**Decision: {decision}.** Evaluation begins `{split_date}`. {methodology}",
        "",
        "## M1–M6 evaluation horse race",
        "",
        (
            "Coefficients are evaluation-set OLS estimates with calendar-date clustered standard errors; "
            "parentheses contain t-statistics. OOS R² uses development-fitted coefficients and the development "
            "outcome mean as the evaluation benchmark."
        ),
        "",
        "| Model | AR(0) | Level | Surprise | Delta | Dev20 | OOS R² |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODEL_SPECS:
        r2 = float(regressions.loc[regressions["model"] == model, "oos_r2"].iloc[0])
        lines.append(
            f"| {model} | {_format_cell(regressions, model, 'ar0')} | {_format_cell(regressions, model, 'level')} | "
            f"{_format_cell(regressions, model, 'surprise')} | {_format_cell(regressions, model, 'delta')} | "
            f"{_format_cell(regressions, model, 'dev20')} | {r2:.6f} |"
        )
    run_note = (
        "- This is the first and only evaluation run for this output directory."
        if config.evaluation_run_number == 1
        else f"- This is disclosed evaluation run {config.evaluation_run_number}: {config.rerun_reason}."
    )
    lines.extend(
        [
            "",
            "## Quintile spreads",
            "",
            "| Signal | Bottom mean | Top mean | Top − bottom | Date-bootstrap 95% CI |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in spreads.itertuples(index=False):
        lines.append(
            f"| {row.signal} | {row.bottom_quintile_mean_car:.6f} | {row.top_quintile_mean_car:.6f} | "
            f"{row.top_minus_bottom:.6f} | [{row.bootstrap_ci_low:.6f}, {row.bootstrap_ci_high:.6f}] |"
        )
    lines.extend(["", "## Attrition", "", "| Step | Unit | Surviving |", "| --- | --- | ---: |"])
    for row in attrition.itertuples(index=False):
        lines.append(f"| {row.step} | {row.unit} | {int(row.surviving):,} |")

    model_r2 = regressions.groupby("model", sort=False)["oos_r2"].first().to_dict()
    m4_t = float(regressions[(regressions["model"] == "M4") & (regressions["term"] == "surprise")]["t_stat"].iloc[0])
    lines.extend(
        [
            "",
            "## Ten-line plain-English summary",
            "",
            f"1. The one-shot pilot decision is **{decision}**.",
            f"2. M4's surprise t-statistic is {m4_t:.2f}; the GO threshold is an absolute value above 2.",
            f"3. M3 surprise OOS R² is {model_r2['M3']:.6f}, versus M2 level at {model_r2['M2']:.6f}.",
            f"4. The naive delta and dev20 OOS R² values are {model_r2['M5']:.6f} and {model_r2['M6']:.6f}.",
            f"5. M1 AR(0)-only OOS R² is {model_r2['M1']:.6f}.",
            "6. Kalman parameters were estimated on development dates only; evaluation observations entered only the one-sided filter.",
            "7. No smoothed or two-sided state estimate was used.",
            "8. CAR(+1,+5) excludes the event-day reaction, which appears only as AR(0).",
            "9. Inference and the portfolio bootstrap both preserve calendar-date clustering.",
            f"10. Mechanical decision reason: {', '.join(decision_reasons)}.",
            "",
            "## Integrity notes",
            "",
            run_note,
            (
                "- FinBERT score is `p_positive - p_negative`, aggregated by target symbol and first tradable "
                "session close after the timestamp."
            ),
            "- `delta` uses the prior firm news-day; `dev20` uses a shifted 20-news-day mean exclusive of today.",
            "- Full one-step forecasts and event-level data contain no licensed headline text.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_pilot(
    scores_path: str | Path,
    stock_prices_path: str | Path,
    market_prices_path: str | Path,
    output_dir: str | Path,
    config: PilotConfig | None = None,
) -> Path:
    config = config or PilotConfig()
    if config.evaluation_run_number < 1:
        raise SentimentSurprisePilotError("evaluation_run_number must be positive")
    if config.evaluation_run_number > 1 and not (config.rerun_reason or "").strip():
        raise SentimentSurprisePilotError("a disclosed rerun_reason is required after evaluation run 1")
    destination = Path(output_dir)
    if destination.exists():
        raise SentimentSurprisePilotError(f"refusing to overwrite pilot output: {destination}")

    stocks = _load_prices(stock_prices_path, label="stock prices")
    market = _load_prices(market_prices_path, label="market prices")
    daily, attrition_rows = build_daily_sentiment(scores_path, stocks, config)
    events = add_market_adjusted_outcomes(daily, stocks, market, config.market_symbol)
    eligible = events[events["eligible_baseline_history"]]
    primary_candidates = eligible.dropna(subset=["ar0", "car_p1_p5", "delta", "dev20"])
    if primary_candidates["session_date"].nunique() < 4:
        raise SentimentSurprisePilotError("fewer than four primary-outcome dates survive")
    split_date = chronological_split_date(primary_candidates["session_date"].dt.date.astype(str).tolist(), config.development_fraction)
    events, local_level_parameters = add_surprise_signals(events, split_date)
    regressions, development, evaluation = run_horse_race(events, split_date)
    spreads = pd.DataFrame(
        [
            quintile_spread(
                evaluation,
                signal,
                samples=config.bootstrap_samples,
                seed=config.random_seed + offset,
            )
            for offset, signal in enumerate(("surprise", "level"))
        ]
    )
    attrition_rows.extend(
        [
            {"step": "events_with_ar0", "unit": "events", "surviving": int(eligible["ar0"].notna().sum())},
            {
                "step": "events_with_primary_car_p1_p5",
                "unit": "events",
                "surviving": int(eligible["car_p1_p5"].notna().sum()),
            },
            {
                "step": "events_with_sensitivity_car_p1_p1",
                "unit": "events",
                "surviving": int(eligible["car_p1_p1"].notna().sum()),
            },
            {
                "step": "events_with_sensitivity_car_p1_p10",
                "unit": "events",
                "surviving": int(eligible["car_p1_p10"].notna().sum()),
            },
            {"step": "common_primary_development_sample", "unit": "events", "surviving": len(development)},
            {"step": "common_primary_evaluation_sample", "unit": "events", "surviving": len(evaluation)},
            {
                "step": "evaluation_calendar_date_clusters",
                "unit": "dates",
                "surviving": evaluation["session_date"].nunique(),
            },
        ]
    )
    attrition = pd.DataFrame(attrition_rows)
    decision, decision_reasons = classify_decision(regressions)
    methodology = (
        f"Outcomes are market-adjusted close-to-close log returns: stock minus {config.market_symbol}; "
        "the pilot did not use a fitted market model."
    )

    destination.mkdir(parents=True)
    regressions_path = destination / "regression_results.csv"
    spreads_path = destination / "quintile_spreads.csv"
    attrition_path = destination / "attrition.csv"
    events_path = destination / "event_panel.csv"
    report_path = destination / "report.md"
    regressions.to_csv(regressions_path, index=False, lineterminator="\n")
    spreads.to_csv(spreads_path, index=False, lineterminator="\n")
    attrition.to_csv(attrition_path, index=False, lineterminator="\n")
    event_output = events.copy()
    event_output["session_date"] = event_output["session_date"].dt.date.astype(str)
    event_output.to_csv(events_path, index=False, lineterminator="\n")
    report_path.write_text(
        render_report(regressions, spreads, attrition, decision, decision_reasons, split_date, methodology, config),
        encoding="utf-8",
        newline="\n",
    )
    inputs = {
        "scores": str(scores_path),
        "stock_prices": str(stock_prices_path),
        "market_prices": str(market_prices_path),
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "evaluation_run_number": config.evaluation_run_number,
        "rerun_reason": config.rerun_reason,
        "decision": decision,
        "decision_reasons": decision_reasons,
        "split": {
            "method": "chronological distinct primary-outcome dates",
            "development_fraction": config.development_fraction,
            "evaluation_start": split_date,
        },
        "signals": {
            "level": "mean FinBERT p_positive - p_negative",
            "surprise": "local-level one-step forecast error divided by forecast standard deviation",
            "state_estimation": ("parameters fit on development only with Powell maximum likelihood; one-sided Kalman filter; no smoother"),
            "delta": "current minus prior firm news-day",
            "dev20": "current minus shifted 20-firm-news-day mean",
            "local_level_parameters": local_level_parameters,
        },
        "outcomes": {
            "method": "market-adjusted close-to-close log returns",
            "market_symbol": config.market_symbol,
            "primary": "CAR(+1,+5)",
            "sensitivities": ["CAR(+1,+1)", "CAR(+1,+10)"],
            "control": "AR(0)",
        },
        "inference": {
            "evaluation_coefficients": "OLS with calendar-date clustered standard errors",
            "oos_r2": "development-fitted prediction relative to development-mean forecast on evaluation",
            "portfolio_ci": "calendar-date cluster bootstrap percentile interval",
            "bootstrap_samples": config.bootstrap_samples,
            "random_seed": config.random_seed,
        },
        "config": config.__dict__,
        "inputs": {name: {"path": path, "sha256": sha256_file(path)} for name, path in inputs.items()},
        "files": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (regressions_path, spreads_path, attrition_path, events_path, report_path)
        },
        "environment": collect_run_environment(),
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return report_path
