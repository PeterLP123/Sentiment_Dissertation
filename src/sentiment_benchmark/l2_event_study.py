from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .artifact_io import atomic_write_json, sha256_file
from .self_consistency import entropy_from_counts
from .trading_effectiveness import benjamini_hochberg

PRIMARY_PROMPT = "target_company_soft_label_base"
HORIZONS = (1, 5, 10)
LABEL_VALUE = {"negative": -1, "neutral": 0, "positive": 1}


class L2AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class L2Result:
    output_dir: Path
    item_metrics_path: Path
    event_returns_path: Path
    hypotheses_path: Path


def _majority(labels: list[str]) -> str:
    counts = {label: labels.count(label) for label in LABEL_VALUE}
    return max(LABEL_VALUE, key=lambda label: (counts[label], LABEL_VALUE[label]))


def summarize_item_scores(scores: pd.DataFrame, primary_prompt: str = PRIMARY_PROMPT) -> pd.DataFrame:
    required = {"item_id", "model_id", "prompt_id", "normalized_label", "status", "metadata"}
    missing = required - set(scores.columns)
    if missing:
        raise L2AnalysisError(f"score matrix is missing columns: {', '.join(sorted(missing))}")
    valid = scores[(scores["prompt_id"] == primary_prompt) & (scores["status"] == "success")].copy()
    valid = valid[valid["normalized_label"].isin(LABEL_VALUE)]
    model_rows: list[dict[str, Any]] = []
    for (item_id, model_id), group in valid.groupby(["item_id", "model_id"], sort=True):
        labels = group["normalized_label"].astype(str).tolist()
        probabilities = [value for value in group.get("label_probabilities", []) if isinstance(value, dict)]
        expected = [float(value.get("positive", 0)) - float(value.get("negative", 0)) for value in probabilities]
        model_rows.append(
            {
                "item_id": item_id,
                "model_id": model_id,
                "model_label": _majority(labels),
                "model_score": float(np.mean(expected)) if expected else float(np.mean([LABEL_VALUE[label] for label in labels])),
                "self_consistency_entropy": entropy_from_counts({label: labels.count(label) for label in LABEL_VALUE}),
                "metadata": group.iloc[0]["metadata"],
            }
        )
    models = pd.DataFrame(model_rows)
    rows: list[dict[str, Any]] = []
    for item_id, group in models.groupby("item_id", sort=True):
        labels = group["model_label"].tolist()
        if len(labels) < 4:
            continue
        agreeing = sum(left == right for index, left in enumerate(labels) for right in labels[index + 1 :])
        pairs = len(labels) * (len(labels) - 1) / 2
        agreement = agreeing / pairs
        bucket = "high" if agreement == 1 else "medium" if agreement > 0.4 else "low"
        metadata = group.iloc[0]["metadata"]
        consensus_score = float(group["model_score"].mean())
        consensus_label = _majority(labels)
        rows.append(
            {
                "item_id": item_id,
                "symbol": metadata.get("symbol"),
                "news_date": metadata.get("news_date"),
                "version_created": metadata.get("version_created"),
                "chronological_split": metadata.get("chronological_split"),
                "model_count": len(labels),
                "pairwise_agreement": agreement,
                "agreement_bucket": bucket,
                "ambiguity_entropy": float(group["self_consistency_entropy"].mean()),
                "score_variance": float(group["model_score"].var(ddof=1)),
                "consensus_score": consensus_score,
                "consensus_label": consensus_label,
                "consensus_sign": LABEL_VALUE[consensus_label],
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    counts = result.groupby(["symbol", "news_date"])["item_id"].transform("count")
    result["company_day_weight"] = 1.0 / counts
    return result


def market_model_event_returns(items: pd.DataFrame, prices: pd.DataFrame, market_symbol: str = "^GSPC") -> pd.DataFrame:
    required = {"symbol", "session_date", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise L2AnalysisError(f"price data is missing columns: {', '.join(sorted(missing))}")
    data = prices.copy()
    data["session_date"] = pd.to_datetime(data["session_date"])
    data = data.sort_values(["symbol", "session_date"])
    data["return"] = data.groupby("symbol")["close"].transform(lambda values: np.log(values.astype(float)).diff())
    market = data[data["symbol"] == market_symbol][["session_date", "return"]].rename(columns={"return": "market_return"})
    rows: list[dict[str, Any]] = []
    for item in items.itertuples(index=False):
        stock = data[data["symbol"] == item.symbol][["session_date", "return"]].merge(market, on="session_date", how="inner")
        event_date = pd.Timestamp(item.news_date)
        future = stock[stock["session_date"] >= event_date].reset_index(drop=True)
        history = stock[stock["session_date"] < event_date].dropna().tail(141).head(120)
        if len(future) < max(HORIZONS) or len(history) < 80:
            continue
        x = np.column_stack([np.ones(len(history)), history["market_return"].to_numpy(float)])
        alpha, beta = np.linalg.lstsq(x, history["return"].to_numpy(float), rcond=None)[0]
        event = future.iloc[: max(HORIZONS)].copy()
        event["abnormal"] = event["return"] - (alpha + beta * event["market_return"])
        event["market_adjusted"] = event["return"] - event["market_return"]
        initial = float(event.iloc[0]["abnormal"])
        for horizon in HORIZONS:
            window = event.iloc[:horizon]
            rows.append(
                {
                    **item._asdict(),
                    "event_session": str(window.iloc[0]["session_date"].date()),
                    "horizon": horizon,
                    "car": float(window["abnormal"].sum()),
                    "market_adjusted_car": float(window["market_adjusted"].sum()),
                    "initial_reaction": initial,
                    "directional_car": float(item.consensus_sign * window["abnormal"].sum()),
                    "market_alpha": float(alpha),
                    "market_beta": float(beta),
                }
            )
    return pd.DataFrame(rows)


def _cluster_meat(x: np.ndarray, residual: np.ndarray, clusters: pd.Series) -> np.ndarray:
    meat = np.zeros((x.shape[1], x.shape[1]))
    for value in clusters.astype(str).unique():
        mask = clusters.astype(str).to_numpy() == value
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    return meat


def clustered_wls(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    data = frame.dropna(subset=[outcome, "pairwise_agreement", "ambiguity_entropy", "initial_reaction"]).copy()
    data["high_agreement"] = (data["agreement_bucket"] == "high").astype(float)
    data["low_agreement"] = (data["agreement_bucket"] == "low").astype(float)
    data["high_x_ambiguity"] = data["high_agreement"] * data["ambiguity_entropy"]
    base = data[["high_agreement", "low_agreement", "ambiguity_entropy", "high_x_ambiguity", "initial_reaction"]]
    effects = pd.get_dummies(data[["symbol", "event_session"]].astype(str), drop_first=True, dtype=float)
    design = pd.concat([base, effects], axis=1)
    x = np.column_stack([np.ones(len(data)), design.to_numpy(float)])
    y = data[outcome].to_numpy(float)
    weights = data["company_day_weight"].to_numpy(float)
    root_w = np.sqrt(weights)
    xw, yw = x * root_w[:, None], y * root_w
    bread_inv = np.linalg.pinv(xw.T @ xw)
    coefficients = bread_inv @ xw.T @ yw
    residual = (yw - xw @ coefficients)
    symbol_meat = _cluster_meat(xw, residual, data["symbol"])
    date_meat = _cluster_meat(xw, residual, data["event_session"])
    intersection = data["symbol"].astype(str) + "|" + data["event_session"].astype(str)
    covariance = bread_inv @ (symbol_meat + date_meat - _cluster_meat(xw, residual, intersection)) @ bread_inv
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    names = ["intercept", *design.columns]
    results = []
    for name, coefficient, standard_error in zip(names, coefficients, standard_errors, strict=True):
        statistic = coefficient / standard_error if standard_error else float("nan")
        results.append(
            {
                "term": name,
                "coefficient": coefficient,
                "standard_error": standard_error,
                "z": statistic,
                "p_value": float(2 * stats.norm.sf(abs(statistic))) if math.isfinite(statistic) else float("nan"),
            }
        )
    return pd.DataFrame(results)


def test_h2_hypotheses(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        regression = clustered_wls(events[events["horizon"] == horizon], "directional_car")
        for hypothesis, term in (("H2a", "high_agreement"), ("H2b", "low_agreement"), ("H2c", "high_x_ambiguity")):
            result = regression[regression["term"] == term].iloc[0]
            rows.append({"hypothesis": hypothesis, "horizon": horizon, **result.to_dict()})
    frame = pd.DataFrame(rows)
    frame["q_bh"] = benjamini_hochberg(frame["p_value"].tolist())
    return frame


def analyze_l2(scores_path: str | Path, prices_path: str | Path, output_dir: str | Path) -> L2Result:
    destination = Path(output_dir)
    if destination.exists():
        raise L2AnalysisError(f"refusing to overwrite L2 output: {destination}")
    score_rows = [json.loads(line) for line in Path(scores_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    items = summarize_item_scores(pd.DataFrame(score_rows))
    prices = pd.read_csv(prices_path)
    events = market_model_event_returns(items, prices)
    hypotheses = test_h2_hypotheses(events)
    destination.mkdir(parents=True)
    item_path = destination / "item_metrics.csv"
    event_path = destination / "event_returns.csv"
    hypothesis_path = destination / "hypotheses.csv"
    items.to_csv(item_path, index=False, lineterminator="\n")
    events.to_csv(event_path, index=False, lineterminator="\n")
    hypotheses.to_csv(hypothesis_path, index=False, lineterminator="\n")
    atomic_write_json(destination / "manifest.json", {
        "schema_version": 1,
        "inputs": {str(scores_path): sha256_file(scores_path), str(prices_path): sha256_file(prices_path)},
        "horizons": HORIZONS,
        "market_model": {"market": "^GSPC", "estimation_sessions": 120, "gap_sessions": 21},
        "files": {path.name: sha256_file(path) for path in (item_path, event_path, hypothesis_path)},
    })
    return L2Result(destination, item_path, event_path, hypothesis_path)
