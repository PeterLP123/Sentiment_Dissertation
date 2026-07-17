from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_benchmark.sentiment_surprise_pilot import (
    PilotConfig,
    _load_prices,
    _local_level_one_step,
    add_market_adjusted_outcomes,
    build_daily_sentiment,
    run_pilot,
)


def _price_rows(symbol: str, dates: pd.DatetimeIndex, returns: np.ndarray) -> list[dict[str, object]]:
    closes = 100 * np.exp(np.cumsum(returns))
    return [
        {
            "symbol": symbol,
            "session_date": date.date().isoformat(),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000,
            "repaired": False,
        }
        for date, close in zip(dates, closes, strict=True)
    ]


def test_session_mapping_uses_first_close_strictly_after_timestamp(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(_price_rows("AAA", pd.DatetimeIndex(["2026-01-02", "2026-01-05"]), np.array([0.0, 0.01]))).to_csv(prices_path, index=False)
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {
                "headline_sha256": "a",
                "matched_symbols": "AAA",
                "first_timestamp": "2026-01-02T20:59:00Z",
                "baseline": "finbert",
                "score": 0.2,
                "status": "success",
            },
            {
                "headline_sha256": "b",
                "matched_symbols": "AAA",
                "first_timestamp": "2026-01-02T21:01:00Z",
                "baseline": "finbert",
                "score": -0.4,
                "status": "success",
            },
        ]
    ).to_csv(scores_path, index=False)
    config = PilotConfig(minimum_symbol_news_days=1, minimum_prior_news_days=0, dev20_window=1)
    daily, _ = build_daily_sentiment(scores_path, _load_prices(prices_path, label="test"), config)
    assert daily["session_date"].dt.date.astype(str).tolist() == ["2026-01-02", "2026-01-05"]
    assert daily["level"].tolist() == [0.2, -0.4]


def test_local_level_forecast_does_not_use_current_or_future_observation() -> None:
    values = np.linspace(-0.3, 0.4, 60) + 0.05 * np.sin(np.arange(60))
    changed = values.copy()
    changed[-1] = 100.0
    prediction, prediction_std, parameters = _local_level_one_step(values, 40)
    changed_prediction, changed_std, changed_parameters = _local_level_one_step(changed, 40)
    assert np.allclose(prediction, changed_prediction)
    assert np.allclose(prediction_std, changed_std)
    assert parameters == changed_parameters
    assert prediction[-1] != changed[-1]


def test_market_adjusted_car_excludes_ar0() -> None:
    dates = pd.bdate_range("2026-01-01", periods=14)
    market_returns = np.array([0.0, *([0.01] * 13)])
    stock_returns = np.array([0.0, 0.03, *([0.02] * 12)])
    stock = pd.DataFrame(_price_rows("AAA", dates, stock_returns))
    market = pd.DataFrame(_price_rows("^GSPC", dates, market_returns))
    stock["session_date"] = pd.to_datetime(stock["session_date"])
    market["session_date"] = pd.to_datetime(market["session_date"])
    events = pd.DataFrame([{"symbol": "AAA", "session_date": dates[1]}])
    result = add_market_adjusted_outcomes(events, stock, market, "^GSPC").iloc[0]
    assert np.isclose(result["ar0"], 0.02)
    assert np.isclose(result["car_p1_p1"], 0.01)
    assert np.isclose(result["car_p1_p5"], 0.05)
    assert np.isclose(result["car_p1_p10"], 0.10)


def test_full_pilot_writes_frozen_outputs(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-12-01", periods=100)
    news_dates = dates[1:91]
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    score_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    market_returns = 0.0005 + 0.0002 * np.sin(np.arange(len(dates)))
    price_rows.extend(_price_rows("^GSPC", dates, market_returns))
    for symbol_index, symbol in enumerate(symbols):
        scores = 0.4 * np.sin(np.arange(len(news_dates)) / 5 + symbol_index)
        for index, (date, score) in enumerate(zip(news_dates, scores, strict=True)):
            score_rows.append(
                {
                    "headline_sha256": f"{symbol}-{index}",
                    "matched_symbols": symbol,
                    "first_timestamp": f"{date.date().isoformat()}T15:00:00Z",
                    "baseline": "finbert",
                    "score": score,
                    "status": "success",
                }
            )
        stock_returns = market_returns + 0.001 * np.roll(scores.mean() + np.sin(np.arange(len(dates))), symbol_index)
        price_rows.extend(_price_rows(symbol, dates, stock_returns))

    scores_path = tmp_path / "scores.csv"
    stocks_path = tmp_path / "stocks.csv"
    market_path = tmp_path / "market.csv"
    pd.DataFrame(score_rows).to_csv(scores_path, index=False)
    prices = pd.DataFrame(price_rows)
    prices[prices["symbol"] != "^GSPC"].to_csv(stocks_path, index=False)
    prices[prices["symbol"] == "^GSPC"].to_csv(market_path, index=False)
    output = tmp_path / "pilot"
    report = run_pilot(
        scores_path,
        stocks_path,
        market_path,
        output,
        PilotConfig(bootstrap_samples=100, random_seed=7),
    )
    assert report.is_file()
    assert (output / "manifest.json").is_file()
    regressions = pd.read_csv(output / "regression_results.csv")
    assert set(regressions["model"]) == {"M1", "M2", "M3", "M4", "M5", "M6"}
    assert len(pd.read_csv(output / "quintile_spreads.csv")) == 2
    assert "Ten-line plain-English summary" in report.read_text(encoding="utf-8")
