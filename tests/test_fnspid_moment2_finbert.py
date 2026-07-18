from __future__ import annotations

import math

import pandas as pd

from sentiment_benchmark.fnspid_moment2_finbert import (
    FnspidMoment2FinbertConfig,
    _attach_risk_inputs,
    estimate_risk_suite,
)


def test_risk_suite_bh_covers_only_six_full_model_sentiment_coefficients() -> None:
    dates = pd.date_range("2011-01-03", periods=40, freq="YS")
    frame = pd.DataFrame(
        {
            "session_date": dates,
            "abs_ar_h1": [0.01 + index / 10_000 for index in range(40)],
            "tail_h1": [float(index % 7 == 0) for index in range(40)],
            "prior5_abs_ar": [0.02 + index / 20_000 for index in range(40)],
            "log_article_count": [math.log1p(1 + index % 3) for index in range(40)],
            "mean_score": [(-1) ** index * 0.1 + index / 1000 for index in range(40)],
            "negative_share": [float(index % 4 == 0) for index in range(40)],
        }
    )
    config = FnspidMoment2FinbertConfig(tail_definition_end_year=2024, tail_evaluation_start_year=2025)
    fits, coefficients, verdict = estimate_risk_suite(
        frame,
        scorer="test",
        mean_predictor="mean_score",
        negative_share_predictor="negative_share",
        config=config,
    )
    assert len(fits) == 6
    adjusted = coefficients["bh_q_value"].notna()
    assert adjusted.sum() == 6
    assert set(coefficients.loc[adjusted, "regressor"]) == {"mean_score", "negative_share"}
    assert verdict in {"YES", "NO", "MARGINAL"}


def test_prior_five_session_volatility_excludes_news_session_return() -> None:
    dates = pd.date_range("2010-12-27", periods=12, freq="B")
    market_close = [100.0] * len(dates)
    stock_returns = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.50, 0.07, 0.08, 0.09, 0.10, 0.11]
    stock_close = [100.0]
    for value in stock_returns[1:]:
        stock_close.append(stock_close[-1] * math.exp(value))
    prices = {
        "SPY": pd.DataFrame({"session_date": dates, "adjusted_close": market_close}),
        "AAA": pd.DataFrame({"session_date": dates, "adjusted_close": stock_close}),
    }
    news_date = dates[6]
    panel = pd.DataFrame({"symbol": ["AAA"], "session_date": [news_date]})
    result, _ = _attach_risk_inputs(
        panel,
        prices,
        FnspidMoment2FinbertConfig(start_year=2010, tail_definition_end_year=2011, horizons=(1,)),
    )
    assert math.isclose(float(result.iloc[0]["prior5_abs_ar"]), 0.03, rel_tol=1e-12)
