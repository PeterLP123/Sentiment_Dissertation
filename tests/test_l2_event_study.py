import pandas as pd

from sentiment_benchmark.l2_event_study import market_model_event_returns, summarize_item_scores


def test_item_metrics_separate_agreement_and_self_consistency() -> None:
    rows = []
    labels = ["positive", "positive", "positive", "positive", "negative"]
    for model_index, label in enumerate(labels):
        for sample in range(5):
            sampled = "neutral" if model_index == 0 and sample == 4 else label
            rows.append({
                "item_id": "event|AAA",
                "model_id": f"m{model_index}",
                "prompt_id": "target_company_soft_label_base",
                "sample_index": sample,
                "normalized_label": sampled,
                "label_probabilities": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
                "status": "success",
                "metadata": {"symbol": "AAA", "news_date": "2026-01-05", "version_created": "2026-01-05T10:00:00Z"},
            })
    result = summarize_item_scores(pd.DataFrame(rows)).iloc[0]
    assert result["pairwise_agreement"] == 0.6
    assert result["agreement_bucket"] == "medium"
    assert result["ambiguity_entropy"] > 0
    assert result["company_day_weight"] == 1


def test_market_model_recovers_zero_abnormal_return() -> None:
    dates = pd.bdate_range("2025-01-01", periods=180)
    market_returns = [0.001] * len(dates)
    market_close = [100.0]
    stock_close = [50.0]
    for value in market_returns[1:]:
        market_close.append(market_close[-1] * (2.718281828 ** value))
        stock_close.append(stock_close[-1] * (2.718281828 ** (0.0002 + 1.5 * value)))
    prices = pd.DataFrame([
        *({"symbol": "^GSPC", "session_date": str(date.date()), "close": close} for date, close in zip(dates, market_close, strict=True)),
        *({"symbol": "AAA", "session_date": str(date.date()), "close": close} for date, close in zip(dates, stock_close, strict=True)),
    ])
    items = pd.DataFrame([{
        "item_id": "e1",
        "symbol": "AAA",
        "news_date": str(dates[150].date()),
        "consensus_sign": 1,
        "company_day_weight": 1.0,
        "pairwise_agreement": 1.0,
        "agreement_bucket": "high",
        "ambiguity_entropy": 0.0,
    }])
    result = market_model_event_returns(items, prices)
    assert len(result) == 3
    assert result["car"].abs().max() < 1e-10
