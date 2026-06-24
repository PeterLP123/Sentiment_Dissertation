from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sentiment_benchmark.trading_analysis import (
    TradingAnalysisError,
    analyze_trading_run,
    bootstrap_mean_interval,
    compare_runs,
    summarize_agreement,
    summarize_horizons,
    summarize_leave_one_company_out,
    summarize_signal_distribution,
    summarize_source_yield,
)

MODELS = (
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
    "meta-llama/llama-3.3-70b-instruct",
)


def test_bootstrap_mean_interval_is_seeded_and_deterministic() -> None:
    first = bootstrap_mean_interval([1.0, 2.0, 3.0], seed=7, resamples=500)
    second = bootstrap_mean_interval([1.0, 2.0, 3.0], seed=7, resamples=500)

    assert first == second
    assert first.mean == 2.0
    assert first.lower <= first.mean <= first.upper
    assert first.n == 3


def test_bootstrap_mean_interval_singleton_is_degenerate() -> None:
    interval = bootstrap_mean_interval([2.5], resamples=10)

    assert interval.mean == interval.lower == interval.upper == 2.5


@pytest.mark.parametrize(
    ("values", "resamples", "confidence"),
    [([], 10, 0.95), ([1.0], 0, 0.95), ([1.0], 10, 1.0)],
)
def test_bootstrap_mean_interval_rejects_invalid_settings(
    values: list[float],
    resamples: int,
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_interval(values, resamples=resamples, confidence=confidence)


def test_summarize_horizons_separates_neutral_events_and_trades() -> None:
    returns = pd.DataFrame(
        [
            {"scorer_id": "consensus/majority", "horizon": 1, "strategy_return_pct": 2.0, "signal_value": 1, "pnl_usd": 200},
            {"scorer_id": "consensus/majority", "horizon": 1, "strategy_return_pct": -1.0, "signal_value": -1, "pnl_usd": -100},
            {"scorer_id": "consensus/majority", "horizon": 1, "strategy_return_pct": 0.0, "signal_value": 0, "pnl_usd": 0},
        ]
    )

    row = summarize_horizons(returns, resamples=200).iloc[0]

    assert row["n_events"] == 3
    assert row["n_trades"] == 2
    assert row["n_neutral"] == 1
    assert row["mean_return_pct"] == pytest.approx(1 / 3)
    assert row["mean_traded_return_pct"] == pytest.approx(0.5)
    assert row["trade_hit_rate"] == pytest.approx(0.5)


def test_summarize_horizons_requires_return_schema() -> None:
    with pytest.raises(TradingAnalysisError, match="missing columns"):
        summarize_horizons(pd.DataFrame({"scorer_id": ["x"]}))


def test_summarize_signal_distribution_preserves_missing_events() -> None:
    signals = pd.DataFrame(
        [
            {"scorer_id": "consensus/majority", "signal": "positive"},
            {"scorer_id": "consensus/majority", "signal": None},
        ]
    )

    result = summarize_signal_distribution(signals).set_index("signal")

    assert result.loc["positive", "event_count"] == 1
    assert result.loc["missing", "event_count"] == 1


def test_summarize_source_yield_separates_provider_coverage() -> None:
    articles = pd.DataFrame(
        [
            {"symbol": "AAA", "screening_decision": "include", "providers": "tavily"},
            {"symbol": "AAA", "screening_decision": "include", "providers": "newsapi"},
            {"symbol": "AAA", "screening_decision": "include", "providers": "newsapi; tavily"},
            {"symbol": "AAA", "screening_decision": "exclude", "providers": "newsapi"},
        ]
    )

    row = summarize_source_yield(articles).iloc[0]

    assert row["discovered_articles"] == 4
    assert row["accepted_articles"] == 3
    assert row["accepted_newsapi_only"] == 1
    assert row["accepted_tavily_only"] == 1
    assert row["accepted_both"] == 1


def test_summarize_agreement_counts_unanimous_majority_and_three_way_split() -> None:
    article_labels = [
        ("a1", ["positive", "positive", "positive"]),
        ("a2", ["positive", "positive", "neutral"]),
        ("a3", ["positive", "neutral", "negative"]),
    ]
    score_rows = []
    for article_id, labels in article_labels:
        score_rows.extend(
            {"article_id": article_id, "scorer_id": scorer, "label": label}
            for scorer, label in zip(MODELS, labels, strict=True)
        )
        score_rows.append({"article_id": article_id, "scorer_id": "baseline/vader", "label": labels[0]})
    signal_rows = [
        {"symbol": "AAA", "news_date": "2026-01-01", "scorer_id": MODELS[0], "signal": "positive"},
        {"symbol": "AAA", "news_date": "2026-01-01", "scorer_id": MODELS[1], "signal": "positive"},
        {"symbol": "AAA", "news_date": "2026-01-01", "scorer_id": MODELS[2], "signal": "positive"},
        {"symbol": "AAA", "news_date": "2026-01-01", "scorer_id": "consensus/majority", "signal": "positive"},
        {"symbol": "AAA", "news_date": "2026-01-01", "scorer_id": "baseline/vader", "signal": "negative"},
    ]

    metrics, pairwise = summarize_agreement(pd.DataFrame(score_rows), pd.DataFrame(signal_rows), MODELS)
    values = metrics.set_index("metric")

    assert values.loc["article_llm_unanimous", "numerator"] == 1
    assert values.loc["article_llm_majority_not_unanimous", "numerator"] == 1
    assert values.loc["article_llm_three_way_split", "numerator"] == 1
    assert values.loc["event_llm_signal_unanimous", "rate"] == 1.0
    assert values.loc["event_consensus_vader_signal_agreement", "rate"] == 0.0
    assert len(pairwise) == 6


def test_leave_one_company_out_recomputes_each_horizon() -> None:
    returns = pd.DataFrame(
        [
            {"symbol": "AAA", "scorer_id": "consensus/majority", "horizon": 1, "strategy_return_pct": 2.0},
            {"symbol": "BBB", "scorer_id": "consensus/majority", "horizon": 1, "strategy_return_pct": -1.0},
        ]
    )

    result = summarize_leave_one_company_out(returns).set_index("omitted_symbol")

    assert result.loc["(none)", "mean_return_pct"] == pytest.approx(0.5)
    assert result.loc["AAA", "mean_return_pct"] == pytest.approx(-1.0)
    assert result.loc["BBB", "mean_return_pct"] == pytest.approx(2.0)


def test_compare_runs_reports_percentage_point_change() -> None:
    current = pd.DataFrame(
        [{"scorer_id": "consensus/majority", "horizon": 1, "strategy_return_pct": 1.5}]
    )
    previous = pd.DataFrame(
        [{"scorer_id": "consensus/majority", "horizon": 1, "strategy_return_pct": 0.5}]
    )

    row = compare_runs(current, previous).iloc[0]

    assert row["mean_return_change_pp"] == pytest.approx(1.0)


def _write_completed_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    articles_path = tmp_path / "derived" / "articles.csv"
    articles_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "article_id": "a1",
                "symbol": "AAA",
                "screening_decision": "include",
                "providers": "tavily",
            }
        ]
    ).to_csv(articles_path, index=False)
    score_rows = [
        {"article_id": "a1", "scorer_id": scorer, "label": "positive"}
        for scorer in (*MODELS, "consensus/majority", "baseline/vader")
    ]
    pd.DataFrame(score_rows).to_csv(run_dir / "sentiment_scores.csv", index=False)
    signal_rows = [
        {
            "symbol": "AAA",
            "news_date": "2026-01-01",
            "scorer_id": scorer,
            "signal": "positive",
            "signal_value": 1,
        }
        for scorer in (*MODELS, "consensus/majority", "baseline/vader")
    ]
    pd.DataFrame(signal_rows).to_csv(run_dir / "daily_signals.csv", index=False)
    return_rows = [
        {
            "symbol": "AAA",
            "news_date": "2026-01-01",
            "scorer_id": scorer,
            "horizon": 1,
            "entry_date": "2026-01-02",
            "market_return": 0.01,
            "strategy_return_pct": 1.0,
            "signal_value": 1,
            "pnl_usd": 100.0,
        }
        for scorer in (*MODELS, "consensus/majority", "baseline/vader")
    ]
    pd.DataFrame(return_rows).to_csv(run_dir / "returns.csv", index=False)
    manifest = {
        "status": "completed",
        "run_id": "fixture",
        "files": {str(articles_path): "unused-in-test"},
        "settings": {"symbols": ["AAA"], "dates": ["2026-01-01"], "models": list(MODELS)},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, tmp_path / "analysis"


def test_analyze_trading_run_writes_manifest_report_tables_and_plots(tmp_path: Path) -> None:
    run_dir, output_dir = _write_completed_fixture(tmp_path)

    result = analyze_trading_run(run_dir, output_dir, resamples=50)

    assert result.summary_path.exists()
    assert result.manifest_path.exists()
    assert (output_dir / "mean_returns_by_horizon.png").exists()
    assert (output_dir / "horizon_summary.csv").exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["bootstrap"]["seed"] == 42
    assert manifest["bootstrap"]["resamples"] == 50
    assert manifest["causal_claim"] is False


def test_analyze_trading_run_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    run_dir, output_dir = _write_completed_fixture(tmp_path)
    output_dir.mkdir()

    with pytest.raises(TradingAnalysisError, match="already exists"):
        analyze_trading_run(run_dir, output_dir)
