from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from sentiment_benchmark.fnspid_moment2_finbert import (
    FnspidMoment2FinbertConfig,
    _attach_risk_inputs,
    _final_report,
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


def test_final_report_preserves_part1_section_and_has_ten_summary_lines(tmp_path: Path) -> None:
    part1 = "# Part 1 — VADER second-moment and tail smoke\n\nimmutable Part 1 result\n\n"
    report_path = tmp_path / "report.md"
    report_path.write_text(
        "# old header\n\n" + part1 + "# Part 2 — full FinBERT\n\n**PENDING.**\n",
        encoding="utf-8",
    )
    lambdas = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "lambda": 0.01,
                "clustered_se": 0.01,
                "t_stat": 1.0,
                "p_value": 0.3,
                "bh_q_value": 0.3,
                "ci_95_low": -0.01,
                "ci_95_high": 0.03,
                "events": 10,
                "date_clusters": 5,
                "r_squared": 0.01,
            }
            for horizon in range(1, 11)
        ]
    )
    split = pd.DataFrame(
        [
            {
                "content_class": label,
                "lambda": value,
                "clustered_se": 0.01,
                "t_stat": 1.0,
                "p_value": 0.3,
                "ci_95_low": -0.01,
                "ci_95_high": 0.03,
                "events": 10,
                "date_clusters": 5,
            }
            for label, value in (("PRICE_RECAP", 0.02), ("OTHER", 0.01))
        ]
    )
    empty = pd.DataFrame()
    model_fits = pd.DataFrame(
        [
            {
                "analysis": "absolute_ar_h1",
                "model": "controls_only",
                "observations": 10,
                "date_clusters": 5,
                "r_squared": 0.1,
                "delta_r_squared_vs_controls": 0.0,
                "mean_outcome": 0.01,
                "verdict_scope": True,
            }
        ]
    )
    coefficients = pd.DataFrame(
        [
            {
                "analysis": "absolute_ar_h1",
                "model": "controls_plus_sentiment",
                "regressor": "finbert_mean",
                "coefficient": 0.01,
                "clustered_se": 0.01,
                "t_stat": 1.0,
                "p_value": 0.3,
                "bh_q_value": 0.3,
                "ci_95_low": -0.01,
                "ci_95_high": 0.03,
            }
        ]
    )
    manifest = {
        "finbert": {"scored": 100, "model_revision": "revision"},
        "interruption_history": [
            {"event": "previous_session_killed_after_event_checkpoint_before_part1_artifacts"},
            {"event": "previous_session_killed_during_finbert_scoring", "finbert_scored_at_resume": 25},
        ],
    }
    _final_report(
        path=report_path,
        part1_model_fits=empty,
        part1_coefficients=empty,
        attrition=empty,
        vader_verdict="YES",
        finbert_lambdas=lambdas,
        finbert_split=split,
        finbert_model_fits=model_fits,
        finbert_coefficients=coefficients,
        mean_verdict="NO",
        finbert_risk_verdict="MARGINAL",
        recap_verdict="NOT ESTABLISHED",
        manifest=manifest,
    )
    result = report_path.read_text(encoding="utf-8")
    assert part1 in result
    summary = result.split("## Ten-line plain-English summary\n\n", 1)[1].split("\n\n", 1)[0]
    assert [line.split(".", 1)[0] for line in summary.splitlines()] == [str(index) for index in range(1, 11)]
    assert "two process kills occurred" in result
    assert "25/100 rows" in result
