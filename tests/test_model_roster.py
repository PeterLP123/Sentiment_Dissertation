"""Tests for the knowledge-cutoff registry and score annotation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sentiment_benchmark.model_roster import (
    is_post_cutoff,
    knowledge_cutoff,
    roster_cutoff,
)
from sentiment_benchmark.trading_strategy import (
    SentimentScore,
    annotate_scores_with_cutoff,
    load_trading_config,
)

PILOT_CONFIG = Path("configs/trading_pilot_3co.toml")


def _score(scorer_id: str, news_date: str) -> SentimentScore:
    return SentimentScore(
        article_id="a1",
        symbol="AAPL",
        news_date=news_date,
        scorer_id=scorer_id,
        scorer_kind="llm",
        label="positive",
        label_value=1,
        status="ok",
        parse_status="ok",
        raw_output="positive",
        compound=None,
        latency_ms=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        generation_id=None,
        total_cost_usd=None,
        error=None,
    )


def test_knowledge_cutoff_full_id_and_unknown() -> None:
    assert knowledge_cutoff("openai/gpt-4o-mini") == date(2023, 10, 1)
    assert knowledge_cutoff("some/unknown-model") is None


def test_knowledge_cutoff_override_takes_precedence() -> None:
    assert knowledge_cutoff("openai/gpt-4o-mini", {"openai/gpt-4o-mini": "2024-04-01"}) == date(2024, 4, 1)
    # An override keyed by the provider-suffix also matches a full provider id.
    assert knowledge_cutoff("openai/gpt-4o-mini", {"gpt-4o-mini": "2024-05-01"}) == date(2024, 5, 1)


def test_roster_cutoff_is_latest_known() -> None:
    models = ("openai/gpt-4o-mini", "google/gemini-2.5-flash-lite", "x/unknown")
    # gemini 2025-01-01 is later than gpt-4o-mini 2023-10-01; unknown ignored.
    assert roster_cutoff(models) == date(2025, 1, 1)
    assert roster_cutoff(("x/unknown", "y/also-unknown")) is None


def test_is_post_cutoff_cases() -> None:
    cut = date(2024, 1, 1)
    assert is_post_cutoff(date(2024, 6, 1), cut) is True
    assert is_post_cutoff(date(2023, 6, 1), cut) is False
    assert is_post_cutoff(date(2024, 6, 1), None) is None
    assert is_post_cutoff(None, cut) is None


def test_annotate_marks_post_cutoff_and_consensus_and_baseline() -> None:
    config = load_trading_config(PILOT_CONFIG)
    scores = [
        _score("openai/gpt-4o-mini", "2026-06-08"),  # well after cutoff
        _score("openai/gpt-4o-mini", "2023-01-01"),  # before cutoff
        _score("consensus/majority", "2026-06-08"),  # roster-max cutoff
        _score("baseline/vader", "2026-06-08"),  # no cutoff
    ]

    annotated = annotate_scores_with_cutoff(scores, config)

    assert annotated[0].is_post_cutoff is True
    assert annotated[0].model_knowledge_cutoff == "2023-10-01"
    assert annotated[1].is_post_cutoff is False
    # consensus uses the latest roster cutoff (gemini-2.5 @ 2025-01-01) -> post.
    assert annotated[2].model_knowledge_cutoff == "2025-01-01"
    assert annotated[2].is_post_cutoff is True
    # dictionary/ML baselines have no knowledge cutoff.
    assert annotated[3].model_knowledge_cutoff is None
    assert annotated[3].is_post_cutoff is None
