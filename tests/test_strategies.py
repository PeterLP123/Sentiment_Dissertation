"""Tests for the pluggable strategy layer: registry, default equivalence, the
conviction-sizing idea, the seams (signal/universe/decision/evaluation), and the
generic sweep."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sentiment_benchmark.backtest import DailySignal, DecisionPolicyConfig, make_trading_decisions
from sentiment_benchmark.prices import PriceRow
from sentiment_benchmark.strategies import (
    AllEventsSelector,
    EventStudyEvaluator,
    MagnitudePolicy,
    MeanSignalBuilder,
    MinArticlesSelector,
    PortfolioEvaluator,
    Strategy,
    ThresholdPolicy,
    available,
    get,
    register,
)
from sentiment_benchmark.strategy_sweep import sweep_strategy


# --- duck-typed stand-ins for MergedArticle / SentimentScore (builder reads attrs) ---
@dataclass
class _Article:
    article_id: str
    symbol: str
    published_date_local: str
    screening_decision: str
    providers: list[str]
    published_at: str | None


@dataclass
class _Score:
    article_id: str
    symbol: str
    news_date: str
    scorer_id: str
    label_value: int | None


def _signal(news_date: str, mean_score: float | None, signal: str = "positive", value: int = 1) -> DailySignal:
    return DailySignal("AAPL", news_date, "model", 3, 3, mean_score, signal, value)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_lists_and_resolves_builtin_strategies() -> None:
    ids = {strategy.id for strategy in available()}
    assert {"sentiment_threshold_v1", "headline_sentiment_threshold_v1", "sentiment_magnitude_v1"} <= ids
    assert get("sentiment_magnitude_v1").id == "sentiment_magnitude_v1"


def test_get_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError):
        get("does_not_exist")


def test_register_duplicate_raises() -> None:
    with pytest.raises(ValueError):
        register(Strategy(id="sentiment_threshold_v1", description="dup", axes=("decision",)))


# --------------------------------------------------------------------------- #
# Seam 1: signal construction
# --------------------------------------------------------------------------- #
def test_mean_signal_builder_aggregates_and_signs() -> None:
    articles = [
        _Article("a1", "AAPL", "2026-06-10", "include", ["lseg"], "2026-06-10T12:00:00+00:00"),
        _Article("a2", "AAPL", "2026-06-10", "include", ["newsapi"], None),
    ]
    scores = [
        _Score("a1", "AAPL", "2026-06-10", "model", 1),
        _Score("a2", "AAPL", "2026-06-10", "model", 1),
    ]
    signals = MeanSignalBuilder().build(articles, scores, ("model",), ("2026-06-10",), ("AAPL",))
    assert len(signals) == 1
    signal = signals[0]
    assert signal.mean_score == 1.0
    assert (signal.signal, signal.signal_value) == ("positive", 1)
    assert signal.article_count == 2  # both included
    assert signal.valid_count == 2
    assert signal.availability_timestamp == "2026-06-10T12:00:00+00:00"  # only the lseg article contributes


def test_mean_signal_builder_neutral_and_missing() -> None:
    articles = [_Article("a1", "AAPL", "2026-06-10", "include", ["x"], None)]
    scores = [
        _Score("a1", "AAPL", "2026-06-10", "model", 1),
        _Score("a2", "AAPL", "2026-06-10", "model", -1),
    ]
    signal = MeanSignalBuilder().build(articles, scores, ("model",), ("2026-06-10",), ("AAPL",))[0]
    assert signal.mean_score == 0.0
    assert (signal.signal, signal.signal_value) == ("neutral", 0)
    # A scorer/day with no scores yields a no-signal row.
    empty = MeanSignalBuilder().build(articles, [], ("model",), ("2026-06-10",), ("AAPL",))[0]
    assert empty.mean_score is None
    assert empty.signal is None and empty.signal_value is None


# --------------------------------------------------------------------------- #
# Seam 2: universe / event selection
# --------------------------------------------------------------------------- #
def test_event_selectors() -> None:
    signals = [_signal("2026-06-10", 1.0), DailySignal("AAPL", "2026-06-11", "model", 0, 0, None, None, None)]
    assert AllEventsSelector().select(signals) == signals
    kept = MinArticlesSelector(min_articles=1).select(signals)
    assert [s.news_date for s in kept] == ["2026-06-10"]  # drops the 0-article day


# --------------------------------------------------------------------------- #
# Seam 3: decision / sizing
# --------------------------------------------------------------------------- #
def test_threshold_policy_is_byte_identical_to_legacy() -> None:
    signals = [
        _signal("2026-06-10", 1.0),
        _signal("2026-06-10", -1.0, "negative", -1),
        _signal("2026-06-10", 0.0, "neutral", 0),
    ]
    config = DecisionPolicyConfig(threshold=0.5, min_valid_stories=1)
    assert ThresholdPolicy().decide(signals, config) == make_trading_decisions(signals, config)


def test_magnitude_policy_scales_and_clips() -> None:
    config = DecisionPolicyConfig(threshold=0.0)
    policy = MagnitudePolicy(scale=2.0, max_position=1.0)
    decisions = policy.decide(
        [
            _signal("2026-06-10", 0.5),  # 0.5/2.0 = 0.25 long
            _signal("2026-06-10", -0.5, "negative", -1),  # -0.25 short
            _signal("2026-06-10", 5.0),  # clipped to +1.0
        ],
        config,
    )
    assert [round(d.position or 0.0, 4) for d in decisions] == [0.25, -0.25, 1.0]
    assert [d.action for d in decisions] == ["buy", "sell", "buy"]
    assert [d.action_value for d in decisions] == [1, -1, 1]
    assert all(d.policy_version == "sentiment_magnitude_v1" for d in decisions)


def test_magnitude_policy_respects_threshold_and_min_valid() -> None:
    policy = MagnitudePolicy(scale=1.0)
    below = policy.decide([_signal("2026-06-10", 0.05)], DecisionPolicyConfig(threshold=0.1))[0]
    assert below.action == "hold" and below.position == 0.0 and below.reason == "inside_no_trade_band"
    thin = DailySignal("AAPL", "2026-06-10", "model", 3, 1, 0.9, "positive", 1)
    gated = policy.decide([thin], DecisionPolicyConfig(threshold=0.0, min_valid_stories=2))[0]
    assert gated.action == "hold" and gated.reason == "insufficient_valid_stories"


# --------------------------------------------------------------------------- #
# Strategy bundle: param split
# --------------------------------------------------------------------------- #
def test_policy_for_splits_shared_and_idea_params() -> None:
    strategy = get("sentiment_magnitude_v1")
    config, policy = strategy.policy_for(
        {"threshold": 0.1, "min_valid_stories": 2, "transaction_cost_bps_per_side": 5.0, "scale": 2.0, "max_position": 0.5}
    )
    assert config.threshold == 0.1
    assert config.min_valid_stories == 2
    assert config.transaction_cost_bps_per_side == 5.0
    assert config.policy_version == "sentiment_magnitude_v1"
    assert isinstance(policy, MagnitudePolicy)
    assert (policy.scale, policy.max_position) == (2.0, 0.5)


# --------------------------------------------------------------------------- #
# Seam 4: evaluation frame
# --------------------------------------------------------------------------- #
def _prices() -> list[PriceRow]:
    return [
        PriceRow("AAPL", "2026-06-10", 100.0, 101.0, 99.0, 100.0, 1000.0, False),
        PriceRow("AAPL", "2026-06-11", 100.0, 111.0, 99.0, 110.0, 1000.0, False),  # entry open 100 -> +10%
        PriceRow("AAPL", "2026-06-12", 110.0, 111.0, 109.0, 110.0, 1000.0, False),
    ]


def test_event_study_evaluator_runs_backtest() -> None:
    result = EventStudyEvaluator().evaluate(
        [_signal("2026-06-10", 1.0)],
        _prices(),
        DecisionPolicyConfig(),
        decision_fn=None,
        horizons=(1,),
        notional_usd=10_000.0,
    )
    assert EventStudyEvaluator().frame == "event_study"
    assert len(result.returns) == 1
    assert result.returns[0].strategy_return == pytest.approx(0.10)
    assert result.equity_curve  # non-empty


def test_portfolio_evaluator_is_not_yet_implemented() -> None:
    assert PortfolioEvaluator().frame == "cross_sectional"
    with pytest.raises(NotImplementedError):
        PortfolioEvaluator().evaluate()


# --------------------------------------------------------------------------- #
# Generic sweep over a strategy's own parameter space
# --------------------------------------------------------------------------- #
def test_sweep_strategy_explores_idea_params() -> None:
    # Per horizon-1 return is the entry session's open->close move; the 06-08 signal
    # enters on 06-09 (+10%) and the 06-09 signal enters on 06-10 (+2%).
    moves = {
        "2026-06-08": (100.0, 100.0),
        "2026-06-09": (100.0, 110.0),
        "2026-06-10": (100.0, 102.0),
        "2026-06-11": (100.0, 100.0),
        "2026-06-15": (100.0, 100.0),
        "2026-06-16": (100.0, 102.0),
        "2026-06-17": (100.0, 110.0),
        "2026-06-18": (100.0, 100.0),
    }
    prices = [PriceRow("AAPL", d, o, max(o, c), min(o, c), c, 1000.0, False) for d, (o, c) in moves.items()]
    signals = [_signal("2026-06-08", 0.5), _signal("2026-06-09", 2.0), _signal("2026-06-15", 0.5), _signal("2026-06-16", 2.0)]
    results = sweep_strategy(
        signals,
        prices,
        get("sentiment_magnitude_v1"),
        horizons=(1,),
        split_date="2026-06-15",
        notional_usd=10_000.0,
        metric="mean_return",
        param_space={"threshold": (0.0,), "scale": (0.5, 1.0, 2.0), "max_position": (1.0,)},
    )
    # 3 scale values × 1 threshold × 1 horizon.
    assert len(results) == 3
    assert all(r.strategy_id == "sentiment_magnitude_v1" for r in results)
    assert {"scale=" in r.params for r in results} == {True}
    # scale=0.5 -> both train scores saturate to full ±1 -> best train mean (0.06).
    best = next(r for r in results if r.selected)
    assert "scale=0.5" in best.params
    assert best.train_metric == pytest.approx(0.06)
