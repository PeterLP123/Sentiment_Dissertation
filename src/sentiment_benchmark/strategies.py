"""Pluggable strategy layer — an *idea* as a first-class, swappable unit.

A sentiment trading idea spans four seams of one pipeline::

    raw scores → [SignalBuilder] → DailySignals → [EventSelector] → tradeable
               → [DecisionPolicy] → TradingDecisions → [Evaluator] → returns

This module makes each seam a small protocol with a default that reproduces the
current behaviour byte-for-byte, bundles a chosen implementation for each seam
into a :class:`Strategy`, and exposes a registry so a run/sweep can name an idea
(``--strategy sentiment_magnitude_v1``) and have the shared machinery
(sweep, effectiveness battery, experiment registry) operate on it generically.

Adding an idea = implement the seam(s) it changes (usually just a
:class:`DecisionPolicy`), then ``register(Strategy(...))``. Everything else is
already generic.

Dependency direction: ``prices`` ← ``backtest`` ← ``strategies``. The signal
builder is duck-typed against the article/score objects so this module does not
import (and cannot cycle with) ``trading_strategy``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .backtest import (
    BacktestResult,
    DailySignal,
    DecisionPolicyConfig,
    IndexFallback,
    TradingDecision,
    make_trading_decisions,
    run_backtest,
)
from .prices import PriceRow

# The four seams an idea can customise (used for discovery in ``list-strategies``).
AXES = ("signal", "universe", "decision", "evaluation")

# Shared decision params that live on ``DecisionPolicyConfig``; every other swept
# parameter is idea-specific and passed to the strategy's ``make_policy``.
CONFIG_PARAM_NAMES = (
    "threshold",
    "min_valid_stories",
    "transaction_cost_bps_per_side",
    "short_borrow_bps_per_day",
)


# --------------------------------------------------------------------------- #
# Seam 1: signal construction  (raw article scores → per-event DailySignal)
# --------------------------------------------------------------------------- #
class ArticleLike(Protocol):
    # Read-only members so concrete attributes (e.g. ``providers: list[str]``)
    # match structurally — settable protocol members are invariant.
    @property
    def article_id(self) -> str: ...
    @property
    def symbol(self) -> str: ...
    @property
    def published_date_local(self) -> str: ...
    @property
    def screening_decision(self) -> str: ...
    @property
    def providers(self) -> Collection[str]: ...
    @property
    def published_at(self) -> str | None: ...


class ScoreLike(Protocol):
    @property
    def article_id(self) -> str: ...
    @property
    def symbol(self) -> str: ...
    @property
    def news_date(self) -> str: ...
    @property
    def scorer_id(self) -> str: ...
    @property
    def label_value(self) -> int | None: ...


class SignalBuilder(Protocol):
    def build(
        self,
        articles: Sequence[ArticleLike],
        scores: Sequence[ScoreLike],
        scorer_ids: Sequence[str],
        dates: Sequence[str],
        symbols: Sequence[str],
    ) -> list[DailySignal]: ...


@dataclass(frozen=True)
class MeanSignalBuilder:
    """Default builder: per ``(symbol, date, scorer)`` the signal is the mean of
    the per-article label values (−1/0/+1), and its sign gives the direction.
    This is the extracted body of the legacy ``trading_strategy.daily_signals``,
    so the default path is unchanged."""

    def build(
        self,
        articles: Sequence[ArticleLike],
        scores: Sequence[ScoreLike],
        scorer_ids: Sequence[str],
        dates: Sequence[str],
        symbols: Sequence[str],
    ) -> list[DailySignal]:
        accepted_counts = Counter(
            (article.symbol, article.published_date_local)
            for article in articles
            if article.screening_decision == "include"
        )
        article_by_id = {article.article_id: article for article in articles}
        grouped: dict[tuple[str, str, str], list[ScoreLike]] = defaultdict(list)
        for score in scores:
            grouped[(score.symbol, score.news_date, score.scorer_id)].append(score)
        results: list[DailySignal] = []
        for news_date in dates:
            for symbol in symbols:
                for scorer_id in scorer_ids:
                    valid_scores = [
                        score
                        for score in grouped.get((symbol, news_date, scorer_id), [])
                        if score.label_value is not None
                    ]
                    values = [int(score.label_value) for score in valid_scores if score.label_value is not None]
                    availability_values = [
                        article_by_id[score.article_id].published_at
                        for score in valid_scores
                        if score.article_id in article_by_id
                        and "lseg" in article_by_id[score.article_id].providers
                        and article_by_id[score.article_id].published_at
                    ]
                    mean = sum(values) / len(values) if values else None
                    if mean is None:
                        signal: str | None = None
                        signal_value: int | None = None
                    elif mean > 0:
                        signal, signal_value = "positive", 1
                    elif mean < 0:
                        signal, signal_value = "negative", -1
                    else:
                        signal, signal_value = "neutral", 0
                    results.append(
                        DailySignal(
                            symbol=symbol,
                            news_date=news_date,
                            scorer_id=scorer_id,
                            article_count=accepted_counts[(symbol, news_date)],
                            valid_count=len(values),
                            mean_score=mean,
                            signal=signal,
                            signal_value=signal_value,
                            availability_timestamp=max(
                                (value for value in availability_values if value is not None), default=None
                            ),
                        )
                    )
        return results


# --------------------------------------------------------------------------- #
# Seam 2: universe / event selection  (which signals are tradeable)
# --------------------------------------------------------------------------- #
class EventSelector(Protocol):
    def select(self, signals: Sequence[DailySignal]) -> list[DailySignal]: ...


@dataclass(frozen=True)
class AllEventsSelector:
    """Default: every company-day is tradeable (decision policy still filters
    holds). Pass-through keeps the default path unchanged."""

    def select(self, signals: Sequence[DailySignal]) -> list[DailySignal]:
        return list(signals)


@dataclass(frozen=True)
class MinArticlesSelector:
    """Only trade company-days backed by at least ``min_articles`` accepted
    stories — a simple demonstration that the universe seam is real."""

    min_articles: int = 1

    def select(self, signals: Sequence[DailySignal]) -> list[DailySignal]:
        return [signal for signal in signals if signal.article_count >= self.min_articles]


# --------------------------------------------------------------------------- #
# Seam 3: decision / sizing  (DailySignal → TradingDecision, primary axis)
# --------------------------------------------------------------------------- #
class DecisionPolicy(Protocol):
    def decide(
        self,
        signals: Sequence[DailySignal],
        config: DecisionPolicyConfig,
    ) -> list[TradingDecision]: ...


@dataclass(frozen=True)
class ThresholdPolicy:
    """Default equal-weight long/short: full ±1 position once the mean sentiment
    clears the (symmetric) no-trade band. Wraps ``make_trading_decisions`` so it
    is byte-identical to the legacy rule."""

    def decide(
        self,
        signals: Sequence[DailySignal],
        config: DecisionPolicyConfig,
    ) -> list[TradingDecision]:
        return make_trading_decisions(list(signals), config)


@dataclass(frozen=True)
class MagnitudePolicy:
    """Conviction-weighted long/short (the worked new idea). Direction is the
    sign of the mean sentiment; the position *size* scales with conviction,
    ``position = clip(mean_score / scale, -max_position, +max_position)``. Same
    entry gate (``threshold``/``min_valid_stories``) and hold reasons as the
    threshold policy, so it is a clean sibling that only changes sizing."""

    scale: float = 1.0
    max_position: float = 1.0
    policy_version: str = "sentiment_magnitude_v1"

    def _size(self, mean_score: float) -> float:
        raw = mean_score / self.scale if self.scale else mean_score
        return max(-self.max_position, min(self.max_position, raw))

    def decide(
        self,
        signals: Sequence[DailySignal],
        config: DecisionPolicyConfig,
    ) -> list[TradingDecision]:
        decisions: list[TradingDecision] = []
        for signal in signals:
            if signal.mean_score is None or signal.valid_count == 0:
                action, action_value, position, reason = "hold", 0, 0.0, "no_valid_sentiment"
            elif signal.valid_count < config.min_valid_stories:
                action, action_value, position, reason = "hold", 0, 0.0, "insufficient_valid_stories"
            elif signal.mean_score > 0 and signal.mean_score >= config.threshold:
                action, action_value = "buy", 1
                position, reason = self._size(signal.mean_score), "positive_mean_meets_threshold"
            elif signal.mean_score < 0 and signal.mean_score <= -config.threshold:
                action, action_value = "sell", -1
                position, reason = self._size(signal.mean_score), "negative_mean_meets_threshold"
            else:
                action, action_value, position, reason = "hold", 0, 0.0, "inside_no_trade_band"
            decisions.append(
                TradingDecision(
                    symbol=signal.symbol,
                    news_date=signal.news_date,
                    scorer_id=signal.scorer_id,
                    article_count=signal.article_count,
                    valid_count=signal.valid_count,
                    mean_score=signal.mean_score,
                    action=action,
                    action_value=action_value,
                    threshold=config.threshold,
                    min_valid_stories=config.min_valid_stories,
                    policy_version=self.policy_version,
                    reason=reason,
                    availability_timestamp=signal.availability_timestamp,
                    position=position,
                )
            )
        return decisions


# --------------------------------------------------------------------------- #
# Seam 4: evaluation frame  (decisions + prices → returns)
# --------------------------------------------------------------------------- #
class Evaluator(Protocol):
    @property
    def frame(self) -> str: ...

    def evaluate(
        self,
        signals: Sequence[DailySignal],
        prices: Sequence[PriceRow],
        config: DecisionPolicyConfig,
        *,
        decision_fn: Callable[[list[DailySignal], DecisionPolicyConfig], list[TradingDecision]] | None,
        horizons: tuple[int, ...],
        notional_usd: float,
        index_fallback: IndexFallback | None = None,
        timezone: str = "America/New_York",
        use_decision_policy: bool = True,
        equity_horizon: int | None = None,
    ) -> BacktestResult: ...


@dataclass(frozen=True)
class EventStudyEvaluator:
    """Default frame: each event is entered at the next session's open and exited
    at each horizon's close, fixed notional per event. Delegates to
    ``backtest.run_backtest``."""

    frame: str = "event_study"

    def evaluate(
        self,
        signals: Sequence[DailySignal],
        prices: Sequence[PriceRow],
        config: DecisionPolicyConfig,
        *,
        decision_fn: Callable[[list[DailySignal], DecisionPolicyConfig], list[TradingDecision]] | None,
        horizons: tuple[int, ...],
        notional_usd: float,
        index_fallback: IndexFallback | None = None,
        timezone: str = "America/New_York",
        use_decision_policy: bool = True,
        equity_horizon: int | None = None,
    ) -> BacktestResult:
        return run_backtest(
            list(signals),
            list(prices),
            config,
            horizons=horizons,
            notional_usd=notional_usd,
            index_fallback=index_fallback,
            timezone=timezone,
            use_decision_policy=use_decision_policy,
            equity_horizon=equity_horizon,
            decision_fn=decision_fn,
        )


@dataclass(frozen=True)
class PortfolioEvaluator:
    """Cross-sectional long/short book (rank events per day, form baskets).

    Interface only in this first pass — the evaluation-frame axis is designed so
    ideas can opt in later; the full build is a flagged fast-follow."""

    frame: str = "cross_sectional"

    def evaluate(self, *args: object, **kwargs: object) -> BacktestResult:
        raise NotImplementedError(
            "cross-sectional portfolio evaluation is a planned fast-follow; use eval_frame='event_study'"
        )


# --------------------------------------------------------------------------- #
# Strategy bundle + registry
# --------------------------------------------------------------------------- #
def _threshold_policy(**_: object) -> DecisionPolicy:
    return ThresholdPolicy()


def _magnitude_policy(scale: float = 1.0, max_position: float = 1.0, **_: object) -> DecisionPolicy:
    return MagnitudePolicy(scale=float(scale), max_position=float(max_position))


@dataclass(frozen=True)
class Strategy:
    """A named idea: one implementation per seam plus the parameter space the
    sweep should explore. Defaults reproduce the legacy threshold strategy."""

    id: str
    description: str
    axes: tuple[str, ...]
    make_policy: Callable[..., DecisionPolicy] = _threshold_policy
    default_param_space: Mapping[str, tuple[float | int, ...]] = field(default_factory=dict)
    signal_builder: SignalBuilder = field(default_factory=MeanSignalBuilder)
    event_selector: EventSelector = field(default_factory=AllEventsSelector)
    evaluator: Evaluator = field(default_factory=EventStudyEvaluator)

    def param_space(self) -> dict[str, tuple[float | int, ...]]:
        """Sweepable parameters → candidate values (horizon is swept separately)."""
        return dict(self.default_param_space)

    def policy_for(self, params: Mapping[str, float]) -> tuple[DecisionPolicyConfig, DecisionPolicy]:
        """Split a flat sweep point into the shared ``DecisionPolicyConfig`` and a
        configured idea policy. Used by the parameter sweep."""
        config = DecisionPolicyConfig(
            min_valid_stories=int(params.get("min_valid_stories", 1)),
            threshold=float(params.get("threshold", 0.0)),
            transaction_cost_bps_per_side=float(params.get("transaction_cost_bps_per_side", 0.0)),
            short_borrow_bps_per_day=float(params.get("short_borrow_bps_per_day", 0.0)),
            policy_version=self.id,
        )
        idea = {name: value for name, value in params.items() if name not in CONFIG_PARAM_NAMES}
        return config, self.make_policy(**idea)


_REGISTRY: dict[str, Strategy] = {}


def register(strategy: Strategy) -> Strategy:
    if strategy.id in _REGISTRY:
        raise ValueError(f"strategy {strategy.id!r} is already registered")
    _REGISTRY[strategy.id] = strategy
    return strategy


def get(strategy_id: str) -> Strategy:
    try:
        return _REGISTRY[strategy_id]
    except KeyError:
        raise KeyError(f"unknown strategy {strategy_id!r}; available: {sorted(_REGISTRY)}") from None


def available() -> list[Strategy]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


register(
    Strategy(
        id="sentiment_threshold_v1",
        description="Equal-weight long/short: full ±1 position once mean sentiment clears the no-trade band.",
        axes=("decision",),
        make_policy=_threshold_policy,
        default_param_space={
            "threshold": (0.0, 0.05, 0.1, 0.2),
            "min_valid_stories": (1,),
            "transaction_cost_bps_per_side": (0.0,),
        },
    )
)

register(
    Strategy(
        id="headline_sentiment_threshold_v1",
        description="Threshold long/short applied to per-headline signals (headline information-value pipeline).",
        axes=("decision",),
        make_policy=_threshold_policy,
        default_param_space={
            "threshold": (0.0,),
            "min_valid_stories": (1,),
            "transaction_cost_bps_per_side": (0.0,),
        },
    )
)

register(
    Strategy(
        id="sentiment_magnitude_v1",
        description="Conviction-weighted long/short: position size scales with |mean sentiment| (clipped).",
        axes=("decision",),
        make_policy=_magnitude_policy,
        default_param_space={
            "threshold": (0.0, 0.1),
            "min_valid_stories": (1,),
            "transaction_cost_bps_per_side": (0.0,),
            "scale": (0.5, 1.0, 2.0),
            "max_position": (1.0,),
        },
    )
)
