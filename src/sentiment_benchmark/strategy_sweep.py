"""Parameter sweep for the trading strategy with train/test discipline.

Tunes the decision policy (threshold, min-valid-stories, cost) and horizon over a
grid, selecting on a *training* split only and reporting *held-out* test metrics,
so tuning cannot leak. Pure (no I/O); consumes the backtest core. Output is a
tidy list of ``SweepResult`` rows that feed the parameter-sweep heatmap.

Dependency direction: ``prices``/``backtest`` ← ``strategy_sweep``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
from statistics import fmean, stdev

from .backtest import DailySignal, DecisionFn, DecisionPolicyConfig, IndexFallback, ReturnRow, run_backtest
from .prices import PriceRow
from .strategies import CONFIG_PARAM_NAMES, Strategy
from .strategies import get as get_strategy

SELECTION_METRICS = ("mean_return", "hit_rate", "sharpe")


def _float_or_none(value: str) -> float | None:
    return float(value) if value not in ("", "None") else None


def _int_or_none(value: str) -> int | None:
    return int(value) if value not in ("", "None") else None


def load_signals_csv(path: str | Path) -> list[DailySignal]:
    """Reconstruct ``DailySignal`` rows from a completed run's daily_signals.csv."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            DailySignal(
                symbol=row["symbol"],
                news_date=row["news_date"],
                scorer_id=row["scorer_id"],
                article_count=int(row["article_count"]),
                valid_count=int(row["valid_count"]),
                mean_score=_float_or_none(row["mean_score"]),
                signal=row["signal"] or None,
                signal_value=_int_or_none(row["signal_value"]),
                availability_timestamp=row.get("availability_timestamp") or None,
            )
            for row in csv.DictReader(handle)
        ]


def load_prices_csv(path: str | Path) -> list[PriceRow]:
    """Reconstruct ``PriceRow`` rows from a completed run's prices.csv."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            PriceRow(
                symbol=row["symbol"],
                session_date=row["session_date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                repaired=row["repaired"] == "True",
            )
            for row in csv.DictReader(handle)
        ]


@dataclass(frozen=True)
class ParameterGrid:
    thresholds: tuple[float, ...]
    horizons: tuple[int, ...]
    min_valid_stories: tuple[int, ...] = (1,)
    transaction_cost_bps_per_side: tuple[float, ...] = (0.0,)

    def points(self) -> Iterator[tuple[float, int, int, float]]:
        yield from product(
            self.thresholds,
            self.horizons,
            self.min_valid_stories,
            self.transaction_cost_bps_per_side,
        )


@dataclass(frozen=True)
class SweepResult:
    threshold: float
    horizon: int
    min_valid_stories: int
    transaction_cost_bps_per_side: float
    n_train: int
    n_test: int
    train_metric: float | None
    test_metric: float | None
    test_mean_return: float | None
    test_hit_rate: float | None
    selected: bool = False
    strategy_id: str = "sentiment_threshold_v1"
    # Idea-specific params for this point (e.g. "scale=1.0") beyond the shared
    # threshold/min-valid/cost columns; empty for the default threshold strategy.
    params: str = ""


def _vector(returns: list[ReturnRow]) -> list[float]:
    return [row.net_strategy_return for row in returns]


def _mean_return(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _hit_rate(values: list[float]) -> float | None:
    return sum(1 for value in values if value > 0) / len(values) if values else None


def _sharpe(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    spread = stdev(values)
    return fmean(values) / spread if spread > 0 else None


def _metric(returns: list[ReturnRow], name: str) -> float | None:
    values = _vector(returns)
    if name == "mean_return":
        return _mean_return(values)
    if name == "hit_rate":
        return _hit_rate(values)
    if name == "sharpe":
        return _sharpe(values)
    raise ValueError(f"unknown selection metric: {name!r}")


def chronological_split_date(news_dates: Sequence[str], train_fraction: float) -> str:
    """Return the ISO date that splits sorted unique ``news_dates`` so that about
    ``train_fraction`` of the distinct dates fall strictly before it (train), the
    rest on/after (test). Splitting on dates — not rows — keeps a news-day intact
    on one side, avoiding leakage across the boundary."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1 (exclusive)")
    unique = sorted(set(news_dates))
    if len(unique) < 2:
        raise ValueError("need at least two distinct news dates to split")
    index = max(1, min(len(unique) - 1, round(len(unique) * train_fraction)))
    return unique[index]


def _returns_for(
    signals: list[DailySignal],
    prices: list[PriceRow],
    policy: DecisionPolicyConfig,
    horizon: int,
    *,
    notional_usd: float,
    index_fallback: IndexFallback | None,
    timezone: str,
    decision_fn: DecisionFn | None = None,
) -> list[ReturnRow]:
    if not signals:
        return []
    return run_backtest(
        signals,
        prices,
        policy,
        horizons=(horizon,),
        notional_usd=notional_usd,
        index_fallback=index_fallback,
        timezone=timezone,
        decision_fn=decision_fn,
    ).returns


def _params_str(params: Mapping[str, float]) -> str:
    """Compact, stable serialisation of idea-specific params for the CSV record."""
    return "; ".join(f"{name}={value}" for name, value in sorted(params.items()))


def _flag_selected(results: list[SweepResult]) -> list[SweepResult]:
    """Flag the single point with the best *train* selection metric."""
    best: SweepResult | None = None
    best_metric = float("-inf")
    for result in results:
        if result.train_metric is not None and result.train_metric > best_metric:
            best_metric = result.train_metric
            best = result
    if best is None:
        return results
    return [replace(result, selected=result is best) for result in results]


def sweep_strategy(
    signals: list[DailySignal],
    prices: list[PriceRow],
    strategy: Strategy,
    *,
    horizons: tuple[int, ...],
    split_date: str,
    notional_usd: float,
    metric: str = "sharpe",
    param_space: Mapping[str, Sequence[float | int]] | None = None,
    index_fallback: IndexFallback | None = None,
    timezone: str = "America/New_York",
) -> list[SweepResult]:
    """Evaluate a registered strategy over its declared parameter space × horizons,
    selecting on the *train* split only and reporting *held-out* test metrics. The
    decision rule comes from the strategy, so any idea plugs into the same tuning
    discipline. ``split_date`` is the first test date; ``param_space`` overrides the
    strategy's default grid."""
    if metric not in SELECTION_METRICS:
        raise ValueError(f"metric must be one of {SELECTION_METRICS}")
    space = dict(param_space) if param_space is not None else strategy.param_space()
    names = list(space)
    train = [signal for signal in signals if signal.news_date < split_date]
    test = [signal for signal in signals if signal.news_date >= split_date]

    results: list[SweepResult] = []
    for combo in product(*(tuple(space[name]) for name in names)):
        point = dict(zip(names, combo, strict=True))
        config, policy = strategy.policy_for(point)
        idea_params = {name: value for name, value in point.items() if name not in CONFIG_PARAM_NAMES}
        for horizon in horizons:
            train_returns = _returns_for(
                train, prices, config, horizon,
                notional_usd=notional_usd, index_fallback=index_fallback,
                timezone=timezone, decision_fn=policy.decide,
            )
            test_returns = _returns_for(
                test, prices, config, horizon,
                notional_usd=notional_usd, index_fallback=index_fallback,
                timezone=timezone, decision_fn=policy.decide,
            )
            results.append(
                SweepResult(
                    threshold=config.threshold,
                    horizon=horizon,
                    min_valid_stories=config.min_valid_stories,
                    transaction_cost_bps_per_side=config.transaction_cost_bps_per_side,
                    n_train=len(train_returns),
                    n_test=len(test_returns),
                    train_metric=_metric(train_returns, metric),
                    test_metric=_metric(test_returns, metric),
                    test_mean_return=_metric(test_returns, "mean_return"),
                    test_hit_rate=_metric(test_returns, "hit_rate"),
                    strategy_id=strategy.id,
                    params=_params_str(idea_params),
                )
            )
    return _flag_selected(results)


def sweep(
    signals: list[DailySignal],
    prices: list[PriceRow],
    grid: ParameterGrid,
    *,
    split_date: str,
    notional_usd: float,
    metric: str = "sharpe",
    index_fallback: IndexFallback | None = None,
    timezone: str = "America/New_York",
) -> list[SweepResult]:
    """Backwards-compatible sweep of the default threshold strategy over a
    :class:`ParameterGrid`; a thin wrapper over :func:`sweep_strategy`."""
    strategy = get_strategy("sentiment_threshold_v1")
    param_space: dict[str, Sequence[float | int]] = {
        "threshold": grid.thresholds,
        "min_valid_stories": grid.min_valid_stories,
        "transaction_cost_bps_per_side": grid.transaction_cost_bps_per_side,
    }
    return sweep_strategy(
        signals,
        prices,
        strategy,
        horizons=grid.horizons,
        split_date=split_date,
        notional_usd=notional_usd,
        metric=metric,
        param_space=param_space,
        index_fallback=index_fallback,
        timezone=timezone,
    )


def write_sweep_csv(results: list[SweepResult], path: str | Path) -> None:
    """Write sweep rows to CSV for the parameter-sweep heatmap and the record."""
    names = [field.name for field in fields(SweepResult)]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for result in results:
            writer.writerow({name: getattr(result, name) for name in names})
