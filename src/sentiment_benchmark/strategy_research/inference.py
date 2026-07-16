"""Paired date-aware inference for fixed strategy comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_daily_difference: float
    confidence_level: float
    confidence_interval_low: float
    confidence_interval_high: float
    mean_direction: str
    confidence_direction: str
    effective_dates: tuple[str, ...]
    block_length: int
    replications: int
    seed: int
    bootstrap_means: tuple[float, ...]


def paired_block_bootstrap(
    strategy_returns: Mapping[str, float],
    comparator_returns: Mapping[str, float],
    *,
    block_length: int = 5,
    replications: int = 2_000,
    seed: int = 20_260_715,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Bootstrap paired daily return differences using contiguous moving blocks."""

    strategy_dates = set(strategy_returns)
    comparator_dates = set(comparator_returns)
    if strategy_dates != comparator_dates:
        missing_strategy = sorted(comparator_dates - strategy_dates)
        missing_comparator = sorted(strategy_dates - comparator_dates)
        raise ValueError(
            "paired returns must have identical dates; "
            f"missing from strategy={missing_strategy[:3]}, missing from comparator={missing_comparator[:3]}"
        )
    dates = tuple(sorted(strategy_dates))
    if not dates:
        raise ValueError("paired returns must not be empty")
    if block_length < 1 or block_length > len(dates):
        raise ValueError("block_length must be between one and the number of dates")
    if replications < 1:
        raise ValueError("replications must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    differences = np.asarray(
        [strategy_returns[date_key] - comparator_returns[date_key] for date_key in dates], dtype=float
    )
    if not np.all(np.isfinite(differences)):
        raise ValueError("paired returns must be finite")
    observed = float(np.mean(differences))
    random = np.random.default_rng(seed)
    maximum_start = len(differences) - block_length
    block_count = math.ceil(len(differences) / block_length)
    bootstrap_means = np.empty(replications, dtype=float)
    for replication in range(replications):
        starts = random.integers(0, maximum_start + 1, size=block_count)
        sampled = np.concatenate([differences[start : start + block_length] for start in starts])[: len(differences)]
        bootstrap_means[replication] = float(np.mean(sampled))

    alpha = 1 - confidence_level
    low = float(np.quantile(bootstrap_means, alpha / 2))
    high = float(np.quantile(bootstrap_means, 1 - alpha / 2))
    mean_direction = "positive" if observed > 0 else "negative" if observed < 0 else "zero"
    confidence_direction = "positive" if low > 0 else "negative" if high < 0 else "inconclusive"
    return PairedBootstrapResult(
        mean_daily_difference=observed,
        confidence_level=confidence_level,
        confidence_interval_low=low,
        confidence_interval_high=high,
        mean_direction=mean_direction,
        confidence_direction=confidence_direction,
        effective_dates=dates,
        block_length=block_length,
        replications=replications,
        seed=seed,
        bootstrap_means=tuple(float(value) for value in bootstrap_means),
    )
