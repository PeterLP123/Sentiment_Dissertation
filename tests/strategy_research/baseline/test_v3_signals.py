from __future__ import annotations

from dataclasses import replace

import pytest

from sentiment_benchmark.strategy_research.baseline.signals import FirmSessionSignal
from sentiment_benchmark.strategy_research.baseline.v3.config import SignalConfig
from sentiment_benchmark.strategy_research.baseline.v3.signals import build_rank_reversal_targets


def _config(**changes: object) -> SignalConfig:
    base = SignalConfig(
        construction="lagged_cross_sectional_rank_reversal",
        formation_lag_sessions=1,
        lookback_sessions=5,
        tail_fraction=0.20,
        minimum_ranked_firms=10,
        minimum_names_per_side=2,
        direction="contrarian",
    )
    return replace(base, **changes)


def _signal(session: str, symbol: str, score: float) -> FirmSessionSignal:
    action = 1 if score > 0 else -1 if score < 0 else 0
    return FirmSessionSignal(
        scorer="finbert",
        symbol=symbol,
        session=session,
        event_count=1,
        positive_events=int(action > 0),
        negative_events=int(action < 0),
        neutral_events=int(action == 0),
        mean_hard_label=float(action),
        mean_continuous_score=score,
        action=action,
    )


def test_rank_reversal_is_lagged_contrarian_diversified_and_neutral() -> None:
    symbols = tuple(f"S{index:02d}" for index in range(10))
    formation = [_signal("2026-01-02", symbol, float(index)) for index, symbol in enumerate(symbols)]
    # Extreme current-session values must not affect the current target.
    current = [_signal("2026-01-05", symbol, float(100 - index)) for index, symbol in enumerate(symbols)]
    result = build_rank_reversal_targets(
        [*formation, *current],
        sessions=("2026-01-02", "2026-01-05"),
        symbols=symbols,
        gross_exposure=1.0,
        config=_config(),
    )
    first, second = result.targets
    assert first.gross_exposure == 0.0
    weights = second.weights()
    assert weights["S00"] == pytest.approx(0.25)
    assert weights["S01"] == pytest.approx(0.25)
    assert weights["S08"] == pytest.approx(-0.25)
    assert weights["S09"] == pytest.approx(-0.25)
    assert sum(weights.values()) == pytest.approx(0.0)
    assert sum(abs(value) for value in weights.values()) == pytest.approx(1.0)
    assert max(abs(value) for value in weights.values()) <= 0.25


def test_rank_reversal_uses_continuous_neutral_scores_and_fails_to_cash_on_thin_breadth() -> None:
    symbols = tuple(f"S{index:02d}" for index in range(10))
    signals = [_signal("2026-01-02", symbol, score) for symbol, score in zip(symbols[:9], range(9), strict=True)]
    result = build_rank_reversal_targets(
        signals,
        sessions=("2026-01-02", "2026-01-05"),
        symbols=symbols,
        gross_exposure=1.0,
        config=_config(),
    )
    assert result.targets[1].gross_exposure == 0.0
    assert result.session_rows[1]["no_trade_reason"] == "minimum_ranked_firms_not_met"
    assert result.targets[1].gross_exposure == 0.0


def test_rank_reversal_rejects_duplicate_firm_session_signal() -> None:
    symbols = tuple(f"S{index:02d}" for index in range(10))
    row = _signal("2026-01-02", "S00", 0.1)
    with pytest.raises(ValueError, match="duplicate"):
        build_rank_reversal_targets(
            [row, row],
            sessions=("2026-01-02", "2026-01-05"),
            symbols=symbols,
            gross_exposure=1.0,
            config=_config(),
        )
