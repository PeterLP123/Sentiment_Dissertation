from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from sentiment_benchmark.strategy_research.directional_event_gate import SelectedSessionEvent
from sentiment_benchmark.strategy_research.independent_replication import (
    BootstrapMean,
    FoldResult,
    ReplicationTrade,
    bootstrap_session_mean,
    calculate_negative_h2_trades,
    evaluate_replication,
    load_frozen_replication,
)


def _selected(session: str, symbol: str, direction: int = -1) -> SelectedSessionEvent:
    return SelectedSessionEvent(
        session=session,
        symbol=symbol,
        event_id=f"{symbol}-{session}",
        direction=direction,
        direction_severity="negative" if direction < 0 else "positive",
        materiality="high",
        novelty="high",
        eligible_event_count=1,
    )


def test_frozen_real_replication_contract_verifies_manifests_without_opening_prices() -> None:
    config_path = Path("experiments/strategy_midcap_negative_h2_replication_20260717.toml")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    local_manifests = (config["news"]["corpus_manifest"], config["prices"]["manifest_path"])
    if any(not Path(path).is_file() for path in local_manifests):
        pytest.skip("historical LSEG manifest check requires authorised local collection and price manifests")
    # Present but changed inputs still fail the frozen loader's hash checks.
    frozen = load_frozen_replication(config_path)

    assert frozen.experiment_id == "strategy-midcap-negative-h2-replication-20260717"
    assert frozen.payload["strategy"]["holding_sessions"] == 2


def test_negative_h2_trades_charge_both_sides_and_exclude_fold_crossings() -> None:
    sessions = ("S0", "S1", "S2", "S3", "S4", "S5")
    folds = (sessions[:3], sessions[3:])
    prices = {
        **{("AAA", session): 100.0 for session in sessions},
        **{("BBB", session): 100.0 for session in sessions},
    }
    prices[("AAA", "S2")] = 98.0
    selected = (
        _selected("S0", "AAA"),
        _selected("S1", "AAA"),
        _selected("S3", "BBB", 1),
    )

    trades = calculate_negative_h2_trades(selected, prices, sessions, ("AAA", "BBB"), folds)

    assert len(trades) == 1
    assert trades[0].gross_short_return == pytest.approx(0.02)
    assert trades[0].net_short_return == pytest.approx(0.018)
    assert trades[0].panel_hedged_net_short_return == pytest.approx(0.008)


def test_session_block_bootstrap_is_deterministic() -> None:
    sessions = tuple(f"S{index:02d}" for index in range(12))
    trades = tuple(
        ReplicationTrade(session, session, f"X{index % 3}", 0.01, value, value)
        for index, (session, value) in enumerate(zip(sessions, (0.01, 0.02, -0.01) * 4, strict=True))
    )

    first = bootstrap_session_mean(
        trades,
        sessions,
        block_length=3,
        replications=250,
        seed=20260715,
        confidence_level=0.99,
    )
    second = bootstrap_session_mean(
        tuple(reversed(trades)),
        sessions,
        block_length=3,
        replications=250,
        seed=20260715,
        confidence_level=0.99,
    )

    assert first == second
    assert first.observed_mean == pytest.approx(0.02 / 3)


def test_replication_gate_requires_support_all_folds_p_value_and_positive_lower_bound() -> None:
    trades = tuple(ReplicationTrade("S0", "S2", f"X{index:02d}", 0.01, 0.008, 0.008) for index in range(18))
    folds = tuple(FoldResult(index, "S0", "S2", 10, 10, 0.008, 0.008) for index in range(3))
    bootstrap = BootstrapMean(0.008, 0.001, 0.015, 0.009, 2000, 2000, 5, 20260715)
    acceptance = {
        "minimum_total_trades": 18,
        "minimum_supported_stocks": 18,
        "minimum_trades_per_fold": 10,
        "required_positive_folds": 3,
    }
    inference = {"one_sided_p_value_maximum": 0.01}

    passed = evaluate_replication(trades, folds, bootstrap, acceptance, inference)
    failed = evaluate_replication(
        trades,
        (replace(folds[0], mean_net_short_return=-0.001), *folds[1:]),
        bootstrap,
        acceptance,
        inference,
    )

    assert passed.passed
    assert not failed.passed
