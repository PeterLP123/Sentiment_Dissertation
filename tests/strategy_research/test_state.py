from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from sentiment_benchmark.strategy_research.state import (
    StateConfig,
    StateEngineError,
    StateEvent,
    apply_state_scales,
    build_state_table,
    estimate_state_scales,
)

SESSIONS = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")


def event(event_id: str, session: str, score: float, *, hour: int = 12, symbol: str = "AAA") -> StateEvent:
    return StateEvent(
        event_id=event_id,
        symbol=symbol,
        session=session,
        available_at_utc=datetime(2026, 1, int(session[-2:]), hour, tzinfo=UTC),
        score=score,
    )


def test_decay_reset_top_up_reduction_flip_and_cap() -> None:
    config = StateConfig(state_cap=1.2)
    rows = build_state_table(
        SESSIONS,
        ["AAA"],
        [
            event("positive", SESSIONS[0], 1.0),
            event("top-up", SESSIONS[0], 1.0, hour=13),
            event("moderate-opposite", SESSIONS[1], -0.5),
            event("extreme-opposite", SESSIONS[2], -1.0),
        ],
        config,
    )
    assert rows[0].final_state == 1.2
    assert 0 < rows[1].final_state < rows[1].state_after_decay
    assert rows[1].reset_reasons == ("opposite_sign_reversal",)
    assert rows[2].final_state < 0
    assert abs(rows[2].final_state) <= config.state_cap
    assert abs(rows[3].final_state) < abs(rows[2].final_state)


def test_same_session_events_are_ordered_by_availability_then_id() -> None:
    timestamp = datetime(2026, 1, 5, 12, tzinfo=UTC)
    rows = build_state_table(
        [SESSIONS[0]],
        ["AAA"],
        [
            StateEvent("b-positive", "AAA", SESSIONS[0], timestamp, 1.0),
            StateEvent("a-negative", "AAA", SESSIONS[0], timestamp, -1.0),
        ],
        StateConfig(),
    )
    assert rows[0].event_ids == ("a-negative", "b-positive")
    assert rows[0].final_state > 0


def test_elapsed_trading_ordinals_drive_decay() -> None:
    rows = build_state_table(
        [SESSIONS[0], SESSIONS[2]],
        ["AAA"],
        [event("positive", SESSIONS[0], 1.0)],
        StateConfig(half_life_sessions=2),
        session_ordinals={SESSIONS[0]: 10, SESSIONS[2]: 12},
    )
    assert rows[1].elapsed_sessions == 2
    assert rows[1].final_state == pytest.approx(0.5)


def test_additive_decay_is_reset_rule_with_zero_reset() -> None:
    events = [event("positive", SESSIONS[0], 1.0), event("opposite", SESSIONS[1], -0.5)]
    additive = build_state_table(SESSIONS[:2], ["AAA"], events, StateConfig(variant="additive_decay"))
    zero_reset = build_state_table(
        SESSIONS[:2], ["AAA"], events, StateConfig(variant="decay_with_reset", reversal_reset=0)
    )
    assert [row.final_state for row in additive] == [row.final_state for row in zero_reset]


def test_cash_and_fixed_hold_baselines_share_interface() -> None:
    scored = [event("signal", SESSIONS[0], -0.5)]
    cash = build_state_table(SESSIONS, ["AAA"], scored, StateConfig(variant="cash"))
    fixed = build_state_table(
        SESSIONS, ["AAA"], scored, StateConfig(variant="last_event_fixed_hold", fixed_hold_sessions=3)
    )
    assert [row.final_state for row in cash] == [0.0] * 4
    assert [row.final_state for row in fixed] == [-0.5, -0.5, -0.5, 0.0]


def test_unknown_event_session_fails_closed() -> None:
    with pytest.raises(StateEngineError, match="undeclared session"):
        build_state_table(SESSIONS[:1], ["AAA"], [event("late", SESSIONS[1], 1.0)], StateConfig())


def test_duplicate_event_identity_is_never_double_counted() -> None:
    duplicate = event("same", SESSIONS[0], 1.0)
    with pytest.raises(StateEngineError, match="duplicate state event_id"):
        build_state_table(SESSIONS[:1], ["AAA"], [duplicate, duplicate], StateConfig())


def test_scales_use_only_development_nonzero_states_and_shrink() -> None:
    events = [
        event("a1", SESSIONS[0], 0.5, symbol="AAA"),
        event("a2", SESSIONS[1], 1.0, symbol="AAA"),
        event("b1", SESSIONS[0], 1.0, symbol="BBB"),
    ]
    rows = build_state_table(SESSIONS, ["AAA", "BBB", "CCC"], events, StateConfig(variant="additive_decay"))
    fitted = estimate_state_scales(rows, SESSIONS[:2], quantile=0.75, shrinkage_k=50, minimum_scale=0.05)
    changed_evaluation = [replace(row, final_state=999.0) if row.session in SESSIONS[2:] else row for row in rows]
    refitted = estimate_state_scales(
        changed_evaluation, SESSIONS[:2], quantile=0.75, shrinkage_k=50, minimum_scale=0.05
    )
    assert fitted == refitted
    estimates = {row.symbol: row for row in fitted.symbols}
    assert estimates["CCC"].support == 0
    assert estimates["CCC"].final_scale == fitted.pooled_scale
    assert 0 < estimates["AAA"].shrinkage_weight < 1


def test_more_support_moves_scale_toward_stock_scale() -> None:
    events = [event(f"a{index}", session, 0.5, symbol="AAA") for index, session in enumerate(SESSIONS)]
    rows = build_state_table(SESSIONS, ["AAA"], events, StateConfig(variant="cash"))
    # Replace the cash path with declared non-zero synthetic development states.
    rows = [replace(row, final_state=value) for row, value in zip(rows, (0.1, 0.2, 0.3, 1.0), strict=True)]
    short = estimate_state_scales(rows, SESSIONS[:1], shrinkage_k=2)
    long = estimate_state_scales(rows, SESSIONS, shrinkage_k=2)
    assert long.symbols[0].shrinkage_weight > short.symbols[0].shrinkage_weight


def test_action_is_bounded_and_applies_strict_no_trade_band() -> None:
    rows = build_state_table(SESSIONS[:1], ["AAA"], [event("small", SESSIONS[0], 0.1)], StateConfig())
    actions = apply_state_scales(rows, {"AAA": 1.0}, no_trade_band=0.2)
    assert actions[0].raw_action == pytest.approx(0.0316122399)
    assert actions[0].action == 0


def test_naive_event_time_and_out_of_range_score_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        StateEvent("event", "AAA", SESSIONS[0], datetime(2026, 1, 5), 1.0)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        event("event", SESSIONS[0], 1.1)
