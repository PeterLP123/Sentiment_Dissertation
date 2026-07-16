from __future__ import annotations

import pytest

from sentiment_benchmark.strategy_research.diagnostics import (
    calculate_portfolio_metrics,
    calculate_stock_diagnostics,
    contribution_concentration,
    leave_one_stock_out_returns,
    returns_by_time_block,
)
from sentiment_benchmark.strategy_research.ledger import (
    LedgerError,
    assert_no_boundary_crossing,
    evaluation_rows,
    run_open_to_open_ledger,
)
from sentiment_benchmark.strategy_research.market import OpenToOpenReturn
from sentiment_benchmark.strategy_research.portfolio import PositionTarget, TargetPortfolio


def target(session: str, weights: dict[str, float]) -> TargetPortfolio:
    positions = tuple(
        PositionTarget(symbol, 1.0 if weight > 0 else -1.0, 0.02, weight / 0.02, weight)
        for symbol, weight in sorted(weights.items())
    )
    long = sum(weight for weight in weights.values() if weight > 0)
    short = sum(-weight for weight in weights.values() if weight < 0)
    return TargetPortfolio(
        session=session,
        positions=positions,
        gross_exposure=long + short,
        net_exposure=long - short,
        long_exposure=long,
        short_exposure=short,
        cash_weight=1 - (long - short),
    )


def test_zero_return_entry_and_reversal_have_exact_turnover_and_cost() -> None:
    targets = [target("2026-01-05", {"AAA": 0.1}), target("2026-01-06", {"AAA": -0.1})]
    returns = [
        OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0.0),
        OpenToOpenReturn("AAA", "2026-01-06", "2026-01-07", 0.0),
    ]
    rows = run_open_to_open_ledger(targets, returns, cost_rate_per_side=0.001)
    assert rows[0].turnover == pytest.approx(0.1)
    assert rows[1].turnover == pytest.approx(0.2)
    assert rows[0].transaction_cost == pytest.approx(0.0001)
    assert rows[1].transaction_cost == pytest.approx(0.0002)
    assert rows[-1].final_liquidation
    assert rows[-1].turnover == pytest.approx(0.1)


def test_current_weight_is_drifted_by_realized_open_return() -> None:
    targets = [target("2026-01-05", {"AAA": 0.1}), target("2026-01-06", {"AAA": 0.1})]
    returns = [
        OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0.10),
        OpenToOpenReturn("AAA", "2026-01-06", "2026-01-07", 0.0),
    ]
    rows = run_open_to_open_ledger(targets, returns, cost_rate_per_side=0, force_final_liquidation=False)
    expected_current = 0.1 * 1.1 / 1.01
    assert dict(rows[1].current_weights)["AAA"] == pytest.approx(expected_current)
    assert rows[1].turnover == pytest.approx(abs(0.1 - expected_current))


def test_target_at_open_earns_that_sessions_open_to_open_return() -> None:
    rows = run_open_to_open_ledger(
        [target("2026-01-05", {"AAA": 0.2, "BBB": -0.1})],
        [
            OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0.10),
            OpenToOpenReturn("BBB", "2026-01-05", "2026-01-06", -0.20),
        ],
        cost_rate_per_side=0,
        force_final_liquidation=False,
    )
    assert rows[0].gross_return == pytest.approx(0.04)
    assert rows[0].next_session == "2026-01-06"
    assert len(rows[0].target_weights) == 2


def test_missing_active_return_fails_closed() -> None:
    with pytest.raises(LedgerError, match="missing return"):
        run_open_to_open_ledger(
            [target("2026-01-05", {"AAA": 0.1, "BBB": 0.1})],
            [OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0)],
        )


def test_ledger_requires_a_target_for_every_decision_session() -> None:
    with pytest.raises(LedgerError, match="supply a target for every decision session"):
        run_open_to_open_ledger(
            [target("2026-01-05", {"AAA": 0.1}), target("2026-01-07", {"AAA": 0.1})],
            [
                OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0),
                OpenToOpenReturn("AAA", "2026-01-07", "2026-01-08", 0),
            ],
        )


def test_boundary_selection_never_includes_pre_boundary_interval() -> None:
    rows = run_open_to_open_ledger(
        [target("2026-01-05", {"AAA": 0.1}), target("2026-01-06", {"AAA": 0.1})],
        [
            OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0),
            OpenToOpenReturn("AAA", "2026-01-06", "2026-01-07", 0),
        ],
        force_final_liquidation=False,
    )
    selected = evaluation_rows(rows, "2026-01-06")
    assert [row.session for row in selected] == ["2026-01-06"]
    assert_no_boundary_crossing(rows, "2026-01-06")


def test_evaluation_accounting_resets_nav_but_carries_revalued_position() -> None:
    rows = run_open_to_open_ledger(
        [target("2026-01-05", {"AAA": 0.1}), target("2026-01-06", {"AAA": -0.1})],
        [
            OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0.10),
            OpenToOpenReturn("AAA", "2026-01-06", "2026-01-07", 0.0),
        ],
        cost_rate_per_side=0.001,
        accounting_reset_session="2026-01-06",
        force_final_liquidation=False,
    )
    carried_weight = 0.1 * 1.1 / 1.01
    evaluation = rows[1]
    assert evaluation.start_nav_usd == 1_000_000
    assert dict(evaluation.current_weights)["AAA"] == pytest.approx(carried_weight)
    assert evaluation.turnover == pytest.approx(abs(-0.1 - carried_weight))
    assert evaluation.orders[0].transaction_cost_usd == pytest.approx(
        1_000_000 * evaluation.orders[0].transaction_cost_return
    )


def test_accounting_reset_must_be_on_the_target_session_spine() -> None:
    with pytest.raises(LedgerError, match="accounting reset session"):
        run_open_to_open_ledger(
            [target("2026-01-05", {"AAA": 0.1})],
            [OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0.0)],
            accounting_reset_session="2026-01-06",
        )


def test_cash_path_records_zero_cost_final_liquidation_for_horizon_alignment() -> None:
    rows = run_open_to_open_ledger(
        [target("2026-01-05", {})],
        [OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0.01)],
    )
    assert len(rows) == 2
    assert rows[-1].session == "2026-01-06"
    assert rows[-1].final_liquidation
    assert rows[-1].turnover == 0
    assert rows[-1].net_return == 0


def test_diagnostics_recombine_fixed_contributions_without_stock_selection() -> None:
    rows = run_open_to_open_ledger(
        [
            target("2026-01-05", {"AAA": 0.2, "BBB": -0.1}),
            target("2026-01-06", {"AAA": 0.1, "BBB": -0.1}),
        ],
        [
            OpenToOpenReturn("AAA", "2026-01-05", "2026-01-06", 0.10),
            OpenToOpenReturn("BBB", "2026-01-05", "2026-01-06", -0.10),
            OpenToOpenReturn("AAA", "2026-01-06", "2026-01-07", -0.05),
            OpenToOpenReturn("BBB", "2026-01-06", "2026-01-07", 0.02),
        ],
        cost_rate_per_side=0.001,
    )
    metrics = calculate_portfolio_metrics(rows)
    stocks = calculate_stock_diagnostics(rows)
    concentration = contribution_concentration(stocks)
    leave_one_out = leave_one_stock_out_returns(rows)
    time_blocks = returns_by_time_block(rows)
    assert metrics.observations == 3  # includes the explicitly costed liquidation
    assert metrics.supported_stocks == 2
    assert metrics.total_transaction_cost_usd > 0
    assert {row.symbol for row in stocks} == {"AAA", "BBB"}
    assert concentration.top_stock in {"AAA", "BBB"}
    assert concentration.top_stock_share is not None
    assert 0.5 <= concentration.top_stock_share <= 1.0
    assert concentration.top_five_share == pytest.approx(1.0)
    assert set(leave_one_out) == {"AAA", "BBB"}
    assert set(time_blocks) == {"2026-01"}
    for row in rows:
        assert sum(contribution.net_return_contribution for contribution in row.contributions) == pytest.approx(
            row.net_return
        )
