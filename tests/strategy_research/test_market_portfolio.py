from __future__ import annotations

import pytest

from sentiment_benchmark.strategy_research.market import (
    AdjustedOpen,
    OpenToOpenReturn,
    PriceDataError,
    calculate_lagged_volatility,
    calculate_open_to_open_returns,
    load_adjusted_opens_csv,
    validate_adjusted_opens,
    validate_exchange_sessions,
)
from sentiment_benchmark.strategy_research.portfolio import PortfolioConstraints, project_target_weights


def test_adjusted_open_validation_and_returns() -> None:
    prices = [
        AdjustedOpen("AAA", "2026-01-05", 100),
        AdjustedOpen("AAA", "2026-01-06", 110),
        AdjustedOpen("AAA", "2026-01-07", 99),
    ]
    returns = calculate_open_to_open_returns(prices)
    assert [(row.session, row.next_session) for row in returns] == [
        ("2026-01-05", "2026-01-06"),
        ("2026-01-06", "2026-01-07"),
    ]
    assert returns[0].value == pytest.approx(0.1)
    assert returns[1].value == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "rows, match",
    [
        (
            [AdjustedOpen("AAA", "2026-01-05", 1), AdjustedOpen("AAA", "2026-01-05", 1)],
            "duplicate",
        ),
        (
            [AdjustedOpen("AAA", "2026-01-06", 1), AdjustedOpen("AAA", "2026-01-05", 1)],
            "non-monotonic",
        ),
        ([AdjustedOpen("AAA", "2026-01-05", 1, adjustment_supported=False)], "unsupported"),
    ],
)
def test_price_panel_fails_closed(rows: list[AdjustedOpen], match: str) -> None:
    with pytest.raises(PriceDataError, match=match):
        validate_adjusted_opens(rows)


def test_csv_loader_requires_the_explicit_adjusted_open_field(tmp_path) -> None:
    panel = tmp_path / "prices.csv"
    panel.write_text("symbol,session_date,adjusted_open\nAAA,2026-01-05,100\nAAA,2026-01-06,101\n", encoding="utf-8")
    loaded = load_adjusted_opens_csv(panel)
    assert [row.adjusted_open for row in loaded] == [100, 101]

    generic = tmp_path / "generic.csv"
    generic.write_text("symbol,session_date,open\nAAA,2026-01-05,100\n", encoding="utf-8")
    with pytest.raises(PriceDataError, match="missing declared columns"):
        load_adjusted_opens_csv(generic)
    assert load_adjusted_opens_csv(generic, price_column="open")[0].adjusted_open == 100


def test_exchange_calendar_rejects_weekend_price_rows() -> None:
    validate_exchange_sessions([AdjustedOpen("AAA", "2026-01-05", 100)])
    with pytest.raises(PriceDataError, match="non-XNYS"):
        validate_exchange_sessions([AdjustedOpen("AAA", "2026-01-04", 100)])


def test_volatility_excludes_return_ending_at_decision_open() -> None:
    returns = [
        OpenToOpenReturn("AAA", "2026-01-01", "2026-01-02", 0.01),
        OpenToOpenReturn("AAA", "2026-01-02", "2026-01-03", -0.01),
        OpenToOpenReturn("AAA", "2026-01-03", "2026-01-04", 0.50),
    ]
    rows = calculate_lagged_volatility(
        returns,
        ["2026-01-04", "2026-01-05"],
        ["AAA", "MISSING"],
        window_sessions=3,
        minimum_sessions=2,
    )
    lookup = {(row.session, row.symbol): row for row in rows}
    at_four = lookup[("2026-01-04", "AAA")]
    at_five = lookup[("2026-01-05", "AAA")]
    assert at_four.observations == 2
    assert at_four.value == pytest.approx(0.0141421356)
    assert at_five.observations == 3
    assert at_five.value > at_four.value
    assert lookup[("2026-01-05", "MISSING")].value is None


def test_projection_enforces_one_sided_net_limit_without_inventing_shorts() -> None:
    actions = {f"S{index}": 1.0 for index in range(10)}
    volatilities = {symbol: 0.02 for symbol in actions}
    portfolio = project_target_weights("2026-01-05", actions, volatilities)
    weights = portfolio.weights()
    assert portfolio.net_exposure == pytest.approx(0.20)
    assert portfolio.gross_exposure == pytest.approx(0.20)
    assert all(0 <= weight <= 0.05 for weight in weights.values())


def test_projection_preserves_signs_and_all_limits_deterministically() -> None:
    actions = {"CCC": 0.2, "AAA": 1.0, "BBB": -1.0, "DDD": -0.5}
    volatilities = {symbol: 0.01 for symbol in actions}
    limits = PortfolioConstraints(gross_limit=0.4, net_limit=0.1, single_name_limit=0.15)
    first = project_target_weights("2026-01-05", actions, volatilities, limits)
    second = project_target_weights("2026-01-05", dict(reversed(list(actions.items()))), volatilities, limits)
    assert first == second
    weights = first.weights()
    assert sum(abs(value) for value in weights.values()) <= 0.4 + 1e-12
    assert abs(sum(weights.values())) <= 0.1 + 1e-12
    assert max(abs(value) for value in weights.values()) <= 0.15 + 1e-12
    assert all(weights[symbol] * action >= 0 for symbol, action in actions.items())


def test_missing_volatility_excludes_nonzero_signal_instead_of_filling() -> None:
    portfolio = project_target_weights("2026-01-05", {"AAA": 1.0, "BBB": 0.0}, {})
    by_symbol = {row.symbol: row for row in portfolio.positions}
    assert by_symbol["AAA"].target_weight == 0
    assert by_symbol["AAA"].exclusion_reason == "missing_lagged_volatility"
    assert by_symbol["BBB"].exclusion_reason is None
