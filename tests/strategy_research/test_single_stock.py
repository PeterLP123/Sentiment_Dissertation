from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentiment_benchmark.strategy_research.market import OpenToOpenReturn
from sentiment_benchmark.strategy_research.single_stock import (
    SingleStockExperimentError,
    SingleStockRule,
    evaluate_single_stock_actions,
    run_single_stock_experiment,
)


def _actions(symbol: str, values: tuple[float, ...]) -> list[dict[str, object]]:
    sessions = ("2026-01-05", "2026-01-06", "2026-01-07")
    return [{"session": session, "symbol": symbol, "action": action} for session, action in zip(sessions, values, strict=True)]


def _returns(symbol: str, values: tuple[float, ...]) -> list[OpenToOpenReturn]:
    sessions = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")
    return [
        OpenToOpenReturn(symbol, current, following, value)
        for current, following, value in zip(sessions[:-1], sessions[1:], values, strict=True)
    ]


def test_direct_action_account_preserves_carry_and_costs() -> None:
    rows, daily = evaluate_single_stock_actions(
        _actions("AAA", (0.5, -0.5, 0.0)),
        _returns("AAA", (0.10, 0.0, 0.0)),
        SingleStockRule(
            evaluation_start="2026-01-06",
            cost_bps_per_side=10,
            initial_nav_usd=100_000,
            bootstrap_replications=20,
            seed=7,
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    carried_weight = 0.5 * 1.1 / 1.05
    expected_first_turnover = abs(-0.5 - carried_weight)
    assert row.observations == 3  # two market intervals plus final liquidation
    assert daily[0]["turnover"] == pytest.approx(expected_first_turnover)
    assert daily[0]["transaction_cost"] == pytest.approx(0.001 * expected_first_turnover)
    assert daily[1]["turnover"] == pytest.approx(0.5)
    assert row.total_turnover == pytest.approx(expected_first_turnover + 0.5)
    assert row.cumulative_net_return < 0


def test_each_stock_result_is_invariant_to_other_stocks() -> None:
    rule = SingleStockRule(evaluation_start="2026-01-06", bootstrap_replications=20, seed=11)
    a_actions = _actions("AAA", (0.2, 0.4, -0.1))
    a_returns = _returns("AAA", (0.01, 0.02, -0.03))
    only_a, _ = evaluate_single_stock_actions(a_actions, a_returns, rule)
    with_b, _ = evaluate_single_stock_actions(
        a_actions + _actions("BBB", (-1.0, -1.0, 1.0)),
        a_returns + _returns("BBB", (-0.2, 0.3, 0.1)),
        rule,
    )

    assert with_b[0] == only_a[0]
    assert [row.symbol for row in with_b] == ["AAA", "BBB"]


def test_completed_run_materialization_is_immutable_and_reusable(tmp_path: Path) -> None:
    run_dir = tmp_path / "completed-run"
    (run_dir / "manifests").mkdir(parents=True)
    (run_dir / "state").mkdir()
    (run_dir / "tuning").mkdir()
    report_manifest = {
        "status": "completed",
        "run_id": "fixture-run",
        "run_identity_sha256": "fixture-identity",
        "config": {
            "run": {"evaluation_start": "2026-01-06"},
            "execution": {"cost_bps_per_side": 10},
            "prices": {"panel_path": (tmp_path / "prices.csv").as_posix()},
            "inference": {"block_length_sessions": 2, "replications": 20, "seed": 5},
        },
    }
    (run_dir / "manifests" / "report.json").write_text(json.dumps(report_manifest), encoding="utf-8")
    action_rows = _actions("AAA", (0.2, 0.4, -0.1)) + _actions("BBB", (-0.2, 0.0, 0.3))
    (run_dir / "state" / "actions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in action_rows),
        encoding="utf-8",
    )
    (run_dir / "tuning" / "selected_config.json").write_text('{"candidate":{}}\n', encoding="utf-8")
    (tmp_path / "prices.csv").write_text(
        "symbol,session_date,adjusted_open\n"
        "AAA,2026-01-05,100\nAAA,2026-01-06,101\nAAA,2026-01-07,103\nAAA,2026-01-08,100\n"
        "BBB,2026-01-05,50\nBBB,2026-01-06,49\nBBB,2026-01-07,48\nBBB,2026-01-08,52\n",
        encoding="utf-8",
    )

    first = run_single_stock_experiment(run_dir)
    second = run_single_stock_experiment(run_dir)

    assert not first.reused
    assert second.reused
    assert first.experiment_id == second.experiment_id
    assert first.summary_path.is_file()
    assert first.daily_path.is_file()
    assert first.report_path.is_file()
    assert first.return_figure_path.is_file()
    assert len(first.rows) == 2

    original_summary = first.summary_path.read_text(encoding="utf-8")
    first.summary_path.write_text(f"{original_summary}tampered\n", encoding="utf-8")
    with pytest.raises(SingleStockExperimentError, match="missing or changed"):
        run_single_stock_experiment(run_dir, output_root=first.output_dir)
    first.summary_path.write_text(original_summary, encoding="utf-8")

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest["identity_sha256"] = "wrong"
    first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SingleStockExperimentError, match="identity mismatch"):
        run_single_stock_experiment(run_dir, output_root=first.output_dir)
