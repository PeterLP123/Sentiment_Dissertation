from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.strategy_research import receipt_pnl
from sentiment_benchmark.strategy_research.receipt_pnl import (
    ReceiptPnlError,
    align_pnl_to_receipt_dates,
    build_receipt_pnl_report,
    calculate_receipt_pnl_performance,
)


def _row(
    scorer: str,
    session: str,
    next_session: str | None,
    *,
    start_nav: float = 1_000.0,
    gross_return: float = 0.0,
    transaction_cost: float = 0.0,
    active_names: int = 0,
    final_liquidation: bool = False,
) -> dict[str, object]:
    net_return = gross_return - transaction_cost
    return {
        "scorer": scorer,
        "session": session,
        "next_session": next_session,
        "start_nav_usd": start_nav,
        "gross_return": gross_return,
        "transaction_cost": transaction_cost,
        "net_return": net_return,
        "end_nav_usd": start_nav * (1 + net_return),
        "active_names": active_names,
        "final_liquidation": final_liquidation,
    }


def test_gross_pnl_moves_to_exit_date_and_zeros_are_filled() -> None:
    rows = [
        _row("finbert", "2026-05-01", "2026-05-04"),
        _row(
            "finbert",
            "2026-05-04",
            "2026-05-05",
            gross_return=0.10,
            transaction_cost=0.001,
            active_names=1,
        ),
        _row(
            "finbert",
            "2026-05-05",
            None,
            transaction_cost=0.001,
            final_liquidation=True,
        ),
    ]

    aligned = align_pnl_to_receipt_dates(rows, scorers=("finbert",))
    by_date = {row.date: row for row in aligned}

    assert by_date["2026-05-01"].zero_filled
    assert by_date["2026-05-04"].gross_pnl_received_usd == 0
    assert by_date["2026-05-04"].zero_filled
    assert by_date["2026-05-05"].gross_pnl_received_usd == pytest.approx(100.0)
    assert by_date["2026-05-05"].active_trade_exits == 1
    assert by_date["2026-05-05"].cumulative_gross_pnl_usd == pytest.approx(100.0)


def test_common_spine_zero_fills_models_and_sharpe_includes_zero_days() -> None:
    rows = [
        _row("finbert", "2026-01-02", "2026-01-05", gross_return=0.01, active_names=2),
        _row("finbert", "2026-01-05", None, final_liquidation=True),
        _row("vader", "2026-01-02", "2026-01-05"),
        _row("vader", "2026-01-05", None, final_liquidation=True),
    ]

    aligned = align_pnl_to_receipt_dates(rows, scorers=("finbert", "vader"))
    assert [(row.date, row.gross_pnl_received_usd) for row in aligned if row.scorer == "finbert"] == [
        ("2026-01-02", 0.0),
        ("2026-01-05", 10.0),
    ]
    assert all(row.zero_filled for row in aligned if row.scorer == "vader")

    performance = {row.scorer: row for row in calculate_receipt_pnl_performance(aligned)}
    assert performance["finbert"].receipt_pnl_annualized_sharpe == pytest.approx(math.sqrt(252) * 5 / math.sqrt(50))
    assert performance["finbert"].zero_receipt_days == 1
    assert performance["vader"].receipt_pnl_annualized_sharpe is None


def test_alignment_rejects_non_reconciling_source_ledger() -> None:
    row = _row("finbert", "2026-01-02", "2026-01-05", gross_return=0.01)
    row["net_return"] = 0.02

    with pytest.raises(ReceiptPnlError, match="do not reconcile"):
        align_pnl_to_receipt_dates([row], scorers=("finbert",))


def test_report_verifies_source_hash_and_writes_gross_receipt_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "daily_pnl.jsonl"
    rows = [
        _row("finbert", "2026-01-02", "2026-01-05", gross_return=0.01, transaction_cost=0.001, active_names=2),
        _row("finbert", "2026-01-05", None, transaction_cost=0.001, final_liquidation=True),
    ]
    source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "synthetic-finbert",
                "outputs": {"daily_pnl": {"sha256": sha256_file(source)}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        receipt_pnl,
        "_render_figure",
        lambda _rows, _scorers, path, _source_run_id: path.write_bytes(b"synthetic png"),
    )

    result = build_receipt_pnl_report(source, manifest, tmp_path / "output", scorers=("finbert",))

    matrix = result.portfolio_matrix_path.read_text(encoding="utf-8")
    assert "2026-01-02,0.000000000000" in matrix
    assert "2026-01-05,10.000000000000" in matrix
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["status"] == "completed"

    source.write_text(source.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ReceiptPnlError, match="SHA-256"):
        build_receipt_pnl_report(source, manifest, tmp_path / "other-output", scorers=("finbert",))
