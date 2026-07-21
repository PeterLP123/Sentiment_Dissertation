from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sentiment_benchmark.strategy_research.baseline.signals import BaselineScoreError, FirmSessionSignal
from sentiment_benchmark.strategy_research.baseline.v2.signals import (
    ArmSignal,
    build_arm_signals,
    build_v2_targets,
    load_vader_compound_scores,
)


def _write_compound(path: Path, rows: list[dict[str, object]]) -> Path:
    columns = ("headline_sha256", "baseline", "label", "compound", "status", "error")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(sha: str, label: str, compound: float) -> dict[str, object]:
    return {
        "headline_sha256": sha,
        "baseline": "vader_compound",
        "label": label,
        "compound": compound,
        "status": "success",
        "error": "",
    }


def test_compound_loader_accepts_only_threshold_consistent_labels(tmp_path: Path) -> None:
    path = _write_compound(
        tmp_path / "compound.csv",
        [
            _row("a" * 64, "positive", 0.6),
            _row("b" * 64, "negative", -0.05),
            _row("c" * 64, "neutral", 0.049),
        ],
    )
    scores = load_vader_compound_scores(path)
    assert scores["a" * 64].label == "positive"
    assert scores["b" * 64].compound == -0.05
    assert scores["c" * 64].label == "neutral"


def test_compound_loader_rejects_threshold_violations_and_duplicates(tmp_path: Path) -> None:
    with pytest.raises(BaselineScoreError, match="compound threshold convention"):
        load_vader_compound_scores(
            _write_compound(tmp_path / "bad_label.csv", [_row("a" * 64, "neutral", 0.6)])
        )
    with pytest.raises(BaselineScoreError, match="duplicate"):
        load_vader_compound_scores(
            _write_compound(
                tmp_path / "duplicate.csv",
                [_row("a" * 64, "positive", 0.6), _row("a" * 64, "positive", 0.6)],
            )
        )
    with pytest.raises(BaselineScoreError, match="no successful scores"):
        load_vader_compound_scores(
            _write_compound(
                tmp_path / "failed.csv",
                [{**_row("a" * 64, "", ""), "status": "error", "error": "boom"}],
            )
        )


def _signal(scorer: str, symbol: str, action: int, score: float, session: str = "2026-03-09") -> FirmSessionSignal:
    return FirmSessionSignal(
        scorer=scorer,
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


def test_agreement_arm_keeps_only_matching_nonzero_directions() -> None:
    signals = (
        _signal("finbert", "AAA", 1, 0.7),
        _signal("finbert", "BBB", -1, -0.6),
        _signal("finbert", "CCC", 1, 0.4),
        _signal("finbert", "DDD", 0, 0.0),
        _signal("vader_compound", "AAA", 1, 0.5),
        _signal("vader_compound", "BBB", -1, -0.4),
        _signal("vader_compound", "CCC", -1, -0.2),
        _signal("vader_compound", "DDD", 1, 0.3),
    )
    agreement = build_arm_signals(signals, "agreement")
    assert [(row.symbol, row.action) for row in agreement] == [("AAA", 1), ("BBB", -1)]
    finbert = build_arm_signals(signals, "finbert")
    assert [(row.symbol, row.action) for row in finbert] == [("AAA", 1), ("BBB", -1), ("CCC", 1)]
    assert [row.magnitude for row in finbert] == [0.7, 0.6, 0.4]


def test_two_name_minimum_forces_cash_when_a_leg_is_thin() -> None:
    signals = (
        ArmSignal("2026-03-09", "AAA", 1, 0.7),
        ArmSignal("2026-03-09", "CCC", 1, 0.4),
        ArmSignal("2026-03-09", "BBB", -1, 0.6),
    )
    result = build_v2_targets(
        signals,
        arm="finbert",
        sessions=("2026-03-09",),
        symbols=("AAA", "BBB", "CCC", "DDD"),
        gross_exposure=1.0,
        minimum_names_per_side=2,
        weighting="equal",
    )
    target = result.targets[0]
    assert target.gross_exposure == 0.0
    assert result.session_rows[0]["no_trade_reason"] == "minimum_names_not_met_on_both_sides"


def test_magnitude_weighting_allocates_sides_proportionally_and_stays_neutral() -> None:
    signals = (
        ArmSignal("2026-03-09", "AAA", 1, 0.6),
        ArmSignal("2026-03-09", "CCC", 1, 0.2),
        ArmSignal("2026-03-09", "BBB", -1, 0.3),
        ArmSignal("2026-03-09", "DDD", -1, 0.1),
    )
    result = build_v2_targets(
        signals,
        arm="finbert_magnitude",
        sessions=("2026-03-09",),
        symbols=("AAA", "BBB", "CCC", "DDD"),
        gross_exposure=1.0,
        minimum_names_per_side=2,
        weighting="magnitude",
    )
    weights = result.targets[0].weights()
    assert weights["AAA"] == pytest.approx(0.5 * 0.6 / 0.8)
    assert weights["CCC"] == pytest.approx(0.5 * 0.2 / 0.8)
    assert weights["BBB"] == pytest.approx(-0.5 * 0.3 / 0.4)
    assert weights["DDD"] == pytest.approx(-0.5 * 0.1 / 0.4)
    assert sum(weights.values()) == pytest.approx(0.0)
    assert sum(abs(value) for value in weights.values()) == pytest.approx(1.0)


def test_magnitude_weighting_falls_back_to_equal_on_zero_magnitude_side() -> None:
    signals = (
        ArmSignal("2026-03-09", "AAA", 1, 0.0),
        ArmSignal("2026-03-09", "CCC", 1, 0.0),
        ArmSignal("2026-03-09", "BBB", -1, 0.3),
        ArmSignal("2026-03-09", "DDD", -1, 0.1),
    )
    result = build_v2_targets(
        signals,
        arm="finbert_magnitude",
        sessions=("2026-03-09",),
        symbols=("AAA", "BBB", "CCC", "DDD"),
        gross_exposure=1.0,
        minimum_names_per_side=2,
        weighting="magnitude",
    )
    weights = result.targets[0].weights()
    assert weights["AAA"] == pytest.approx(0.25)
    assert weights["CCC"] == pytest.approx(0.25)
