from __future__ import annotations

from sentiment_benchmark.strategy_research.persistent_finbert import FirmSessionScore, build_persistent_targets


def _scores() -> tuple[FirmSessionScore, ...]:
    return (
        FirmSessionScore("2026-01-02", "A", 0.8, 1),
        FirmSessionScore("2026-01-02", "B", 0.7, 1),
        FirmSessionScore("2026-01-02", "C", -0.8, 1),
        FirmSessionScore("2026-01-02", "D", -0.7, 1),
    )


def test_persistent_targets_hold_for_exact_number_of_intervals() -> None:
    sessions = ("2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07")
    targets = build_persistent_targets(
        _scores(),
        sessions=sessions,
        symbols=("A", "B", "C", "D"),
        threshold=0.5,
        holding_sessions=3,
        minimum_names_per_side=2,
        gross_exposure=1.0,
    )
    assert [row.gross_exposure for row in targets] == [1.0, 1.0, 1.0, 0.0]
    assert targets[0].weights() == {"A": 0.25, "B": 0.25, "C": -0.25, "D": -0.25}


def test_new_weak_score_supersedes_and_flattens_existing_state() -> None:
    sessions = ("2026-01-02", "2026-01-05")
    scores = _scores() + (FirmSessionScore("2026-01-05", "A", 0.1, 1),)
    targets = build_persistent_targets(
        scores,
        sessions=sessions,
        symbols=("A", "B", "C", "D"),
        threshold=0.5,
        holding_sessions=3,
        minimum_names_per_side=2,
        gross_exposure=1.0,
    )
    assert targets[0].gross_exposure == 1.0
    assert targets[1].gross_exposure == 0.0


def test_strategy_requires_both_sides_and_is_dollar_neutral() -> None:
    sessions = ("2026-01-02",)
    targets = build_persistent_targets(
        _scores()[:3],
        sessions=sessions,
        symbols=("A", "B", "C", "D"),
        threshold=0.5,
        holding_sessions=1,
        minimum_names_per_side=2,
        gross_exposure=1.0,
    )
    assert targets[0].gross_exposure == 0.0
    full = build_persistent_targets(
        _scores(),
        sessions=sessions,
        symbols=("A", "B", "C", "D"),
        threshold=0.5,
        holding_sessions=1,
        minimum_names_per_side=2,
        gross_exposure=1.0,
    )[0]
    assert full.gross_exposure == 1.0
    assert full.net_exposure == 0.0
