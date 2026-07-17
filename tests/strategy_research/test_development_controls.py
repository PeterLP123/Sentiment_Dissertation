from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_file,
    sha256_text,
)
from sentiment_benchmark.strategy_research.development_controls import (
    DevelopmentControlError,
    build_control_exposures,
    select_control_scores,
    shuffle_scores_within_stock,
)
from sentiment_benchmark.strategy_research.development_tests import SessionSignal, aggregate_stock_session_scores
from sentiment_benchmark.strategy_research.run_validation import (
    CompletedRunSnapshot,
    CompletedRunValidationError,
    load_completed_run_snapshot,
    validate_adjusted_open_price_panel,
    validate_completed_stage_output,
)


def signal(session: str, symbol: str, score: float) -> SessionSignal:
    return SessionSignal(
        session=session,
        symbol=symbol,
        event_count=1,
        mean_score=score,
        positive_events=int(score > 0),
        neutral_events=int(score == 0),
        negative_events=int(score < 0),
    )


def test_non_refreshing_hold_ignores_same_direction_and_reverses_opposite() -> None:
    sessions = tuple(f"2026-01-{day:02d}" for day in range(1, 9))
    signals = (
        signal(sessions[0], "AAA", 1.0),
        signal(sessions[2], "AAA", 1.0),
        signal(sessions[4], "AAA", -1.0),
    )

    exposures = build_control_exposures(
        sessions,
        ("AAA",),
        signals,
        threshold=0.75,
        hold_sessions=4,
        control="sentiment_v2",
    )["AAA"]

    assert exposures == (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0)


def test_refresh_control_restarts_same_direction_clock() -> None:
    sessions = tuple(f"2026-01-{day:02d}" for day in range(1, 8))
    signals = (signal(sessions[0], "AAA", 1.0), signal(sessions[2], "AAA", 1.0))

    non_refreshing = build_control_exposures(
        sessions,
        ("AAA",),
        signals,
        threshold=0.75,
        hold_sessions=4,
        control="sentiment_v2",
    )["AAA"]
    refreshing = build_control_exposures(
        sessions,
        ("AAA",),
        signals,
        threshold=0.75,
        hold_sessions=4,
        control="refreshing_signal",
    )["AAA"]

    assert non_refreshing == (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert refreshing == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)


def test_inverted_and_news_timing_controls_ignore_or_reverse_direction() -> None:
    sessions = ("2026-01-01", "2026-01-02", "2026-01-03")
    signals = (signal(sessions[0], "AAA", 1.0), signal(sessions[1], "BBB", 0.0))

    inverted = build_control_exposures(
        sessions,
        ("AAA", "BBB"),
        signals,
        threshold=0.75,
        hold_sessions=2,
        control="inverted_sentiment",
    )
    timing = build_control_exposures(
        sessions,
        ("AAA", "BBB"),
        signals,
        threshold=0.75,
        hold_sessions=2,
        control="news_timing_long",
    )

    assert inverted["AAA"] == (-1.0, -1.0, 0.0)
    assert inverted["BBB"] == (0.0, 0.0, 0.0)
    assert timing["AAA"] == (1.0, 1.0, 0.0)
    assert timing["BBB"] == (0.0, 1.0, 1.0)


def test_shuffle_preserves_stock_score_multisets_and_session_event_counts() -> None:
    events = [
        {"event_id": "a1", "eligible_execution_session": "2026-01-01", "symbol": "AAA"},
        {"event_id": "a2", "eligible_execution_session": "2026-01-01", "symbol": "AAA"},
        {"event_id": "a3", "eligible_execution_session": "2026-01-02", "symbol": "AAA"},
        {"event_id": "b1", "eligible_execution_session": "2026-01-01", "symbol": "BBB"},
        {"event_id": "b2", "eligible_execution_session": "2026-01-02", "symbol": "BBB"},
        {"event_id": "b3", "eligible_execution_session": "2026-01-02", "symbol": "BBB"},
    ]
    scores = [
        {"event_id": "a1", "status": "success", "score": -1.0},
        {"event_id": "a2", "status": "success", "score": 0.0},
        {"event_id": "a3", "status": "success", "score": 1.0},
        {"event_id": "b1", "status": "success", "score": 1.0},
        {"event_id": "b2", "status": "success", "score": 1.0},
        {"event_id": "b3", "status": "invalid", "score": 0.0},
    ]

    first = shuffle_scores_within_stock(events, scores, ("2026-01-01", "2026-01-02"), seed=7)
    second = shuffle_scores_within_stock(events, scores, ("2026-01-01", "2026-01-02"), seed=7)

    assert first == second
    counts = {(row.session, row.symbol): row.event_count for row in first}
    assert counts == {
        ("2026-01-01", "AAA"): 2,
        ("2026-01-02", "AAA"): 1,
        ("2026-01-01", "BBB"): 1,
        ("2026-01-02", "BBB"): 1,
    }
    shuffled_multisets: dict[str, Counter[float]] = defaultdict(Counter)
    for row in first:
        shuffled_multisets[row.symbol].update(
            {
                -1.0: row.negative_events,
                0.0: row.neutral_events,
                1.0: row.positive_events,
            }
        )
    assert shuffled_multisets["AAA"] == Counter({-1.0: 1, 0.0: 1, 1.0: 1})
    assert shuffled_multisets["BBB"] == Counter({1.0: 2})


def test_eligible_only_aggregation_prevents_ineligible_zero_dilution() -> None:
    events = [
        {"event_id": "material", "eligible_execution_session": "2026-01-01", "symbol": "AAA"},
        {"event_id": "irrelevant", "eligible_execution_session": "2026-01-01", "symbol": "AAA"},
    ]
    scores = [
        {"event_id": "material", "status": "success", "score": 1.0, "eligible": True},
        {"event_id": "irrelevant", "status": "success", "score": 0.0, "eligible": False},
    ]

    all_scores = select_control_scores(scores, "all_scored_events")
    eligible_scores = select_control_scores(scores, "eligible_nonzero_events")
    all_signal = aggregate_stock_session_scores(events, all_scores, ("2026-01-01",))[0]
    eligible_signal = aggregate_stock_session_scores(events, eligible_scores, ("2026-01-01",))[0]

    assert all_signal.mean_score == 0.5
    assert all_signal.event_count == 2
    assert eligible_signal.mean_score == 1.0
    assert eligible_signal.event_count == 1


@pytest.mark.parametrize(
    "row, message",
    [
        ({"event_id": "missing-flag", "status": "success", "score": 0.0}, "eligibility flag"),
        (
            {"event_id": "bad-ineligible", "status": "success", "score": 1.0, "eligible": False},
            "explicit zero",
        ),
        ({"event_id": "bad-eligible", "status": "success", "score": 0.0, "eligible": True}, "sign-only"),
    ],
)
def test_eligible_only_aggregation_fails_closed(row: dict[str, object], message: str) -> None:
    with pytest.raises(DevelopmentControlError, match=message):
        select_control_scores([row], "eligible_nonzero_events")


@pytest.mark.parametrize(
    ("stage", "output_name"),
    (("events", "events"), ("tuning", "fold_definitions")),
)
def test_completed_upstream_stage_outputs_fail_closed_after_tampering(
    tmp_path: Path,
    stage: str,
    output_name: str,
) -> None:
    root = tmp_path / "source-run"
    output_path = (
        tmp_path / "derived" / root.name / "events" / "events.jsonl"
        if stage == "events"
        else root / "tuning" / "fold_definitions.json"
    )
    atomic_write_text(output_path, "frozen\n")
    snapshot = _completed_snapshot(root, outputs={output_path.as_posix(): {"sha256": sha256_file(output_path)}})
    atomic_write_json(
        root / "manifests" / f"{stage}.json",
        {
            "manifest_schema_version": 1,
            "pipeline_schema_version": 1,
            "status": "completed",
            "run_id": root.name,
            "run_identity_sha256": snapshot.report["run_identity_sha256"],
            "outputs": {output_name: {"sha256": sha256_file(output_path)}},
        },
    )
    suffix = f"{root.name}/events/events.jsonl" if stage == "events" else f"{root.name}/tuning/fold_definitions.json"
    validate_completed_stage_output(
        snapshot,
        stage=stage,
        output_name=output_name,
        output_path=output_path,
        root_output_suffix=suffix,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("changed\n")
    stage_manifest_path = root / "manifests" / f"{stage}.json"
    stage_manifest = {
        "manifest_schema_version": 1,
        "pipeline_schema_version": 1,
        "status": "completed",
        "run_id": root.name,
        "run_identity_sha256": snapshot.report["run_identity_sha256"],
        "outputs": {output_name: {"sha256": sha256_file(output_path)}},
    }
    atomic_write_json(stage_manifest_path, stage_manifest)

    with pytest.raises(CompletedRunValidationError, match=f"{stage} stage.*changed"):
        validate_completed_stage_output(
            snapshot,
            stage=stage,
            output_name=output_name,
            output_path=output_path,
            root_output_suffix=suffix,
        )


def test_adjusted_open_price_manifest_fails_closed_after_tampering(tmp_path: Path) -> None:
    root = tmp_path / "source-run"
    price_path = tmp_path / "prices.csv"
    manifest_path = tmp_path / "prices.manifest.json"
    atomic_write_text(price_path, "symbol,session_date,adjusted_open\nAAA,2026-01-02,100\n")
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "adjustment_supported": True,
            "execution_field": "adjusted_open",
            "files": {"price_panel": {"sha256": sha256_file(price_path)}},
        },
    )
    config = {"prices": {"manifest_path": manifest_path.as_posix()}}
    snapshot = _completed_snapshot(
        root,
        config=config,
        input_identities={
            "price_panel": sha256_file(price_path),
            "price_manifest": sha256_file(manifest_path),
        },
    )
    validate_adjusted_open_price_panel(snapshot, price_path)
    with price_path.open("a", encoding="utf-8") as handle:
        handle.write("BBB,2026-01-02,200\n")
    price_manifest = {
        "schema_version": 1,
        "status": "completed",
        "adjustment_supported": True,
        "execution_field": "adjusted_open",
        "files": {"price_panel": {"sha256": sha256_file(price_path)}},
    }
    atomic_write_json(manifest_path, price_manifest)

    with pytest.raises(CompletedRunValidationError, match="price panel.*changed"):
        validate_adjusted_open_price_panel(snapshot, price_path)


def _completed_snapshot(
    root: Path,
    *,
    config: dict[str, object] | None = None,
    input_identities: dict[str, str] | None = None,
    outputs: dict[str, dict[str, str]] | None = None,
) -> CompletedRunSnapshot:
    resolved_config = config or {"run": {"evaluation_start": "2026-02-01"}}
    config_hash = sha256_text(canonical_json(resolved_config))
    root_manifest_path = root / "manifest.json"
    atomic_write_json(
        root_manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": root.name,
            "run_identity_sha256": "source-identity",
            "config": resolved_config,
            "config_sha256": config_hash,
            "input_identities": input_identities or {},
            "outputs": outputs or {},
        },
    )
    atomic_write_json(
        root / "manifests" / "report.json",
        {
            "manifest_schema_version": 1,
            "pipeline_schema_version": 1,
            "status": "completed",
            "run_id": root.name,
            "run_identity_sha256": "source-identity",
            "config": resolved_config,
            "config_sha256": config_hash,
            "outputs": {"manifest": {"sha256": sha256_file(root_manifest_path)}},
        },
    )
    return load_completed_run_snapshot(root)
