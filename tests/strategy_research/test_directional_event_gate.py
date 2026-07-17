from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sentiment_benchmark.artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
)
from sentiment_benchmark.strategy_research.directional_event_gate import (
    BootstrapSpread,
    DirectionalEventGateError,
    DirectionalEventGateSpec,
    FoldSpreadSummary,
    HorizonObservation,
    HorizonSpreadSummary,
    SelectedSessionEvent,
    _draw_contiguous_session_blocks,
    _optional_session_block_bootstrap_spread,
    calculate_horizon_observations,
    evaluate_directional_gate,
    run_directional_event_gate,
    select_strongest_session_events,
    session_block_bootstrap_spread,
    summarize_validation_folds,
)


def _event(event_id: str, symbol: str, session: str, minute: int) -> dict[str, object]:
    headline = "LICENSED_SENTINEL must not reach aggregate outputs"
    body = f"Synthetic body for {event_id}."
    return {
        "event_id": event_id,
        "story_family_id": f"family-{event_id}",
        "revision_id": f"revision-{event_id}",
        "symbol": symbol,
        "target_company_name": f"{symbol} Corp",
        "version_created_utc": f"{session}T13:{minute:02d}:00+00:00",
        "eligible_execution_session": session,
        "available_at_utc": f"{session}T14:{minute:02d}:00+00:00",
        "source": "lseg",
        "headline": headline,
        "lead_or_body": body,
        "text_sha256": sha256_text(f"{headline}\n\n{body}"),
        "target_relevance": "include",
        "event_session": session,
        "exclusion_reason": None,
        "input_manifest_hash": "source-manifest",
    }


def _score(
    event_id: str,
    direction: str,
    materiality: str,
    novelty: str,
    *,
    eligible: bool = True,
) -> dict[str, object]:
    sign = -1.0 if "negative" in direction else 1.0 if "positive" in direction else 0.0
    return {
        "event_id": event_id,
        "status": "success",
        "score": sign if eligible else 0.0,
        "eligible": eligible,
        "model_direction_severity": direction,
        "model_materiality": materiality,
        "model_novelty": novelty,
    }


def _selected(session: str, symbol: str, direction: int) -> SelectedSessionEvent:
    label = "positive" if direction > 0 else "negative"
    return SelectedSessionEvent(
        session=session,
        symbol=symbol,
        event_id=f"{symbol}-{session}",
        direction=direction,
        direction_severity=label,
        materiality="high",
        novelty="high",
        eligible_event_count=1,
    )


def _observation(
    session: str,
    symbol: str,
    direction: int,
    adjusted_return: float,
    *,
    horizon: int = 1,
) -> HorizonObservation:
    return HorizonObservation(
        horizon_sessions=horizon,
        start_session=session,
        end_session=session,
        symbol=symbol,
        direction=direction,
        raw_return=adjusted_return,
        market_return=0.0,
        market_adjusted_return=adjusted_return,
    )


def test_selects_strongest_event_lexicographically_and_excludes_conflicting_top_ties() -> None:
    session = "2026-01-05"
    events = [
        _event("a-novel-severe", "AAA", session, 1),
        _event("a-material", "AAA", session, 2),
        _event("b-severe", "BBB", session, 3),
        _event("b-novel", "BBB", session, 4),
        _event("c-positive", "CCC", session, 5),
        _event("c-very-negative", "CCC", session, 6),
        _event("d-positive", "DDD", session, 7),
        _event("d-negative", "DDD", session, 8),
        _event("e-later", "EEE", session, 10),
        _event("e-earlier", "EEE", session, 9),
        _event("f-ineligible", "FFF", session, 11),
    ]
    scores = [
        _score("a-novel-severe", "very_positive", "moderate", "very_high"),
        _score("a-material", "positive", "high", "moderate"),
        _score("b-severe", "very_positive", "high", "moderate"),
        _score("b-novel", "positive", "high", "high"),
        _score("c-positive", "positive", "high", "high"),
        _score("c-very-negative", "very_negative", "high", "high"),
        _score("d-positive", "positive", "high", "high"),
        _score("d-negative", "negative", "high", "high"),
        _score("e-later", "positive", "high", "high"),
        _score("e-earlier", "positive", "high", "high"),
        _score("f-ineligible", "neutral", "none", "none", eligible=False),
    ]

    first = select_strongest_session_events(events, scores, (session,))
    second = select_strongest_session_events(tuple(reversed(events)), tuple(reversed(scores)), (session,))

    assert first == second
    assert [(row.symbol, row.event_id, row.direction) for row in first.selected_events] == [
        ("AAA", "a-material", 1),
        ("BBB", "b-novel", 1),
        ("CCC", "c-very-negative", -1),
        ("EEE", "e-earlier", 1),
    ]
    assert first.input_events == 11
    assert first.eligible_events == 10
    assert first.ineligible_events == 1
    assert first.conflicting_tie_stock_sessions == 1
    assert first.selected_events[-1].eligible_event_count == 2


def test_selection_rejects_ineligible_nonzero_scores() -> None:
    session = "2026-01-05"
    event = _event("bad", "AAA", session, 1)
    score = _score("bad", "positive", "high", "high", eligible=False)
    score["score"] = 1.0

    with pytest.raises(DirectionalEventGateError, match="ineligible.*zero"):
        select_strongest_session_events((event,), (score,), (session,))


def test_horizon_returns_are_market_adjusted_and_stop_before_evaluation() -> None:
    sessions = (
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
    )
    symbols = ("AAA", "BBB")
    prices = {
        **{("AAA", session): 100.0 * 1.01**index for index, session in enumerate(sessions)},
        **{("BBB", session): 100.0 for session in sessions},
    }
    selected = (
        _selected(sessions[0], "AAA", 1),
        _selected(sessions[9], "BBB", -1),
    )

    observations = calculate_horizon_observations(
        selected,
        prices,
        sessions,
        symbols,
        sessions[10],
        (1, 2, 3, 5, 10),
    )

    assert [row.horizon_sessions for row in observations] == [1, 2, 3, 5]
    assert all(row.end_session < sessions[10] for row in observations)
    assert observations[0].raw_return == pytest.approx(0.01)
    assert observations[0].market_return == pytest.approx(0.005)
    assert observations[0].market_adjusted_return == pytest.approx(0.005)


def test_fold_summaries_require_start_and_endpoint_in_one_validation_fold() -> None:
    validation_sessions = (
        ("2026-01-05", "2026-01-06"),
        ("2026-01-07", "2026-01-08"),
        ("2026-01-09", "2026-01-12"),
    )
    observations = [
        _observation("2026-01-02", "TRAIN", 1, -99.0),
        replace(
            _observation("2026-01-06", "CROSS_FOLD", 1, 99.0),
            end_session="2026-01-07",
        ),
    ]
    for fold_index, sessions in enumerate(validation_sessions):
        for session_index, session in enumerate(sessions):
            observations.extend(
                (
                    _observation(session, f"P{fold_index}{session_index}", 1, 0.02),
                    _observation(session, f"N{fold_index}{session_index}", -1, -0.01),
                )
            )
    folds = [
        {
            "fold_index": index,
            "training_sessions": ["2026-01-02"],
            "validation_sessions": list(sessions),
        }
        for index, sessions in enumerate(validation_sessions)
    ]

    summaries = summarize_validation_folds(observations, folds, (1,))

    assert len(summaries) == 3
    assert [row.fold_index for row in summaries] == [0, 1, 2]
    assert all(row.positive_observations == row.negative_observations == 2 for row in summaries)
    assert all(row.spread == pytest.approx(0.03) for row in summaries)


def test_session_block_bootstrap_is_deterministic_and_keeps_session_clusters() -> None:
    positive = (0.03, 0.04, -0.01, 0.02, 0.05, 0.01)
    negative = (-0.01, -0.02, -0.03, 0.00, -0.01, -0.02)
    observations: list[HorizonObservation] = []
    for index, (positive_return, negative_return) in enumerate(zip(positive, negative, strict=True)):
        session = f"2026-01-{index + 2:02d}"
        observations.extend(
            (
                _observation(session, f"P{index}", 1, positive_return),
                _observation(session, f"N{index}", -1, negative_return),
            )
        )

    first = session_block_bootstrap_spread(
        observations,
        block_length=2,
        replications=250,
        seed=20_260_715,
        alpha=0.01,
    )
    second = session_block_bootstrap_spread(
        tuple(reversed(observations)),
        block_length=2,
        replications=250,
        seed=20_260_715,
        alpha=0.01,
    )

    assert first == second
    assert first.observed_spread == pytest.approx(sum(positive) / 6 - sum(negative) / 6)
    assert first.block_length == 2
    assert first.replications == 250
    assert first.seed == 20_260_715
    assert 0.0 < first.one_sided_p_value <= 1.0
    assert first.ci_low <= first.observed_spread <= first.ci_high


def test_contiguous_bootstrap_blocks_never_wrap_the_sample_boundary() -> None:
    sessions = tuple(f"S{index}" for index in range(6))
    sampled = _draw_contiguous_session_blocks(
        sessions,
        block_length=3,
        block_count=100,
        generator=np.random.default_rng(7),
    )

    for offset in range(0, len(sampled), 3):
        block = sampled[offset : offset + 3]
        ordinals = [sessions.index(session) for session in block]
        assert ordinals == list(range(ordinals[0], ordinals[0] + 3))


def test_sparse_two_sided_horizon_with_no_valid_bootstrap_draw_fails_without_aborting() -> None:
    sessions = tuple(f"S{index:02d}" for index in range(10))
    observations = (
        _observation(sessions[0], "POS", 1, 0.02),
        _observation(sessions[-1], "NEG", -1, -0.01),
    )

    bootstrap = _optional_session_block_bootstrap_spread(
        observations,
        block_length=3,
        replications=1,
        seed=0,
        alpha=0.01,
        session_order=sessions,
    )
    summary = _horizon_summary(1, positive=1, negative=1)
    acceptance, selected = evaluate_directional_gate(
        (summary,),
        tuple(_fold_summary(1, fold) for fold in range(3)),
        (),
        DirectionalEventGateSpec(
            horizons=(1,),
            bootstrap_replications=1,
            block_length=3,
            minimum_observations_per_direction=1,
            minimum_observations_per_direction_per_fold=1,
            minimum_supported_stocks=1,
        ),
    )

    assert bootstrap is None
    assert selected is None
    assert "missing_bootstrap_result" in acceptance[0].rejection_reasons


def _horizon_summary(horizon: int, *, positive: int = 60, negative: int = 60) -> HorizonSpreadSummary:
    return HorizonSpreadSummary(
        horizon_sessions=horizon,
        positive_observations=positive,
        negative_observations=negative,
        supported_stocks=30,
        positive_mean_market_adjusted_return=0.02,
        negative_mean_market_adjusted_return=-0.01,
        spread=0.03,
    )


def _fold_summary(horizon: int, fold: int, spread: float = 0.03) -> FoldSpreadSummary:
    return FoldSpreadSummary(
        horizon_sessions=horizon,
        fold_index=fold,
        positive_observations=20,
        negative_observations=20,
        positive_mean_market_adjusted_return=0.02,
        negative_mean_market_adjusted_return=0.02 - spread,
        spread=spread,
    )


def _bootstrap(horizon: int, p_value: float = 0.01, ci_low: float = 0.005) -> BootstrapSpread:
    return BootstrapSpread(
        horizon_sessions=horizon,
        observed_spread=0.03,
        ci_low=ci_low,
        ci_high=0.05,
        one_sided_p_value=p_value,
        block_length=5,
        replications=2_000,
        seed=20_260_715,
    )


def test_gate_requires_bonferroni_p_value_and_three_positive_folds() -> None:
    spec = DirectionalEventGateSpec(
        minimum_observations_per_direction=50,
        minimum_supported_stocks=25,
        required_positive_folds=3,
        maximum_one_sided_p_value=0.01,
    )
    horizons = [_horizon_summary(horizon) for horizon in (1, 2, 3, 5, 10)]
    horizons[3] = replace(horizons[3], positive_observations=49)
    folds = [_fold_summary(horizon, fold) for horizon in (1, 2, 3, 5, 10) for fold in range(3)]
    folds = [replace(row, spread=-0.001) if row.horizon_sessions == 2 and row.fold_index == 2 else row for row in folds]
    bootstraps = [_bootstrap(horizon, 0.02 if horizon == 3 else 0.01) for horizon in (1, 2, 3, 5, 10)]

    acceptance, selected_horizon = evaluate_directional_gate(horizons, folds, bootstraps, spec)

    by_horizon = {row.horizon_sessions: row for row in acceptance}
    assert selected_horizon == 1
    assert by_horizon[1].passed
    assert not by_horizon[2].passed
    assert any("fold" in reason for reason in by_horizon[2].rejection_reasons)
    assert not by_horizon[3].passed
    assert any("p" in reason for reason in by_horizon[3].rejection_reasons)
    assert not by_horizon[5].passed
    assert any("observation" in reason for reason in by_horizon[5].rejection_reasons)
    assert by_horizon[10].passed


def test_gate_rejects_nonpositive_familywise_confidence_bound() -> None:
    spec = DirectionalEventGateSpec(
        horizons=(1,),
        minimum_observations_per_direction=50,
        minimum_supported_stocks=25,
        required_positive_folds=3,
        maximum_one_sided_p_value=0.01,
    )
    horizon = _horizon_summary(1)
    folds = [_fold_summary(1, fold) for fold in range(3)]

    acceptance, selected_horizon = evaluate_directional_gate(
        [horizon],
        folds,
        [_bootstrap(1, p_value=0.001, ci_low=-0.001)],
        spec,
    )

    assert selected_horizon is None
    assert not acceptance[0].passed
    assert any("lower_bound" in reason for reason in acceptance[0].rejection_reasons)


def test_run_directional_event_gate_is_aggregate_only_reusable_and_immutable(tmp_path: Path) -> None:
    sessions = (
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
    )
    validation_folds = (sessions[0:3], sessions[3:6], sessions[6:9])
    price_path = tmp_path / "prices.csv"
    price_rows = ["symbol,session_date,adjusted_open"]
    for symbol, direction in (("AAA", 1), ("BBB", -1)):
        price_rows.extend(f"{symbol},{session},{100.0 + direction * 2.0 * index:.2f}" for index, session in enumerate(sessions))
    atomic_write_text(price_path, "\n".join(price_rows) + "\n")

    run_dir = tmp_path / "results" / "source-run"
    derived_root = tmp_path / "derived"
    events_path = derived_root / run_dir.name / "events" / "events.jsonl"
    events: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    for session_index in (0, 1, 3, 4, 6, 7):
        session = sessions[session_index]
        for symbol, label, score in (("AAA", "positive", 1.0), ("BBB", "negative", -1.0)):
            event_id = f"event-{symbol}-{session_index}"
            events.append(_event(event_id, symbol, session, 0))
            scores.append(
                {
                    "event_id": event_id,
                    "status": "success",
                    "score": score,
                    "eligible": True,
                    "model_direction_severity": label,
                    "model_materiality": "high",
                    "model_novelty": "high",
                }
            )
    atomic_write_jsonl(events_path, events)
    price_manifest_path = tmp_path / "prices.manifest.json"
    atomic_write_json(
        price_manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "adjustment_supported": True,
            "execution_field": "adjusted_open",
            "files": {"price_panel": {"sha256": sha256_file(price_path)}},
        },
    )
    atomic_write_json(
        run_dir / "manifests" / "report.json",
        {
            "manifest_schema_version": 1,
            "pipeline_schema_version": 1,
            "status": "completed",
            "run_id": run_dir.name,
            "run_identity_sha256": "source-identity",
            "config": {
                "run": {"evaluation_start": sessions[-1]},
                "outputs": {"derived_root": derived_root.as_posix()},
                "prices": {
                    "panel_path": price_path.as_posix(),
                    "manifest_path": price_manifest_path.as_posix(),
                },
            },
        },
    )
    atomic_write_json(
        run_dir / "manifests" / "events.json",
        {
            "manifest_schema_version": 1,
            "pipeline_schema_version": 1,
            "status": "completed",
            "run_id": run_dir.name,
            "run_identity_sha256": "source-identity",
            "outputs": {"events": {"sha256": sha256_file(events_path)}},
        },
    )
    folds_path = run_dir / "tuning" / "fold_definitions.json"
    atomic_write_json(
        folds_path,
        {
            "folds": [
                {
                    "fold_index": index,
                    "training_sessions": list(sessions[: 3 * index]),
                    "validation_sessions": list(validation),
                }
                for index, validation in enumerate(validation_folds)
            ]
        },
    )
    atomic_write_json(
        run_dir / "manifests" / "tuning.json",
        {
            "manifest_schema_version": 1,
            "pipeline_schema_version": 1,
            "status": "completed",
            "run_id": run_dir.name,
            "run_identity_sha256": "source-identity",
            "outputs": {"fold_definitions": {"sha256": sha256_file(folds_path)}},
        },
    )
    report_path = run_dir / "manifests" / "report.json"
    report = read_json(report_path)
    config_hash = sha256_text(canonical_json(report["config"]))
    root_manifest = {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_dir.name,
        "run_identity_sha256": report["run_identity_sha256"],
        "config": report["config"],
        "config_sha256": config_hash,
        "input_identities": {
            "price_panel": sha256_file(price_path),
            "price_manifest": sha256_file(price_manifest_path),
        },
        "outputs": {
            events_path.as_posix(): {"sha256": sha256_file(events_path)},
            folds_path.as_posix(): {"sha256": sha256_file(folds_path)},
        },
    }
    root_manifest_path = run_dir / "manifest.json"
    atomic_write_json(root_manifest_path, root_manifest)
    report["config_sha256"] = config_hash
    report["outputs"] = {"manifest": {"sha256": sha256_file(root_manifest_path)}}
    atomic_write_json(report_path, report)

    universe_dir = tmp_path / "universe"
    items_path = universe_dir / "audit_items.csv"
    atomic_write_text(
        items_path,
        "audit_id,event_id\n" + "".join(f"audit-{index},{event['event_id']}\n" for index, event in enumerate(events)),
    )
    assumption_path = universe_dir / "model_only_assumption.md"
    atomic_write_text(assumption_path, "Synthetic model-only waiver assumption.\n")
    universe_identity = {
        "contract": "model_only_filtered_stock_strategy_v1",
        "source_run_id": run_dir.name,
        "source_run_identity": "source-identity",
        "evaluation_start": sessions[-1],
        "events_sha256": sha256_file(events_path),
    }
    universe_identity_hash = sha256_text(canonical_json(universe_identity))
    atomic_write_json(
        universe_dir / "sample_manifest.json",
        {
            "schema_version": 1,
            "status": "completed",
            "identity_sha256": universe_identity_hash,
            "identity": universe_identity,
            "outputs": {
                items_path.name: {"sha256": sha256_file(items_path)},
                assumption_path.name: {"sha256": sha256_file(assumption_path)},
            },
        },
    )
    strategy_dir = tmp_path / "filtered-strategy"
    scores_path = strategy_dir / "scores.jsonl"
    atomic_write_jsonl(scores_path, scores)
    strategy_identity = {
        "contract": "model_only_filtered_stock_strategy_v1",
        "universe_identity_sha256": universe_identity_hash,
    }
    atomic_write_json(
        strategy_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "completed",
            "identity_sha256": sha256_text(canonical_json(strategy_identity)),
            "identity": strategy_identity,
            "outputs": {scores_path.name: {"sha256": sha256_file(scores_path)}},
        },
    )

    output_dir = tmp_path / "gate-output"
    spec = DirectionalEventGateSpec(
        horizons=(1, 9),
        bootstrap_replications=50,
        block_length=3,
        familywise_alpha=0.05,
        maximum_one_sided_p_value=0.025,
        minimum_observations_per_direction=1,
        minimum_observations_per_direction_per_fold=1,
        minimum_supported_stocks=2,
        required_positive_folds=3,
    )
    first = run_directional_event_gate(
        run_dir,
        universe_dir,
        strategy_dir,
        output_root=output_dir,
        spec=spec,
    )
    frozen_hashes = {path.relative_to(output_dir).as_posix(): sha256_file(path) for path in output_dir.rglob("*") if path.is_file()}
    second = run_directional_event_gate(
        run_dir,
        universe_dir,
        strategy_dir,
        output_root=output_dir,
        spec=spec,
    )

    assert first.passed
    assert first.selected_horizon == 1
    assert not first.acceptance[1].passed
    assert "missing_bootstrap_result" in first.acceptance[1].rejection_reasons
    assert "| 9 | 0 | 0 | 0 | n/a | n/a | n/a |" in first.report_path.read_text(encoding="utf-8")
    assert not first.reused
    assert second.reused
    assert second.experiment_id == first.experiment_id
    assert second.acceptance == first.acceptance
    assert read_json(first.manifest_path)["status"] == "completed"
    assert frozen_hashes == {path.relative_to(output_dir).as_posix(): sha256_file(path) for path in output_dir.rglob("*") if path.is_file()}
    aggregate_text = "\n".join(path.read_text(encoding="utf-8") for path in output_dir.rglob("*") if path.is_file())
    assert "event_id" not in aggregate_text
    assert "LICENSED_SENTINEL" not in aggregate_text
    assert set(frozen_hashes) == {
        "acceptance_checks.csv",
        "bootstrap.csv",
        "figures/direction_spread.svg",
        "fold_summary.csv",
        "horizon_summary.csv",
        "manifest.json",
        "report.md",
        "subgroup_summary.csv",
    }

    original_manifest = read_json(first.manifest_path)
    identity_tamper = read_json(first.manifest_path)
    identity_tamper["identity"]["warning"] = "changed without rehashing"
    atomic_write_json(first.manifest_path, identity_tamper)
    with pytest.raises(DirectionalEventGateError, match="identity mismatch"):
        run_directional_event_gate(
            run_dir,
            universe_dir,
            strategy_dir,
            output_root=output_dir,
            spec=spec,
        )
    atomic_write_json(first.manifest_path, original_manifest)

    tampered_manifest = dict(original_manifest)
    tampered_manifest["passed"] = "False"
    tampered_manifest["selected_horizon"] = True
    atomic_write_json(first.manifest_path, tampered_manifest)
    with pytest.raises(DirectionalEventGateError, match="decision fields disagree"):
        run_directional_event_gate(
            run_dir,
            universe_dir,
            strategy_dir,
            output_root=output_dir,
            spec=spec,
        )
    atomic_write_json(first.manifest_path, original_manifest)

    frozen_assumption = assumption_path.read_text(encoding="utf-8")
    atomic_write_text(assumption_path, frozen_assumption + "changed\n")
    with pytest.raises(DirectionalEventGateError, match="scoring input changed"):
        run_directional_event_gate(
            run_dir,
            universe_dir,
            strategy_dir,
            output_root=output_dir,
            spec=spec,
        )
    atomic_write_text(assumption_path, frozen_assumption)

    unexpected = output_dir / "unexpected.txt"
    unexpected.write_text("not manifested\n", encoding="utf-8")
    with pytest.raises(DirectionalEventGateError, match="unexpected or missing files"):
        run_directional_event_gate(
            run_dir,
            universe_dir,
            strategy_dir,
            output_root=output_dir,
            spec=spec,
        )
    unexpected.unlink()

    frozen_folds = folds_path.read_text(encoding="utf-8")
    atomic_write_text(folds_path, frozen_folds + "\n")
    with pytest.raises(DirectionalEventGateError, match="tuning stage.*changed"):
        run_directional_event_gate(
            run_dir,
            universe_dir,
            strategy_dir,
            output_root=output_dir,
            spec=spec,
        )
    atomic_write_text(folds_path, frozen_folds)

    with (output_dir / "horizon_summary.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(DirectionalEventGateError, match="output changed"):
        run_directional_event_gate(
            run_dir,
            universe_dir,
            strategy_dir,
            output_root=output_dir,
            spec=spec,
        )
