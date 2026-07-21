from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import sha256_text
from sentiment_benchmark.headline_value import headline_norm_sha256
from sentiment_benchmark.strategy_research.baseline.config import (
    BaselineConfigurationError,
    load_baseline_config,
)
from sentiment_benchmark.strategy_research.baseline.signals import (
    BaselineScoreError,
    EventScreeningMetadata,
    FirmSessionSignal,
    HeadlineScore,
    build_equal_weight_targets,
    build_event_labels,
    load_headline_scores,
)
from sentiment_benchmark.strategy_research.schemas import StrategyEvent

BASE_CONFIG = Path("configs/strategy_research/baselines/lseg_midcap_finbert_vader_v1.toml")
SCORE_COLUMNS = (
    "headline_sha256",
    "matched_symbols",
    "explicit_target",
    "contextual",
    "market_price_technical",
    "baseline",
    "label",
    "p_positive",
    "p_negative",
    "p_neutral",
    "score",
    "status",
)


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "baseline.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _score_row(
    headline_sha256: str,
    scorer: str,
    *,
    label: str = "positive",
    score: str | None = None,
) -> dict[str, str]:
    probabilities = {
        "positive": ("0.8", "0.1", "0.1", "0.7"),
        "negative": ("0.1", "0.8", "0.1", "-0.7"),
        "neutral": ("0.1", "0.1", "0.8", "0.0"),
    }
    positive, negative, neutral, expected_score = probabilities[label]
    return {
        "headline_sha256": headline_sha256,
        "matched_symbols": "AAA",
        "explicit_target": "true",
        "contextual": "false",
        "market_price_technical": "false",
        "baseline": scorer,
        "label": label,
        "p_positive": positive,
        "p_negative": negative,
        "p_neutral": neutral,
        "score": score if score is not None else expected_score,
        "status": "success",
    }


def _write_scores(path: Path, rows: list[dict[str, str]], *, columns: tuple[str, ...] = SCORE_COLUMNS) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _event(event_id: str, headline: str, *, symbol: str = "AAA") -> StrategyEvent:
    body = f"Body for {event_id}."
    timestamp = datetime(2026, 1, 5, 13, tzinfo=UTC)
    return StrategyEvent(
        event_id=event_id,
        story_family_id=f"family-{event_id}",
        revision_id=f"revision-{event_id}",
        symbol=symbol,
        target_company_name=f"{symbol} Corp",
        version_created_utc=timestamp,
        available_at_utc=timestamp,
        source="lseg",
        headline=headline,
        lead_or_body=body,
        text_sha256=sha256_text(f"{headline}\n\n{body}"),
        target_relevance="include",
        event_session=date(2026, 1, 5),
        eligible_execution_session=date(2026, 1, 6),
        exclusion_reason=None,
        input_manifest_hash="manifest-hash",
    )


def _headline_score(
    headline: str,
    scorer: str,
    label: str,
    *,
    matched_symbols: tuple[str, ...] = ("AAA",),
    explicit_target: bool = True,
    market_price_technical: bool = False,
) -> HeadlineScore:
    probabilities = {
        "positive": (0.8, 0.1, 0.1),
        "negative": (0.1, 0.8, 0.1),
        "neutral": (0.1, 0.1, 0.8),
    }
    positive, negative, neutral = probabilities[label]
    return HeadlineScore(
        headline_sha256=headline_norm_sha256(headline),
        scorer=scorer,
        matched_symbols=matched_symbols,
        explicit_target=explicit_target,
        contextual=False,
        market_price_technical=market_price_technical,
        label=label,
        p_positive=positive,
        p_negative=negative,
        p_neutral=neutral,
        score=positive - negative,
    )


def _add_event_scores(
    scores: dict[tuple[str, str], HeadlineScore],
    event: StrategyEvent,
    labels: tuple[str, str] = ("neutral", "neutral"),
    **metadata: object,
) -> None:
    for scorer, label in zip(("finbert", "vader"), labels, strict=True):
        item = _headline_score(event.headline, scorer, label, **metadata)
        scores[(item.headline_sha256, scorer)] = item


def _signal(scorer: str, symbol: str, session: str, action: int) -> FirmSessionSignal:
    return FirmSessionSignal(
        scorer=scorer,
        symbol=symbol,
        session=session,
        event_count=1,
        positive_events=int(action > 0),
        negative_events=int(action < 0),
        neutral_events=int(action == 0),
        mean_hard_label=float(action),
        mean_continuous_score=0.7 * action,
        action=action,
    )


def test_config_rejects_unknown_keys_and_vader_compound_semantics(tmp_path: Path) -> None:
    base = BASE_CONFIG.read_text(encoding="utf-8")
    unknown_key = base.replace(
        'sample_status = "previously_explored"',
        'sample_status = "previously_explored"\nunexpected = "not-allowed"',
        1,
    )
    with pytest.raises(BaselineConfigurationError, match=r"unknown \[run\] keys.*unexpected"):
        load_baseline_config(_write_config(tmp_path, unknown_key))

    vader_compound = base.replace('score_definition = "vader_pos_minus_neg"', 'score_definition = "vader_compound"', 1)
    with pytest.raises(BaselineConfigurationError, match="stores pos - neg, not canonical VADER compound"):
        load_baseline_config(_write_config(tmp_path, vader_compound))

    changed_revision = base.replace(
        "4556d13015211d73dccd3fdd39d39232506f3e43",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        1,
    )
    with pytest.raises(BaselineConfigurationError, match="v1 FinBERT revision must be exactly"):
        load_baseline_config(_write_config(tmp_path, changed_revision))

    changed_cost = base.replace("cost_bps_per_side = 10.0", "cost_bps_per_side = 5.0", 1)
    with pytest.raises(BaselineConfigurationError, match="portfolio, NAV, cost, and liquidation rules are frozen"):
        load_baseline_config(_write_config(tmp_path, changed_cost))


def test_score_csv_requires_schema_and_exact_positive_minus_negative_score(tmp_path: Path) -> None:
    headline_hash = "a" * 64
    valid_path = tmp_path / "valid.csv"
    _write_scores(
        valid_path,
        [_score_row(headline_hash, "finbert"), _score_row(headline_hash, "vader", label="negative")],
    )

    loaded = load_headline_scores(valid_path, ("finbert", "vader"))

    assert loaded[(headline_hash, "finbert")].score == pytest.approx(0.7)
    assert loaded[(headline_hash, "vader")].score == pytest.approx(-0.7)

    missing_column_path = tmp_path / "missing-column.csv"
    _write_scores(
        missing_column_path,
        [_score_row(headline_hash, "finbert")],
        columns=tuple(column for column in SCORE_COLUMNS if column != "p_neutral"),
    )
    with pytest.raises(BaselineScoreError, match="missing columns.*p_neutral"):
        load_headline_scores(missing_column_path, ("finbert",))

    wrong_difference_path = tmp_path / "wrong-difference.csv"
    _write_scores(wrong_difference_path, [_score_row(headline_hash, "finbert", score="0.6")])
    with pytest.raises(BaselineScoreError, match="score is not p_positive - p_negative"):
        load_headline_scores(wrong_difference_path, ("finbert",))


def test_event_labels_join_on_normalized_headline_hash_filter_and_aggregate() -> None:
    config = load_baseline_config(BASE_CONFIG)
    selected_positive = _event("selected-positive", "AAA   wins a major contract!!!")
    selected_neutral = _event("selected-neutral", "AAA gives an unchanged outlook")
    non_explicit = _event("non-explicit", "AAA appears in an incidental list")
    multi_target = _event("multi-target", "AAA and BBB announce a partnership")
    technical = _event("technical", "AAA shares rise after the market opens")
    scores: dict[tuple[str, str], HeadlineScore] = {}
    # Simulate a later repeat that polluted the unique-headline metadata. The
    # chosen occurrence below remains the sole authority for screening.
    _add_event_scores(
        scores,
        selected_positive,
        ("positive", "negative"),
        matched_symbols=("AAA", "FUTURE"),
        explicit_target=False,
        market_price_technical=True,
    )
    _add_event_scores(scores, selected_neutral)
    _add_event_scores(scores, non_explicit, explicit_target=False)
    _add_event_scores(scores, multi_target, matched_symbols=("AAA", "BBB"))
    _add_event_scores(scores, technical, market_price_technical=True)

    metadata = {
        selected_positive.event_id: EventScreeningMetadata(("AAA",), True, False),
        selected_neutral.event_id: EventScreeningMetadata(("AAA",), True, False),
        non_explicit.event_id: EventScreeningMetadata(("AAA",), False, False),
        multi_target.event_id: EventScreeningMetadata(("AAA", "BBB"), True, False),
        technical.event_id: EventScreeningMetadata(("AAA",), True, True),
    }
    result = build_event_labels(
        (selected_positive, selected_neutral, non_explicit, multi_target, technical),
        scores,
        metadata,
        config,
    )

    assert len(result.labels) == 4
    assert {label.event_id for label in result.labels} == {"selected-positive", "selected-neutral"}
    assert all(label.headline_sha256 == headline_norm_sha256(next(
        event.headline for event in (selected_positive, selected_neutral) if event.event_id == label.event_id
    )) for label in result.labels)
    assert result.attrition == {
        "input_events": 5,
        "market_price_technical": 1,
        "missing_score": 0,
        "multi_target": 1,
        "non_explicit_target": 1,
        "selected_events": 2,
        "symbol_mismatch": 0,
    }
    signals = {signal.scorer: signal for signal in result.signals}
    assert signals["finbert"].event_count == 2
    assert signals["finbert"].mean_hard_label == pytest.approx(0.5)
    assert signals["finbert"].mean_continuous_score == pytest.approx(0.35)
    assert signals["finbert"].action == 1
    assert signals["vader"].mean_hard_label == pytest.approx(-0.5)
    assert signals["vader"].mean_continuous_score == pytest.approx(-0.35)
    assert signals["vader"].action == -1


def test_event_label_join_fails_closed_when_any_configured_score_is_missing() -> None:
    config = load_baseline_config(BASE_CONFIG)
    event = _event("missing-vader", "AAA announces a new contract")
    finbert = _headline_score(event.headline, "finbert", "positive")

    with pytest.raises(BaselineScoreError, match=r"event missing-vader.*\['vader'\]"):
        build_event_labels(
            (event,),
            {(finbert.headline_sha256, "finbert"): finbert},
            {event.event_id: EventScreeningMetadata(("AAA",), True, False)},
            config,
        )


def test_equal_weight_targets_require_both_legs_and_never_carry() -> None:
    sessions = ("2026-01-06", "2026-01-07", "2026-01-08")
    signals = (
        _signal("finbert", "AAA", sessions[0], 1),
        _signal("finbert", "BBB", sessions[0], 1),
        _signal("finbert", "CCC", sessions[0], -1),
        _signal("finbert", "DDD", sessions[0], -1),
        _signal("finbert", "AAA", sessions[2], 1),
    )

    result = build_equal_weight_targets(
        signals,
        scorer="finbert",
        sessions=sessions,
        symbols=("AAA", "BBB", "CCC", "DDD"),
        gross_exposure=1.0,
        minimum_names_per_side=1,
    )

    first, no_news, one_sided = result.targets
    assert first.weights() == {"AAA": 0.25, "BBB": 0.25, "CCC": -0.25, "DDD": -0.25}
    assert first.gross_exposure == pytest.approx(1.0)
    assert first.net_exposure == pytest.approx(0.0)
    assert no_news.weights() == {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "DDD": 0.0}
    assert one_sided.weights() == no_news.weights()
    assert no_news.gross_exposure == 0.0
    assert one_sided.gross_exposure == 0.0
    assert result.session_rows[0]["eligible"] is True
    assert result.session_rows[1]["no_trade_reason"] == "minimum_names_not_met_on_both_sides"
    assert result.session_rows[2]["no_trade_reason"] == "minimum_names_not_met_on_both_sides"
    assert next(position for position in one_sided.positions if position.symbol == "AAA").exclusion_reason == "insufficient_opposite_leg"
