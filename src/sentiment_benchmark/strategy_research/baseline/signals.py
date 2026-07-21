"""Headline-score validation, point-in-time joins, and frozen baseline targets."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ...headline_value import headline_norm_sha256
from ..portfolio import PositionTarget, TargetPortfolio
from ..schemas import StrategyEvent
from .config import LiteratureBaselineConfig

LABEL_VALUE = {"negative": -1, "neutral": 0, "positive": 1}


class BaselineScoreError(ValueError):
    """Raised when headline scores cannot support a fail-closed baseline."""


def _strict_bool(value: str, *, field: str, line_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise BaselineScoreError(f"invalid {field} boolean on score row {line_number}: {value!r}")


@dataclass(frozen=True)
class HeadlineScore:
    headline_sha256: str
    scorer: str
    matched_symbols: tuple[str, ...]
    explicit_target: bool
    contextual: bool
    market_price_technical: bool
    label: str
    p_positive: float
    p_negative: float
    p_neutral: float
    score: float


@dataclass(frozen=True)
class EventLabel:
    event_id: str
    story_family_id: str
    symbol: str
    eligible_execution_session: str
    headline_sha256: str
    scorer: str
    label: str
    hard_label: int
    score: float
    p_positive: float
    p_negative: float
    p_neutral: float

    def to_payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "story_family_id": self.story_family_id,
            "symbol": self.symbol,
            "eligible_execution_session": self.eligible_execution_session,
            "headline_sha256": self.headline_sha256,
            "scorer": self.scorer,
            "label": self.label,
            "hard_label": self.hard_label,
            "score": self.score,
            "p_positive": self.p_positive,
            "p_negative": self.p_negative,
            "p_neutral": self.p_neutral,
        }


@dataclass(frozen=True)
class FirmSessionSignal:
    scorer: str
    symbol: str
    session: str
    event_count: int
    positive_events: int
    negative_events: int
    neutral_events: int
    mean_hard_label: float
    mean_continuous_score: float
    action: int

    def to_payload(self) -> dict[str, object]:
        return {
            "scorer": self.scorer,
            "symbol": self.symbol,
            "session": self.session,
            "event_count": self.event_count,
            "positive_events": self.positive_events,
            "negative_events": self.negative_events,
            "neutral_events": self.neutral_events,
            "mean_hard_label": self.mean_hard_label,
            "mean_continuous_score": self.mean_continuous_score,
            "action": self.action,
        }


@dataclass(frozen=True)
class LabelBuildResult:
    labels: tuple[EventLabel, ...]
    signals: tuple[FirmSessionSignal, ...]
    attrition: dict[str, int]


@dataclass(frozen=True)
class PortfolioBuildResult:
    targets: tuple[TargetPortfolio, ...]
    session_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class EventScreeningMetadata:
    """Occurrence-specific flags derived from the chosen corpus revision."""

    matched_symbols: tuple[str, ...]
    explicit_target: bool
    market_price_technical: bool


def load_headline_scores(path: str | Path, scorers: Sequence[str]) -> dict[tuple[str, str], HeadlineScore]:
    """Load and validate the complete local headline score artifact."""

    target_scorers = set(scorers)
    required = {
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
    }
    loaded: dict[tuple[str, str], HeadlineScore] = {}
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise BaselineScoreError(f"headline score artifact is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                scorer = str(row.get("baseline") or "").strip()
                if scorer not in target_scorers:
                    continue
                headline_sha = str(row.get("headline_sha256") or "").strip()
                key = (headline_sha, scorer)
                if not headline_sha or len(headline_sha) != 64:
                    raise BaselineScoreError(f"invalid headline hash on score row {line_number}")
                if key in loaded:
                    raise BaselineScoreError(f"duplicate score for {headline_sha}/{scorer}")
                if row.get("status") != "success":
                    # Retain the missing key so any required event fails coverage below.
                    continue
                label = str(row.get("label") or "").strip().lower()
                if label not in LABEL_VALUE:
                    raise BaselineScoreError(f"invalid label on score row {line_number}: {label!r}")
                try:
                    probabilities = {
                        "positive": float(row["p_positive"]),
                        "negative": float(row["p_negative"]),
                        "neutral": float(row["p_neutral"]),
                    }
                    score = float(row["score"])
                except (TypeError, ValueError) as exc:
                    raise BaselineScoreError(f"non-numeric score row {line_number}") from exc
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
                    raise BaselineScoreError(f"probability outside [0, 1] on score row {line_number}")
                if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
                    raise BaselineScoreError(f"probabilities do not sum to one on score row {line_number}")
                if not math.isfinite(score) or not math.isclose(
                    score,
                    probabilities["positive"] - probabilities["negative"],
                    abs_tol=1e-12,
                ):
                    raise BaselineScoreError(f"score is not p_positive - p_negative on row {line_number}")
                maximum = max(probabilities.values())
                if not math.isclose(probabilities[label], maximum, abs_tol=1e-12):
                    raise BaselineScoreError(f"label is not an argmax class on score row {line_number}")
                symbols = tuple(
                    sorted({value.strip().upper() for value in str(row.get("matched_symbols") or "").split("|") if value.strip()})
                )
                if not symbols:
                    raise BaselineScoreError(f"score row {line_number} has no matched symbols")
                loaded[key] = HeadlineScore(
                    headline_sha256=headline_sha,
                    scorer=scorer,
                    matched_symbols=symbols,
                    explicit_target=_strict_bool(row["explicit_target"], field="explicit_target", line_number=line_number),
                    contextual=_strict_bool(row["contextual"], field="contextual", line_number=line_number),
                    market_price_technical=_strict_bool(
                        row["market_price_technical"], field="market_price_technical", line_number=line_number
                    ),
                    label=label,
                    p_positive=probabilities["positive"],
                    p_negative=probabilities["negative"],
                    p_neutral=probabilities["neutral"],
                    score=score,
                )
    except OSError as exc:
        raise BaselineScoreError(f"cannot read headline scores {path}: {exc}") from exc
    if not loaded:
        raise BaselineScoreError("headline score artifact contains no successful configured scores")
    return loaded


def build_event_labels(
    events: Sequence[StrategyEvent],
    scores: Mapping[tuple[str, str], HeadlineScore],
    event_metadata: Mapping[str, EventScreeningMetadata],
    config: LiteratureBaselineConfig,
) -> LabelBuildResult:
    """Join point-in-time events to headline-only labels and aggregate firm sessions."""

    attrition: Counter[str] = Counter(
        {
            "input_events": len(events),
            "missing_score": 0,
            "symbol_mismatch": 0,
            "non_explicit_target": 0,
            "multi_target": 0,
            "market_price_technical": 0,
            "selected_events": 0,
        }
    )
    scorer_ids = tuple(item.id for item in config.scorers)
    selected: list[EventLabel] = []
    for event in events:
        headline_sha = headline_norm_sha256(event.headline)
        event_scores: list[HeadlineScore] = []
        missing = [scorer for scorer in scorer_ids if (headline_sha, scorer) not in scores]
        if missing:
            attrition["missing_score"] += 1
            raise BaselineScoreError(
                f"event {event.event_id} is missing successful headline scores for {missing}"
            )
        event_scores = [scores[(headline_sha, scorer)] for scorer in scorer_ids]
        occurrence = event_metadata.get(event.event_id)
        if occurrence is None:
            raise BaselineScoreError(f"event {event.event_id} has no occurrence-specific screening metadata")
        if event.symbol not in occurrence.matched_symbols:
            attrition["symbol_mismatch"] += 1
            raise BaselineScoreError(
                f"event symbol {event.symbol} is absent from its chosen corpus occurrence"
            )
        if config.screening.require_explicit_target and not occurrence.explicit_target:
            attrition["non_explicit_target"] += 1
            continue
        if config.screening.require_single_target and len(occurrence.matched_symbols) != 1:
            attrition["multi_target"] += 1
            continue
        if config.screening.exclude_market_price_technical and occurrence.market_price_technical:
            attrition["market_price_technical"] += 1
            continue
        attrition["selected_events"] += 1
        for item in event_scores:
            selected.append(
                EventLabel(
                    event_id=event.event_id,
                    story_family_id=event.story_family_id,
                    symbol=event.symbol,
                    eligible_execution_session=event.eligible_execution_session.isoformat(),
                    headline_sha256=headline_sha,
                    scorer=item.scorer,
                    label=item.label,
                    hard_label=LABEL_VALUE[item.label],
                    score=item.score,
                    p_positive=item.p_positive,
                    p_negative=item.p_negative,
                    p_neutral=item.p_neutral,
                )
            )
    selected.sort(key=lambda row: (row.eligible_execution_session, row.symbol, row.event_id, row.scorer))
    grouped: dict[tuple[str, str, str], list[EventLabel]] = defaultdict(list)
    for row in selected:
        grouped[(row.scorer, row.symbol, row.eligible_execution_session)].append(row)
    signals: list[FirmSessionSignal] = []
    for (scorer, symbol, session), rows in sorted(grouped.items()):
        if len(rows) < config.signal.minimum_events:
            continue
        mean_hard = sum(row.hard_label for row in rows) / len(rows)
        mean_score = sum(row.score for row in rows) / len(rows)
        action = 1 if mean_hard > 0 else -1 if mean_hard < 0 else 0
        counts = Counter(row.label for row in rows)
        signals.append(
            FirmSessionSignal(
                scorer=scorer,
                symbol=symbol,
                session=session,
                event_count=len(rows),
                positive_events=counts["positive"],
                negative_events=counts["negative"],
                neutral_events=counts["neutral"],
                mean_hard_label=mean_hard,
                mean_continuous_score=mean_score,
                action=action,
            )
        )
    return LabelBuildResult(tuple(selected), tuple(signals), dict(sorted(attrition.items())))


def build_equal_weight_targets(
    signals: Sequence[FirmSessionSignal],
    *,
    scorer: str,
    sessions: Sequence[str],
    symbols: Sequence[str],
    gross_exposure: float,
    minimum_names_per_side: int,
) -> PortfolioBuildResult:
    """Create daily dollar-neutral targets with no news-state carry."""

    signal_lookup = {
        (row.session, row.symbol): row.action
        for row in signals
        if row.scorer == scorer
    }
    if len(signal_lookup) != sum(row.scorer == scorer for row in signals):
        raise ValueError(f"duplicate firm-session signal for scorer {scorer}")
    symbol_tuple = tuple(sorted(set(symbols)))
    targets: list[TargetPortfolio] = []
    audit_rows: list[dict[str, object]] = []
    for session in sessions:
        actions = {symbol: signal_lookup.get((session, symbol), 0) for symbol in symbol_tuple}
        longs = sorted(symbol for symbol, action in actions.items() if action > 0)
        shorts = sorted(symbol for symbol, action in actions.items() if action < 0)
        eligible = len(longs) >= minimum_names_per_side and len(shorts) >= minimum_names_per_side
        long_weight = gross_exposure / 2 / len(longs) if eligible else 0.0
        short_weight = -gross_exposure / 2 / len(shorts) if eligible else 0.0
        weights = {
            symbol: long_weight if symbol in longs and eligible else short_weight if symbol in shorts and eligible else 0.0
            for symbol in symbol_tuple
        }
        positions = tuple(
            PositionTarget(
                symbol=symbol,
                action=float(actions[symbol]),
                volatility=None,
                raw_weight=weights[symbol],
                target_weight=weights[symbol],
                exclusion_reason=None if eligible or actions[symbol] == 0 else "insufficient_opposite_leg",
            )
            for symbol in symbol_tuple
        )
        long_exposure = sum(weight for weight in weights.values() if weight > 0)
        short_exposure = sum(-weight for weight in weights.values() if weight < 0)
        gross = long_exposure + short_exposure
        net = long_exposure - short_exposure
        if gross > gross_exposure + 1e-12 or abs(net) > 1e-12:
            raise ValueError(f"equal-weight target violates exposure constraints on {session}")
        targets.append(
            TargetPortfolio(
                session=session,
                positions=positions,
                gross_exposure=gross,
                net_exposure=net,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                cash_weight=1 - net,
            )
        )
        audit_rows.append(
            {
                "scorer": scorer,
                "session": session,
                "long_candidates": len(longs),
                "short_candidates": len(shorts),
                "eligible": eligible,
                "no_trade_reason": None if eligible else "minimum_names_not_met_on_both_sides",
                "gross_exposure": gross,
                "net_exposure": net,
                "active_names": sum(weight != 0 for weight in weights.values()),
            }
        )
    return PortfolioBuildResult(tuple(targets), tuple(audit_rows))
