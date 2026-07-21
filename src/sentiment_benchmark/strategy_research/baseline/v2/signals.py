"""V2 label joins, arm actions, and dollar-neutral target construction.

The v2 signal layer differs from v1 in three predeclared ways: the comparator
labels come from a canonical compound-threshold VADER artifact, arms may weight
legs by score magnitude, and an agreement arm trades only firm-sessions where
both scorers agree. Screening, point-in-time joins, and the sign-of-mean rule
for directions are unchanged from v1.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ....baselines import VADER_THRESHOLD
from ....headline_value import headline_norm_sha256
from ...portfolio import PositionTarget, TargetPortfolio
from ...schemas import StrategyEvent
from ..signals import (
    LABEL_VALUE,
    BaselineScoreError,
    EventScreeningMetadata,
    FirmSessionSignal,
    HeadlineScore,
    PortfolioBuildResult,
)
from .config import LiteratureBaselineV2Config


@dataclass(frozen=True)
class CompoundScore:
    """One canonical compound-threshold VADER classification of a headline."""

    headline_sha256: str
    label: str
    compound: float


@dataclass(frozen=True)
class V2EventLabel:
    """A per-event, per-scorer label; probabilities are FinBERT-only."""

    event_id: str
    story_family_id: str
    symbol: str
    eligible_execution_session: str
    headline_sha256: str
    scorer: str
    label: str
    hard_label: int
    score: float
    p_positive: float | None
    p_negative: float | None
    p_neutral: float | None

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
class V2LabelBuildResult:
    labels: tuple[V2EventLabel, ...]
    signals: tuple[FirmSessionSignal, ...]
    attrition: dict[str, int]


@dataclass(frozen=True)
class ArmSignal:
    """The traded direction and weighting magnitude for one firm-session."""

    session: str
    symbol: str
    action: int
    magnitude: float


def load_vader_compound_scores(path: str | Path) -> dict[str, CompoundScore]:
    """Load and validate the canonical compound-threshold VADER artifact."""

    required = {
        "headline_sha256",
        "baseline",
        "label",
        "compound",
        "status",
    }
    loaded: dict[str, CompoundScore] = {}
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise BaselineScoreError(f"VADER compound artifact is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                if str(row.get("baseline") or "").strip() != "vader_compound":
                    raise BaselineScoreError(
                        f"unexpected baseline on VADER compound row {line_number}: {row.get('baseline')!r}"
                    )
                headline_sha = str(row.get("headline_sha256") or "").strip()
                if not headline_sha or len(headline_sha) != 64:
                    raise BaselineScoreError(f"invalid headline hash on VADER compound row {line_number}")
                if headline_sha in loaded:
                    raise BaselineScoreError(f"duplicate VADER compound score for {headline_sha}")
                if row.get("status") != "success":
                    # Retain the missing key so any required event fails coverage.
                    continue
                label = str(row.get("label") or "").strip().lower()
                if label not in LABEL_VALUE:
                    raise BaselineScoreError(f"invalid label on VADER compound row {line_number}: {label!r}")
                try:
                    compound = float(row["compound"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise BaselineScoreError(f"non-numeric compound on row {line_number}") from exc
                if not math.isfinite(compound) or not -1 <= compound <= 1:
                    raise BaselineScoreError(f"compound outside [-1, 1] on row {line_number}")
                expected = (
                    "positive"
                    if compound >= VADER_THRESHOLD
                    else "negative"
                    if compound <= -VADER_THRESHOLD
                    else "neutral"
                )
                if label != expected:
                    raise BaselineScoreError(
                        f"label does not follow the compound threshold convention on row {line_number}"
                    )
                loaded[headline_sha] = CompoundScore(headline_sha256=headline_sha, label=label, compound=compound)
    except OSError as exc:
        raise BaselineScoreError(f"cannot read VADER compound scores {path}: {exc}") from exc
    if not loaded:
        raise BaselineScoreError("VADER compound artifact contains no successful scores")
    return loaded


def build_v2_event_labels(
    events: Sequence[StrategyEvent],
    finbert_scores: Mapping[tuple[str, str], HeadlineScore],
    compound_scores: Mapping[str, CompoundScore],
    event_metadata: Mapping[str, EventScreeningMetadata],
    config: LiteratureBaselineV2Config,
) -> V2LabelBuildResult:
    """Join events to FinBERT and canonical-VADER labels with v1 screening."""

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
    selected: list[V2EventLabel] = []
    for event in events:
        headline_sha = headline_norm_sha256(event.headline)
        finbert = finbert_scores.get((headline_sha, "finbert"))
        compound = compound_scores.get(headline_sha)
        if finbert is None or compound is None:
            attrition["missing_score"] += 1
            missing = [name for name, value in (("finbert", finbert), ("vader_compound", compound)) if value is None]
            raise BaselineScoreError(f"event {event.event_id} is missing successful headline scores for {missing}")
        occurrence = event_metadata.get(event.event_id)
        if occurrence is None:
            raise BaselineScoreError(f"event {event.event_id} has no occurrence-specific screening metadata")
        if event.symbol not in occurrence.matched_symbols:
            attrition["symbol_mismatch"] += 1
            raise BaselineScoreError(f"event symbol {event.symbol} is absent from its chosen corpus occurrence")
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
        common = {
            "event_id": event.event_id,
            "story_family_id": event.story_family_id,
            "symbol": event.symbol,
            "eligible_execution_session": event.eligible_execution_session.isoformat(),
            "headline_sha256": headline_sha,
        }
        selected.append(
            V2EventLabel(
                **common,
                scorer="finbert",
                label=finbert.label,
                hard_label=LABEL_VALUE[finbert.label],
                score=finbert.score,
                p_positive=finbert.p_positive,
                p_negative=finbert.p_negative,
                p_neutral=finbert.p_neutral,
            )
        )
        selected.append(
            V2EventLabel(
                **common,
                scorer="vader_compound",
                label=compound.label,
                hard_label=LABEL_VALUE[compound.label],
                score=compound.compound,
                p_positive=None,
                p_negative=None,
                p_neutral=None,
            )
        )
    selected.sort(key=lambda row: (row.eligible_execution_session, row.symbol, row.event_id, row.scorer))
    grouped: dict[tuple[str, str, str], list[V2EventLabel]] = defaultdict(list)
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
    return V2LabelBuildResult(tuple(selected), tuple(signals), dict(sorted(attrition.items())))


def build_arm_signals(
    signals: Sequence[FirmSessionSignal],
    arm: str,
) -> tuple[ArmSignal, ...]:
    """Resolve one predeclared arm's firm-session directions and magnitudes.

    Directions always come from the sign-of-mean hard label; magnitudes are the
    absolute mean continuous score and are only used by magnitude-weighted arms.
    The agreement arm keeps a FinBERT direction only when the canonical VADER
    direction is identical and non-zero.
    """

    by_scorer: dict[str, dict[tuple[str, str], FirmSessionSignal]] = defaultdict(dict)
    for row in signals:
        key = (row.session, row.symbol)
        if key in by_scorer[row.scorer]:
            raise ValueError(f"duplicate firm-session signal for {row.scorer} on {key}")
        by_scorer[row.scorer][key] = row
    if arm in {"finbert", "finbert_magnitude"}:
        source = by_scorer.get("finbert", {})
        return tuple(
            ArmSignal(session=key[0], symbol=key[1], action=row.action, magnitude=abs(row.mean_continuous_score))
            for key, row in sorted(source.items())
            if row.action != 0
        )
    if arm == "vader_compound":
        source = by_scorer.get("vader_compound", {})
        return tuple(
            ArmSignal(session=key[0], symbol=key[1], action=row.action, magnitude=abs(row.mean_continuous_score))
            for key, row in sorted(source.items())
            if row.action != 0
        )
    if arm == "agreement":
        finbert = by_scorer.get("finbert", {})
        vader = by_scorer.get("vader_compound", {})
        agreed: list[ArmSignal] = []
        for key, row in sorted(finbert.items()):
            other = vader.get(key)
            if other is None or row.action == 0 or row.action != other.action:
                continue
            agreed.append(
                ArmSignal(session=key[0], symbol=key[1], action=row.action, magnitude=abs(row.mean_continuous_score))
            )
        return tuple(agreed)
    raise ValueError(f"unknown v2 arm: {arm!r}")


def build_v2_targets(
    arm_signals: Sequence[ArmSignal],
    *,
    arm: str,
    sessions: Sequence[str],
    symbols: Sequence[str],
    gross_exposure: float,
    minimum_names_per_side: int,
    weighting: str,
) -> PortfolioBuildResult:
    """Create daily dollar-neutral targets with equal or magnitude side weights.

    Magnitude weighting allocates each side's half of the gross exposure in
    proportion to the absolute mean continuous score; a side whose magnitudes
    are all zero falls back to equal weight rather than dropping the session.
    """

    if weighting not in {"equal", "magnitude"}:
        raise ValueError(f"unknown weighting: {weighting!r}")
    lookup: dict[tuple[str, str], ArmSignal] = {}
    for row in arm_signals:
        key = (row.session, row.symbol)
        if key in lookup:
            raise ValueError(f"duplicate arm signal for {arm} on {key}")
        lookup[key] = row
    symbol_tuple = tuple(sorted(set(symbols)))
    targets: list[TargetPortfolio] = []
    audit_rows: list[dict[str, object]] = []
    for session in sessions:
        rows = {symbol: lookup.get((session, symbol)) for symbol in symbol_tuple}
        longs = sorted(symbol for symbol, row in rows.items() if row is not None and row.action > 0)
        shorts = sorted(symbol for symbol, row in rows.items() if row is not None and row.action < 0)
        eligible = len(longs) >= minimum_names_per_side and len(shorts) >= minimum_names_per_side
        weights = dict.fromkeys(symbol_tuple, 0.0)
        if eligible:
            for side_symbols, sign in ((longs, 1.0), (shorts, -1.0)):
                side_budget = gross_exposure / 2
                if weighting == "magnitude":
                    magnitudes = {symbol: rows[symbol].magnitude for symbol in side_symbols}  # type: ignore[union-attr]
                    total = sum(magnitudes.values())
                else:
                    magnitudes = dict.fromkeys(side_symbols, 1.0)
                    total = float(len(side_symbols))
                if total <= 0:
                    magnitudes = dict.fromkeys(side_symbols, 1.0)
                    total = float(len(side_symbols))
                for symbol in side_symbols:
                    weights[symbol] = sign * side_budget * magnitudes[symbol] / total
        positions = tuple(
            PositionTarget(
                symbol=symbol,
                action=float(rows[symbol].action) if rows[symbol] is not None else 0.0,
                volatility=None,
                raw_weight=weights[symbol],
                target_weight=weights[symbol],
                exclusion_reason=(
                    None
                    if eligible or rows[symbol] is None or rows[symbol].action == 0  # type: ignore[union-attr]
                    else "insufficient_opposite_leg"
                ),
            )
            for symbol in symbol_tuple
        )
        long_exposure = sum(weight for weight in weights.values() if weight > 0)
        short_exposure = sum(-weight for weight in weights.values() if weight < 0)
        gross = long_exposure + short_exposure
        net = long_exposure - short_exposure
        if gross > gross_exposure + 1e-9 or abs(net) > 1e-9:
            raise ValueError(f"v2 target violates exposure constraints on {session}")
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
                "scorer": arm,
                "session": session,
                "long_candidates": len(longs),
                "short_candidates": len(shorts),
                "eligible": eligible,
                "no_trade_reason": None if eligible else "minimum_names_not_met_on_both_sides",
                "gross_exposure": gross,
                "net_exposure": net,
                "active_names": sum(weight != 0 for weight in weights.values()),
                "max_name_weight": max((abs(weight) for weight in weights.values()), default=0.0),
            }
        )
    return PortfolioBuildResult(tuple(targets), tuple(audit_rows))


__all__ = [
    "ArmSignal",
    "CompoundScore",
    "V2EventLabel",
    "V2LabelBuildResult",
    "build_arm_signals",
    "build_v2_event_labels",
    "build_v2_targets",
    "load_vader_compound_scores",
]
