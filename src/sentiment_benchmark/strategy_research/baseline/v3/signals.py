"""Point-in-time FinBERT labels and lagged cross-sectional rank targets."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ....headline_value import headline_norm_sha256
from ...portfolio import PositionTarget, TargetPortfolio
from ...schemas import StrategyEvent
from ..signals import (
    LABEL_VALUE,
    BaselineScoreError,
    EventLabel,
    EventScreeningMetadata,
    FirmSessionSignal,
    HeadlineScore,
    LabelBuildResult,
)
from .config import ScreeningConfig, SignalConfig


@dataclass(frozen=True)
class RankPortfolioBuildResult:
    targets: tuple[TargetPortfolio, ...]
    session_rows: tuple[dict[str, object], ...]


def build_finbert_labels(
    events: Sequence[StrategyEvent],
    scores: Mapping[tuple[str, str], HeadlineScore],
    event_metadata: Mapping[str, EventScreeningMetadata],
    screening: ScreeningConfig,
) -> LabelBuildResult:
    """Join earliest story-family events to one immutable FinBERT score each."""

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
    selected: list[EventLabel] = []
    for event in events:
        headline_sha = headline_norm_sha256(event.headline)
        score = scores.get((headline_sha, "finbert"))
        if score is None:
            attrition["missing_score"] += 1
            raise BaselineScoreError(f"event {event.event_id} is missing a successful FinBERT score")
        occurrence = event_metadata.get(event.event_id)
        if occurrence is None:
            raise BaselineScoreError(f"event {event.event_id} has no occurrence-specific screening metadata")
        if event.symbol not in occurrence.matched_symbols:
            attrition["symbol_mismatch"] += 1
            raise BaselineScoreError(f"event symbol {event.symbol} is absent from its chosen occurrence")
        if screening.require_explicit_target and not occurrence.explicit_target:
            attrition["non_explicit_target"] += 1
            continue
        if screening.require_single_target and len(occurrence.matched_symbols) != 1:
            attrition["multi_target"] += 1
            continue
        if screening.exclude_market_price_technical and occurrence.market_price_technical:
            attrition["market_price_technical"] += 1
            continue
        selected.append(
            EventLabel(
                event_id=event.event_id,
                story_family_id=event.story_family_id,
                symbol=event.symbol,
                eligible_execution_session=event.eligible_execution_session.isoformat(),
                headline_sha256=headline_sha,
                scorer="finbert",
                label=score.label,
                hard_label=LABEL_VALUE[score.label],
                score=score.score,
                p_positive=score.p_positive,
                p_negative=score.p_negative,
                p_neutral=score.p_neutral,
            )
        )
    attrition["selected_events"] = len(selected)
    grouped: dict[tuple[str, str], list[EventLabel]] = defaultdict(list)
    for row in selected:
        grouped[(row.symbol, row.eligible_execution_session)].append(row)
    signals: list[FirmSessionSignal] = []
    for (symbol, session), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        mean_hard = sum(row.hard_label for row in rows) / len(rows)
        mean_score = sum(row.score for row in rows) / len(rows)
        action = 1 if mean_hard > 0 else -1 if mean_hard < 0 else 0
        signals.append(
            FirmSessionSignal(
                scorer="finbert",
                symbol=symbol,
                session=session,
                event_count=len(rows),
                positive_events=sum(row.hard_label > 0 for row in rows),
                negative_events=sum(row.hard_label < 0 for row in rows),
                neutral_events=sum(row.hard_label == 0 for row in rows),
                mean_hard_label=mean_hard,
                mean_continuous_score=mean_score,
                action=action,
            )
        )
    return LabelBuildResult(labels=tuple(selected), signals=tuple(signals), attrition=dict(sorted(attrition.items())))


def _percentile_ranks(rows: Mapping[str, float]) -> dict[str, float]:
    """Return deterministic centred percentile ranks in [-1, 1]."""

    ordered = sorted(rows.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    if count == 1:
        return {ordered[0][0]: 0.0}
    return {symbol: 2.0 * index / (count - 1) - 1.0 for index, (symbol, _value) in enumerate(ordered)}


def build_rank_reversal_targets(
    signals: Sequence[FirmSessionSignal],
    *,
    sessions: Sequence[str],
    symbols: Sequence[str],
    gross_exposure: float,
    config: SignalConfig,
) -> RankPortfolioBuildResult:
    """Build equal-weight tail portfolios from lagged rolling sentiment ranks.

    Each formation session ranks the continuous FinBERT firm score across the
    firms with target-specific news. A decision uses only the previous five
    formation sessions, averages those ranks by firm, then takes the opposite
    side of the extreme tails. Current-session news cannot affect the target.
    """

    if not 0 < gross_exposure <= 1:
        raise ValueError("gross exposure must be in (0, 1]")
    symbol_tuple = tuple(sorted(set(symbols)))
    if not symbol_tuple:
        raise ValueError("rank strategy requires symbols")
    session_tuple = tuple(sessions)
    if tuple(sorted(set(session_tuple))) != session_tuple:
        raise ValueError("sessions must be sorted and unique")
    raw: dict[str, dict[str, float]] = defaultdict(dict)
    for row in signals:
        if row.scorer != "finbert":
            continue
        if row.symbol not in symbol_tuple:
            raise ValueError(f"signal symbol is absent from portfolio universe: {row.symbol}")
        if not math.isfinite(row.mean_continuous_score):
            raise ValueError(f"non-finite FinBERT score for {row.symbol}/{row.session}")
        if row.symbol in raw[row.session]:
            raise ValueError(f"duplicate FinBERT firm-session signal: {row.symbol}/{row.session}")
        raw[row.session][row.symbol] = row.mean_continuous_score
    ranks = {session: _percentile_ranks(values) for session, values in raw.items()}

    targets: list[TargetPortfolio] = []
    audit: list[dict[str, object]] = []
    for index, session in enumerate(session_tuple):
        end = index - config.formation_lag_sessions + 1
        start = max(0, end - config.lookback_sessions)
        formation_sessions = session_tuple[start:end] if end > 0 else ()
        history: dict[str, list[float]] = defaultdict(list)
        for formation_session in formation_sessions:
            for symbol, value in ranks.get(formation_session, {}).items():
                history[symbol].append(value)
        rolling = {symbol: sum(values) / len(values) for symbol, values in history.items() if values}
        ranked_firms = len(rolling)
        tail_size = max(config.minimum_names_per_side, int(math.floor(ranked_firms * config.tail_fraction)))
        eligible = ranked_firms >= config.minimum_ranked_firms and tail_size * 2 <= ranked_firms
        weights = dict.fromkeys(symbol_tuple, 0.0)
        long_symbols: tuple[str, ...] = ()
        short_symbols: tuple[str, ...] = ()
        reason = ""
        if eligible:
            ordered = sorted(rolling.items(), key=lambda item: (item[1], item[0]))
            # Contrarian: low/negative sentiment ranks are long, high ranks short.
            long_symbols = tuple(symbol for symbol, _value in ordered[:tail_size])
            short_symbols = tuple(symbol for symbol, _value in ordered[-tail_size:])
            side_weight = gross_exposure / 2 / tail_size
            for symbol in long_symbols:
                weights[symbol] = side_weight
            for symbol in short_symbols:
                weights[symbol] = -side_weight
        else:
            reason = "minimum_ranked_firms_not_met"
        positions = tuple(
            PositionTarget(
                symbol=symbol,
                action=1.0 if weights[symbol] > 0 else -1.0 if weights[symbol] < 0 else 0.0,
                volatility=None,
                raw_weight=weights[symbol],
                target_weight=weights[symbol],
                exclusion_reason=None if eligible else reason,
            )
            for symbol in symbol_tuple
        )
        long_exposure = sum(weight for weight in weights.values() if weight > 0)
        short_exposure = sum(-weight for weight in weights.values() if weight < 0)
        gross = long_exposure + short_exposure
        net = long_exposure - short_exposure
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
        audit.append(
            {
                "arm": "finbert_rank_reversal",
                "session": session,
                "formation_sessions": list(formation_sessions),
                "ranked_firms": ranked_firms,
                "tail_size": tail_size if eligible else 0,
                "long_symbols": list(long_symbols),
                "short_symbols": list(short_symbols),
                "active_names": sum(weight != 0 for weight in weights.values()),
                "gross_exposure": gross,
                "net_exposure": net,
                "max_name_weight": max((abs(weight) for weight in weights.values()), default=0.0),
                "no_trade_reason": reason,
            }
        )
    return RankPortfolioBuildResult(targets=tuple(targets), session_rows=tuple(audit))
