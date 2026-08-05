"""Sparse exact-extrema portfolio construction for the LSEG/Gemma arm.

The helper deliberately does one thing: turn exact +1/-1 firm-open signals
into an equal-leg dollar-neutral target, or cash when either leg is absent.
Execution, drifted-weight accounting, costs, and liquidation remain delegated
to the validated strategy-research ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sentiment_benchmark.strategy_research.ledger import DailyLedgerRow
from sentiment_benchmark.strategy_research.portfolio import PositionTarget, TargetPortfolio


def build_exact_extrema_targets(
    panel: pd.DataFrame,
    *,
    signal_col: str = "strongest_event",
    gross_exposure: float = 1.0,
    minimum_names_per_side: int = 1,
    single_name_cap: float | None = None,
    tolerance: float = 1e-12,
) -> tuple[tuple[TargetPortfolio, ...], pd.DataFrame]:
    """Build one-session equal-leg targets from exact positive/negative events.

    Every session is represented. A session trades only when it has at least
    ``minimum_names_per_side`` exact +1 names and the same minimum of exact -1
    names. Each active leg receives half the requested gross exposure unless a
    ``single_name_cap`` makes either leg too sparse. In that case both legs are
    reduced to the same feasible budget and unused gross capacity remains cash.
    """
    if not math.isfinite(gross_exposure) or not 0 < gross_exposure <= 1:
        raise ValueError("gross_exposure must be finite and in (0, 1]")
    if minimum_names_per_side < 1:
        raise ValueError("minimum_names_per_side must be positive")
    if single_name_cap is not None and (
        not math.isfinite(single_name_cap)
        or not 0 < single_name_cap <= gross_exposure / 2.0
    ):
        raise ValueError("single_name_cap must be finite and in (0, gross_exposure / 2]")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    required = {"session_date", "symbol", signal_col}
    if missing := required - set(panel.columns):
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    frame = panel.loc[:, ["session_date", "symbol", signal_col]].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if frame[["session_date", "symbol"]].isna().any().any():
        raise ValueError("session_date and symbol must be non-missing")
    if frame.duplicated(["session_date", "symbol"]).any():
        raise ValueError("panel must be unique by session_date and symbol")

    values = pd.to_numeric(frame[signal_col], errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isnan(values) & (~np.isfinite(values) | (np.abs(values) > 1.0 + tolerance))
    if invalid.any():
        raise ValueError("signals must be missing or finite values in [-1, 1]")
    frame[signal_col] = values

    sessions = pd.DatetimeIndex(sorted(frame["session_date"].unique()))
    symbols = tuple(sorted(frame["symbol"].unique()))
    if not len(sessions) or not symbols:
        raise ValueError("panel must contain at least one session and symbol")
    session_sets = frame.groupby("session_date")["symbol"].agg(lambda x: frozenset(x))
    expected = frozenset(symbols)
    if not session_sets.map(lambda value: value == expected).all():
        raise ValueError("every session must contain the same symbol population")

    targets: list[TargetPortfolio] = []
    audit_rows: list[dict[str, Any]] = []
    for session in sessions:
        day = frame.loc[frame["session_date"].eq(session)].set_index("symbol").reindex(symbols)
        signal = day[signal_col].to_numpy(dtype=float)
        positive = np.isclose(signal, 1.0, rtol=0.0, atol=tolerance, equal_nan=False)
        negative = np.isclose(signal, -1.0, rtol=0.0, atol=tolerance, equal_nan=False)
        n_positive = int(positive.sum())
        n_negative = int(negative.sum())
        eligible = (
            n_positive >= minimum_names_per_side
            and n_negative >= minimum_names_per_side
        )
        weights = np.zeros(len(symbols), dtype=float)
        leg_exposure = 0.0
        if eligible:
            leg_exposure = gross_exposure / 2.0
            if single_name_cap is not None:
                leg_exposure = min(
                    leg_exposure,
                    single_name_cap * n_positive,
                    single_name_cap * n_negative,
                )
            weights[positive] = leg_exposure / n_positive
            weights[negative] = -leg_exposure / n_negative

        positions = tuple(
            PositionTarget(
                symbol=symbol,
                action=1.0 if is_positive else -1.0 if is_negative else 0.0,
                volatility=None,
                raw_weight=float(weight),
                target_weight=float(weight),
                exclusion_reason=(
                    "minimum_names_per_side_not_met"
                    if (is_positive or is_negative) and not eligible
                    else None
                ),
            )
            for symbol, weight, is_positive, is_negative in zip(
                symbols,
                weights,
                positive,
                negative,
                strict=True,
            )
        )
        long_exposure = float(weights[weights > 0].sum())
        short_exposure = float(-weights[weights < 0].sum())
        net_exposure = long_exposure - short_exposure
        gross = long_exposure + short_exposure
        if abs(net_exposure) > 1e-12 or gross > gross_exposure + 1e-12:
            raise RuntimeError("exact-extrema target violates exposure constraints")
        targets.append(
            TargetPortfolio(
                session=str(pd.Timestamp(session).date()),
                positions=positions,
                gross_exposure=gross,
                net_exposure=net_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                cash_weight=1.0 - net_exposure,
            )
        )
        audit_rows.append(
            {
                "session_date": session,
                "positive_names": n_positive,
                "negative_names": n_negative,
                "eligible": eligible,
                "gross_exposure": gross,
                "net_exposure": net_exposure,
                "active_names": int(np.count_nonzero(weights)),
                "leg_exposure": leg_exposure,
                "unused_gross_capacity": gross_exposure - gross,
                "single_name_cap": single_name_cap,
                "max_abs_name_weight": float(np.max(np.abs(weights))),
                "long_hhi": float(np.square(weights[weights > 0] / long_exposure).sum())
                if long_exposure > 0
                else 0.0,
                "short_hhi": float(np.square(-weights[weights < 0] / short_exposure).sum())
                if short_exposure > 0
                else 0.0,
            }
        )
    return tuple(targets), pd.DataFrame(audit_rows)


def sector_neutralize_targets(
    targets: Sequence[TargetPortfolio],
    *,
    sector_by_symbol: Mapping[str, str],
    gross_exposure: float = 1.0,
    single_name_cap: float = 0.25,
    tolerance: float = 1e-12,
) -> tuple[tuple[TargetPortfolio, ...], pd.DataFrame]:
    """Demean target weights within sectors and retain unused capacity as cash.

    The transformation is deterministic and return-blind. For every session it
    subtracts each sector's equal-weight mean target from all names in that
    sector, thereby making every sector's dollar exposure zero. The complete
    centred portfolio is then scaled by one common multiplier only when needed
    to respect the gross and single-name limits. It never scales up a sparse
    portfolio to consume unused capacity.
    """
    if not math.isfinite(gross_exposure) or not 0 < gross_exposure <= 1:
        raise ValueError("gross_exposure must be finite and in (0, 1]")
    if (
        not math.isfinite(single_name_cap)
        or not 0 < single_name_cap <= gross_exposure
    ):
        raise ValueError("single_name_cap must be finite and in (0, gross_exposure]")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    target_rows = tuple(targets)
    if not target_rows:
        raise ValueError("targets must contain at least one session")

    expected_symbols = tuple(position.symbol for position in target_rows[0].positions)
    if len(expected_symbols) != len(set(expected_symbols)) or not expected_symbols:
        raise ValueError("each target must contain a non-empty unique symbol population")
    missing_sector = sorted(set(expected_symbols) - set(sector_by_symbol))
    if missing_sector:
        raise ValueError(f"sector mapping missing symbols: {missing_sector}")
    sectors = {symbol: str(sector_by_symbol[symbol]).strip() for symbol in expected_symbols}
    if any(not sector for sector in sectors.values()):
        raise ValueError("sector labels must be non-empty")
    sector_members: dict[str, list[int]] = {}
    for index, symbol in enumerate(expected_symbols):
        sector_members.setdefault(sectors[symbol], []).append(index)
    if any(len(members) < 2 for members in sector_members.values()):
        raise ValueError("every represented sector must contain at least two symbols")

    neutral_targets: list[TargetPortfolio] = []
    audit_rows: list[dict[str, Any]] = []
    for target in target_rows:
        symbols = tuple(position.symbol for position in target.positions)
        if symbols != expected_symbols:
            raise ValueError("every target must use the same ordered symbol population")
        base = np.asarray(
            [position.target_weight for position in target.positions], dtype=float
        )
        if not np.isfinite(base).all():
            raise ValueError("target weights must be finite")
        centred = base.copy()
        base_sector_exposure: dict[str, float] = {}
        for sector, members in sector_members.items():
            indices = np.asarray(members, dtype=int)
            base_sector_exposure[sector] = float(base[indices].sum())
            centred[indices] -= float(base[indices].mean())

        pre_scale_gross = float(np.abs(centred).sum())
        pre_scale_max = float(np.abs(centred).max())
        scale = 1.0
        if pre_scale_gross > gross_exposure:
            scale = min(scale, gross_exposure / pre_scale_gross)
        if pre_scale_max > single_name_cap:
            scale = min(scale, single_name_cap / pre_scale_max)
        weights = centred * scale
        weights[np.abs(weights) <= tolerance] = 0.0

        long_exposure = float(weights[weights > 0].sum())
        short_exposure = float(-weights[weights < 0].sum())
        net_exposure = long_exposure - short_exposure
        gross = long_exposure + short_exposure
        neutral_sector_exposure = {
            sector: float(weights[np.asarray(members, dtype=int)].sum())
            for sector, members in sector_members.items()
        }
        max_sector_exposure = max(map(abs, neutral_sector_exposure.values()))
        max_abs_weight = float(np.abs(weights).max())
        if (
            abs(net_exposure) > tolerance
            or gross > gross_exposure + tolerance
            or max_abs_weight > single_name_cap + tolerance
            or max_sector_exposure > tolerance
        ):
            raise RuntimeError("sector-neutral target violates exposure constraints")

        positions = tuple(
            PositionTarget(
                symbol=position.symbol,
                action=(
                    position.action
                    if abs(position.target_weight) > tolerance
                    else float(np.sign(weight))
                ),
                volatility=position.volatility,
                raw_weight=float(raw_weight),
                target_weight=float(weight),
                exclusion_reason=(
                    "sector_neutral_hedge"
                    if abs(position.target_weight) <= tolerance and abs(weight) > tolerance
                    else position.exclusion_reason
                ),
            )
            for position, raw_weight, weight in zip(
                target.positions, centred, weights, strict=True
            )
        )
        signal_mask = np.abs(base) > tolerance
        hedge_mask = ~signal_mask & (np.abs(weights) > tolerance)
        neutral_targets.append(
            TargetPortfolio(
                session=target.session,
                positions=positions,
                gross_exposure=gross,
                net_exposure=net_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                cash_weight=1.0 - net_exposure,
            )
        )
        audit_rows.append(
            {
                "session_date": pd.Timestamp(target.session),
                "base_gross_exposure": float(np.abs(base).sum()),
                "pre_scale_neutral_gross": pre_scale_gross,
                "neutral_gross_exposure": gross,
                "scale_multiplier": scale,
                "base_max_name_weight": float(np.abs(base).max()),
                "neutral_max_name_weight": max_abs_weight,
                "base_max_abs_sector_exposure": max(map(abs, base_sector_exposure.values())),
                "neutral_max_abs_sector_exposure": max_sector_exposure,
                "signal_names": int(signal_mask.sum()),
                "hedge_names": int(hedge_mask.sum()),
                "active_names": int(np.count_nonzero(weights)),
                "signal_gross_after_neutralization": float(np.abs(weights[signal_mask]).sum()),
                "hedge_gross_after_neutralization": float(np.abs(weights[hedge_mask]).sum()),
            }
        )
    return tuple(neutral_targets), pd.DataFrame(audit_rows)


def sector_project_targets(
    targets: Sequence[TargetPortfolio],
    *,
    sector_by_symbol: Mapping[str, str],
    gross_exposure: float = 1.0,
    single_name_cap: float = 0.25,
    tolerance: float = 1e-12,
) -> tuple[tuple[TargetPortfolio, ...], pd.DataFrame]:
    """Project targets onto equal-weight sector baskets without scaling up.

    For each session and sector, every constituent receives the sector's mean
    input target. This is the between-sector component removed by within-sector
    demeaning. A single common multiplier is applied only if needed to enforce
    the requested gross and name limits; unused capacity remains cash.
    """
    if not math.isfinite(gross_exposure) or not 0 < gross_exposure <= 1:
        raise ValueError("gross_exposure must be finite and in (0, 1]")
    if (
        not math.isfinite(single_name_cap)
        or not 0 < single_name_cap <= gross_exposure
    ):
        raise ValueError("single_name_cap must be finite and in (0, gross_exposure]")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    target_rows = tuple(targets)
    if not target_rows:
        raise ValueError("targets must contain at least one session")

    expected_symbols = tuple(position.symbol for position in target_rows[0].positions)
    if len(expected_symbols) != len(set(expected_symbols)) or not expected_symbols:
        raise ValueError("each target must contain a non-empty unique symbol population")
    missing_sector = sorted(set(expected_symbols) - set(sector_by_symbol))
    if missing_sector:
        raise ValueError(f"sector mapping missing symbols: {missing_sector}")
    sectors = {symbol: str(sector_by_symbol[symbol]).strip() for symbol in expected_symbols}
    if any(not sector for sector in sectors.values()):
        raise ValueError("sector labels must be non-empty")
    sector_members: dict[str, list[int]] = {}
    for index, symbol in enumerate(expected_symbols):
        sector_members.setdefault(sectors[symbol], []).append(index)
    if any(len(members) < 2 for members in sector_members.values()):
        raise ValueError("every represented sector must contain at least two symbols")

    projected_targets: list[TargetPortfolio] = []
    audit_rows: list[dict[str, Any]] = []
    for target in target_rows:
        symbols = tuple(position.symbol for position in target.positions)
        if symbols != expected_symbols:
            raise ValueError("every target must use the same ordered symbol population")
        base = np.asarray(
            [position.target_weight for position in target.positions], dtype=float
        )
        if not np.isfinite(base).all():
            raise ValueError("target weights must be finite")
        if abs(float(base.sum())) > tolerance:
            raise ValueError("sector projection requires dollar-neutral input targets")

        projected = np.zeros_like(base)
        sector_exposure: dict[str, float] = {}
        for sector, members in sector_members.items():
            indices = np.asarray(members, dtype=int)
            sector_exposure[sector] = float(base[indices].sum())
            projected[indices] = float(base[indices].mean())

        pre_scale_gross = float(np.abs(projected).sum())
        pre_scale_max = float(np.abs(projected).max())
        scale = 1.0
        if pre_scale_gross > gross_exposure:
            scale = min(scale, gross_exposure / pre_scale_gross)
        if pre_scale_max > single_name_cap:
            scale = min(scale, single_name_cap / pre_scale_max)
        weights = projected * scale
        weights[np.abs(weights) <= tolerance] = 0.0

        long_exposure = float(weights[weights > 0].sum())
        short_exposure = float(-weights[weights < 0].sum())
        net_exposure = long_exposure - short_exposure
        gross = long_exposure + short_exposure
        max_abs_weight = float(np.abs(weights).max())
        if (
            abs(net_exposure) > tolerance
            or gross > gross_exposure + tolerance
            or max_abs_weight > single_name_cap + tolerance
        ):
            raise RuntimeError("sector-projected target violates exposure constraints")

        positions = tuple(
            PositionTarget(
                symbol=position.symbol,
                action=float(np.sign(weight)),
                volatility=position.volatility,
                raw_weight=float(raw_weight),
                target_weight=float(weight),
                exclusion_reason=(
                    "sector_projection_peer"
                    if abs(position.target_weight) <= tolerance and abs(weight) > tolerance
                    else position.exclusion_reason
                ),
            )
            for position, raw_weight, weight in zip(
                target.positions, projected, weights, strict=True
            )
        )
        signal_mask = np.abs(base) > tolerance
        peer_mask = ~signal_mask & (np.abs(weights) > tolerance)
        active_sectors = sum(abs(value) > tolerance for value in sector_exposure.values())
        projected_targets.append(
            TargetPortfolio(
                session=target.session,
                positions=positions,
                gross_exposure=gross,
                net_exposure=net_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                cash_weight=1.0 - net_exposure,
            )
        )
        audit_rows.append(
            {
                "session_date": pd.Timestamp(target.session),
                "base_gross_exposure": float(np.abs(base).sum()),
                "projected_gross_exposure": gross,
                "pre_scale_projected_gross": pre_scale_gross,
                "scale_multiplier": scale,
                "base_max_name_weight": float(np.abs(base).max()),
                "projected_max_name_weight": max_abs_weight,
                "base_max_abs_sector_exposure": max(map(abs, sector_exposure.values())),
                "active_sectors": active_sectors,
                "signal_names": int(signal_mask.sum()),
                "projected_names": int(np.count_nonzero(weights)),
                "non_signal_peer_names": int(peer_mask.sum()),
            }
        )
    return tuple(projected_targets), pd.DataFrame(audit_rows)


def ledger_rows_to_frame(rows: Sequence[DailyLedgerRow]) -> pd.DataFrame:
    """Convert validated ledger rows into a compact analysis frame."""
    records = [
        {
            "session_date": pd.Timestamp(row.session),
            "return_end_date": pd.Timestamp(row.next_session or row.session),
            "gross_return": row.gross_return,
            "cost": row.transaction_cost,
            "net_return": row.net_return,
            "turnover": row.turnover,
            "gross_exposure": row.gross_exposure,
            "long_exposure": row.long_exposure,
            "short_exposure": row.short_exposure,
            "net_exposure": row.net_exposure,
            "active_names": row.active_names,
            "start_nav": row.start_nav_usd,
            "end_nav": row.end_nav_usd,
            "final_liquidation": row.final_liquidation,
        }
        for row in rows
    ]
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["downside_sq"] = np.minimum(frame["net_return"], 0.0) ** 2
    return frame
