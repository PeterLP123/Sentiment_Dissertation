"""Exact market- and sector-residual targets for a fixed long selection path."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from sentiment_benchmark.strategy_research.portfolio import (
    PositionTarget,
    TargetPortfolio,
)

RESIDUAL_MODES = ("market", "sector")


def build_exposure_residual_targets(
    selected_long_targets: Sequence[TargetPortfolio],
    *,
    mode: str,
    sector_by_symbol: Mapping[str, str] | None = None,
    tolerance: float = 1e-12,
) -> tuple[tuple[TargetPortfolio, ...], pd.DataFrame]:
    """Subtract a matched equal-weight exposure control from fixed long targets.

    ``market`` distributes each session's selected-long notional equally over
    the full symbol population. ``sector`` distributes each sector's selected
    notional equally over that sector's members. Signal and hedge weights are
    netted by symbol before the returned targets are formed.
    """
    if mode not in RESIDUAL_MODES:
        raise ValueError(f"mode must be one of {RESIDUAL_MODES}")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    source_targets = tuple(selected_long_targets)
    if not source_targets:
        raise ValueError("selected_long_targets must not be empty")

    symbols = tuple(position.symbol for position in source_targets[0].positions)
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("targets must contain one ordered position per unique symbol")
    sessions = tuple(target.session for target in source_targets)
    if len(sessions) != len(set(sessions)) or tuple(sorted(sessions)) != sessions:
        raise ValueError("target sessions must be unique and chronological")

    sector_labels: dict[str, str] = {}
    sector_members: dict[str, np.ndarray] = {}
    if mode == "sector":
        if sector_by_symbol is None:
            raise ValueError("sector_by_symbol is required for sector residuals")
        missing = sorted(set(symbols) - set(sector_by_symbol))
        if missing:
            raise ValueError(f"sector mapping missing symbols: {missing}")
        sector_labels = {symbol: str(sector_by_symbol[symbol]).strip() for symbol in symbols}
        if any(not label for label in sector_labels.values()):
            raise ValueError("sector labels must be non-empty")
        for sector in sorted(set(sector_labels.values())):
            indices = np.asarray(
                [index for index, symbol in enumerate(symbols) if sector_labels[symbol] == sector],
                dtype=int,
            )
            if len(indices) < 2:
                raise ValueError("every represented sector must contain at least two symbols")
            sector_members[sector] = indices

    residual_targets: list[TargetPortfolio] = []
    audit_rows: list[dict[str, object]] = []
    for target in source_targets:
        target_symbols = tuple(position.symbol for position in target.positions)
        if target_symbols != symbols:
            raise ValueError("every target must use the same ordered symbol population")
        selected = np.asarray([position.target_weight for position in target.positions], dtype=float)
        if not np.isfinite(selected).all():
            raise ValueError("selected-long weights must be finite")
        if (selected < -tolerance).any():
            raise ValueError("selected-long targets cannot contain short weights")
        selected[np.abs(selected) <= tolerance] = 0.0
        selected_exposure = float(selected.sum())

        control = np.zeros_like(selected)
        if mode == "market":
            control[:] = selected_exposure / len(symbols)
        else:
            for indices in sector_members.values():
                control[indices] = float(selected[indices].sum()) / len(indices)

        residual = selected - control
        residual[np.abs(residual) <= tolerance] = 0.0
        control_exposure = float(control.sum())
        net_exposure = float(residual.sum())
        if abs(control_exposure - selected_exposure) > tolerance:
            raise RuntimeError("control does not match selected-long exposure")
        if abs(net_exposure) > tolerance:
            raise RuntimeError("residual target is not dollar neutral")

        sector_error = 0.0
        if mode == "sector":
            sector_error = max(abs(float(residual[indices].sum())) for indices in sector_members.values())
            if sector_error > tolerance:
                raise RuntimeError("sector residual is not sector neutral")

        long_exposure = float(residual[residual > 0].sum())
        short_exposure = float(-residual[residual < 0].sum())
        gross_exposure = long_exposure + short_exposure
        positions = tuple(
            PositionTarget(
                symbol=symbol,
                action=float(np.sign(weight)),
                volatility=position.volatility,
                raw_weight=float(weight),
                target_weight=float(weight),
                exclusion_reason=(f"{mode}_equal_weight_hedge" if selected_weight == 0.0 and weight != 0.0 else position.exclusion_reason),
            )
            for symbol, position, selected_weight, weight in zip(symbols, target.positions, selected, residual, strict=True)
        )
        residual_targets.append(
            TargetPortfolio(
                session=target.session,
                positions=positions,
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                cash_weight=1.0 - net_exposure,
            )
        )
        audit_rows.append(
            {
                "session_date": pd.Timestamp(target.session),
                "mode": mode,
                "selected_long_exposure": selected_exposure,
                "matched_control_exposure": control_exposure,
                "exposure_match_error": abs(control_exposure - selected_exposure),
                "residual_gross_exposure": gross_exposure,
                "residual_long_exposure": long_exposure,
                "residual_short_exposure": short_exposure,
                "residual_net_exposure": net_exposure,
                "max_abs_name_weight": float(np.abs(residual).max()),
                "max_abs_sector_exposure": sector_error,
                "selected_names": int(np.count_nonzero(selected)),
                "hedge_only_names": int(np.count_nonzero((selected == 0.0) & (residual != 0.0))),
                "active_names": int(np.count_nonzero(residual)),
            }
        )

    return tuple(residual_targets), pd.DataFrame(audit_rows)
