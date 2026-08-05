"""Fixed sector-neutral portfolio translations for the expanded LSEG panel.

The module deliberately changes only the portfolio construction applied to the
existing Gemma ``strongest_event`` firm-open signal.  No rule is fitted to
returns.  The 44-company sector map comes from the two frozen, sector-balanced
LSEG collection configurations (three incumbents plus one breadth addition in
each of 11 sectors).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.aggregators import benjamini_hochberg
from final_experiments.lib.evaluate import (
    TradeConfig,
    annualized_sharpe,
    build_daily_portfolio,
    cross_sectional_rank_scores,
    summarize_daily_portfolio,
)

SECTOR_MEMBERS: dict[str, tuple[str, ...]] = {
    "information_technology": ("AAPL", "MSFT", "NVDA", "AVGO"),
    "communication_services": ("GOOGL", "META", "NFLX", "TMUS"),
    "consumer_discretionary": ("AMZN", "TSLA", "HD", "MCD"),
    "financials": ("JPM", "BAC", "GS", "WFC"),
    "healthcare": ("JNJ", "UNH", "LLY", "ABBV"),
    "energy": ("XOM", "CVX", "COP", "EOG"),
    "industrials": ("CAT", "BA", "GE", "RTX"),
    "consumer_staples": ("WMT", "PG", "KO", "COST"),
    "utilities": ("NEE", "DUK", "SO", "AEP"),
    "real_estate": ("PLD", "AMT", "EQIX", "WELL"),
    "materials": ("LIN", "FCX", "NEM", "SHW"),
}
SECTOR_BY_SYMBOL: dict[str, str] = {
    symbol: sector for sector, symbols in SECTOR_MEMBERS.items() for symbol in symbols
}

STRATEGY_RULES: tuple[str, ...] = (
    "global_rank",
    "sector_rank",
    "sector_extremes",
    "sector_hysteresis",
)
MULTIPLICITY_FAMILY = (
    "four return-blind strongest-event portfolio translations x h1; "
    "two-sided five-session circular-block mean test; BH-FDR q=0.05"
)


@dataclass(frozen=True)
class SectorPortfolioConfig:
    """Frozen accounting and inference choices for the four-arm family."""

    signal_col: str = "strongest_event"
    outcome: str = "raw_open_h1"
    split: str = "exploratory_full_period"
    cost_bps_per_side: float = 10.0
    min_global_names: int = 10
    min_active_sectors: int = 6
    annualization_periods: int = 252
    bootstrap_block_length: int = 5
    bootstrap_replications: int = 4_999
    random_seed: int = 20260805

    def __post_init__(self) -> None:
        if self.cost_bps_per_side < 0:
            raise ValueError("cost_bps_per_side must be non-negative")
        if self.min_global_names < 2:
            raise ValueError("min_global_names must be at least two")
        if not 1 <= self.min_active_sectors <= len(SECTOR_MEMBERS):
            raise ValueError("min_active_sectors is outside the available sector count")
        if self.bootstrap_block_length < 1:
            raise ValueError("bootstrap_block_length must be positive")
        if self.bootstrap_replications < 1:
            raise ValueError("bootstrap_replications must be positive")


def validate_sector_map(
    sector_members: Mapping[str, Sequence[str]] = SECTOR_MEMBERS,
) -> dict[str, Any]:
    """Fail closed unless the frozen map is exactly 11 disjoint groups of four."""

    groups = {str(sector): tuple(map(str, members)) for sector, members in sector_members.items()}
    symbols = [symbol for members in groups.values() for symbol in members]
    if len(groups) != 11 or any(len(members) != 4 for members in groups.values()):
        raise ValueError("sector map must contain exactly 11 sectors with four symbols each")
    if len(symbols) != len(set(symbols)):
        raise ValueError("sector map symbols must be unique")
    if set(symbols) != set(SECTOR_BY_SYMBOL):
        raise ValueError("sector map does not match the frozen 44-company universe")
    return {
        "sectors": len(groups),
        "symbols": len(symbols),
        "members_per_sector": sorted({len(members) for members in groups.values()}),
    }


def _normalise_gross(values: np.ndarray) -> np.ndarray:
    gross = float(np.abs(values).sum())
    if not math.isfinite(gross) or gross <= 0:
        return np.zeros_like(values, dtype=float)
    return values.astype(float) / gross


def _prepare_panel(panel: pd.DataFrame, cfg: SectorPortfolioConfig) -> pd.DataFrame:
    required = {"session_date", "symbol", "split", cfg.signal_col, cfg.outcome}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    frame = panel.loc[panel["split"].eq(cfg.split)].dropna(
        subset=[cfg.signal_col, cfg.outcome]
    ).copy()
    frame["symbol"] = frame["symbol"].astype(str)
    unknown = sorted(set(frame["symbol"]) - set(SECTOR_BY_SYMBOL))
    if unknown:
        raise ValueError(f"panel contains symbols outside the frozen sector map: {unknown}")
    if frame.duplicated(["symbol", "session_date"]).any():
        raise ValueError("panel contains duplicate symbol-session rows")
    frame["sector"] = frame["symbol"].map(SECTOR_BY_SYMBOL)
    if frame.empty:
        raise ValueError("panel has no eligible signal/return rows for the requested split")
    return frame.sort_values(["session_date", "sector", "symbol"], kind="mergesort")


def _eligible_sector_groups(day: pd.DataFrame, signal_col: str) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {}
    for sector, group in day.groupby("sector", sort=True):
        if len(group) < 2 or group[signal_col].nunique(dropna=True) < 2:
            continue
        groups[str(sector)] = group.sort_values("symbol", kind="mergesort")
    return groups


def _sector_rank_weight_map(
    groups: Mapping[str, pd.DataFrame], signal_col: str
) -> dict[str, float]:
    if not groups:
        return {}
    sector_budget = 1.0 / len(groups)
    weights: dict[str, float] = {}
    for group in groups.values():
        raw = cross_sectional_rank_scores(group[signal_col].to_numpy(dtype=float))
        local = sector_budget * _normalise_gross(raw)
        weights.update(
            {
                str(symbol): float(weight)
                for symbol, weight in zip(group["symbol"], local, strict=True)
                if weight != 0.0
            }
        )
    return weights


def _extreme_symbols(group: pd.DataFrame, signal_col: str) -> tuple[str, str]:
    ordered = group.sort_values(
        [signal_col, "symbol"], ascending=[False, True], kind="mergesort"
    )
    return str(ordered.iloc[0]["symbol"]), str(ordered.iloc[-1]["symbol"])


def _sector_extreme_weight_map(
    groups: Mapping[str, pd.DataFrame], signal_col: str
) -> dict[str, float]:
    if not groups:
        return {}
    sector_half_leg = 0.5 / len(groups)
    weights: dict[str, float] = {}
    for group in groups.values():
        long_symbol, short_symbol = _extreme_symbols(group, signal_col)
        weights[long_symbol] = sector_half_leg
        weights[short_symbol] = -sector_half_leg
    return weights


def _retained_extremes(
    group: pd.DataFrame,
    signal_col: str,
    previous_long: str | None,
    previous_short: str | None,
) -> tuple[str, str]:
    ordered = group.sort_values(
        [signal_col, "symbol"], ascending=[False, True], kind="mergesort"
    )
    symbols = ordered["symbol"].astype(str).tolist()
    half = int(math.ceil(len(symbols) / 2.0))
    top_half = set(symbols[:half])
    bottom_half = set(symbols[-half:])
    long_symbol = previous_long if previous_long in top_half else symbols[0]
    short_symbol = previous_short if previous_short in bottom_half else symbols[-1]
    if long_symbol == short_symbol:
        long_symbol, short_symbol = symbols[0], symbols[-1]
    return long_symbol, short_symbol


def _sector_hysteresis_weight_map(
    groups: Mapping[str, pd.DataFrame],
    signal_col: str,
    previous: dict[str, tuple[str, str]],
) -> tuple[dict[str, float], dict[str, tuple[str, str]]]:
    if not groups:
        return {}, {}
    sector_half_leg = 0.5 / len(groups)
    weights: dict[str, float] = {}
    current: dict[str, tuple[str, str]] = {}
    for sector, group in groups.items():
        prior = previous.get(sector, (None, None))
        long_symbol, short_symbol = _retained_extremes(
            group, signal_col, prior[0], prior[1]
        )
        weights[long_symbol] = sector_half_leg
        weights[short_symbol] = -sector_half_leg
        current[sector] = (long_symbol, short_symbol)
    return weights, current


def _append_liquidation(
    daily: pd.DataFrame,
    *,
    final_weights: Mapping[str, float],
    cost_bps_per_side: float,
) -> pd.DataFrame:
    out = daily.copy()
    liquidation_turnover = 0.5 * sum(abs(float(weight)) for weight in final_weights.values())
    if out.empty or liquidation_turnover <= 0:
        return out
    liquidation_cost = 2.0 * liquidation_turnover * cost_bps_per_side / 10_000.0
    last = out.index[-1]
    out.loc[last, "turnover"] += liquidation_turnover
    out.loc[last, "cost"] += liquidation_cost
    out.loc[last, "net_return"] -= liquidation_cost
    return out


def _build_sector_daily_portfolio(
    frame: pd.DataFrame,
    *,
    strategy: str,
    cfg: SectorPortfolioConfig,
) -> pd.DataFrame:
    if strategy not in {"sector_rank", "sector_extremes", "sector_hysteresis"}:
        raise ValueError(f"unsupported sector strategy: {strategy}")

    previous_weights: dict[str, float] = {}
    previous_extremes: dict[str, tuple[str, str]] = {}
    rows: list[dict[str, Any]] = []
    cost_rate = cfg.cost_bps_per_side / 10_000.0

    for session, day in frame.groupby("session_date", sort=True):
        groups = _eligible_sector_groups(day, cfg.signal_col)
        if len(groups) < cfg.min_active_sectors:
            weight_map: dict[str, float] = {}
            previous_extremes = {}
        elif strategy == "sector_rank":
            weight_map = _sector_rank_weight_map(groups, cfg.signal_col)
        elif strategy == "sector_extremes":
            weight_map = _sector_extreme_weight_map(groups, cfg.signal_col)
        else:
            weight_map, previous_extremes = _sector_hysteresis_weight_map(
                groups, cfg.signal_col, previous_extremes
            )

        names = set(previous_weights) | set(weight_map)
        turnover = 0.5 * sum(
            abs(weight_map.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
            for symbol in names
        )
        returns = day.set_index("symbol")[cfg.outcome].astype(float)
        gross = float(
            sum(weight * float(returns.get(symbol, 0.0)) for symbol, weight in weight_map.items())
        )
        cost = float(2.0 * turnover * cost_rate)
        weights = np.fromiter(weight_map.values(), dtype=float)
        rows.append(
            {
                "session_date": session,
                "split": cfg.split,
                "strategy": strategy,
                "n_names": int(len(day)),
                "n_active_sectors": int(len(groups)),
                "n_long": int((weights > 0).sum()),
                "n_short": int((weights < 0).sum()),
                "net_exposure": float(weights.sum()) if len(weights) else 0.0,
                "gross_exposure": float(np.abs(weights).sum()) if len(weights) else 0.0,
                "max_abs_weight": float(np.abs(weights).max()) if len(weights) else 0.0,
                "gross_return": gross,
                "turnover": float(turnover),
                "cost": cost,
                "net_return": gross - cost,
            }
        )
        previous_weights = weight_map

    daily = pd.DataFrame(rows)
    return _append_liquidation(
        daily,
        final_weights=previous_weights,
        cost_bps_per_side=cfg.cost_bps_per_side,
    )


def build_sector_strategy_family(
    panel: pd.DataFrame,
    *,
    config: SectorPortfolioConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the four frozen strategies from one unchanged firm-open signal."""

    cfg = config or SectorPortfolioConfig()
    validate_sector_map()
    frame = _prepare_panel(panel, cfg)
    trade_cfg = TradeConfig(
        outcome=cfg.outcome,
        position_mode="cs_rank",
        threshold=0.0,
        min_names=cfg.min_global_names,
        cost_bps_per_side=cfg.cost_bps_per_side,
        annualization_periods=cfg.annualization_periods,
        bootstrap_block_length=cfg.bootstrap_block_length,
        bootstrap_replications=cfg.bootstrap_replications,
        random_seed=cfg.random_seed,
    )
    global_daily = build_daily_portfolio(
        frame,
        cfg.signal_col,
        config=trade_cfg,
        orient=1.0,
        split=cfg.split,
    )
    final_global_weights_gross = 1.0 if not global_daily.empty and (
        int(global_daily.iloc[-1]["n_long"] + global_daily.iloc[-1]["n_short"]) > 0
    ) else 0.0
    global_daily = _append_liquidation(
        global_daily,
        final_weights={"book": final_global_weights_gross},
        cost_bps_per_side=cfg.cost_bps_per_side,
    )
    global_daily["strategy"] = "global_rank"
    active_by_session = (
        frame.groupby("session_date")["sector"].nunique().rename("n_active_sectors")
    )
    global_daily = global_daily.merge(
        active_by_session, left_on="session_date", right_index=True, how="left", validate="one_to_one"
    )
    global_daily["gross_exposure"] = np.where(
        global_daily["n_long"].add(global_daily["n_short"]).gt(0), 1.0, 0.0
    )
    global_max_weight: dict[pd.Timestamp, float] = {}
    for session, day in frame.groupby("session_date", sort=True):
        raw = cross_sectional_rank_scores(day[cfg.signal_col].to_numpy(dtype=float))
        weights = _normalise_gross(raw)
        if int(np.count_nonzero(weights)) < cfg.min_global_names:
            weights = np.zeros_like(weights)
        global_max_weight[pd.Timestamp(session)] = (
            float(np.abs(weights).max()) if len(weights) else 0.0
        )
    global_daily["max_abs_weight"] = global_daily["session_date"].map(global_max_weight)

    outputs = {"global_rank": global_daily}
    for strategy in STRATEGY_RULES[1:]:
        outputs[strategy] = _build_sector_daily_portfolio(
            frame, strategy=strategy, cfg=cfg
        )
    return outputs


def circular_block_mean_test(
    values: np.ndarray,
    *,
    block_length: int,
    replications: int,
    seed: int,
) -> dict[str, float]:
    """Percentile interval plus a centred two-sided circular-block mean test."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 or block_length >= n:
        point = float(x.mean()) if n else float("nan")
        return {
            "mean": point,
            "se": float("nan"),
            "ci_low": point,
            "ci_high": point,
            "p_two_sided": float("nan"),
        }
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(replications, int(math.ceil(n / block_length))))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets) % n
    indices = indices.reshape(replications, -1)[:, :n]
    means = x[indices].mean(axis=1)
    centered_means = (x - x.mean())[indices].mean(axis=1)
    point = float(x.mean())
    p_value = (1.0 + float(np.count_nonzero(np.abs(centered_means) >= abs(point)))) / (
        replications + 1.0
    )
    return {
        "mean": point,
        "se": float(means.std(ddof=1)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_two_sided": float(p_value),
    }


def _compounded_return(values: pd.Series) -> float:
    return float((1.0 + values.astype(float)).prod() - 1.0)


def _max_drawdown(values: pd.Series) -> float:
    wealth = (1.0 + values.astype(float)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def evaluate_sector_strategy_family(
    panel: pd.DataFrame,
    *,
    config: SectorPortfolioConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate all frozen arms and apply BH to their gross-mean tests."""

    cfg = config or SectorPortfolioConfig()
    outputs = build_sector_strategy_family(panel, config=cfg)
    summary_cfg = TradeConfig(
        outcome=cfg.outcome,
        position_mode="cs_rank",
        threshold=0.0,
        min_names=cfg.min_global_names,
        cost_bps_per_side=cfg.cost_bps_per_side,
        annualization_periods=cfg.annualization_periods,
        bootstrap_block_length=cfg.bootstrap_block_length,
        bootstrap_replications=cfg.bootstrap_replications,
        random_seed=cfg.random_seed,
    )
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGY_RULES:
        daily = outputs[strategy]
        active = daily["gross_exposure"].gt(0)
        gross_test = circular_block_mean_test(
            daily["gross_return"].to_numpy(dtype=float),
            block_length=cfg.bootstrap_block_length,
            replications=cfg.bootstrap_replications,
            seed=cfg.random_seed,
        )
        summary = summarize_daily_portfolio(daily, config=summary_cfg)
        midpoint = len(daily) // 2
        first = daily.iloc[:midpoint]
        second = daily.iloc[midpoint:]
        rows.append(
            {
                "strategy": strategy,
                **summary,
                "gross_mean_se": gross_test["se"],
                "gross_mean_ci_low": gross_test["ci_low"],
                "gross_mean_ci_high": gross_test["ci_high"],
                "gross_mean_p_two_sided": gross_test["p_two_sided"],
                "gross_total_return": _compounded_return(daily["gross_return"]),
                "net_total_return": _compounded_return(daily["net_return"]),
                "net_max_drawdown": _max_drawdown(daily["net_return"]),
                "max_abs_net_exposure": float(daily["net_exposure"].abs().max()),
                "mean_active_sectors": float(daily["n_active_sectors"].mean()),
                "active_sessions": int(active.sum()),
                "active_session_share": float(active.mean()),
                "mean_gross_exposure": float(daily["gross_exposure"].mean()),
                "mean_max_abs_weight": float(daily["max_abs_weight"].mean()),
                "maximum_abs_weight": float(daily["max_abs_weight"].max()),
                "mean_turnover_active_session": (
                    float(daily.loc[active, "turnover"].mean()) if active.any() else float("nan")
                ),
                "mean_active_sectors_when_trading": (
                    float(daily.loc[active, "n_active_sectors"].mean())
                    if active.any()
                    else float("nan")
                ),
                "first_half_mean_gross": float(first["gross_return"].mean()),
                "second_half_mean_gross": float(second["gross_return"].mean()),
                "first_half_sharpe_gross": annualized_sharpe(
                    first["gross_return"], periods_per_year=cfg.annualization_periods
                ),
                "second_half_sharpe_gross": annualized_sharpe(
                    second["gross_return"], periods_per_year=cfg.annualization_periods
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["bh_reject_gross_mean_q05"] = benjamini_hochberg(
        table["gross_mean_p_two_sided"].fillna(1.0).tolist()
    )
    table["positive_both_halves"] = table["first_half_mean_gross"].gt(0) & table[
        "second_half_mean_gross"
    ].gt(0)
    table["economically_viable_10bps"] = (
        table["sharpe_net"].gt(0)
        & table["breakeven_bps_per_side"].ge(cfg.cost_bps_per_side)
        & table["bootstrap_net_ci_low"].gt(0)
    )
    table["alpha_claim_gate"] = (
        table["bh_reject_gross_mean_q05"]
        & table["positive_both_halves"]
        & table["economically_viable_10bps"]
    )
    table["multiplicity_family"] = MULTIPLICITY_FAMILY
    table["status"] = "retrospective_exploratory_non_pooled_portfolio_translation"
    return table, outputs


def build_cost_curve(
    daily_outputs: Mapping[str, pd.DataFrame],
    *,
    cost_bps_per_side: Sequence[float] = (0.0, 1.0, 2.0, 4.0, 10.0),
    annualization_periods: int = 252,
) -> pd.DataFrame:
    """Reprice fixed weights at disclosed costs; weights never depend on cost."""

    rows: list[dict[str, float | str]] = []
    for strategy in STRATEGY_RULES:
        daily = daily_outputs[strategy]
        for bps in cost_bps_per_side:
            net = daily["gross_return"] - 2.0 * daily["turnover"] * float(bps) / 10_000.0
            rows.append(
                {
                    "strategy": strategy,
                    "cost_bps_per_side": float(bps),
                    "mean_net": float(net.mean()),
                    "sharpe_net": annualized_sharpe(
                        net, periods_per_year=annualization_periods
                    ),
                    "total_net_return": _compounded_return(net),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "MULTIPLICITY_FAMILY",
    "SECTOR_BY_SYMBOL",
    "SECTOR_MEMBERS",
    "STRATEGY_RULES",
    "SectorPortfolioConfig",
    "build_cost_curve",
    "build_sector_strategy_family",
    "circular_block_mean_test",
    "evaluate_sector_strategy_family",
    "validate_sector_map",
]
