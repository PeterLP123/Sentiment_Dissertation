"""Development-gated persistent FinBERT strategy for recent Reuters news.

The module deliberately separates candidate selection from evaluation.  The
development command sees only pre-boundary returns.  Evaluation accepts the
immutable development protocol and applies the selected quantile and holding
period without return-driven changes.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifact_io import canonical_json, read_json, sha256_file, sha256_text
from .baseline.pipeline import _metrics_payload, _statistical_rows
from .inference import paired_block_bootstrap
from .ledger import DailyLedgerRow, run_open_to_open_ledger
from .market import OpenToOpenReturn, calculate_open_to_open_returns, load_adjusted_opens_csv
from .portfolio import PositionTarget, TargetPortfolio


class PersistentFinbertError(ValueError):
    """Raised when the frozen persistence experiment contract is violated."""


@dataclass(frozen=True)
class FirmSessionScore:
    session: str
    symbol: str
    score: float
    event_count: int


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    run_id: str
    sample_status: str
    development_boundary: str
    robustness_boundary: str
    development_signals: Path
    development_prices: Path
    development_prices_manifest: Path
    robustness_signals: Path
    robustness_prices: Path
    robustness_prices_manifest: Path
    threshold_quantiles: tuple[float, ...]
    holding_sessions: tuple[int, ...]
    minimum_names_per_side: int
    minimum_development_active_days: int
    minimum_positive_subperiods: int
    gross_exposure: float
    initial_nav_usd: float
    cost_bps_per_side: float
    cost_grid_bps: tuple[float, ...]
    block_length_sessions: int
    replications: int
    seed: int
    config_sha256: str


@dataclass(frozen=True)
class PeriodRun:
    metrics: dict[str, Any]
    ledger: tuple[DailyLedgerRow, ...]
    sessions: tuple[str, ...]
    threshold: float
    holding_sessions: int


def _implementation_sha256() -> str:
    root = Path(__file__).parent
    files = (
        Path(__file__),
        root / "inference.py",
        root / "ledger.py",
        root / "market.py",
        root / "portfolio.py",
        root / "baseline" / "pipeline.py",
    )
    return sha256_text("\n".join(f"{path.name}:{sha256_file(path)}" for path in files))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistentFinbertError(message)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    raw = source.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    _require(set(data) == {"run", "development", "robustness", "signal", "portfolio", "inference"}, "unexpected config tables")
    run = data["run"]
    development = data["development"]
    robustness = data["robustness"]
    signal = data["signal"]
    portfolio = data["portfolio"]
    inference = data["inference"]
    quantiles = tuple(float(value) for value in signal["threshold_quantiles"])
    holds = tuple(int(value) for value in signal["holding_sessions"])
    costs = tuple(float(value) for value in portfolio["cost_grid_bps"])
    _require(run["sample_status"] == "previously_explored", "recent corpora must not be presented as untouched")
    _require(quantiles == (0.0, 0.5, 0.7), "threshold-quantile grid is frozen")
    _require(holds == (1, 3, 5, 7), "holding grid is frozen")
    _require(int(signal["minimum_names_per_side"]) == 2, "two names per side are required")
    _require(float(portfolio["gross_exposure"]) == 1.0, "gross exposure is frozen at one")
    _require(float(portfolio["cost_bps_per_side"]) == 10.0, "headline cost is frozen at 10 bps")
    _require(costs == (5.0, 10.0, 20.0), "cost grid is frozen")
    _require(int(inference["block_length_sessions"]) == 5, "bootstrap block is frozen at five sessions")
    _require(int(inference["replications"]) == 2000, "bootstrap replications are frozen")
    return ExperimentConfig(
        path=source,
        run_id=str(run["id"]),
        sample_status=str(run["sample_status"]),
        development_boundary=str(run["development_boundary"]),
        robustness_boundary=str(run["robustness_boundary"]),
        development_signals=Path(development["signals_path"]),
        development_prices=Path(development["prices_path"]),
        development_prices_manifest=Path(development["prices_manifest_path"]),
        robustness_signals=Path(robustness["signals_path"]),
        robustness_prices=Path(robustness["prices_path"]),
        robustness_prices_manifest=Path(robustness["prices_manifest_path"]),
        threshold_quantiles=quantiles,
        holding_sessions=holds,
        minimum_names_per_side=2,
        minimum_development_active_days=int(signal["minimum_development_active_days"]),
        minimum_positive_subperiods=int(signal["minimum_positive_subperiods"]),
        gross_exposure=1.0,
        initial_nav_usd=float(portfolio["initial_nav_usd"]),
        cost_bps_per_side=10.0,
        cost_grid_bps=costs,
        block_length_sessions=5,
        replications=2000,
        seed=int(inference["seed"]),
        config_sha256=sha256_text(raw.decode("utf-8")),
    )


def load_firm_session_scores(path: str | Path) -> tuple[FirmSessionScore, ...]:
    loaded: list[FirmSessionScore] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("scorer") != "finbert":
                continue
            score = float(row["mean_continuous_score"])
            key = (str(row["session"]), str(row["symbol"]))
            _require(key not in seen, f"duplicate firm-session score at line {line_number}: {key}")
            _require(math.isfinite(score) and -1 <= score <= 1, f"invalid FinBERT score at line {line_number}")
            seen.add(key)
            loaded.append(FirmSessionScore(key[0], key[1], score, int(row["event_count"])))
    _require(bool(loaded), "no FinBERT firm-session scores loaded")
    loaded.sort(key=lambda row: (row.session, row.symbol))
    return tuple(loaded)


def _threshold(scores: Sequence[FirmSessionScore], sessions: set[str], quantile: float) -> float:
    _require(0 <= quantile < 1, "threshold quantile must be in [0, 1)")
    if quantile == 0:
        return 0.0
    values = [abs(row.score) for row in scores if row.session in sessions]
    _require(bool(values), "no development scores are available for threshold calibration")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _sessions_and_returns(prices_path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[OpenToOpenReturn, ...]]:
    opens = load_adjusted_opens_csv(prices_path)
    returns = tuple(calculate_open_to_open_returns(opens))
    symbols = tuple(sorted({row.symbol for row in opens}))
    sessions = tuple(sorted({row.session for row in returns}))
    _require(bool(symbols) and bool(sessions), "price panel has no executable intervals")
    return symbols, sessions, returns


def build_persistent_targets(
    scores: Sequence[FirmSessionScore],
    *,
    sessions: Sequence[str],
    symbols: Sequence[str],
    threshold: float,
    holding_sessions: int,
    minimum_names_per_side: int,
    gross_exposure: float,
) -> tuple[TargetPortfolio, ...]:
    """Carry each eligible score for H open-to-open intervals, then expire it.

    A new score for a symbol always supersedes the older state.  If its absolute
    value is below the frozen gate it explicitly flattens that symbol.  Current
    session scores are eligible because upstream event construction maps news to
    the first open strictly after timestamp plus processing buffer.
    """

    _require(holding_sessions > 0, "holding_sessions must be positive")
    _require(threshold >= 0, "threshold must be non-negative")
    session_tuple = tuple(sessions)
    _require(session_tuple == tuple(sorted(set(session_tuple))), "sessions must be sorted and unique")
    symbol_tuple = tuple(sorted(set(symbols)))
    by_session: dict[str, list[FirmSessionScore]] = defaultdict(list)
    for row in scores:
        if row.symbol in symbol_tuple and row.session in set(session_tuple):
            by_session[row.session].append(row)
    state: dict[str, tuple[int, int]] = {}
    targets: list[TargetPortfolio] = []
    for index, session in enumerate(session_tuple):
        state = {symbol: value for symbol, value in state.items() if index < value[1]}
        for row in sorted(by_session.get(session, ()), key=lambda value: value.symbol):
            direction = 1 if row.score > 0 else -1 if row.score < 0 else 0
            if direction and abs(row.score) >= threshold:
                state[row.symbol] = (direction, index + holding_sessions)
            else:
                state.pop(row.symbol, None)
        longs = sorted(symbol for symbol, (direction, _expiry) in state.items() if direction > 0)
        shorts = sorted(symbol for symbol, (direction, _expiry) in state.items() if direction < 0)
        eligible = len(longs) >= minimum_names_per_side and len(shorts) >= minimum_names_per_side
        weights = dict.fromkeys(symbol_tuple, 0.0)
        if eligible:
            for symbol in longs:
                weights[symbol] = gross_exposure / 2 / len(longs)
            for symbol in shorts:
                weights[symbol] = -gross_exposure / 2 / len(shorts)
        positions = tuple(
            PositionTarget(
                symbol=symbol,
                action=1.0 if weights[symbol] > 0 else -1.0 if weights[symbol] < 0 else 0.0,
                volatility=None,
                raw_weight=weights[symbol],
                target_weight=weights[symbol],
                exclusion_reason=None if eligible else "minimum_names_per_side_not_met",
            )
            for symbol in symbol_tuple
        )
        long_exposure = sum(value for value in weights.values() if value > 0)
        short_exposure = sum(-value for value in weights.values() if value < 0)
        targets.append(
            TargetPortfolio(
                session=session,
                positions=positions,
                gross_exposure=long_exposure + short_exposure,
                net_exposure=long_exposure - short_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                cash_weight=1 - (long_exposure - short_exposure),
            )
        )
    return tuple(targets)


def _run_on_sessions(
    scores: Sequence[FirmSessionScore],
    symbols: Sequence[str],
    returns: Sequence[OpenToOpenReturn],
    sessions: Sequence[str],
    *,
    threshold: float,
    holding_sessions: int,
    minimum_names_per_side: int,
    gross_exposure: float,
    cost_bps_per_side: float,
    initial_nav_usd: float,
) -> PeriodRun:
    targets = build_persistent_targets(
        scores,
        sessions=sessions,
        symbols=symbols,
        threshold=threshold,
        holding_sessions=holding_sessions,
        minimum_names_per_side=minimum_names_per_side,
        gross_exposure=gross_exposure,
    )
    ledger = tuple(
        run_open_to_open_ledger(
            targets,
            returns,
            cost_rate_per_side=cost_bps_per_side / 10_000,
            initial_nav_usd=initial_nav_usd,
            force_final_liquidation=True,
        )
    )
    return PeriodRun(
        metrics=_metrics_payload(_statistical_rows(ledger)),
        ledger=ledger,
        sessions=tuple(sessions),
        threshold=threshold,
        holding_sessions=holding_sessions,
    )


def _chunks(values: Sequence[str], count: int = 3) -> tuple[tuple[str, ...], ...]:
    indices = np.array_split(np.arange(len(values)), count)
    return tuple(tuple(values[int(index)] for index in group) for group in indices if len(group))


def develop(config: ExperimentConfig, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    signal_path = (root / config.development_signals).resolve()
    prices_path = (root / config.development_prices).resolve()
    manifest_path = (root / config.development_prices_manifest).resolve()
    scores = load_firm_session_scores(signal_path)
    symbols, sessions, returns = _sessions_and_returns(prices_path)
    development_sessions = tuple(session for session in sessions if session < config.development_boundary)
    _require(len(development_sessions) >= 60, "development period is too short")
    development_set = set(development_sessions)
    grid: list[dict[str, Any]] = []
    for quantile in config.threshold_quantiles:
        threshold = _threshold(scores, development_set, quantile)
        for hold in config.holding_sessions:
            run = _run_on_sessions(
                scores,
                symbols,
                returns,
                development_sessions,
                threshold=threshold,
                holding_sessions=hold,
                minimum_names_per_side=config.minimum_names_per_side,
                gross_exposure=config.gross_exposure,
                cost_bps_per_side=config.cost_bps_per_side,
                initial_nav_usd=config.initial_nav_usd,
            )
            subperiod_sharpes: list[float | None] = []
            for chunk in _chunks(development_sessions):
                subrun = _run_on_sessions(
                    scores,
                    symbols,
                    returns,
                    chunk,
                    threshold=threshold,
                    holding_sessions=hold,
                    minimum_names_per_side=config.minimum_names_per_side,
                    gross_exposure=config.gross_exposure,
                    cost_bps_per_side=config.cost_bps_per_side,
                    initial_nav_usd=config.initial_nav_usd,
                )
                subperiod_sharpes.append(subrun.metrics["after_cost_sharpe"])
            finite = [float(value) for value in subperiod_sharpes if value is not None and math.isfinite(float(value))]
            positive_subperiods = sum(value > 0 for value in finite)
            candidate_viable = (
                int(run.metrics["active_day_count_excluding_liquidation"]) >= config.minimum_development_active_days
                and run.metrics["after_cost_sharpe"] is not None
                and float(run.metrics["after_cost_sharpe"]) > 0
                and positive_subperiods >= config.minimum_positive_subperiods
            )
            grid.append(
                {
                    "threshold_quantile": quantile,
                    "absolute_threshold": threshold,
                    "holding_sessions": hold,
                    "development_metrics": run.metrics,
                    "subperiod_sharpes": subperiod_sharpes,
                    "median_subperiod_sharpe": float(np.median(finite)) if finite else None,
                    "worst_subperiod_sharpe": min(finite) if finite else None,
                    "positive_subperiods": positive_subperiods,
                    "viable": candidate_viable,
                }
            )
    viable_candidates = [row for row in grid if row["viable"]]
    identities = {
        "config_sha256": config.config_sha256,
        "implementation_sha256": _implementation_sha256(),
        "development_signals_sha256": sha256_file(signal_path),
        "development_prices_sha256": sha256_file(prices_path),
        "development_prices_manifest_sha256": sha256_file(manifest_path),
    }
    if not viable_candidates:
        failure_id = f"{config.run_id}-{sha256_text(canonical_json({'identities': identities, 'grid': grid}))[:12]}"
        return {
            "schema_version": 1,
            "status": "development_gate_failed",
            "protocol_id": failure_id,
            "sample_status": config.sample_status,
            "selection_data": f"sessions before {config.development_boundary}",
            "candidate_count": len(grid),
            "candidate_grid": grid,
            "selected": None,
            "input_identities": identities,
            "failure_reason": (
                "No candidate met the frozen requirements for active days, positive full-development "
                "Sharpe, and positive Sharpe in at least two of three chronological subperiods."
            ),
            "evaluation_opened": False,
        }

    def rank(row: Mapping[str, Any]) -> tuple[float, float, float, float, int, float]:
        metrics = row["development_metrics"]
        return (
            float(row["median_subperiod_sharpe"]),
            float(row["worst_subperiod_sharpe"]),
            float(metrics["after_cost_sharpe"]),
            -float(metrics["total_turnover"]),
            -int(row["holding_sessions"]),
            -float(row["threshold_quantile"]),
        )

    selected = max(viable_candidates, key=rank)
    protocol_id = f"{config.run_id}-{sha256_text(canonical_json({'identities': identities, 'selected': selected}))[:12]}"
    return {
        "schema_version": 1,
        "status": "development_rule_frozen",
        "protocol_id": protocol_id,
        "sample_status": config.sample_status,
        "selection_data": f"sessions before {config.development_boundary}",
        "candidate_count": len(grid),
        "candidate_grid": grid,
        "selected": selected,
        "input_identities": identities,
        "rule": {
            "score": "mean continuous FinBERT score per target firm and eligible execution session",
            "threshold_calibration": "absolute-score quantile on development scores only",
            "direction": "same-direction sentiment momentum",
            "persistence": "new scores supersede; weak scores flatten; otherwise expire after H intervals",
            "portfolio": "equal-weight dollar-neutral; at least two names on each side; cash otherwise",
            "execution": "split-adjusted open-to-open after upstream 15-minute availability buffer",
            "cost_bps_per_side": config.cost_bps_per_side,
        },
        "evaluation_opened": False,
    }


def _bootstrap_payload(result: Any) -> dict[str, Any]:
    return {
        "mean_daily_difference": result.mean_difference,
        "confidence_interval_low": result.ci_low,
        "confidence_interval_high": result.ci_high,
        "confidence_direction": result.direction,
        "block_length": result.block_length,
        "replications": result.replications,
        "seed": result.seed,
    }


def _daily_returns(run: PeriodRun) -> dict[str, float]:
    return {row.session: row.net_return for row in _statistical_rows(run.ledger)}


def evaluate(config: ExperimentConfig, protocol_path: str | Path, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = read_json(Path(protocol_path))
    _require(protocol.get("status") == "development_rule_frozen", "evaluation requires a frozen development protocol")
    _require(protocol.get("evaluation_opened") is False, "protocol was not sealed before evaluation")
    _require(protocol["input_identities"]["config_sha256"] == config.config_sha256, "config changed after development")
    expected_development_identities = {
        "implementation_sha256": _implementation_sha256(),
        "development_signals_sha256": sha256_file((root / config.development_signals).resolve()),
        "development_prices_sha256": sha256_file((root / config.development_prices).resolve()),
        "development_prices_manifest_sha256": sha256_file((root / config.development_prices_manifest).resolve()),
    }
    for name, expected in expected_development_identities.items():
        _require(protocol["input_identities"].get(name) == expected, f"{name} changed after development")
    selected = protocol["selected"]
    quantile = float(selected["threshold_quantile"])
    hold = int(selected["holding_sessions"])

    def load_period(signal_rel: Path, price_rel: Path, manifest_rel: Path, boundary: str) -> tuple[PeriodRun, dict[str, str]]:
        signal_path = (root / signal_rel).resolve()
        price_path = (root / price_rel).resolve()
        manifest_path = (root / manifest_rel).resolve()
        scores = load_firm_session_scores(signal_path)
        symbols, sessions, returns = _sessions_and_returns(price_path)
        calibration_sessions = {session for session in sessions if session < boundary}
        threshold = _threshold(scores, calibration_sessions, quantile)
        evaluation_sessions = tuple(session for session in sessions if session >= boundary)
        _require(len(evaluation_sessions) >= config.block_length_sessions, "evaluation period is too short")
        run = _run_on_sessions(
            scores,
            symbols,
            returns,
            evaluation_sessions,
            threshold=threshold,
            holding_sessions=hold,
            minimum_names_per_side=config.minimum_names_per_side,
            gross_exposure=config.gross_exposure,
            cost_bps_per_side=config.cost_bps_per_side,
            initial_nav_usd=config.initial_nav_usd,
        )
        return run, {
            "signals_sha256": sha256_file(signal_path),
            "prices_sha256": sha256_file(price_path),
            "prices_manifest_sha256": sha256_file(manifest_path),
        }

    midcap, midcap_ids = load_period(
        config.development_signals,
        config.development_prices,
        config.development_prices_manifest,
        config.development_boundary,
    )
    sector, sector_ids = load_period(
        config.robustness_signals,
        config.robustness_prices,
        config.robustness_prices_manifest,
        config.robustness_boundary,
    )
    cost_grid: dict[str, dict[str, Any]] = {"midcap": {}, "sector33": {}}
    for name, signal_rel, price_rel, boundary in (
        ("midcap", config.development_signals, config.development_prices, config.development_boundary),
        ("sector33", config.robustness_signals, config.robustness_prices, config.robustness_boundary),
    ):
        scores = load_firm_session_scores(root / signal_rel)
        symbols, sessions, returns = _sessions_and_returns(root / price_rel)
        calibration = {session for session in sessions if session < boundary}
        threshold = _threshold(scores, calibration, quantile)
        eval_sessions = tuple(session for session in sessions if session >= boundary)
        for cost in config.cost_grid_bps:
            cost_grid[name][f"{cost:g}"] = _run_on_sessions(
                scores,
                symbols,
                returns,
                eval_sessions,
                threshold=threshold,
                holding_sessions=hold,
                minimum_names_per_side=config.minimum_names_per_side,
                gross_exposure=config.gross_exposure,
                cost_bps_per_side=cost,
                initial_nav_usd=config.initial_nav_usd,
            ).metrics
    midcap_returns = _daily_returns(midcap)
    sector_returns = _daily_returns(sector)
    bootstraps = {
        "midcap_vs_cash": _bootstrap_payload(
            paired_block_bootstrap(
                midcap_returns,
                {session: 0.0 for session in midcap_returns},
                block_length=config.block_length_sessions,
                replications=config.replications,
                seed=config.seed,
            )
        ),
        "sector33_vs_cash": _bootstrap_payload(
            paired_block_bootstrap(
                sector_returns,
                {session: 0.0 for session in sector_returns},
                block_length=config.block_length_sessions,
                replications=config.replications,
                seed=config.seed + 1,
            )
        ),
    }
    accepted = (
        midcap.metrics["after_cost_sharpe"] is not None
        and sector.metrics["after_cost_sharpe"] is not None
        and float(midcap.metrics["after_cost_sharpe"]) > 0
        and float(sector.metrics["after_cost_sharpe"]) > 0
        and float(midcap.metrics["cumulative_net_return"]) > 0
        and float(sector.metrics["cumulative_net_return"]) > 0
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(Path(protocol_path)),
        "selected_rule": {
            "threshold_quantile": quantile,
            "holding_sessions": hold,
            "midcap_absolute_threshold": midcap.threshold,
            "sector33_absolute_threshold": sector.threshold,
        },
        "evaluation": {"midcap": midcap.metrics, "sector33": sector.metrics},
        "cost_grid": cost_grid,
        "bootstrap": bootstraps,
        "input_identities": {"midcap": midcap_ids, "sector33": sector_ids},
        "acceptance": {
            "passed": accepted,
            "rule": "positive 10-bps net cumulative return and Sharpe in both post-boundary universes",
        },
        "interpretation": {
            "confirmatory": False,
            "reason": "both recent corpora have prior exploratory exposure",
            "lseg_requests_used": 0,
        },
    }
