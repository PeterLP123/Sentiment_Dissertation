"""Independent per-stock evaluation of one frozen strategy-research signal.

This diagnostic deliberately removes portfolio construction. Each symbol is a
separate account whose target exposure is the already-frozen state action in
``[-1, 1]``. Results are never summed into a synthetic portfolio.
"""

from __future__ import annotations

import csv
import html
import io
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from ..artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from .diagnostics import calculate_portfolio_metrics
from .inference import paired_block_bootstrap
from .ledger import evaluation_rows, run_open_to_open_ledger
from .market import OpenToOpenReturn, calculate_open_to_open_returns, load_adjusted_opens_csv
from .portfolio import PositionTarget, TargetPortfolio


class SingleStockExperimentError(RuntimeError):
    """Raised when a per-stock experiment cannot be reproduced safely."""


@dataclass(frozen=True)
class SingleStockRule:
    evaluation_start: str
    cost_bps_per_side: float = 10.0
    initial_nav_usd: float = 100_000.0
    block_length: int = 5
    bootstrap_replications: int = 2_000
    seed: int = 20_260_715

    def __post_init__(self) -> None:
        if not self.evaluation_start:
            raise ValueError("evaluation_start cannot be blank")
        if not math.isfinite(self.cost_bps_per_side) or self.cost_bps_per_side < 0:
            raise ValueError("cost_bps_per_side must be finite and non-negative")
        if not math.isfinite(self.initial_nav_usd) or self.initial_nav_usd <= 0:
            raise ValueError("initial_nav_usd must be finite and positive")
        if self.block_length < 1 or self.bootstrap_replications < 1:
            raise ValueError("bootstrap settings must be positive")


@dataclass(frozen=True)
class SingleStockResult:
    symbol: str
    observations: int
    active_days: int
    long_days: int
    short_days: int
    cumulative_gross_return: float
    cumulative_net_return: float
    buy_and_hold_return: float
    excess_vs_buy_and_hold: float
    after_cost_sharpe: float | None
    maximum_drawdown: float
    average_turnover: float
    total_turnover: float
    total_transaction_cost_usd: float
    profitable_active_day_rate: float
    directional_hit_rate: float
    final_nav_usd: float
    mean_daily_return: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_direction: str


@dataclass(frozen=True)
class SingleStockExperimentResult:
    experiment_id: str
    output_dir: Path
    summary_path: Path
    daily_path: Path
    report_path: Path
    manifest_path: Path
    return_figure_path: Path
    benchmark_figure_path: Path
    rows: tuple[SingleStockResult, ...]
    reused: bool


def _target(session: str, symbol: str, action: float) -> TargetPortfolio:
    if not math.isfinite(action) or not -1 <= action <= 1:
        raise SingleStockExperimentError(f"action for {symbol} on {session} is outside [-1, 1]")
    position = PositionTarget(
        symbol=symbol,
        action=action,
        volatility=None,
        raw_weight=action,
        target_weight=action,
    )
    long_exposure = max(action, 0.0)
    short_exposure = max(-action, 0.0)
    return TargetPortfolio(
        session=session,
        positions=(position,),
        gross_exposure=abs(action),
        net_exposure=action,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        cash_weight=1 - action,
    )


def evaluate_single_stock_actions(
    actions: Sequence[Mapping[str, Any]],
    returns: Sequence[OpenToOpenReturn],
    rule: SingleStockRule,
) -> tuple[tuple[SingleStockResult, ...], tuple[dict[str, Any], ...]]:
    """Evaluate the same direct-action rule in one standalone account per stock."""

    action_by_symbol: dict[str, list[tuple[str, float]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for raw in actions:
        symbol = str(raw.get("symbol") or "").strip().upper()
        session = str(raw.get("session") or "").strip()
        try:
            action = float(raw["action"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SingleStockExperimentError("action rows require numeric action values") from exc
        key = (session, symbol)
        if not symbol or not session or key in seen:
            raise SingleStockExperimentError(f"invalid or duplicate action key: {key}")
        seen.add(key)
        action_by_symbol[symbol].append((session, action))

    returns_by_symbol: dict[str, list[OpenToOpenReturn]] = defaultdict(list)
    for return_row in returns:
        returns_by_symbol[return_row.symbol].append(return_row)
    result_rows: list[SingleStockResult] = []
    daily_output: list[dict[str, Any]] = []
    for symbol in sorted(action_by_symbol):
        symbol_actions = sorted(action_by_symbol[symbol])
        sessions = [session for session, _ in symbol_actions]
        if len(sessions) != len(set(sessions)) or sessions != sorted(sessions):
            raise SingleStockExperimentError(f"actions are not a unique chronological spine for {symbol}")
        if rule.evaluation_start not in sessions:
            raise SingleStockExperimentError(f"evaluation boundary {rule.evaluation_start} is absent for {symbol}")
        symbol_returns = sorted(returns_by_symbol.get(symbol, ()), key=lambda row: row.session)
        return_lookup = {row.session: row for row in symbol_returns}
        missing = [session for session in sessions if session not in return_lookup]
        if missing:
            raise SingleStockExperimentError(f"missing executable returns for {symbol}: {missing[:3]}")

        targets = [_target(session, symbol, action) for session, action in symbol_actions]
        ledger = run_open_to_open_ledger(
            targets,
            symbol_returns,
            cost_rate_per_side=rule.cost_bps_per_side / 10_000,
            initial_nav_usd=rule.initial_nav_usd,
            accounting_reset_session=rule.evaluation_start,
            force_final_liquidation=True,
        )
        evaluated = evaluation_rows(ledger, rule.evaluation_start)
        metrics = calculate_portfolio_metrics(evaluated)
        market_intervals = [row for row in evaluated if not row.final_liquidation]
        buy_and_hold = math.prod(1 + return_lookup[row.session].value for row in market_intervals) - 1
        active_intervals = [row for row in market_intervals if row.active_names]
        long_days = sum(dict(row.target_weights).get(symbol, 0.0) > 0 for row in market_intervals)
        short_days = sum(dict(row.target_weights).get(symbol, 0.0) < 0 for row in market_intervals)
        directional_hits = sum(row.gross_return > 0 for row in active_intervals)
        total_turnover = sum(row.turnover for row in evaluated)
        daily_returns = {row.session: row.net_return for row in evaluated}
        cash_returns = {row.session: 0.0 for row in evaluated}
        inference = paired_block_bootstrap(
            daily_returns,
            cash_returns,
            block_length=min(rule.block_length, len(evaluated)),
            replications=rule.bootstrap_replications,
            seed=rule.seed,
        )
        result_rows.append(
            SingleStockResult(
                symbol=symbol,
                observations=metrics.observations,
                active_days=metrics.active_portfolio_days,
                long_days=long_days,
                short_days=short_days,
                cumulative_gross_return=metrics.cumulative_gross_return,
                cumulative_net_return=metrics.cumulative_net_return,
                buy_and_hold_return=buy_and_hold,
                excess_vs_buy_and_hold=metrics.cumulative_net_return - buy_and_hold,
                after_cost_sharpe=metrics.after_cost_sharpe,
                maximum_drawdown=metrics.maximum_drawdown,
                average_turnover=metrics.average_turnover,
                total_turnover=total_turnover,
                total_transaction_cost_usd=metrics.total_transaction_cost_usd,
                profitable_active_day_rate=metrics.profitable_active_day_rate,
                directional_hit_rate=directional_hits / len(active_intervals) if active_intervals else 0.0,
                final_nav_usd=evaluated[-1].end_nav_usd,
                mean_daily_return=sum(row.net_return for row in evaluated) / len(evaluated),
                bootstrap_ci_low=inference.confidence_interval_low,
                bootstrap_ci_high=inference.confidence_interval_high,
                bootstrap_direction=inference.confidence_direction,
            )
        )
        for ledger_row in evaluated:
            weight = dict(ledger_row.target_weights).get(symbol, 0.0)
            daily_output.append(
                {
                    "symbol": symbol,
                    "session": ledger_row.session,
                    "next_session": ledger_row.next_session,
                    "target_exposure": weight,
                    "gross_return": ledger_row.gross_return,
                    "transaction_cost": ledger_row.transaction_cost,
                    "net_return": ledger_row.net_return,
                    "turnover": ledger_row.turnover,
                    "start_nav_usd": ledger_row.start_nav_usd,
                    "end_nav_usd": ledger_row.end_nav_usd,
                    "final_liquidation": ledger_row.final_liquidation,
                }
            )
    daily_output.sort(key=lambda row: (row["session"], row["symbol"]))
    return tuple(result_rows), tuple(daily_output)


def run_single_stock_experiment(
    run_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    price_panel: str | Path | None = None,
) -> SingleStockExperimentResult:
    """Run and immutably materialize the standalone-stock diagnostic."""

    root = Path(run_dir)
    report_manifest_path = root / "manifests" / "report.json"
    actions_path = root / "state" / "actions.jsonl"
    selected_path = root / "tuning" / "selected_config.json"
    for required in (report_manifest_path, actions_path, selected_path):
        if not required.is_file():
            raise SingleStockExperimentError(f"required completed-run artifact is missing: {required}")
    report_manifest = read_json(report_manifest_path)
    if report_manifest.get("status") != "completed":
        raise SingleStockExperimentError("source strategy report stage is not completed")
    config = report_manifest.get("config")
    if not isinstance(config, dict):
        raise SingleStockExperimentError("report manifest has no normalized configuration")
    run_config = config.get("run")
    execution = config.get("execution")
    prices = config.get("prices")
    if not isinstance(run_config, dict) or not isinstance(execution, dict) or not isinstance(prices, dict):
        raise SingleStockExperimentError("report manifest is missing run, execution, or price settings")

    panel_path = Path(price_panel) if price_panel is not None else Path(str(prices["panel_path"]))
    if not panel_path.is_file():
        raise SingleStockExperimentError(f"price panel is missing: {panel_path}")
    rule = SingleStockRule(
        evaluation_start=str(run_config["evaluation_start"]),
        cost_bps_per_side=float(execution["cost_bps_per_side"]),
        block_length=int(config["inference"]["block_length_sessions"]),
        bootstrap_replications=int(config["inference"]["replications"]),
        seed=int(config["inference"]["seed"]),
    )
    identity_payload = {
        "schema_version": 1,
        "source_run_id": report_manifest.get("run_id", root.name),
        "source_run_identity": report_manifest.get("run_identity_sha256"),
        "actions_sha256": sha256_file(actions_path),
        "selected_config_sha256": sha256_file(selected_path),
        "price_panel_sha256": sha256_file(panel_path),
        "rule": asdict(rule),
        "target_rule": "standalone account target exposure equals frozen state action",
        "excluded_portfolio_steps": [
            "volatility scaling",
            "gross normalization",
            "single-name portfolio cap",
            "net-exposure correction",
            "cross-stock capital allocation",
        ],
    }
    identity_hash = sha256_text(canonical_json(identity_payload))
    experiment_id = f"direct-action-{identity_hash[:12]}"
    output_dir = Path(output_root) if output_root is not None else root / "single_stock" / experiment_id
    summary_path = output_dir / "stock_summary.csv"
    daily_path = output_dir / "daily_returns.jsonl"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"
    return_figure_path = output_dir / "figures" / "net_return_by_stock.svg"
    benchmark_figure_path = output_dir / "figures" / "strategy_vs_buy_and_hold.svg"
    expected_outputs = (summary_path, daily_path, report_path, return_figure_path, benchmark_figure_path)

    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("identity_sha256") != identity_hash:
            raise SingleStockExperimentError(f"existing experiment identity mismatch: {output_dir}")
        for output in expected_outputs:
            expected_hash = existing.get("outputs", {}).get(output.name, {}).get("sha256")
            if not output.is_file() or sha256_file(output) != expected_hash:
                raise SingleStockExperimentError(f"completed experiment output is missing or changed: {output}")
        rows = tuple(SingleStockResult(**row) for row in existing["results"])
        return SingleStockExperimentResult(
            experiment_id,
            output_dir,
            summary_path,
            daily_path,
            report_path,
            manifest_path,
            return_figure_path,
            benchmark_figure_path,
            rows,
            True,
        )

    actions = read_jsonl(actions_path)
    adjusted_opens = load_adjusted_opens_csv(panel_path)
    returns = calculate_open_to_open_returns(adjusted_opens)
    rows, daily = evaluate_single_stock_actions(actions, returns, rule)
    output_dir.mkdir(parents=True, exist_ok=False)
    return_figure_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(summary_path, _summary_csv(rows))
    atomic_write_jsonl(daily_path, daily)
    atomic_write_text(return_figure_path, _return_bar_svg(rows))
    atomic_write_text(benchmark_figure_path, _benchmark_scatter_svg(rows))
    atomic_write_text(report_path, _render_report(experiment_id, rows, rule, return_figure_path, benchmark_figure_path))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": experiment_id,
        "identity_sha256": identity_hash,
        "identity": identity_payload,
        "interpretation": "standalone stock accounts; results are not an investable portfolio and must not be summed",
        "results": [asdict(row) for row in rows],
        "outputs": {
            path.name: {"path": path.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in expected_outputs
        },
    }
    atomic_write_json(manifest_path, manifest)
    return SingleStockExperimentResult(
        experiment_id,
        output_dir,
        summary_path,
        daily_path,
        report_path,
        manifest_path,
        return_figure_path,
        benchmark_figure_path,
        rows,
        False,
    )


def _summary_csv(rows: Sequence[SingleStockResult]) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    return buffer.getvalue()


def _render_report(
    experiment_id: str,
    rows: Sequence[SingleStockResult],
    rule: SingleStockRule,
    return_figure: Path,
    benchmark_figure: Path,
) -> str:
    ordered = sorted(rows, key=lambda row: (-row.cumulative_net_return, row.symbol))
    positive = sum(row.cumulative_net_return > 0 for row in rows)
    positive_ci = sum(row.bootstrap_direction == "positive" for row in rows)
    negative_ci = sum(row.bootstrap_direction == "negative" for row in rows)
    median_return = median(row.cumulative_net_return for row in rows)
    median_excess = median(row.excess_vs_buy_and_hold for row in rows)
    lines = [
        f"# Independent stock strategy test: {experiment_id}",
        "",
        "> Each ticker is a separate standalone account. These results are not a portfolio and are never summed.",
        "",
        "## Frozen rule",
        "",
        "- Signal: the completed run's frozen 10-session decay-with-reset action.",
        "- Target exposure: action directly, bounded from -100% short to +100% long.",
        "- Removed: volatility scaling, gross normalization, name caps, net correction, and cross-stock allocation.",
        f"- Evaluation: `{rule.evaluation_start}` through final liquidation; costs `{rule.cost_bps_per_side:g}` bps per side.",
        f"- Standalone illustrative NAV: `${rule.initial_nav_usd:,.0f}` per stock.",
        "",
        "## Aggregate diagnostic across separate tests",
        "",
        f"- Positive after-cost return: **{positive}/{len(rows)} stocks**.",
        f"- Median after-cost return: **{median_return:.2%}**.",
        f"- Median excess versus each stock's buy-and-hold return: **{median_excess:.2%}**.",
        (
            f"- Bootstrap interval entirely above/below cash: **{positive_ci} positive / {negative_ci} negative**; "
            "these 33 intervals are unadjusted for multiple comparisons."
        ),
        "",
        f"![After-cost return by stock]({return_figure.relative_to(return_figure.parents[1]).as_posix()})",
        "",
        f"![Strategy versus buy and hold]({benchmark_figure.relative_to(benchmark_figure.parents[1]).as_posix()})",
        "",
        "## Stock-by-stock observations",
        "",
        (
            "| Stock | Net return | Gross return | Buy & hold | Excess vs B&H | Sharpe | Max DD | Turnover | "
            "Active days | Direction hit | Bootstrap vs cash |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ordered:
        sharpe = "n/a" if row.after_cost_sharpe is None else f"{row.after_cost_sharpe:.2f}"
        lines.append(
            f"| {row.symbol} | {row.cumulative_net_return:.2%} | {row.cumulative_gross_return:.2%} | "
            f"{row.buy_and_hold_return:.2%} | {row.excess_vs_buy_and_hold:.2%} | {sharpe} | "
            f"{row.maximum_drawdown:.2%} | {row.total_turnover:.2f}x | {row.active_days} | "
            f"{row.directional_hit_rate:.1%} | {row.bootstrap_direction} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- This is a post-evaluation diagnostic using a strategy selected before this stock-level view; "
                "it must not be used to drop losing stocks or retain winners."
            ),
            "- Stocks share market conditions and news processes, so the 33 outcomes are not independent replications.",
            (
                "- The evaluation is short. Per-stock Sharpe ratios and bootstrap intervals are unstable, "
                "and the intervals have no multiple-testing correction."
            ),
            (
                "- Prices are split-adjusted but not dividend-adjusted; borrow, financing, liquidity, capacity, "
                "and market impact are unmodelled."
            ),
            (
                "- The correct use is to diagnose heterogeneity and decide what to test on a new untouched block, "
                "not to claim profitable alpha."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _return_bar_svg(rows: Sequence[SingleStockResult]) -> str:
    ordered = sorted(rows, key=lambda row: (row.cumulative_net_return, row.symbol))
    width, height = 1100, 760
    left, right, top, bottom = 90, 35, 55, 90
    plot_width, plot_height = width - left - right, height - top - bottom
    low = min((row.cumulative_net_return for row in ordered), default=-0.01)
    high = max((row.cumulative_net_return for row in ordered), default=0.01)
    low, high = min(low, 0.0), max(high, 0.0)
    span = max(high - low, 0.01)
    zero_y = top + plot_height * high / span
    bar_width = plot_width / max(1, len(ordered)) * 0.72
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="20">Standalone after-cost return by stock</text>',
        (
            f'<text x="{left}" y="47" font-family="sans-serif" font-size="12" fill="#555">'
            "Direct frozen action; May 21 to July 1, 2026; 10 bps per side</text>"
        ),
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}" stroke="#333"/>',
    ]
    for index, row in enumerate(ordered):
        x = left + (index + 0.5) * plot_width / len(ordered)
        value_y = top + plot_height * (high - row.cumulative_net_return) / span
        y = min(zero_y, value_y)
        bar_height = max(1.0, abs(value_y - zero_y))
        fill = "#2864a8" if row.cumulative_net_return >= 0 else "#e09045"
        pieces.append(f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{fill}"/>')
        pieces.append(
            f'<text x="{x:.2f}" y="{height - bottom + 16}" transform="rotate(60 {x:.2f} {height - bottom + 16})" '
            f'font-family="sans-serif" font-size="11">{html.escape(row.symbol)}</text>'
        )
    pieces.extend(
        [
            f'<text x="10" y="{top + 8}" font-family="sans-serif" font-size="12">{high:.1%}</text>',
            f'<text x="10" y="{height - bottom}" font-family="sans-serif" font-size="12">{low:.1%}</text>',
            "</svg>\n",
        ]
    )
    return "\n".join(pieces)


def _benchmark_scatter_svg(rows: Sequence[SingleStockResult]) -> str:
    width, height = 900, 700
    left, right, top, bottom = 80, 35, 55, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [value for row in rows for value in (row.buy_and_hold_return, row.cumulative_net_return)] or [-0.01, 0.01]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.08, 0.01)
    low, high = low - padding, high + padding
    span = high - low

    def x(value: float) -> float:
        return left + plot_width * (value - low) / span

    def y(value: float) -> float:
        return top + plot_height * (high - value) / span

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="20">Standalone strategy versus buy and hold</text>',
        (
            f'<text x="{left}" y="47" font-family="sans-serif" font-size="12" fill="#555">'
            "Points above the dashed diagonal outperformed their own stock benchmark</text>"
        ),
        f'<line x1="{x(low):.2f}" y1="{y(low):.2f}" x2="{x(high):.2f}" y2="{y(high):.2f}" stroke="#777" stroke-dasharray="5 5"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222"/>',
    ]
    for row in rows:
        cx, cy = x(row.buy_and_hold_return), y(row.cumulative_net_return)
        fill = "#2864a8" if row.excess_vs_buy_and_hold >= 0 else "#e09045"
        pieces.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" fill="{fill}" stroke="white" stroke-width="1"/>')
        pieces.append(f'<text x="{cx + 7:.2f}" y="{cy - 5:.2f}" font-family="sans-serif" font-size="9">{html.escape(row.symbol)}</text>')
    pieces.extend(
        [
            (
                f'<text x="{left + plot_width / 2:.2f}" y="{height - 18}" text-anchor="middle" '
                'font-family="sans-serif" font-size="13">Buy-and-hold return</text>'
            ),
            (
                f'<text x="18" y="{top + plot_height / 2:.2f}" '
                f'transform="rotate(-90 18 {top + plot_height / 2:.2f})" text-anchor="middle" '
                'font-family="sans-serif" font-size="13">Strategy after-cost return</text>'
            ),
            "</svg>\n",
        ]
    )
    return "\n".join(pieces)
