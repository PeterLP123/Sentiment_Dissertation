"""Deterministic technical reporting for completed strategy-research runs."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from .config import StrategyResearchConfig
from .inference import PairedBootstrapResult
from .ledger import DailyLedgerRow
from .tuning import CandidateResult

LEAKAGE_CHECKLIST: tuple[tuple[str, str, str], ...] = (
    ("future_article_revisions", "mechanically_checked", "Earliest eligible story-family x symbol revision is selected."),
    ("precise_timestamp", "mechanically_checked", "Availability uses timezone-aware version_created plus the fixed buffer."),
    ("fills_before_availability", "mechanically_checked", "Execution is the first XNYS open strictly after availability."),
    ("price_and_volatility_lookahead", "mechanically_checked", "Volatility uses intervals ending before the decision open."),
    ("evaluation_scaling", "mechanically_checked", "Final scales are fitted on development sessions only."),
    ("evaluation_parameter_selection", "mechanically_checked", "Candidates are selected only from development folds."),
    ("stock_removal_after_inspection", "mechanically_checked", "Diagnostics retain every supported stock and do not re-optimize."),
    ("random_split", "mechanically_checked", "Only chronological split and expanding folds are implemented."),
    (
        "hidden_parameter_trials",
        "evidenced_assumption",
        "The manifest records the declared grid; off-repository experiments cannot be detected.",
    ),
    ("overlapping_event_double_counting", "mechanically_checked", "One state and one position exist per symbol."),
    ("corporate_actions", "evidenced_assumption", "The price manifest must establish split-adjusted opens; dividends are excluded."),
    ("survivorship_bias", "open_decision", "The fixed 33-company universe can contain survivorship bias."),
    ("shared_model_lineage", "open_decision", "Heterogeneous scorers are not statistically independent."),
    ("training_contamination", "untested_deployment_issue", "Provider training data cannot be fully audited here."),
    ("phrasebank_domain_shift", "open_decision", "Benchmark competence may not transfer to Reuters company news."),
    ("daily_price_timing", "open_decision", "Daily adjusted opens do not model intraday market microstructure."),
    ("multiple_testing", "open_decision", "Candidate and variant comparisons require cautious interpretation."),
    ("borrow_liquidity_impact", "untested_deployment_issue", "Borrow, liquidity, financing, capacity and market impact are not modelled."),
)


def leakage_payload(config: StrategyResearchConfig) -> list[dict[str, str]]:
    checks = list(LEAKAGE_CHECKLIST)
    if config.scoring.provider == "fixture":
        model_check = (
            "model_selection_from_portfolio_pnl",
            "mechanically_checked",
            "The synthetic smoke fixture makes no model-selection claim.",
        )
    elif config.scoring.competence_evidence_path and config.scoring.competence_experiment_id:
        model_check = (
            "model_selection_from_portfolio_pnl",
            "evidenced_assumption",
            "Model selection cites run-hashed competence experiment "
            f"{config.scoring.competence_experiment_id} in {config.scoring.competence_evidence_path}.",
        )
    else:
        model_check = (
            "model_selection_from_portfolio_pnl",
            "open_decision",
            "No frozen competence evidence is configured for model selection.",
        )
    checks.insert(6, model_check)
    return [
        {"check": check, "classification": classification, "evidence": evidence}
        for check, classification, evidence in checks
    ]


def render_summary(
    config: StrategyResearchConfig,
    *,
    resolved_run_id: str,
    selected: CandidateResult,
    evaluation_metrics: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, PairedBootstrapResult],
    attrition: Mapping[str, int],
    warnings: Sequence[str],
) -> str:
    lines = [
        f"# Strategy research run: {resolved_run_id}",
        "",
        "> This is a chronological, sample-specific research backtest. It is not deployable alpha, investment advice, or causal evidence.",
        "",
        "## Frozen design",
        "",
        f"- Source: `{config.data.source}`",
        f"- Evaluation begins: `{config.run.evaluation_start}`",
        f"- Scorer: `{config.scoring.provider}/{config.scoring.model}` using `{config.scoring.prompt_id}`",
        f"- Scoring endpoint: `{config.scoring.endpoint or 'provider default locked in run identity'}`",
        f"- Competence evidence: `{config.scoring.competence_experiment_id or 'not applicable to synthetic fixture'}`",
        f"- Score scheme: `{config.scoring.scheme}`",
        f"- Execution: `{config.prices.execution_field}` / `{config.prices.return_convention}`",
        f"- Cost: `{config.execution.cost_bps_per_side:g}` bps per side",
        f"- Selected reset candidate: half-life `{selected.candidate.half_life_sessions:g}`, scale quantile "
        f"`{selected.candidate.state_scale_quantile:g}`, no-trade band `{selected.candidate.no_trade_band:g}`",
        "",
        "## Observations: evaluation metrics",
        "",
        "| Variant | Cumulative net return | After-cost Sharpe | Max drawdown | Avg turnover | Active days | Supported stocks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("cash", "last_event_fixed_hold", "additive_decay", "decay_with_reset"):
        metrics = evaluation_metrics[name]
        lines.append(
            f"| `{name}` | {_percent(metrics.get('cumulative_net_return'))} | {_number(metrics.get('after_cost_sharpe'))} | "
            f"{_percent(metrics.get('maximum_drawdown'))} | {_number(metrics.get('average_turnover'))} | "
            f"{metrics.get('active_portfolio_days', 0)} | {metrics.get('supported_stocks', 0)} |"
        )
    lines.extend(["", "## Paired contiguous-block comparisons", ""])
    lines.extend(
        f"- Reset minus `{name}`: mean daily difference {_number(result.mean_daily_difference)}, "
        f"{result.confidence_level:.0%} interval [{_number(result.confidence_interval_low)}, "
        f"{_number(result.confidence_interval_high)}], `{result.confidence_direction}`, "
        f"{len(result.effective_dates)} paired dates."
        for name, result in sorted(comparisons.items())
    )
    lines.extend(["", "## Attrition", ""])
    if attrition:
        lines.extend(f"- `{name}`: {count}" for name, count in sorted(attrition.items()))
    else:
        lines.append("- No downstream event exclusions were recorded.")
    lines.extend(["", "## Leakage and research-degree-of-freedom checks", ""])
    lines.extend(
        f"- `{classification}` — `{check}`: {evidence}"
        for check, classification, evidence in LEAKAGE_CHECKLIST
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The table and diagnostics are observed properties of this frozen sample and cost model.",
            "- Bootstrap intervals describe paired sample uncertainty; they do not establish deployable or profitable alpha.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "Null and adverse results are retained. The manifest records the scorer, company universe, boundary, "
            "parameters, and price convention used by this run; it cannot rule out off-repository trials.",
            "",
        ]
    )
    return "\n".join(lines)


def cumulative_return_svg(rows: Sequence[DailyLedgerRow], *, title: str) -> str:
    """Render a small dependency-free deterministic SVG line chart."""

    width, height = 900, 420
    left, right, top, bottom = 70, 25, 40, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [0.0]
    cumulative = 1.0
    for row in rows:
        cumulative *= 1 + row.net_return
        values.append(cumulative - 1)
    low = min(values, default=0.0)
    high = max(values, default=0.0)
    if high == low:
        high, low = high + 0.01, low - 0.01
    points = []
    denominator = max(1, len(values) - 1)
    for index, value in enumerate(values):
        x = left + plot_width * index / denominator
        y = top + plot_height * (high - value) / (high - low)
        points.append(f"{x:.2f},{y:.2f}")
    zero_y = top + plot_height * (high - 0.0) / (high - low)
    safe_title = html.escape(title)
    first_session = html.escape(rows[0].session if rows else "")
    last_session = html.escape(rows[-1].session if rows else "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="18">{safe_title}</text>\n'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>\n'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>\n'
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#aaa" stroke-dasharray="4 4"/>\n'
        f'<polyline fill="none" stroke="#145da0" stroke-width="2.5" points="{" ".join(points)}"/>\n'
        f'<text x="8" y="{top+8}" font-family="sans-serif" font-size="12">{high:.2%}</text>\n'
        f'<text x="8" y="{height-bottom}" font-family="sans-serif" font-size="12">{low:.2%}</text>\n'
        f'<text x="{left}" y="{height-18}" font-family="sans-serif" font-size="12">{first_session}</text>\n'
        f'<text x="{width-right-80}" y="{height-18}" font-family="sans-serif" font-size="12">{last_session}</text>\n'
        '</svg>\n'
    )


def exposure_turnover_svg(rows: Sequence[DailyLedgerRow], *, title: str) -> str:
    width, height = 900, 420
    left, right, top, bottom = 70, 25, 40, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    series = [
        ("gross exposure", [row.gross_exposure for row in rows], "#2a9d8f"),
        ("turnover", [row.turnover for row in rows], "#e76f51"),
    ]
    maximum = max((value for _, values, _ in series for value in values), default=1.0) or 1.0
    polylines = []
    for _label, values, colour in series:
        denominator = max(1, len(values) - 1)
        points = [
            f"{left + plot_width * index / denominator:.2f},{top + plot_height * (maximum-value) / maximum:.2f}"
            for index, value in enumerate(values)
        ]
        polylines.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{" ".join(points)}"/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="18">{html.escape(title)}</text>\n'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>\n'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>\n'
        + "\n".join(polylines)
        + f'\n<text x="{left+10}" y="{top+18}" fill="#2a9d8f" font-family="sans-serif" font-size="12">gross exposure</text>\n'
        f'<text x="{left+125}" y="{top+18}" fill="#e76f51" font-family="sans-serif" font-size="12">turnover</text>\n'
        '</svg>\n'
    )


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"
