#!/usr/bin/env python3
"""Build the canonical artifact for the recent-news FinBERT results dashboard.

The dashboard is a compact, plot-first companion to the long-form HTML
explainer. It reads only text-free aggregate artifacts from the frozen local
replay, reconciles every displayed headline metric, and writes the portable
dashboard contract consumed by the Data Analytics artifact builder.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPOSITORY_ROOT / "results" / "strategy_research" / "baselines" / "recent-news-midcap-finbert-event-v1-dfdd88113a5f"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reports" / "data" / "recent_news_midcap_finbert_dashboard_artifact.json"
SOURCE_SQL = REPOSITORY_ROOT / "reports" / "data" / "recent_news_midcap_finbert_dashboard_sources.sql"
PERIODS = ("Development", "Evaluation", "Combined")
COST_LEVELS_BPS = (0, 5, 10, 15, 20)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _compound(values: Iterable[float]) -> float:
    return math.prod(1.0 + float(value) for value in values) - 1.0


def _maximum_drawdown(returns: Iterable[float]) -> float:
    wealth = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in returns:
        wealth *= 1.0 + float(value)
        peak = max(peak, wealth)
        maximum_drawdown = min(maximum_drawdown, wealth / peak - 1.0)
    return maximum_drawdown


def _assert_close(label: str, actual: float, expected: float, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: reconstructed {actual:.15g}, frozen {expected:.15g}")


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"dashboard source must remain inside the repository: {path}") from exc


def _source_timestamp(paths: Iterable[Path]) -> str:
    timestamp = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(timestamp, tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _period_rows(daily_rows: list[dict[str, Any]], evaluation_start: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "development": [row for row in daily_rows if str(row["session"]) < evaluation_start],
        "evaluation": [row for row in daily_rows if str(row["session"]) >= evaluation_start],
        "all": daily_rows,
    }


def _reconcile_daily_metrics(rows_by_period: dict[str, list[dict[str, Any]]], metrics: dict[str, Any]) -> None:
    for period, rows in rows_by_period.items():
        frozen = metrics[period]
        if len(rows) != int(frozen["observations"]):
            raise ValueError(f"{period} observation mismatch: reconstructed {len(rows)}, frozen {frozen['observations']}")
        _assert_close(
            f"{period} cumulative gross return",
            _compound(float(row["gross_return"]) for row in rows),
            float(frozen["cumulative_gross_return"]),
        )
        _assert_close(
            f"{period} cumulative net return",
            _compound(float(row["net_return"]) for row in rows),
            float(frozen["cumulative_net_return"]),
        )
        _assert_close(
            f"{period} maximum drawdown",
            _maximum_drawdown(float(row["net_return"]) for row in rows),
            float(frozen["maximum_drawdown"]),
        )
        active_sessions = sum(int(row["active_names"]) > 0 for row in rows)
        if active_sessions != int(frozen["active_day_count_excluding_liquidation"]):
            raise ValueError(
                f"{period} active-session mismatch: reconstructed {active_sessions}, "
                f"frozen {frozen['active_day_count_excluding_liquidation']}"
            )
        _assert_close(
            f"{period} total turnover",
            sum(float(row["turnover"]) for row in rows),
            float(frozen["total_turnover"]),
        )


def _equity_rows(daily_rows: list[dict[str, Any]], evaluation_start: str) -> list[dict[str, Any]]:
    gross_wealth = 1.0
    net_wealth = 1.0
    net_peak = 1.0
    output: list[dict[str, Any]] = []
    for row in daily_rows:
        gross_return = float(row["gross_return"])
        net_return = float(row["net_return"])
        gross_wealth *= 1.0 + gross_return
        net_wealth *= 1.0 + net_return
        net_peak = max(net_peak, net_wealth)
        session = str(row["session"])
        output.append(
            {
                "date": session,
                "period": "Evaluation" if session >= evaluation_start else "Development",
                "cumulative_gross_return": gross_wealth - 1.0,
                "cumulative_net_return": net_wealth - 1.0,
                "net_drawdown": net_wealth / net_peak - 1.0,
                "daily_gross_return": gross_return,
                "daily_net_return": net_return,
                "active_names": int(row["active_names"]),
                "turnover": float(row["turnover"]),
            }
        )
    return output


def _period_summary(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {"development": "Development", "evaluation": "Evaluation", "all": "Combined"}
    output = []
    for key in ("development", "evaluation", "all"):
        row = metrics[key]
        output.append(
            {
                "period": labels[key],
                "gross_return": float(row["cumulative_gross_return"]),
                "net_return": float(row["cumulative_net_return"]),
                "net_sharpe": float(row["after_cost_sharpe"]),
                "maximum_drawdown": float(row["maximum_drawdown"]),
                "annualized_mean_return": float(row["annualized_mean_return"]),
                "annualized_volatility": float(row["annualized_volatility"]),
                "active_sessions": int(row["active_day_count_excluding_liquidation"]),
                "total_sessions": int(row["observations"]),
                "active_session_share": int(row["active_day_count_excluding_liquidation"]) / int(row["observations"]),
                "profitable_active_session_rate": float(row["profitable_active_day_rate"]),
                "breakeven_cost_bps": float(row["breakeven_cost_bps_per_side_approx"]),
                "maximum_name_concentration": float(row["maximum_name_concentration"]),
            }
        )
    return output


def _monthly_activity(daily_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    monthly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in daily_rows:
        month = str(row["session"])[:7]
        if int(row["active_names"]) > 0:
            status = "Invested"
        elif float(row["turnover"]) > 0:
            status = "Exit only"
        else:
            status = "Cash"
        monthly[month][status] += 1

    output: list[dict[str, Any]] = []
    for month in sorted(monthly):
        counts = monthly[month]
        total = sum(counts.values())
        for status in ("Invested", "Exit only", "Cash"):
            output.append(
                {
                    "month": f"{month}-01",
                    "status": status,
                    "sessions": counts[status],
                    "total_sessions": total,
                    "active_share": counts["Invested"] / total,
                }
            )
    return output


def _cost_sensitivity(evaluation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for cost_bps in COST_LEVELS_BPS:
        hypothetical_returns = (float(row["gross_return"]) - float(row["turnover"]) * cost_bps / 10_000.0 for row in evaluation_rows)
        output.append(
            {
                "cost_label": f"{cost_bps} bps",
                "cost_bps": cost_bps,
                "evaluation_net_return": _compound(hypothetical_returns),
                "evaluation_sessions": len(evaluation_rows),
            }
        )
    return output


def _stock_contributions(stock_metrics: dict[str, Any], evaluation_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(stock_metrics.get("finbert") or [])
    if not rows:
        raise ValueError("stock_metrics.json has no FinBERT rows")
    _assert_close(
        "evaluation stock transaction costs",
        sum(float(row["transaction_cost_usd"]) for row in rows),
        float(evaluation_metrics["total_transaction_cost_usd"]),
        tolerance=1e-6,
    )
    ranked = sorted(rows, key=lambda row: float(row["net_pnl_usd"]))
    selected = [*ranked[:5], *ranked[-5:]]
    return [
        {
            "symbol": str(row["symbol"]),
            "net_pnl_usd": float(row["net_pnl_usd"]),
            "gross_pnl_usd": float(row["gross_pnl_usd"]),
            "transaction_cost_usd": float(row["transaction_cost_usd"]),
            "active_days": int(row["active_days"]),
            "maximum_weight": float(row["maximum_weight"]),
            "selection": "Bottom five" if index < 5 else "Top five",
        }
        for index, row in enumerate(selected)
    ]


def _source(
    *,
    source_id: str,
    label: str,
    path: Path,
) -> dict[str, Any]:
    return {"id": source_id, "label": label, "path": _repo_path(path)}


def build_artifact(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    daily_path = run_dir / "daily_pnl.jsonl"
    stock_path = run_dir / "stock_metrics.json"
    bootstrap_path = run_dir / "bootstrap.json"
    source_paths = (manifest_path, metrics_path, daily_path, stock_path, bootstrap_path)

    manifest = _read_json(manifest_path)
    metrics_payload = _read_json(metrics_path)
    stock_metrics = _read_json(stock_path)
    bootstrap = _read_json(bootstrap_path)
    raw_daily = _read_jsonl(daily_path)

    if manifest.get("status") != "completed":
        raise ValueError(f"run is not complete: {manifest.get('status')!r}")
    if manifest.get("sharing", {}).get("contains_licensed_headline_text") is not False:
        raise ValueError("dashboard input must explicitly exclude licensed headline text")
    evaluation_start = str(manifest["config"]["run"]["evaluation_start"])
    finbert_metrics = metrics_payload.get("finbert")
    if not isinstance(finbert_metrics, dict):
        raise ValueError("metrics.json has no FinBERT metrics")

    liquidation_rows = [row for row in raw_daily if row.get("scorer") == "finbert" and bool(row.get("final_liquidation"))]
    if any(abs(float(row["net_return"])) > 1e-15 for row in liquidation_rows):
        raise ValueError("non-zero final-liquidation return requires explicit dashboard handling")
    daily_rows = [row for row in raw_daily if row.get("scorer") == "finbert" and not bool(row.get("final_liquidation"))]
    if not daily_rows:
        raise ValueError("daily_pnl.jsonl has no non-liquidation FinBERT rows")
    sessions = [str(row["session"]) for row in daily_rows]
    if sessions != sorted(sessions) or len(set(sessions)) != len(sessions):
        raise ValueError("FinBERT daily rows must have unique, ascending sessions")

    rows_by_period = _period_rows(daily_rows, evaluation_start)
    _reconcile_daily_metrics(rows_by_period, finbert_metrics)
    period_summary = _period_summary(finbert_metrics)
    cost_sensitivity = _cost_sensitivity(rows_by_period["evaluation"])
    frozen_cost = float(manifest["config"]["portfolio"]["cost_bps_per_side"])
    frozen_cost_row = next((row for row in cost_sensitivity if row["cost_bps"] == frozen_cost), None)
    if frozen_cost_row is None:
        raise ValueError(f"frozen cost {frozen_cost} bps is absent from dashboard cost levels")
    _assert_close(
        "evaluation cost-sensitivity return at the frozen cost",
        float(frozen_cost_row["evaluation_net_return"]),
        float(finbert_metrics["evaluation"]["cumulative_net_return"]),
    )

    generated_at = _source_timestamp(source_paths)
    sources = [
        _source(
            source_id="frozen_metrics",
            label="Frozen FinBERT strategy metrics",
            path=SOURCE_SQL,
        ),
        _source(
            source_id="frozen_daily_pnl",
            label="Frozen FinBERT daily P&L",
            path=SOURCE_SQL,
        ),
        _source(
            source_id="frozen_stock_metrics",
            label="Frozen evaluation stock contributions",
            path=SOURCE_SQL,
        ),
        _source(
            source_id="frozen_bootstrap",
            label="Frozen evaluation bootstrap",
            path=SOURCE_SQL,
        ),
        _source(
            source_id="frozen_manifest",
            label="Frozen strategy manifest",
            path=SOURCE_SQL,
        ),
    ]
    source_inventory = [{"id": source["id"], "label": source["label"], "path": source["path"]} for source in sources]

    bootstrap_cash = bootstrap["primary_vs_cash"]
    caveat_body = (
        "## Interpretation boundary\n\n"
        "The positive realised return and Sharpe are exploratory, not confirmed alpha. The evaluation bootstrap "
        "interval crosses zero, the sample was previously examined, and several active sessions carried concentrated "
        "single-name weights. Use the [long-form explainer](sentiment_trading_baseline_explainer.html) "
        "for the full rule, limitations, failed alternatives, and reproduction trail."
    )
    equity_rows = _equity_rows(daily_rows, evaluation_start)
    datasets = {
        "period_summary": period_summary,
        "period_returns": [
            {
                "period": row["period"],
                "return_type": return_type,
                "compounded_return": row[field],
                "net_sharpe": row["net_sharpe"],
                "active_sessions": row["active_sessions"],
            }
            for row in period_summary
            for return_type, field in (("Gross return", "gross_return"), ("Net return", "net_return"))
        ],
        "equity_curve": [
            {
                "date": row["date"],
                "period": row["period"],
                "return_type": return_type,
                "cumulative_return": row[field],
                "daily_return": row[daily_field],
                "active_names": row["active_names"],
                "turnover": row["turnover"],
            }
            for row in equity_rows
            for return_type, field, daily_field in (
                ("Net return", "cumulative_net_return", "daily_net_return"),
                ("Gross return", "cumulative_gross_return", "daily_gross_return"),
            )
        ],
        "drawdown_curve": [{"date": row["date"], "period": row["period"], "net_drawdown": row["net_drawdown"]} for row in equity_rows],
        "monthly_activity": _monthly_activity(daily_rows),
        "cost_sensitivity": cost_sensitivity,
        "stock_contributions": _stock_contributions(stock_metrics, finbert_metrics["evaluation"]),
        "bootstrap_summary": [
            {
                "comparison": "FinBERT vs cash",
                "mean_daily_net_return": float(bootstrap_cash["mean_daily_difference"]),
                "confidence_interval_low": float(bootstrap_cash["confidence_interval_low"]),
                "confidence_interval_high": float(bootstrap_cash["confidence_interval_high"]),
                "confidence_level": float(bootstrap_cash["confidence_level"]),
                "replications": int(bootstrap_cash["replications"]),
                "block_length_sessions": int(bootstrap_cash["block_length"]),
                "confidence_direction": str(bootstrap_cash["confidence_direction"]),
            }
        ],
    }

    return {
        "surface": "dashboard",
        "manifest": {
            "version": 1,
            "surface": "dashboard",
            "title": "Recent-News FinBERT Strategy Results",
            "description": ("A plot-first snapshot of the frozen 22-stock, one-session, market-neutral strategy replay."),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "evaluation_return",
                    "description": "Compounded evaluation return after the frozen 10 bps per-side transaction cost.",
                    "dataset": "period_summary",
                    "filter": {"period": "Evaluation"},
                    "sourceId": "frozen_metrics",
                    "metrics": [{"label": "Evaluation net return", "field": "net_return", "format": "percent", "signed": True}],
                },
                {
                    "id": "evaluation_sharpe",
                    "description": "Annualized evaluation mean net return divided by annualized daily net volatility.",
                    "dataset": "period_summary",
                    "filter": {"period": "Evaluation"},
                    "sourceId": "frozen_metrics",
                    "metrics": [{"label": "Evaluation net Sharpe", "field": "net_sharpe", "format": "number"}],
                },
                {
                    "id": "evaluation_drawdown",
                    "description": "Largest peak-to-trough loss in the evaluation net equity curve.",
                    "dataset": "period_summary",
                    "filter": {"period": "Evaluation"},
                    "sourceId": "frozen_metrics",
                    "metrics": [{"label": "Evaluation max drawdown", "field": "maximum_drawdown", "format": "percent", "signed": True}],
                },
                {
                    "id": "evaluation_activity",
                    "description": "Sessions with both an eligible long and short leg in the evaluation block.",
                    "dataset": "period_summary",
                    "filter": {"period": "Evaluation"},
                    "sourceId": "frozen_metrics",
                    "metrics": [
                        {"label": "Active evaluation sessions", "field": "active_sessions", "format": "number"},
                        {"label": "Total sessions", "field": "total_sessions", "format": "number"},
                    ],
                },
                {
                    "id": "evaluation_breakeven",
                    "description": "Approximate per-side cost that would reduce evaluation mean daily net return to zero.",
                    "dataset": "period_summary",
                    "filter": {"period": "Evaluation"},
                    "sourceId": "frozen_metrics",
                    "metrics": [{"label": "Break-even cost (bps/side)", "field": "breakeven_cost_bps", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "equity_curve",
                    "title": "Cumulative portfolio return",
                    "subtitle": "Development and evaluation compounded continuously; evaluation begins 2 Jan 2026.",
                    "intent": "trend",
                    "question": "How did gross and cost-aware net performance evolve through time?",
                    "rationale": "A two-series line chart shows path, timing, and the persistent cost wedge without hiding reversals.",
                    "type": "line",
                    "dataset": "equity_curve",
                    "sourceId": "frozen_daily_pnl",
                    "encodings": {
                        "x": {"field": "date", "type": "temporal", "label": "Session"},
                        "y": {"field": "cumulative_return", "type": "quantitative", "label": "Cumulative return", "format": "percent"},
                        "color": {"field": "return_type", "type": "nominal", "label": "Return type"},
                        "tooltip": [
                            {"field": "period", "type": "nominal", "label": "Period"},
                            {"field": "daily_return", "type": "quantitative", "label": "Daily return", "format": "percent"},
                            {"field": "active_names", "type": "quantitative", "label": "Active names"},
                        ],
                    },
                    "comparisonContext": {"unit": "compounded return", "grain": "XNYS session"},
                    "valueFormat": "percent",
                    "legend": {"position": "bottom", "sort": "spec"},
                    "labels": {"values": "endpoints"},
                    "referenceLines": [
                        {"axis": "x", "value": evaluation_start, "label": "Evaluation starts", "color": "neutral", "lineStyle": "dashed"},
                        {"axis": "y", "value": 0, "label": "Start", "color": "neutral", "lineStyle": "solid"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "net_drawdown",
                    "title": "Net portfolio drawdown",
                    "subtitle": "Peak-to-trough loss from the combined cost-aware equity curve.",
                    "intent": "trend",
                    "question": "When and how deeply did the net portfolio fall below its prior peak?",
                    "rationale": "A single filled time series makes the depth and duration of losses visible against a zero peak line.",
                    "type": "area",
                    "dataset": "drawdown_curve",
                    "sourceId": "frozen_daily_pnl",
                    "encodings": {
                        "x": {"field": "date", "type": "temporal", "label": "Session"},
                        "y": {"field": "net_drawdown", "type": "quantitative", "label": "Drawdown", "format": "percent"},
                        "tooltip": [{"field": "period", "type": "nominal", "label": "Period"}],
                    },
                    "comparisonContext": {"baseline": "prior net equity peak", "unit": "percent"},
                    "valueFormat": "percent",
                    "referenceLines": [{"axis": "y", "value": 0, "label": "Peak", "color": "neutral", "lineStyle": "solid"}],
                    "layout": "half",
                },
                {
                    "id": "period_returns",
                    "title": "Gross and net return by period",
                    "subtitle": "Compounded within each chronological block and across both blocks.",
                    "intent": "comparison",
                    "question": "Did returns remain positive after transaction costs in each period?",
                    "rationale": "Grouped bars directly compare same-unit gross and net returns across the two blocks and combined sample.",
                    "type": "bar",
                    "dataset": "period_returns",
                    "sourceId": "frozen_metrics",
                    "encodings": {
                        "x": {"field": "period", "type": "ordinal", "label": "Period"},
                        "y": {"field": "compounded_return", "type": "quantitative", "label": "Compounded return", "format": "percent"},
                        "color": {"field": "return_type", "type": "nominal", "label": "Return type"},
                        "tooltip": [
                            {"field": "net_sharpe", "type": "quantitative", "label": "Net Sharpe"},
                            {"field": "active_sessions", "type": "quantitative", "label": "Active sessions"},
                        ],
                    },
                    "comparisonContext": {"unit": "compounded return", "grain": "chronological block"},
                    "valueFormat": "percent",
                    "legend": {"position": "bottom", "sort": "spec"},
                    "labels": {"values": "all"},
                    "referenceLines": [{"axis": "y", "value": 0, "label": "Cash", "color": "neutral", "lineStyle": "solid"}],
                    "layout": "half",
                },
                {
                    "id": "monthly_activity",
                    "title": "Monthly portfolio activity",
                    "subtitle": "Invested sessions have both portfolio legs; exit-only sessions carry turnover without exposure.",
                    "intent": "composition",
                    "question": "How sparse was the strategy's deployment through the sample?",
                    "rationale": (
                        "A 100 percent stacked monthly view separates invested, exit-only, and cash sessions while preserving seasonality."
                    ),
                    "type": "stackedBar100",
                    "dataset": "monthly_activity",
                    "sourceId": "frozen_daily_pnl",
                    "encodings": {
                        "x": {"field": "month", "type": "temporal", "label": "Month"},
                        "y": {"field": "sessions", "type": "quantitative", "label": "Share of sessions"},
                        "color": {"field": "status", "type": "nominal", "label": "Session state"},
                        "tooltip": [
                            {"field": "total_sessions", "type": "quantitative", "label": "Total sessions"},
                            {"field": "active_share", "type": "quantitative", "label": "Invested share", "format": "percent"},
                        ],
                    },
                    "comparisonContext": {"denominator": "all XNYS sessions in each month", "unit": "share"},
                    "valueFormat": "number",
                    "legend": {"position": "bottom", "sort": "spec"},
                    "layout": "half",
                },
                {
                    "id": "cost_sensitivity",
                    "title": "Evaluation return across transaction costs",
                    "subtitle": "Recomputed from frozen daily gross returns and turnover; the replay uses 10 bps per side.",
                    "intent": "comparison",
                    "question": "How much of the evaluation return survives plausible per-side cost assumptions?",
                    "rationale": (
                        "Ordered bars keep the zero-return threshold visible and avoid implying more precision "
                        "than five cost scenarios provide."
                    ),
                    "type": "bar",
                    "dataset": "cost_sensitivity",
                    "sourceId": "frozen_daily_pnl",
                    "encodings": {
                        "x": {"field": "cost_label", "type": "ordinal", "label": "Cost per side"},
                        "y": {
                            "field": "evaluation_net_return",
                            "type": "quantitative",
                            "label": "Compounded evaluation return",
                            "format": "percent",
                        },
                        "tooltip": [
                            {"field": "cost_bps", "type": "quantitative", "label": "Cost (bps/side)"},
                            {"field": "evaluation_sessions", "type": "quantitative", "label": "Sessions"},
                        ],
                    },
                    "comparisonContext": {"unit": "compounded evaluation return", "grain": "cost scenario"},
                    "valueFormat": "percent",
                    "labels": {"values": "all"},
                    "referenceLines": [{"axis": "y", "value": 0, "label": "Break-even", "color": "neutral", "lineStyle": "solid"}],
                    "layout": "half",
                },
                {
                    "id": "stock_contributions",
                    "title": "Largest evaluation stock contributions",
                    "subtitle": "Five largest positive and negative contributions after transaction costs.",
                    "intent": "comparison",
                    "question": "Which stocks contributed most to evaluation profit and loss?",
                    "rationale": (
                        "Sorted horizontal bars make signed stock-level contributions and concentration easy "
                        "to compare without a dense table."
                    ),
                    "type": "horizontalBar",
                    "dataset": "stock_contributions",
                    "sourceId": "frozen_stock_metrics",
                    "encodings": {
                        "x": {"field": "symbol", "type": "nominal", "label": "Stock"},
                        "y": {"field": "net_pnl_usd", "type": "quantitative", "label": "Net P&L", "format": "currency"},
                        "tooltip": [
                            {"field": "gross_pnl_usd", "type": "quantitative", "label": "Gross P&L", "format": "currency"},
                            {"field": "transaction_cost_usd", "type": "quantitative", "label": "Transaction cost", "format": "currency"},
                            {"field": "active_days", "type": "quantitative", "label": "Active days"},
                            {"field": "maximum_weight", "type": "quantitative", "label": "Maximum weight", "format": "percent"},
                        ],
                    },
                    "comparisonContext": {"unit": "USD net P&L", "grain": "stock in evaluation block"},
                    "valueFormat": "currency",
                    "labels": {"values": "all"},
                    "referenceLines": [{"axis": "y", "value": 0, "label": "Zero", "color": "neutral", "lineStyle": "solid"}],
                    "settings": {"sort": "ascending", "orientation": "horizontal"},
                    "layout": "full",
                },
            ],
            "sources": source_inventory,
            "blocks": [
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "evaluation_return",
                        "evaluation_sharpe",
                        "evaluation_drawdown",
                        "evaluation_activity",
                        "evaluation_breakeven",
                    ],
                },
                {"id": "equity_curve_block", "type": "chart", "chartId": "equity_curve", "layout": "full"},
                {"id": "drawdown_block", "type": "chart", "chartId": "net_drawdown", "layout": "half"},
                {"id": "period_returns_block", "type": "chart", "chartId": "period_returns", "layout": "half"},
                {"id": "activity_block", "type": "chart", "chartId": "monthly_activity", "layout": "half"},
                {"id": "cost_block", "type": "chart", "chartId": "cost_sensitivity", "layout": "half"},
                {"id": "stock_block", "type": "chart", "chartId": "stock_contributions", "layout": "full"},
                {"id": "interpretation_boundary", "type": "markdown", "body": caveat_body},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite to replace it")
    artifact = build_artifact(args.run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    datasets = artifact["snapshot"]["datasets"]
    print(
        json.dumps(
            {
                "output": str(output),
                "generated_at": artifact["snapshot"]["generatedAt"],
                "datasets": {name: len(rows) for name, rows in datasets.items()},
                "charts": len(artifact["manifest"]["charts"]),
                "validated": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
