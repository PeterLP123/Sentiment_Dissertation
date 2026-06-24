from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .agreement import compute_agreement
from .constants import is_valid_label
from .latex_tables import generate_latex_tables
from .metrics import _response_prediction, bootstrap_metric_ci, load_metric_json, mcnemar_test
from .plotting import generate_figures
from .storage import BenchmarkStore


def _row_prediction(row: dict[str, Any]) -> str:
    normalized = row.get("normalized_label")
    if normalized is None or (isinstance(normalized, float) and math.isnan(normalized)):
        normalized = None
    return _response_prediction(
        {
            "status": row.get("status"),
            "parse_status": row.get("parse_status"),
            "normalized_label": normalized,
        }
    )


def _compute_statistics(responses: pd.DataFrame, seed: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scope": "primary",
        "seed": seed,
        "per_model": {},
        "pairwise_mcnemar": [],
        "agreement": None,
    }
    if responses.empty or "has_conflicting_duplicate" not in responses.columns:
        return payload

    primary = responses[responses["has_conflicting_duplicate"] == 0]
    if primary.empty:
        return payload

    predictions_by_model: dict[str, dict[int, str]] = {}
    aligned: dict[str, dict[int, tuple[str, str]]] = {}
    for model_id, group in primary.groupby("model_id"):
        records = group.to_dict(orient="records")
        y_true = [str(record["hidden_label"]) for record in records]
        y_pred = [_row_prediction(record) for record in records]
        payload["per_model"][str(model_id)] = {
            "accuracy_ci": bootstrap_metric_ci(y_true, y_pred, "accuracy", seed=seed),
            "macro_f1_ci": bootstrap_metric_ci(y_true, y_pred, "macro_f1", seed=seed),
            "n": len(y_true),
        }
        aligned[str(model_id)] = {
            int(record["row_number"]): (str(record["hidden_label"]), prediction)
            for record, prediction in zip(records, y_pred, strict=True)
        }
        valid = {
            int(record["row_number"]): prediction
            for record, prediction in zip(records, y_pred, strict=True)
            if is_valid_label(prediction)
        }
        if valid:
            predictions_by_model[str(model_id)] = valid

    model_ids = sorted(aligned)
    for first_index in range(len(model_ids)):
        for second_index in range(first_index + 1, len(model_ids)):
            model_a = model_ids[first_index]
            model_b = model_ids[second_index]
            common = sorted(set(aligned[model_a]) & set(aligned[model_b]))
            if not common:
                continue
            y_true = [aligned[model_a][row_number][0] for row_number in common]
            y_pred_a = [aligned[model_a][row_number][1] for row_number in common]
            y_pred_b = [aligned[model_b][row_number][1] for row_number in common]
            result = mcnemar_test(y_true, y_pred_a, y_pred_b)
            payload["pairwise_mcnemar"].append({"model_a": model_a, "model_b": model_b, **result})

    if len(predictions_by_model) >= 2:
        payload["agreement"] = asdict(compute_agreement(predictions_by_model))
    return payload


def _compute_operational(responses: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-model operational statistics over every attempted row in the run.

    Unlike the classification metrics these are scope-independent: cost, latency,
    token usage, and failure rates describe API behavior, not label quality, so
    conflicting-duplicate rows are included. Cost depends on provider-reported
    generation metadata and is reported as None when no row has it (baselines,
    local models, or providers without cost data).
    """
    per_model: dict[str, dict[str, Any]] = {}
    if responses.empty:
        return per_model
    for model_id, group in responses.groupby("model_id"):
        n_rows = len(group)
        latencies = pd.to_numeric(group["latency_ms"], errors="coerce").dropna()
        costs = pd.to_numeric(group["total_cost"], errors="coerce").dropna()
        prompt_tokens = pd.to_numeric(group["prompt_tokens"], errors="coerce").dropna()
        completion_tokens = pd.to_numeric(group["completion_tokens"], errors="coerce").dropna()
        total_tokens = pd.to_numeric(group["total_tokens"], errors="coerce").dropna()
        invalid_count = int((group["parse_status"] == "invalid").sum())
        api_error_count = int((group["status"] != "success").sum())
        per_model[str(model_id)] = {
            "n_rows": n_rows,
            "latency_ms": {
                "n": int(latencies.size),
                "mean": float(latencies.mean()) if latencies.size else None,
                "p50": float(latencies.quantile(0.5)) if latencies.size else None,
                "p95": float(latencies.quantile(0.95)) if latencies.size else None,
            },
            "cost": {
                "n_rows_with_cost": int(costs.size),
                "total_usd": float(costs.sum()) if costs.size else None,
                "usd_per_1k_rows": float(costs.sum() / costs.size * 1000.0) if costs.size else None,
            },
            "tokens": {
                "total_prompt": int(prompt_tokens.sum()),
                "total_completion": int(completion_tokens.sum()),
                "mean_total_per_row": float(total_tokens.mean()) if total_tokens.size else None,
            },
            "invalid_count": invalid_count,
            "invalid_rate": invalid_count / n_rows,
            "api_error_count": api_error_count,
            "api_error_rate": api_error_count / n_rows,
        }
    return per_model


def dataset_sha256(path: str | Path) -> str | None:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return None
    digest = hashlib.sha256()
    with dataset_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _summary_run_meta_lines(runs: pd.DataFrame) -> list[str]:
    if runs.empty:
        return []
    run = runs.iloc[0].to_dict()
    machine = _optional_text(run.get("machine_label")) or _optional_text(run.get("machine_id")) or "unknown"
    return [
        f"- Mode: `{run['mode']}`",
        f"- Status: `{run['status']}`",
        f"- Output mode: `{run['output_mode']}`",
        f"- Created: `{run['created_at']}`",
        f"- Machine: `{machine}`",
        "",
    ]


def _summary_metrics_lines(metric_records: list[dict[str, Any]]) -> list[str]:
    if not metric_records:
        return []
    lines = ["## Metrics", ""]
    for metric in metric_records:
        lines.extend(
            [
                f"### {metric['model_id']} ({metric['scope']})",
                "",
                f"- Rows: {metric['row_count']}",
                f"- Accuracy: {metric['accuracy']:.4f}",
                f"- Balanced accuracy: {metric.get('balanced_accuracy', 0.0):.4f}",
                f"- MCC: {metric.get('mcc', 0.0):.4f}",
                f"- Macro F1: {metric['macro_f1']:.4f}",
                f"- Weighted F1: {metric['weighted_f1']:.4f}",
                f"- Invalid outputs: {metric['invalid_output_count']}",
                f"- API errors: {metric['api_error_count']}",
            ]
        )
        calibration = metric.get("calibration")
        if isinstance(calibration, dict):
            lines.extend(
                [
                    f"- Brier score: {calibration['brier_score']:.4f} (soft-label, n={calibration['n_scored']})",
                    f"- ECE: {calibration['ece']:.4f} ({calibration['n_bins']} bins)",
                ]
            )
        lines.append("")
    return lines


def _summary_statistics_lines(statistics: dict[str, Any]) -> list[str]:
    per_model_statistics = statistics["per_model"]
    if not per_model_statistics:
        return []
    lines = ["## Statistics (primary scope)", ""]
    for model_id, model_statistics in per_model_statistics.items():
        accuracy_ci = model_statistics["accuracy_ci"]
        macro_f1_ci = model_statistics["macro_f1_ci"]
        confidence_pct = f"{accuracy_ci['confidence'] * 100:.0f}"
        accuracy_interval = f"[{accuracy_ci['lower']:.4f}, {accuracy_ci['upper']:.4f}]"
        macro_f1_interval = f"[{macro_f1_ci['lower']:.4f}, {macro_f1_ci['upper']:.4f}]"
        lines.extend(
            [
                f"### {model_id}",
                "",
                f"- Rows: {model_statistics['n']}",
                f"- Accuracy: {accuracy_ci['point']:.4f} ({confidence_pct}% CI {accuracy_interval})",
                f"- Macro F1: {macro_f1_ci['point']:.4f} ({confidence_pct}% CI {macro_f1_interval})",
                "",
            ]
        )
    pairwise = statistics["pairwise_mcnemar"]
    if pairwise:
        lines.extend(["### Pairwise McNemar", ""])
        for pair in pairwise:
            lines.append(
                f"- {pair['model_a']} vs {pair['model_b']}: p = {pair['p_value']:.4f} "
                f"(n_discordant = {pair['n_discordant']}, {pair['method']})"
            )
        lines.append("")
    return lines


def _summary_operational_lines(statistics: dict[str, Any]) -> list[str]:
    operational_models = statistics["operational"]["per_model"]
    if not operational_models:
        return []

    def _cell(value: Any, decimals: int = 0) -> str:
        return f"{float(value):.{decimals}f}" if isinstance(value, (int, float)) else "-"

    def _cost_cell(value: Any) -> str:
        return f"{float(value):.4f}" if isinstance(value, (int, float)) else "-"

    lines = [
        "## Operational metrics (OQ4)",
        "",
        "Computed over every attempted row in the run (both scopes). "
        "Cost requires provider-reported generation metadata and is `-` for baselines and local models.",
        "",
        "| Model | Rows | Cost (USD) | USD/1k rows | Latency p50 (ms) | Latency p95 (ms) | Tokens/row | Invalid | API errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id in sorted(operational_models):
        entry = operational_models[model_id]
        latency = entry["latency_ms"]
        cost = entry["cost"]
        lines.append(
            f"| {model_id} | {entry['n_rows']} | {_cost_cell(cost['total_usd'])} "
            f"| {_cost_cell(cost['usd_per_1k_rows'])} | {_cell(latency['p50'])} | {_cell(latency['p95'])} "
            f"| {_cell(entry['tokens']['mean_total_per_row'], 1)} "
            f"| {entry['invalid_count']} ({entry['invalid_rate'] * 100:.1f}%) "
            f"| {entry['api_error_count']} ({entry['api_error_rate'] * 100:.1f}%) |"
        )
    lines.append("")
    return lines


def _summary_agreement_lines(statistics: dict[str, Any]) -> list[str]:
    agreement = statistics.get("agreement")
    if not agreement:
        return []

    def _fmt(value: float | None) -> str:
        return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"

    lines = [
        "## Inter-model agreement (primary scope)",
        "",
        f"- Raters (models): {agreement['n_raters']}",
        f"- Observed agreement: {_fmt(agreement['observed_agreement'])}",
        f"- Fleiss' kappa: {_fmt(agreement['fleiss_kappa'])}",
        f"- Krippendorff's alpha: {_fmt(agreement['krippendorff_alpha'])}",
        "",
    ]
    for pair in agreement["pairwise_cohen_kappa"]:
        lines.append(
            f"- Cohen's kappa {pair['rater_a']} vs {pair['rater_b']}: {pair['kappa']:.4f} (n = {pair['n']})"
        )
    lines.append("")
    return lines


def _build_summary_markdown(
    run_id: int,
    runs: pd.DataFrame,
    metric_records: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> list[str]:
    lines = [f"# Sentiment Benchmark Run {run_id}", ""]
    lines.extend(_summary_run_meta_lines(runs))
    lines.extend(_summary_metrics_lines(metric_records))
    lines.extend(_summary_statistics_lines(statistics))
    lines.extend(_summary_operational_lines(statistics))
    lines.extend(_summary_agreement_lines(statistics))
    return lines


def export_run(db_path: str | Path, run_id: int, output_dir: str | Path | None = None) -> list[Path]:
    db = Path(db_path)
    destination = Path(output_dir) if output_dir else Path("results/exports") / f"run_{run_id}"
    destination.mkdir(parents=True, exist_ok=True)

    with BenchmarkStore(db).connect() as connection:
        responses = pd.read_sql_query(
            """
            SELECT
                r.run_id,
                r.model_id,
                r.row_number,
                d.sentence,
                d.hidden_label,
                d.has_conflicting_duplicate,
                r.normalized_label,
                CASE
                    WHEN r.status = 'success' AND r.parse_status = 'valid' AND r.normalized_label = d.hidden_label THEN 1
                    ELSE 0
                END AS is_correct,
                r.parse_status,
                r.status,
                r.raw_content,
                r.explanation,
                r.label_probabilities,
                r.latency_ms,
                r.prompt_tokens,
                r.completion_tokens,
                r.total_tokens,
                r.generation_id,
                gm.total_cost,
                gm.provider_name,
                r.error
            FROM responses r
            JOIN dataset_items d ON d.row_number = r.row_number
            LEFT JOIN generation_metadata gm
                ON gm.run_id = r.run_id
                AND gm.row_number = r.row_number
                AND gm.model_id = r.model_id
                AND gm.generation_id = r.generation_id
            WHERE r.run_id = ?
            ORDER BY r.model_id, r.row_number
            """,
            connection,
            params=(run_id,),
        )
        metrics = pd.read_sql_query(
            "SELECT run_id, model_id, scope, metrics_json, created_at FROM metrics WHERE run_id = ? ORDER BY model_id, scope",
            connection,
            params=(run_id,),
        )
        prompts = pd.read_sql_query(
            """
            SELECT p.*
            FROM prompts p
            JOIN runs r ON r.prompt_hash = p.prompt_hash
            WHERE r.id = ?
            """,
            connection,
            params=(run_id,),
        )
        runs = pd.read_sql_query("SELECT * FROM runs WHERE id = ?", connection, params=(run_id,))

    paths: list[Path] = []
    responses_csv = destination / "responses.csv"
    responses.to_csv(responses_csv, index=False)
    paths.append(responses_csv)

    responses_json = destination / "responses.json"
    responses.to_json(responses_json, orient="records", indent=2)
    paths.append(responses_json)

    metrics_json_path = destination / "metrics.json"
    metric_records = []
    for record in metrics.to_dict(orient="records"):
        parsed = load_metric_json(record.pop("metrics_json"))
        metric_records.append({**record, **parsed})
    metrics_json_path.write_text(json.dumps(metric_records, indent=2), encoding="utf-8")
    paths.append(metrics_json_path)

    run_json_path = destination / "run.json"
    run_records = runs.to_dict(orient="records")
    prompt_records = prompts.to_dict(orient="records")
    run_record = run_records[0] if run_records else {}
    request_settings = json.loads(run_record.get("request_json") or "{}") if run_record else {}
    models = json.loads(run_record.get("models_json") or "[]") if run_record else []
    dataset_path = str(run_record.get("dataset_path") or "")
    run_environment = _json_object(run_record.get("environment_json")) if run_record else {}
    machine = run_environment.get("machine") if isinstance(run_environment.get("machine"), dict) else {}
    machine_id = _optional_text(run_record.get("machine_id")) if run_record else None
    machine_label = _optional_text(run_record.get("machine_label")) if run_record else None
    run_payload = {
        "run": run_records,
        "prompt": prompt_records,
        "metadata": {
            "package_version": __version__,
            "exported_at": datetime.now(UTC).isoformat(),
            "dataset_path": dataset_path,
            "dataset_sha256": dataset_sha256(dataset_path) if dataset_path else None,
            "prompt_hash": run_record.get("prompt_hash"),
            "seed": request_settings.get("seed"),
            "mode": run_record.get("mode"),
            "models": models,
            "provider": request_settings.get("provider", "openrouter"),
            "base_url": run_record.get("base_url"),
            "machine_id": machine_id or machine.get("id"),
            "machine_label": machine_label or machine.get("label"),
            "run_environment": run_environment,
            "request_settings": request_settings,
        },
    }
    run_json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    paths.append(run_json_path)

    seed_setting = request_settings.get("seed")
    seed = seed_setting if isinstance(seed_setting, int) else 42
    statistics = _compute_statistics(responses, seed)
    statistics["operational"] = {
        "note": "Computed over every attempted row in the run (both scopes); rates are per attempted row.",
        "per_model": _compute_operational(responses),
    }
    statistics_path = destination / "statistics.json"
    statistics_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    paths.append(statistics_path)

    summary_path = destination / "summary.md"
    summary_path.write_text(
        "\n".join(_build_summary_markdown(run_id, runs, metric_records, statistics)),
        encoding="utf-8",
    )
    paths.append(summary_path)

    # Dissertation-ready booktabs tables (pure text; no optional dependencies).
    table_paths = generate_latex_tables(
        metric_records, statistics, run_payload["metadata"], run_id, destination / "tables"
    )
    paths.extend(table_paths)

    # Publication-ready figures (no-op if matplotlib is not installed).
    figure_paths = generate_figures(metric_records, statistics, destination / "figures")
    paths.extend(figure_paths)
    return paths
