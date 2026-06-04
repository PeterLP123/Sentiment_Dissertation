from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import __version__
from .metrics import load_metric_json


def dataset_sha256(path: str | Path) -> str | None:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return None
    digest = hashlib.sha256()
    with dataset_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_run(db_path: str | Path, run_id: int, output_dir: str | Path | None = None) -> list[Path]:
    db = Path(db_path)
    destination = Path(output_dir) if output_dir else Path("results/exports") / f"run_{run_id}"
    destination.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db) as connection:
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
            "base_url": run_record.get("base_url"),
            "request_settings": request_settings,
        },
    }
    run_json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    paths.append(run_json_path)

    summary_path = destination / "summary.md"
    lines = [f"# Sentiment Benchmark Run {run_id}", ""]
    if not runs.empty:
        run = runs.iloc[0].to_dict()
        lines.extend(
            [
                f"- Mode: `{run['mode']}`",
                f"- Status: `{run['status']}`",
                f"- Output mode: `{run['output_mode']}`",
                f"- Created: `{run['created_at']}`",
                "",
            ]
        )
    if metric_records:
        lines.extend(["## Metrics", ""])
        for metric in metric_records:
            lines.extend(
                [
                    f"### {metric['model_id']} ({metric['scope']})",
                    "",
                    f"- Rows: {metric['row_count']}",
                    f"- Accuracy: {metric['accuracy']:.4f}",
                    f"- Macro F1: {metric['macro_f1']:.4f}",
                    f"- Weighted F1: {metric['weighted_f1']:.4f}",
                    f"- Invalid outputs: {metric['invalid_output_count']}",
                    f"- API errors: {metric['api_error_count']}",
                    "",
                ]
            )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    paths.append(summary_path)
    return paths
