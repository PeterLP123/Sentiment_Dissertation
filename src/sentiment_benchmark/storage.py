from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .models import DatasetRow, EvaluationResult, LLMResponseRecord, PromptConfig, RunConfig


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class BenchmarkStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_items (
                    row_number INTEGER PRIMARY KEY,
                    sentence TEXT NOT NULL,
                    hidden_label TEXT NOT NULL,
                    is_duplicate INTEGER NOT NULL,
                    has_conflicting_duplicate INTEGER NOT NULL,
                    duplicate_group_size INTEGER NOT NULL,
                    dataset_path TEXT NOT NULL,
                    loaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prompts (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    output_mode TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_template TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    mode TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    output_mode TEXT NOT NULL,
                    models_json TEXT NOT NULL,
                    selected_rows_json TEXT NOT NULL,
                    dataset_path TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_models (
                    run_id INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    PRIMARY KEY (run_id, model_id)
                );

                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    row_number INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    raw_content TEXT,
                    normalized_label TEXT,
                    parse_status TEXT NOT NULL,
                    explanation TEXT,
                    raw_response_json TEXT,
                    latency_ms REAL,
                    status TEXT NOT NULL,
                    error TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    generation_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, row_number, model_id, prompt_hash)
                );

                CREATE TABLE IF NOT EXISTS generation_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    row_number INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    total_cost REAL,
                    provider_name TEXT,
                    latency_ms REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, row_number, model_id, generation_id)
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, model_id, scope)
                );
                """
            )

    def upsert_dataset(self, rows: list[DatasetRow], dataset_path: str) -> None:
        loaded_at = utc_now()
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO dataset_items (
                    row_number, sentence, hidden_label, is_duplicate, has_conflicting_duplicate,
                    duplicate_group_size, dataset_path, loaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(row_number) DO UPDATE SET
                    sentence=excluded.sentence,
                    hidden_label=excluded.hidden_label,
                    is_duplicate=excluded.is_duplicate,
                    has_conflicting_duplicate=excluded.has_conflicting_duplicate,
                    duplicate_group_size=excluded.duplicate_group_size,
                    dataset_path=excluded.dataset_path,
                    loaded_at=excluded.loaded_at
                """,
                [
                    (
                        row.row_number,
                        row.sentence,
                        row.hidden_label,
                        int(row.is_duplicate),
                        int(row.has_conflicting_duplicate),
                        row.duplicate_group_size,
                        dataset_path,
                        loaded_at,
                    )
                    for row in rows
                ],
            )

    def save_prompt(self, prompt: PromptConfig) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO prompts (
                    prompt_hash, prompt_id, output_mode, system_prompt, user_template, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(prompt_hash) DO UPDATE SET
                    prompt_id=excluded.prompt_id,
                    output_mode=excluded.output_mode,
                    system_prompt=excluded.system_prompt,
                    user_template=excluded.user_template
                """,
                (
                    prompt.prompt_hash,
                    prompt.prompt_id,
                    prompt.output_mode,
                    prompt.system_prompt,
                    prompt.user_template,
                    utc_now(),
                ),
            )

    def create_run(self, config: RunConfig, selected_row_numbers: list[int]) -> int:
        request = {
            "temperature": config.temperature,
            "max_completion_tokens": config.max_completion_tokens,
            "concurrency": config.concurrency,
            "retries": config.retries,
            "sample_per_class": config.sample_per_class,
            "seed": config.seed,
        }
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    created_at, mode, prompt_hash, output_mode, models_json, selected_rows_json,
                    dataset_path, base_url, request_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    config.mode,
                    config.prompt.prompt_hash,
                    config.prompt.output_mode,
                    json.dumps(config.models),
                    json.dumps(selected_row_numbers),
                    config.dataset_path,
                    config.base_url,
                    json.dumps(request, sort_keys=True),
                    "running",
                ),
            )
            return int(cursor.lastrowid)

    def get_run_selected_rows(self, run_id: int) -> list[int]:
        with self.connect() as connection:
            row = connection.execute("SELECT selected_rows_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} does not exist")
        return [int(value) for value in json.loads(row["selected_rows_json"])]

    def mark_run_complete(self, run_id: int, status: str = "completed") -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE id = ?",
                (status, utc_now(), run_id),
            )

    def upsert_run_model(self, run_id: int, model_id: str, status: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO run_models (run_id, model_id, status, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, model_id) DO UPDATE SET
                    status=excluded.status,
                    completed_at=excluded.completed_at
                """,
                (run_id, model_id, status, now if status == "running" else None, now if status == "completed" else None),
            )

    def response_exists(self, run_id: int, model_id: str, row_number: int, prompt_hash: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM responses
                WHERE run_id = ? AND model_id = ? AND row_number = ? AND prompt_hash = ?
                """,
                (run_id, model_id, row_number, prompt_hash),
            ).fetchone()
        return row is not None

    def save_response(self, run_id: int, record: LLMResponseRecord) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO responses (
                    run_id, row_number, model_id, prompt_hash, raw_content, normalized_label,
                    parse_status, explanation, raw_response_json, latency_ms, status, error,
                    prompt_tokens, completion_tokens, total_tokens, generation_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, row_number, model_id, prompt_hash) DO UPDATE SET
                    raw_content=excluded.raw_content,
                    normalized_label=excluded.normalized_label,
                    parse_status=excluded.parse_status,
                    explanation=excluded.explanation,
                    raw_response_json=excluded.raw_response_json,
                    latency_ms=excluded.latency_ms,
                    status=excluded.status,
                    error=excluded.error,
                    prompt_tokens=excluded.prompt_tokens,
                    completion_tokens=excluded.completion_tokens,
                    total_tokens=excluded.total_tokens,
                    generation_id=excluded.generation_id
                """,
                (
                    run_id,
                    record.row_number,
                    record.model_id,
                    record.prompt_hash,
                    record.raw_content,
                    record.normalized_label,
                    record.parse_status,
                    record.explanation,
                    json.dumps(record.raw_response_json) if record.raw_response_json is not None else None,
                    record.latency_ms,
                    record.status,
                    record.error,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.generation_id,
                    utc_now(),
                ),
            )

    def save_generation_metadata(
        self,
        run_id: int,
        row_number: int,
        model_id: str,
        generation_id: str,
        metadata: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_metadata (
                    run_id, row_number, model_id, generation_id, metadata_json,
                    total_cost, provider_name, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, row_number, model_id, generation_id) DO UPDATE SET
                    metadata_json=excluded.metadata_json,
                    total_cost=excluded.total_cost,
                    provider_name=excluded.provider_name,
                    latency_ms=excluded.latency_ms
                """,
                (
                    run_id,
                    row_number,
                    model_id,
                    generation_id,
                    json.dumps(metadata, sort_keys=True),
                    metadata.get("total_cost"),
                    metadata.get("provider_name"),
                    metadata.get("latency"),
                    utc_now(),
                ),
            )

    def fetch_responses(self, run_id: int, model_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM responses
                WHERE run_id = ? AND model_id = ?
                ORDER BY row_number
                """,
                (run_id, model_id),
            ).fetchall()
        return list(rows)

    def save_metrics(self, run_id: int, result: EvaluationResult) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO metrics (run_id, model_id, scope, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, model_id, scope) DO UPDATE SET
                    metrics_json=excluded.metrics_json,
                    created_at=excluded.created_at
                """,
                (run_id, result.model_id, result.scope, json.dumps(asdict(result), sort_keys=True), utc_now()),
            )

    def list_runs(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, completed_at, mode, output_mode, models_json, status
                FROM runs
                ORDER BY id DESC
                LIMIT 50
                """
            ).fetchall()
        return list(rows)

