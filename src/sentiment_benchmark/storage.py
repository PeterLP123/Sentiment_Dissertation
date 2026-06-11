from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .env import load_env_file
from .libsql_backend import LibsqlConnection
from .models import DatasetRow, EvaluationResult, LLMResponseRecord, PromptConfig, RunConfig, RunResumeSettings
from .runtime_metadata import collect_run_environment
from .self_consistency import (
    RowSelfConsistency,
    SelfConsistencyResult,
    compute_row_consistency,
    compute_self_consistency,
)

# libSQL/Turso embedded-replica connections cannot be accessed concurrently:
# two open connections to the same local replica file deadlock on the replica
# file lock and hang forever (the connect `timeout` only covers SQL busy waits,
# not this). In the TUI a benchmark worker thread continuously opens connections
# to write responses while the UI thread opens connections to read the runs
# table / leaderboard for refreshes -- the two collide and freeze the app. This
# process-wide lock serializes every libSQL connection so they never overlap.
# Plain SQLite uses its own busy-timeout file locking and is left unguarded.
_LIBSQL_LOCK = threading.RLock()

# Cap the number of bound parameters in a single multi-row INSERT so it stays
# well under SQLite's variable limit (SQLITE_MAX_VARIABLE_NUMBER) across versions.
_MAX_BULK_PARAMS = 900


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _chunked(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for index in range(0, len(seq), size):
        yield seq[index : index + size]


def _table_columns(connection: Any, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _ensure_column(connection: Any, table_name: str, column_name: str, column_sql: str) -> None:
    if column_name not in _table_columns(connection, table_name):
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


class BenchmarkStore:
    def __init__(self, db_path: str | Path) -> None:
        load_env_file()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = os.getenv("SENTIMENT_BENCH_DB_BACKEND", "sqlite").strip().lower()

    def storage_label(self) -> str:
        if self.backend in {"libsql", "turso"}:
            mode = os.getenv("TURSO_CONNECTION_MODE", "replica").strip().lower() or "replica"
            if mode == "hosted":
                return "Turso/libSQL hosted"
            replica_path = os.getenv("TURSO_REPLICA_PATH", "results/turso_replica.db").strip()
            return f"Turso/libSQL replica: {replica_path} (syncs to hosted)"
        return f"SQLite: {self.db_path}"

    @contextmanager
    def connect(self) -> Iterator[Any]:
        use_lock = self.backend in {"libsql", "turso"}
        if use_lock:
            _LIBSQL_LOCK.acquire()
        try:
            if self.backend in {"libsql", "turso"}:
                connection = LibsqlConnection.from_env()
            elif self.backend in {"sqlite", ""}:
                connection = sqlite3.connect(self.db_path)
                connection.row_factory = sqlite3.Row
            else:
                raise ValueError("SENTIMENT_BENCH_DB_BACKEND must be 'sqlite' or 'libsql'")
            try:
                yield connection
                connection.commit()
            finally:
                connection.close()
        finally:
            if use_lock:
                _LIBSQL_LOCK.release()

    def sync_backend(self) -> bool:
        if self.backend not in {"libsql", "turso"}:
            return False
        with self.connect() as connection:
            sync = getattr(connection, "sync", None)
            if not callable(sync):
                return False
            synced = sync()
            return bool(synced) if synced is not None else True

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

                CREATE TABLE IF NOT EXISTS dataset_snapshots (
                    dataset_path TEXT PRIMARY KEY,
                    row_count INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL
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
                    machine_id TEXT,
                    machine_label TEXT,
                    environment_json TEXT NOT NULL DEFAULT '{}',
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

                CREATE TABLE IF NOT EXISTS sc_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    model_id TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    num_samples INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    dataset_path TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'primary',
                    status TEXT NOT NULL DEFAULT 'running',
                    total_cost REAL
                );

                CREATE TABLE IF NOT EXISTS sc_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sc_run_id INTEGER NOT NULL,
                    row_number INTEGER NOT NULL,
                    sample_index INTEGER NOT NULL,
                    normalized_label TEXT,
                    parse_status TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    generation_id TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (sc_run_id) REFERENCES sc_runs(id)
                );

                CREATE TABLE IF NOT EXISTS sc_results (
                    sc_run_id INTEGER PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    num_samples INTEGER NOT NULL,
                    scope TEXT NOT NULL,
                    n_rows INTEGER NOT NULL,
                    mean_entropy REAL NOT NULL,
                    median_entropy REAL NOT NULL,
                    mean_majority_fraction REAL NOT NULL,
                    consistency_rate REAL NOT NULL,
                    majority_vote_accuracy REAL NOT NULL,
                    majority_vote_balanced_accuracy REAL NOT NULL,
                    conflicting_entropy REAL,
                    non_conflicting_entropy REAL,
                    conflicting_rows INTEGER NOT NULL DEFAULT 0,
                    non_conflicting_rows INTEGER NOT NULL DEFAULT 0,
                    total_cost REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (sc_run_id) REFERENCES sc_runs(id)
                );
                """
            )
            self._migrate_schema(connection)

    def _migrate_schema(self, connection: Any) -> None:
        _ensure_column(connection, "runs", "machine_id", "TEXT")
        _ensure_column(connection, "runs", "machine_label", "TEXT")
        _ensure_column(connection, "runs", "environment_json", "TEXT NOT NULL DEFAULT '{}'")

    def upsert_dataset(self, rows: list[DatasetRow], dataset_path: str) -> None:
        loaded_at = utc_now()
        columns = (
            "row_number",
            "sentence",
            "hidden_label",
            "is_duplicate",
            "has_conflicting_duplicate",
            "duplicate_group_size",
            "dataset_path",
            "loaded_at",
        )
        records = [
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
        ]
        placeholder = "(" + ", ".join(["?"] * len(columns)) + ")"
        conflict_clause = """
                ON CONFLICT(row_number) DO UPDATE SET
                    sentence=excluded.sentence,
                    hidden_label=excluded.hidden_label,
                    is_duplicate=excluded.is_duplicate,
                    has_conflicting_duplicate=excluded.has_conflicting_duplicate,
                    duplicate_group_size=excluded.duplicate_group_size,
                    dataset_path=excluded.dataset_path,
                    loaded_at=excluded.loaded_at
        """
        chunk_size = max(1, _MAX_BULK_PARAMS // len(columns))
        with self.connect() as connection:
            # libSQL/Turso embedded replicas forward every statement to the remote
            # primary, so a per-row executemany of the whole dataset becomes
            # thousands of network round-trips (minutes -- the run looks frozen).
            # Batch into multi-row INSERTs so each chunk is a single round-trip.
            for chunk in _chunked(records, chunk_size):
                values_sql = ", ".join([placeholder] * len(chunk))
                connection.execute(
                    f"INSERT INTO dataset_items ({', '.join(columns)}) VALUES {values_sql}"
                    + conflict_clause,
                    [value for record in chunk for value in record],
                )
            connection.execute(
                """
                INSERT INTO dataset_snapshots (dataset_path, row_count, recorded_at)
                VALUES (?, ?, ?)
                ON CONFLICT(dataset_path) DO UPDATE SET
                    row_count=excluded.row_count,
                    recorded_at=excluded.recorded_at
                """,
                (dataset_path, len(rows), loaded_at),
            )

    def dataset_snapshot_exists(self, dataset_path: str, row_count: int) -> bool:
        with self.connect() as connection:
            try:
                row = connection.execute(
                    "SELECT row_count FROM dataset_snapshots WHERE dataset_path = ?",
                    (dataset_path,),
                ).fetchone()
            except Exception:
                row = None
            if row is not None and int(row["row_count"]) == row_count:
                return True

            # Backward-compatible fallback for databases populated before
            # dataset_snapshots existed. This avoids re-uploading thousands of
            # unchanged source rows to hosted libSQL on every run.
            row = connection.execute(
                "SELECT COUNT(*) AS row_count FROM dataset_items WHERE dataset_path = ?",
                (dataset_path,),
            ).fetchone()
        return row is not None and int(row["row_count"]) == row_count

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
        environment = collect_run_environment()
        machine = environment.get("machine") if isinstance(environment.get("machine"), dict) else {}
        request = {
            "provider": config.provider,
            "temperature": config.temperature,
            "max_completion_tokens": config.max_completion_tokens,
            "reasoning_max_completion_tokens": config.reasoning_max_completion_tokens,
            "model_max_completion_tokens": config.model_max_completion_tokens,
            "concurrency": config.concurrency,
            "retries": config.retries,
            "ollama_think": config.ollama_think,
            "sample_per_class": config.sample_per_class,
            "seed": config.seed,
            "few_shot_k": config.few_shot_k,
            "few_shot_seed": config.few_shot_seed,
        }
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    created_at, mode, prompt_hash, output_mode, models_json, selected_rows_json,
                    dataset_path, base_url, request_json, machine_id, machine_label, environment_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    machine.get("id"),
                    machine.get("label"),
                    json.dumps(environment, sort_keys=True),
                    "running",
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Could not create benchmark run")
            return int(cursor.lastrowid)

    def get_run_selected_rows(self, run_id: int) -> list[int]:
        with self.connect() as connection:
            row = connection.execute("SELECT selected_rows_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} does not exist")
        return [int(value) for value in json.loads(row["selected_rows_json"])]

    def get_run_resume_settings(self, run_id: int) -> RunResumeSettings:
        """Stored prompt hash and few-shot settings required to resume a run safely."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT prompt_hash, request_json FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} does not exist")
        request = json.loads(row["request_json"])
        few_shot_seed = request.get("few_shot_seed")
        return RunResumeSettings(
            prompt_hash=str(row["prompt_hash"]),
            few_shot_k=int(request.get("few_shot_k", 0)),
            few_shot_seed=int(few_shot_seed) if few_shot_seed is not None else None,
        )

    def run_prompt_id(self, run_id: int) -> str | None:
        """The human-readable prompt id used by a run (joined via prompt_hash)."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT p.prompt_id
                FROM runs r
                JOIN prompts p ON p.prompt_hash = r.prompt_hash
                WHERE r.id = ?
                """,
                (run_id,),
            ).fetchone()
        return row["prompt_id"] if row is not None else None

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
                (
                    run_id,
                    model_id,
                    status,
                    now if status == "running" else None,
                    now if status in {"completed", "cancelled"} else None,
                ),
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

    def successful_response_exists(self, run_id: int, model_id: str, row_number: int, prompt_hash: str) -> bool:
        """True only if a *successful* response is already stored.

        Resuming a run relies on this so that previously failed rows
        (api_error/transport_error/malformed_response) are re-attempted instead
        of being treated as permanently complete.
        """
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM responses
                WHERE run_id = ? AND model_id = ? AND row_number = ? AND prompt_hash = ?
                  AND status = 'success'
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

    def fetch_misclassifications(
        self,
        run_id: int,
        model_id: str | None = None,
        scope: str = "all",
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        filters = ["r.run_id = ?"]
        params: list[Any] = [run_id]
        if model_id:
            filters.append("r.model_id = ?")
            params.append(model_id)
        if scope == "primary":
            filters.append("d.has_conflicting_duplicate = 0")
        params.append(limit)
        where = " AND ".join(filters)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    r.row_number,
                    r.model_id,
                    d.hidden_label,
                    r.normalized_label,
                    r.parse_status,
                    r.status,
                    r.raw_content,
                    r.error,
                    r.latency_ms,
                    d.sentence,
                    d.has_conflicting_duplicate,
                    d.duplicate_group_size
                FROM responses r
                JOIN dataset_items d ON d.row_number = r.row_number
                WHERE {where}
                  AND (
                    r.status != 'success'
                    OR r.parse_status != 'valid'
                    OR r.normalized_label IS NULL
                    OR r.normalized_label != d.hidden_label
                  )
                ORDER BY r.model_id, r.row_number
                LIMIT ?
                """,
                params,
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

    def fetch_predictions(self, run_id: int, model_id: str, scope: str = "all") -> list[sqlite3.Row]:
        """Aligned (gold, prediction) rows for one model, used for paired comparison."""
        filters = ["r.run_id = ?", "r.model_id = ?"]
        params: list[Any] = [run_id, model_id]
        if scope == "primary":
            filters.append("d.has_conflicting_duplicate = 0")
        where = " AND ".join(filters)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT r.row_number, d.hidden_label, r.normalized_label, r.parse_status, r.status
                FROM responses r
                JOIN dataset_items d ON d.row_number = r.row_number
                WHERE {where}
                ORDER BY r.row_number
                """,
                params,
            ).fetchall()
        return list(rows)

    def run_model_ids(self, run_id: int) -> list[str]:
        """Distinct model ids that produced responses for a run, in stable order."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT model_id FROM responses WHERE run_id = ? ORDER BY model_id",
                (run_id,),
            ).fetchall()
        return [row["model_id"] for row in rows]

    def fetch_metrics(self, run_id: int) -> list[sqlite3.Row]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT model_id, scope, metrics_json FROM metrics WHERE run_id = ? ORDER BY model_id, scope",
                (run_id,),
            ).fetchall()
        return list(rows)

    def fetch_all_metrics(self) -> list[sqlite3.Row]:
        """All metric rows across every run (newest run first) for cross-run aggregation."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT run_id, model_id, scope, metrics_json FROM metrics ORDER BY run_id DESC, model_id, scope"
            ).fetchall()
        return list(rows)

    def run_cost_by_model(self, run_id: int) -> dict[str, float]:
        """Sum observed provider cost per model for a run, if recorded."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT model_id, SUM(total_cost) AS cost
                FROM generation_metadata
                WHERE run_id = ? AND total_cost IS NOT NULL
                GROUP BY model_id
                """,
                (run_id,),
            ).fetchall()
        return {row["model_id"]: float(row["cost"]) for row in rows if row["cost"] is not None}

    def list_runs(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            columns = _table_columns(connection, "runs")
            machine_select = (
                "machine_id, machine_label"
                if {"machine_id", "machine_label"} <= columns
                else "NULL AS machine_id, NULL AS machine_label"
            )
            rows = connection.execute(
                f"""
                SELECT id, created_at, completed_at, mode, output_mode, models_json, status, {machine_select}
                FROM runs
                ORDER BY id DESC
                LIMIT 50
                """
            ).fetchall()
        return list(rows)

    # ------------------------------------------------------------------
    # Self-consistency run storage
    # ------------------------------------------------------------------

    def create_sc_run(
        self,
        *,
        model_id: str,
        temperature: float,
        num_samples: int,
        mode: str,
        dataset_path: str,
        prompt_hash: str,
        scope: str = "primary",
    ) -> int:
        """Create a new self-consistency run entry and return its id."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sc_runs (created_at, model_id, temperature, num_samples, mode,
                                     dataset_path, prompt_hash, scope, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (utc_now(), model_id, temperature, num_samples, mode,
                 dataset_path, prompt_hash, scope, "running"),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Could not create self-consistency run")
            return int(cursor.lastrowid)

    def mark_sc_run_complete(
        self,
        sc_run_id: int,
        *,
        status: str = "completed",
        total_cost: float | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE sc_runs SET status = ?, completed_at = ?, total_cost = ? WHERE id = ?",
                (status, utc_now(), total_cost, sc_run_id),
            )

    def save_sc_sample(
        self,
        sc_run_id: int,
        *,
        row_number: int,
        sample_index: int,
        normalized_label: str | None,
        parse_status: str,
        status: str,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        generation_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sc_samples (
                    sc_run_id, row_number, sample_index, normalized_label,
                    parse_status, status, latency_ms, prompt_tokens,
                    completion_tokens, total_tokens, generation_id,
                    error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sc_run_id, row_number, sample_index, normalized_label,
                    parse_status, status, latency_ms,
                    prompt_tokens, completion_tokens, total_tokens,
                    generation_id, error, utc_now(),
                ),
            )

    def save_sc_result(self, sc_run_id: int, result: SelfConsistencyResult) -> None:
        """Store aggregated self-consistency metrics."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sc_results (
                    sc_run_id, model_id, temperature, num_samples, scope, n_rows,
                    mean_entropy, median_entropy, mean_majority_fraction,
                    consistency_rate, majority_vote_accuracy,
                    majority_vote_balanced_accuracy,
                    conflicting_entropy, non_conflicting_entropy,
                    conflicting_rows, non_conflicting_rows, total_cost, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sc_run_id) DO UPDATE SET
                    mean_entropy=excluded.mean_entropy,
                    median_entropy=excluded.median_entropy,
                    mean_majority_fraction=excluded.mean_majority_fraction,
                    consistency_rate=excluded.consistency_rate,
                    majority_vote_accuracy=excluded.majority_vote_accuracy,
                    majority_vote_balanced_accuracy=excluded.majority_vote_balanced_accuracy,
                    conflicting_entropy=excluded.conflicting_entropy,
                    non_conflicting_entropy=excluded.non_conflicting_entropy,
                    conflicting_rows=excluded.conflicting_rows,
                    non_conflicting_rows=excluded.non_conflicting_rows,
                    total_cost=excluded.total_cost
                """,
                (
                    sc_run_id, result.model_id, result.temperature,
                    result.num_samples, result.scope, result.n_rows,
                    result.mean_entropy, result.median_entropy,
                    result.mean_majority_fraction, result.consistency_rate,
                    result.majority_vote_accuracy,
                    result.majority_vote_balanced_accuracy,
                    result.conflicting_entropy, result.non_conflicting_entropy,
                    result.conflicting_rows, result.non_conflicting_rows,
                    result.total_cost, utc_now(),
                ),
            )

    def fetch_sc_samples(self, sc_run_id: int) -> list[sqlite3.Row]:
        """All raw samples for a self-consistency run, ordered by row then sample."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM sc_samples
                WHERE sc_run_id = ?
                ORDER BY row_number, sample_index
                """,
                (sc_run_id,),
            ).fetchall()
        return list(rows)

    def fetch_sc_result(self, sc_run_id: int) -> sqlite3.Row | None:
        """Stored aggregate result for a self-consistency run."""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM sc_results WHERE sc_run_id = ?", (sc_run_id,)
            ).fetchone()

    def sc_run_by_id(self, sc_run_id: int) -> sqlite3.Row | None:
        """Metadata for a self-consistency run."""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM sc_runs WHERE id = ?", (sc_run_id,)
            ).fetchone()

    def list_sc_runs(self) -> list[sqlite3.Row]:
        """List self-consistency runs (most recent first)."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, completed_at, model_id, temperature,
                       num_samples, mode, status, total_cost
                FROM sc_runs
                ORDER BY id DESC
                LIMIT 50
                """
            ).fetchall()
        return list(rows)

    def fetch_sc_row_samples(
        self, sc_run_id: int
    ) -> dict[int, list[str]]:
        """Group all sample labels by row_number for a sc_run.

        Returns {row_number: [label_0, label_1, ..., label_{N-1}]} with
        invalid/error labels included (caller filters as needed).
        """
        samples = self.fetch_sc_samples(sc_run_id)
        by_row: dict[int, list[str]] = {}
        for row in samples:
            rn = int(row["row_number"])
            label = row["normalized_label"] if row["normalized_label"] else "__invalid__"
            by_row.setdefault(rn, []).append(str(label))
        return by_row

    def build_sc_row_consistency(
        self,
        sc_run_id: int,
        *,
        model_id: str,
        temperature: float,
        num_samples: int,
        scope: str = "primary",
    ) -> SelfConsistencyResult:
        """Rebuild SelfConsistencyResult from raw samples in the DB.

        Looks up dataset rows to get hidden labels and conflict metadata,
        then computes per-row entropy and aggregate statistics.
        """
        samples_by_row = self.fetch_sc_row_samples(sc_run_id)
        run_info = self.sc_run_by_id(sc_run_id)
        if run_info is None:
            raise ValueError(f"Self-consistency run {sc_run_id} not found")

        # Load dataset rows
        dataset_path = str(run_info["dataset_path"])
        from .dataset import load_dataset
        all_rows = load_dataset(dataset_path)
        row_map = {r.row_number: r for r in all_rows}

        rows_consistency: list[RowSelfConsistency] = []
        for rn in sorted(samples_by_row):
            ds_row = row_map.get(rn)
            if ds_row is None:
                continue
            if scope == "primary" and ds_row.has_conflicting_duplicate:
                continue

            row_cons = compute_row_consistency(
                samples_by_row[rn],
                row_number=rn,
                sentence=ds_row.sentence,
                hidden_label=ds_row.hidden_label,
                is_conflicting_duplicate=ds_row.has_conflicting_duplicate,
            )
            rows_consistency.append(row_cons)

        result = compute_self_consistency(
            model_id=model_id,
            temperature=temperature,
            num_samples=num_samples,
            scope=scope,
            rows=rows_consistency,
            total_cost=float(run_info["total_cost"]) if run_info["total_cost"] else None,
        )
        return result
