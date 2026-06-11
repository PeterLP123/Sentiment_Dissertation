import json
import sqlite3
import sys
from types import SimpleNamespace

from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, EvaluationResult, LLMResponseRecord, PromptConfig, RunConfig
from sentiment_benchmark.storage import BenchmarkStore


def test_storage_insert_and_resume_check(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    record = LLMResponseRecord(
        row_number=2,
        model_id="test/model",
        prompt_hash="abc123",
        raw_content="positive",
        normalized_label="positive",
        parse_status="valid",
        status="success",
    )
    assert not store.response_exists(1, "test/model", 2, "abc123")
    store.save_response(1, record)
    assert store.response_exists(1, "test/model", 2, "abc123")
    store.save_response(1, record)
    responses = store.fetch_responses(1, "test/model")
    assert len(responses) == 1


def _prompt() -> PromptConfig:
    return PromptConfig(
        prompt_id="test",
        system_prompt="Return a label.",
        user_template="{sentence}",
        output_mode="label_only",
        prompt_hash="prompt-hash",
    )


def test_create_run_records_machine_and_environment_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SENTIMENT_BENCH_MACHINE_ID", "lab desktop")
    monkeypatch.setenv("SENTIMENT_BENCH_MACHINE_LABEL", "RTX 3070")
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    prompt = _prompt()
    config = RunConfig(
        models=["gemma4:12b"],
        prompt=prompt,
        mode="pilot",
        dataset_path="Data/data.csv",
        db_path=str(tmp_path / "test.sqlite"),
        base_url="http://localhost:11434",
        provider="ollama",
    )

    run_id = store.create_run(config, [1, 2, 3])

    with sqlite3.connect(tmp_path / "test.sqlite") as connection:
        row = connection.execute(
            "SELECT machine_id, machine_label, environment_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    environment = json.loads(row[2])
    assert row[0] == "lab-desktop"
    assert row[1] == "RTX 3070"
    assert environment["machine"]["id"] == "lab-desktop"
    assert environment["machine"]["label"] == "RTX 3070"
    assert environment["database"]["backend"] == "sqlite"
    assert environment["package_version"] == "0.1.0"


def test_initialize_migrates_old_runs_table_for_environment_metadata(tmp_path) -> None:
    db_path = tmp_path / "old.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
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
            )
            """
        )
        connection.commit()

    store = BenchmarkStore(db_path)
    store.initialize()

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
    assert {"machine_id", "machine_label", "environment_json"} <= columns


def test_storage_label_describes_active_backend(monkeypatch, tmp_path) -> None:
    sqlite_store = BenchmarkStore(tmp_path / "test.sqlite")
    assert sqlite_store.storage_label() == f"SQLite: {tmp_path / 'test.sqlite'}"

    monkeypatch.setenv("SENTIMENT_BENCH_DB_BACKEND", "libsql")
    monkeypatch.delenv("TURSO_CONNECTION_MODE", raising=False)
    monkeypatch.setenv("TURSO_REPLICA_PATH", str(tmp_path / "replica.db"))
    replica_store = BenchmarkStore(tmp_path / "ignored.sqlite")
    assert replica_store.storage_label() == f"Turso/libSQL replica: {tmp_path / 'replica.db'} (syncs to hosted)"

    monkeypatch.setenv("TURSO_CONNECTION_MODE", "hosted")
    hosted_store = BenchmarkStore(tmp_path / "ignored.sqlite")
    assert hosted_store.storage_label() == "Turso/libSQL hosted"


def test_upsert_dataset_records_snapshot(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    rows = [
        DatasetRow(1, "a", "positive", False, False, 1),
        DatasetRow(2, "b", "negative", False, False, 1),
    ]

    store.initialize()
    assert store.dataset_snapshot_exists("Data/data.csv", len(rows)) is False

    store.upsert_dataset(rows, "Data/data.csv")

    assert store.dataset_snapshot_exists("Data/data.csv", len(rows)) is True
    assert store.dataset_snapshot_exists("Data/data.csv", len(rows) + 1) is False


def test_upsert_dataset_batches_inserts_instead_of_one_per_row(tmp_path, monkeypatch) -> None:
    """Bulk dataset writes must use multi-row INSERTs, not one statement per row.

    The libSQL/Turso backend forwards every statement to the remote primary as
    its own network round-trip, so a per-row write of a full dataset (~5,800
    rows) takes ~20 minutes and looks like a frozen run. This guards against a
    regression back to a per-row ``execute``/``executemany`` by asserting the
    number of dataset INSERT statements scales by chunk, not by row count.
    """
    from contextlib import contextmanager

    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()

    insert_statements = 0

    class CountingConnection:
        def __init__(self, inner) -> None:
            self._inner = inner

        def execute(self, sql, params=None):
            nonlocal insert_statements
            if "INSERT INTO dataset_items" in sql:
                insert_statements += 1
            return self._inner.execute(sql) if params is None else self._inner.execute(sql, params)

        def executemany(self, sql, seq):
            nonlocal insert_statements
            seq = list(seq)
            if "INSERT INTO dataset_items" in sql:
                # executemany against libSQL is a per-row round-trip -- the very
                # pattern this test exists to forbid -- so count each row.
                insert_statements += len(seq)
            return self._inner.executemany(sql, seq)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    real_connect = store.connect

    @contextmanager
    def counting_connect():
        with real_connect() as inner:
            yield CountingConnection(inner)

    monkeypatch.setattr(store, "connect", counting_connect)

    def make_rows(count: int):
        return [DatasetRow(i, f"sentence {i}", "positive", False, False, 1) for i in range(1, count + 1)]

    rows = make_rows(500)
    insert_statements = 0
    store.upsert_dataset(rows, "Data/data.csv")

    # A per-row write would emit 500 statements; batching keeps it to a handful
    # (>= 10 rows per statement). The exact chunk size may be tuned, so assert the
    # property, not a magic number.
    assert insert_statements <= len(rows) // 10
    # And the rows must actually be written correctly through the batched path.
    with sqlite3.connect(tmp_path / "test.sqlite") as connection:
        written = connection.execute("SELECT COUNT(*) FROM dataset_items").fetchone()[0]
    assert written == len(rows)


def test_sync_backend_syncs_libsql(monkeypatch, tmp_path) -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.synced = 0
            self.committed = 0
            self.closed = False

        def sync(self) -> None:
            self.synced += 1

        def commit(self) -> None:
            self.committed += 1

        def close(self) -> None:
            self.closed = True

    fake_connection = FakeConnection()

    def fake_connect(path, **kwargs):
        return fake_connection

    monkeypatch.setitem(sys.modules, "libsql", SimpleNamespace(connect=fake_connect))
    monkeypatch.setenv("SENTIMENT_BENCH_DB_BACKEND", "libsql")
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://example.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TURSO_REPLICA_PATH", str(tmp_path / "replica.db"))

    store = BenchmarkStore(tmp_path / "ignored.sqlite")

    assert store.sync_backend() is True
    assert fake_connection.synced == 1
    assert fake_connection.committed == 1
    assert fake_connection.closed is True


def test_metrics_counts_invalid_as_wrong_and_excludes_conflicts() -> None:
    rows = [
        DatasetRow(2, "a", "positive", False, False, 1),
        DatasetRow(3, "b", "negative", False, False, 1),
        DatasetRow(4, "c", "neutral", True, True, 2),
    ]
    responses = [
        {
            "row_number": 2,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "positive",
            "latency_ms": 10,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 3,
            "status": "success",
            "parse_status": "invalid",
            "normalized_label": None,
            "latency_ms": 20,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 4,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "negative",
            "latency_ms": 30,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    ]
    primary = evaluate_responses(rows, responses, "test/model", scope="primary")
    audit = evaluate_responses(rows, responses, "test/model", scope="all")
    assert primary.row_count == 2
    assert primary.accuracy == 0.5
    assert primary.balanced_accuracy == 0.5
    assert primary.mcc == 0.0
    assert primary.invalid_output_count == 1
    assert audit.row_count == 3


def test_balanced_accuracy_and_mcc_count_invalid_as_wrong() -> None:
    rows = [
        DatasetRow(1, "a", "positive", False, False, 1),
        DatasetRow(2, "b", "negative", False, False, 1),
        DatasetRow(3, "c", "neutral", False, False, 1),
        DatasetRow(4, "d", "positive", False, False, 1),
    ]
    responses = [
        {
            "row_number": 1,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "positive",
            "latency_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 2,
            "status": "success",
            "parse_status": "invalid",
            "normalized_label": None,
            "latency_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 3,
            "status": "success",
            "parse_status": "valid",
            "normalized_label": "neutral",
            "latency_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        {
            "row_number": 4,
            "status": "api_error",
            "parse_status": "error",
            "normalized_label": None,
            "latency_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    ]
    result = evaluate_responses(rows, responses, "test/model", scope="primary")
    assert result.accuracy == 0.5
    # Scored only over ALLOWED_LABELS: invalid/error map to a wrong class, not extra categories.
    assert result.balanced_accuracy == 0.5
    assert result.mcc == 0.2


def _eval_result(model_id: str, scope: str, accuracy: float) -> EvaluationResult:
    return EvaluationResult(
        model_id=model_id,
        scope=scope,
        row_count=10,
        accuracy=accuracy,
        balanced_accuracy=accuracy,
        mcc=accuracy,
        macro_f1=accuracy,
        weighted_f1=accuracy,
        per_class={},
        confusion_matrix={},
        invalid_output_count=0,
        api_error_count=0,
        mean_latency_ms=100.0,
        total_prompt_tokens=10,
        total_completion_tokens=10,
        total_tokens=20,
    )


def test_fetch_metrics_returns_all_scopes(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    store.save_metrics(1, _eval_result("a/model", "primary", 0.9))
    store.save_metrics(1, _eval_result("a/model", "all", 0.8))
    store.save_metrics(2, _eval_result("b/model", "primary", 0.5))

    rows = store.fetch_metrics(1)
    assert {(row["model_id"], row["scope"]) for row in rows} == {("a/model", "primary"), ("a/model", "all")}


def test_run_cost_by_model_sums_recorded_generation_cost(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    store.save_generation_metadata(1, 2, "a/model", "gen-1", {"total_cost": 0.001})
    store.save_generation_metadata(1, 3, "a/model", "gen-2", {"total_cost": 0.002})
    store.save_generation_metadata(1, 4, "b/model", "gen-3", {"total_cost": 0.005})
    # Missing cost should be ignored rather than counted as zero.
    store.save_generation_metadata(1, 5, "b/model", "gen-4", {})

    costs = store.run_cost_by_model(1)
    assert costs["a/model"] == 0.003
    assert costs["b/model"] == 0.005
    assert store.run_cost_by_model(99) == {}



def test_reconcile_orphaned_runs_marks_running_as_interrupted(tmp_path) -> None:
    store = BenchmarkStore(tmp_path / "test.sqlite")
    store.initialize()
    config = RunConfig(
        models=["gemma4:12b"],
        prompt=_prompt(),
        mode="pilot",
        dataset_path="Data/data.csv",
        db_path=str(tmp_path / "test.sqlite"),
        base_url="http://localhost:11434",
        provider="ollama",
    )
    orphan_id = store.create_run(config, [1, 2, 3])
    done_id = store.create_run(config, [1])
    store.mark_run_complete(done_id)

    assert store.reconcile_orphaned_runs() == [orphan_id]

    with sqlite3.connect(tmp_path / "test.sqlite") as connection:
        statuses = dict(connection.execute("SELECT id, status FROM runs").fetchall())
    assert statuses[orphan_id] == "interrupted"
    assert statuses[done_id] == "completed"
    # Idempotent: a second pass finds nothing left to reconcile.
    assert store.reconcile_orphaned_runs() == []
