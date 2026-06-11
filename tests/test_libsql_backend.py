import sys
from types import SimpleNamespace

import pytest

from sentiment_benchmark.libsql_backend import LibsqlConnection, ResultRow, split_sql_script


def test_result_row_supports_name_and_index_access() -> None:
    row = ResultRow(["id", "model"], [7, "gemma4:12b"])

    assert row[0] == 7
    assert row["model"] == "gemma4:12b"
    assert row.keys() == ["id", "model"]
    assert list(row) == [7, "gemma4:12b"]


def test_split_sql_script_ignores_comments_and_semicolons() -> None:
    script = """
    -- comment
    CREATE TABLE IF NOT EXISTS runs (id INTEGER);
    INSERT INTO runs (id) VALUES (1);
    """

    assert split_sql_script(script) == [
        "CREATE TABLE IF NOT EXISTS runs (id INTEGER)",
        "INSERT INTO runs (id) VALUES (1)",
    ]


def test_libsql_connection_from_env_uses_hosted_replica_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakeCursor:
        description = [("id",), ("model",)]
        lastrowid = 7

        def fetchone(self):
            return (7, "gemma4:12b")

        def fetchall(self):
            return [(7, "gemma4:12b")]

    class FakeConnection:
        def __init__(self) -> None:
            self.synced = 0
            self.committed = 0
            self.closed = False

        def sync(self) -> None:
            self.synced += 1

        def execute(self, sql, params=()):
            calls["execute"] = (sql, params)
            return FakeCursor()

        def executemany(self, sql, params_seq):
            calls["executemany"] = (sql, params_seq)
            return FakeCursor()

        def commit(self) -> None:
            self.committed += 1

        def close(self) -> None:
            self.closed = True

    fake_connection = FakeConnection()

    def fake_connect(path, **kwargs):
        calls["connect"] = (path, kwargs)
        return fake_connection

    monkeypatch.setitem(sys.modules, "libsql", SimpleNamespace(connect=fake_connect))
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://example.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TURSO_REPLICA_PATH", str(tmp_path / "replica.db"))

    connection = LibsqlConnection.from_env()
    cursor = connection.execute("SELECT id, model FROM runs WHERE id = ?", (7,))
    row = cursor.fetchone()
    connection.commit()

    assert calls["connect"] == (
        str(tmp_path / "replica.db"),
        {"sync_url": "libsql://example.turso.io", "auth_token": "token", "timeout": 5.0},
    )
    assert calls["execute"] == ("SELECT id, model FROM runs WHERE id = ?", (7,))
    assert row["model"] == "gemma4:12b"
    assert fake_connection.synced == 0
    assert fake_connection.committed == 1

    dbapi_cursor = connection.cursor()
    dbapi_cursor.execute("SELECT id, model FROM runs WHERE id = ?", (7,))
    rows = dbapi_cursor.fetchall()
    assert rows[0]["id"] == 7


def test_libsql_connection_can_use_direct_hosted_mode(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakeConnection:
        def sync(self) -> None:
            pass

        def close(self) -> None:
            pass

    def fake_connect(path, **kwargs):
        calls["connect"] = (path, kwargs)
        return FakeConnection()

    monkeypatch.setitem(sys.modules, "libsql", SimpleNamespace(connect=fake_connect))
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://example.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TURSO_REPLICA_PATH", str(tmp_path / "replica.db"))
    monkeypatch.setenv("TURSO_CONNECTION_MODE", "hosted")

    LibsqlConnection.from_env()

    assert calls["connect"] == (
        "libsql://example.turso.io",
        {"auth_token": "token", "timeout": 5.0},
    )


def test_libsql_connection_sync_can_be_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakeCursor:
        description = []
        lastrowid = None

        def fetchone(self):
            return None

        def fetchall(self):
            return []

    class FakeConnection:
        def __init__(self) -> None:
            self.synced = 0
            self.committed = 0

        def sync(self) -> None:
            self.synced += 1

        def execute(self, sql, params=()):
            return FakeCursor()

        def commit(self) -> None:
            self.committed += 1

        def close(self) -> None:
            pass

    fake_connection = FakeConnection()

    def fake_connect(path, **kwargs):
        calls["connect"] = (path, kwargs)
        return fake_connection

    monkeypatch.setitem(sys.modules, "libsql", SimpleNamespace(connect=fake_connect))
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://example.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TURSO_REPLICA_PATH", str(tmp_path / "replica.db"))
    monkeypatch.setenv("TURSO_CONNECTION_MODE", "replica")
    monkeypatch.setenv("TURSO_SYNC_ON_CONNECT", "1")
    monkeypatch.setenv("TURSO_SYNC_ON_COMMIT", "true")

    connection = LibsqlConnection.from_env()
    connection.commit()

    assert calls["connect"] == (
        str(tmp_path / "replica.db"),
        {"sync_url": "libsql://example.turso.io", "auth_token": "token", "timeout": 5.0},
    )
    assert fake_connection.synced == 2
    assert fake_connection.committed == 1


def test_libsql_cursor_exposes_rowcount() -> None:
    from sentiment_benchmark.libsql_backend import LibsqlCursor

    class RawWithRowcount:
        description = []
        lastrowid = None
        rowcount = 5

    class RawWithoutRowcount:
        description = []
        lastrowid = None

    assert LibsqlCursor(RawWithRowcount()).rowcount == 5
    assert LibsqlCursor(RawWithoutRowcount()).rowcount == -1
