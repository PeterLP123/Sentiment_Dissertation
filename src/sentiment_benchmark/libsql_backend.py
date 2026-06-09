from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .env import load_env_file


class LibsqlConfigurationError(RuntimeError):
    """Raised when libSQL/Turso settings are missing or invalid."""


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise LibsqlConfigurationError(f"{name} must be one of: 1, 0, true, false, yes, no, on, off")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LibsqlConfigurationError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise LibsqlConfigurationError(f"{name} must be greater than 0")
    return parsed


class ResultRow:
    """Small row wrapper with sqlite3.Row-like name and index access."""

    def __init__(self, columns: Sequence[str], values: Sequence[Any]) -> None:
        self._columns = list(columns)
        self._values = list(values)
        self._by_name = dict(zip(self._columns, self._values, strict=False))

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._by_name[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def keys(self) -> list[str]:
        return list(self._columns)


def split_sql_script(script: str) -> list[str]:
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buffer += line + "\n"
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            buffer = ""
    tail = buffer.strip().rstrip(";").strip()
    if tail:
        statements.append(tail)
    return statements


def _columns_from_description(description: Any) -> list[str]:
    columns: list[str] = []
    for item in description or []:
        if isinstance(item, str):
            columns.append(item)
        elif isinstance(item, Sequence) and item:
            columns.append(str(item[0]))
        else:
            columns.append(str(item))
    return columns


def _wrap_row(columns: Sequence[str], row: Any) -> Any:
    if row is None:
        return None
    if hasattr(row, "keys"):
        return row
    if isinstance(row, Mapping):
        return ResultRow(list(row), list(row.values()))
    return ResultRow(columns, list(row))


class LibsqlCursor:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    @property
    def description(self) -> Any:
        return getattr(self._cursor, "description", None)

    @property
    def lastrowid(self) -> int | None:
        value = getattr(self._cursor, "lastrowid", None)
        return int(value) if value is not None else None

    def fetchone(self) -> Any:
        columns = _columns_from_description(self.description)
        return _wrap_row(columns, self._cursor.fetchone())

    def fetchall(self) -> list[Any]:
        columns = _columns_from_description(self.description)
        return [_wrap_row(columns, row) for row in self._cursor.fetchall()]

    def close(self) -> None:
        closer = getattr(self._cursor, "close", None)
        if callable(closer):
            closer()


class LibsqlConnectionCursor:
    def __init__(self, connection: LibsqlConnection) -> None:
        self._connection = connection
        self._cursor: LibsqlCursor | None = None

    @property
    def description(self) -> Any:
        return self._cursor.description if self._cursor is not None else None

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid if self._cursor is not None else None

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> LibsqlConnectionCursor:
        self._cursor = self._connection.execute(sql, params)
        return self

    def fetchone(self) -> Any:
        return self._cursor.fetchone() if self._cursor is not None else None

    def fetchall(self) -> list[Any]:
        return self._cursor.fetchall() if self._cursor is not None else []

    def close(self) -> None:
        if self._cursor is not None:
            self._cursor.close()


class LibsqlConnection:
    def __init__(self, connection: Any, *, sync_on_commit: bool = True) -> None:
        self._connection = connection
        self.sync_on_commit = sync_on_commit

    @classmethod
    def from_env(cls) -> LibsqlConnection:
        load_env_file()
        database_url = os.getenv("TURSO_DATABASE_URL", "").strip()
        auth_token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
        if not database_url:
            raise LibsqlConfigurationError("TURSO_DATABASE_URL is required when SENTIMENT_BENCH_DB_BACKEND=libsql")
        if not auth_token:
            raise LibsqlConfigurationError("TURSO_AUTH_TOKEN is required when SENTIMENT_BENCH_DB_BACKEND=libsql")
        timeout = _env_float("TURSO_TIMEOUT_SECONDS", 5.0)
        mode = os.getenv("TURSO_CONNECTION_MODE", "replica").strip().lower()
        if mode not in {"hosted", "replica"}:
            raise LibsqlConfigurationError("TURSO_CONNECTION_MODE must be 'hosted' or 'replica'")

        try:
            import libsql
        except ImportError as exc:
            raise LibsqlConfigurationError(
                "The libsql package is required for SENTIMENT_BENCH_DB_BACKEND=libsql. "
                "Use Python 3.12 and install dependencies with: python -m pip install -e \".[dev]\""
            ) from exc

        if mode == "hosted":
            connection = libsql.connect(database_url, auth_token=auth_token, timeout=timeout)
            return cls(connection, sync_on_commit=False)

        replica_path = Path(os.getenv("TURSO_REPLICA_PATH", "results/turso_replica.db"))
        replica_path.parent.mkdir(parents=True, exist_ok=True)
        sync_interval_raw = os.getenv("TURSO_SYNC_INTERVAL_SECONDS", "").strip()
        sync_on_connect = _env_flag("TURSO_SYNC_ON_CONNECT", False)
        sync_on_commit = _env_flag("TURSO_SYNC_ON_COMMIT", False)
        kwargs: dict[str, Any] = {
            "sync_url": database_url,
            "auth_token": auth_token,
            "timeout": timeout,
        }
        if sync_interval_raw:
            kwargs["sync_interval"] = int(sync_interval_raw)
        connection = libsql.connect(str(replica_path), **kwargs)
        sync = getattr(connection, "sync", None)
        if sync_on_connect and callable(sync):
            sync()
        return cls(connection, sync_on_commit=sync_on_commit)

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> LibsqlCursor:
        cursor = self._connection.execute(sql, tuple(params or ()))
        return LibsqlCursor(cursor)

    def executemany(self, sql: str, params_seq: Iterable[Sequence[Any]]) -> LibsqlCursor:
        cursor = self._connection.executemany(sql, [tuple(params) for params in params_seq])
        return LibsqlCursor(cursor)

    def executescript(self, script: str) -> LibsqlCursor:
        cursor: LibsqlCursor | None = None
        for statement in split_sql_script(script):
            cursor = self.execute(statement)
        if cursor is None:
            cursor = self.execute("SELECT 1 WHERE 0")
        return cursor

    def cursor(self) -> Any:
        return LibsqlConnectionCursor(self)

    def commit(self) -> None:
        self._connection.commit()
        if self.sync_on_commit:
            self.sync()

    def sync(self) -> bool:
        sync = getattr(self._connection, "sync", None)
        if callable(sync):
            sync()
            return True
        return False

    def close(self) -> None:
        self._connection.close()
