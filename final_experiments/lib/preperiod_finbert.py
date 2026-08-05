"""Resumable, local-only FinBERT checkpoint for a frozen FNSPID pre-period.

The main FNSPID checkpoint is intentionally immutable and starts in 2011.
This module creates a separate gitignored SQLite checkpoint for an earlier
calendar window while reusing the established event identity and XNYS timing
rules.  Raw licensed headlines remain inside the local checkpoint and are
never returned by the public loading function.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_benchmark.baselines import (
    _disable_hf_progress_bars,
    _soft_sentiment_from_scores,
)
from sentiment_benchmark.fnspid_feasibility import (
    _hash64,
    _normal,
    _open_zstd_csv,
    _timestamp_parts,
)
from sentiment_benchmark.fnspid_moment2_finbert import _local_finbert_snapshot
from sentiment_benchmark.fnspid_vader_smoke import _SessionMapper

MODEL_ID = "ProsusAI/finbert"
EVENT_HASH_SEED = "fnspid-feasibility-v1"


class PreperiodCheckpointError(RuntimeError):
    """Raised when the local pre-period checkpoint cannot be trusted."""


@dataclass(frozen=True)
class PreperiodConfig:
    start_session: str = "2009-10-01"
    end_session: str = "2010-12-31"
    event_hash_seed: str = EVENT_HASH_SEED
    finbert_batch_size: int = 64
    finbert_checkpoint_size: int = 4096
    calendar_name: str = "XNYS"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _metadata_get(connection: sqlite3.Connection, key: str) -> Any:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else json.loads(str(row[0]))


def _metadata_set(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, json.dumps(value, sort_keys=True)),
    )


def _initialise_store(path: Path, config: PreperiodConfig) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT NOT NULL UNIQUE,
            symbol TEXT NOT NULL,
            session_date TEXT NOT NULL,
            headline TEXT NOT NULL,
            p_positive REAL,
            p_negative REAL,
            p_neutral REAL,
            finbert_score REAL
        )
        """
    )
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    stored = _metadata_get(connection, "config")
    expected = asdict(config)
    if stored is None:
        _metadata_set(connection, "config", expected)
        _metadata_set(connection, "created_at", _utc_now())
        connection.commit()
    elif stored != expected:
        connection.close()
        raise PreperiodCheckpointError("existing pre-period checkpoint has a different frozen configuration")
    return connection


def _validate_source_records(
    archives: tuple[tuple[Path, bool], ...],
    source_records: tuple[dict[str, Any], ...],
) -> None:
    expected = {Path(item["path"]).resolve(): item for item in source_records}
    actual = {path.resolve() for path, _ in archives}
    if actual != set(expected):
        raise PreperiodCheckpointError("news archives differ from the frozen source records")
    for path in sorted(actual):
        if not path.is_file():
            raise PreperiodCheckpointError(f"missing immutable news archive: {path}")
        recorded_size = int(expected[path]["size_bytes"])
        if path.stat().st_size != recorded_size:
            raise PreperiodCheckpointError(f"news archive size changed for {path}: {path.stat().st_size} != {recorded_size}")


def ensure_events_extracted(
    *,
    db_path: Path,
    archives: tuple[tuple[Path, bool], ...],
    source_records: tuple[dict[str, Any], ...],
    symbols: set[str],
    config: PreperiodConfig,
) -> dict[str, Any]:
    """Build the pre-period event store, safely rescanning after interruption.

    Re-running an interrupted extraction scans the immutable archives again and
    uses the unique event key to retain prior progress.  The completed audit is
    based on the fresh full scan, so partial-run counters cannot contaminate it.
    """

    _validate_source_records(archives, source_records)
    connection = _initialise_store(db_path, config)
    try:
        if bool(_metadata_get(connection, "event_extraction_complete")):
            audit = _metadata_get(connection, "event_extraction_audit")
            if not isinstance(audit, dict):
                raise PreperiodCheckpointError("completed checkpoint lacks extraction audit")
            stored_events = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            if stored_events != int(audit["deduplicated_nonempty_events"]):
                raise PreperiodCheckpointError("event count disagrees with extraction audit")
            return audit

        mapper = _SessionMapper(config.calendar_name)
        csv.field_size_limit(2**31 - 1)
        counts: Counter[str] = Counter()
        pending: list[tuple[str, str, str, str]] = []
        start = pd.Timestamp(config.start_session)
        end = pd.Timestamp(config.end_session)
        started = time.perf_counter()

        for archive, exclude_timed_rows in archives:
            with _open_zstd_csv(archive) as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, start=1):
                    counts["physical_rows_scanned"] += 1
                    if row_number % 1_000_000 == 0:
                        elapsed = time.perf_counter() - started
                        print(
                            f"Pre-period scan {archive.name}: {row_number:,} rows ({elapsed / 60:.1f} min)",
                            flush=True,
                        )
                    symbol = str(row.get("Stock_symbol") or "").strip().upper()
                    if symbol not in symbols:
                        continue
                    counts["rows_for_frozen_cohort"] += 1
                    raw_timestamp = str(row.get("Date") or "")
                    parts = _timestamp_parts(raw_timestamp)
                    if parts is None:
                        counts["invalid_timestamp_rows"] += 1
                        continue
                    if exclude_timed_rows and parts[3]:
                        counts["nasdaq_timed_rows_excluded"] += 1
                        continue
                    mapped = mapper.map(raw_timestamp)
                    if mapped is None:
                        counts["invalid_timestamp_rows"] += 1
                        continue
                    session_day, precise = mapped
                    counts["precise_timestamp_rows" if precise else "date_only_rows"] += 1
                    session = pd.Timestamp(session_day)
                    if not start <= session <= end:
                        counts["mapped_outside_window"] += 1
                        continue
                    counts["mapped_window_events_before_dedup"] += 1
                    headline = str(row.get("Article_title") or "").strip()
                    if not headline:
                        counts["empty_headlines_excluded"] += 1
                        continue
                    url = str(row.get("Url") or "").strip()
                    identity = _normal(url).rstrip("/") or _normal(headline)
                    event_key = _hash64(
                        symbol,
                        session_day,
                        identity,
                        seed=config.event_hash_seed,
                    )
                    pending.append((f"{event_key:016x}", symbol, session_day, headline))
                    if len(pending) >= 10_000:
                        connection.executemany(
                            """
                            INSERT OR IGNORE INTO events(event_key, symbol, session_date, headline)
                            VALUES (?, ?, ?, ?)
                            """,
                            pending,
                        )
                        connection.commit()
                        pending.clear()
        if pending:
            connection.executemany(
                """
                INSERT OR IGNORE INTO events(event_key, symbol, session_date, headline)
                VALUES (?, ?, ?, ?)
                """,
                pending,
            )
            connection.commit()

        unique_events = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        counts["deduplicated_nonempty_events"] = unique_events
        counts["exact_firm_event_duplicates"] = max(
            0,
            counts["mapped_window_events_before_dedup"] - counts["empty_headlines_excluded"] - unique_events,
        )
        connection.execute("CREATE INDEX IF NOT EXISTS events_symbol_session ON events(symbol, session_date)")
        audit: dict[str, Any] = {
            **{key: int(value) for key, value in counts.items()},
            "symbols_in_frozen_cohort": len(symbols),
            "symbols_with_events": int(connection.execute("SELECT COUNT(DISTINCT symbol) FROM events").fetchone()[0]),
            "firm_sessions": int(
                connection.execute("SELECT COUNT(*) FROM (SELECT 1 FROM events GROUP BY symbol, session_date)").fetchone()[0]
            ),
            "minimum_session": connection.execute("SELECT MIN(session_date) FROM events").fetchone()[0],
            "maximum_session": connection.execute("SELECT MAX(session_date) FROM events").fetchone()[0],
            "completed_at": _utc_now(),
            "runtime_seconds": time.perf_counter() - started,
        }
        _metadata_set(connection, "event_extraction_audit", audit)
        _metadata_set(connection, "source_records", list(source_records))
        _metadata_set(connection, "event_extraction_complete", True)
        connection.commit()
        return audit
    finally:
        connection.close()


def ensure_finbert_scored(*, db_path: Path, config: PreperiodConfig) -> dict[str, Any]:
    """Score every checkpoint headline locally, committing resumable chunks."""

    snapshot, revision = _local_finbert_snapshot()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from transformers import pipeline as hf_pipeline

    _disable_hf_progress_bars()
    if torch.cuda.is_available():
        device: int | str = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1

    connection = _initialise_store(db_path, config)
    try:
        if not bool(_metadata_get(connection, "event_extraction_complete")):
            raise PreperiodCheckpointError("event extraction must finish before scoring")
        expected = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        completed_before = int(connection.execute("SELECT COUNT(*) FROM events WHERE finbert_score IS NOT NULL").fetchone()[0])
        if completed_before == expected and bool(_metadata_get(connection, "finbert_scoring_complete")):
            audit = _metadata_get(connection, "finbert_scoring_audit")
            if not isinstance(audit, dict):
                raise PreperiodCheckpointError("completed checkpoint lacks scoring audit")
            return audit

        classifier = hf_pipeline(
            "text-classification",
            model=str(snapshot),
            truncation=True,
            device=device,
        )
        started = time.perf_counter()
        while True:
            pending = connection.execute(
                """
                SELECT event_id, headline
                FROM events
                WHERE finbert_score IS NULL
                ORDER BY event_id
                LIMIT ?
                """,
                (config.finbert_checkpoint_size,),
            ).fetchall()
            if not pending:
                break
            outputs: Any = classifier(
                [str(row[1]) for row in pending],
                top_k=None,
                batch_size=config.finbert_batch_size,
            )
            if len(outputs) != len(pending):
                raise PreperiodCheckpointError("FinBERT returned an unexpected row count")
            updates = []
            for (event_id, _), raw in zip(pending, outputs, strict=True):
                result = _soft_sentiment_from_scores(raw)
                updates.append(
                    (
                        result.p_positive,
                        result.p_negative,
                        result.p_neutral,
                        result.score,
                        int(event_id),
                    )
                )
            connection.executemany(
                """
                UPDATE events
                SET p_positive=?, p_negative=?, p_neutral=?, finbert_score=?
                WHERE event_id=?
                """,
                updates,
            )
            connection.commit()
            current = int(connection.execute("SELECT COUNT(*) FROM events WHERE finbert_score IS NOT NULL").fetchone()[0])
            elapsed = time.perf_counter() - started
            rate = (current - completed_before) / elapsed if elapsed > 0 else 0.0
            print(
                f"Pre-period FinBERT: {current:,}/{expected:,} ({current / expected:.1%}); {rate:.1f} headlines/s",
                flush=True,
            )

        probability_errors = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM events
                WHERE finbert_score IS NULL
                   OR ABS((p_positive + p_negative + p_neutral) - 1.0) > 1e-5
                   OR finbert_score < -1.0 OR finbert_score > 1.0
                """
            ).fetchone()[0]
        )
        if probability_errors:
            raise PreperiodCheckpointError(f"FinBERT probability integrity failures: {probability_errors}")
        audit = {
            "model_id": MODEL_ID,
            "model_revision": revision,
            "local_snapshot": str(snapshot),
            "local_only": True,
            "device": "cuda" if device == 0 else str(device),
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "batch_size": config.finbert_batch_size,
            "checkpoint_size": config.finbert_checkpoint_size,
            "events": expected,
            "resumed_from": completed_before,
            "runtime_seconds_this_invocation": time.perf_counter() - started,
            "completed_at": _utc_now(),
        }
        _metadata_set(connection, "finbert_scoring_audit", audit)
        _metadata_set(connection, "finbert_scoring_complete", True)
        connection.commit()
        return audit
    finally:
        connection.close()


def load_aggregate_signal(db_path: Path) -> pd.DataFrame:
    """Return only aggregate, licence-safe firm-session FinBERT values."""

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        if not bool(_metadata_get(connection, "finbert_scoring_complete")):
            raise PreperiodCheckpointError("FinBERT checkpoint is incomplete")
        panel = pd.read_sql_query(
            """
            SELECT symbol, session_date,
                   AVG(finbert_score) AS mean_continuous,
                   COUNT(*) AS article_count
            FROM events
            GROUP BY symbol, session_date
            ORDER BY session_date, symbol
            """,
            connection,
        )
    finally:
        connection.close()
    panel["session_date"] = pd.to_datetime(panel["session_date"], errors="raise").dt.normalize()
    if panel.duplicated(["symbol", "session_date"]).any():
        raise PreperiodCheckpointError("aggregate signal is not unique by firm-session")
    if panel["mean_continuous"].isna().any():
        raise PreperiodCheckpointError("aggregate signal contains missing values")
    return panel


__all__ = [
    "EVENT_HASH_SEED",
    "MODEL_ID",
    "PreperiodCheckpointError",
    "PreperiodConfig",
    "ensure_events_extracted",
    "ensure_finbert_scored",
    "load_aggregate_signal",
]
