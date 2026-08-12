"""Immutable headline snapshots for incremental backward-arm FinBERT scoring.

This module deliberately reads only terminal LSEG company-date checkpoints.
It never queries LSEG and it never loads prices or returns.  Each snapshot is a
standalone, compact LSEG-style collection that can be passed to the existing
resumable ``score_collection_baselines`` implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentiment_benchmark.artifact_io import atomic_write_json, atomic_write_jsonl, sha256_file, sha256_text
from sentiment_benchmark.headline_value import collect_scorable_headline_records
from sentiment_benchmark.lseg_source import (
    _normalize_headline,
    collection_windows,
    load_lseg_collection_config,
)


class IncrementalFinbertSnapshotError(ValueError):
    """Raised when saved LSEG checkpoints cannot form a valid snapshot."""


@dataclass(frozen=True)
class IncrementalFinbertSnapshotSummary:
    completed_company_dates: int
    source_pages: int
    source_rows: int
    unique_story_ids: int
    unique_scorable_headlines: int
    population_sha256: str
    output_dir: Path
    headlines_path: Path
    manifest_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncrementalFinbertSnapshotError(message)


def _terminal(payload: dict[str, Any]) -> bool:
    return not str(payload.get("cursor_out") or "").strip() or bool(
        str(payload.get("pagination_terminal_reason") or "").strip()
    )


def _merge_compact_headline(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key in ("matched_symbols", "matched_queries", "matched_rics", "subjects", "entities"):
        target[key] = sorted(set(target.get(key, ())) | set(incoming.get(key, ())))
    if len(str(incoming.get("headline") or "")) > len(str(target.get("headline") or "")):
        target["headline"] = incoming["headline"]
    incoming_first = str(incoming.get("first_created") or "")
    target_first = str(target.get("first_created") or "")
    if incoming_first and (not target_first or incoming_first < target_first):
        target["first_created"] = incoming_first
    incoming_version = str(incoming.get("version_created") or "")
    target_version = str(target.get("version_created") or "")
    if incoming_version and (not target_version or incoming_version > target_version):
        target["version_created"] = incoming_version
    for key in ("source_code", "language"):
        if not target.get(key) and incoming.get(key):
            target[key] = incoming[key]


def materialize_incremental_finbert_snapshot(
    repo_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> IncrementalFinbertSnapshotSummary:
    """Build one immutable snapshot from terminal company-date page chains."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_lseg_collection_config(config_file)
    raw_dir = config.raw_dir if config.raw_dir.is_absolute() else root / config.raw_dir
    pages_dir = raw_dir / "headline_pages"
    source_manifest_path = raw_dir / "manifest.json"
    target = Path(output_dir)
    if not target.is_absolute():
        target = root / target
    _require(pages_dir.is_dir(), f"missing LSEG checkpoint directory: {pages_dir}")
    _require(source_manifest_path.is_file(), f"missing LSEG source manifest: {source_manifest_path}")
    _require(not target.exists() or not any(target.iterdir()), f"snapshot output is not empty: {target}")

    companies = {company.symbol: company for company in config.companies}
    windows = {window.index: window for window in collection_windows(config)}
    expected_company_dates = len(companies) * len(windows)
    # Retain pagination metadata only.  The raw collection can be many GB, so
    # holding every page's bytes and parsed rows until validation completes is
    # not memory safe.  Page contents are read again in the materialisation pass.
    groups: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = {}
    for page_path in sorted(pages_dir.glob("*.json")):
        raw = page_path.read_bytes()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IncrementalFinbertSnapshotError(f"invalid checkpoint JSON: {page_path}") from exc
        _require(isinstance(payload, dict), f"checkpoint is not an object: {page_path}")
        symbol = str(payload.get("symbol") or "").strip().upper()
        try:
            window_index = int(payload.get("window_index"))
            page_number = int(payload.get("page_number"))
        except (TypeError, ValueError) as exc:
            raise IncrementalFinbertSnapshotError(f"checkpoint identity is invalid: {page_path}") from exc
        _require(symbol in companies, f"checkpoint has unexpected symbol {symbol!r}: {page_path}")
        _require(window_index in windows, f"checkpoint has unexpected window {window_index}: {page_path}")
        _require(page_number >= 1, f"checkpoint has invalid page number: {page_path}")
        metadata = {
            "symbol": symbol,
            "window_index": window_index,
            "page_number": page_number,
            "query": payload.get("query"),
            "ric": payload.get("ric"),
            "window_start": payload.get("window_start"),
            "window_end": payload.get("window_end"),
            "cursor_in": payload.get("cursor_in"),
            "cursor_out": payload.get("cursor_out"),
            "pagination_terminal_reason": payload.get("pagination_terminal_reason"),
        }
        groups.setdefault((symbol, window_index), []).append((page_path, metadata))

    completed: list[tuple[str, int]] = []
    source_pages: list[Path] = []
    for key, pages in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        terminal_pages = [item for item in pages if _terminal(item[1])]
        if not terminal_pages:
            continue
        _require(len(terminal_pages) == 1, f"multiple terminal checkpoints for {key}")
        terminal_number = int(terminal_pages[0][1]["page_number"])
        selected = sorted(
            (item for item in pages if int(item[1]["page_number"]) <= terminal_number),
            key=lambda item: int(item[1]["page_number"]),
        )
        numbers = [int(item[1]["page_number"]) for item in selected]
        _require(numbers == list(range(1, terminal_number + 1)), f"non-contiguous checkpoint chain for {key}")
        _require(len(selected) == len(pages), f"checkpoint pages exist after the terminal page for {key}")

        symbol, window_index = key
        company = companies[symbol]
        window = windows[window_index]
        expected_cursor: str | None = None
        for position, (page_path, payload) in enumerate(selected, start=1):
            _require(payload.get("query") == company.news_query, f"query drift in {page_path}")
            _require(str(payload.get("ric") or "") == company.ric, f"RIC drift in {page_path}")
            _require(payload.get("window_start") == window.start, f"window start drift in {page_path}")
            _require(payload.get("window_end") == window.end, f"window end drift in {page_path}")
            _require(payload.get("cursor_in") == expected_cursor, f"cursor chain mismatch in {page_path}")
            if position < terminal_number:
                _require(not _terminal(payload), f"early terminal checkpoint in {page_path}")
                expected_cursor = str(payload.get("cursor_out") or "").strip()
                _require(bool(expected_cursor), f"missing cursor in non-terminal checkpoint: {page_path}")
            else:
                _require(_terminal(payload), f"last checkpoint is not terminal: {page_path}")
        completed.append(key)
        source_pages.extend(page_path for page_path, _payload in selected)

    _require(bool(completed), "no terminal company-date checkpoints are available")
    checkpoint_hash = hashlib.sha256()
    headlines: dict[str, dict[str, Any]] = {}
    source_rows = 0
    for page_path in source_pages:
        raw = page_path.read_bytes()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IncrementalFinbertSnapshotError(f"invalid checkpoint JSON: {page_path}") from exc
        relative = page_path.relative_to(pages_dir).as_posix().encode("utf-8")
        checkpoint_hash.update(len(relative).to_bytes(8, "big"))
        checkpoint_hash.update(relative)
        checkpoint_hash.update(len(raw).to_bytes(8, "big"))
        checkpoint_hash.update(raw)
        company = companies[str(payload["symbol"]).strip().upper()]
        rows = payload.get("rows")
        _require(isinstance(rows, list), f"checkpoint rows are not a list: {page_path}")
        source_rows += len(rows)
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = _normalize_headline(row, company)
            if normalized is None:
                continue
            normalized.pop("raw_rows", None)
            story_id = str(normalized["story_id"])
            if story_id in headlines:
                _merge_compact_headline(headlines[story_id], normalized)
            else:
                headlines[story_id] = normalized

    target.mkdir(parents=True, exist_ok=True)
    headlines_path = atomic_write_jsonl(target / "headlines.jsonl", [headlines[key] for key in sorted(headlines)])
    manifest_path = target / "manifest.json"
    base_manifest = {
        "schema_version": 1,
        "status": "completed",
        "config_sha256": config.config_sha256,
        "config": config.to_payload(),
        "snapshot": {
            "kind": "terminal_company_date_incremental_finbert_input",
            "created_at": datetime.now(UTC).isoformat(),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "completed_company_dates": len(completed),
            "completed_company_dates_sha256": sha256_text(
                "\n".join(f"{symbol}|{window_index}" for symbol, window_index in completed)
            ),
            "source_pages": len(source_pages),
            "source_checkpoint_tree_sha256": checkpoint_hash.hexdigest(),
            "partial_collection": len(completed) < expected_company_dates,
            "prices_or_returns_loaded": False,
        },
        "counts": {
            "headline_pages": len(source_pages),
            "raw_headline_rows": source_rows,
            "headlines": len(headlines),
            "fetch_story_bodies": False,
        },
        "files": {"headlines_jsonl": {"path": headlines_path.name, "sha256": sha256_file(headlines_path)}},
        "sharing": {"licensed_headlines": True, "redistribute": False, "source_control": False},
    }
    atomic_write_json(manifest_path, base_manifest)
    scorable = collect_scorable_headline_records(target)
    hashes = [record.headline_sha256 for record in scorable]
    population_sha256 = sha256_text("\n".join(sorted(hashes)))
    base_manifest["snapshot"]["unique_scorable_headlines"] = len(hashes)
    base_manifest["snapshot"]["population_sha256"] = population_sha256
    atomic_write_json(manifest_path, base_manifest)
    return IncrementalFinbertSnapshotSummary(
        completed_company_dates=len(completed),
        source_pages=len(source_pages),
        source_rows=source_rows,
        unique_story_ids=len(headlines),
        unique_scorable_headlines=len(hashes),
        population_sha256=population_sha256,
        output_dir=target,
        headlines_path=headlines_path,
        manifest_path=manifest_path,
    )


__all__ = [
    "IncrementalFinbertSnapshotError",
    "IncrementalFinbertSnapshotSummary",
    "materialize_incremental_finbert_snapshot",
]
