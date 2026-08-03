#!/usr/bin/env python3
"""Merge completed LSEG headline collections without modifying their source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

MERGER_VERSION = "lseg_headline_merge_v1"
LIST_FIELDS = ("matched_symbols", "matched_queries", "matched_rics", "subjects", "entities")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object on line {line_number} of {path}")
            yield value


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value if str(item)})


def _unique_raw_rows(rows: Sequence[Any]) -> list[Any]:
    unique: dict[str, Any] = {}
    for row in rows:
        unique.setdefault(canonical_json(row), row)
    return [unique[key] for key in sorted(unique)]


def prepare_row(row: dict[str, Any], collection_id: str) -> dict[str, Any]:
    story_id = str(row.get("story_id") or "").strip()
    if not story_id:
        raise ValueError(f"headline row is missing story_id in {collection_id}")
    prepared = dict(row)
    prepared["story_id"] = story_id
    for field in LIST_FIELDS:
        prepared[field] = _strings(prepared.get(field))
    raw_rows = prepared.get("raw_rows")
    prepared["raw_rows"] = list(raw_rows) if isinstance(raw_rows, list) else []
    prepared["source_collection_ids"] = [collection_id]
    return prepared


def merge_rows(target: dict[str, Any], incoming: dict[str, Any], conflicts: Counter[str]) -> dict[str, Any]:
    if target.get("story_id") != incoming.get("story_id"):
        raise ValueError("cannot merge different story IDs")
    for field in LIST_FIELDS:
        target[field] = sorted(set(_strings(target.get(field))) | set(_strings(incoming.get(field))))
    target["source_collection_ids"] = sorted(
        set(_strings(target.get("source_collection_ids"))) | set(_strings(incoming.get("source_collection_ids")))
    )
    target["raw_rows"] = _unique_raw_rows(
        [*(target.get("raw_rows") if isinstance(target.get("raw_rows"), list) else []),
         *(incoming.get("raw_rows") if isinstance(incoming.get("raw_rows"), list) else [])]
    )
    if len(str(incoming.get("headline") or "")) > len(str(target.get("headline") or "")):
        target["headline"] = incoming.get("headline") or ""
    incoming_first = str(incoming.get("first_created") or "")
    target_first = str(target.get("first_created") or "")
    if incoming_first and (not target_first or incoming_first < target_first):
        target["first_created"] = incoming_first
    incoming_version = str(incoming.get("version_created") or "")
    target_version = str(target.get("version_created") or "")
    if incoming_version and (not target_version or incoming_version > target_version):
        target["version_created"] = incoming_version
    for field in ("source_code", "language"):
        current = str(target.get(field) or "")
        candidate = str(incoming.get(field) or "")
        if current and candidate and current != candidate:
            conflicts[field] += 1
        elif not current and candidate:
            target[field] = incoming[field]
    return target


def _source_metadata(source: Path) -> dict[str, Any]:
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"source manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"source collection is not completed: {source}")
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    collection = config.get("collection") if isinstance(config.get("collection"), dict) else {}
    collection_id = str(collection.get("id") or "").strip()
    if not collection_id:
        raise ValueError(f"source manifest has no collection ID: {manifest_path}")
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    entry = files.get("headlines_jsonl") if isinstance(files.get("headlines_jsonl"), dict) else {}
    if not entry.get("path") or not entry.get("sha256"):
        raise ValueError(f"source manifest has no verified headlines file: {manifest_path}")
    headlines_path = source / str(entry["path"])
    if not headlines_path.exists():
        raise ValueError(f"source headlines file is missing: {headlines_path}")
    actual_sha256 = sha256_file(headlines_path)
    if actual_sha256 != entry["sha256"]:
        raise ValueError(f"source headlines hash mismatch: {headlines_path}")
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    expected_rows = int(counts.get("headlines", -1))
    if expected_rows < 0:
        raise ValueError(f"source manifest has no headline count: {manifest_path}")
    return {
        "collection_id": collection_id,
        "collection_start": collection.get("start"),
        "collection_end": collection.get("end"),
        "source_dir": str(source),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "headlines_path": str(headlines_path),
        "headlines_sha256": actual_sha256,
        "expected_rows": expected_rows,
        "config": config,
    }


def _merged_config(sources: Sequence[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Build the minimal frozen config consumed by downstream headline scorers."""
    companies: dict[str, dict[str, Any]] = {}
    starts: list[str] = []
    ends: list[str] = []
    languages: set[str] = set()
    for source in sources:
        config = source["config"]
        collection = config.get("collection") or {}
        starts.append(str(collection["start"]))
        ends.append(str(collection["end"]))
        if collection.get("language"):
            languages.add(str(collection["language"]))
        for item in config.get("companies") or []:
            symbol = str(item.get("symbol") or "").strip()
            if not symbol:
                continue
            current = companies.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "name": str(item.get("name") or symbol),
                    "ric": str(item.get("ric") or ""),
                    "news_query": str(item.get("news_query") or ""),
                    "aliases": [],
                },
            )
            for field in ("name", "ric", "news_query"):
                candidate = str(item.get(field) or "")
                if current[field] and candidate and current[field] != candidate:
                    raise ValueError(f"conflicting company {field} for {symbol}")
                if not current[field] and candidate:
                    current[field] = candidate
            current["aliases"] = sorted(
                set(current["aliases"])
                | {str(alias) for alias in item.get("aliases") or [] if str(alias).strip()}
            )
    if not companies:
        raise ValueError("source manifests contain no company definitions")
    if len(languages) > 1:
        raise ValueError(f"source collections use different languages: {sorted(languages)}")
    if output_dir.parent.name == "derived":
        collection_id = output_dir.parent.parent.name
    else:
        collection_id = "merged__" + "__".join(source["collection_id"] for source in sources)
    return {
        "collection": {
            "id": collection_id,
            "start": min(starts),
            "end": max(ends),
            "language": next(iter(languages), "en"),
            "fetch_story_bodies": False,
            "source_collection_ids": [source["collection_id"] for source in sources],
        },
        "companies": [companies[symbol] for symbol in sorted(companies)],
    }


def _source_signature(sources: Sequence[dict[str, Any]]) -> str:
    payload = {
        "merger_version": MERGER_VERSION,
        "sources": [
            {
                "collection_id": source["collection_id"],
                "manifest_sha256": source["manifest_sha256"],
                "headlines_sha256": source["headlines_sha256"],
            }
            for source in sources
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _current_manifest(
    output_dir: Path,
    source_signature: str,
    merged_config: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("merger_version") != MERGER_VERSION
        or manifest.get("source_signature") != source_signature
    ):
        return None
    entry = manifest.get("files", {}).get("headlines_jsonl", {})
    output_path = output_dir / str(entry.get("path") or "")
    if not output_path.exists() or sha256_file(output_path) != entry.get("sha256"):
        return None
    if manifest.get("config") != merged_config:
        manifest["config"] = merged_config
        manifest_temp = output_dir / ".manifest.json.tmp"
        manifest_temp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(manifest_temp, output_dir / "manifest.json")
    return manifest


def _temporary_text(path: Path) -> tuple[TextIO, Path]:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    return handle, Path(handle.name)


def build(sources: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if len(sources) < 2:
        raise ValueError("at least two source collections are required")
    source_metadata = [_source_metadata(source) for source in sources]
    collection_ids = [source["collection_id"] for source in source_metadata]
    if len(collection_ids) != len(set(collection_ids)):
        raise ValueError("source collection IDs must be unique")
    signature = _source_signature(source_metadata)
    merged_config = _merged_config(source_metadata, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    current = _current_manifest(output_dir, signature, merged_config)
    if current is not None:
        return current
    if any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-current output directory: {output_dir}")

    database_handle, database_name = tempfile.mkstemp(prefix=".headline-merge.", suffix=".sqlite3", dir=output_dir)
    os.close(database_handle)
    database_path = Path(database_name)
    output_path = output_dir / "headlines.jsonl"
    output_handle: TextIO | None = None
    output_temp: Path | None = None
    source_rows: dict[str, int] = {}
    duplicate_story_ids = 0
    scalar_conflicts: Counter[str] = Counter()
    all_symbols: set[str] = set()
    earliest_version = ""
    latest_version = ""

    try:
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE headlines (story_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for source in source_metadata:
            collection_id = str(source["collection_id"])
            row_count = 0
            for row in read_jsonl(Path(source["headlines_path"])):
                prepared = prepare_row(row, collection_id)
                story_id = str(prepared["story_id"])
                version_created = str(prepared.get("version_created") or "")
                if version_created:
                    earliest_version = min(earliest_version, version_created) if earliest_version else version_created
                    latest_version = max(latest_version, version_created)
                all_symbols.update(_strings(prepared.get("matched_symbols")))
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO headlines (story_id, payload) VALUES (?, ?)",
                    (story_id, canonical_json(prepared)),
                )
                if cursor.rowcount == 0:
                    duplicate_story_ids += 1
                    existing_value = connection.execute(
                        "SELECT payload FROM headlines WHERE story_id = ?", (story_id,)
                    ).fetchone()
                    if existing_value is None:
                        raise RuntimeError(f"missing SQLite merge row for {story_id}")
                    existing = json.loads(existing_value[0])
                    merged = merge_rows(existing, prepared, scalar_conflicts)
                    connection.execute(
                        "UPDATE headlines SET payload = ? WHERE story_id = ?",
                        (canonical_json(merged), story_id),
                    )
                row_count += 1
                if row_count % 10_000 == 0:
                    connection.commit()
            connection.commit()
            if row_count != source["expected_rows"]:
                raise ValueError(
                    f"source row count mismatch for {collection_id}: expected {source['expected_rows']}, found {row_count}"
                )
            source_rows[collection_id] = row_count

        output_rows = int(connection.execute("SELECT COUNT(*) FROM headlines").fetchone()[0])
        output_handle, output_temp = _temporary_text(output_path)
        for (payload,) in connection.execute("SELECT payload FROM headlines ORDER BY story_id"):
            output_handle.write(payload)
            output_handle.write("\n")
        output_handle.flush()
        os.fsync(output_handle.fileno())
        output_handle.close()
        output_handle = None
        connection.close()
        os.replace(output_temp, output_path)
        output_temp = None

        manifest = {
            "schema_version": 1,
            "status": "completed",
            "built_at": datetime.now(UTC).isoformat(),
            "merger_version": MERGER_VERSION,
            "source_signature": signature,
            "sources": [
                {key: value for key, value in source.items() if key != "config"}
                for source in source_metadata
            ],
            "config": merged_config,
            "deduplication": {
                "key": "story_id",
                "list_fields": "sorted set union",
                "headline": "longest non-empty value",
                "first_created": "earliest non-empty value",
                "version_created": "latest non-empty value",
                "source_code_and_language": (
                    "first non-empty value in declared source order; conflicting non-empty values are counted"
                ),
                "raw_rows": "exact canonical-JSON deduplication when a story occurs in multiple collections",
                "source_collection_ids": "sorted set union",
            },
            "counts": {
                "source_collections": len(source_metadata),
                "source_rows": source_rows,
                "input_rows": sum(source_rows.values()),
                "duplicate_story_ids_across_collections": duplicate_story_ids,
                "output_rows": output_rows,
                "symbols": len(all_symbols),
                "scalar_conflicts": dict(sorted(scalar_conflicts.items())),
            },
            "coverage": {
                "earliest_version_created": earliest_version,
                "latest_version_created": latest_version,
                "matched_symbols": sorted(all_symbols),
            },
            "label_mapping": None,
            "split": None,
            "random_seed": None,
            "files": {
                "headlines_jsonl": {
                    "path": output_path.name,
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                }
            },
            "sharing": {
                "licensed_lseg_headlines": True,
                "redistribute": False,
                "note": "This derived corpus contains licensed LSEG headlines and must remain local.",
            },
        }
        manifest_temp = output_dir / ".manifest.json.tmp"
        manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        os.replace(manifest_temp, output_dir / "manifest.json")
        return manifest
    except BaseException:
        if output_handle is not None:
            output_handle.close()
        if output_temp is not None:
            output_temp.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        (output_dir / ".manifest.json.tmp").unlink(missing_ok=True)
        raise
    finally:
        database_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path, help="Completed raw LSEG directory; repeat")
    parser.add_argument("--output-dir", required=True, type=Path, help="Empty directory for merged output")
    args = parser.parse_args()
    manifest = build(args.source, args.output_dir)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
