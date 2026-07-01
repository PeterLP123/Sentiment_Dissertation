#!/usr/bin/env python3
"""Build analysis-ready headline and main-body CSVs from a completed LSEG collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from sentiment_benchmark.news_cleaning import CLEANER_VERSION, QUALITY_OK, clean_news_html

EXPORTER_VERSION = "lseg_clean_csv_v1"

HEADLINE_FIELDS = [
    "revision_id",
    "story_id",
    "story_family_id",
    "version_created",
    "source_code",
    "language",
    "matched_symbols",
    "matched_rics",
    "headline_sha256",
    "headline",
]

BODY_FIELDS = [
    "revision_id",
    "story_id",
    "story_family_id",
    "version_created",
    "source_code",
    "language",
    "matched_symbols",
    "matched_rics",
    "headline",
    "body_format",
    "text_quality",
    "scoring_eligible",
    "original_chars",
    "cleaned_chars",
    "removed_trailing_lines",
    "raw_body_sha256",
    "main_body_sha256",
    "main_body",
]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalized_timestamp(value: Any) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def story_family_id(story_id: str) -> str:
    return re.sub(r":\d+$", "", story_id)


def pipe_values(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "|".join(sorted({normalized_text(item) for item in value if normalized_text(item)}))


def revision_id(story_id: str, version_created: str) -> str:
    return sha256_text(f"lseg|{story_id}|{version_created}")[:16]


def resolve_manifest_file(source: Path, manifest: dict[str, Any], key: str) -> Path:
    files = manifest.get("files")
    entry = files.get(key) if isinstance(files, dict) else None
    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
        raise ValueError(f"raw manifest is missing files.{key}")
    path = source / str(entry["path"])
    if not path.exists():
        raise ValueError(f"raw collection file is missing: {path}")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ValueError(f"raw collection hash mismatch for {path}")
    return path


def temporary_csv(path: Path) -> tuple[TextIO, Path]:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    return handle, Path(handle.name)


def output_is_current(output_dir: Path, source_manifest_sha256: str) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("source_manifest_sha256") != source_manifest_sha256
        or manifest.get("cleaner_version") != CLEANER_VERSION
        or manifest.get("exporter_version") != EXPORTER_VERSION
    ):
        return False
    files = manifest.get("files")
    if not isinstance(files, dict):
        return False
    for entry in files.values():
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            return False
        path = output_dir / str(entry["path"])
        if not path.exists() or sha256_file(path) != entry["sha256"]:
            return False
    return True


def build(source: Path, output_dir: Path) -> dict[str, Any]:
    raw_manifest_path = source / "manifest.json"
    if not raw_manifest_path.exists():
        raise ValueError(f"raw collection manifest is missing: {raw_manifest_path}")
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("status") != "completed":
        raise ValueError(f"raw collection is not completed: {source}")

    source_manifest_sha256 = sha256_file(raw_manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_is_current(output_dir, source_manifest_sha256):
        return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if any(output_dir.iterdir()):
        existing_manifest_path = output_dir / "manifest.json"
        existing_names = {path.name for path in output_dir.iterdir()}
        allowed_names = {"headlines.csv", "main_bodies.csv", "manifest.json"}
        existing_manifest = (
            json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            if existing_manifest_path.exists()
            else {}
        )
        if (
            existing_manifest.get("source_manifest_sha256") != source_manifest_sha256
            or not existing_names <= allowed_names
        ):
            raise ValueError(f"refusing to overwrite non-current output directory: {output_dir}")

    headlines_path = resolve_manifest_file(source, raw_manifest, "headlines_jsonl")
    stories_path = resolve_manifest_file(source, raw_manifest, "stories_jsonl")
    config = raw_manifest.get("config") if isinstance(raw_manifest.get("config"), dict) else {}
    collection = config.get("collection") if isinstance(config.get("collection"), dict) else {}
    cleaning = config.get("cleaning") if isinstance(config.get("cleaning"), dict) else {}
    configured_language = normalized_text(collection.get("language")) or "en"
    min_text_chars = int(cleaning.get("min_text_chars", 100))

    story_ids = {normalized_text(row.get("story_id")) for row in read_jsonl(stories_path)}
    story_ids.discard("")
    story_metadata: dict[str, dict[str, str]] = {}
    headline_counts: Counter[str] = Counter({"invalid_timestamp": 0})

    headlines_output = output_dir / "headlines.csv"
    headline_handle, headline_temp = temporary_csv(headlines_output)
    try:
        writer = csv.DictWriter(headline_handle, fieldnames=HEADLINE_FIELDS, lineterminator="\n")
        writer.writeheader()
        seen_story_ids: set[str] = set()
        for row in read_jsonl(headlines_path):
            story_id = normalized_text(row.get("story_id"))
            headline = normalized_text(row.get("headline"))
            if not story_id:
                headline_counts["excluded_missing_story_id"] += 1
                continue
            if not headline:
                headline_counts["excluded_empty_headline"] += 1
                continue
            if story_id in seen_story_ids:
                raise ValueError(f"duplicate story_id in consolidated headlines: {story_id}")
            seen_story_ids.add(story_id)
            version_created = normalized_timestamp(row.get("version_created"))
            if not version_created:
                headline_counts["invalid_timestamp"] += 1
            output_row = {
                "revision_id": revision_id(story_id, version_created),
                "story_id": story_id,
                "story_family_id": story_family_id(story_id),
                "version_created": version_created,
                "source_code": normalized_text(row.get("source_code")),
                "language": normalized_text(row.get("language")) or configured_language,
                "matched_symbols": pipe_values(row.get("matched_symbols")),
                "matched_rics": pipe_values(row.get("matched_rics")),
                "headline_sha256": sha256_text(headline),
                "headline": headline,
            }
            writer.writerow(output_row)
            headline_counts["written"] += 1
            if story_id in story_ids:
                story_metadata[story_id] = output_row
        headline_handle.flush()
        os.fsync(headline_handle.fileno())
        headline_handle.close()
    except BaseException:
        headline_handle.close()
        headline_temp.unlink(missing_ok=True)
        raise

    body_output = output_dir / "main_bodies.csv"
    body_handle, body_temp = temporary_csv(body_output)
    body_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    try:
        writer = csv.DictWriter(body_handle, fieldnames=BODY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in read_jsonl(stories_path):
            story_id = normalized_text(row.get("story_id"))
            body_counts["source_rows"] += 1
            if normalized_text(row.get("status")) != "success":
                body_counts["excluded_unsuccessful"] += 1
                continue
            body = row.get("body") if isinstance(row.get("body"), str) else ""
            cleaned = clean_news_html(body, min_chars=min_text_chars)
            quality_counts[cleaned.quality] += 1
            if not cleaned.text:
                body_counts[f"excluded_{cleaned.quality}"] += 1
                continue
            metadata = story_metadata.get(story_id, {})
            version_created = metadata.get("version_created", "")
            writer.writerow(
                {
                    "revision_id": metadata.get("revision_id") or revision_id(story_id, version_created),
                    "story_id": story_id,
                    "story_family_id": metadata.get("story_family_id") or story_family_id(story_id),
                    "version_created": version_created,
                    "source_code": metadata.get("source_code", ""),
                    "language": metadata.get("language", configured_language),
                    "matched_symbols": metadata.get("matched_symbols", ""),
                    "matched_rics": metadata.get("matched_rics", ""),
                    "headline": metadata.get("headline", ""),
                    "body_format": normalized_text(row.get("body_format")),
                    "text_quality": cleaned.quality,
                    "scoring_eligible": str(cleaned.quality == QUALITY_OK).lower(),
                    "original_chars": cleaned.original_chars,
                    "cleaned_chars": cleaned.cleaned_chars,
                    "removed_trailing_lines": cleaned.removed_trailing_lines,
                    "raw_body_sha256": sha256_text(body),
                    "main_body_sha256": sha256_text(cleaned.text),
                    "main_body": cleaned.text,
                }
            )
            body_counts["written"] += 1
            if not metadata:
                body_counts["missing_headline_metadata"] += 1
        body_handle.flush()
        os.fsync(body_handle.fileno())
        body_handle.close()
        os.replace(headline_temp, headlines_output)
        os.replace(body_temp, body_output)
    except BaseException:
        body_handle.close()
        headline_temp.unlink(missing_ok=True)
        body_temp.unlink(missing_ok=True)
        raise

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "built_at": datetime.now(UTC).isoformat(),
        "source_manifest": str(raw_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "exporter_version": EXPORTER_VERSION,
        "source_files": {
            "headlines_jsonl": {"path": str(headlines_path), "sha256": sha256_file(headlines_path)},
            "stories_jsonl": {"path": str(stories_path), "sha256": sha256_file(stories_path)},
        },
        "cleaner_version": CLEANER_VERSION,
        "cleaning": {
            "headline": "collapse Unicode whitespace and trim leading/trailing whitespace",
            "main_body": (
                "parse HTML, drop non-content elements/comments, preserve paragraph boundaries, normalize whitespace, "
                "and remove trailing Reuters boilerplate"
            ),
            "min_text_chars": min_text_chars,
            "timestamp": "interpret LSEG's timezone-naive version_created values as UTC, then emit UTC ISO 8601",
            "list_columns": "deduplicate, sort, and join values with |",
            "body_inclusion": (
                "include successful story fetches whose cleaned main body is non-empty; retain too_short rows with "
                "text_quality and scoring_eligible flags"
            ),
        },
        "label_mapping": None,
        "split": None,
        "random_seed": None,
        "counts": {
            "headlines": dict(sorted(headline_counts.items())),
            "main_bodies": dict(sorted(body_counts.items())),
            "body_text_quality": dict(sorted(quality_counts.items())),
        },
        "files": {
            "headlines_csv": {
                "path": headlines_output.name,
                "bytes": headlines_output.stat().st_size,
                "sha256": sha256_file(headlines_output),
            },
            "main_bodies_csv": {
                "path": body_output.name,
                "bytes": body_output.stat().st_size,
                "sha256": sha256_file(body_output),
            },
        },
        "sharing": {
            "licensed_lseg_text": True,
            "redistribute": False,
            "note": "Both CSVs contain licensed LSEG text and are local research artifacts.",
        },
    }
    manifest_temp = output_dir / ".manifest.json.tmp"
    manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(manifest_temp, output_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Completed raw LSEG collection directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Empty directory for the two CSVs")
    args = parser.parse_args()
    manifest = build(args.source, args.output_dir)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
