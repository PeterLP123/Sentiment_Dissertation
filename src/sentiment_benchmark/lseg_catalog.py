from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_io import atomic_write_json, atomic_write_text, read_json, sha256_file
from .lseg_source import DEFAULT_LSEG_DERIVED_ROOT, LsegNewsError, utc_now

LSEG_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LsegCatalogResult:
    derived_root: Path
    catalog_json: Path
    catalog_csv: Path
    corpus_count: int
    eligible_count: int


def _manifest_entries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    entries: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return entries
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            continue
        if manifest.get("status") == "completed":
            entries.append((manifest_path, manifest))
    return entries


def _catalog_entry(root: Path, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    collection = config.get("collection") if isinstance(config.get("collection"), dict) else {}
    companies = config.get("companies") if isinstance(config.get("companies"), list) else []
    symbols = sorted(str(company.get("symbol")) for company in companies if isinstance(company, dict) and company.get("symbol"))
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    text_quality = counts.get("text_quality") if isinstance(counts.get("text_quality"), dict) else {}
    return {
        "collection_id": str(collection.get("id") or manifest_path.parent.name),
        "start": str(collection.get("start") or ""),
        "end": str(collection.get("end") or ""),
        "window_days": collection.get("window_days"),
        "company_count": len(symbols),
        "companies": symbols,
        "articles": int(counts.get("articles", 0) or 0),
        "eligible": int(counts.get("eligible", 0) or 0),
        "ineligible": int(counts.get("ineligible", 0) or 0),
        "text_quality": dict(sorted((str(key), int(value)) for key, value in text_quality.items())),
        "built_at": str(manifest.get("built_at") or ""),
        "cleaner_version": str(manifest.get("cleaner_version") or ""),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "relative_manifest_path": str(manifest_path.relative_to(root)) if manifest_path.is_relative_to(root) else str(manifest_path),
        "redistribute": bool((manifest.get("sharing") if isinstance(manifest.get("sharing"), dict) else {}).get("redistribute", False)),
        "licensed_full_text": bool(
            (manifest.get("sharing") if isinstance(manifest.get("sharing"), dict) else {}).get("licensed_full_text", False)
        ),
    }


def _write_catalog_csv(path: Path, entries: list[dict[str, Any]]) -> Path:
    fields = [
        "collection_id",
        "start",
        "end",
        "window_days",
        "company_count",
        "companies",
        "articles",
        "eligible",
        "ineligible",
        "text_quality",
        "built_at",
        "cleaner_version",
        "manifest_path",
        "manifest_sha256",
        "redistribute",
        "licensed_full_text",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        row = {field: entry.get(field, "") for field in fields}
        row["companies"] = "|".join(entry.get("companies") or [])
        row["text_quality"] = json.dumps(entry.get("text_quality") or {}, sort_keys=True)
        writer.writerow(row)
    return atomic_write_text(path, stream.getvalue())


def build_lseg_catalog(derived_root: str | Path = DEFAULT_LSEG_DERIVED_ROOT) -> LsegCatalogResult:
    root = Path(derived_root)
    root.mkdir(parents=True, exist_ok=True)
    entries = [_catalog_entry(root, path, manifest) for path, manifest in _manifest_entries(root)]
    entries.sort(key=lambda row: (str(row["collection_id"]), str(row["start"]), str(row["end"])))
    catalog_csv = _write_catalog_csv(root / "catalog.csv", entries)
    catalog_json = atomic_write_json(
        root / "catalog.json",
        {
            "schema_version": LSEG_CATALOG_SCHEMA_VERSION,
            "generated_at": utc_now(),
            "derived_root": str(root),
            "corpus_count": len(entries),
            "eligible_count": sum(int(entry["eligible"]) for entry in entries),
            "corpora": entries,
            "files": {
                "catalog_csv": {"path": catalog_csv.name, "sha256": sha256_file(catalog_csv)},
            },
            "sharing": {
                "licensed_full_text": False,
                "redistribute": True,
                "note": "Catalog files contain metadata and hashes only, not story text.",
            },
        },
    )
    return LsegCatalogResult(
        derived_root=root,
        catalog_json=catalog_json,
        catalog_csv=catalog_csv,
        corpus_count=len(entries),
        eligible_count=sum(int(entry["eligible"]) for entry in entries),
    )


def read_lseg_catalog(path: str | Path = DEFAULT_LSEG_DERIVED_ROOT / "catalog.json") -> dict[str, Any]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise LsegNewsError(f"LSEG catalog does not exist: {catalog_path}")
    return read_json(catalog_path)
