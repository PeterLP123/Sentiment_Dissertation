#!/usr/bin/env python3
"""Build the contiguous balanced LSEG sector-33 headline corpus.

The source collections remain immutable. A date is admitted only when terminal
pagination checkpoints exist for every company in the common universe. Empty
company-days remain represented in the coverage ledger; they are not invented
as headline rows.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_experiments.lib.backward_validation import (  # noqa: E402
    LsegCollectionCoverageAudit,
    audit_lseg_collection_coverage,
)
from scripts.merge_lseg_headline_collections import build, sha256_file  # noqa: E402
from sentiment_benchmark.lseg_source import load_lseg_collection_config  # noqa: E402

OUTPUT_DIR = (
    REPO_ROOT
    / "Data"
    / "collections"
    / "lseg_us_sector_33_balanced_20240101_20260806"
    / "derived"
    / "merged"
)
COLLECTIONS = (
    (
        "backward",
        Path("configs/lseg_us_sector_33_backward_20240101_20251026_headlines.toml"),
    ),
    ("back2m", Path("configs/lseg_us_sector_33_back2m_headlines.toml")),
    ("recent", Path("configs/lseg_us_sector_33_6m.toml")),
    (
        "prospective",
        Path("configs/lseg_us_sector_33_prospective_20260626_20260806_headlines.toml"),
    ),
)


class BalancedCoverageError(ValueError):
    """The declared collections do not form a complete balanced panel."""


@dataclass(frozen=True)
class BalancedCoverageSummary:
    companies: int
    dates: int
    company_date_cells: int
    complete_company_date_cells: int
    start_inclusive: str
    end_exclusive: str
    symbols: tuple[str, ...]
    source_collection_ids: tuple[str, ...]


def validate_balanced_coverage(
    audit: LsegCollectionCoverageAudit,
) -> BalancedCoverageSummary:
    """Fail closed unless every common company-date acquisition cell is complete."""

    summary = audit.summary.copy()
    ledger = audit.company_date_completion.copy()
    if summary.empty or ledger.empty:
        raise BalancedCoverageError("coverage audit is empty")
    if not summary["manifest_status"].eq("completed").all():
        raise BalancedCoverageError("every source manifest must be completed")
    if not summary["all_windows_completed"].astype(bool).all():
        raise BalancedCoverageError("at least one source has incomplete company-date windows")
    if summary["unsafe_pagination_reasons"].fillna("").astype(str).str.len().gt(0).any():
        raise BalancedCoverageError("at least one source has an unsafe pagination anomaly")
    if ledger.duplicated(["symbol", "window_start"]).any():
        raise BalancedCoverageError("company-date coverage contains duplicates")
    if not ledger["complete"].astype(bool).all():
        raise BalancedCoverageError("at least one company-date acquisition cell is incomplete")

    symbols = tuple(dict.fromkeys(ledger["symbol"].astype(str)))
    dates = tuple(dict.fromkeys(ledger["window_start"].astype(str)))
    expected_cells = len(symbols) * len(dates)
    if len(ledger) != expected_cells:
        raise BalancedCoverageError(
            f"coverage is not rectangular: expected {expected_cells} cells, found {len(ledger)}"
        )

    return BalancedCoverageSummary(
        companies=len(symbols),
        dates=len(dates),
        company_date_cells=len(ledger),
        complete_company_date_cells=int(ledger["complete"].astype(bool).sum()),
        start_inclusive=str(summary.iloc[0]["start"]),
        end_exclusive=str(summary.iloc[-1]["end"]),
        symbols=symbols,
        source_collection_ids=tuple(summary["collection_id"].astype(str)),
    )


def write_coverage_ledger(
    audit: LsegCollectionCoverageAudit,
    output_path: Path,
) -> dict[str, Any]:
    """Write the licence-safe rectangular acquisition ledger atomically."""

    rows = audit.company_date_completion.sort_values(["window_start", "symbol"])
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "symbol",
                "collection_id",
                "collection_label",
                "collection_complete",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows.itertuples(index=False):
            writer.writerow(
                {
                    "date": str(row.window_start)[:10],
                    "symbol": row.symbol,
                    "collection_id": row.collection_id,
                    "collection_label": row.collection_label,
                    "collection_complete": str(bool(row.complete)).lower(),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)
    return {
        "path": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "rows": len(rows),
    }


def main() -> None:
    collection_args = [(label, config) for label, config in COLLECTIONS]
    audit = audit_lseg_collection_coverage(REPO_ROOT, collection_args)
    coverage = validate_balanced_coverage(audit)

    source_dirs: list[Path] = []
    for _label, relative_config in COLLECTIONS:
        config = load_lseg_collection_config(REPO_ROOT / relative_config)
        source_dirs.append(REPO_ROOT / config.raw_dir)

    manifest = build(source_dirs, OUTPUT_DIR)
    coverage_file = write_coverage_ledger(audit, OUTPUT_DIR / "company_date_coverage.csv")
    manifest["balanced_coverage"] = {
        **asdict(coverage),
        "symbols": list(coverage.symbols),
        "source_collection_ids": list(coverage.source_collection_ids),
        "definition": (
            "Common 33-company universe; contiguous daily UTC acquisition windows; "
            "every retained symbol-date has a terminal pagination checkpoint."
        ),
        "zero_headline_semantics": (
            "A complete company-date may have zero matching headlines. Such cells are "
            "present in company_date_coverage.csv and do not create placeholder news rows."
        ),
        "excluded_universes": [
            "lseg_us_sector_add11_8m_headlines",
            "lseg_us_midcap_22_1y",
        ],
    }
    manifest.setdefault("files", {})["company_date_coverage_csv"] = coverage_file
    manifest_path = OUTPUT_DIR / "manifest.json"
    temp_manifest = OUTPUT_DIR / ".manifest.json.tmp"
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temp_manifest, manifest_path)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "balanced_coverage": manifest["balanced_coverage"],
                "files": manifest["files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
