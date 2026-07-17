#!/usr/bin/env python3
"""Fetch one hash-manifested broad-market series from LSEG Workspace."""

from __future__ import annotations

import argparse
import csv
import io
from dataclasses import asdict
from pathlib import Path

from sentiment_benchmark.artifact_io import atomic_write_json, atomic_write_text, sha256_file
from sentiment_benchmark.prices import LsegPriceProvider
from sentiment_benchmark.runtime_metadata import collect_run_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="^GSPC", help="Canonical symbol written to the output.")
    parser.add_argument("--ric", required=True, help="LSEG historical-pricing instrument.")
    parser.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        parser.error(f"refusing to overwrite existing output: {output}")

    provider = LsegPriceProvider(ric_overrides={args.symbol: args.ric})
    rows = provider.fetch([args.symbol], args.start, args.end)
    if not rows:
        parser.error(f"LSEG returned no rows for {args.symbol} ({args.ric})")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0].__dataclass_fields__), lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    atomic_write_text(output, buffer.getvalue())
    atomic_write_json(
        manifest,
        {
            "schema_version": 1,
            "status": "completed",
            "provider": "lseg",
            "symbol": args.symbol,
            "ric": args.ric,
            "requested_start": args.start,
            "requested_end": args.end,
            "return_convention": "split-adjusted price returns; dividends not back-adjusted",
            "counts": {"rows": len(rows), "sessions": len({row.session_date for row in rows})},
            "coverage": {
                "first_session": min(row.session_date for row in rows),
                "last_session": max(row.session_date for row in rows),
            },
            "file": {"path": output.name, "sha256": sha256_file(output)},
            "environment": collect_run_environment(),
        },
    )
    print(f"Fetched {len(rows)} rows for {args.symbol} ({args.ric}): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
