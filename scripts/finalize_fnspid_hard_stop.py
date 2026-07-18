#!/usr/bin/env python3
"""Finalize the aggregate-only FNSPID report after the Gate 1 hard stop."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_benchmark.artifact_io import atomic_write_json, sha256_file
from sentiment_benchmark.runtime_metadata import collect_run_environment

HF_REVISION = "bf9189c41527198897d1af3e17b1a0095279fc45"
GITHUB_REVISION = "4054842ec476953b30ee874d4b7e8eea786a21fa"
HF_CARD_SHA256 = "422d56f3f354bd23a28e286210862c0f95a26c2a2c187761f9786e8571c3f377"
GITHUB_LICENSE_SHA256 = "0d1b0093ff52b3a207c7d33691bc7ecf5bed9eb624d107e04b38236559e094b1"
GITHUB_README_SHA256 = "8946c9f27793ee1dfa0e2ccd3d9148cc29f37109c02a51f43724f1403b9d9f8f"


def _license_record() -> str:
    return f"""# FNSPID Gate 0 license record

Frozen Hugging Face revision: `{HF_REVISION}`
Frozen GitHub revision: `{GITHUB_REVISION}`

Document hashes:

- Hugging Face `README.md`: `{HF_CARD_SHA256}`
- GitHub `LICENSE`: `{GITHUB_LICENSE_SHA256}`
- GitHub `README.md`: `{GITHUB_README_SHA256}`

## Hugging Face dataset card — verbatim

> This dataset is available under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC-4.0) license. The use of this dataset for commercial purposes is strictly prohibited without prior authorization.

> Commercial use of this code or dataset without explicit permission from the original authors is strictly prohibited. For commercial use or licensing inquiries, please contact us at `puma122707@gmail.com`.

Source: https://huggingface.co/datasets/Zihan1004/FNSPID/blob/{HF_REVISION}/README.md

## GitHub `LICENSE` — governing terms verbatim

> Creative Commons Attribution-NonCommercial 4.0 International Public License

> NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation.

> Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to: a. reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and b. produce, reproduce, and Share Adapted Material for NonCommercial purposes only.

> If You Share the Licensed Material (including in modified form), You must: a. retain the following if it is supplied by the Licensor with the Licensed Material: i. identification of the creator(s) of the Licensed Material and any others designated to receive attribution, in any reasonable manner requested by the Licensor (including by pseudonym if designated); ii. a copyright notice; iii. a notice that refers to this Public License; iv. a notice that refers to the disclaimer of warranties; v. a URI or hyperlink to the Licensed Material to the extent reasonably practicable; b. indicate if You modified the Licensed Material and retain an indication of any previous modifications; and c. indicate the Licensed Material is licensed under this Public License, and include the text of, or the URI or hyperlink to, this Public License.

Source: https://github.com/Zdong104/FNSPID_Financial_News_Dataset/blob/{GITHUB_REVISION}/LICENSE

## GitHub README contradiction — verbatim

> 07242025： We will stop the maintainance for this repo. All the right for commercial use and research use are released. Feel free to use :) .

> The use of this code for commercial purposes is strictly prohibited without prior authorization. If you wish to utilize this code in a commercial setting or for any revenue-generating activities, you are required to obtain explicit permission from the original authors.

Source: https://github.com/Zdong104/FNSPID_Financial_News_Dataset/blob/{GITHUB_REVISION}/README.md

## Gate interpretation

The formal repository license and dataset card clearly permit non-commercial academic research with attribution. Their CC BY-NC 4.0 terms are treated as controlling because the later README release statement conflicts with the unchanged formal license and with the prohibition later in the same README. **Gate 0: PASS for this dissertation's non-commercial academic research use.** This is a research-use interpretation, not legal advice.
"""


def _report(source: pd.DataFrame, counts: dict[str, int], download: dict[str, Any]) -> str:
    full_with_symbol = int(source["full_datetime_rows_with_nonempty_symbol"].sum())
    full_fraction = counts["full_datetime_rows"] / counts["rows"]
    mappable_fraction = full_with_symbol / counts["rows"]
    lines = [
        "# FNSPID dissertation-corpus feasibility gate",
        "",
        "**Decision: NO-GO at Gate 1. Gates 2–4 were not run under the predeclared cost-ordered hard-stop rule.**",
        "",
        "## Gate status",
        "",
        "| Gate | Status | Reason |",
        "| --- | --- | --- |",
        "| 0 — License | PASS | Formal CC BY-NC 4.0 terms permit non-commercial academic research with attribution. |",
        "| 1 — Structure and timestamps | HARD FAIL | No coherent ≥3-year, ≥300-firm window exists; zero firm-years reach 20 mappable news-days. |",
        "| 2 — Price validation | NOT RUN | Stopped after Gate 1. The downloaded price ZIP was not opened for outcome validation. |",
        "| 3 — Simulation precision | NOT RUN | Measured Gate 1 dimensions do not meet the corpus specification. |",
        "| 4 — Scoring and λ(0) smoke test | NOT RUN | No qualifying coherent window exists to sample. |",
        "",
        "## Gate 0 — license and immutable release",
        "",
        f"Hugging Face revision `{HF_REVISION}` and GitHub revision `{GITHUB_REVISION}` were frozen before download. The full verbatim governing excerpts and the contradictory README statements are in [`license_record.md`](license_record.md).",
        "",
        "| Upstream file | Bytes | SHA-256 | Local storage |",
        "| --- | ---: | --- | ---: |",
    ]
    for file in download["files"]:
        row = dict(file)
        lines.append(
            f"| `{row['upstream_path']}` | {int(row['upstream_size_bytes']):,} | `{row['upstream_sha256']}` | "
            f"{int(row['local_size_bytes']):,} bytes |"
        )
    lines.extend(
        [
            "",
            "All upstream hashes matched. The two CSV byte streams were stored outside git using lossless Zstandard compression; the price ZIP was already compressed. Raw content was not modified.",
            "",
            "## Gate 1 — structure, timestamps, and coverage",
            "",
            f"The physical release contains **{counts['rows']:,} CSV rows**, not 15.7M rows when both published news files are scanned together. {counts['full_datetime_rows']:,} rows ({full_fraction:.1%}) have a non-midnight clock time, but most are unusable Reuters records with no firm symbol.",
            "",
            f"Only **{full_with_symbol:,} rows ({mappable_fraction:.3%} of physical rows)** contain both a real clock time and a non-empty symbol. Deduplication leaves **{counts['mappable_deduplicated_firm_events']:,} firm events** and {counts['unique_story_events']:,} unique mappable story-days.",
            "",
            "Exact `00:00:00` values are classified conservatively as date-only or ambiguous. FNSPID does not preserve whether those midnights are genuine publication times, so they cannot support the primary session mapping.",
            "",
            "| Source site | Rows | With symbol | Full datetime | Full datetime + symbol | Deduplicated mappable firm events |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in source.sort_values("rows", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.source_site} | {int(row.rows):,} | {int(row.rows_with_nonempty_symbol):,} | "
            f"{int(row.full_datetime_rows):,} | {int(row.full_datetime_rows_with_nonempty_symbol):,} | "
            f"{int(row.mappable_deduplicated_firm_events):,} |"
        )
    lines.extend(
        [
            "",
            "The core incompatibility is structural: Reuters contributes 17,091,322 non-midnight rows but **zero symbols**. Benzinga and Nasdaq contain symbols, but only 3.66% and 1.07% of their rows respectively retain non-midnight times.",
            "",
            "The symbol×year matrix contains 6,463 nominal symbols, yet the maximum observed coverage is only **10 mappable news-days for any firm-year**. Therefore zero firms meet the required 20 news-days/year in every observed year, and no candidate window can satisfy the 300-firm condition. The 50,000-event count alone is not enough.",
            "",
            f"Deduplication removed {counts['exact_firm_event_duplicates']:,} repeated firm-day URL/headline associations. The remaining mappable events comprise {counts['unique_story_events']:,} unique story-days plus {counts['cross_symbol_story_associations']:,} retained cross-symbol associations. Two >3× source-month coverage breaks were detected, both in Benzinga (2011-05 and 2013-10).",
            "",
            "Full aggregate evidence: [`source_timestamp_audit.csv`](gate1_audit_rerun2/source_timestamp_audit.csv), [`source_year_timestamp_audit.csv`](gate1_audit_rerun2/source_year_timestamp_audit.csv), [`symbol_year_coverage.csv`](gate1_audit_rerun2/symbol_year_coverage.csv), [`candidate_windows.csv`](gate1_audit_rerun2/candidate_windows.csv), and [`coverage_breaks.csv`](gate1_audit_rerun2/coverage_breaks.csv).",
            "",
            "## Attrition",
            "",
            "| Step | Surviving | Share of raw rows |",
            "| --- | ---: | ---: |",
            f"| Physical CSV rows | {counts['rows']:,} | 100.000% |",
            f"| Valid timestamp strings | {counts['valid_timestamp_rows']:,} | {counts['valid_timestamp_rows'] / counts['rows']:.3%} |",
            f"| Non-midnight full datetimes | {counts['full_datetime_rows']:,} | {full_fraction:.3%} |",
            f"| Full datetimes with symbols | {full_with_symbol:,} | {mappable_fraction:.3%} |",
            f"| Deduplicated mappable firm events | {counts['mappable_deduplicated_firm_events']:,} | {counts['mappable_deduplicated_firm_events'] / counts['rows']:.3%} |",
            "| Firms with ≥20 mappable news-days in any year | 0 | 0.000% |",
            "| Coherent windows meeting 3y / 300 firms / 50k events | 0 | 0.000% |",
            "",
            "## Invalidated run disclosure",
            "",
            "The first Gate 1 output (`gate1_audit/`) was superseded before the decision was finalized because its source table omitted the explicit symbol-completeness columns needed to diagnose Reuters. The rerun changed no rule, threshold, input, or result; it added only `rows_with_nonempty_symbol` and `full_datetime_rows_with_nonempty_symbol`. `gate1_audit_rerun2/` is authoritative.",
            "",
            "## Ten-line plain-English summary",
            "",
            "1. The FNSPID replacement-corpus verdict is **NO-GO**.",
            "2. Its CC BY-NC 4.0 license permits this non-commercial academic audit with attribution.",
            f"3. The two published news files physically contain {counts['rows']:,} rows.",
            f"4. {counts['full_datetime_rows']:,} rows look precisely timed, but almost all of those are Reuters rows with no symbol.",
            f"5. Only {full_with_symbol:,} rows contain both a usable clock time and a firm symbol.",
            f"6. Deduplication leaves {counts['mappable_deduplicated_firm_events']:,} mappable firm events.",
            "7. No firm-year reaches the required 20 distinct mappable news-days; the observed maximum is 10.",
            "8. Therefore there is no three-year, 300-firm coherent window, regardless of the headline marketing count.",
            "9. Prices, simulations, FinBERT throughput, and λ(0) were not tested because Gate 1 is a hard stop.",
            "10. FNSPID cannot rescue the term-structure dissertation design without reconstructing symbol and timestamp metadata from original sources.",
            "",
            "## Integrity notes",
            "",
            "- Bundled FNSPID sentiment scores and summaries were never read into the audit logic.",
            "- No price return, FinBERT score, or evaluation outcome was opened after the Gate 1 failure.",
            "- Raw data remains outside git and byte-for-byte recoverable from the verified archives.",
            "- All committed outputs are aggregate-only and contain no article text.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authoritative-audit", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root
    audit_manifest = json.loads((args.authoritative_audit / "audit_manifest.json").read_text(encoding="utf-8"))
    download = json.loads(args.download_manifest.read_text(encoding="utf-8"))
    source = pd.read_csv(args.authoritative_audit / "source_timestamp_audit.csv")
    counts = {key: int(value) for key, value in audit_manifest["counts"].items()}

    artifacts = {
        "license_record.md": _license_record(),
        "report.md": _report(source, counts, download),
    }
    for name, text in artifacts.items():
        path = output / name
        if path.exists():
            parser.error(f"refusing to overwrite finalized artifact: {path}")
        path.write_text(text, encoding="utf-8", newline="\n")
    attrition = pd.DataFrame(
        [
            {"step": "physical_csv_rows", "unit": "rows", "surviving": counts["rows"]},
            {"step": "valid_timestamp_strings", "unit": "rows", "surviving": counts["valid_timestamp_rows"]},
            {"step": "non_midnight_full_datetimes", "unit": "rows", "surviving": counts["full_datetime_rows"]},
            {
                "step": "full_datetimes_with_nonempty_symbol",
                "unit": "rows",
                "surviving": int(source["full_datetime_rows_with_nonempty_symbol"].sum()),
            },
            {
                "step": "deduplicated_mappable_firm_events",
                "unit": "events",
                "surviving": counts["mappable_deduplicated_firm_events"],
            },
            {"step": "firms_with_20_mappable_news_days_in_any_year", "unit": "firms", "surviving": 0},
            {"step": "coherent_windows_meeting_all_dimensions", "unit": "windows", "surviving": 0},
        ]
    )
    attrition.to_csv(output / "attrition.csv", index=False, lineterminator="\n")
    gate_status = pd.DataFrame(
        [
            {"gate": 0, "name": "license", "status": "PASS", "reason": "CC BY-NC 4.0 academic use permitted"},
            {"gate": 1, "name": "structure_and_timestamps", "status": "HARD FAIL", "reason": "no qualifying coherent window"},
            {"gate": 2, "name": "price_panel", "status": "NOT RUN", "reason": "cost-ordered stop after Gate 1"},
            {"gate": 3, "name": "simulation_precision", "status": "NOT RUN", "reason": "cost-ordered stop after Gate 1"},
            {"gate": 4, "name": "scoring_and_smoke_test", "status": "NOT RUN", "reason": "cost-ordered stop after Gate 1"},
        ]
    )
    gate_status.to_csv(output / "gate_status.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {
                "run": 1,
                "path": "gate1_audit",
                "status": "superseded",
                "reason": "source table omitted explicit symbol-completeness diagnostics",
                "changed_rules_or_thresholds": False,
            },
            {
                "run": 2,
                "path": "gate1_audit_rerun2",
                "status": "authoritative",
                "reason": "same inputs and rules with diagnostic-only columns added",
                "changed_rules_or_thresholds": False,
            },
        ]
    ).to_csv(output / "invalidated_runs.csv", index=False, lineterminator="\n")

    finalized = [
        output / "license_record.md",
        output / "report.md",
        output / "attrition.csv",
        output / "gate_status.csv",
        output / "invalidated_runs.csv",
    ]
    manifest = {
        "schema_version": 1,
        "status": "completed_hard_stop",
        "decision": "NO-GO",
        "stopped_after_gate": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "dataset_revision": HF_REVISION,
        "github_revision": GITHUB_REVISION,
        "inputs": {
            "download_manifest": {
                "path": str(args.download_manifest),
                "sha256": sha256_file(args.download_manifest),
            },
            "authoritative_audit_manifest": {
                "path": str(args.authoritative_audit / "audit_manifest.json"),
                "sha256": sha256_file(args.authoritative_audit / "audit_manifest.json"),
            },
        },
        "files": {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in finalized},
        "environment": collect_run_environment(),
    }
    atomic_write_json(output / "manifest.json", manifest)
    print(output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
