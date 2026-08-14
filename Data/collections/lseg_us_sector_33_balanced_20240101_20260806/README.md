# LSEG US Sector 33 — Balanced 2024-01-01 to 2026-08-06

This is the combined, contiguous headline dataset for the original 33-company
LSEG sector universe. It admits a UTC acquisition date only when all 33
companies have terminal pagination checkpoints for that date.

The generated dataset is under `derived/merged/` and contains:

- `headlines.jsonl`: story-ID-deduplicated LSEG headline rows with source
  collection provenance and merged company associations;
- `company_date_coverage.csv`: the complete 33-company by acquisition-date
  ledger, including complete dates on which a company returned no headlines;
- `manifest.json`: source hashes, merge rules, counts, coverage definition and
  output hashes.

The source collections are immutable and remain separate. The combined corpus
uses these four contiguous collections:

1. sector-33 backward headlines, 2024-01-01 to 2025-10-26;
2. sector-33 back-two-month headlines, 2025-10-26 to 2025-12-26;
3. sector-33 six-month headlines, 2025-12-26 to 2026-06-26;
4. sector-33 prospective headlines, 2026-06-26 to 2026-08-06.

The added-11 and mid-cap-22 collections are excluded because those companies
are not observed over the full interval. Reuters bodies are also not copied:
body retrieval exists only for part of the six-month source period, whereas
headline coverage is consistent across the complete balanced interval.

Build from the repository root:

```bash
.venv/bin/python scripts/build_lseg_balanced_headline_collection.py
```

All generated files contain or describe licensed LSEG data, are ignored by Git,
and must remain local unless recipients are authorised under the applicable
data agreement. The tracked README contains no licensed text.
