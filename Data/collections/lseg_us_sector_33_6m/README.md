# LSEG US Sector 33 — Six-Month Collection

This folder contains all generated artifacts for the 33-company, 11-sector LSEG
Workspace collection covering `2025-12-26T00:00:00Z` through
`2026-06-26T00:00:00Z`.

The canonical collection definition is
[`configs/lseg_us_sector_33_6m.toml`](../../../configs/lseg_us_sector_33_6m.toml).
Its queries request all entitled English-language sources; there is no
Reuters-only source filter.

## Layout

- `raw/lseg_us_sector_33_6m/` — resumable headline and story checkpoints,
  aggregate raw files, and the raw manifest.
- `derived/us_sector_33_6m/` — deterministic cleaned corpus, screening index,
  and derived manifest.
- `reports/` — collection-flow visualizations and research summaries created
  from the manifests.
- `packages/` — ZIP archives prepared for authorized sharing.

Generated contents under `raw/`, `derived/`, `reports/`, and `packages/` are
ignored by Git. They may contain licensed LSEG text and must not be committed or
redistributed without the necessary permission.

## Commands

Run these commands from the repository root:

```bash
.venv/bin/sentiment-bench lseg-news-check \
  --config configs/lseg_us_sector_33_6m.toml

.venv/bin/sentiment-bench fetch-lseg-news \
  --config configs/lseg_us_sector_33_6m.toml

.venv/bin/sentiment-bench build-lseg-corpus \
  --source Data/collections/lseg_us_sector_33_6m/raw/lseg_us_sector_33_6m
```

The fetch command displays a resume-aware live dashboard. Its headline phase
tracks completed company/day windows and estimates the phase ETA; after all
headlines are known, it switches to story-level progress and the final
completion ETA. Pressing `Ctrl-C` is safe—run the same command to resume from
the saved checkpoints.

## Clean CSV Export

For local analysis, build two flat UTF-8 CSVs from the hash-verified raw
collection:

```bash
.venv/bin/python scripts/build_lseg_clean_csvs.py \
  --source Data/collections/lseg_us_sector_33_6m/raw/lseg_us_sector_33_6m \
  --output-dir Data/collections/lseg_us_sector_33_6m/derived/us_sector_33_6m/csv
```

- `headlines.csv` contains one normalized headline per unique LSEG story
  revision, with UTC timestamps, deterministic IDs and hashes, source, ticker,
  and RIC fields. LSEG's timezone-naive `version_created` values are interpreted
  as UTC, consistent with the collection windows and analysis pipeline.
- `main_bodies.csv` contains successful Reuters story bodies converted from HTML
  to paragraph-preserving plain text. It records cleaning quality and retains
  short but non-empty bodies with `scoring_eligible=false`.
- `manifest.json` records input hashes, cleaning rules, row counts, exclusions,
  output hashes, and the no-redistribution constraint.

The CSVs contain licensed LSEG text. Keep them local unless every recipient is
authorized to receive that content.

## Package For Sharing

The cleaned corpus is normally the useful package. After collection and
cleaning complete, run:

```bash
cd Data/collections/lseg_us_sector_33_6m
zip -r packages/us_sector_33_6m_cleaned.zip \
  derived/us_sector_33_6m \
  reports \
  README.md
```

To preserve every raw checkpoint as well, create a complete archive instead:

```bash
cd Data/collections/lseg_us_sector_33_6m
zip -r packages/us_sector_33_6m_complete.zip \
  raw/lseg_us_sector_33_6m \
  derived/us_sector_33_6m \
  reports \
  README.md
```

Only create or distribute either archive when every recipient is authorized to
receive the included LSEG story text. The complete archive is substantially
larger and is mainly useful for reproducibility or offline recleaning.
