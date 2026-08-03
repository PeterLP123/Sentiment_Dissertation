# LSEG 44-company, eight-month headline collection

**Frozen:** 2026-08-01, before retrieval of either extension.

## Objective

Build a broader and longer LSEG headline corpus while preserving the completed
33-company collection as an immutable source artifact. The target analysis
coverage is 44 companies across all 11 sectors and the fixed interval
2025-10-26 through 2026-06-26.

This collection is headline-only and retains every entitled English-language
source. It does not request story bodies and does not apply a Reuters filter.

## Frozen design

The final corpus is assembled offline from three separately auditable inputs:

1. Existing `us_sector_33_6m`: 33 companies, 2025-12-26 to 2026-06-26.
2. New `us_sector_33_back2m_headlines`: the same 33 companies, 2025-10-26 to
   2025-12-26.
3. New `us_sector_add11_8m_headlines`: 11 additional companies, one per sector,
   2025-10-26 to 2026-06-26.

The 11 additions are AVGO, TMUS, MCD, WFC, ABBV, EOG, RTX, COST, AEP, WELL and
SHW. They are a deliberately transparent, purposive large-cap/liquidity
extension, not a random sample and not a claim to be the exact fourth-largest
constituent of each sector. The list was frozen without inspecting its returns
or sentiment results.

Any analysis must continue to report the original sector-33 result separately.
The 44-company corpus is a breadth robustness population, not a retroactive
replacement of the original estimand.

## Retrieval controls

- 100 headlines per page.
- Daily query windows and at most 50 pages per company-day.
- Three request starts per second.
- Hard ceiling of 8,000 requests per invocation.
- Atomic page checkpoints and resumable manifests.
- No story requests (`fetch_story_bodies = false`).
- Raw and derived licensed text stays local and gitignored.

## Two-day schedule

Day 1 validates both configs and collects the two-month sector-33 backward
extension. Day 2 collects the 11-company eight-month breadth extension. Based on
the completed sector-33 request density, the two new collections are expected
to require roughly 13,000 headline-page calls in total, below two conservative
8,000-request daily budgets. Actual request counts and pagination anomalies,
not the estimate, are authoritative.

## Commands

```bash
.venv/bin/sentiment-bench lseg-news-check \
  --config configs/lseg_us_sector_33_back2m_headlines.toml \
  --all-companies

.venv/bin/sentiment-bench lseg-news-check \
  --config configs/lseg_us_sector_add11_8m_headlines.toml \
  --all-companies

caffeinate -is .venv/bin/sentiment-bench fetch-lseg-news \
  --config configs/lseg_us_sector_33_back2m_headlines.toml

# Run after the LSEG daily quota resets.
caffeinate -is .venv/bin/sentiment-bench fetch-lseg-news \
  --config configs/lseg_us_sector_add11_8m_headlines.toml
```

## Completion checks

For each collection, require a completed manifest, a consolidated
`headlines.jsonl`, zero story requests, no unresolved pagination anomaly, and a
recorded output hash. Offline consolidation must deduplicate exact shared-boundary
records using the stable headline identity fields; source collections remain
unchanged.

## Completed merged artifact

The three inputs were merged offline on 2026-08-03 with
`scripts/merge_lseg_headline_collections.py`. The merger verifies every source
manifest and headline-file hash, uses LSEG `story_id` as the deduplication key,
unions ticker/query/source-collection associations, and leaves all source files
unchanged.

- Input rows: 1,166,116.
- Duplicate story IDs across collections: 7,354.
- Output rows: 1,158,762.
- Symbols: 44.
- Coverage: 2025-10-26 00:01:06.326 through 2026-06-26 00:00:00.
- Output: `Data/collections/lseg_us_sector_44_8m_headlines/derived/merged/headlines.jsonl`.
- SHA-256: `7f48c46690f3293cbfe846ddecc701852daf3c35643f2747dcbc39f829927878`.

There were 345 duplicate-story cases with conflicting non-empty top-level
`source_code` values. The merge deterministically retains the first non-empty
value in declared source order, records the conflict count, and preserves the
underlying `raw_rows` plus all source-collection identities for audit. No rows
were removed on the basis of this metadata conflict.
