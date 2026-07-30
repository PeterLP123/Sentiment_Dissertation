# Licence and provenance — FNSPID earnings calendar (LSEG)

**Access date (UTC):** 2026-07-30T14:21:28Z (initial pull); filters refreshed same day.  
**Source:** LSEG Workspace desktop session via the `lseg.data` Python library
(EventType `RES` = earnings / results releases).  
**Script:** `scripts/fetch_fnspid_earnings_lseg.py`  
**Cohort:** `reports/loop_vader_scale_20260719/cohort_symbols.csv` (574 FNSPID
coherent symbols, window 2011–2023).

## Redistribution

Rows are LSEG-derived reference data. They stay under
`final_experiments/data/earnings/` and are gitignored. Do not commit the CSV /
JSON payloads or redistribute them outside authorised Workspace entitlements.

## Fields requested

- `TR.EventStartDate`
- `TR.EventStartTime`
- `TR.EventType` (parameter `EventType=RES`)
- `TR.EventTitle`
- `TR.EventStatus`
- `TR.EventEventID`

## Aggregate outcome (safe to cite)

| Metric | Value |
| --- | ---: |
| Cohort symbols | 574 |
| RICs resolved | 574 |
| Issuer-matched calendar rows | 23,139 |
| Quarterly subset rows (WS6 default) | 22,883 |
| Symbols with ≥1 quarterly event | 490 |
| Quarterly timing BMO / AMC / during / unknown | 12,303 / 9,764 / 202 / 614 |

Symbols with zero kept events (84 of 574) break down as: 35 ETF/fund, 22
delisted/acquired, 10 tickers mis-resolved onto futures/options roots, and 17
equity/ADR names dropped by the issuer-title filter (name-change /
parenthetical-phrase failures). The last two groups are recoverable coverage
holes; see `README.md` quality notes and `final_experiments/00_data_inventory`
§8 for the classification.

## yfinance fallback

Not used for this gather. LSEG was preferred while Workspace was available.
