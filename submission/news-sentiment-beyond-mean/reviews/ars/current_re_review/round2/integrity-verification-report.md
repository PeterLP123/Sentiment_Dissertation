# Round-2 Chain-Start Integrity Verification

## Scope and verdict

- Exact checked draft: `round2/original_manuscript.md`.
- Draft SHA-256: `5f73b895261c314d22538104b3dfc797b71f06425b3a8554b1df9269466bd7e0`.
- Verdict: **PASS**.
- Open integrity issues: **0**.
- Cross-model verification: not configured; this is a single-family check.

## Evidence

- The exact draft is the hash-bound output of Round 1. Its ARS apply report
  records 29 of 38 blocks preserved byte-identically, and its re-review checker
  returned an apply-chain witness of `pass`.
- The bibliography contains 37 retained sources. `research/source_verification.md`
  records primary or persistent-source checks for all 37, including all three
  references introduced during Round 1.
- Repository validation on 14 August 2026 found no missing or dangling citation
  key, no missing manuscript include or graphic, no raw-data boundary breach and
  no obvious secret. The prose checker passed all 16 TeX files.
- The locked Python 3.12 environment completed all 64 tests; five warnings came
  from third-party `exchange_calendars` deprecations. No notebook was executed,
  including sealed Notebook 75.
- The initial chain-start originality screen sampled 46 of 153 eligible body
  paragraphs (30.1%) and found no suspicious exact or close overlap. The Round-1
  revision delta is fully represented by its authorized patch and contains no
  copied external passage. A separate from-scratch originality and research-
  integrity pass remains required on the final post-revision manuscript.

This receipt establishes the exact pre-Round-2 draft for deterministic replay.
It does not substitute for the final post-revision integrity gate.
