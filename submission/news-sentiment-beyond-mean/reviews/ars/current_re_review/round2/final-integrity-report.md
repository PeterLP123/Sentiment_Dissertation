# ARS Final Integrity Report

## Scope and verdict

- Exact accepted draft: `round2/revised_manuscript.md`.
- Accepted-draft SHA-256: `d15e6c3762b3637a7a79bd686ff6b6b3f381093688844417779a0d69473599ce`.
- Revision round: 2.
- Verdict: **PASS**.
- Open integrity issues: **0**.
- Cross-model verification: not configured; this is a same-family review.

The authorised Round-2 patch changed only canonical blocks `B0014` and
`B0018`. It preserved 36 of 38 blocks byte-identically, introduced no
structural change and replayed with an apply-chain witness of `pass`. No
experiment, aggregate result or frozen specification was changed, and sealed
Notebook 75 was not executed.

## Seven-mode research-integrity audit

| Mode | Verdict | Basis |
|---|---|---|
| Implementation bug | CLEAR | The aggregate-artifact generator completed successfully, the manuscript artefacts reproduced their promoted values, and the locked Python 3.12 environment passed all 64 tests. |
| Citation hallucination | CLEAR | All 37 bibliography records were reconfirmed through DOI or persistent scholarly records; all 37 keys are used, all 46 citation-command contexts were checked, and repository validation found no dangling or ghost citation. |
| Hallucinated result | CLEAR | All 18 promoted claim families replay to the committed aggregate evidence snapshot and evidence map. No new empirical result was introduced by the revision. |
| Shortcut reliance | CLEAR | The manuscript retains the singleton hard-boundary and repeated-story limitations, reports story-count strata, and does not claim that either shortcut is solved. |
| Bug presented as insight | CLEAR | No anomalous computation was recast as a finding. The frozen nulls, fragility diagnostics and failed economic gates remain visible. |
| Methodology fabrication | CLEAR | Methods remain consistent with the frozen specifications, manifests and provenance table; Round 2 introduced no method. |
| Frame lock | CLEAR | The adverse temporal non-replication, transaction-cost failure, prompt failures and LSEG imprecision remain intact, while prospective work is kept separate from the completed evidence. |

No mode was escalated to `SUSPECTED` during the final audit.

## Claim, citation and originality checks

- Phase E claim verification: **18/18 VERIFIED**, with zero minor distortion,
  major distortion, unverifiable claim or access failure. The formal
  `evidence-row/1.0` validator passed all 18 rows. Repeated manuscript
  restatements were reconciled to those same promoted claim families.
- E4 is recorded as `[E4-SKIPPED: no scope context]` because the workflow has
  no formal RQ Brief `scope` object. A manual boundary check nevertheless
  confirmed that FNSPID and LSEG remain separate source regimes.
- E5 found no absolute novelty or primacy claim; the manuscript explicitly
  states that dispersion itself is not new.
- E6 found no unauthorised claim-strength movement. The only wording shifts
  narrow hypothesis and interpretation language exactly as authorised by the
  two roadmap items. Numeric-token changes are confined to the authorised
  appendix clarification and reproduce existing aggregate estimands.
- The final originality screen sampled 73 of 145 eligible chapter paragraphs
  (50.34%) plus both modified passages. Seventy-four nine-word exact-fragment
  searches produced zero exact, close or verbatim match. Determination:
  **NO_SUSPICIOUS_MATCH_OBSERVED**. This public-search screen is not a Turnitin
  or iThenticate certificate; no author-publication corpus was supplied, so
  self-plagiarism was not independently checked.

## Stage-5 advisories

- Tortured-phrase screening: `not_checked / unresolved`, reason
  `SNAPSHOT_NOT_PROVIDED`. No lawful local phrase-list snapshot or detached
  manifest was supplied, so the receipt does not claim a clean result.
- Cross-document consistency: three performed pairs (abstract/results,
  discussion/results and methods/reported analyses) returned
  `NO_LISTED_INCONSISTENCY_LOCATED`. This is not proof of complete agreement.
  The manuscript/preregistration pair is explicitly `not_checked`, reason
  `COUNTERPART_DOCUMENT_MISSING`, because no completed preregistration was
  provided.
- Both advisory carriers replay-validate against accepted-draft SHA-256
  `d15e6c3762b3637a7a79bd686ff6b6b3f381093688844417779a0d69473599ce`.

## Build and presentation verification

- The six numbered chapters contain **10,889 words**, within the stated
  9,000--11,000 range and 8.89% above the approximate 10,000-word target.
- The final PDF has 57 A4 pages and SHA-256
  `53ab7883bee6e9dda94107ac74033250023e0edf1d729960746a6aad61d9c078`.
- All 57 rendered pages were inspected in contact sheets, with detailed checks
  of the abstract, the revised Results passage and the revised appendix
  Outcome. No clipping, overlap, broken glyph, undefined citation/reference,
  duplicate destination or overfull box was observed. The only LaTeX warning
  was an underfull bibliography URL line.

## Reproducibility boundary

Repository validation passed with 64 tests and five third-party
`exchange_calendars` deprecation warnings in the existing locked Python 3.12
environment. The default shell Python 3.13 environment lacks
`exchange_calendars`; its three import failures were environmental and were
not treated as manuscript regressions. No raw/licensed data, identifier-bearing
records, prompts, responses, database, Parquet panel or JSONL scoring log was
added or modified.
