# Fresh academic re-review after external-panel remediation — 16 August 2026

## Decision

**Scientific/editorial decision: Minor Revision (dissertation scope), improved
from Major Revision.** The panel's blocking timing objection is fully addressed
by the requested return-leg test and by withdrawing the unsupported predictive
interpretation. No further model search, retuning or evaluation opening is
needed. The remaining revision status is driven by external submission
conditions and by evidence that cannot be reconstructed from the retained
checkpoint, not by a known arithmetic or inferential defect.

**Submission decision: Blocked.** The manuscript should not be submitted until
the approved project summary, assessment-specific GenAI permission and
Reuters/LSEG external-processing authority are obtained. The author must also
confirm the restricted-distribution wording and perform the final personal
read-through.

No venue was supplied, so this is a dissertation-scope judgement rather than a
journal acceptance decision. Cross-model verification was not configured; the
review was performed in the same model family that implemented the revision.

## Disposition of the major finding

The external review correctly identified that the original open/close
asymmetry and the data chapter's timestamp caveat had not been joined. That
defect no longer remains.

- The association is present assigned-open-to-close (`-0.010291`, 95% HAC(5)
  CI `[-0.015960,-0.004622]`, BH q=`0.001121`).
- It is absent assigned-close-to-next-open (`+0.001548`,
  `[-0.004158,+0.007254]`, q=`0.594922`).
- The previous-open-to-assigned-open probe is strongly negative (`-0.019257`,
  `[-0.025796,-0.012717]`, p=`7.868e-09`).
- The aggregate checkpoint cannot identify when each story became available.
  Intraday reaction, stale assignment and earlier information are therefore
  not separable.

The manuscript now says plainly that predictive timing and “next-open
specificity” are not established. This is the correct resolution: the new test
changes the claim rather than being used to defend the old one.

## What remains supported

1. **Development association, conditional and selected.** The baseline is
   `-0.00831`; price-path controls give `-0.00914`, and trailing-beta
   alternatives remain negative. The paper reports selection and winner's
   curse and does not call this independent confirmation.
2. **Temporal non-replication.** The frozen evaluation baseline is `+0.00583`;
   evaluation minus development is `+0.01414` with 95% HAC interval
   `[+0.00250,+0.02579]` and `p=0.0173`. HAC lags 0--42 and trailing-beta
   benchmarks preserve the contrast. The cause remains unidentified.
3. **The singleton mechanism is material but insufficient.** Evaluation is
   more singleton-heavy (56.53% versus 48.32%), so a pure singleton-composition
   account does not explain the sign reversal. Class, zero-share and coverage
   differences still prevent a stronger mechanism claim.
4. **Economic evidence is adverse.** The attainable-rank translation is only
   `-0.512` percentile points. The best per-side break-even estimate is `0.625`
   basis points against 10 assumed. FNSPID overlay drawdown is worse than HAR
   (`-16.36%` versus `-15.73%`). LSEG matched-control and prompt gates fail.
5. **The narrow FNSPID downside result is not a one-episode artefact.** All 44
   leave-one-episode-out intervals and the February--April 2020 exclusion
   preserve it, but this does not reverse the worse overall return/drawdown or
   create cross-source portability.

Prospective power is now separated from realised precision: 35.4% prospective
versus 26.5% realised, ex-post power at the first BH-2 hurdle. The evaluation
null is therefore not treated as proof of an exact zero.

## Integrity and literature review

The current bibliography has 39 cited records: 36 DOI records, two arXiv
records and one JSTOR record. Metadata and citation context were rechecked. The
material addition is Cakici et al. (2025), whose published replication
qualifies Farmer et al. (2023) by identifying look-ahead in the implemented
two-sided kernel and reporting that the correction removes most claimed pocket
performance. No dangling, ghost or uncited bibliography key remains.

The repository-level integrity outcome is **PASS WITH NOTES**. All seven
checked modes are clear or transparently disclosed; the qualification records
the absence of an institutional similarity checker, an author-publication
corpus and the three external permissions. See
`reviews/academic_integrity_verification_20260816.md`.

## Validation record

- PDF: `manuscript/main.pdf`, SHA-256
  `59cffc24f2d6a440db762396a3818390520c99e4cf5f06639ba6040aae52bcf6`;
  60 A4 pages; 942,293 bytes.
- Build: clean final LaTeX/BibTeX convergence, no undefined citations or
  references, no overfull boxes and all fonts embedded. Ghostscript parsed all
  pages successfully.
- Visual QA: the prior full-document render was checked page by page. Every
  changed literature, timing, conclusion and bibliography page was then
  re-rendered at 144 dpi after the final build. Tables are legible, no content
  clips or overlaps, and the nearly empty conclusion spill page was removed.
- Locked validation: repository boundary and prose checks pass; **74 tests
  pass**; fresh-cache artifact reproduction is **PASS for 27 files, including
  18 figure files**.
- Word count: **10,314** across the six numbered chapters, 3.14% above the
  programme's approximate 10,000-word target; full `texcount` total 12,848.
- `git diff --check`: clean.
- Notebook 75 was not executed. Notebooks 86 and 87 independently reproduced
  every promoted CSV byte for byte, and no raw or licensed text was added.

## Required author actions

1. Replace reconstructed Appendix B with the original approved project-summary
   form.
2. Obtain written confirmation of the dissertation's GenAI assessment category
   and the permitted scope of the disclosed assistance.
3. Establish the Reuters/LSEG licence, data-processing and ethics authority for
   the completed external prompt processing; follow programme direction if the
   prompt arm must be removed.
4. Confirm the restricted-distribution sentence and read every final page,
   method, result and citation personally before submission.

Subject to those external actions, the manuscript's scientific correction is
complete. The defensible contribution is a transparent negative-result and
model-risk sequence, not a prospectively timed trading signal.
