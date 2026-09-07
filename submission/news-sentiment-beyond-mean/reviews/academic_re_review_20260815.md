# Fresh academic re-review — 15 August 2026

## Scope and criteria binding

This is a fresh, read-only review of the current manuscript PDF and its
supporting aggregate evidence. It assesses scientific validity, internal
consistency, reproducibility and dissertation submission readiness. No journal
or conference target was supplied, so venue-fit criteria remain
`criteria_binding_unavailable` and no journal acceptance claim is made.

The current PDF is `manuscript/main.pdf`, SHA-256
`3b8e858f2e6b4adddae3334b9f1f14f79e939eca60cab941f24418b006afcf60`.
It is not the byte-identical artifact accepted by the older formal ARS Round-2
chain. This report therefore supersedes that report as a review of the current
draft, but it is not represented as a new checker-backed ARS contract replay.

## Decision

**Scientific/editorial decision: Minor Revision.**

The central, bounded conclusion is supported: a selected development-era
association fails the frozen later-era replication, is not economically useful
after costs, and does not establish a portable trading or risk rule. The draft
now separates the matched development baseline from the price-control
robustness estimate, preserves the adverse replication result, reports power
without converting a null into proof of zero, and limits the era contrast to
instability in the measured pipeline association. No new accounting or result-
reconciliation error was found.

**Submission decision: Blocked.** The approved project-summary form and the
assessment-specific GenAI permission remain external administrative gates.
They are not scientific defects, but the present PDF should not be submitted
until both are resolved.

## Verification of the repair set

| Repair | Verdict | Verification |
|---|---|---|
| Rebalance from return-drifted weights | **Fully addressed** | `experiments/lib/evaluate.py` and `experiments/lib/thresholds.py` now use raw asset returns to drift the prior book before the next rebalance; focused regression tests pass. |
| Charge opening and final liquidation | **Fully addressed** | Both portfolio paths add terminal half-L1 turnover and its two-sided cost to the last interval; one-period and final-exit tests pass. |
| Handle leveraged-book insolvency | **Fully addressed** | Non-positive wealth raises a dated `PortfolioInsolvencyError`; the label-shuffled negative control is reported as insolvent rather than assigned misleading cost metrics. |
| Correct the adjusted-open return description | **Fully addressed** | The data chapter defines `open * (adjusted close / close)` as a Yahoo-ratio proxy carrying the adjustments encoded in the archive, without calling it an independently audited total-return series. |
| Synchronise economic outputs | **Fully addressed** | Notebook 03, 10, 71, 77 and 78 aggregate outputs, provenance hashes, figures, tables, evidence map and manuscript prose agree on the corrected accounting. The best direct-signal break-even is 0.625 bps per side and the 10 bps economic conclusion remains negative. |
| Separate baseline, robustness and replication estimands | **Fully addressed** | The abstract, results, conclusion and project-summary appendix consistently pair development `-0.00831` with frozen evaluation `+0.00583`; `-0.00914` is separately labelled as development price-control robustness. |
| Bound the cross-era interpretation | **Fully addressed** | The draft consistently calls the result instability/non-replication of the measured pipeline association and leaves market, composition, timing and classifier explanations unidentified. |
| Bound the sparse-versus-dense news interpretation | **Fully addressed** | The 48.3% singleton share, story-count attenuation, LSEG medians of 61/90 versus 2, and failure to support a rich-news distribution effect are all explicit. |
| Surface administrative gates | **Partly addressed by design** | The manuscript and `docs/submission_gates.md` disclose the missing approved form and unresolved GenAI category. Only external confirmation can close them. |

## Remaining required revisions

### 1. Keep the FNSPID outcome language inside the timing evidence

**Severity for the present central claim: minor. Severity for any predictive or
executable claim: major.**

The date-only branch is conservative, but the retained checkpoint has neither
row-level timestamp precision nor the share/rule of the timestamped branch.
The open-to-open association is absent in the close-to-close and FF3-residual
sensitivities. The manuscript now discloses this well, so the temporal
non-replication conclusion survives. A few phrases still outrun the evidence,
notably “one executable session” in Chapter 3 and language that can read as
strictly prospective “next-open information”.

Minimum text remedy: call the primary outcome the **assigned-session
open-to-open interval** and state once that timestamped rows may make part of
the association contemporaneous rather than prospectively tradable. A stronger
predictive claim would require the earlier requested intraday/overnight
decomposition and the timestamped-versus-date-only branch share; the current
clean repository cannot supply those row-level tests.

### 2. Specify the HAR benchmark sufficiently for reproduction

**Severity: minor.**

Section 4.9 says only that daily, weekly and monthly log variance predicts
five-session log variance. The committed notebook/helper reveals material
details that should be in the methodology: OLS; squared SPY adjusted-open
returns known at the decision open; 1-, 5- and 22-session components; the mean
of the next five squared returns as target; exponentiation followed by the
positive development-only QLIKE scale; and the 21-session expanding
walk-forward refit in the historical robustness arm. Two compact sentences or
one equation would close the gap.

### 3. Close the two external submission gates

**Severity: blocking for submission, not a scientific revision.**

1. Replace the reconstructed Appendix B scope page with the original approved
   project-summary form.
2. Obtain written confirmation of the assessment-specific GenAI category and
   that the assistance disclosed in Appendix A is permitted. Disclosure alone
   does not establish permission.

Also confirm that the restricted-distribution sentence on the title page is
the intended option.

## Repository advisories

These do not change the scientific decision:

- A fresh artifact run reproduced all generated numbers and TeX tables exactly,
  but the nine PNG/PDF figures were not pixel- or dimension-identical under a
  fresh Matplotlib font cache. The plotted values and current PDF were visually
  correct. Pinning or bundling one font and avoiding environment-sensitive
  tight bounding boxes would make figure reproduction stronger.
- Historical Notebook 76/77/78 specifications or manifests still say Notebook
  75 “remains deliberately unexecuted”. That was true when those studies were
  frozen, but not now. Preserve the frozen specifications; add an explicit
  current-status note beside the result manifests so a reader cannot mistake a
  historical execution-state statement for the repository's present state.

## Validation record

- PDF structural preflight: **PASS**, 58 declared/enumerated/read pages, no
  warnings.
- Visual inspection: all 58 pages reviewed; no clipping, broken tables,
  malformed figures or pagination defect found.
- `make validate`: repository boundary and prose checks passed; **68 tests
  passed**. The five warnings are third-party `exchange_calendars`/NumPy
  deprecations.
- `git diff --check`: clean.
- No manuscript source, bibliography or included figure is newer than the
  reviewed PDF.
- Regenerated numerical/TeX artifacts match the committed versions exactly.
- Sixteen locally available provenance targets were re-hashed; all sixteen
  matched their ledger entries. The remaining ledger targets refer to the
  private source archive and were not reverified here.
- The sealed Notebook 75 was not executed during this review.

## Overall assessment

This is now a strong negative-results dissertation rather than a weak trading
paper. Its best contribution is the disciplined evidence sequence: selected
development association, frozen adverse replication, direct era contrast,
economic translation, matched comparators, power bounds and preserved nulls.
The two scientific revisions above are small and do not require changing any
reported result. The administrative gates do require the author's external
action before submission.

## Reviewer-family disclosure

Cross-model verification was not configured. This verification round ran on
the same model family that drove the revisions; over-optimization to this
judge's latent biases is possible (Ren et al. 2026, arXiv:2607.13104, section
8.1.2).
