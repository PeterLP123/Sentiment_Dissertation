# Academic revision completion and focused re-review — 16 August 2026

## Scope and criteria binding

This note closes the two scientific revisions and two repository advisories in
`reviews/academic_re_review_20260815.md`. It reviews the current LaTeX sources,
aggregate evidence and rebuilt PDF. No journal or conference target was
supplied, so venue-fit criteria remain `criteria_binding_unavailable`; the
decision below is a dissertation-scope readiness judgement, not a journal
acceptance claim.

## Decision

**Focused scientific/editorial verdict: Accept for the current dissertation
scope.** The two minor revisions identified on 15 August are fully addressed,
and this focused pass found no new scientific, accounting, citation or layout
defect that requires another manuscript change. The central conclusion remains
bounded: the development association does not replicate in the frozen later
era and does not support a portable, economically useful trading or risk rule.

**Submission verdict: Blocked only by external administrative evidence.** The
repository cannot supply or infer the approved project-summary form or the
assessment-specific GenAI permission. The author must also confirm the chosen
restricted-distribution sentence on the title page before submission.

## Revision closure

| Re-review finding | Status | Closure evidence |
|---|---|---|
| FNSPID timing language outran the observable timing data | **Closed** | The abstract, research question, data, methodology, results, conclusion and reconstructed scope appendix now call the outcome the assigned-session open-to-open abnormal return. Chapter 3 states that the interval is not guaranteed to begin at the first open after verified publication and that timestamped rows may make part of the association contemporaneous rather than prospectively tradable. |
| HAR benchmark was underspecified | **Closed** | Section 4.9 now gives the OLS log-variance equation, adjusted-open return information set, 1/5/22-session inputs, next-five-session target, exponentiation and positive development-only QLIKE scale, evaluation cutoff, 504-row start, 21-session expanding refit, completed-target restriction, exposure rule and costs. These details agree with `experiments/lib/volatility_target.py` and Notebooks 21 and 25. |
| Figure files varied with font-cache state and tight bounding boxes | **Closed** | `uv.lock` pins the declared Python 3.12 numerical and figure toolchain; the generator rejects a mismatched environment and uses only DejaVu Sans's concrete normal and bold faces. `make validate` now performs a real fresh-cache byte comparison that fails on any mismatch or font fallback. It reproduced all 25 generated artifacts exactly, including all 18 PNG/PDF figure files, and every figure remains fully visible in the rebuilt paper. |
| Historical manifests could be mistaken for the current Notebook 75 state | **Closed without rewriting frozen history** | `experiments/results/README.md` records that the older manifests describe their execution-time state, while Notebook 75 was later authorised and executed once on 14 August 2026. The evidence map points readers to that note. Notebook 75 was not executed during this revision. |
| Approved scope form, GenAI permission and title-page distribution choice | **External blocker remains** | `docs/submission_gates.md` records the missing evidence and the exact actions needed. No approval or form was fabricated. |

## Validation record

- Current PDF: `manuscript/main.pdf`, SHA-256
  `d79332af05cf2fc745f22f983f5b3df37f5dd214c241e5585a3764140e1c51fc`;
  58 A4 pages; 849,990 bytes.
- LaTeX/BibTeX: clean build, zero BibTeX warnings, no undefined citations or
  references, no overfull boxes and all fonts embedded. `qpdf` was unavailable;
  Ghostscript successfully parsed every page as the structural substitute.
- Visual QA: all pages were inspected in the preceding full review; all nine
  figures and the revised reproducibility appendix were re-rendered in context
  and inspected at high resolution. No clipping, collision, broken figure,
  near-empty spill page or other pagination defect remains.
- Artifact reproducibility: locked fresh-cache comparison **IDENTICAL (25
  files, including all 18 PNG/PDF figures)**, with no font fallback.
- `make validate` under the declared Python 3.12 environment: repository
  boundary valid, prose style passed and **68 tests passed**.
- `make lint`: **passed**.
- `git diff --check`: **clean**.
- Numbered chapters: **10,999 words**, within the manuscript workflow's
  $\pm10\%$ planning tolerance around the programme's approximate 10,000-word
  target; full `texcount` total including front matter, appendices and generated
  tables: 13,228.
- The sealed Notebook 75 was not executed, no source regime was pooled, and no
  licensed or raw-text input was added.

## Remaining author actions

1. Replace reconstructed Appendix B with the original approved project-summary
   form.
2. Obtain written confirmation of the dissertation's assessment-specific GenAI
   category and that the Appendix A disclosure is permitted.
3. Confirm that the restricted-distribution sentence on the title page is the
   intended option.

Until those three actions are complete, the paper is scientifically revised
but not administratively ready to submit.
