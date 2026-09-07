# ARS Re-review Report

**Round:** `ars-re-review-2026-08-14-round-1`
**Roadmap:** `21c899169d4aad8d516d23a51c0ec98f829c5f4174bfe17582f474b2728d3ff9`
**Decision:** **Minor Revision**
**Checker result:** `re-review synthesis ok`
**Apply-chain witness:** `pass`

## Decision basis

The authorized nine-operation patch applied cleanly. It preserved 29 of 38
canonical blocks byte-identically and changed only the mapped blocks. The
three-gate re-review then found both original `must_fix` items partially
addressed with lower-magnitude `should_fix` residuals. All four original
`should_fix` items are fully addressed. Under the ARS decision rules, that
combination yields **Minor Revision**.

| Roadmap item | Class | Phase 2A verdict | Residual | Final status |
|---|---|---|---|---|
| `REV-ESTIMAND-ALIGNMENT` | must fix | Partially addressed | The appendix project summary does not name `-0.00831` as the development baseline paired with `+0.00583`; its transition from the `-0.00914` robustness result remains ambiguous. | Partial |
| `REV-INTERPRETATION-BOUNDARY` | must fix | Partially addressed | The results synthesis and appendix project summary still use broad temporal-instability wording without consistently limiting the finding to the measured baseline pipeline or leaving its cause unidentified. | Partial |
| `REV-HYPOTHESIS-LANGUAGE` | should fix | Fully addressed | None. H1b is “not supported,” and the final summary correctly distinguishes H1a from H1b. | Complete |
| `REV-TEMPORAL-LITERATURE` | should fix | Fully addressed | None. Welch--Goyal, McLean--Pontiff and Farmer--Schmidt--Timmermann are used to bound the transport claim. | Complete |
| `REV-WORD-COUNT` | should fix | Fully addressed | None. The six numbered chapters contain 10,873 TeXcount words, 8.73% above the approximate 10,000-word target. | Complete |
| `REV-PRESENTATION` | should fix | Fully addressed | None. The rendered PDF uses Roman front matter, Arabic main matter, and “99.1 percentile.” | Complete |

## Verification record

- The chain-start integrity receipt is `PASS`: 34 of 34 original references
  were checked, no dangling or ghost citations were found, and a deterministic
  46-of-153-paragraph originality sample (30.1%) found no suspicious exact or
  close overlap.
- The added predictor-stability references were checked against their primary
  publication metadata: [Welch and Goyal (2008)](https://doi.org/10.1093/rfs/hhm014),
  [McLean and Pontiff (2016)](https://doi.org/10.1111/jofi.12365), and
  [Farmer, Schmidt and Timmermann (2023)](https://doi.org/10.1111/jofi.13229).
- Repository validation passed with 64 tests. The sealed FNSPID evaluation
  notebook was not executed.
- The manuscript built successfully to 57 A4 pages. Visual inspection covered
  all front-matter pages, the Chapter 1 transition, and the timing-placebo
  figure page. The LaTeX log contained no duplicate-destination, undefined
  citation, undefined reference, or overfull-box warning.
- No new Phase 2A issue, dissent, escalation exception, or Phase 2B adjustment
  was recorded.

## Required residual revision

The original authorization did not include canonical block `B0018`
(`manuscript/appendices/project_summary.tex`), so the remaining ambiguity was
not silently edited. A validated Round-2 roadmap now confines the correction to:

- `REV-ESTIMAND-ALIGNMENT-R2` -> `B0018`
- `REV-INTERPRETATION-BOUNDARY-R2` -> `B0014`, `B0018`

Its roadmap SHA-256 is
`8d9412dbb29253b0bf2dc2cb30e81b5e3593f07785afbc6461c124cccd0737d1`.

## Judge disclosure

Cross-model verification was not configured. This verification round ran on
the same model family that drove the revisions; over-optimization to this
judge's latent biases is possible (Ren et al. 2026, arXiv:2607.13104 §8.1.2).
