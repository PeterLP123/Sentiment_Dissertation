# Dissertation quality review

Review date: 21 August 2026.

This is a source-level check against the supplied UCL requirements and marking rubric. It records what the repository can establish and separates the remaining administrative decisions that only the author or programme can resolve.

## Rubric coverage

| Criterion | Evidence in the manuscript | Review outcome |
|---|---|---|
| Structure, clarity and research hypothesis | One research problem separated into training-period and testing-period questions, five declared claim components (H1a, H1b and H2--H4) and a results section that answers each | Covered |
| Computational-finance content | Assigned-window, intraday, overnight and prior-open timing; market- and trailing-beta-adjusted returns; cross-sectional ranks; HAR volatility targeting; turnover, costs and exposure controls | Covered |
| Choice of models | FinBERT revision, nine aggregation rules, rank regression, model-free sort, HAR base, matched comparators and a declared post-hoc FF3/soft-mass family are defined and justified | Covered |
| Results | Exact estimates and uncertainty appear beside eleven figures and eight numbered results tables, including the nine-rule family, outcome-leg decomposition, sensitivity panel, MDEs and final verdicts; favourable and adverse findings use the same claim standard | Covered |
| Validation | Chronology, outcome-leg identities, prior-open timing, HAC lag and dependence diagnostics, BH correction, trailing-beta adjustment, block bootstraps, price-path controls, story-count strata, episode and crash exclusions, MDEs, matched exposure and cross-source checks | Covered |
| Consistency and notation | A4, 12-point type, 2.5 cm margins, one-and-a-half spacing, numbered equations, explicit return-leg notation and source-specific return labels | Covered |
| Conclusions and further work | Direct answer, explicit investment boundary and five testable follow-on designs | Covered |
| Background literature | 39 retained sources; 36 DOI records, two arXiv records and one JSTOR record checked against current primary or authoritative metadata and substantive content | Covered |

## Claim audit

- The main coefficient is described as an exploratory conditional association, never as a causal effect or independent confirmation.
- The outcome-leg decomposition is reported in full. The training-period association is intraday (`-0.01029`, BH q=`0.0011`) and absent post-close (`+0.00155`, q=`0.595`). A strongly negative previous-open probe keeps assignment lag and earlier information live. The manuscript therefore withdraws next-open specificity and predictive timing rather than reading the asymmetry as confirmation.
- The story-count strata are reported in full. The coefficient retains about 22% of its full-sample size once three same-day stories are required, and the testing period is more singleton-heavy (56.5% versus 48.3%). The threshold mechanism remains material but cannot by itself explain the sign reversal.
- The price-path test addresses reversal, five-session momentum and recent volatility. It does not claim to remove all omitted-variable bias.
- The double sort and rank translation are labelled differently: one is in basis points and the other is in return-rank percentile points. The translation uses the observed tied-regressor 90th-minus-10th spread of 0.56056; the former hypothetical 0.8 move is explicitly identified as unattainable.
- FNSPID and LSEG are not pooled. Figure captions state that their return outcomes differ.
- The LSEG nulls are paired with MDEs and are described as imprecise, not as proof of zero effect.
- Losses avoided by the firm-level LSEG rule are reported with upside forgone, extra cost and ending wealth against the matched control.
- Prompt results stop at the 2024 selection gate; no 2025 prompt portfolio is reported.
- The backward LSEG downside result is reported with its concentration: two of eighteen risk-off episodes supply 65.6% of the squared-downside reduction, alongside the leave-one-episode-out and block-length checks that survive.
- The frozen 56-family ledger is labelled with its Notebook 73 cut-off. Later focal families appear in a separate appendix table rather than being hidden or added retrospectively to the old total.
- Close-to-close, FF3 residual and soft-mass tests are labelled as one post-hoc family frozen after the headline result. None survives BH correction. They do not upgrade or reopen H1.
- The selected FNSPID association was repeated once on the frozen 2020--2023 block. Neither declared conditional coefficient nor any of the nine aggregation rules survives correction. The predeclared testing-minus-training contrast is reported directly. Prospective power was 35.4%; realised power at the first hurdle is 26.5% because the testing-period HAC standard error was 15.8% above projection. The non-replication is not misdescribed as proof of zero.
- The abstract and conclusion distinguish the training-period baseline coefficient
  (-0.00831) from the full price-control robustness coefficient (-0.00914). The
  reported +0.01414 era contrast is explicitly tied to the baseline coefficient
  series used by Notebook 75.
- The era contrast is described as instability in the measured pipeline
  association. Firm/news composition, mapping, timing and classifier changes
  remain competing explanations rather than being relabelled as a market-regime
  mechanism.
- Trailing-beta adjustment, beta control, HAC lags 0--42 and daily-series dependence diagnostics preserve the era contrast. Forty-four episode deletions and a February--April 2020 exclusion preserve the adverse FNSPID overlay-minus-base return. These checks narrow concentration explanations but do not convert either result into a causal or portable mechanism.
- FNSPID maximum drawdown is reported symmetrically with LSEG: the overlay is worse (`-16.36%`) than the HAR base (`-15.73%`).

## Language review

The prose was read for directness as well as technical correctness. An automated check rejects a narrow list of stock machine-written phrases, including `delve`, `pivotal`, `multifaceted`, `this underscores`, `important to note`, `not only ... but also`, and generic `in conclusion` openings. The manual pass also removed unsupported emphasis, decorative claims of novelty and a section title that called a period-specific timing result “real”. The remaining prose normally states the sample, estimate and limitation in the same paragraph.

## Reproducibility and presentation checks

- `scripts/generate_dissertation_artifacts.py` reads committed aggregate outputs only and rebuilds ten figures, seven generated results tables, derived metrics and a headline-number file: 30 locked artifacts in total.
- `scripts/validate_repository.py` checks LaTeX includes, graphics, citation keys, unreferenced figure and table labels, draft markers, raw-data formats, raw-text columns and obvious secrets.
- `scripts/check_prose_style.py` checks all manuscript TeX sources.
- Seventy-four focused analytical tests pass in the locked Python 3.12 research environment.
- `latexmk` compiles the bibliography and cross-references without undefined citations, undefined references or overfull boxes. The revised build is 62 A4 pages; its remaining underfull boxes are confined to long URLs and bibliography entries.
- All 62 pages of the revised PDF were rendered and visually inspected after the terminology, timing, equation and compression pass. The analytical figures were also inspected at source resolution; no clipping, overlap or unreadable labels remain.
- `texcount` reports the main-chapter total separately from front matter and appendices in `docs/word_count.md`. The main chapters contain 9,265 words, 7.35% below the approximate 10,000-word target and inside the manuscript workflow's planning range. The compression removed 4,110 words from the 13,375-word draft while preserving equations, estimates, intervals, corrections and evidence boundaries.
- Front matter uses Roman page numbers and the main matter resets to Arabic, so the build no longer emits duplicate PDF destinations.

## Reader-first compression of 29 August 2026

All six chapters were edited for a reader with minimal prior knowledge. The pass
replaced reader-facing development terminology with training-period and
testing-period language, removed repeated motivation and separated paragraphs
that carried several ideas. Technical detail now appears once, in its primary
section. The introduction and conclusion retain the explanations needed to
interpret timing, testing-period uncertainty and trading costs. No equation,
estimate, interval, correction, cost assumption, hypothesis rule or evidence
boundary changed in this pass.

## Administrative checks still owned by the author

1. Confirm that the restricted-distribution sentence on the title page is the intended UCL option.
2. Replace the reconstructed project summary in Appendix C with the original approved form. A 15 August filesystem search of the source archive and a narrow connected-Outlook check did not recover it; this is a submission blocker rather than an optional tidy-up.
3. Obtain written confirmation that the programme's assessment category permits the disclosed OpenAI Codex assistance. No programme-specific category was found. The concise acknowledgement does not itself establish permission; see `docs/submission_gates.md`.
4. Establish the Reuters/LSEG licence, data-processing and ethics basis for the completed external prompt processing. Safeguards and aggregate-only publication do not themselves establish authority.
5. Read the full submission version and be prepared to explain every method, result and citation in a viva or supervisor review.

## Corrective proofread of 21 August 2026

The proofread corrected three prompt-reporting errors without changing an
estimate or gate outcome. The manuscript now states that no prompt passed the
complete five-part gate, rather than claiming that every prompt failed every
component. It also records that three variants beat the same-date FinBERT net
mean while remaining loss-making, and distinguishes the one complete reproduced
request from the five retained task paragraphs. Redundant prose was cut to bring
the six chapters to 10,986 words, inside the stated planning tolerance.
The rebuilt A4 PDF has 67 pages, 1,356,290 bytes and SHA-256
`6cc14739a7888fb881ae4783ef18ddaf828866976bedbeec6717912445260318`.

## Final proofread of 19 August 2026

No notebook or model was rerun, and no estimate, interval, correction, cost,
placebo, MDE or gate verdict changed.

1. **Bounded the remaining categorical claims.** Marginal class and singleton
   shifts no longer purport to rule out composition; recent-return controls do
   not rule out reversal generally; the period-specific downside result is not
   called genuine timing; and the LSEG evidence fails to establish portability
   rather than disproving it.
2. **Kept the established voice.** The revision uses short, declarative
   paragraphs, first-person plural where the author acts, and the same
   estimate--uncertainty--boundary sequence as the surrounding manuscript.
   Repeated exposition was cut rather than replaced with new argument.
3. **Closed the page-flow defects.** The contents depth was reduced by one level,
   the closing assessment was integrated into further work, the disclosure was
   tightened and the prompt-selection table was moved after the full discussion.
   The final 65-page render has no one-line spill pages or float inside a sentence.

## Author-voice revision of 20 August 2026

The abstract and all six numbered chapters were revised against three earlier
reports bearing the author's name. The profile favoured direct topic sentences,
short paragraphs, ordinary reporting verbs, numerical results before
interpretation and procedural method explanations. The pass removed generic
signposting, over-balanced contrasts, slogan-like closing lines and repeated
section templates. It did not add deliberate errors, informal filler or
detector-targeted perturbations.

No estimate, interval, correction, cost, placebo, MDE, hypothesis or gate verdict
changed. The existing artificial-intelligence disclosure remains in Appendix A.
The manuscript was not uploaded to GPTZero or another external detector. That
revision's validation was 74 tests, 30 reproducible artefacts and a 65-page A4
PDF with SHA-256
`7e4f511faa29bff1981badb3712f1992d9084462312d1b6b3dd49c9f3b04fd63`.

## External-review remediation of 16 August 2026

The external panel's major-revision decision was driven by one joined issue:
the paper interpreted an assigned-window/close-to-close asymmetry as
specificity while acknowledging that some stories could arrive after the
assigned opening price. Notebooks 86 and 87 now test the available return legs
without reopening sealed Notebook 75.

1. **Decomposed the outcome.** The training-period association is intraday
   (`-0.01029`, BH q=`0.0011`) and absent post-close (`+0.00155`, q=`0.595`).
   Previous-close-to-assigned-open is negative, and the separate
   previous-open-to-assigned-open probe is strongly negative (`-0.01926`,
   p=`7.87e-09`). The manuscript withdraws predictive-window specificity and
   treats timing as unidentified.
2. **Preserved the era result without over-explaining it.** The full baseline
   contrast remains `+0.01414` (`p=0.0173`); the intraday contrast is
   `+0.01366` (`p=0.0204`, BH q=`0.0613`). The testing period is more singleton-heavy,
   not less, so a pure threshold-composition account does not explain the sign
   reversal. Timing, coverage, mapping and classifier change remain competing
   explanations.
3. **Corrected scale and uncertainty.** The effect translation now uses the
   attainable tied-rank spread of 0.56056 (`-0.512` percentile points).
   Prospective power remains 35.4%, while realised, ex-post power is 26.5%
   because the testing-period standard error is 15.8% larger than projected.
4. **Reported the adverse economic result.** FNSPID overlay maximum drawdown is
   `-16.36%`, worse than the HAR base's `-15.73%`. Forty-four episode deletions
   and a February--April 2020 exclusion preserve the narrow downside/return
   timing statistic but do not repair the failed economic gate.
5. **Added requested robustness and measurement context.** Trailing-beta
   outcomes, beta controls, HAC lags 0--42, autocorrelation and Ljung--Box
   diagnostics, class rates, zero-share rates, singleton shares and average
   names per session are now reported. The absence of a human FinBERT audit is
   explicit.
6. **Rechecked the literature.** All 39 retained records and citation contexts
   were reverified. Cakici et al. (2025) is added because its published
   replication identifies look-ahead in Farmer et al.'s implemented kernel and
   materially qualifies the original pockets-of-predictability result.

The complete point-by-point response is in
`reviews/external_review_remediation_20260816.md`; the integrity audit is in
`reviews/academic_integrity_verification_20260816.md`.

## Revision of 13 August 2026

A second review pass made the following changes. No model was re-estimated; every result added was already present in the committed aggregate evidence.

1. **Reported the Notebook 74 robustness set.** The evidence map lists six diagnostic tables for that notebook, but the manuscript cited only two of them. `sec:robustness-diagnostics` in the results chapter now reports the story-count strata (including the `n >= 3` row, where the coefficient falls to -0.00183 with p = 0.766), the Newey-West lag sensitivity, and, for the backward LSEG block, the episode attribution, leave-one-episode-out and bootstrap block-length checks. The story-count evidence weakens the headline claim and is carried into the abstract and the conclusion's limits.
2. **Fixed a turnover/cost ambiguity.** The net-return equation summed the full absolute weight change while the results chapter reported one-way turnover of 0.712. A reader combining the two recovered -6.2 rather than the reported -13.34 basis points. Turnover was defined explicitly as half the traded notional, the cost was written as `2c x TO`, and the break-even arithmetic was shown inline. The contemporaneous claim that the underlying code was already correct was overturned by the 15 August accounting audit below.
3. **Restated H3 and H4 as refutable propositions.** Both were previously written as normative standards ("must cover its turnover", "should be examined separately"), which the results chapter then reported as rejected or failed. They now state outcomes the evidence can contradict.
4. **Included the orphaned MDE table.** `tables/tab_null_mde.tex` was generated by Notebook 77 and quoted in prose but never `\input` anywhere in the document. It is now included and cross-referenced. Its column widths were adjusted to fit the A4 text block; no value changed.
5. **Added two generated tables.** `write_aggregation_table` and `write_robustness_table` in the artifact script now emit the exact nine-rule family results and the sensitivity panel from the committed CSVs, with a drift check on the `n >= 3` diagnostic. Previously the nine-rule numbers existed only in a figure.
6. **Cross-referenced every float.** Four figures and three tables were previously not called out anywhere in the text.
7. **Separated panel-row counts from regression sample sizes.** Notebook 74's story-count file records panel rows before complete-case outcome filtering. Four full-panel rows have no return, so the robustness table now labels these as panel rows and the prose reports the `n >= 3` stratum as roughly 137,000 rows rather than presenting 137,142 as the regression sample. The nine-rule table also reports the actual range of 512,145--512,149 usable observations across estimators.

The Notebook 74 robustness reporting was retained. That intermediate revision
contained about 10,402 main-chapter words; later Notebook 75 and 85 additions
increased the count before the current compression recorded above.

## Revision of 14 August 2026 after fresh peer review

No model or notebook was re-executed. The revision changed claim alignment,
literature positioning and presentation while preserving every promoted result.

1. **Aligned the central estimands.** The abstract, introduction, methodology,
   results and conclusion now separate the training-period full-control result from
   the baseline coefficient repeated in Notebook 75. The era contrast is shown
   as -0.00831 versus +0.00583; -0.00914 is reported separately as training-period
   robustness.
2. **Bounded the non-replication claim.** The data and conclusion chapters state
   that the direct contrast establishes pipeline-level instability but does not
   distinguish market change from firm/news composition, timing, mapping or
   classifier behaviour. No unrecorded common-universe analysis is implied.
3. **Added predictor-stability context.** Welch--Goyal (2008),
   McLean--Pontiff (2016) and Farmer--Schmidt--Timmermann (2023) now position
   temporal transport within empirical finance. Their metadata, DOIs and
   publisher abstracts were verified and added to the literature records.
4. **Corrected hypothesis and presentation language.** H1b is “not supported”
   rather than “rejected”; the final hypothesis summary names H1a explicitly;
   and Figure 5.5 uses “99.1 percentile” rather than “99.1th”.
5. **Reduced the submission-risk count.** Secondary risk, LSEG and prompt prose
   was compressed from 12,107 to 10,872 main-chapter words without deleting the
   frozen replication, uncertainty, multiplicity, cost, power or matched-control
   evidence.

## Turnover-accounting repair of 15 August 2026

A code-to-manuscript audit found that the portfolio helpers still compared each
target with the prior target rather than the prior return-drifted book. Notebook
03 also omitted terminal liquidation despite the manuscript saying it was
charged. The shared helpers, focused tests, frozen amendment and Notebooks 03,
10, 71 and 77 were updated and replayed from authorised local inputs in an
isolated workspace. The negative-share annualised one-way turnover moved from
179.33 to 179.48, the net mean from -13.34 to -13.35 basis points and the
break-even cost from 0.62533 to 0.62482 basis points per side. Statistical
coefficients, multiplicity decisions and the substantive economic null did not
change.

The label-shuffled threshold control exposed a genuine leverage boundary: its
26 January 2021 book ends the interval with pre-cost wealth growth -0.511.
Drift-normalised turnover is undefined after insolvency, so the aggregate result
now reports the arm as `INSOLVENT` with cost-dependent metrics missing rather
than capping the return or reverting to target-to-target turnover.

## Figure revision of 13 August 2026

The figure programme was rebuilt so that ten plots are generated from committed
aggregate evidence by one script, in one visual system. The HAR exposure/equity
figure is the documented exception: it is regenerated separately from private
Notebook 24 daily paths because those row-level paths cannot be committed.

1. **Three notebook PNGs were replaced by reproducible equivalents.** `fig_aggregation_ic`
   carried a raw column identifier (`ar_open_h1`) in its title, 45-degree rotated tick
   labels and no visible emphasis on the one rule that survives correction.
   `fig_conditional_double_sort` and `fig_downside_placebo` used matplotlib default
   colours rather than the validated house palette; the schedule-shift figure spent three
   different hues on three panels whose identity was already given by their titles.
   Their replacements are `fig_aggregation_family`, `fig_double_sort` and
   `fig_timing_placebo`. The schedule-shift figure is now a null-versus-observed comparison
   rather than three histograms, because the full circular-shift distributions live in
   uncommitted parquet and cannot be rebuilt from the evidence snapshot.
2. **Three new figures were added** for evidence that existed only as prose or tables:
   `fig_estimate_stability` (price-path controls beside story-count strata on one shared
   scale), `fig_risk_regime_stability` (overlay-minus-base return by period) and
   `fig_prompt_gate` (gross-to-net returns for all six prompt variants).
3. **Formatting fixes across all nine.** Hardcoded axis limits in the three existing
   generated figures were made data-driven so a drifting number cannot push a label off
   canvas; gridlines were moved behind bars and markers; legend positions were moved off
   colliding labels; literal `--` sequences were replaced with en-dashes; and value
   labels were aligned into one column rather than tracking each interval's right end.
4. **Figure text was sized for print.** Figures are authored wide so annotations fit and
   are then scaled to the A4 text block, which shrinks every label. A `dissertation_style`
   helper raises the base sizes so labels land near 8pt beside 12pt body text.

The house categorical palette was checked with the six-check validator before use: it
passes the lightness band, chroma floor, adjacent-pair CVD separation and normal-vision
floor, with a contrast warning on three slots that the direct-labelling convention
already relieves.

## Currency labelling correction

The LSEG economic decomposition was reported in sterling. Tracing it showed no FX
conversion anywhere in the repository: the value is computed as a return difference
multiplied by a notional of 1,000,000, and the label came from the frozen spec's
wording, "ending wealth and loss-day/up-day decomposition per GBP 1 million".

The underlying cash flows are open-to-open returns on 33 US-listed equities, so they
are USD. A sterling label implied a GBP-funded investor carrying GBP/USD exposure that
the backtest does not model — over the 167-session testing period that exposure could
easily exceed the 30,870-per-million ending-wealth gap being reported, which is only
3.1% of notional.

The manuscript, figure axis, derived metrics and provenance row now report USD. No
number changed, because no conversion had ever been applied. The saved
`gbp_per_million` column name is preserved so the aggregate snapshot still matches the
frozen specification, and the deviation is recorded in Appendix A and the evidence map.

## Post-hoc measurement sensitivities of 13 August 2026

Notebook 85 adds one declared post-hoc family: close-to-close hard share, a trailing
FF3 residual of that close-to-close return, and next-open soft negative mass. The
family was frozen after the headline next-open result. None of the three coefficients
survives Benjamini--Hochberg correction at $q=0.05$. The predeclared next-open
hard-share coefficient is reported beside them and is unchanged. These tests do not
join the original nine-rule family or the Notebook 73 ledger of 56 families, and they
do not upgrade H1. Soft mass was computed without a human label audit; the audit
remains the step that would interpret why hard and soft agree or differ.


## Typographic pass of 13 August 2026

1. **Numeric tables now use `siunitx` columns.** Every negative number in a table was set with a
   text-mode hyphen, so a table reading `-0.00831` sat on the same page as prose reading
   $-0.00831$ with a true minus. Numeric columns are now `S` columns: real minus signs,
   alignment on the decimal point, and digit grouping handled by the package rather than
   by hand-written commas. Confidence intervals are set in maths mode for the same reason.
2. **`p{}` columns were set ragged-right.** Justified fixed-width columns were producing
   stretched inter-word spacing and mid-word hyphenation (`coeffi-cient`) in the headline,
   MDE and appendix provenance tables.
3. **Captions were configured.** `caption` was loaded but never set up. Labels are now
   bold with a hanging indent, one size below body text.
4. **A constant column was removed.** `tab_reversal_confound` carried a `Sessions` column
   reading 2,264 on every row; that fact moved to the caption and the table stopped
   overrunning the text block.
5. **Dead preamble removed.** `fancyhdr` was loaded but never configured, and the
   `\FNSPID`, `\LSEG`, `\FinBERT`, `\bps` and `\E` macros were defined but never used.
6. **Widow and orphan penalties set**, and paragraphs that grammatically continue the
   sentence introducing a display equation no longer indent as if they began a new one.

Underfull horizontal boxes fell from 95 to 6, and the document still compiles with no
overfull boxes, no warnings and no undefined references.

## Multiplicity record removed

The appendix section reproducing the 56-family ledger and the post-ledger family table
was cut at the author's direction, together with `tab_multiplicity_families.tex` and
`tab_post_ledger_families.tex`. The substantive claims are unchanged and still appear in
\cref{ch:results}: 56 families and 264 tests or decisions through Notebook 73, of which
27 are within-family BH survivors and 36 families are all-null. Those totals remain
sourced from `experiments/results/78_null_results_and_provenance_tables/`, and the
manuscript now points a reader to the repository ledger rather than to an appendix table.

## Figure taste pass of 13 August 2026

The figures were carrying three layers of labelling: an in-image title, an in-image
source line and a LaTeX caption. In a bound document the caption is the single place a
reader looks, so the in-image titles and source lines were removed and their content
folded into the captions, which now also name the notebooks each figure draws on. That
returned roughly a fifth of each canvas to the data.

Two figures were redesigned rather than adjusted:

- **Cross-source coefficients.** The MDE panel was two bars — not a chart — on a second
  axis that competed with the estimates and clipped its own x-label, while a footnote
  overlapped the left panel's label. It is now a single forest plot of the four
  estimates, with each LSEG block's power written beside the interval it explains.
- **Break-even costs.** The charged-cost line sat at the extreme right with two-thirds of
  the panel empty, and its annotation collided with the title. The shortfall between the
  best break-even and the charged cost is now a shaded band, so the gap is an object
  rather than absence, and the multiple is stated once.

Chrome was reduced across the set: no left spine or tick marks on categorical axes, grid
under the marks and along the value axis only, value labels in one aligned column outside
the plot, and quiet grey panel labels where a figure has two panels. Axis labels were
shortened once the captions carried the full description, which also fixed a clipped
label on the double-sort figure.
