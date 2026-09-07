# Editorial Decision and Revision Roadmap

> **Status update, 14 August 2026.** This review predates the authorised
> one-shot opening of Notebook 75. The frozen 2020--2023 result now fails both
> declared primary families, and the predeclared testing-minus-training
> contrast supports temporal instability. The manuscript has therefore been
> reframed as a temporal non-replication paper. The recommendations below remain
> a historical review record; suggestions to keep Notebook 75 sealed or add
> further evaluation tests are superseded.

Simulated five-seat peer review, 14 August 2026. Produced by the
`academic-paper-reviewer` skill (v1.10.0) in `full` mode. This is a separate
document: no reviewer modified the manuscript.

**Criteria binding:** `criteria_binding_unavailable` for venue. There is no journal
target. The panel was bound to the UCL MSc Computational Finance marking rubric
in `docs/ucl_requirements.md`. No journal-fit claim is made anywhere below.

---

## Conflict of interest — read this first

The synthesiser (Claude) edited this manuscript earlier in the same session:
the `sec:robustness-diagnostics` section, the nine-figure programme, the
turnover/cost definition, the GBP→USD relabelling, and the typographic pass.
Several findings below land on that work. Where they do, it is marked
**[synthesiser-authored]**. Treat the adjudication of those specific items as
the least reliable part of this document, and note that one reviewer's praise
(Methodology S2, verifying the turnover convention as correct) is praise for
synthesiser-authored text. A 15 August code-to-manuscript audit subsequently
disproved that verification: the implementation used prior target weights and
Notebook 03 omitted terminal liquidation. The repaired evidence is governed by
`experiments/specs/portfolio_turnover_accounting_amendment_v1_20260815.json`.

---

## Decision

**MAJOR REVISION.**

| Seat | Recommendation | Criticals |
|---|---|---|
| Examiner-fit (`EIC`) | Minor Revision — weighted 78.5, Distinction band | 0 |
| Peer Reviewer 1 (Methodology) | Major Revision — 73.4 | 0 (explicitly declined to manufacture one) |
| Peer Reviewer 2 (Domain) | Major Revision | 0 |
| Peer Reviewer 3 (Practitioner) | Major Revision | 0 |
| Devil's Advocate | — | **2** |

Three of four scoring seats recommend Major Revision. Two Devil's Advocate
CRITICAL findings are adjudicated below; under Checkpoint Rule #4 neither may be
silently bypassed.

**The decision is not a judgement that the work is weak.** All five seats
independently credited the negative-result discipline, and the Practitioner
seat stated plainly that the paper's central economic conclusion is *correct
and if anything understated*. Every required repair is either a rewrite or a
cheap test on data already held. No new data collection is implied.

---

## Adjudication of Devil's Advocate CRITICALs

### DA-C1 — Outcome-window specificity — **GENUINELY UNRESOLVED. Blocks Accept.**

The Notebook 85 sensitivities show the association exists in exactly one
outcome window:

| Outcome | Estimate | *p* |
|---|---|---|
| Open-to-open (headline) | −0.00831 | 0.005 |
| Close-to-close | +0.00027 | 0.929 |
| FF3 residual | +0.00182 | 0.543 |

*Verified by the synthesiser against
`experiments/results/85_fnspid_development_outcome_and_label_sensitivities/sensitivity_coefficients.csv`.*

The DA's inference: open-to-open = intraday(*t*) + overnight(*t*→*t*+1);
close-to-close = overnight(*t*→*t*+1) + intraday(*t*+1). The two windows share
the overnight leg. Since close-to-close is flat, the −0.00831 must load almost
entirely on intraday(*t*) — the session to which the news is assigned. That is
legitimate if the news is genuinely pre-open, which the date-only rule
guarantees; but Chapter 3 specifies the timestamped branch only as "more
precise timestamps retain the checkpoint's earlier mapping", concedes the
checkpoint holds no verifiable availability timestamp at row grain, and never
reports the branch share.

**Adjudication:** the arithmetic is confirmed; the leakage inference is
plausible but **not established**. The paper has not run the test that would
settle it. Until it does, "next-session information" is asserted rather than
demonstrated. This is the single highest-priority item in the roadmap (R1).

### DA-C2 — The distributional claim is not supported — **VALIDATED.**

Independently corroborated by four of five seats: Examiner W1, Methodology W6,
Domain W2/W4/W12, and the DA. Averaging is inoperative on 48.3% of the panel
(247,470 of 512,153 rows are single-story — *synthesiser-verified*), the
training-period median is two stories, and the coefficient is 22% of full size and
insignificant at *n*≥3 (−0.00183, *p*=0.766). The one reading the data leave
open — hard label beats soft score — is itself unsupported: soft negative mass
gives −0.00378, *p*=0.112, *q*=0.337.

**Adjudication:** validated. This blocks Accept and drives R2. It is a
**framing** defect, not a data defect: the concession is already in the
manuscript (§5.3, §6.1) but has not been allowed to reshape the title, abstract
or Chapter 1. **[synthesiser-authored — §5.3 and the abstract clause are mine;
I added the concession but did not fix the framing it contradicts.]**

---

## Consensus findings (independent corroboration)

Reviewers were blind to each other. Convergence is therefore signal.

**1. Framing outruns evidence — 4 of 5 seats.** Examiner W1, Methodology W6,
Domain W2/W4/W12, DA C2. Unanimous on the diagnosis and on the remedy being a
rewrite rather than re-analysis.

**2. Available evidence exists but is not reported — 3 seats, four instances.**
- Methodology W4: Notebook 13 ran "two scorers × nine pre-existing aggregators"
  on LSEG — **18 tests, 0 BH survivors, all-null** — plus an 8-test metadata
  family, also all-null. This is the closest existing out-of-sample re-run of
  the paper's own aggregation family, and Chapter 5 attributes LSEG
  non-detection purely to low power. *Synthesiser-verified in
  `78_null_results_and_provenance_tables/multiplicity_family_results.csv`.*
- Domain W8: the checkpoint carries a **recap flag**, used only as a tie-break
  inside rule 7. A recap-conditioned negative share is computable now, which
  weakens §6.4.1's claim that "FNSPID would require rebuilding the checkpoint".
  *Synthesiser-verified: flag present per §3.1; sole use is rule 7.*
- Methodology W3 / DA M3: the "56 families / 264 tests" ledger spans notebooks
  3–73 only, so the focal families (71, 74, 76, 79, 85) sit outside the figure
  offered as multiplicity discipline. *Synthesiser-verified: `first_notebook=3,
  last_notebook=73`.*
- DA M2: the *n*=1 stratum coefficient — the single number that would settle
  C2 — is never reported, though the ≥2 and ≥3 strata are.

**3. A cheap decisive test is missing — all 5 seats, four different tests.**
DA C1 (window decomposition), Methodology W5 (story-count interaction on the
full sample, rather than comparing two imprecise subsamples), Domain W3
(negative-class dummy horse race), Practitioner W1 (price-based volatility
control under identical hysteresis). Each is hours of work on existing data and
each would close a question the manuscript currently leaves open.

**4. Evidence concentration — 2 seats.** DA M1: the nine yearly coefficients
average to the pooled estimate; dropping 2014 and 2016, the only individually
significant years, leaves seven years averaging −0.005405 with implied SE
0.0033, **t ≈ −1.63** — not significant. *Synthesiser-recomputed.* Practitioner
W2: in backward LSEG, the first z-score is 2024-07-03 and the largest downside
episode begins 2024-07-18, when the rolling normaliser sits near its
126-observation seeding minimum. *Synthesiser-verified.*

---

## Disagreement, and its arbitration

**Examiner-fit says Minor Revision / Distinction 78.5; three seats say Major.**

This is a criteria difference, not a factual one. The Examiner scored against
the UCL rubric, where Validation carries 20% and this dissertation is genuinely
top-band on it (chronological boundary, HAC lag sweep, story-count strata,
MDEs, exposure-matched comparators, leave-one-out at two levels, circular-shift
placebos, a full provenance ledger). The other three scored against
research-contribution norms, where the framing/evidence gap and the unreported
null families weigh heavier.

**Arbitration:** both are correct within their own binding. For the *degree
assessment* the Examiner's reading governs and the work sits comfortably in the
Distinction band. For the *scientific claim* the majority governs: the
manuscript should not go forward asserting a distributional finding. The
roadmap below serves both — every item is a correction or a cheap test, and
none puts the classification at risk.

**A blocking compliance item sits outside this disagreement.** Examiner W5: the
title-page **distribution statement is absent**. It was present in `main.tex`
at the start of the 13 August session and was lost in the title-page redesign;
`grep -i distribut manuscript/main.tex` now returns nothing.
*Synthesiser-verified.* This is a required front-matter element.

---

## Revision Roadmap

Ordered by the panel's evidence, not by effort. No item requires new data.

### R1 — Settle the outcome-window question (DA-C1)
Run two regressions on the existing panel: open(*t*)→close(*t*), and
close(*t*)→open(*t*+1). Report both with the same controls. Additionally report
the share of FNSPID rows assigned by the timestamped branch versus the
date-only branch, and state the timestamped rule explicitly in §3.1. If the
association sits in intraday(*t*) and a material share of rows is timestamped,
the "next-session" characterisation must be withdrawn or heavily qualified.
**This is the only finding that could invalidate the headline.**

### R2 — Bring the framing down to what survives (DA-C2, consensus 1)
Retitle away from "Beyond the Mean". Rewrite the Chapter 1 motivating example
around one- and two-story days. State the 48.3% single-story share in §3.1 with
the full story-count distribution (n=1, 2, 3–5, >5). Narrow the abstract and
the Chapter 6 closing sentence ("a firm-day average is not a sufficient
description of its news flow" asserts a property of *flow* the evidence does
not reach). The supportable claim: *a hard negative-class flag retains weak
conditional information that a signed mean discards, concentrated in
sparse-coverage firm-days.*

### R3 — Report the evidence already in hand (consensus 2)
Add the Notebook 13 18-arm all-null LSEG family to §5.8 with its MDE, or state
in one sentence why the LSEG-44 firm set makes it uninformative for LSEG-33.
Report the *n*=1 stratum coefficient. Report the story-type interaction
coefficients, not only their MDEs — they exist in
`77_null_minimum_detectable_effects/`. Relabel the 56-family figure as
"families closed before the focal analysis" wherever it appears.

### R4 — Run the four cheap tests (consensus 3)
(a) Story-count interaction on the full sample — settles whether the decay is
attenuation or the arithmetic of halving twice. (b) Negative-class dummy and
signed-arm decomposition — separates "distribution shape survives the mean"
from "the negative arm survives", the latter being known negativity asymmetry.
(c) Threshold sensitivity: define the negative class by *p*(neg) > τ for
τ ∈ {0.4, 0.5, 0.6} instead of argmax — the cheapest available evidence that
the finding is not an artefact of a PhraseBank-calibrated boundary.
(d) Price-based volatility z-score under identical 1.5/0.5 hysteresis — if news
pressure does not beat it, the prospective risk test in §6.4.5 should be
withdrawn rather than recommended.

### R5 — Report the yearly and episode concentration (consensus 4)
State that removing 2014 and 2016 leaves the pooled estimate indistinguishable
from zero. For backward LSEG, report the observation count backing the z-score
at each episode, or restrict to sessions with a full 252-observation
normaliser.

### R6 — Compliance and specification
Restore the title-page distribution statement (**blocks submission**). Specify
the HAR model fully — estimator, windows, lags, retransformation, one
forecast-accuracy diagnostic (Examiner W3; largest available gain on the
20%-weighted lecture-topics criterion). Annotate the provenance ledger to
explain the `none` frozen-spec rows, including Notebook 03, the family that
selected the dissertation's subject.

### R7 — Literature (Domain W5/W6/W9)
Add investor attention (Da/Engelberg/Gao 2011; Barber/Odean 2008) and
differences of opinion (Diether/Malloy/Scherbina 2002; Miller 1977) — both
supply competing readings of the *same sign* through non-information channels,
and the paper invokes "attention" by name in a rule title and in its own
preferred alternative reading with zero grounding. Add the supervised
return-trained branch (Ke/Kelly/Xiu, SESTM) which the gap claim currently
sidesteps. State the measurement unit of `isakin2023` and `chen2024`, on which
the distinguishing sentence depends. Repair the `politis1994` attribution
(fixed circular blocks → Künsch 1989). **Verify every venue before citing —
the Domain reviewer explicitly flagged Ke/Kelly/Xiu and Bybee et al. as
venue-unconfirmed and declined to name a RavenPack paper it could not
substantiate.**

### R8 — Practitioner hardening (strengthens the existing conclusion)
Quantify borrow on the news-selected short leg (plausibly 0.20–1.20 bps/day
against 0.890 bps gross — independently decisive, no cost assumption needed).
State that horizon extension is *closed* by the close-to-close null and
event-only entry is *closed* by the story-count decay, converting "we assumed
10 bps and it failed" into "no implementation of this signal pays for itself".
Note the §6.4.4 tension: proposing to trade "first releases judged material"
selects the subsample where the signal vanishes. Add the ~226 names/session
figure and the 0.695 pre-cost gross Sharpe. Raise the firm-level rule's 2 bps
ETF-grade cost to a single-name range.

### R9 — Minor
Split H4 into H4a (coefficient, inconclusive on power) and H4b (economic gate,
refuted) and carry both verdicts to the abstract. Rename the LSEG
"backward"/"recent" blocks, which invert chronological intuition. Delete or
reorder `manuscript/figures/` — `\graphicspath` searches 18 stale PNGs *before*
`artifacts/`, so a future name collision would silently substitute an obsolete
figure. **[synthesiser-authored — I left those files as "archive"; the hazard
is mine.]** Normalise `tab_outcome_label_sensitivities` to the siunitx column
style used by the other Chapter 5 tables. Trim the 361-word abstract.

---

## What the panel affirmed

Recorded because the decision is Major Revision and the reasons for it should
not eclipse this. All five seats, independently:

- The negative results are reported as findings, not buried — break-even costs
  across all nine rules, exposure-matched controls, circular-shift diagnostics,
  leave-one-episode-out and leave-one-company-out, a to-the-dollar wealth
  decomposition, an honoured prompt stopping rule that cost the author a
  chapter, and a recorded model-spend ledger.
- Every headline number the Methodology seat spot-checked (~25 of them)
  reproduced exactly from committed aggregate CSVs, with internally consistent
  derived arithmetic.
- The exposure-matched comparator — the Practitioner's single strongest
  compliment — is the control that kills the author's own result, and most
  backtests never build it.
- Power analysis keeps nulls interpretable in both directions: the
  learned-threshold null is *informative* (MDE 6.9 bps below observed losses of
  7.71–16.88), the LSEG transfer null is *underpowered*, and the paper does not
  conflate them.
- The Devil's Advocate **retracted two of its six assigned attack lines** after
  inspection — the "salvage narrative" and "not-pooled defence" charges — and
  logged them as observations rather than defects.

The Practitioner's closing assessment is worth preserving verbatim in spirit:
as an allocator there is nothing here to allocate to, and the paper says so
without hedging; as a hiring manager, constructing the comparator that kills
your own result is the habit that separates a researcher who saves a desk money
from one who costs it money.

---

## Revision log — 14 August 2026

Applied from this roadmap. Items requiring row-level panel data are **not** done
and remain with the author.

| Item | Status |
|---|---|
| R6 title-page distribution statement | **Done** — restored to `main.tex` |
| R9 `\graphicspath` hazard | **Done** — `artifacts/` now searched before `figures/` |
| R2 reframing | **Done** — abstract opening, Ch1 motivating example, Ch6 closing sentence, and the story-count distribution added to §3.1 (48.3% / 24.9% / 26.8%) |
| R5 yearly concentration | **Done** — §5.3 now states that removing 2014 and 2016 leaves *t* ≈ −1.6 |
| R3 unreported evidence | **Partly done** — the 18-arm and 8-test all-null LSEG families are now reported in §5.8 with the firm-set caveat; the 56-family ledger is relabelled "closed before the focal analysis" in §4.6 and §5.11. The *n*=1 stratum coefficient is **not** added: it does not exist in the committed aggregates |
| R9 H4 split | **Done** — H4a/H4b in Ch1, split verdict in §5.11 |
| R9 LSEG block naming | **Done** — clarifying sentence at first use in §3.3, rather than a global rename |
| R9 table styling | **Done** — `tab_outcome_label_sensitivities` converted to siunitx |
| **R1 window decomposition** | **NOT DONE — author** |
| **R4 four cheap tests** | **NOT DONE — author** |
| R6 HAR specification | **NOT DONE — author** (needs the notebook's estimator/window detail) |
| R7 literature | **NOT DONE — author** (see caution below) |
| R8 practitioner hardening | **NOT DONE — author** (borrow range is a substantive claim to own) |

**Why R1 and R4 were not attempted.** Both need the row-level FNSPID panel,
which is deliberately excluded from this repository per `AGENTS.md`. Only
aggregate outputs are committed, so the window decomposition, the story-count
interaction, the negative-class horse race, the argmax-threshold sensitivity and
the price-volatility control all have to be run where the panel lives.

**Caution on R7.** Do not paste the reviewer's reference list in. The Domain
seat marked Ke/Kelly/Xiu and Bybee et al. as venue-unconfirmed and declined to
name a RavenPack paper it could not substantiate. Verify every entry against
primary metadata before it enters `references.bib` — the existing bibliography
is clean and should stay that way.

Main-chapter word count rose from 10,456 to 11,172 (about 11.7% above the
10,000 target). `docs/word_count.md` records the trimming order.


---

# Round 2 — 14 August 2026

Two seats (Devil's Advocate, Methodology) re-ran blind on the revised manuscript,
briefed to attack the ~700 words of new prose hardest. **Both found errors in the
Round 1 revisions themselves.** Wave 2 (Examiner-fit, Domain, Practitioner) was
held pending this result.

## Errors introduced by the Round 1 revision — corrected

### E1. The yearly-concentration passage was an invalid inference. **Retracted.**

The revision added: *"removing 2014 and 2016 leaves seven years averaging −0.00541
with an implied standard error of 0.0033, a t-statistic of about −1.6 ... not a
steady annual regularity so much as a pooled average over a period containing two
strong years."* Three independent defects, all verified:

1. **Selection on extremes.** 2014 and 2016 were removed *because* they were the two
   most negative and the only significant years. Removing the k most extreme
   observations mechanically shrinks the mean and inflates the p-value, so t = −1.6
   is not a test of stability.
2. **The artifact says the opposite.** The sample deviation of the nine yearly
   estimates is **0.00703**, against a mean within-year standard error of
   **0.00875**. Dispersion is *below* sampling noise, so there is no evidence of
   between-year heterogeneity at all; a Cochran Q would not reject homogeneity.
   Per-year power at the pooled effect size is ≈ 0.16, so ≈ 1.4 significant years
   out of nine is expected under a perfectly constant effect. Observing two is
   unremarkable.
3. **The standard error was undocumented and construction-dependent.** Propagating
   within-year SEs gives 0.00333 (t = −1.63); the between-year dispersion gives
   0.00164 (**t = −3.29**). The two constructions give opposite verdicts and the
   manuscript defined neither.

The passage now reports what the data support: individual years are underpowered,
the yearly series shows no evidence the coefficient varies, and the pooled estimate
does not rest on any particular year. **This reverses the direction of the Round 1
edit, which over-corrected against the author's own result.**

### E2. The ledger relabelling was false. **Corrected.**

Round 1 relabelled the 56-family ledger as *"closed before the focal analysis ...
audits the exploratory work that preceded the headline rather than the headline
itself."* The ledger contains Notebooks 45, 72 and 73, and the Notebook 45 and 73
families are exactly the *q* = 0.0301, *q* = 0.0336 and *q* = 0.0482 results
reported as findings in the same chapter. The relabelling made a false claim more
explicit. The text now states both boundaries: Notebook 71's focal families are
excluded (so no family-level correction reaches the headline), while the
downside-alignment and backward-LSEG survivors sit *inside* the audited 264-test
search.

### E3–E5. Three smaller defects introduced or left by the revision. **Corrected.**
- The 8-test all-null family tests the **mean continuous** aggregator, not negative
  share; described accurately now.
- §5.3's lead-in still announced "two diagnostics" after a third was added.
- The abstract attached 48.3% to a paragraph naming the 715,546-row panel; it is a
  training-period figure and now says so.

## New findings — author-owned, not yet actioned

**N1 (DA C2). The LSEG transfer arm sits entirely in the regime where the paper's
own evidence predicts nothing.** `71_.../population_audit.csv` gives median story
count **2.0** for the FNSPID training period against **61.0** and **90.0** for the LSEG
blocks. §5.3 reports the coefficient is −0.00183 (*p* = 0.766) at *n* ≥ 3. The LSEG
blocks are therefore deep inside the dense-coverage regime already shown null, yet
Chapter 5 and Chapter 6 attribute non-transfer solely to power, and recommend
collecting a *larger* sample on that basis. If regime rather than power is the
binding constraint, that recommendation is wrong. *Synthesiser-verified.* This is
the most consequential new finding of Round 2 and needs a manuscript response.

**N2 (DA M3). The headline −0.00914 has no robustness evidence of its own.** Every
diagnostic — story-count strata, HAC lag sweep, yearly series, outcome-window
sensitivities — is run on the **base** −0.00831 specification. The abstract and
Chapter 6 lead with −0.00914. Every caveat attached to the headline is imported
from a different specification. *Synthesiser-verified across four artifacts.*

**N3 (Methodology W2/W5, carried).** The story-count decay is still asserted from
nested subsamples rather than tested. One pooled regression with a
negative-share × 1{*n* ≥ 3} interaction would settle it. Requires row-level data.

**N4 (Methodology W6).** The newly added all-null LSEG families carry no MDE, while
the manuscript insists elsewhere that nulls must be power-bounded. Applying the
standard inconsistently is a coherence defect in its own framework.

**N5 (Methodology W1, carried).** Post-selection inference remains unadjusted. The
reviewer notes a Bonferroni-9 floor leaves H1 nominally significant (0.0049 × 9 =
0.044; 0.00118 × 9 = 0.011) — worth stating, since it shows the core claim survives
a crude correction.

## Round 2 verdict

Methodology: **Major Revision, 71.2** (down from 73.4 in Round 1 — the revision
introduced defects faster than it closed them). Devil's Advocate: **2 CRITICALs**,
one carried (outcome window, still unresolved) and one new (N1).

Both seats independently confirmed that every *arithmetic* claim in the new prose
reconciles exactly to the artifacts, including the four new §3.1 counts, which sum
to 512,153 to the unit. The defects were inferential and structural, not numerical.
