# Academic integrity verification — 16 August 2026

## Verdict and limits

**Repository-level verdict: PASS WITH NOTES.** No fabricated citation,
unsupported numerical result, hidden adverse result, method/result mismatch or
claim-strength inflation was found in the current source. The notes are
external limits, not detected defects: this was not a Turnitin/iThenticate
check; no author-publication corpus was supplied for self-plagiarism screening;
and three submission permissions remain outside the repository.

This audit covers the current LaTeX manuscript, bibliography, committed
aggregate evidence, evidence map, frozen specifications, promoted post-review
diagnostics and validation scripts. It does not certify the legality of the
Reuters/LSEG processing or the programme's permitted GenAI category.

## Seven-mode integrity screen

| Mode | Finding | Basis |
|---|---|---|
| Implementation bug | **Clear in the checked surface** | Return-leg identities reconcile to at most `1.776e-15`; independent reruns of Notebooks 86 and 87 reproduce every promoted CSV byte for byte; focused tests cover timing legs, rolling beta and rank-regression edge cases. |
| Citation hallucination | **Clear** | All 39 bibliography records resolve through 36 DOI records, two arXiv records and one JSTOR record. All keys are cited, no cited key is missing, and every use was checked against the source's substantive result. |
| Hallucinated result | **Clear** | The manuscript generator consumes committed aggregate results, re-derives displayed translations and fails on drift. The timing, power, effect-size and drawdown corrections reconcile to their promoted outputs. |
| Shortcut reliance | **Disclosed, not laundered** | Singleton thresholding, tied ranks, missing story-family identity, unavailable row-level timestamps, classifier validity and source-density differences are explicit limitations. |
| Bug presented as insight | **Clear** | The timing test weakens the earlier interpretation; the manuscript withdraws specificity. The adverse FNSPID drawdown and failed economic gates remain visible. |
| Methodology fabrication | **Clear** | Methods, specifications, runner code, executed notebooks, manifests, result files and provenance rows agree for the post-review studies. Notebook 75 was not rerun. |
| Frame lock | **Clear** | The paper changes its title, question and conclusion when the timing evidence changes. It does not preserve a predictive or deployable-alpha frame. |

## Citation and context audit

The current bibliography contains 39 records and the manuscript contains 49
citation commands. Repository validation checks for missing and uncited keys.
The fresh metadata/context pass produced this ledger:

| Key | Authoritative locator | Result |
|---|---|---|
| `allee2015` | <https://doi.org/10.1111/1475-679X.12072> | Verified |
| `araci2019` | <https://arxiv.org/abs/1908.10063> | Verified |
| `benjamini1995` | <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x> | Verified |
| `bloom1995` | <https://doi.org/10.1177/0193841X9501900504> | Verified |
| `cakici2025` | <https://doi.org/10.1111/jofi.13484> | Verified; added because it directly qualifies the implementation evidence in `farmer2023` |
| `chan2003news` | <https://doi.org/10.1016/S0304-405X(03)00146-6> | Verified |
| `chen2024` | <https://doi.org/10.1016/j.ribaf.2024.102415> | Verified |
| `corsi2009har` | <https://doi.org/10.1093/jjfinec/nbp001> | Verified |
| `dong2024` | <https://doi.org/10.1145/3637528.3671629> | Verified |
| `engelberg2011` | <https://doi.org/10.1111/j.1540-6261.2010.01626.x> | Verified |
| `famafrench1993` | <https://doi.org/10.1016/0304-405X(93)90023-5> | Verified |
| `fang2009` | <https://doi.org/10.1111/j.1540-6261.2009.01493.x> | Verified |
| `farmer2023` | <https://doi.org/10.1111/jofi.13229> | Verified and qualified by the published replication |
| `garcia2013` | <https://doi.org/10.1111/jofi.12027> | Verified |
| `glasserman2023` | <https://arxiv.org/abs/2309.17322> | Verified |
| `harvey2016multiple` | <https://doi.org/10.1093/rfs/hhv059> | Verified |
| `heston2017` | <https://doi.org/10.2469/faj.v73.n3.3> | Verified |
| `huang2023` | <https://doi.org/10.1111/1911-3846.12832> | Verified |
| `isakin2023` | <https://doi.org/10.1016/j.irfa.2023.102761> | Verified |
| `kirtac2024` | <https://doi.org/10.1016/j.frl.2024.105227> | Verified |
| `korajczyk2004` | <https://doi.org/10.1111/j.1540-6261.2004.00656.x> | Verified |
| `kunsch1989` | <https://doi.org/10.1214/aos/1176347265> | Verified; cited for fixed dependent-data blocks, not the stationary bootstrap |
| `lopezlira2026` | <https://doi.org/10.1016/j.jfineco.2026.104335> | Verified; updated to volume 184, article 104335 |
| `loughran2011liability` | <https://doi.org/10.1111/j.1540-6261.2010.01625.x> | Verified |
| `loughran2016` | <https://doi.org/10.1111/1475-679X.12123> | Verified |
| `mackinlay1997` | <https://www.jstor.org/stable/2729691> | Verified |
| `malo2014` | <https://doi.org/10.1002/asi.23062> | Verified |
| `mclean2016` | <https://doi.org/10.1111/jofi.12365> | Verified |
| `moreira2017` | <https://doi.org/10.1111/jofi.12513> | Verified |
| `newey1987` | <https://doi.org/10.2307/1913610> | Verified |
| `novymarx2016` | <https://doi.org/10.1093/rfs/hhv063> | Verified |
| `politis1994` | <https://doi.org/10.1080/01621459.1994.10476870> | Verified |
| `savor2012information` | <https://doi.org/10.1016/j.jfineco.2012.06.011> | Verified |
| `tetlock2007` | <https://doi.org/10.1111/j.1540-6261.2007.01232.x> | Verified |
| `tetlock2008words` | <https://doi.org/10.1111/j.1540-6261.2008.01362.x> | Verified |
| `tetlock2011stale` | <https://doi.org/10.1093/rfs/hhq141> | Verified |
| `uhl2014` | <https://doi.org/10.1080/15427560.2014.967852> | Verified |
| `uhl2021` | <https://doi.org/10.1080/15427560.2020.1821375> | Verified |
| `welch2008` | <https://doi.org/10.1093/rfs/hhm014> | Verified |

The material correction from this pass is the addition of Cakici et al. (2025).
The earlier literature paragraph could otherwise leave Farmer et al. (2023)
standing as uncontested evidence for local predictability. The revised text
states that the replication identifies look-ahead in the implemented
two-sided kernel and reports that correcting it removes most claimed pocket
performance. That is directly relevant to this paper's chronology standard.

## Quantitative and cross-document checks

- The development baseline (`-0.00831`), full-control robustness (`-0.00914`),
  frozen evaluation baseline (`+0.00583`) and era contrast (`+0.01414`) retain
  their distinct estimands everywhere they are compared.
- Notebook 86 reconciles the return decomposition, realised power, observed-rank
  translation and FNSPID drawdown. Notebook 87 reconciles beta-adjusted
  estimates, dependence checks, episode exclusions and composition rates.
- The abstract, research question, data limitation, timing methods, result,
  conclusion, project summary, evidence map and repository summary agree that
  predictive timing is not established.
- FNSPID and LSEG remain separate source regimes and their different return
  definitions are labelled.
- The adverse FNSPID drawdown, LSEG matched-control failure, transaction-cost
  shortfall, prompt failures, multiplicity corrections, MDEs and nulls remain
  present.

## Originality screen

Eight distinctive passages from the revised introduction, methodology,
results and conclusion were searched as exact quoted strings on the public
web. No exact or close match was returned. This is a limited public-search
screen, not an institutional similarity certificate. It cannot check closed
student-paper repositories or self-overlap against an unpublished author
corpus.

## Open notes

1. Replace the reconstructed project summary with the approved form.
2. Confirm the assessment-specific GenAI category and the permissibility of
   the disclosed assistance.
3. Establish the Reuters/LSEG licence, data-processing and ethics basis for
   the completed external prompt processing.
4. Confirm the restricted-distribution sentence and perform the author's full
   final read-through.
