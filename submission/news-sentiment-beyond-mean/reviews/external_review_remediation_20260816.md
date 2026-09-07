# External-review remediation — 16 August 2026

## Outcome

The external panel's blocking timing objection is resolved by new analysis and
by narrowing the paper's claim. The new evidence does **not** rescue a
predictive next-open interpretation. It places the development association in
the assigned session's intraday leg, finds no post-close continuation, and
also finds a strong negative association before the assigned open. With no
recoverable row-level availability timestamps, stale assignment, earlier
information and contemporaneous reaction remain observationally equivalent.

The revised paper therefore treats timing ambiguity as a central result rather
than as a caveat in another chapter. Its scientific spine is now: a selected
development association, unresolved event time, frozen temporal
non-replication and adverse economic evidence.

## Blocking finding

| Review objection | Action | Result and interpretation | Status |
|---|---|---|---|
| The same timing defect could explain next-open/close asymmetry, singleton concentration and the era reversal | Notebook 86 decomposes the assigned-window return into assigned-open-to-close, assigned-close-to-next-open and previous-close-to-assigned-open legs. Notebook 87 adds the requested previous-open-to-assigned-open probe. Exact gross-return identities are checked before abnormal returns are formed. | Development intraday: `-0.010291`, BH q=`0.001121`; post-close: `+0.001548`, q=`0.594922`; previous-close-to-open: `-0.006381`, q=`0.068578`; previous-open-to-open: `-0.019257`, p=`7.868e-09`. The evidence is incompatible with a clean predictive-window claim. | **Closed by analysis and claim withdrawal** |
| The timing caveat and the apparent specificity result were separated across chapters | The abstract, introduction, data, methodology, results, conclusion, project summary, repository summary and evidence map now state the same timing boundary. | The phrases “next-open specificity” and predictive timing are explicitly withdrawn. The paper says what the aggregate evidence can and cannot identify. | **Closed** |
| The timing explanation might also account for the era result | The leg decomposition is repeated by era and the full baseline contrast is preserved. Observable composition diagnostics are added. | The full contrast remains `+0.014143` (`p=0.017284`). The intraday contrast is `+0.013664` (`p=0.020423`, BH q=`0.061269`). Evaluation is more, not less, singleton-heavy: 56.53% versus 48.32%. Timing and composition remain possible causes, but neither is identified as the cause. | **Closed within the available data; causal explanation remains out of scope** |

Notebook 75 was not executed. Its SHA-256 remained
`df0b6c5105158198f62c828842b3ff3e1051ccaf664e7a778c9c66e11cffc455`
before and after both post-review studies.

## Major and minor review points

| Topic | Completed change | Status |
|---|---|---|
| Title, research question and sparse-news framing | The title now names a hard threshold rather than implying a generally rich distributional statistic. The research question separates development, timing and temporal validation. The 48.3% singleton share and median of two stories appear at the start of the paper. | **Closed** |
| Selection and winner's curse | Negative share is described as selected from the nine-rule development comparison. The baseline p-value is placed beside a simple Bonferroni-over-nine screen but is not called selective inference. Evaluation is described as a frozen temporal replication, not independent confirmation. | **Closed** |
| Prospective versus realised power | The manuscript retains the prospective 35.4% calculation and adds realised, ex-post power of 26.5%; the evaluation HAC standard error is 15.8% above its square-root projection. | **Closed** |
| Effect-size translation | The unattainable 0.8 rank move is withdrawn. The observed tied-regressor 90th-minus-10th spread is 0.56056, implying `-0.512` return-rank percentile points under the full-control coefficient. This is not presented as basis points. | **Closed** |
| FNSPID drawdown omission | HAR base maximum drawdown (`-15.73%`) and overlay maximum drawdown (`-16.36%`) are now reported together in the results, conclusion, generated audit and provenance table. | **Closed** |
| Episode and crash concentration | Forty-four leave-one-risk-off-episode estimates and a February–April 2020 exclusion were added. Every leave-one-out return interval stays above zero and the crash exclusion remains positive (`p=0.0034`), while the worse overall return and drawdown remain visible. | **Closed; does not rescue the economic gate** |
| Hypothesis alignment | The final structure uses H1a/H1b and H2–H4. LSEG coefficient transfer is a separate precision-bounded assessment; H4 is the matched-control economic gate. | **Closed** |
| FinBERT validity and composition | The exact checkpoint revision, Financial PhraseBank domain, absence of a human-label audit or independent LLM comparator, class rates, zero-share rates and singleton shares are stated. | **Closed as a limitation; external semantic validation remains future work** |
| Literature positioning | The paper narrows novelty to its evidence sequence, not sentiment dispersion. It integrates stale-news evidence, Garcia's state-dependent result and the Cakici et al. replication that challenges the original “pockets” implementation on look-ahead grounds. | **Closed** |
| Market adjustment and dependence | Trailing 252-session beta adjustment, beta control, HAC lags 0/5/21/42, first-order autocorrelation and Ljung–Box diagnostics were added. The era contrast survives each declared benchmark. | **Closed** |
| Administrative readiness | The missing approved project summary, assessment-specific GenAI permission and Reuters/LSEG external-processing authority are now explicit submission blockers. The restricted-distribution sentence remains an author confirmation. | **Closed as disclosure; external author action remains** |

Additional points from the report are also explicit: the model-free double sort
is descriptive; the FNSPID break-even cost is a point estimate without an
interval; the US$1,767 LSEG compounding residual is reconciled; fixed circular
blocks are distinguished from the stationary bootstrap; average names per
session and class-base-rate changes are reported; and the frozen registration
is described as repository-self-attested with a dirty execution worktree.

## Reproducibility record

- Notebook 86 was independently rerun in
  `/private/tmp/news-sentiment-nb86.1QfgQj`; all nine CSV files were
  byte-identical and the manifest was canonically identical after excluding
  run path and execution time.
- Notebook 87 was independently rerun in
  `/private/tmp/news-sentiment-nb87.1ebBWQ`; all fourteen CSV files were
  byte-identical under the same comparison.
- Promotion records and claim boundaries are stored beside each result set in
  `experiments/results/86_fnspid_outcome_leg_decomposition/` and
  `experiments/results/87_fnspid_external_review_robustness_pack/`.
- Only licence-safe aggregate outputs were promoted. FNSPID and LSEG remain
  separate, and no raw text, identifiers, Parquet panels or scoring logs were
  added.
