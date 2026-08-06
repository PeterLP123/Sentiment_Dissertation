# Cash-Flow-Distance Human Audit Codebook

Version: `cash_flow_distance_v1`, frozen 2026-08-05.

Reliability estimand details below were frozen on 2026-08-06, before either
coder entered a label.

## Construct

Label how many unresolved economic prerequisites separate the event described
in the Reuters story from a realised firm cash-flow consequence. Judge the
event described by the story, not whether the headline sounds positive or
negative and not whether the stock price moved.

| Label | Meaning | Typical evidence |
| --- | --- | --- |
| `0` | Realised or directly booked cash-flow outcome | Reported revenue/profit/cash flow, payment received or made, completed distribution, completed repurchase or dividend payment. |
| `1` | Binding near-term commercial or financing step | Signed contract, firm order, completed acquisition, final regulatory approval, launched product with an operative commercial channel. One implementation step may remain. |
| `2` | Material but contingent intermediate step | Pending approval, clinical or technical milestone, non-final proposal, guidance or forecast, transaction still subject to stated conditions. Multiple execution steps remain. |
| `3` | Speculative, strategic, or distant possibility | Exploring, considering, seeking, planning, preliminary/non-binding talks, long-horizon target, or possibility with no committed transaction. |
| `NA` | Cannot be coded reliably | Insufficient context, no identifiable firm cash-flow pathway, conflicting descriptions, or story text is unusable. |

## Coding fields

- `human_distance`: one of `0`, `1`, `2`, `3`, or `NA`.
- `unresolved_prerequisites`: short list of the remaining prerequisites that
  justify the label; write `none` for label 0.
- `evidence_span`: the shortest passage supporting the decision.
- `confidence`: `high`, `medium`, or `low`.
- `notes`: optional ambiguity note. Do not search for price reactions or read
  any model score while coding.

## Arbitration and acceptance gate

Coder B independently labels the frozen 60-event subset. Compare weighted
Cohen's kappa and exact/adjacent agreement before opening any return model that
uses the human labels. Disagreements are preserved, then adjudicated using the
same codebook. The construct is not accepted merely because the machine proxy
predicts the human label. The frozen acceptance gate is: quadratic-weighted
Cohen's kappa must be at least 0.60, its 95% event-bootstrap lower bound must be
at least 0.40, and adjacent agreement (absolute coder difference at most one)
must be at least 0.80. Report exact agreement and the full confusion matrix
without an additional pass/fail cut. If any gate fails, revise the codebook and
collect a fresh reliability sample before opening returns.

`NA` is not an ordinal distance. Quadratic-weighted kappa therefore uses only
double-coded events where both coders selected 0-3, and at least 48 of the 60
events must be jointly numeric for the gate to pass. Exact and adjacent
agreement use all 60 events: two `NA` labels agree, while a numeric/`NA` pair
does not agree. The 95% kappa interval uses 9,999 event-level bootstrap
resamples of the complete 60-event overlap with seed `20260819`; kappa is
recomputed on the jointly numeric pairs in each resample and undefined
replicates are omitted. These rules cannot be changed after labels are opened.

The worksheets are deliberately blinded: they omit sentiment scores, proxy
strata, source hashes, dates, symbols, and all realised returns. Licensed story
text remains in ignored local outputs and must not be committed.
