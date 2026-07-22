# Recent-news mid-cap FinBERT strategy protocol

## Decision

Freeze the existing mid-cap FinBERT event rule under the new strategy ID
`recent-news-midcap-finbert-event-v1` and replay it without changing its
parameters. This is a new immutable strategy artifact built from an existing
rule and existing recent Reuters/LSEG data; it is not presented as a newly
discovered model edge.

The strategy is acceptable only if the clean replay preserves all of the
following pre-existing results at 10 basis points per side:

- positive net return and positive annualised Sharpe in development;
- positive net return and positive annualised Sharpe in the evaluation period;
- at least 20 active sessions in each period;
- no input-integrity, point-in-time, or reproducibility failure; and
- a positive break-even transaction cost above the frozen 10 basis-point cost.

The evaluation start remains `2026-01-02`. The sample is explicitly marked
`previously_explored`, so the evaluation result is corroborating evidence, not
a pristine untouched holdout.

## Frozen inputs and rules

The complete machine-readable specification is
`configs/strategy_research/baselines/recent_news_midcap_finbert_event_v1.toml`.
No model inference, LSEG download, universe change, threshold search, holding
period search, or cost-model change is authorised for the replay.

The replay uses:

- the existing 22-stock Reuters/LSEG mid-cap collection;
- the pinned `ProsusAI/finbert` revision
  `4556d13015211d73dccd3fdd39d39232506f3e43`;
- cached point-in-time headline labels and the existing adjusted-open panel;
- the first eligible XNYS open after a 15-minute processing buffer;
- one-session open-to-open holdings with no carry;
- equal-weight, dollar-neutral exposure when both sides exist; and
- 10 basis points per dollar traded, including final liquidation.

## Search boundary

The same-day search also tested broader 33-stock and persistent-sentiment
alternatives. Their negative or unstable results remain preserved; this replay
must not tune itself in response to those failures. No new LSEG requests are
needed, leaving the 10,000-request allowance unused.

## Required handoff

After the clean replay, publish a comprehensive explainer containing exact
event eligibility, signal, execution, portfolio, cost, cash, and exit rules;
development and evaluation metrics; cost capacity; statistical uncertainty;
known biases; reproducibility evidence; and the failed-strategy search trail.

## Result fixed on 22 July 2026

The clean replay resolved to
`recent-news-midcap-finbert-event-v1-dfdd88113a5f` and passed every acceptance
criterion. At 10 bps per side, development returned +3.419% with 0.780 Sharpe
over 25 active sessions; evaluation returned +4.004% with 0.593 Sharpe over 37
active sessions. The approximate break-even costs were 18.18 and 16.52 bps per
side respectively.

The five-session moving-block evaluation interval for mean daily net return was
[-0.0886%, +0.1934%], so statistical confidence remains inconclusive. A second
identical command validated and reused the immutable output. See
`docs/recent_news_midcap_finbert_strategy.md` for the full rules, material
passport, research-risk audit, search trail, and reproduction hashes.
