# Exploratory Experiment Ledger

Last updated: 2026-08-04. This is the current completion ledger for the
`final_experiments` closing programme. Generated row-level outputs remain under
the ignored `final_experiments/outputs/` boundary; the figures below are
aggregate and licence-safe.

## What Has Been Run

| Workstream / arm | Status | Design and result |
| --- | --- | --- |
| W1 data consolidation | Complete | FNSPID primary spine: 715,546 news-bearing firm-days, 570 priced symbols, 3,262 sessions. LSEG sector-33 and midcap-22 remain separate robustness regimes. Development ends 2019-12-31; chronological evaluation starts 2020-01-01. |
| W2 distribution EDA | Complete | Firm-day score moments and story-count strata are in notebook 02. |
| W2 novelty / repetition | Machine arm complete; human validation blocked | Strictly-earlier comparison yields 45,955 repetition and 1,553,496 novel-or-unclassified stories. Market recap is a separate flag. The blinded 180-story audit exists but has no human labels. |
| W2 breaking / repetition / routine | Provisional machine arm complete | Notebook 08 combines the novelty screen with the existing ordered event regexes. “Breaking” and “routine” remain candidate labels until human audit; market recaps are not called routine. |
| W2 publisher inventory / weighting | Complete on LSEG only | Notebook 11 inventories 594,287 sector-33 events and tests one ex-ante Reuters-2× weight against unweighted, Reuters-only, non-Reuters, and actionable-only arms. FNSPID publisher identity is absent. Opaque non-Reuters codes cannot support honest aggregator/syndicator/promotional tiers. None of five sector-33 arms survives BH. |
| W3 aggregation | Complete | Nine rules × h1 on development. Only `negative_share` survives BH: mean daily cross-sectional IC −0.005433, HAC t −3.119. All net Sharpes are negative at 10 bps/side; best break-even is 0.625 bps/side. |
| W4 sentiment surprise | Complete | Against the control-only nest, firm-baseline and full-surprise additions have small positive OOS R² deltas (+0.000093 and +0.000054 respectively) with positive block-bootstrap intervals. These are predictive-loss improvements, not usable trading alpha. |
| W5 story-type conditioning | Provisional machine arm complete; human validation blocked | Notebook 08 fits one pooled 12-slope model on development with session-date clustering and BH over 12 types × h1. Zero of 12 slopes survives. The smallest raw p-value is `commodity_rates_fx` (slope 0.001593, p=0.0209), but it fails the frozen correction. The conditional type-weighting gate therefore retains the unweighted default. |
| W6 earnings-date effect | Complete, subject to calendar/taxonomy caveats | Notebook 09 maps 22,856 quarterly events and freezes pre −5…−1, event 0, post +1…+5. Zero of three window interactions survives BH. The post raw p-value is 0.0339 but fails correction. Excluding every ±5-session earnings window leaves `negative_share` as the only W3 BH survivor (IC −0.005508, HAC t −2.901). Machine `earnings_guidance` firm-days fall within ±5 sessions of a mapped event 42.27% of the time; this is not a human taxonomy validation. |
| Learned trade/no-trade thresholds | Pooled arm complete; sector/stock variants blocked | Notebook 10 fits through 2017 and selects on 2018–2019. The corrected historical fixed band, activity-matched fixed band, logistic, gradient-boosted, MLP, and shuffled-label MLP have evaluation net Sharpes of −0.108, −5.043, −2.073, −5.467, −1.003, and −0.757 at 10 bps/side after final liquidation. Cash is the best policy. All learned-model evaluation AUCs are approximately 0.49–0.50, and all repaired books have mean net exposure below 4e−18 in absolute value. Historical band `0.4` traded on only 2.98% of validation sessions; applying the same five-name / 50%-session activity floor selects band `0.0`, whose authorised evaluation recomputation is labelled iterative/retrospective. Earlier evaluation tables without bilateral normalisation or final-liquidation cost are invalid. A point-in-time FNSPID sector map is absent, so the protocol-required sector-first and subsequent stock-specific variants cannot be fit honestly. |
| LSEG sector-33 robustness | Complete | Five-arm h1 family on 5,985 news-bearing firm-days / 124 dates; zero survives BH. Reuters-2× IC is +0.00153 (p=0.924). Actionable-only is the largest magnitude (IC −0.03180, p=0.0602) but is not corrected-significant. Target is raw next-open return, so this is not pooled with FNSPID. |
| LSEG midcap-22 robustness | Complete | Reuters-only mean and actionable-only arms on 1,053 news-bearing firm-days / 175–177 usable dates; neither survives BH (IC +0.00481 and +0.00064). Publisher weighting is unidentified because all events are Reuters. |
| LSEG 44-company headline expansion | Automated labelling and LLM agreement audit complete; human audit outstanding | Separate all-source robustness corpus: 1,158,762 story rows / 888,155 unique scorable headlines over 2025-10-26 to 2026-06-26. Exact hashes reused 568,704 frozen FinBERT labels and required 319,451 fresh calls; final coverage is 888,155/888,155. FinBERT hard labels are 21.03% negative, 46.30% neutral, and 32.66% positive. VADER is excluded. The researcher-supplied structured-output LLM prompt passed its 200/200 gate; across the frozen 1,000-attempt hash-order sample, 997 rows were valid (99.7%) and three deterministic format failures remained after an identical resume. On n=997, Gemma–FinBERT hard-label agreement is 69.31%, Cohen's κ is 0.532, and score Spearman ρ is 0.750. These are agreement diagnostics, not accuracy. No human labels or price/return claim exist for this corpus yet. |
| OpenRouter Gemma 4 26B scorer | Public gate complete; licensed LSEG scoring running | Frozen label-blind SHA-256 sample of 1,000 rows from `financial_sentiment_v2.csv`; exact investor prompt, DeepInfra FP8 only, ZDR, data collection denied, no fallback, reasoning off, strict probability JSON. After preserving 76 initial upstream 429s and appending a lower-concurrency retry, coverage and format validity are both 100%. Accuracy is 0.808 [0.784, 0.832], macro-F1 0.813 [0.789, 0.837], Brier 0.274, ECE 0.0445, and cost $0.03786. On the same rows, FinBERT accuracy is 0.797 and macro-F1 0.797; paired differences include zero (McNemar p=0.505), so the result supports comparability, not superiority. The researcher confirmed hosted processing is permitted on 2026-08-04. A private 10-headline LSEG gate achieved 10/10 valid DeepInfra responses for $0.00037782. The concurrency-20 gate achieved 2,000/2,000 successes in 144.405 seconds; a subsequent concurrency-25 gate achieved 2,000/2,000 with zero terminal errors, invalid outputs, or retries in 89.133 seconds (22.44/second) for $0.07565025. The full resumable run now continues at concurrency 25 on UCL project storage from a 90,055-success checkpoint. No full-corpus result or return claim exists yet. |
| Strategy sweep / event-time diagnostics | Complete | Zero of 135 strategy cells clears 10 bps/side and zero has positive net Sharpe. Among 120 cells meeting the history/breadth floor, zero clears the cost-and-uncertainty deployment gate, so the recomputed policy chooses **cash**. On the 1,004-session evaluation block this changes total net return from −19.74% for the historical traded arm (Sharpe −0.944, max drawdown −29.67%) to 0.00% under the notebook's zero-cash-return convention: a +19.74 percentage-point iterative improvement with zero turnover and drawdown. Zero of 80 event-time signal × lag cells survives BH. |
| VaR / ES tail-risk factorial | Closed and registered; not extended | Existing result is retained: news arrival matters, semantics do not. The final-experiments protocol explicitly de-prioritises further tail-risk work. |

## Experiments That Cannot Be Completed From Current Inputs

These are not silently replaced with proxies:

1. Human precision/recall for breaking, repetition, and routine reporting: the
   seeded 180-story blinded worksheet is unlabelled.
2. Human validation of the 12-type taxonomy: the FNSPID and LSEG audit sheets
   have no manual labels.
3. FNSPID publisher weighting and four-way publisher prestige tiers: publisher
   identity is absent from the checkpoint; LSEG's non-Reuters codes lack a
   defensible source-category mapping.
4. FNSPID sector-conditioned thresholds: no explicit point-in-time sector map
   exists. The frozen protocol requires this arm before stock-specific fitting.

Completing any of these requires new human labels or new source metadata, not
more computation on the existing checkpoint.

## Academic Interpretation

The exploratory programme now gives a consistent answer: small statistical
associations exist, especially for the share of negative stories and for the
surprise decomposition, but conditioning, weighting, learned gates, and trading
translations do not produce robust economic performance after costs. Raw
subgroup p-values in story type and earnings do not survive their declared
multiplicity families. Null arms are retained rather than retuned.

Gate F1 remains a research-framing decision, not an unfinished experiment. The
evidence favours a bounded question about what negative-share aggregation and
sentiment residualisation measure, with economic non-viability stated as a
central result. It does not support a publisher, event-type, earnings-timing, or
neural-threshold alpha claim.
