# Exploratory Experiment Ledger

Last updated: 2026-08-05. This is the current completion ledger for the
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
| W2 expanded-LSEG publisher conditioning | Complete; clean null | Notebook 15 maps all 888,155 successful Gemma hashes to 3,112 source codes; 33,250 hashes (3.74%) have Reuters and 857,461 have a non-Reuters source. The frozen all-source, Reuters-only, non-Reuters-only, and Reuters-2× h1 family has zero BH survivors and zero economically viable arms at 10 bps/side. Reuters-only is the least favourable arm (IC −0.02134, p=0.189, gross Sharpe −1.325, break-even −2.658 bps/side, net Sharpe −6.290). The all-source arm exactly reproduces Notebook 13's Gemma mean-continuous result. Do not search additional source tiers or weights on this window. |
| Full-text cash-flow-distance audit | Prepared; human labels outstanding | Notebook 16 exact-hash matches 17,508 non-empty Reuters body rows to the successful Gemma population, yielding 16,861 unique full-text headline hashes. A return-blind 200-event sample is frozen across lexical enrichment strata d0/d1/d2/d3/unmatched at 35/35/35/60/35, with a separately ordered 60-event double-code subset (12 per stratum). Coder sheets omit sentiment, proxy labels, hashes, symbols, dates, and returns. The private key and licensed text remain ignored/local. Weighted kappa and agreement must be evaluated before any human-distance return model is opened. |
| W3 aggregation | Complete | Nine rules × h1 on development. Only `negative_share` survives BH: mean daily cross-sectional IC −0.005433, HAC t −3.119. All net Sharpes are negative at 10 bps/side; best break-even is 0.625 bps/side. |
| W4 sentiment surprise | Complete | Against the control-only nest, firm-baseline and full-surprise additions have small positive OOS R² deltas (+0.000093 and +0.000054 respectively) with positive block-bootstrap intervals. These are predictive-loss improvements, not usable trading alpha. |
| W5 story-type conditioning | Provisional machine arm complete; human validation blocked | Notebook 08 fits one pooled 12-slope model on development with session-date clustering and BH over 12 types × h1. Zero of 12 slopes survives. The smallest raw p-value is `commodity_rates_fx` (slope 0.001593, p=0.0209), but it fails the frozen correction. The conditional type-weighting gate therefore retains the unweighted default. |
| W6 earnings-date effect | Complete, subject to calendar/taxonomy caveats | Notebook 09 maps 22,856 quarterly events and freezes pre −5…−1, event 0, post +1…+5. Zero of three window interactions survives BH. The post raw p-value is 0.0339 but fails correction. Excluding every ±5-session earnings window leaves `negative_share` as the only W3 BH survivor (IC −0.005508, HAC t −2.901). Machine `earnings_guidance` firm-days fall within ±5 sessions of a mapped event 42.27% of the time; this is not a human taxonomy validation. |
| Learned trade/no-trade thresholds | Pooled arm complete; sector/stock variants blocked | Notebook 10 fits through 2017 and selects on 2018–2019. The corrected historical fixed band, activity-matched fixed band, logistic, gradient-boosted, MLP, and shuffled-label MLP have evaluation net Sharpes of −0.108, −5.043, −2.073, −5.467, −1.003, and −0.757 at 10 bps/side after final liquidation. Cash is the best policy. All learned-model evaluation AUCs are approximately 0.49–0.50, and all repaired books have mean net exposure below 4e−18 in absolute value. Historical band `0.4` traded on only 2.98% of validation sessions; applying the same five-name / 50%-session activity floor selects band `0.0`, whose authorised evaluation recomputation is labelled iterative/retrospective. Earlier evaluation tables without bilateral normalisation or final-liquidation cost are invalid. A point-in-time FNSPID sector map is absent, so the protocol-required sector-first and subsequent stock-specific variants cannot be fit honestly. |
| LSEG sector-33 robustness | Complete | Five-arm h1 family on 5,985 news-bearing firm-days / 124 dates; zero survives BH. Reuters-2× IC is +0.00153 (p=0.924). Actionable-only is the largest magnitude (IC −0.03180, p=0.0602) but is not corrected-significant. Target is raw next-open return, so this is not pooled with FNSPID. |
| LSEG midcap-22 robustness | Complete | Reuters-only mean and actionable-only arms on 1,053 news-bearing firm-days / 175–177 usable dates; neither survives BH (IC +0.00481 and +0.00064). Publisher weighting is unidentified because all events are Reuters. |
| LSEG 44-company headline expansion | Both automated scorers complete; human audit outstanding | Separate all-source robustness corpus: 1,158,762 story rows / 888,155 unique normalized headlines over 2025-10-26 to 2026-06-26. FinBERT is 21.03% negative, 46.30% neutral, and 32.66% positive; Gemma 4 26B is 23.10% negative, 40.26% neutral, and 36.64% positive. Across all 888,155 hashes, hard-label agreement is 74.03%, Cohen's κ is 0.598, score Spearman ρ is 0.771, and mean absolute score difference is 0.270. These are agreement diagnostics, not accuracy. VADER remains excluded and the human audit remains open. |
| OpenRouter Gemma 4 26B scorer | Public gate and licensed full-corpus scoring complete | Public n=1,000 gate: accuracy 0.808 [0.784, 0.832], macro-F1 0.813 [0.789, 0.837], Brier 0.274, ECE 0.0445, and cost $0.03786; same-row differences from FinBERT include zero. Licensed run: exact investor prompt, DeepInfra FP8 only, ZDR, data collection denied, no fallback, reasoning off. The append-only CSV contains 889,103 attempts: 888,155 unique successes plus 919 API errors, 27 invalid outputs, and two malformed responses retained as history. Selecting `status == "success"` yields 100% coverage, zero duplicate successes, zero invalid successful probabilities, SHA-256 `afb2d9c3…`, and total reported cost $33.61534475. |
| LSEG 44-company Gemma/FinBERT return robustness | Complete; clean null | Notebook 13 maps 929,691 unique-headline/company associations to the first XNYS open at least 15 minutes after the first timestamp, producing 7,348 priced firm-open rows over 167 entry sessions. The primary family is two scorers × nine pre-existing aggregators × h1 with daily cross-sectional IC, HAC(5), BH, fixed 10 bps/side costs, explicit liquidation, and a 1,999-replication five-session block bootstrap. Zero of 18 IC arms survives BH; zero is economically viable; zero of eight pre-declared metadata-filter sensitivities survives BH. The most favourable gross arm is Gemma `strongest_event`: IC +0.03461 (raw p=0.0992), gross Sharpe 1.250, but break-even is only 3.709 bps/side, net Sharpe −2.119, and the net-mean bootstrap interval is [−0.001877, +0.000182]. This retrospective eight-month raw-return snapshot is not a pristine holdout and is not promoted. |
| LSEG sector-neutral / hysteresis translation | Complete; improved gross/cost efficiency but clean 10-bps null | Notebook 17 changes only the portfolio construction applied to Gemma `strongest_event`, comparing global rank, within-sector rank, sector extremes, and a top-half/bottom-half extreme-retention rule. Sector hysteresis is the best arm: gross Sharpe 1.393 versus 1.250 for the exact Notebook 13 reproduction, mean half-L1 turnover 0.280 versus 0.665, and break-even 4.053 versus 3.709 bps/side. It is positive in both chronological halves and remains positive at 1–2 bps/side (2-bps net Sharpe 0.706; total net return +1.882%), but its gross block-bootstrap p=0.226, its 95% gross-mean interval [−0.000114, +0.000615] spans zero, and at 10 bps/side net Sharpe is −2.015 with a net-mean interval [−0.000683, +0.000044]. Zero of four gross-mean tests survives BH and zero arm clears the alpha gate. The six-sector breadth floor leaves only 57/167 active sessions; on those same days the paired hysteresis-minus-global gross difference is +3.73 bps/day but remains uncertain (p=0.141; interval [−1.13, +8.62] bps/day). This is a useful turnover/coverage result, not validated alpha; no more tuning is allowed on this opened window. |
| LSEG sparse novel-material-event strategy | Complete; clean null | Notebook 14 freezes two Gemma signals × 1/3/5-session state before opening their returns: strongest direct single-company novel material event, and the same selected event attenuated by an unvalidated headline-only cash-flow-distance proxy. The 30-day/Jaccard-0.80 screen leaves 172,198 novel material headlines; strongest-event selection yields 6,127 company-entry events across all 44 symbols and 167 entry sessions, including 2,570 with a distance proxy. Zero of six IC tests survives BH and zero arm is economically viable at 10 bps/side. The best new gross arm is unattenuated material h5: IC +0.00689 (raw p=0.596), gross Sharpe 0.663, break-even 1.904 bps/side, net Sharpe −2.820, and net-mean bootstrap interval [−0.001161, −0.000140]. It is worse than Notebook 13's Gemma `strongest_event` magnitude reference (gross Sharpe 1.250; break-even 3.709). The cash-flow-distance arms do not improve the result; preserve the retrospective null and do not tune this window again. |
| Strategy sweep / event-time diagnostics | Complete | Zero of 135 strategy cells clears 10 bps/side and zero has positive net Sharpe. Among 120 cells meeting the history/breadth floor, zero clears the cost-and-uncertainty deployment gate, so the recomputed policy chooses **cash**. On the 1,004-session evaluation block this changes total net return from −19.74% for the historical traded arm (Sharpe −0.944, max drawdown −29.67%) to 0.00% under the notebook's zero-cash-return convention: a +19.74 percentage-point iterative improvement with zero turnover and drawdown. Zero of 80 event-time signal × lag cells survives BH. |
| VaR / ES tail-risk factorial | Closed and registered; not extended | Existing result is retained: news arrival matters, semantics do not. The final-experiments protocol explicitly de-prioritises further tail-risk work. |

## Experiments That Cannot Be Completed From Current Inputs

These are not silently replaced with proxies:

1. Human precision/recall for breaking, repetition, and routine reporting: the
   seeded 180-story blinded worksheet is unlabelled.
2. Human validation of the 12-type taxonomy: the FNSPID and LSEG audit sheets
   have no manual labels.
3. Human validation of cash-flow distance: the 200-event/60-double-code LSEG
   pack now exists, but both coder worksheets are unlabelled.
4. FNSPID publisher weighting and four-way publisher prestige tiers: publisher
   identity is absent from the checkpoint; LSEG's non-Reuters codes lack a
   defensible source-category mapping.
5. FNSPID sector-conditioned thresholds: no explicit point-in-time sector map
   exists. The frozen protocol requires this arm before stock-specific fitting.

Completing any of these requires new human labels or new source metadata, not
more computation on the existing checkpoint.

## Academic Interpretation

The exploratory programme now gives a consistent answer: small statistical
associations exist, especially for the share of negative stories and for the
surprise decomposition, but conditioning, publisher weighting, learned gates, scorer
replacement, sparse event selection, cash-flow-distance sizing, sector control,
rank hysteresis, and trading
translations do not produce robust economic performance after costs. Raw
subgroup p-values in story type and earnings do not survive their declared
multiplicity families. Null arms are retained rather than retuned.

The strongest new portfolio result is still informative about implementation:
sector hysteresis reduces turnover by about 58% and modestly raises gross Sharpe,
but it trades on only 34% of sessions, remains statistically uncertain, and
breaks even at roughly 4 bps/side rather than the frozen 10 bps. That supports a
cost-and-coverage limitation, not a claim that the sentiment signal is valueless.

Gate F1 remains a research-framing decision, not an unfinished experiment. The
evidence favours a bounded question about what negative-share aggregation and
sentiment residualisation measure, with economic non-viability stated as a
central result. It does not support a publisher, event-type, earnings-timing, or
neural-threshold alpha claim.
