# Sentiment Trading Baseline

For a visual walkthrough of the data, models, trading rule, sources and observed
result, open the [interactive HTML explainer](sentiment_trading_baseline_explainer.html).

This section defines the deliberately simple strategy baseline used while the
dissertation research question is being finalised. It is isolated under the
existing `strategy_research` namespace and does not replace the dissertation's
core analysis, historical LLM strategy, Week 6 experiments, or their artifacts.

> **Evidence status:** this is a retrospective, conceptual adaptation of the
> literature, not an exact paper replication, a pristine out-of-sample test, a
> causal design, deployable alpha, or investment advice. The initial Reuters
> mid-cap sample has already been examined elsewhere in the repository. A null
> result is valid evidence and must be retained without outcome-driven retuning.

## Research basis and replication claim

The v1 baseline combines established ideas rather than claiming to reproduce
one paper's data, model, or return series.

| Primary source | Relevant contribution | Use in this baseline |
| --- | --- | --- |
| [Tetlock (2007), *Giving Content to Investor Sentiment*](https://doi.org/10.1111/j.1540-6261.2007.01232.x) | Relates pre-existing media pessimism to subsequent market returns and reversal. | Motivates a timestamped text signal, a short holding horizon, and cost-aware interpretation. The WSJ market-level corpus and General Inquirer method are not reproduced. |
| [Loughran and McDonald (2011), *When Is a Liability Not a Liability?*](https://doi.org/10.1111/j.1540-6261.2010.01625.x) | Shows why general-language polarity can be misleading in financial text. | Motivates a finance-domain primary model and a transparent lexical comparator. The 10-K dictionary study is not itself a trading strategy and is not implemented as the v1 signal. |
| [Hutto and Gilbert (2014), VADER](https://doi.org/10.1609/icwsm.v8i1.14550) | Provides a deterministic rule-based sentiment model whose published classification convention uses the compound score. | Motivates the lexical control, but the reusable score artifact's `pos/neg/neu` share-argmax is a derived **VADER share-argmax** comparator, not canonical VADER classification. |
| [Araci (2019), FinBERT](https://doi.org/10.48550/arXiv.1908.10063) and the [ProsusAI model card](https://huggingface.co/ProsusAI/finbert) | Adapts BERT to financial text and fine-tunes three-class financial sentiment. | Supplies the primary local scorer. Its classification performance does not establish return predictability. |
| [López-Lira and Tang (arXiv:2304.07619v6), *Can ChatGPT Forecast Stock Price Movements?*](https://arxiv.org/abs/2304.07619v6) ([SSRN DOI](https://doi.org/10.2139/ssrn.4412788)) | Aggregates company-linked headline labels into daily long/short portfolios and evaluates tradable drift, turnover, and costs. | Supplies the closest portfolio template. The historical GPT-4 signal, broader universe, target-specific prompt, timing windows, approximately 200% gross exposure, two-name leg minimum, and one-sided fallback are not reproduced. |
| [Ke, Kelly, and Xiu (2019), *Predicting Returns With Text Data*](https://doi.org/10.3386/w26186) | Learns return-specific text weights from a very large news corpus. | Deferred: the current 22-stock, one-year cohort is too small for a defensible supervised return-text model without severe overfitting risk. |

Accordingly, the checked-in config records
`fidelity = "conceptual_adaptation"` and lists the deviations explicitly. In dissertation
writing, call the result the **Reuters FinBERT/VADER-share-argmax adjusted-open baseline** or
an **adapted literature baseline**, never a replication of Tetlock,
López-Lira–Tang, or FinBERT trading returns.

## Frozen v1 contract

The authoritative specification is
[`configs/strategy_research/baselines/lseg_midcap_finbert_vader_v1.toml`](../configs/strategy_research/baselines/lseg_midcap_finbert_vader_v1.toml).
The strict loader rejects unknown fields and changes to the v1 invariants.

| Component | Frozen rule |
| --- | --- |
| Sample status | `previously_explored`; the chronological evaluation begins `2026-01-02` as an interpretation aid, not a claim of an untouched holdout. |
| News | Reuters-only, headline-only LSEG mid-cap collection; earliest eligible `story_family_id × symbol` revision. |
| Relevance screen | Require an explicit, single-company target and exclude market-price/technical recap headlines. A missing or invalid configured score fails the run; it is never changed to neutral. |
| Availability | `version_created_utc + 15 minutes`; execute at the first XNYS open strictly after that timestamp in `America/New_York`. |
| Primary scorer | `ProsusAI/finbert`, exact complete snapshot revision `4556d13015211d73dccd3fdd39d39232506f3e43`; retain the three probabilities and `p_positive - p_negative`. |
| Comparator | **VADER share-argmax** from `nltk.sentiment.vader`: the artifact labels by the largest `pos`, `neg`, or `neu` share and stores `positive share - negative share` as a diagnostic. This is not canonical compound-threshold VADER classification. |
| Signal | Map hard labels to positive `+1`, neutral `0`, and negative `-1`; average one or more events within `scorer × symbol × eligible session`; trade only the sign of a non-zero mean. One-session holding period, no signal carry and no return-fitted threshold. |
| Portfolio | Predeclared FinBERT primary and VADER-share-argmax comparator; `$1,000,000` initial NAV and `1.0` gross equal-weight dollar-neutral exposure, split equally across the two legs. At least one long and one short are required; otherwise the session stays in cash. |
| Return | Split-adjusted open at entry to split-adjusted open one XNYS session later. Dividends are not back-adjusted. Each interval is assigned by its entry session; the evaluation begins with the interval entered on the configured boundary. |
| Friction | Charge `10` basis points per dollar traded on `sum(abs(change in portfolio weight))` and force final liquidation. This is equivalent to `20` basis points for an unchanged position's full open-and-close round trip. The pinned paper instead defines one-way turnover as half that weight change and applies costs per round trip, so its cost figures are not directly comparable. Borrow availability, spread and market impact remain limitations. |
| Inference | Five-session contiguous block bootstrap with `2,000` replications and seed `20260718`, comparing evaluation-period FinBERT with zero-return cash and with VADER share-argmax. Cash yield and financing are omitted. Report FinBERT even if the comparator performs better. The terminal liquidation is retained as an audit row but its exit cost is folded into the final holding interval for daily statistics. |

The continuous FinBERT and VADER scores remain diagnostics. They must not be
substituted for the frozen hard-label signal after returns are observed.

## End-to-end workflow

Install the local scorer and LSEG extras on the relevant machines:

```bash
python -m pip install -e ".[dev,lseg,baselines,finbert]"
```

### 1. Define, check and collect the news cohort

The tracked source definition is
[`configs/lseg_us_midcap_22_1y.toml`](../configs/lseg_us_midcap_22_1y.toml).
It freezes the 22 symbols, Reuters-only queries, UTC interval and local output
roots. For a new collection, use a new collection ID and dates; never reuse an
existing ID for different settings.

```bash
sentiment-bench lseg-news-check --config configs/lseg_us_midcap_22_1y.toml
sentiment-bench fetch-lseg-news --config configs/lseg_us_midcap_22_1y.toml
```

Collection is atomic and resumable. Raw pages and manifests preserve the LSEG
timestamps and query associations. The baseline must not use a web scrape or a
later article revision as though it had been available earlier.

### 2. Build the canonical local corpus

```bash
sentiment-bench build-lseg-corpus \
  --source Data/collections/lseg_us_midcap_22_1y/raw/lseg_us_midcap_22_1y
```

The completed manifest is
`Data/collections/lseg_us_midcap_22_1y/derived/us_midcap_22_1y/manifest.json`.
The baseline consumes only a completed, hash-verified corpus. It uses the event
timestamp and target association from that corpus; the score file's global
`first_timestamp` field never determines execution timing.

### 3. Score every frozen headline locally

FinBERT and VADER scoring is an upstream, price-independent step:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_lseg_headline_study.py score-baselines \
  --collection-root Data/collections/lseg_us_midcap_22_1y \
  --output Data/collections/lseg_us_midcap_22_1y/derived/headline_scores_finbert4556_vader_baseline_v1_cacheonly_20260718.csv \
  --finbert-batch-size 256 \
  --finbert-revision 4556d13015211d73dccd3fdd39d39232506f3e43
```

The scorer is append-only and resumable by normalized headline hash and model.
Its adjacent manifest records the population hash, successful counts, package
versions, local device, the enforced FinBERT revision, the VADER lexicon-text
checksum and final CSV hash. The formal baseline rejects a manifest that merely
names a FinBERT revision without proving it was passed to model loading. Run
inference from the cached repository snapshot; if a different snapshot,
lexicon or package stack is required, create a new score artifact and new
baseline config rather than rewriting this one.

FinBERT probabilities are reduced to `p_positive - p_negative`. The existing
VADER-share-argmax uses its `pos`, `neg`, and `neu` shares, stores `pos - neg`,
and retains the largest-share class. Do not describe that label or numeric score
as canonical VADER compound classification or silently change thresholds.

### 4. Build and verify the adjusted-open panel

First collect the LSEG price source through the normal hash-manifested price
command (omit `--overwrite` so an existing export remains immutable):

```bash
sentiment-bench fetch-lseg-prices \
  --config configs/lseg_us_midcap_22_1y.toml \
  --start 2025-06-20 \
  --end 2026-07-10 \
  --output Data/derived/prices/lseg_us_midcap_22_1y.csv
```

Then convert the existing 22-symbol export into the strict strategy artifact:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_strategy_price_panel.py \
  --source-panel Data/derived/prices/lseg_us_midcap_22_1y.csv \
  --source-manifest Data/derived/prices/lseg_us_midcap_22_1y.manifest.json \
  --output-panel Data/derived/prices/lseg_us_midcap_22_strategy_adjusted_open.csv \
  --output-manifest Data/derived/prices/lseg_us_midcap_22_strategy_adjusted_open.manifest.json \
  --expected-symbol-count 22
```

The converter verifies the source hash and adjustment provenance, requires a
complete 22-symbol XNYS symbol-session grid, and refuses to overwrite an
existing artifact. The resulting manifest must state split-adjusted opens and
`dividend_adjusted = false`; close prices are never substituted as a fallback.

### 5. Inspect the frozen run

```bash
sentiment-bench strategy baseline inspect \
  --config configs/strategy_research/baselines/lseg_midcap_finbert_vader_v1.toml
```

Use the inspection output to confirm the resolved identity, hashes, coverage,
evaluation boundary and output roots. Resolve missing or invalid inputs before
running, but do not change the signal, holding period, costs, scorer roles or
universe in response to return information.

### 6. Execute once and retain the result

```bash
sentiment-bench strategy baseline run \
  --config configs/strategy_research/baselines/lseg_midcap_finbert_vader_v1.toml
```

The run consumes completed scores and prices and makes no model or network
calls. Its resolved run ID incorporates the frozen configuration and verified
input identities. Matching completed artifacts are verified and reused;
incompatible inputs fail closed rather than overwriting prior evidence.

## Point-in-time and no-retuning rules

These rules are part of the research contract, not optional robustness choices:

1. Select the earliest relevance-passing revision for each
   `story_family_id × symbol`; later revisions never replace earlier text.
2. Join a score by normalized headline hash, but retain each event's own target
   timestamp, matched-symbol set and screening flags. A repeated headline does
   not inherit another occurrence's availability or later-unioned metadata.
3. Reject missing, naive or invalid timestamps. Add the 15-minute processing
   buffer before resolving the first strictly later XNYS open, including
   weekends, holidays and daylight-saving transitions.
4. Build text labels and session signals without opening prices or returns.
   Prices enter only after the event and signal identities are frozen.
5. Use the configured evaluation boundary. A return interval is assigned by
   its entry session; no outcome is moved between blocks after it is observed.
   Limit the accounting spine to the first through last selected event session,
   then liquidate at the immediately following open.
6. Do not choose between FinBERT and VADER, tune thresholds, filter stocks,
   change costs, select horizons or revise screening after seeing performance.
7. If a materially different rule is worth testing, assign a new experiment ID,
   freeze a new config before opening its outcomes, and report it separately.
   A failed historical sample is not another development set.
8. Preserve gross and net null results. The next defensible escalation after a
   failed retrospective baseline is prospective collection or a genuinely
   independent cohort, not repeated optimisation on this sample.

## Artifacts and isolation

| Location | Contents | Tracking policy |
| --- | --- | --- |
| `configs/lseg_us_midcap_22_1y.toml` | Source, universe, dates, queries and collection identity. | Tracked source definition. |
| `Data/collections/lseg_us_midcap_22_1y/raw/` | Licensed LSEG checkpoints and raw manifest. | Local and ignored; never redistribute. |
| `Data/collections/lseg_us_midcap_22_1y/derived/us_midcap_22_1y/` | Canonical headline corpus, screening index and corpus manifest. | Local and ignored; contains licensed material. |
| `Data/collections/lseg_us_midcap_22_1y/derived/headline_scores_finbert4556_vader_baseline_v1_cacheonly_20260718.csv*` | Text-bearing FinBERT/VADER-share-argmax score file and completed cache-only scoring manifest. | Local and ignored; never commit or redistribute. |
| `Data/derived/prices/lseg_us_midcap_22_strategy_adjusted_open.csv*` | Verified split-adjusted-open panel and provenance manifest. | Local generated input. |
| `configs/strategy_research/baselines/lseg_midcap_finbert_vader_v1.toml` | Frozen replication claim, scorer roles, signal, accounting, inference and roots. | Tracked protocol. |
| `Data/derived/strategy_research/baselines/<resolved-run-id>/` | `event_labels.jsonl`, `firm_session_signals.jsonl`, and `attrition.json`; no headline text is copied. | Local and ignored; retains licensed-linked identities and hashes. |
| `results/strategy_research/baselines/<resolved-run-id>/` | `daily_pnl.jsonl`, `session_construction.jsonl`, `metrics.json`, `stock_metrics.json`, `bootstrap.json`, `report.md`, and `manifest.json`. | Local, text-free generated evidence; share only aggregates permitted by the licence. |

Never point the baseline roots at existing Week 6, event-study or historical
LLM strategy directories. A completed stage is immutable; use a new run ID for
a changed input or protocol.

## Required reporting

Report the primary FinBERT result first and VADER share-argmax as the fixed comparator. At a
minimum retain:

- eligible and excluded event counts with reason-specific attrition;
- scorer coverage, active sessions, supported stocks and long/short counts;
- gross and net return, transaction costs, turnover, annualised Sharpe and
  maximum drawdown;
- two-sided construction counts/exposures and stock-level contributions;
- paired contiguous-block bootstrap intervals for FinBERT versus zero-return
  cash and FinBERT versus the fixed VADER-share-argmax comparator;
- exact config, corpus, score, price and output hashes; and
- the retrospective status, deviations from the papers and all limitations.

## Limitations

- The one-year 22-stock panel is selected and previously explored. It can have
  coverage, survivorship and cohort-selection bias, low power and unstable
  annualised statistics.
- Reuters headlines omit article context. Even after single-target screening,
  FinBERT and VADER share-argmax measure generic textual tone rather than the target firm's
  expected return or the novelty/materiality of the event.
- FinBERT's Financial PhraseBank performance is classification evidence, not a
  guarantee of Reuters-domain accuracy or trading predictability. VADER was not
  designed for finance, and share-argmax is not its published compound convention.
- Open-to-open returns are an adaptation, not López-Lira–Tang's open-to-close
  window for overnight news and close-to-close window for intraday news. Split
  adjustments are included; dividends are not.
- Ten basis points per dollar traded is an assumption. Spread, impact, short
  borrow, locate availability, financing, taxes and capacity can make realised
  results worse.
- The cash comparator earns exactly zero; T-bill yield, cash interest and
  financing are not modelled.
- No historical backtest establishes causality or future profitability. Results
  should be positioned as a reproducible comparator for later research designs.

## v2 protocol

The v2 baseline is a second frozen protocol under
[`configs/strategy_research/baselines/lseg_midcap_finbert_vaderc_v2.toml`](../configs/strategy_research/baselines/lseg_midcap_finbert_vaderc_v2.toml),
implemented in the isolated `strategy_research.baseline.v2` package. It exists
because the v1 run surfaced two structural defects — the share-argmax VADER
comparator produced zero negative labels and therefore never traded, and the
one-name-per-side floor allowed 50% single-name weights that dominated the
variance — plus an inference design (37 active portfolio days) with almost no
power. The v1 artifacts are unchanged and remain the reported v1 evidence.

> **Provenance:** the v2 protocol was specified *after* the v1 result on this
> cohort was observed. The config's mandatory
> `specified_after_v1_results = true` records this. Every v2 result on this
> sample is exploratory/diagnostic; the frozen protocol is intended for
> confirmatory reuse on an independent cohort (for example the all-headlines
> Reuters collection), not for another pass over this one.

Changes relative to v1, all predeclared in the config and frozen by the strict
loader:

| Component | v2 rule |
| --- | --- |
| Comparator | Canonical VADER: compound score with the published `+/-0.05` thresholds, from a separate cache-only score artifact (`score-vader-compound`) whose manifest pins the lexicon hash and threshold. |
| Portfolio | At least **two names per side** (the reference paper's rule), capping single-name weight at 25% of NAV; cash otherwise. |
| Primary inference | Firm-session event-level moving-block bootstrap of signed **gross** open-to-open returns (session means, block 5, 2,000 replications, seed `20260721`), instead of the low-power portfolio-day test. |
| Variant arms | `finbert_magnitude` (legs weighted by absolute mean continuous score) and `agreement` (FinBERT direction kept only when canonical VADER agrees and is non-zero) — the agreement arm operationalises the dissertation's cross-model agreement question. |
| Cost reporting | Fixed 5/10/20 bps sensitivity grid alongside the frozen 10 bps headline assumption. |

Workflow (after the v1 corpus, FinBERT/VADER scores, and price panel exist):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_lseg_headline_study.py score-vader-compound \
  --collection-root Data/collections/lseg_us_midcap_22_1y \
  --output Data/collections/lseg_us_midcap_22_1y/derived/headline_scores_vader_compound_v2_cacheonly_20260721.csv

sentiment-bench strategy baseline v2 inspect \
  --config configs/strategy_research/baselines/lseg_midcap_finbert_vaderc_v2.toml
sentiment-bench strategy baseline v2 run \
  --config configs/strategy_research/baselines/lseg_midcap_finbert_vaderc_v2.toml
```

The executed v2 run on this cohort is
`lseg-midcap-finbert-vaderc-v2-299b11fa8e0a`. Headline observations, retained
under the no-retuning rules above: the event-level FinBERT signal is positive
but inconclusive (+0.245% mean signed gross session return, 95% CI
[-0.090%, 0.616%] over 216 evaluation firm-sessions); canonical VADER is
*adversely* signed (-0.392%, CI [-0.781%, -0.010%], though that interval barely
excludes zero and would not survive a multiple-comparisons correction across
the seven predeclared tests); and agreement-conditioning does not improve on
unconditional FinBERT. The two-name minimum leaves only 5 active portfolio
days in the evaluation period on 22 stocks, confirming that this universe is
too narrow for the paper-style portfolio and that breadth, not further rule
changes, is the binding constraint.

For implementation and accounting details shared with the broader historical
pipeline, see the [historical strategy-research guide](strategy_research_pipeline.md).
