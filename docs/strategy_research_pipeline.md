# Historical Strategy-Research Pipeline

This is a separate research pipeline for translating point-in-time, target-specific LSEG news sentiment into an auditable daily portfolio. It does not change the dissertation research question, scorer roster, event-study design, or existing trading outputs. Results are research backtests, not deployable alpha or causal evidence.

## Architecture and boundaries

```text
verified canonical LSEG corpus
  -> full screened strategy event universe
  -> frozen target-specific scores
  -> stock-level sentiment state
  -> volatility-scaled target weights
  -> one holdings/order/P&L ledger
  -> development-only tuning
  -> chronological evaluation
```

The package lives under `src/sentiment_benchmark/strategy_research/`. Licensed text and scoring records are written under `Data/derived/strategy_research/<resolved-run-id>/`; state, orders, positions, P&L, diagnostics, and reports are written under `results/strategy_research/<resolved-run-id>/`. Formal configs cannot redirect either root into existing research trees. Both roots are ignored by Git.

The implementation reuses only generic infrastructure from the main package: verified corpus loading, existing relevance rules, provider clients, price-row conventions, canonical JSON and SHA-256 helpers, atomic writes, and runtime metadata. It does not consume fitted event-study coefficients or existing event-study, Week 6, fixed-horizon, or rotating-sleeve outputs.

## Event identity and availability

The strategy event is the earliest relevance-passing revision for each `story_family_id x symbol`. A multi-company story produces one target-specific event per eligible symbol. Later revisions never replace earlier text. The semantic event ID is independent of the text hash, so a content change invalidates the artifact without silently redefining the event.

`version_created` must already be timezone-aware in the canonical corpus. The strategy layer rejects naive or invalid timestamps. It adds the configured processing buffer, 15 minutes by default:

```text
available_at_utc = version_created_utc + processing_buffer
```

The eligible execution session is the first XNYS open strictly after `available_at_utc`. This handles pre-open, intraday, after-close, weekend, holiday, and daylight-saving cases without using a date-only proxy. The source adapter records deterministic attrition for relevance, revisions, timestamps, text, scores, sessions, and price coverage.

## Scoring

Two versioned target-company schemes map strict labels to a continuous score in `[-1, 1]`:

| Scheme | Mapping |
| --- | --- |
| `three_class` | negative `-1`, neutral `0`, positive `+1` |
| `five_level` | `very_negative` `-1`, `negative` `-0.5`, `neutral` `0`, `positive` `+0.5`, `very_positive` `+1` |

The prompts judge the supplied text's directional implication for the named company. They do not predict stock returns or use price response. Missing or unexpected labels are excluded, never coerced to neutral.

Cache identity includes the event, target, content hash, provider/model identity, exact resolved endpoint, optional model digest, prompt hash, scheme, request settings, and sample index. The checked-in formal scorer also freezes the completed `clean-llm-full-20260703` competence record from `experiments/manifest.toml`; the evidence file hash and experiment ID are part of run identity. Ollama configurations must declare a digest, which is verified against the server before classification. Formal settings use temperature zero, one sample, and sequential scoring in v1. Hosted scoring is never entered by `strategy run` unless `--allow-paid` is supplied; `--dry-run` never constructs a client or writes files.

## State and action

At each decision session, previous state decays by elapsed trading sessions:

```text
z_decay = z_previous * 2 ** (-elapsed_sessions / half_life_sessions)
u_event = sign(score) * abs(score) ** severity_power
reset = reversal_reset * abs(u_event) if u_event * z_current < 0 else 0
z_next = clip((1 - reset) * z_current + impulse_scale * u_event,
              -state_cap, state_cap)
```

Same-session events are applied sequentially by `(available_at_utc, event_id)` with no extra decay. The variants are cash, three-session last-event hold, additive decay (`reversal_reset = 0`), and decay with reset.

State becomes a bounded action through `tanh(state / scale)`. Stock scales use the configured quantile of non-zero absolute development states, shrink toward the pooled scale with `n / (n + 50)`, and have a positive floor. Validation-fold scales use only that fold's training history; final evaluation scales use development history only. Actions inside the no-trade band become zero.

## Prices, positions, and ledger

Pipeline v1 requires a verified, hash-manifested, split-adjusted open panel. Its manifest must identify the exact symbol set and the loader requires every symbol on every session of the complete XNYS spine, record the adjustment source and provenance, and explicitly set `split_adjusted=true` and `dividend_adjusted=false`. Formal runs require an exact match to the canonical 33-company universe and enough pre-decision warm-up for every symbol. LSEG returns are price returns because dividends are not back-adjusted. A future close-based convention would require a separate pipeline version and is never a fallback.

Trailing volatility at session `t` uses open-to-open returns ending no later than `t-1`, defaults to a 20-session window with 15 observations, and uses a fixed `0.005` floor. Missing history makes a symbol ineligible; future values are never backfilled.

Raw weights are `action / max(volatility, floor)`. Projection is deterministic: normalize gross to at most 1.0, cap each name at 5%, and reduce only the overweight side when net exposure exceeds `+/-20%`. It preserves every non-zero signal's sign and leaves unused capacity in cash.

There is one position per symbol. Events change state rather than opening overlapping lots. Targets filled at open `t` earn the open-`t` to open-`t+1` return. Current holdings are revalued before trading, turnover is the absolute target-minus-current change, and costs default to 10 basis points per side. Final positions are explicitly liquidated.

## Tuning and evaluation

The reset variant uses the predeclared 45-candidate grid:

- half-life: `1, 2, 3, 5, 10` sessions;
- state-scale quantile: `0.50, 0.75, 0.90`;
- no-trade band: `0.00, 0.10, 0.20`.

Severity, reset, impulse scale, and state cap stay fixed. Expanding folds begin with at least 60 training sessions. The runner targets three 20-session validation folds; if development history is shorter, it constructs the largest deterministic two-or-three-fold arrangement with equal contiguous windows of at least 10 sessions. Fewer than two valid folds blocks formal tuning.

Candidates are ranked on after-cost returns by:

```text
median(fold Sharpe) - 0.5 * IQR(fold Sharpe)
```

Every candidate and fold is retained. Undefined results or hard-guardrail failures are rejected. Ties use lower dispersion, lower turnover, lower complexity, then the stable parameter tuple. The selected configuration is hashed before one chronological evaluation. The evaluation block is not described as pristine because the period has had prior exploratory exposure.

Paired strategy comparisons resample the same daily net-return dates with deterministic contiguous blocks. Defaults are block length 5, 2,000 replications, and seed `20260715`.

## CLI

```bash
sentiment-bench strategy run --config configs/strategy_research/smoke.toml --dry-run
sentiment-bench strategy run --config configs/strategy_research/smoke.toml
sentiment-bench strategy run --config configs/strategy_research/lseg_llm_decay_v1.toml --dry-run

sentiment-bench strategy build-events --config <config>
sentiment-bench strategy score --config <config>
sentiment-bench strategy build-state --config <config>
sentiment-bench strategy tune --config <config>
sentiment-bench strategy backtest --config <config>
sentiment-bench strategy report --config <config>
sentiment-bench strategy paper
```

For a local Ollama canary, first freeze the formal evaluation boundary, endpoint,
model tag, and digest. Then bound the number of new calls while retaining the
same resumable cache used by the eventual full run:

```bash
sentiment-bench strategy score \
  --config <local-ollama-config> \
  --max-new-scores 200
```

Bounded scoring is Ollama-only and selects events strictly before
`evaluation_start`. It leaves the score stage `in_progress` and does not create
the immutable `scores.jsonl` until every event has a successful cached score.
Repeating the command reuses prior successes and makes at most the requested
number of additional calls. The configured Ollama digest is verified before any
classification request.

Build the strategy price artifact from a completed, hash-verified LSEG export:

```bash
sentiment-bench fetch-lseg-prices \
  --config configs/lseg_us_sector_33_6m.toml \
  --ric-overrides configs/lseg_us_sector_33_strategy_price_rics.toml \
  --start 2025-11-13 --end 2026-07-01 \
  --output Data/derived/prices/lseg_us_sector_33_strategy_source_v2.csv

python scripts/build_strategy_price_panel.py \
  --source-panel Data/derived/prices/lseg_us_sector_33_strategy_source_v2.csv \
  --source-manifest Data/derived/prices/lseg_us_sector_33_strategy_source_v2.manifest.json \
  --output-panel Data/derived/prices/lseg_us_sector_33_strategy_adjusted_open.csv \
  --output-manifest Data/derived/prices/lseg_us_sector_33_strategy_adjusted_open.manifest.json
```

The converter refuses incomplete symbol-session grids, hash mismatches, ambiguous
adjustment provenance, or overwrites. If adding the price manifest changes the
overall run identity after scoring, the complete price-independent cache can be
verified and imported without another model call:

```bash
sentiment-bench strategy score --config <config> \
  --cache-from Data/derived/strategy_research/<prior-run>/scores/cache
```

The main LSEG configuration intentionally remains blocked until it has an explicit evaluation boundary, a completed canonical corpus, and a sufficiently long verified adjusted-open price panel. Dry-run reports these blockers and the resolved identity before any scoring.

## Artifacts and replay

The resolved run name combines a logical ID and a prefix of an identity hash covering source, event rule, processing buffer, scoring identity, price manifest, calendar, split, grid, state, risk, costs, inference, and schema version. A changed research choice therefore creates a different directory.

Each stage records its input identity, status, row counts, exclusions, output hashes, command, and runtime context. Matching completed stages are validated and reused. A mismatch or attempted overwrite fails closed. Scoring can resume from its cache while incomplete, then materializes a deterministically sorted immutable score file.

## Prospective paper mode

The `strategy paper` command is deliberately gated until the formal historical v1 run passes acceptance; today it exits without writing. The implemented journal primitive is append-only and never routes broker orders, but it is not exposed for collection yet. When activated after acceptance, each observation will record publication, retrieval, first-seen, and scoring times, source URL, target, content hash, proposed weights, and paper orders before later returns are known. Prospective availability is `max(first_seen, scoring_completed_at)`. Historical rows and proposed orders cannot be rewritten. A future Tavily adapter remains prospective-only unless a genuine point-in-time archive exists.

## Known limitations

- The fixed company universe can contain survivorship bias.
- LSEG adjusted prices account for corrections and capital changes, not dividends.
- Daily prices leave intraday timing and execution uncertainty.
- Short borrow, financing, liquidity, capacity, slippage beyond the fixed cost, and market impact are not modelled.
- Scorers are heterogeneous, not statistically independent; shared lineage and training contamination remain possible.
- PhraseBank competence may not transfer to Reuters target-company news.
- Multiple testing and prior exploratory exposure limit confirmatory interpretation.
- A successful backtest is sample-specific research evidence, not deployable alpha.
