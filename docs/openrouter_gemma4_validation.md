# OpenRouter Gemma 4 validation gate

This gate evaluates `google/gemma-4-26b-a4b-it` on a frozen 1,000-row sample of
the public `financial_sentiment_v2.csv` benchmark before it is considered as an
additional LLM scorer. It does **not** send LSEG headline text to OpenRouter.
The LSEG corpus is licensed local data; external processing requires a separate
licence decision.

## Frozen contract

- Prompt: `investor_headline_soft_label_v1`, hash `81596538d99b29b8`.
- Sample: first 1,000 primary rows in SHA-256(sentence) order. Labels do not
  enter selection.
- Model: `google/gemma-4-26b-a4b-it`.
- Provider: DeepInfra only (`deepinfra` provider slug); provider fallback disabled.
- Privacy: zero-data-retention required and provider data collection denied.
- Decoding: temperature 0, reasoning effort `none`, `max_tokens=128`.
- Output: strict JSON schema with exactly `positive`, `negative`, and `neutral`.
- Metrics: accuracy, balanced accuracy, MCC, macro/weighted F1, class metrics,
  confusion matrix, multiclass Brier score, 10-bin ECE, multiclass log loss,
  and seeded percentile intervals for accuracy and macro-F1. Log loss clips the
  true-class probability to `[1e-15, 1]` only for the logarithm.
- Format gate: at least 99% valid probability objects among provider completions.
- Coverage gate: a successful completion for every selected row. Transport and
  rate-limit failures are reported separately from model-format failures.

OpenRouter currently lists DeepInfra at $0.07 per million input tokens and
$0.34 per million output tokens for this model. The request includes those
values as a hard routing price ceiling, so a price increase fails closed rather
than silently changing the experiment.

The routing, privacy, and output controls follow OpenRouter's official
[provider-routing](https://openrouter.ai/docs/guides/routing/provider-selection),
[zero-data-retention](https://openrouter.ai/docs/guides/features/zdr), and
[structured-output](https://openrouter.ai/docs/guides/features/structured-outputs)
contracts.

## Prepare without spending

From the repository root:

```bash
uv run python scripts/run_openrouter_gemma4_validation.py
```

This writes the frozen manifest under the ignored
`final_experiments/outputs/12_lseg_44_labelling/openrouter_gemma4_validation/`
directory and sends no API requests.

## Run or resume the paid 1,000-row gate

Set `OPENROUTER_API_KEY` in the process environment or the repository `.env`,
then run:

```bash
uv run python scripts/run_openrouter_gemma4_validation.py --confirm-paid
```

Responses are appended one at a time to a config-bound JSONL checkpoint. The
same command resumes missing rows and rejects a checkpoint produced by any
different dataset, sample, prompt, model, provider, or request contract. Do not
change concurrency while interpreting timing as a like-for-like measurement.

Failed API rows are not silently replaced. After inspecting the recorded error,
append an explicit retry at lower concurrency with:

```bash
uv run python scripts/run_openrouter_gemma4_validation.py \
  --concurrency 10 --retry-api-errors --confirm-paid
```

Both the original failure and the later attempt remain in `responses.jsonl`;
metrics use each row's latest attempt, while cost and token accounting use all
attempt records.

The API receives only the headline/sentence. Hidden labels remain local. Raw
LSEG/FNSPID text, model responses, and checkpoints stay ignored; only aggregate,
licence-safe results may later be promoted.

## Completed result: 2026-08-04

The frozen gate completed on all 1,000 selected public rows. The first
concurrency-25 pass produced 924 successful DeepInfra completions and 76
recorded upstream `429 engine_overloaded` failures. One explicit
concurrency-10 retry appended 76 new attempts; no original record was removed.
The final checkpoint therefore contains 1,076 attempt records for 1,000 unique
rows.

| Measure | Result |
| --- | ---: |
| Coverage | 1,000 / 1,000 |
| Correct model/provider route | 1,000 / 1,000 Gemma 4 26B / DeepInfra |
| Valid strict probability JSON among completions | 100.0% |
| Accuracy | 0.808 (95% bootstrap CI 0.784–0.832) |
| Macro-F1 | 0.813 (95% bootstrap CI 0.789–0.837) |
| Balanced accuracy | 0.822 |
| MCC | 0.690 |
| Multiclass Brier score | 0.274 |
| 10-bin ECE | 0.0445 |
| Multiclass log loss | 0.821 |
| Recorded OpenRouter usage cost | $0.03786 |

On these exact 1,000 rows, the existing pinned FinBERT predictions score 0.797
accuracy and 0.797 macro-F1. Gemma 4's paired differences are +0.011 accuracy
(95% row-bootstrap CI −0.019 to +0.041) and +0.0167 macro-F1 (−0.0128 to
+0.0467). The exact McNemar p-value is 0.505 (118 Gemma-only correct versus 107
FinBERT-only correct). This sample therefore shows comparable performance, not
evidence that Gemma 4 is superior. The earlier local Gemma 4 12B result used a
different 90-row balanced sample and a hard-label prompt, so it is not treated
as a same-design comparison. The paired bootstrap and exact McNemar calculation
are implemented in `paired_classification_comparison` beside the validation
runner.

The ignored evidence is at
`final_experiments/outputs/12_lseg_44_labelling/openrouter_gemma4_validation/`.
Its final manifest records config hash
`d84e80434d0ca3db57e236602577745cca4f19e65333d0cf56e8204a9c75ed50`.
