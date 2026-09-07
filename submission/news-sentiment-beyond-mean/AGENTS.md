# Research repository instructions

This is the clean dissertation repository. Keep it smaller than the source archive.

- Treat `experiments/results/` as an aggregate evidence snapshot. New runs write to `experiments/generated/`.
- Never commit Reuters or FNSPID text, LSEG identifiers that expose licensed records, raw prompts/responses, API credentials, model caches, SQLite databases, Parquet panels or JSONL scoring logs.
- Keep FNSPID and LSEG as separate source regimes. Do not pool them.
- Distinguish statistical information from economic usefulness. Do not turn a small conditional association into a trading or causal claim.
- Preserve nulls, multiplicity corrections, transaction costs, placebo tests and minimum detectable effects.
- Before changing a result used by the manuscript, update its notebook, frozen specification, aggregate result, evidence map and manuscript text together.
- Do not execute the sealed FNSPID evaluation notebook (`75_...`) unless the research decision authorising Gate F1 has been recorded.
- Prefer a focused notebook and a clear figure over new framework code. Add tests only for helpers whose silent failure would corrupt multiple results.
- Run `make validate` before committing.
