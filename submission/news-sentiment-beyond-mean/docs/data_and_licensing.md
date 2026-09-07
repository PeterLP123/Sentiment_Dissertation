# Data and licensing

## What may be committed

- source code and notebook logic;
- frozen specifications, seeds and parameter choices;
- aggregate coefficient, portfolio, uncertainty, power and gate tables;
- aggregate figures and LaTeX tables;
- manifests and cryptographic hashes that do not reveal licensed text;
- documentation of sample sizes, dates and transformations;
- the public Ken French daily three-factor file used by the close-to-close residual check.

## What must remain local

- Reuters/LSEG headlines, bodies, story identifiers and revisions;
- FNSPID article text and large scored checkpoints;
- OpenRouter prompts containing licensed news, model responses and attempt logs;
- SQLite databases, Parquet panels, JSONL event/scoring files and model caches;
- credentials, API keys and provider metadata that can identify an account.

The `.gitignore` and repository validator enforce common cases, but they are not a substitute for inspecting staged changes.

## Derived-data rule

Never edit a source dataset in place. Create a derived file, record the source path and hash, list cleaning and mapping steps, and state the date split and seed. Aggregate outputs promoted to Git must be insufficient to reconstruct the licensed news text.

## External model experiments

The August 2026 OpenRouter experiments were conducted under a user-authorised scope with DeepInfra-only routing and no-retention controls. This clean repository includes only the prompt configuration, frozen governance specifications and aggregate selection results. It excludes the selected Reuters headlines, company context, earlier same-company headlines, raw responses and provider attempt logs.

That operational record is not proof of third-party permission. No Reuters/LSEG
licence instrument, data-processing agreement or recorded ethics determination
authorising transfer of the licensed headlines through OpenRouter to DeepInfra
is present in the research archive. The data-use basis must be confirmed before
submission or any further external processing. Do not describe no-retention
routing or aggregate-only publication as resolving the underlying licence or
data-governance question.

## Pre-commit check

Run:

```bash
python scripts/validate_repository.py
git diff --cached --name-only
git diff --cached
```

If a file contains readable news text or row-level licensed identifiers, do not commit it even if an ignore rule missed it.
