# CLAUDE.md

`AGENTS.md` is the single source of truth for agent instructions in this repository.
Read it first. This file exists so Claude-based tools pick up the same rules, and it
repeats only the points that most change how you should behave right now.

## Current phase: Final Experiments (opened 2026-07-30)

All closing work lives in `final_experiments/`. Read `final_experiments/README.md` for
the notebook/data map and `final_experiments/plan.html` for live decisions and checklist
status. `docs/research_protocol.md` holds current scientific guardrails;
`docs/dissertation_execution_plan.md` holds stage order. Dated plans/results are
historical evidence.

### The research question is not decided

Do not hard-code a single RQ into code, docs, or dissertation text. Gate F1 follows the
filtering/distribution EDA and precedes evaluation-return comparison. "Beyond the Mean"
cross-model agreement is the fallback/default, not a decision. Record which candidate RQ
each result speaks to.

### Work like a researcher, not a production engineer

`src/sentiment_benchmark` is a hardened pipeline with 66 test files, frozen TOML configs,
content-addressed manifests and immutable artifacts. **Do not extend that pattern into
`final_experiments/`.**

- Jupyter notebooks (jupytext `# %%` `.py` pairs) with plots and tables, not new CLI
  subcommands.
- **Fewer tests.** Only test a reusable data-processing or metric helper where a silent
  error would corrupt every downstream result. No tests for exploratory analysis,
  plotting, or notebook glue.
- No full-suite runs, no `mypy`, no manifest/hash ceremony unless a result is being
  promoted into the dissertation.
- Rigour comes from stating the estimand, the chronological split, the clustering, and the
  multiplicity family - and from a figure that shows the effect. Not from abstraction.
- New helpers go in `final_experiments/lib/`, importing `sentiment_benchmark` as a library.

### Current state and six workstreams

FNSPID 2011-2023 is the frozen primary spine; LSEG sector-33 + midcap-22 is a non-pooled
robustness arm. Workstream 1 has produced a 715,546-row firm-day panel for 570 priced
symbols and 3,262 sessions. Development ends 2019-12-31; evaluation starts 2020-01-01.
Workstream 2 is next.

1. Data consolidation - complete for the current checkpoint chain.
2. Story filtering and weighting: publisher/novelty/repetition/routine features and
   within-company-day score-distribution EDA. Current FNSPID checkpoint lacks publisher
   and story-family fields; do not invent them.
3. Aggregation rules: hard-label mean, continuous mean, median, trimmed mean, negative
   share, dispersion, strongest event, attention weighting, and decayed state.
4. Sentiment surprise, net of market return, general sentiment level, and firm baseline.
5. Story-type conditioning using the existing 12-type taxonomy.
6. Earnings-date effect using the local LSEG calendar in
   `final_experiments/data/earnings/`; mapping/filter coverage caveats remain.

### Scope limits

- **De-prioritise VaR / ES tail risk.** That factorial is finished and registered. Cite it,
  do not extend it.
- Neural nets only for learning a trade/no-trade threshold (possibly per stock or sector)
  in place of a fixed no-trade band. Development-only fit, compared against the fixed band.

### Non-negotiable even at speed

Freeze downstream fitting/model selection before opening evaluation outcomes. Use
date-clustered or block-bootstrap inference. Report costs and break-even where portfolio
outcomes appear. Licensed LSEG/FNSPID text stays local and gitignored. Nulls are results.
`final_experiments/outputs/` is gitignored.
