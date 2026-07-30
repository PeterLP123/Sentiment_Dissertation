# CLAUDE.md

`AGENTS.md` is the single source of truth for agent instructions in this repository.
Read it first. This file exists so Claude-based tools pick up the same rules, and it
repeats only the points that most change how you should behave right now.

## Current phase: Final Experiments (opened 2026-07-30)

All closing work lives in `final_experiments/`. The live plan and checklist is
`final_experiments/plan.html` - read it before starting anything, and update it as items
land.

### The research question is not decided

Do not hard-code a single RQ into code, docs, or dissertation text. Candidate framings are
enumerated in the plan. The "Beyond the Mean" cross-model-agreement RQ in
`docs/research_protocol.md` is the standing default, not a decision. Record which
candidate RQ each result speaks to.

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

### Six workstreams

1. Data consolidation into one firm-day news panel (larger universe, longer window,
   high-impact stories).
2. Story filtering and weighting: publisher weight; breaking news vs repetition vs routine
   scheduled reporting; EDA of the within-company-day score distribution.
3. Aggregation rules: mean of hard labels vs mean score vs median vs trimmed mean vs
   negative share vs dispersion vs strongest event vs attention weighting vs decayed state.
4. Sentiment surprise, net of the market return and of the general sentiment level.
5. Story-type conditioning using the existing 12-type event taxonomy.
6. Earnings-date effect - needs a scheduled-earnings calendar that does not exist yet.

### Scope limits

- **De-prioritise VaR / ES tail risk.** That factorial is finished and registered. Cite it,
  do not extend it.
- Neural nets only for learning a trade/no-trade threshold (possibly per stock or sector)
  in place of a fixed no-trade band. Development-only fit, compared against the fixed band.

### Non-negotiable even at speed

Chronological splits frozen before scoring. Date-clustered or block-bootstrap inference.
Costs and break-even reported. Licensed LSEG/FNSPID text stays local and gitignored. Nulls
are results. `final_experiments/outputs/` is gitignored.
