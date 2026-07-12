# Dissertation LaTeX Workspace

Research/code plan: [dissertation execution plan](../docs/dissertation_execution_plan.md)

## Compile and inspect

```bash
cd dissertation
make pdf                  # full build: pdflatex + biber + makeglossaries
make assets               # refresh generated tables from registered runs
make count                # words per chapter and total vs 10,000 target
make clean                # remove auxiliary files, retaining the PDF
```

`latexmk -pdf main.tex` also works directly. Build outputs are ignored; `.tex`, `.bib`, and registered generated assets are source-controlled.

## Refocused writing contract — 12 July 2026

Working title: **Beyond the Mean: Cross-Model Agreement and Post-News Return Resolution**

Primary question:

> Does cross-model agreement—assessed against graded human annotation agreement—add out-of-sample information about firm-level post-news abnormal returns beyond mean sentiment and the initial price reaction?

The former L1–L4 chapter programme is no longer the critical path. The core document contains:

1. benchmark competence and PhraseBank construct validation;
2. one timestamp-valid LSEG event study;
3. a held-out strongest-scorer vs mean-only vs mean-plus-agreement comparison;
4. one economic translation only if the nice-to-have gate opens.

G-theory/reliability sizing, futures, prompt/self-consistency factorials, training-cutoff, heterogeneity, and writer–scorer Echo remain optional/future work.

## Target seven-chapter structure

| Target chapter | Words | Current source disposition |
| --- | ---: | --- |
| **1. Introduction** | 800–900 | Reframe `chapters/01-introduction.tex` around one RQ and bounded contributions. |
| **2. Literature Review** | 1,700–1,900 | Refocus `chapters/02-literature-review.tex` on sentiment measurement, human/model disagreement, closest finance studies, and event-study methods. |
| **3. Data and Provenance** | 1,200–1,400 | Convert `chapters/03-data-pipeline.tex` to the actual benchmark, LSEG, price, timing, roster, split and attrition evidence. |
| **4. Methodology** | 1,700–1,900 | Consolidate agreement validation, event construction, abnormal returns, nested OOS comparison and inference. Mine valid methods from the current scaffold. |
| **5. Results** | 2,200–2,500 | Use construct-validity and event-study assets; `chapters/05-layer2-consensus.tex` is the main source but must lose crowding claims and illustrative values. |
| **6. Discussion and Limitations** | 1,200–1,400 | Reframe `chapters/08-discussion.tex` around measurement vs market value and bounded interpretation. |
| **7. Conclusion** | 400–600 | Reframe `chapters/09-conclusion.tex` around the primary and two supporting questions. |

Target main-text total: **9,800–10,100 words**. Abstract: 250–350 words, written last.

### Existing material to preserve but demote

- `chapters/04-layer1-baseline.tex`: mine valid exploratory results and limitations for a short context/appendix subsection; its mid-August live result is not a core dependency.
- `chapters/06-layer3-reliability.tex`: optional G-theory/reliability extension; exclude from the core build after useful material is preserved.
- `chapters/07-layer4-echo.tex`: optional future-work source; exclude from the core build.

Do not delete or rename source material until its useful content has been moved and the change is reviewable in Git.

## Evidence inventory

Required main-text assets:

1. clean-benchmark model table;
2. agreement-by-PhraseBank-tier figure;
3. construct-validity/error table;
4. LSEG cohort/attrition table;
5. held-out nested-model/event-study table;
6. return-relationship figure;
7. limitations/provenance summary.

Optional: one fixed after-cost economic comparison. Alternative horizons, agreement definitions, leave-one-model-out analysis, prompt/configuration and manifests belong in appendices.

## Results tables: never hand-type final numbers

`make assets` (or `python ../scripts/refresh_dissertation_assets.py`) mirrors exported LaTeX tables from experiments registered in `experiments/manifest.toml` into `generated/<experiment-id>/`, with provenance headers.

```latex
\input{generated/phrasebank-agreement-core-v1/tier_summary}
```

Check the header before inclusion. A hand-typed value can drift; a generated registered table remains traceable. Illustrative placeholder tables must be deleted before any supervisor draft.

## Writing checkpoints

| Date | Deliverable |
| --- | --- |
| **14 Jul** | New title/RQ and seven-chapter core structure compile; L3/L4 excluded from the core contract |
| **18 Jul** | Data and fixed Methods first drafts; 3,000–3,500 substantive words |
| **23 Jul** | Literature and Data complete; 4,500–5,000 words; agreement assets inserted |
| **27 Jul** | Methodology complete; Results shell populated; 6,000–6,500 words |
| **31 Jul** | Results synchronized to frozen artifacts; 7,500–8,000 words |
| **7 Aug** | 9,800–10,100 words plus abstract, project summary, appendices, bibliography and AI-use log |
| **8 Aug** | Full document QA and visual PDF inspection |
| **9–10 Aug** | Complete supervisor draft |

Dissertation prose is written by Peter under the supervisor's no-AI-prose rule. Planning notes and generated evidence do not count toward the prose milestones.

## Writing/build habits

- Use one sentence per source line for reviewable diffs.
- Use `\cref{...}` rather than hard-coded figure/chapter numbers.
- Include result tables/figures from registered generated assets.
- Verify bibliography metadata when a source enters the text.
- Record prompts, model IDs, hashes, run IDs and AI-tool use in the appendix/log.
- Compile frequently; visually inspect the PDF rather than trusting source checks alone.

## Before the supervisor draft

- [ ] Exact RQ is identical in title/abstract/introduction/methods/conclusion.
- [ ] Every bracketed drafting prompt, visible `\todo`, and invented example value is removed.
- [ ] Every result traces to a registered run and generated asset.
- [ ] Every figure/table is discussed, captioned and cited.
- [ ] Nulls, attrition, intervals, limitations and deviations are reported.
- [ ] `make assets`, `make count`, and `make pdf` pass without undefined citations/references.
- [ ] PDF is inspected page by page.

## Before submission

- [ ] Verify current Moodle deadline/time and required uploads.
- [ ] Confirm programme/degree/department, prescribed title-page wording and distribution statement.
- [ ] Complete the declaration, AI-use log and current AI-policy requirements.
- [ ] Include the required project summary at the end.
- [ ] Verify every bibliography entry against the actual source.
- [ ] Spell-check and visually inspect the final PDF.
- [ ] Upload by the 31 August operational target and verify the receipt; 1 September is contingency only.
