# Dissertation LaTeX Template

## Compile

```bash
cd dissertation
make pdf                  # full build (latexmk: pdflatex + biber + makeglossaries)
make assets               # refresh generated/ tables from registered runs
make count                # words per chapter + total vs the 10,000-word budget
make clean                # remove aux files (keeps the PDF)
```

(`latexmk -pdf main.tex` still works directly; the Makefile just wraps it.)

The included `.latexmkrc` wires in `biber` (bibliography) and
`makeglossaries` (the List of Acronyms) automatically — a plain
`pdflatex` run alone would leave both empty. Build outputs (`main.pdf`,
`*.aux`, etc.) are git-ignored; only the `.tex`/`.bib` sources are tracked.

Or use [Overleaf](https://overleaf.com): zip this folder and upload — it works as-is
(set the compiler to pdfLaTeX and main document to `main.tex`).

## Layout

| Path | Purpose |
|---|---|
| `main.tex` | packages, metadata, document skeleton |
| `frontmatter/` | abstract, declaration, acknowledgements |
| `chapters/01-introduction.tex` | insufficient-statistic claim, RQ1–RQ4 |
| `chapters/02-literature-review.tex` | four shelves: sentiment–prices, extraction, LLM measurement, disagreement + G-theory |
| `chapters/03-data-pipeline.tex` | data, roster, aggregation, leakage, rigor protocol |
| `chapters/04-layer1-baseline.tex` | L1 — threshold strategy (supervisor's baseline) |
| `chapters/05-layer2-consensus.tex` | L2 — consensus → crowding/reversal (headline chapter) |
| `chapters/06-layer3-reliability.tex` | L3 — G-study → shrinkage/sizing |
| `chapters/07-layer4-echo.tex` | L4 — echo experiment (**gated**: delete if M2 gate closed) |
| `chapters/08-discussion.tex` | what the distribution knew; live-deployment gap |
| `chapters/09-conclusion.tex` | RQ-by-RQ answers, future work |
| `chapters/appendix-a.tex` | prompts/config + pre-registration record |
| `references.bib` | bibliography (seeded from `.raw/` + theory shelves — **verify entries**) |
| `figures/` | drop PDF/PNG figures here |

Chapter structure mirrors the chapter map in the wiki's *Beyond the Score Plan*.
Sections 3.1–3.4 can be drafted now from the repo — they don't depend on results.

## Results tables: never hand-type numbers

`make assets` (or `python ../scripts/refresh_dissertation_assets.py`) mirrors the
exported LaTeX tables of every experiment registered in `experiments/manifest.toml`
into `generated/<experiment-id>/`, each file stamped with a provenance header
(experiment id, run ids, commit, dataset sha). Include them instead of transcribing:

```latex
\input{generated/clean-baselines-full-20260628/leaderboard}
```

Re-run the experiment → re-export → `make assets`, and the document updates itself.
Anything hand-typed can silently drift from the run it claims to report; anything
`\input` cannot. `generated/` is tracked in git so the folder still zips straight
into Overleaf. Check the provenance header before including a table — legacy
(corrupted-dataset) experiments are mirrored too, and their headers say so.

## Habits that pay off

- **Compile only the chapter you're writing**: comment out the other
  `\include` lines in `main.tex`.
- **One sentence per line** in the `.tex` source — makes git diffs and
  supervisor feedback line-precise.
- **Never hard-code** "Figure 3.1" / "Chapter 2" — always `\cref{...}`.
- **`\todo{...}`** leaves orange margin notes; add `final` to the
  todonotes options in `main.tex` to hide them all for a clean draft.
- Core rules are recorded in `wiki/sources/Dissertation Instructions.md`:
  about 10,000 words / 50 pages, 40-45 main-body pages, A4, 1.5 spacing,
  recommended 12-point type, prescribed title-page wording, and a project
  summary at the end. Verify the 2017 instructions against the current
  Moodle page for submission and AI/declaration requirements.

## Before submission checklist

- [ ] Replace all `[bracketed placeholders]` and `\todo` notes
- [ ] Verify every `references.bib` entry against the actual paper
- [ ] Title page wording matches programme handbook
- [ ] Programme name and free/restricted distribution statement confirmed
- [ ] Copy of the project summary included at the end
- [ ] Current Moodle submission and AI/declaration requirements checked
- [ ] Every figure/table referenced in text; every RQ answered in Ch. 6
- [ ] Run a spell-check on the PDF, not just the source
- [ ] `latexmk -pdf` finishes with zero warnings about undefined references
