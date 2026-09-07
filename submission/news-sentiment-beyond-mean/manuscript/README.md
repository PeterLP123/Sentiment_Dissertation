# Manuscript

`main.tex` is the active UCL dissertation on training-period evidence, frozen
temporal non-replication and economic limits in firm-day sentiment
aggregation. `figures/`, `tables/` and `artifacts/` contain only aggregate,
licence-safe evidence selected for the main text or appendix.

Compile from the repository root with:

```bash
make manuscript
```

The main chapters contain 10,258 words. The build regenerates the derived figures and numbers, runs focused prose checks, and compiles the PDF. See `docs/word_count.md` for the counting convention and target note, and `manuscript/artifacts/validation_report.md` for the arithmetic checks.
