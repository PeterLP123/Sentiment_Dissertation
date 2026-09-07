# Experiments

`notebooks/` contains the selected scientific chain. `lib/` contains only helpers used by that chain. `specs/` records frozen choices and governance amendments. `results/` is the aggregate evidence snapshot used by the writing. `generated/` is ignored and receives rerun outputs.

Start Jupyter from the repository root so notebook-relative paths resolve consistently:

```bash
jupyter lab
```

The numbering is inherited from the source archive to preserve traceability. See `docs/evidence_map.md` for the intended order and the reason each notebook is present.

Notebook 75 is a special case. It was opened once in the source archive under a
frozen parent specification and outcome-blind amendment. The exact executed
notebook is preserved with its promoted results; the curated notebook is an
audit-only view and must not be used for a second evaluation run.
