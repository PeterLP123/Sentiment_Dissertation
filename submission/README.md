# Final dissertation submission

[Read the dissertation PDF](news-sentiment-beyond-mean/Peter-Prendergast-COMP0077-Dissertation.pdf) · [Evidence map](news-sentiment-beyond-mean/docs/evidence_map.md) · [Reproduce from the project root](../docs/reproducing_the_submission.md)

`news-sentiment-beyond-mean/` is a licence-safe snapshot based on clean dissertation repository commit `59577da111f69f1d7678b922d3d753208bfba1ef`, copied on 1 September 2026. On 8 September 2026, the assessment identifier was removed from the title-page source and the named PDF for portfolio use. The research content is unchanged. The current PDF's SHA-256 is recorded in [`source_manifest.json`](source_manifest.json).

The snapshot contains the final LaTeX manuscript, licence-safe aggregate results, frozen specifications, selected notebooks, supporting code, tests and reproducibility documentation. No experiment was rerun during the transfer. Raw Reuters, LSEG and FNSPID text, licensed identifiers, model caches, API responses, credentials, databases, Parquet panels and JSONL scoring logs were not copied.

Keep this package separate from `final_experiments/`. The historical workspace has different in-progress studies numbered 85 and 86, so merging by notebook number would damage provenance.

The project-root commands `make setup`, `make validate` and `make manuscript`
select this package's locked environment automatically. Inside the package,
the original Makefile commands are also preserved. Notebook 75 is an audit-only
view of the one-time testing-period execution and must not be rerun.

## Integration record

The September 2026 project tidy makes this package the main research entry point
without merging it into the earlier experiment tree. All 386 source-controlled
files from the final repository are included, plus the named submission PDF.
The title page carries the documented privacy amendment. The source Git blob is
authoritative for LF line endings in `manuscript/ormsv080.bst`.

[`source_manifest.json`](source_manifest.json) records the source commit,
amendment and SHA-256 of every included file. The root
[`check_project.py`](../scripts/check_project.py) verifies this inventory,
Git visibility and the public project file boundary before the final package's
own checks run. Validation regenerates aggregate manuscript artifacts in a
temporary directory; it does not rerun the scientific experiments.

## Verification on 7 September 2026

The integrated project was checked in a separate temporary Git checkout
containing only the proposed public files, with Git-normalized line endings
and newly installed locked environments.

| Check | Result |
| --- | --- |
| Submission inventory | All 387 files match their recorded SHA-256 values. |
| Final-study validation | 74 analytical tests, manuscript-reference/prose checks and lint pass. |
| Figure and table reproduction | All 30 artifacts, including 20 figure files, reproduce byte for byte. |
| Benchmark and integration checks | 18 focused tests and lint pass; the 5,947-row public benchmark has no duplicate-label conflicts. |
| Manuscript build | 62 pages; the final LaTeX log has no unresolved references or warnings. |
| Named submission PDF | Matched the assessment copy at integration; the later privacy amendment is recorded above. |
| Presentation | Main README visually inspected; public entry-point links checked against Git-visible files. |

These are local macOS checks. The GitHub Actions workflow is configured separately;
this record does not claim a completed hosted CI run. Scientific notebooks,
model calls and licensed-data computations were not rerun during the integration.
