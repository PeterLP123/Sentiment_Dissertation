# Reproducing the final submission

The canonical final work is the self-contained
[`submission/news-sentiment-beyond-mean/`](../submission/news-sentiment-beyond-mean/README.md)
snapshot. The surrounding repository is the broader historical workspace. These are
separate locked Python projects and should not share an environment.

Prerequisites: Python 3.12, `uv`, and Make. The locked environments install
the Python dependencies; TeX is needed only for a manuscript build.

## Validate the public snapshot

From the repository root, install the final package and run its repository, prose,
test, lint, and artifact-reproducibility checks:

```bash
make setup
make validate
```

This clone-only audit uses the committed aggregate results, frozen specifications,
figures, tables, manuscript sources, and public factor fixture. It does not reconstruct
the licensed source corpora or every upstream panel.

The source inventory in [`submission/source_manifest.json`](../submission/source_manifest.json)
pins every final-package file, including the named PDF. Changes to that preserved
snapshot fail the root check until explicitly reviewed and recorded.

### Without Make

From the repository root:

```bash
python3 scripts/check_project.py
cd submission/news-sentiment-beyond-mean
uv sync --locked --extra dev
uv run --locked --extra dev python scripts/validate_repository.py
uv run --locked --extra dev python scripts/check_prose_style.py
uv run --locked --extra dev python -m pytest -q
uv run --locked --extra dev python scripts/check_artifact_reproducibility.py
uv run --locked --extra dev python -m ruff check experiments/lib tests scripts
```

On Windows, use `py -3.12` for the first command if `python3` is unavailable.

To rebuild the manuscript, install a TeX distribution providing `latexmk`, then run:

```bash
make manuscript
```

The rebuild writes
`submission/news-sentiment-beyond-mean/manuscript/main.pdf`. The named
`submission/news-sentiment-beyond-mean/Peter-Prendergast-COMP0077-Dissertation.pdf`
is the preserved final submission and is not overwritten.

## Historical benchmark checks

The root package contains the earlier benchmark, news-collection, CLI, and TUI tooling.
Install and check it separately:

```bash
make benchmark-setup
make benchmark-check
```

These commands validate the public benchmark and focused statistical tests. They do not
run paid model calls, collect news, or execute the historical notebook chain.

## Full computational replay

A full replay requires independently authorised local FNSPID and LSEG inputs, price
exports, scorer checkpoints, and staged panels that cannot be distributed in this
repository. Place or map those inputs according to the first executable cells and
manifests in the relevant notebook. Generated reruns belong under the package's ignored
`experiments/generated/` directory; compare them with the committed aggregate snapshot
before promoting any result.

Some notebooks and manifests preserve `/Users/peterprendergast/...` and UCL
`/cs/student/project_msc/...` strings as historical provenance. Those strings record
where the original computation ran. They are not portable requirements and should be
mapped to an authorised local input mount rather than recreated literally.

Notebook 75 is the immutable audit view of the one-time 2020--2023 testing-period
execution. Do not rerun, retune, or extend it. The later diagnostics verify its recorded
hash and write only to generated paths. See the package's full
[reproducibility guide](../submission/news-sentiment-beyond-mean/docs/reproducibility.md)
for the supported dependency chains, frozen inference choices, and safe-rerun process.

Availability of code, manifests, hashes, or aggregate evidence is not permission to
obtain or process the omitted data. Read [data and rights](data_and_rights.md) before
attempting a licensed replay.
