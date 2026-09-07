# Contributing

This repository preserves a completed MSc dissertation alongside the larger research
workspace from which it was produced. Contributions that improve reproducibility,
documentation, tests, or clearly identified historical tooling are welcome. Changes
must preserve the evidential record.

## Before opening a change

1. Read the [project README](README.md), the
   [reproduction guide](docs/reproducing_the_submission.md), and the
   [data and rights policy](docs/data_and_rights.md).
2. Keep the final submission under `submission/news-sentiment-beyond-mean/`
   self-contained. Do not renumber its notebooks or silently replace its committed
   aggregate evidence.
3. Treat frozen specifications, manifests, null results, invalidation records, and
   the named submission PDF as provenance. Explain any necessary correction rather
   than rewriting history.
4. Never add credentials, raw Reuters/LSEG/FNSPID text, provider responses, local
   databases, model checkpoints, or generated row-level panels.

## Development checks

Install and validate the final submission:

```bash
make setup
make validate
```

The historical benchmark is a separate Python project and environment:

```bash
make benchmark-setup
make benchmark-check
```

`benchmark-check` validates the public dataset, runs Ruff across `src/`, `tests/`
and `scripts/`, runs mypy on the library, and executes the full root pytest suite.
The setup includes the locked plotting extra for figure tests. Optional model
integrations skip when their dependencies are absent. A historical LSEG manifest
test skips when its licensed local inputs are absent; present inputs still undergo
the original hash checks. Pytest lists skip reasons in its summary. Model weights
and licensed data are not needed for the public suite. Use `make benchmark-test`,
`make benchmark-lint`, or `make benchmark-types` to run one gate during development.

Run focused tests first for the code you change. A manuscript rebuild requires
`latexmk` and can be checked with `make manuscript`; the preserved named submission
PDF is not overwritten by that command.

## Pull requests

Keep changes focused and describe:

- the problem and resulting behaviour;
- the files and evidence affected;
- the commands run and their outcome;
- whether outputs are generated, source-controlled, or require licensed local data;
- any change to an assumption, sample, split, seed, estimand, or interpretation.

Do not present an exploratory result as confirmatory, a local validation as a full
data replay, or publicly inspectable material as permission to redistribute it.
