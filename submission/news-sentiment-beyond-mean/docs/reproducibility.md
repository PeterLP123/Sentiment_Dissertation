# Reproducibility guide

## Two reproducibility levels

This repository supports two distinct checks:

1. **Evidence audit:** available immediately from aggregate CSV/JSON results, manifests, frozen specifications, figures and the LaTeX manuscript.
2. **Full computational replay:** requires authorised local FNSPID and LSEG inputs plus several large staged panels that cannot be committed.

Do not confuse the absence of licensed inputs on GitHub with absent provenance. The committed manifests retain historical input hashes and the original repository remains the private data archive.

## Environment

Use the locked Python 3.12 environment:

```bash
uv sync --extra dev --locked
source .venv/bin/activate
make validate
```

The committed `uv.lock` pins the numerical and figure toolchain. Validation
regenerates all 30 manuscript artifacts with a fresh Matplotlib cache and
requires byte-identical output, including all 20 PNG/PDF figure files.

The selected helper tests cover aggregation, rank regression, return-leg identities, rolling-beta timing, turnover accounting, fixed-share portfolio logic, panel/session joins, risk overlays, thresholds and volatility targeting.

## Input placement

Keep raw and staged inputs outside Git. Either place them under `data/` / `Data/` using the source-compatible paths expected by a notebook, or symlink authorised local files into those locations. The most common dependencies are:

- `experiments/generated/01_panel/fnspid_firm_day_panel.parquet`;
- `experiments/generated/03_aggregation/firm_day_aggregators.parquet`;
- staged HAR daily paths from Notebooks 21, 24 and 25;
- staged LSEG FinBERT firm-open aggregates from the historical joined-long-window analysis;
- LSEG sector-33 price CSVs under `Data/derived/prices/`.

Each notebook states its exact input paths near the first executable cell. Because the clean repository intentionally omits upstream data-engineering notebooks, prepare these inputs from the original private archive or a separately verified input bundle.

## Execution order

Run only the chain needed for the claim being refreshed.

Primary statistical chain:

```text
01 staged panel -> 03 aggregation -> 71 conditional model
                                  -> 76 double sort
                                  -> 79 price-path controls
                                  -> 85 outcome/label sensitivities
                                  -> 86 outcome-leg decomposition
                                  -> 87 external-review robustness pack
                                  -> 77 MDE -> 78 provenance tables
```

Risk chain:

```text
03 aggregation -> 20 risk overlay -> 21 HAR base -> 24 hysteresis
                                               -> 25 walk-forward -> 45 schedule-shift diagnostic
LSEG staged inputs -> 72 recent transfer -> 73 backward transfer -> 74 diagnostics
```

Economic-value chain:

```text
staged FNSPID/LSEG daily paths -> 80/81 aggregate searches
LSEG firm-level inputs        -> 82 firm loss-control search
```

Notebook 75 was executed exactly once in the source repository. The exact
executed notebook and source, plus the licence-safe aggregate snapshot, are
committed under `experiments/results/75_fnspid_evaluation_beyond_mean_replication/`.
The notebook in `experiments/notebooks/` audits that promotion and does not
reopen the testing-period study. Do not rerun, retune or extend the testing-period
estimands.

Notebooks 86 and 87 are authorised post-review diagnostics. They may be rerun
only against the frozen inputs and write to `experiments/generated/`. Each
runner verifies that Notebook 75's hash is unchanged and does not execute it.
The promoted aggregate outputs were reproduced byte for byte in independent
temporary directories.

## Safe reruns

All ported notebooks write to `experiments/generated/` and manuscript-ready graphics to `manuscript/figures/`. Before replacing a committed result snapshot:

1. compare the new output with `experiments/results/`;
2. verify sample counts, date blocks, seed, costs and inference settings;
3. explain any difference in a frozen-spec amendment;
4. update the evidence map and manuscript together;
5. copy only aggregate, licence-safe files into `experiments/results/`;
6. run `make validate`.

## Determinism and inference

- preserve random seeds from each frozen specification;
- use the recorded five-session HAC or block length unless an amendment is declared before inspection;
- keep transaction costs per side and final liquidation in the accounting;
- keep all multiplicity-family members, including nulls;
- never use the testing period to choose thresholds, horizons or prompts.

## Manuscript

Install a TeX distribution with `latexmk`, then run:

```bash
make manuscript
```

The output PDF is deliberately ignored. Figures and LaTeX tables are tracked because they contain only aggregate evidence.
