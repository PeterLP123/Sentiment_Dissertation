# A Hard Negative-Story Threshold

Clean dissertation repository for two linked questions:

> In the training period, is hard negative-story share conditionally associated with
> the assigned-window return beyond rank-linear mean sentiment and news volume,
> where is that association measured, and does the unchanged baseline persist
> in the frozen 2020--2023 testing period?

The selected FNSPID training-period baseline is -0.00831; recent-price controls give
-0.00914. A post-review decomposition places the association intraday, not
post-close, and a previous-open probe is also strongly negative. Missing
row-level availability times therefore prevent a predictive interpretation.
The frozen 2020--2023 baseline is +0.00583 and the predeclared era contrast is
+0.01414 (95% HAC interval [+0.00250, +0.02579]). Prospective power was 35.4%;
observed precision implies 26.5% ex-post power at the same hurdle. The result is
a bounded temporal non-replication and timing diagnosis, not proof of zero or a
market mechanism. Trading, mapped-session risk diagnostics and prompt tests add adverse economic
and portability evidence.

This repository is the slim research and writing surface. It excludes the source project's abandoned branches, duplicate dashboards, model caches, raw news, licensed text, API responses, and large intermediate panels.

## Repository map

```text
docs/                    Research design, evidence map, writing plan and data policy
experiments/
  notebooks/             Twenty-two selected analysis notebooks
  lib/                   Reusable statistical and portfolio helpers
  specs/                 Frozen specifications and amendments
  results/               Licence-safe aggregate result snapshots
  generated/             Local rerun outputs; ignored by Git
manuscript/              LaTeX dissertation, bibliography, figures and tables
scripts/                 Repository validation
tests/                   Focused tests for high-risk analytical helpers
data/                    Local input mount; documentation only in Git
```

## Start here

1. Read [the research design](docs/research_design.md).
2. Use [the evidence map](docs/evidence_map.md) to trace each claim to a notebook, specification and result.
3. Follow [the reproducibility guide](docs/reproducibility.md) before rerunning anything.
4. Read [the dissertation quality review](docs/dissertation_quality_review.md), then build the manuscript.

## Quick setup

```bash
uv sync --extra dev --locked
source .venv/bin/activate
make validate
```

`uv.lock` is the canonical Python 3.12 environment for manuscript artifacts and
validation. The artifact generator fails closed if the active Python,
Matplotlib, NumPy or pandas version differs from that lock.

Build the manuscript with:

```bash
make manuscript
```

The six numbered chapters contain 10,258 words. The generated
PDF is `manuscript/main.pdf` and is ignored by Git; the LaTeX source, aggregate
figures and derived-number audit are tracked.

The committed `experiments/results/` directory is an immutable, aggregate-only evidence snapshot. Notebook reruns write to ignored `experiments/generated/`; they do not silently overwrite the evidence used by the draft.

## Claim boundary

This work supports a disciplined timing and non-replication finding, not a deployable-alpha claim:

- training-period evidence: negative-story share survives recent-return, volatility and trailing-beta controls on the assigned window;
- timing boundary: the association is intraday and also appears before the assigned open, so the current data do not establish predictive timing;
- frozen temporal result: the baseline coefficient does not replicate in 2020--2023 and differs across eras under the predeclared HAC contrast; this is consistent with pipeline-level instability, not its cause;
- not established: a stable within-day distribution effect beyond one historical training period;
- not supported: close-to-close ranks, FF3 residual ranks, or treating soft negative mass as equivalent to hard share;
- not supported: a cost-covering daily trading rule;
- not established: precise transfer to the shorter LSEG blocks;
- not supported: prompt engineering as a rescue for LSEG economic value;
- bounded: FNSPID squared-downside alignment is contemporaneous because current
  mapped-session news sets the same session's modifier; overall drawdown is
  adverse, and schedule-shift, episode and transfer evidence must be reported
  together.

See [data and licensing](docs/data_and_licensing.md) before adding any input or output file.
