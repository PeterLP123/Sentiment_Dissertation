# A Hard Negative-Story Threshold

*Training-Period Association, Temporal Non-Replication and Economic Limits*

![A Hard Negative-Story Threshold: Training-Period Association, Temporal Non-Replication and Economic Limits](docs/assets/project-banner.svg)

**Does the mix of financial news tell us more than its average sentiment?** An MSc Computational Finance dissertation at UCL, with the submitted manuscript, reproducible aggregate evidence and the research code that led to it.

[Read the dissertation](submission/news-sentiment-beyond-mean/Peter-Prendergast-COMP0077-Dissertation.pdf) · [Explore the evidence](submission/news-sentiment-beyond-mean/docs/evidence_map.md) · [Reproduce the figures](docs/reproducing_the_submission.md) · [Browse the research history](docs/research_archive.md)

## The study

Financial-news pipelines often reduce a day's stories to one average score. This project asks whether the **share of stories classified as negative by FinBERT** carries additional information after controlling for mean sentiment and news volume. It then tests the association's timing, stability across periods, economic value after costs and transfer to a separate news source.

The primary FNSPID panel spans **2011–2023, 570 priced firms and 715,546 news-bearing firm-days**. LSEG/Reuters provides separate transfer checks. The final study contains 22 selected notebooks, frozen specifications, aggregate result snapshots and the complete LaTeX manuscript.

**The negative training-period association did not replicate in 2020–2023.** Post-review diagnostics also find an association before the assigned trading window. Without row-level news availability times, the data cannot establish a predictive signal. This is a result about measurement, timing and temporal instability.

## Results at a glance

| Test | Final evidence | Interpretation |
| --- | --- | --- |
| Training, 2011–2019 | Coefficient **−0.00831**; 95% HAC interval **[−0.01410, −0.00252]** | Conditional association beyond mean sentiment and news count. |
| Frozen testing, 2020–2023 | **+0.00583**; interval **[−0.00426, +0.01593]** | The negative baseline association does not replicate. |
| Testing minus training | **+0.01414**; interval **[+0.00250, +0.02579]** | The predeclared contrast supports a difference across periods. |
| Timing diagnostics | Association measured intraday and before the assigned open | News availability and session assignment limit a predictive reading. |
| Trading costs | Best break-even cost approximately **0.625 bps per side**, against **10 bps** charged | The tested daily trading rules do not cover costs. |

Coefficients describe daily cross-sectional **rank regressions**, not percentage returns. Intervals use five-lag HAC inference. The testing period had limited power and had informed some earlier project design; this is neither proof of a zero effect nor a pristine confirmatory holdout. [Full design, estimates and limitations →](submission/news-sentiment-beyond-mean/docs/research_design.md)

![Annual negative-story-share coefficients and uncertainty: negative training-period mean, positive testing-period mean](submission/news-sentiment-beyond-mean/manuscript/artifacts/fig_coefficient_drift.png)

*Annual estimates from the submitted dissertation. The shaded block is the frozen testing period; horizontal lines show the two period means. The figure is regenerated from the included aggregate evidence during validation.*

## Reproduce the published evidence

For exact artifact reproduction, use **macOS on Apple silicon, Python 3.12, uv and Make**. Run from the repository root:

```bash
make setup
make validate
```

This installs the final study's locked environment, checks the submission inventory and public file boundary, validates manuscript references and prose, runs the analytical helper tests and lint checks, and regenerates **30 manuscript artifacts, including 20 figure files**, for byte-for-byte comparison. It uses the included aggregate evidence; it does not call a model provider or reopen the testing-period experiment.

To build the manuscript, install a TeX distribution with `latexmk` and run:

```bash
make manuscript
```

The build writes `submission/news-sentiment-beyond-mean/manuscript/main.pdf`. The [named submission PDF](submission/news-sentiment-beyond-mean/Peter-Prendergast-COMP0077-Dissertation.pdf) remains the preserved final copy. Full notebook replay requires authorised local inputs; [the reproducibility guide](docs/reproducing_the_submission.md) explains the distinction and direct commands for systems without Make.

## Inside the repository

The final submission is a self-contained Python project. Its paths and environment are preserved so that every claim still points to the exact code and evidence used in the dissertation.

```text
submission/news-sentiment-beyond-mean/    Final dissertation and evidence
├── manuscript/                         LaTeX, bibliography, figures and tables
├── experiments/notebooks/              22 selected analysis notebooks
├── experiments/lib/                    Statistical and portfolio helpers
├── experiments/specs/                  Frozen decisions and amendments
├── experiments/results/                Aggregate evidence used in the manuscript
├── tests/                              Analytical correctness checks
└── uv.lock                             Final study's Python environment

src/sentiment_benchmark/                 Benchmark, collection and scoring library
final_experiments/                      Earlier notebook research and local inputs
Data/                                  Public benchmark; licensed inputs stay local
configs/ · notebooks/ · reports/        Supporting studies and configurations
docs/                                  Project guides and research history
```

| Start with | What you will find |
| --- | --- |
| [Final submission](submission/README.md) | The complete submitted package, source provenance and PDF identity. |
| [Evidence map](submission/news-sentiment-beyond-mean/docs/evidence_map.md) | Claim → notebook → specification → aggregate result. |
| [Research design](submission/news-sentiment-beyond-mean/docs/research_design.md) | Estimands, chronology, timing checks, multiplicity and null results. |
| [Documentation index](docs/README.md) | Final-study guides, benchmark tooling and archived plans. |
| [Research archive](docs/research_archive.md) | How the earlier experiments relate to the final study. |

Historical paths remain stable for provenance. Separate, unpublished novelty studies in the local archive also use notebook numbers 85 and 86; they must not be merged with the final package's notebooks by number. The older `dissertation/` directory contains an earlier manuscript.

## Benchmark and research tooling

The original `sentiment-bench` CLI/TUI supports dataset validation, model benchmarking, news collection, scoring, evaluation and exports. It has its own root environment:

```bash
make benchmark-setup
make benchmark-check
uv run sentiment-bench --help
```

The benchmark check runs the full root test suite, Ruff and mypy, alongside public
dataset validation. The [contribution guide](CONTRIBUTING.md) lists individual
development commands, and the [script index](scripts/README.md) maps the earlier
collection, scoring and reporting workflows.

The default benchmark is the **5,947-row provenance-clean Financial PhraseBank/FiQA dataset**. The legacy `Data/data.csv` contains documented label corruption and is not the default. See the [dataset card](docs/dataset_card.md), [getting-started guide](docs/getting_started.md) and [CLI reference](docs/cli_reference.md).

## Data, citation and contributions

Licensed FNSPID and Reuters/LSEG text, raw model responses, credentials and large intermediate panels stay local. Public evidence consists of aggregate results, specifications, source code and figures. [Data and rights](docs/data_and_rights.md) explains the mixed third-party terms and the recorded data-governance limitations.

**Dissertation:** Peter Prendergast (2026), *A Hard Negative-Story Threshold: Training-Period Association, Temporal Non-Replication and Economic Limits*, MSc Computational Finance, University College London. Citation metadata is available in [CITATION.cff](CITATION.cff).

See [CONTRIBUTING.md](CONTRIBUTING.md) for reproducible changes and [SECURITY.md](SECURITY.md) for handling sensitive reports. The [CI workflow](.github/workflows/ci.yml) checks the final submission and the public benchmark separately.
