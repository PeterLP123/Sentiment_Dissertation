# Project Documentation

The completed dissertation, *A Hard Negative-Story Threshold: Training-Period Association, Temporal Non-Replication and Economic Limits*, is in [`submission/news-sentiment-beyond-mean/`](../submission/news-sentiment-beyond-mean/README.md). That package contains the final manuscript, frozen specifications, selected experiments, aggregate evidence, a locked Python environment, and its own validation workflow.

The rest of this directory covers the wider research workspace: reusable benchmark tools, historical protocols, results, and runbooks. Use the submission package to cite or reproduce the final study. Use these pages to follow the work that led to it or to run the benchmark application.

## Find what you need

| Task | Start here |
| --- | --- |
| Read the dissertation | [Dissertation PDF](../submission/news-sentiment-beyond-mean/Peter-Prendergast-COMP0077-Dissertation.pdf) |
| Understand the final design and findings | [Submission README](../submission/news-sentiment-beyond-mean/README.md) |
| Trace claims to frozen evidence | [Final evidence map](../submission/news-sentiment-beyond-mean/docs/evidence_map.md) |
| Reproduce the submitted study | [Commands and prerequisites](reproducing_the_submission.md) |
| Install and run the benchmark application | [Getting started](getting_started.md) |
| Browse earlier and exploratory work | [Research archive](research_archive.md) |

The submission package also documents the [research design](../submission/news-sentiment-beyond-mean/docs/research_design.md), [data and licensing](../submission/news-sentiment-beyond-mean/docs/data_and_licensing.md), [reproducibility](../submission/news-sentiment-beyond-mean/docs/reproducibility.md), and [submission gates](../submission/news-sentiment-beyond-mean/docs/submission_gates.md).

## Benchmark application

| Guide | Covers |
| --- | --- |
| [CLI reference](cli_reference.md) | Commands, options, and defaults |
| [Dataset card](dataset_card.md) | Provenance, label policy, limitations, and ethics |
| [Architecture](architecture.md) | Package boundaries and data flow |
| [Model providers](model_providers.md) | OpenRouter, Cerebras, and Ollama setup |
| [Results and exports](results_and_exports.md) | Result storage and artifact schemas |
| [TUI guide](tui_guide.md) | Interactive use |
| [News sourcing](news_sourcing.md) | Public-web collection and provenance |

## Evidence boundary

The final study reports a training-period association followed by temporal non-replication and adverse trading evidence after costs. Row-level availability times are missing, and the unchanged coefficient reverses sign in the frozen 2020–2023 evaluation. These results do not establish a deployable signal. The submission package gives the exact estimands, intervals, specifications, and limitations.

Raw FNSPID and licensed LSEG text, model caches, API responses, databases, and large intermediate panels remain local. Research outputs in Git are limited to licence-safe aggregate evidence.
