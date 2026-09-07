# Project Documentation

The completed dissertation project is [`submission/news-sentiment-beyond-mean/`](../submission/news-sentiment-beyond-mean/README.md). It contains the submitted manuscript, selected experiments, frozen specifications, aggregate evidence, a locked Python environment, and its own validation workflow.

This directory documents the broader research workspace from which that final package was distilled. It includes reusable benchmark tooling and historical protocols, results, and runbooks. Use it to understand how the project developed; use the submission package when citing or reproducing the final study.

![News sentiment beyond the mean](assets/project-banner.svg)

## Start here

| Goal | Document |
| --- | --- |
| Understand the final study | [Submission README](../submission/news-sentiment-beyond-mean/README.md) |
| Trace final claims to frozen evidence | [Final evidence map](../submission/news-sentiment-beyond-mean/docs/evidence_map.md) |
| Reproduce and validate the submission | [Project commands and prerequisites](reproducing_the_submission.md) |
| Read the dissertation | [Submitted PDF](../submission/news-sentiment-beyond-mean/Peter-Prendergast-COMP0077-Dissertation.pdf) |
| Install the broader benchmark application | [Getting started](getting_started.md) |
| Explore the pre-submission research programme | [Research archive](research_archive.md) |

## Documentation map

### Final study

The canonical project lives under [`submission/news-sentiment-beyond-mean/`](../submission/news-sentiment-beyond-mean/README.md). Its local documentation covers the [research design](../submission/news-sentiment-beyond-mean/docs/research_design.md), [data and licensing](../submission/news-sentiment-beyond-mean/docs/data_and_licensing.md), [reproducibility](../submission/news-sentiment-beyond-mean/docs/reproducibility.md), and [submission gates](../submission/news-sentiment-beyond-mean/docs/submission_gates.md).

### Reusable benchmark application

| Document | Purpose |
| --- | --- |
| [Getting started](getting_started.md) | Install, validate the benchmark data, run a pilot, and export results. |
| [CLI reference](cli_reference.md) | Commands, options, and defaults. |
| [Dataset card](dataset_card.md) | Benchmark provenance, label policy, limitations, and ethics. |
| [Architecture](architecture.md) | Package boundaries and data flow. |
| [Model providers](model_providers.md) | OpenRouter and Ollama configuration. |
| [Results and exports](results_and_exports.md) | Result storage and artifact schemas. |
| [TUI guide](tui_guide.md) | Interactive workflow. |
| [News sourcing](news_sourcing.md) | Public-web collection and provenance. |

### Research archive

The [research archive map](research_archive.md) separates the exploratory notebook programme, frozen protocols, historical result records, and operational runbooks. The large [`final_experiments/`](../final_experiments/README.md) tree is preserved for auditability; the final package provides the concise reproduction surface.

## Evidence boundary

The final result is a bounded temporal non-replication and timing diagnosis. The training-period negative-story-share association does not establish a deployable signal: row-level availability times are missing, the unchanged coefficient reverses sign in the frozen 2020–2023 evaluation, and trading evidence is adverse after costs. See the final package for the exact estimands, intervals, specifications, and limitations.

Raw FNSPID and licensed LSEG text, model caches, API responses, databases, and large intermediate panels remain local. Only licence-safe aggregate evidence belongs in Git.
