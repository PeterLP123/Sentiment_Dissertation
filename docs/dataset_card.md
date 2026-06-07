# Dataset Card

Last updated: 2026-06-07

Related documentation:

- [Getting started](getting_started.md)
- [Results and exports](results_and_exports.md)
- [Research protocol](research_protocol.md)
- [Tavily news sourcing](news_sourcing.md)

## Dataset Identity

- Repository path: `Data/data.csv`
- SHA-256: `3d86ea04e694471479b7473a84c005a5456491ddfdc43ce5340cb2ae1b6c6b05`
- Planned public source citation: Kaggle Financial Sentiment Analysis, `sbhatti/financial-sentiment-analysis`
- Source URL: <https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis>
- Current repository role: source dataset for dissertation sentiment benchmark experiments

The Kaggle source should be verified against the final dissertation bibliography before submission. If the local file was transformed from the downloaded source, the transformation steps should be added to this card.

## Dataset Shape

The current file has:

- Rows: 5,842
- Columns: `Sentence`, `Sentiment`
- Labels: `positive`, `negative`, `neutral`

Label counts:

| Label | Rows |
| --- | ---: |
| positive | 1,852 |
| negative | 860 |
| neutral | 3,130 |

Duplicate and conflict summary:

| Measure | Count |
| --- | ---: |
| Duplicate sentence groups | 520 |
| Extra duplicate rows | 520 |
| Conflicting duplicate groups | 514 |
| Rows in conflicting duplicate groups | 1,028 |
| Primary scoring rows after excluding conflicting duplicates | 4,814 |

Primary-scope label counts:

| Label | Rows |
| --- | ---: |
| positive | 1,852 |
| negative | 346 |
| neutral | 2,616 |

## Column Definitions

`Sentence` contains one financial-domain text item. Items may include company names, market commentary, financial results, headlines, stock symbols, or short social-media-style statements.

`Sentiment` is the assigned sentiment label:

- `positive`: text indicates favorable financial performance, outlook, market movement, or sentiment.
- `negative`: text indicates unfavorable financial performance, outlook, market movement, or sentiment.
- `neutral`: text is primarily factual, mixed, procedural, or lacks clear positive or negative sentiment.

## Intended Use

This dataset is used to evaluate and compare sentiment classification systems in a dissertation context. Suitable uses include:

- Benchmarking LLMs and non-LLM sentiment classifiers.
- Comparing prompt variants and model families on shared rows.
- Measuring robustness under prompt perturbation.
- Studying ambiguity through duplicate label conflict, model disagreement, and self-consistency entropy.

The dataset should not be used as evidence for trading decisions, financial advice, or claims about future asset performance.

News article corpora sourced through Tavily are separate derived source material under `Data/news/`. They are not part of this labeled benchmark dataset unless a later labeling workflow creates a documented derived dataset.

## Data Handling Rules

Do not modify `Data/data.csv` in place. Treat it as source material.

If cleaning, filtering, relabeling, or deduplicating is needed, write a derived file and document:

- Source file and SHA-256 hash
- Cleaning steps
- Label mapping
- Inclusion/exclusion rules
- Train/validation/test split logic
- Random seed
- Date and responsible experiment/run

Benchmark tooling uses two scoring scopes:

- `primary`: excludes rows from duplicate sentence groups with conflicting labels.
- `all`: includes every selected row and is used as an audit or ambiguity-analysis scope.

## Known Limitations

The dataset is imbalanced, with `neutral` as the majority class. Accuracy alone can therefore be misleading; macro-F1, balanced accuracy, MCC, and per-class metrics should be reported.

Many duplicate sentence groups have conflicting labels. These conflicts may reflect genuine ambiguity, annotation disagreement, source inconsistencies, or preprocessing issues. They should be analyzed explicitly rather than silently removed without documentation.

The dataset is financial-domain text. Results may not generalize to product reviews, political text, general social media, or other sentiment-analysis domains.

The public source and license should be rechecked before final dissertation submission. Any redistribution limits from Kaggle or the original upstream data source should be respected.

## Ethical and Privacy Considerations

The file appears to contain public financial text, headlines, and market commentary. It may include company names, stock symbols, and short social-media-style statements. Avoid printing large raw samples in logs, exported reports, or final dissertation appendices unless the citation, licensing, and privacy implications are clear.

Models evaluated on this dataset may reproduce biases from training data or provider-specific moderation/parsing behavior. Report observed results without overstating model understanding or treating benchmark performance as financial expertise.

## Reproducibility Notes

Formal experiments should record:

- Dataset path and SHA-256 hash
- Code commit SHA
- Run IDs and export paths
- Prompt ID and prompt hash
- Model IDs and provider route
- Mode, seed, sample logic, temperature, token limits, retries, and concurrency
- Metrics, statistical tests, and interpretation notes

The experiment registry at `experiments/manifest.toml` is the source-controlled index for formal dissertation runs.
