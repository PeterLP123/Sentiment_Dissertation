# Data and rights

This repository contains material with mixed ownership and licensing. Its public
availability permits inspection and use of the reproducibility materials only to the
extent allowed by the applicable rights. It is not a blanket licence or a grant of
permission to copy, redistribute, republish, train on, or commercially use every file.

## Repository components

- Original source code and documentation remain subject to the rights stated by the
  repository owner. No repository-wide open-source licence has been granted here.
- The dissertation, its figures, tables, and named submission PDF are authored works.
  The submitted manuscript states that it may not be copied or distributed without
  the author's permission.
- Financial PhraseBank, FiQA, Ken French factor data, the UCL template and branding,
  bibliographic styles, and other third-party materials retain their original rights
  and attribution requirements. Consult the upstream source before reuse.
- Reuters/LSEG and FNSPID article-level material, model responses derived from licensed
  text, large panels, databases, and checkpoints are deliberately excluded from the
  final public package.

The [dataset card](dataset_card.md) records the benchmark sources, transformations,
known corruption in the legacy merge, and the licence information available during the
project. Recheck upstream terms for the intended use; a tracked copy or citation does
not establish that every downstream use is permitted.

## Final submission boundary

The final package publishes code, frozen specifications, aggregate evidence, selected
notebooks, figures, tables, and provenance records. Its detailed
[data and licensing note](../submission/news-sentiment-beyond-mean/docs/data_and_licensing.md)
also records an unresolved caveat: no archived licence instrument, data-processing
agreement, or ethics determination establishes permission for the historical transfer
of licensed headlines to an external model provider. No-retention routing and
aggregate-only publication do not resolve that question.

Do not infer a right to the omitted inputs from the ability to audit aggregate results.
Anyone seeking a full computational replay must obtain the data independently under
appropriate terms and confirm that the proposed processing is authorised.

## Adding material

Before committing data or generated evidence, verify its source, licence, sensitivity,
and whether a row could reconstruct restricted text or identifiers. Keep credentials,
raw licensed news, provider payloads, databases, Parquet panels, model artifacts, and
large generated outputs outside Git. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) and
run `make validate` before publication.
