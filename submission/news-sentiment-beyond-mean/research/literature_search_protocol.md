# Literature search protocol

Search date: 13--14 August 2026.

## Purpose

The search supports five parts of the dissertation:

1. whether financial news tone is associated with returns and fundamentals;
2. why negative information may behave differently from positive information;
3. how text is scored and how multiple scores are aggregated;
4. how predictive finance studies should handle dependence, multiplicity,
   costs and power; and
5. why a HAR volatility target is an appropriate sentiment-free comparator;
   and
6. how empirical finance treats out-of-sample decay and time-local
   predictability.

This is a structured, targeted literature search for an MSc dissertation, not
a registered systematic review. The novelty statement is therefore bounded to
the located literature.

## Sources searched

- publisher and journal pages from Wiley, Oxford Academic, Elsevier,
  Taylor & Francis, SAGE, ACM and CFA Institute;
- official working-paper or institutional repositories, including NBER, the
  Federal Reserve, UCL Discovery and arXiv;
- DOI resolution and Crossref-linked metadata;
- backward references from the Loughran--McDonald survey and the closest
  empirical papers; and
- targeted forward searches for 2024--2026 work on news-sentiment dispersion,
  large language models and return prediction.

## Query families

- `financial news sentiment stock returns negative news`
- `Reuters sentiment stock returns aggregation`
- `multiple news stories daily sentiment aggregation mean dispersion`
- `negative share news sentiment next day return`
- `tone dispersion financial disclosure` and `news sentiment dispersion`
- `FinBERT financial text sentiment return prediction`
- `LLM headline sentiment stock return look-ahead bias`
- `HAC multiple testing block bootstrap asset pricing`
- `trading costs turnover anomaly returns`
- `HAR realised volatility volatility managed portfolios`
- `minimum detectable effect null results`
- `stock return predictor out of sample decay temporal instability`

## Inclusion rules

- Peer-reviewed finance, accounting, econometrics or natural-language
  processing work is preferred.
- Dataset and model papers are included when they define an input used here.
- Working papers are included only when they are directly relevant to a
  current frontier issue and are labelled as such.
- Adjacent dispersion research is included even when it studies disclosures,
  mergers or within-document dispersion, because it constrains the novelty
  claim.
- Each included source must support a specific sentence or design choice.

## Exclusion rules

- Generic sentiment-classification papers with no financial-domain or
  measurement relevance;
- social-media-only prediction papers unless needed for a model definition;
- papers cited only because they report a large backtest Sharpe ratio;
- secondary summaries when a primary article or official repository is
  available; and
- unverified 2025--2026 preprints inherited from the earlier disagreement
  framing.

## Search outcome and bounded gap

The retained matrix contains 37 core, adjacent or methodological sources. The literature
clearly establishes that news tone, coverage, staleness, domain-specific
language and negative-news asymmetry matter. It also contains work on tone
dispersion within disclosures and event-specific news sets. The narrower gap is
that the located firm-level news-return studies generally choose an aggregation
rule---often a mean, sum, fraction of negative words or selected top news---and
do not compare a broad set of firm-day story-distribution summaries under one
return definition, correction family, cost model and cross-source transfer
design. This is a targeted-search finding, not proof of global absence.

## Verification rule

- Resolve every available DOI and match title, author and venue metadata.
- For sources without a DOI, verify the persistent arXiv or institutional
  identifier.
- Inspect the publisher abstract or substantive primary-source content for
  every retained source and every claim central to the research gap.
- Record corrections and exclusions in `source_verification.md`.
