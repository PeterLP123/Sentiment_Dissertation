# Source verification record

Verification date: 30 August 2026.

## Coverage

- In-text citation keys: **39**.
- Bibliography entries: **39**.
- Missing citations or uncited bibliography entries: **0**.
- Sources with a DOI: **36**.
- DOI resolver, Crossref or official publisher checks completed: **36/36**.
- Persistent non-DOI records: **3/3** (two arXiv records and one JSTOR stable
  record).
- Abstract or substantive content checks: **39/39** through a publisher,
  journal, official institutional repository or primary preprint page.
- Fabricated sources found: **0**.
- Retraction or correction notices found in the checked records: **0**.

Thirty-five DOI records matched the expected title, first author and venue in
the live Crossref check. Bloom (1995) did not match through Crossref's title
search, but its DOI resolves to the exact title, author, journal, year, volume,
issue and pages on the official SAGE page. The three non-DOI records were
matched to their primary arXiv or JSTOR pages.

Some records have an online-first year before their printed issue year. The
bibliography uses the issue year, which agrees with the final journal record.

## Corrected metadata

- Huang, Wang and Yang (2023) occupies pages **806--841**, not 765--805.
- Heston and Sinha (2017) uses the published title *News vs. Sentiment:
  Predicting Stock Returns from News Stories* and journal DOI
  `10.2469/faj.v73.n3.3`.
- Dong, Fan and Peng (2024) is cited as a KDD conference paper at pages
  **4918--4927** with DOI `10.1145/3637528.3671629`, not only as an arXiv item.
- Lopez-Lira and Tang (2026) is an official *Journal of Financial Economics*
  article, number **104335**, DOI `10.1016/j.jfineco.2026.104335`. On the
  verification date, it was available online and assigned to the October 2026
  issue.
- Uhl's Reuters article is volume **15**, issue 4, pages 287--298.
- Chen et al. (2024) is volume **71**, article 102415.
- Isakin and Pu (2023) is volume **89**, article 102761.
- Welch and Goyal (2008) is in *The Review of Financial Studies* **21**(4),
  1455--1508, DOI `10.1093/rfs/hhm014`.
- McLean and Pontiff (2016) is in *The Journal of Finance* **71**(1), 5--32,
  DOI `10.1111/jofi.12365`.
- Farmer, Schmidt and Timmermann (2023) is in *The Journal of Finance*
  **78**(3), 1279--1341, DOI `10.1111/jofi.13229`.
- Fama and French (1993), Cakici et al. (2025) and Künsch (1989) were added to
  the audit after their use in the current manuscript was confirmed.

## Non-DOI records

| Record | Verification |
|---|---|
| Araci (2019) | arXiv:1908.10063; title and sole author matched on the primary arXiv page. |
| Glasserman and Lin (2023) | arXiv:2309.17322; title, authors and anonymisation experiment matched on the primary arXiv page. |
| MacKinlay (1997) | Journal of Economic Literature 35(1), 13--39; JSTOR stable record 2729691 matched. |

## Central claim checks

- **Negative language and returns:** publisher abstracts confirm Tetlock et al.
  (2008), Uhl (2014) and Heston--Sinha (2017) report stronger or slower effects
  for negative news in their respective settings.
- **Recent-price confound:** Chan (2003) and Savor (2012) distinguish drift
  after information from reversal after no-information moves.
- **Novelty boundary:** publisher content confirms Allee--DeAngelis (2015),
  Isakin--Pu (2023) and Chen et al. (2024) already study tone or news-sentiment
  dispersion. The manuscript therefore makes no claim to invent dispersion.
- **Story repetition:** Tetlock (2011) defines staleness using similarity to
  earlier firm stories and documents a different market response, supporting
  the explicit story-family limitation.
- **Costs and multiplicity:** Novy-Marx--Velikov (2016), Korajczyk--Sadka (2004)
  and Harvey--Liu--Zhu (2016) directly support the decision to separate gross
  association from net economic value and to expose the complete test family.
- **Temporal transport:** Welch--Goyal (2008), McLean--Pontiff (2016) and
  Farmer--Schmidt--Timmermann (2023) support treating later-period performance
  as a separate evidential hurdle. They do not identify why the FNSPID
  coefficient changes across eras.
- **Model choice:** Loughran--McDonald (2011), Araci (2019) and Huang et al.
  (2023) support domain adaptation; none is used as proof that the pinned model
  is correctly labelled on the final corpora.
- **Historical LLM scoring:** Glasserman--Lin (2023) compare original and
  company-anonymised headlines. The manuscript now describes the two effects
  they separate: look-ahead knowledge and distraction from company names. It no
  longer says that anonymisation itself reveals leakage.
- **Chronology:** Cakici et al. (2025) show that replacing a future-looking step
  with a one-sided method sharply weakens forecasting value. The manuscript no
  longer says that the correction removes most local pockets.
- **Time-series methods:** Newey--West (1987), Künsch (1989) and
  Politis--Romano (1994) support dependence-aware inference and block
  resampling. Künsch supports fixed-length blocks; wrapping blocks at the sample
  boundary is identified as this dissertation's implementation detail.
- **Risk benchmark:** Corsi (2009) supports the daily, weekly and monthly HAR
  components. Moreira--Muir (2017) supports inverse-volatility exposure. The
  manuscript separates those findings from its own exposure rule.
- **Factor and event-study methods:** Fama--French (1993) supports the FF3
  controls, while MacKinlay (1997) is the general event-study reference. Neither
  is presented as direct evidence for the news signal.

## Excluded inherited references

The earlier bibliography contained several items tied to the abandoned
cross-model-disagreement framing or unverified 2025--2026 frontier papers.
They were removed because no final sentence required them:

- algorithmic monoculture and rater-disagreement papers;
- placeholder SSRN entries with incomplete author names;
- unverified EACL/arXiv papers inherited from the disagreement search; and
- a purported leakage-safe trading paper whose metadata and final findings were
  not sufficiently verified.

This exclusion avoids using citations as decoration and leaves a smaller,
claim-linked bibliography.

## Boundaries retained in the paper

1. The exact pinned FinBERT checkpoint is stated separately from papers about
   related FinBERT models.
2. FNSPID's published headline scale describes the source dataset, not the
   cleaned panel analysed here.
3. LSEG is licensed. The paper identifies the source without reproducing text or
   implying redistribution rights.
4. The 2026 Lopez-Lira--Tang record is treated as available online and assigned
   to a future issue as of the verification date.

## Reproducible check

All retained identifiers and the claim supported by each source are recorded in
`literature_matrix.md` and `manuscript/references.bib`. A live metadata check was
rerun on 30 August 2026 after the final three methodology sources were included.
