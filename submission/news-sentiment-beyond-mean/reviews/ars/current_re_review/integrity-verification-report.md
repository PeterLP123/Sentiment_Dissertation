# Academic Integrity Verification Report

## Verification mode

Initial verification for the exact ARS chain-start snapshot.

- Verification date: 14 August 2026.
- Checked draft SHA-256: `df7809760f530e043117c4741a2a54575f1ae2d8666a336587f324d8d634a524`.
- Verdict: **PASS**.
- Open integrity issues: **0**.
- Cross-model verification: not configured; this is a single-family verification.

## Verification summary

| Category | Coverage | Result |
|---|---:|---|
| Reference existence and metadata | 34/34 chain-start bibliography entries | Pass |
| Ghost citations | 34 cited keys and 34 bibliography keys | 0 orphan; 0 dangling |
| Citation-context checks | 34/34, using the source-verification record and primary abstracts/pages | Pass |
| Statistical and internal consistency | Complete repository validator, prose checker and test suite | Pass: 64 tests; 5 third-party deprecation warnings |
| Evidence-map decisions | 17/17 committed claim/decision rows | Pass |
| Originality screening | 46/153 eligible body paragraphs (30.1%); every numbered chapter represented | 46 original/no exact quoted match; 0 close or verbatim matches |
| Self-plagiarism | Author publication corpus not supplied | Not checked; no adverse finding inferred |

## Reference audit trail

Each row was searched by exact DOI or persistent identifier. The linked primary or persistent record matched title, author, year and venue metadata. Context checks also used `research/source_verification.md`, whose central-claim checks are bounded to what the cited sources establish.

| Key | Primary or persistent record | Determination |
|---|---|---|
| `allee2015` | <https://doi.org/10.1111/1475-679X.12072> | VERIFIED |
| `araci2019` | <https://arxiv.org/abs/1908.10063> | VERIFIED |
| `benjamini1995` | <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x> | VERIFIED |
| `bloom1995` | <https://doi.org/10.1177/0193841X9501900504> | VERIFIED |
| `chan2003news` | <https://doi.org/10.1016/S0304-405X(03)00146-6> | VERIFIED |
| `chen2024` | <https://doi.org/10.1016/j.ribaf.2024.102415> | VERIFIED |
| `corsi2009har` | <https://doi.org/10.1093/jjfinec/nbp001> | VERIFIED |
| `dong2024` | <https://doi.org/10.1145/3637528.3671629> | VERIFIED |
| `engelberg2011` | <https://doi.org/10.1111/j.1540-6261.2010.01626.x> | VERIFIED |
| `famafrench1993` | <https://doi.org/10.1016/0304-405X(93)90023-5> | VERIFIED |
| `fang2009` | <https://doi.org/10.1111/j.1540-6261.2009.01493.x> | VERIFIED |
| `garcia2013` | <https://doi.org/10.1111/jofi.12027> | VERIFIED |
| `glasserman2023` | <https://arxiv.org/abs/2309.17322> | VERIFIED |
| `harvey2016multiple` | <https://doi.org/10.1093/rfs/hhv059> | VERIFIED |
| `heston2017` | <https://doi.org/10.2469/faj.v73.n3.3> | VERIFIED |
| `huang2023` | <https://doi.org/10.1111/1911-3846.12832> | VERIFIED |
| `isakin2023` | <https://doi.org/10.1016/j.irfa.2023.102761> | VERIFIED |
| `kirtac2024` | <https://doi.org/10.1016/j.frl.2024.105227> | VERIFIED |
| `korajczyk2004` | <https://doi.org/10.1111/j.1540-6261.2004.00656.x> | VERIFIED |
| `lopezlira2026` | <https://doi.org/10.1016/j.jfineco.2026.104335> | VERIFIED; advance-online status retained |
| `loughran2011liability` | <https://doi.org/10.1111/j.1540-6261.2010.01625.x> | VERIFIED |
| `loughran2016` | <https://doi.org/10.1111/1475-679X.12123> | VERIFIED |
| `mackinlay1997` | <https://www.jstor.org/stable/2729691> | VERIFIED |
| `malo2014` | <https://doi.org/10.1002/asi.23062> | VERIFIED |
| `moreira2017` | <https://doi.org/10.1111/jofi.12513> | VERIFIED |
| `newey1987` | <https://doi.org/10.2307/1913610> | VERIFIED |
| `novymarx2016` | <https://doi.org/10.1093/rfs/hhv063> | VERIFIED |
| `politis1994` | <https://doi.org/10.1080/01621459.1994.10476870> | VERIFIED |
| `savor2012information` | <https://doi.org/10.1016/j.jfineco.2012.06.011> | VERIFIED |
| `tetlock2007` | <https://doi.org/10.1111/j.1540-6261.2007.01232.x> | VERIFIED |
| `tetlock2008words` | <https://doi.org/10.1111/j.1540-6261.2008.01362.x> | VERIFIED |
| `tetlock2011stale` | <https://doi.org/10.1093/rfs/hhq141> | VERIFIED |
| `uhl2014` | <https://doi.org/10.1080/15427560.2014.967852> | VERIFIED |
| `uhl2021` | <https://doi.org/10.1080/15427560.2020.1821375> | VERIFIED |

## Data and claim checks

The clean `HEAD` tree was exported to an isolated temporary directory and validated without executing notebooks. `make validate` reported:

- repository boundary valid across 259 files;
- prose style passed across 16 TeX files;
- 64 tests passed;
- five deprecation warnings originated in `exchange_calendars`;
- the sealed FNSPID evaluation notebook was not executed.

The validator and tests bind the manuscript's own-result statements to the committed aggregate snapshot and enforce the repository's data and licensing boundary. The 17-row evidence map was also inspected for null, multiplicity, power, cost, provenance and sealed-evaluation preservation.

## Originality screening

The sampling frame contained 153 body paragraphs of at least 20 words across the six numbered chapters. A deterministic SHA-256 ordering with seed `ars-originality-v1` selected 46 paragraphs, or 30.1%, with representation from every chapter. For each selected paragraph, an exact quoted fragment of 8--10 characteristic words was searched. No exact external match, close match or 20-word verbatim match was observed.

This screening uses public web search and is not a substitute for Turnitin or iThenticate. It cannot establish complete plagiarism absence.

## Integrity boundary

This PASS establishes the exact chain-start snapshot for revision replay. It does not approve the manuscript's scholarly quality, identify causal validity, or replace the separate ARS re-review verdict.
