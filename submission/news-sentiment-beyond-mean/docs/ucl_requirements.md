# UCL dissertation requirements and marking contract

This note records the requirements used to design the manuscript. It was
checked on 13 August 2026 against the supplied programme instructions,
marking-rubric PDF and UCL MSc Computational Finance LaTeX example. The source
PDFs remain outside this clean repository.

## Submission and length

- The expected scale is approximately 50 pages and about 10,000 content words,
  plus figures, tables and equations.
- The main body is expected to occupy roughly 40--45 pages. Front matter,
  references, appendices and code do not form part of that guide.
- LaTeX is recommended and the supplied example uses the `book` class.
- The final document must include a copy of the project summary after the
  appendices.

## Required front matter

The title page must contain this wording exactly:

> This dissertation is submitted as partial requirement for the MSc
> Computational Finance degree at UCL. It is substantially the result of my
> own work except where explicitly indicated in the text.

It must also contain the selected distribution statement. The current draft
uses the restricted-distribution option, which must be confirmed before
submission.

The abstract follows the title page. UCL describes it as short and accessible
across technical backgrounds. It should explain:

1. what the project studies, its aim and the central challenge;
2. how the work was carried out and what was learned; and
3. the main results.

The contents follow the abstract. Lists of figures and tables are appropriate
for this dissertation. Acknowledgements are optional.

## Expected chapter logic

- **Introduction:** a clear thesis or research hypothesis and a concise account
  of how it will be tested.
- **Literature review:** a documented, critical account of the relevant field,
  not a list of paper summaries.
- **Methodology:** enough detail for a technically competent non-specialist to
  check the design and reproduce the logic.
- **Results:** clear tables and figures, with uncertainty and validation close
  to each claim.
- **Conclusions and future directions:** direct answers, limitations and
  specific research extensions.
- **References and appendices:** consistent citation details, including DOI or
  URL where appropriate, plus reproducibility material without licensed text.

## Presentation

- A4 paper, approximately 2.5 cm margins, 12-point type recommended and never
  below 10 point.
- One-and-a-half line spacing.
- Page numbering after the title page.
- Numbered chapters, sections and equations.
- Figures and tables should appear near their first discussion, be numbered and
  have descriptive captions.
- Notation and citation style must be consistent throughout.

## Marking rubric

| Criterion | Weight | What the top band requires here |
|---|---:|---|
| Structure, clarity and research hypothesis | 10% | One sharply formulated, theoretically motivated question; explicit alignment between question, method and evidence; a coherent narrative accessible to a knowledgeable non-specialist. |
| Adherence to lecture topics | 20% | Accurate and creative use of computational-finance concepts, especially event timing, return measurement, volatility forecasting, portfolio accounting and statistical inference. |
| Choice of models | 20% | Thorough justification of FinBERT, aggregation rules, rank regression, HAR and comparators; plausible alternatives evaluated critically. |
| Results | 15% | Original, clearly presented results with academic or industry value, including honest negative results. |
| Validation of results | 20% | Systematic robustness work, chronological boundaries, clustered uncertainty, multiplicity control, costs, placebos, power and limitations. |
| Consistency and notation | 5% | Production-quality language, formulas, variable definitions, tables and figures. |
| Conclusions and further challenges | 5% | Coherent action points and future research tied to both finance practice and theory. |
| Background literature | 5% | A thorough, documented search and demonstrated understanding of the closest literature. |

## Manuscript consequences

1. Use a single primary research question; treat economic value, risk
   translation and LSEG portability as ordered objectives, not competing
   questions.
2. Keep statistical detectability, economic usability and portability as three
   separate claims.
3. Give validation unusually high visibility because it carries 20% of the
   mark and is central to the dissertation's contribution.
4. Allocate approximately 300 words to the abstract and 1,500 words to the
   final chapter. The abstract sits outside the 10,000-word main-body budget.
5. Change the manuscript to 12-point type, one-and-a-half spacing and margins
   close to 2.5 cm before final compilation.
6. Include a specific artificial-intelligence-use statement. UCL's current
   central guidance requires acknowledgement of how generative AI assisted an
   assessed work; the assessment brief still determines what uses are allowed.
   The supplied materials and private source archive do not contain a
   programme-specific declaration or category. This is a blocking submission
   check, not something the disclosure itself resolves; see
   `docs/submission_gates.md`.
