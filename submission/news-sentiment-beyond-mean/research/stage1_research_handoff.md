# Stage 1 research handoff

Date: 13 August 2026.

## Recommended dissertation angle

The strongest dissertation is not a claim that sentiment is tradable and not a
claim that the LSEG risk rule works. It is a disciplined study of an apparently
small modelling choice---how several same-company news stories are compressed
into one daily signal---and of the gap between statistical information and
economic usefulness.

**Working title:** *News Sentiment Beyond the Mean: Statistical Information,
Economic Limits and Cross-Dataset Portability*

**Primary question:** Does the share of negative stories in a firm's same-day
news distribution contain incremental information about its next-session
abnormal return after controlling for mean sentiment, news volume and recent
price dynamics?

## Answer supported by the evidence

The answer is **yes in the large FNSPID training panel, but only as a small
conditional association**. Negative-story share is the only corrected survivor
in the nine-rule aggregation family. Its coefficient remains negative after
mean sentiment, news volume, lagged one-day return, lagged five-day return and
recent volatility are controlled. The economic translation is much weaker: the
model-free spread is about -0.62 basis points per session, its interval crosses
zero, and the best break-even cost is about 0.625 basis points per side against
the 10 basis points charged.

The secondary downside result is a bounded case study. Scaling a HAR volatility
target down during high aggregate negative pressure improves downside outcomes
in one FNSPID period and has unusual mapped-session alignment among circular
shifts there. Because current mapped-session news sets the same session's
modifier, this is not an ex-ante timing result. The exact rule does not
activate in the recent LSEG block; flexible aggregate, firm-level and prompt
searches do not pass complete economic gates. LSEG coefficient signs are
negative but too imprecise to establish or reject an FNSPID-sized effect.

## Contribution ladder

1. **Primary contribution:** a like-for-like comparison of nine firm-day
   aggregation rules and a direct estimate of what negative share adds beyond
   the mean, coverage and recent price path.
2. **Economic contribution:** an explicit demonstration that statistical
   information can be real yet too small for daily implementation after costs.
3. **Validation contribution:** source-separated LSEG transfer, minimum
   detectable effects, schedule-shift diagnostics, matched-exposure controls and a complete
   null/provenance ledger.
4. **Practitioner implication:** model-risk review should test aggregation and
   turnover before upgrading a classifier or engineering prompts.

The novelty is bounded. Prior work studies negative-word fractions, tone
dispersion, news-sentiment dispersion and top-news selection. The targeted
search found little work combining broad firm-day aggregation comparison,
direct conditional controls, realistic costs and cross-source transfer under
one design.

## Proposed argument structure

1. **Introduction:** averaging is a lossy modelling choice; ask one question and
   pre-state the statistical/economic/portability distinction.
2. **Literature review:** move from media effects to negative asymmetry, domain
   scoring, aggregation/dispersion, predictive inference and costs; finish with
   the narrow gap.
3. **Data and methodology:** define session assignment, the three non-pooled
   regimes, the nine rules, the daily rank estimand, price controls, HAC/BH,
   model-free sort, costs, HAR base and validation gates.
4. **Results:** aggregation family; information beyond the mean; price-path
   confound; basis-point scale and costs; bounded risk translation; LSEG and
   prompt nulls.
5. **Conclusions and future directions:** answer directly, explain practical
   meaning, assess strengths/limitations, then prioritise story-family,
   prospective and human-label tests.

## Abstract and final-chapter budget

- Abstract: approximately **300 words**, outside the 10,000-word main body.
- Final chapter: approximately **1,500 words**.

The longer final chapter should not repeat every result. It should allocate
roughly 375 words to the answer and contribution, 275 to practical meaning, 375
to strengths and limitations, and 425 to prioritised further work.

## Adversarial finding that changes the write-up

The most serious unresolved alternative explanation is **story repetition**.
News count is controlled, but the FNSPID checkpoint cannot reliably identify
whether several negative records are independent events or rewrites of the same
event. This must appear in the abstract's challenge clause only if space permits,
and must appear explicitly in the methodology, limitations and first future-
work proposal.

## Stage 2 authoring inputs

- `docs/ucl_requirements.md`: format and rubric contract.
- `docs/exemplar_style_notes.md`: adopted narrative style.
- `research/rq_and_methodology.md`: question, estimand and claim rules.
- `research/literature_matrix.md`: 33 retained sources.
- `research/source_verification.md`: DOI and content audit.
- `research/devils_advocate_checkpoint_1.md`: strongest objections and required
  wording changes.
- `docs/evidence_map.md`: notebook-to-claim map.
- `docs/writing_plan.md`: 10,000-word allocation and expanded abstract/final
  chapter.

## Stage decision

Proceed to authoring only on this bounded thesis. Do not reopen prompt tuning or
search for a profitable LSEG rule during writing. The evidence is sufficient
for a strong dissertation because the main positive result survives the most
obvious price-path confound and the null economic/portability results are
quantified rather than hidden.
