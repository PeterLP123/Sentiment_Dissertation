# financial_sentiment_v2

Provenance-clean rebuild of the labeled sentiment benchmark, built by
`scripts/build_labeled_dataset.py` directly from the original sources. It
replaces `Data/data.csv` (the Kaggle merge), whose 514 "conflicting
duplicates" were found on 2026-06-12 to be corrupt rows — every negative
PhraseBank sentence duplicated under a wrong `neutral` label — and which
silently dropped the 50-65% agreement tier (the most human-contested
sentences).

- Rows: 5947 (4836 PhraseBank + 1111 FiQA; every sentence unique)
- Labels: {'negative': 981, 'neutral': 2884, 'positive': 2082}
- PhraseBank agreement tiers (strongest file containing the sentence): {50: 627, 66: 761, 75: 1189, 100: 2259}
- PhraseBank sentences dropped for in-source label conflicts: 2
- SHA-256: `514dfc90316ec43617b7d59ac89a29d1063549ed91b08ef21cd11f5a8ac9ea9e`

## Sources

- Financial PhraseBank v1.0 (Malo et al. 2014), CC BY-NC-SA 3.0 —
  `Data/source/financial_phrasebank/FinancialPhraseBank-v1.0.zip`
- FiQA 2018 Task 1 (Maia et al. 2018) via `pauri32/fiqa-2018` on Hugging Face —
  `Data/source/fiqa_2018/{train,validation,test}.csv`

## Column notes

- `pb_agreement_tier`: 100/75/66/50 — minimum annotator-agreement file the
  sentence appears in (16 annotators, 5-8 annotations per sentence). This is
  the graded human-disagreement ground truth for ambiguity-proxy validation.
- FiQA rows are sentence-level aggregates of per-target continuous scores;
  the label is the sign of the mean score. FiQA has **no annotated neutral
  class** — restrict neutral-sensitive analyses to `source ==
  financial_phrasebank`. `fiqa_sign_conflict = 1` marks sentences whose
  target-level scores disagree in sign.

## Caveats

- Both sources are public and predate all candidate-model training cutoffs;
  the contamination/look-ahead caveat in `docs/research_protocol.md` applies.
- PhraseBank license is CC BY-NC-SA 3.0 (non-commercial; fine for the
  dissertation with attribution).
