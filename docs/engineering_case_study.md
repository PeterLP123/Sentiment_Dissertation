# Three engineering decisions

The project combines a completed financial-news study with the benchmark and research
tools developed along the way. These examples explain how I handled data quality,
reproducibility and financial accounting. The linked tests document the corresponding
contracts; they do not establish predictive performance.

## Rebuilding a corrupted benchmark

**Problem.** The legacy Kaggle merge contained 514 negative PhraseBank sentences a
second time with incorrect neutral labels. Treating these conflicts as annotation
ambiguity would have produced a misleading comparison. Excluding all conflicting
groups would also have removed every PhraseBank negative from that comparison.

**Implementation.** I retained the original file for audit and rebuilt a derived
benchmark from Financial PhraseBank and FiQA sources. The build preserves source
identity, PhraseBank agreement tiers and FiQA continuous scores alongside the
classification labels. The default dataset contains 5,947 unique sentences with
zero conflicting duplicate groups.

**Verification.** The validator reports class counts, duplicate groups and label
conflicts. Dataset tests check both the clean default and the legacy file, keeping
the known corruption visible. This checks provenance and consistency; it does not
make every individual sentiment label unambiguous.

[Build script](../scripts/build_labeled_dataset.py) · [Dataset card](dataset_card.md) · [Tests](../tests/test_dataset.py)

## Making runs traceable

**Problem.** Results are hard to compare when model identity, prompts, input data or
configuration change without being recorded. Resuming a partially completed pipeline
can also mix incompatible artifacts.

**Implementation.** The benchmark records prompt hashes and runtime metadata. The
separate historical strategy pipeline derives run identities from its configuration
and input identities, and records stage manifests with output hashes. Completed
stages validate existing artifacts before reuse and reject mismatches. Atomic writes
protect readers from partially written files.

**Verification.** Tests exercise interrupted-stage resumption, matching-stage reuse,
changed-input rejection, output tampering and concurrent immutable writes. The
offline demo uses fresh output directories so it cannot reuse a historical run.
Hashes establish artifact identity, not scientific validity.

[Stage manifests](../src/sentiment_benchmark/strategy_research/artifacts.py) · [Atomic I/O](../src/sentiment_benchmark/artifact_io.py) · [Tests](../tests/strategy_research/test_artifacts.py)

## Accounting for overlapping holdings

**Problem.** Summing event returns can obscure how much capital is committed when
holding periods overlap. A strategy can look attractive before transaction costs
while failing after realistic accounting.

**Implementation.** The funded portfolio evaluator allocates overlapping cohorts
through rotating capital sleeves, marks positions daily, applies costs and
reconciles stock-level profit and loss to portfolio net asset value. Concentration
limits apply across overlapping sleeves; unused capacity remains in cash.

**Verification.** Tests reconcile holdings, daily marks, terminal profit and loss,
costs and net asset value. They also check concentration limits and the strict
exchange-open availability boundary. The dissertation separately reports missing
row-level news availability times as a binding limitation on predictive claims.

[Portfolio evaluator](../src/sentiment_benchmark/portfolio.py) · [Tests](../tests/test_portfolio.py) · [Final research design](../submission/news-sentiment-beyond-mean/docs/research_design.md)

## Research outcome

The training-period association did not replicate in the later evaluation. Timing
diagnostics and costs further limited the interpretation. The completed project
therefore demonstrates a documented evaluation process with null and adverse
findings preserved. The synthetic demo illustrates software execution only.

[Back to the project](../README.md) · [Run the offline demo](portfolio_demo.md)
