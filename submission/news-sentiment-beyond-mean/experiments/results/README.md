# Aggregate evidence snapshot

Each numbered directory is a frozen aggregate result snapshot. Statements in a
manifest describe the repository state at that analysis's execution and are not
rewritten retrospectively when later work closes.

## Current Notebook 75 status

As of 16 August 2026, Notebook 75 is no longer sealed or unexecuted. The
outcome-blind amendment in
`experiments/specs/fnspid_evaluation_beyond_mean_preexecution_amendment_v1_20260814.json`
authorised one opening. The analysis executed once on 14 August 2026, and its
licence-safe aggregate outputs were promoted into
`experiments/results/75_fnspid_evaluation_beyond_mean_replication/` without a
second execution. The result manifest, `result_summary.json`,
`postflight_validation.md` and `promotion_manifest.json` preserve the audit
chain.

Accordingly, older Notebook 76--78 specifications and manifests that say
Notebook 75 remained deliberately unexecuted are accurate historical records,
not descriptions of the repository's current state. Preserve those frozen
files and do not rerun Notebook 75.

## Post-review diagnostics

Notebooks 86 and 87 were authorised on 16 August 2026 to answer the external
review without reopening Notebook 75. Notebook 86 decomposes the assigned
open-to-next-open outcome into intraday and overnight legs and reconciles the
tied-rank translation, prospective versus realised precision, and FNSPID
maximum drawdown. Notebook 87 adds the requested previous-open probe,
trailing-beta and HAC checks, FNSPID episode exclusions, and class/tie base
rates. Both studies are post-review and post-selection; neither is independent
confirmation. Their CSV outputs reproduced byte for byte in separate temporary
runs, and their manifests record an unchanged Notebook 75 hash.
