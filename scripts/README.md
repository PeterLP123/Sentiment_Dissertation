# Research Scripts

These command-line scripts support the broader, historical research workspace. They
are not required to read or validate the submitted dissertation. For the preserved
submission workflow, start with [`make validate`](../docs/reproducing_the_submission.md).

## Public and local utilities

| Task | Entry point | Notes |
| --- | --- | --- |
| Build the public benchmark dataset | `build_labeled_dataset.py` | Rebuilds the tracked PhraseBank/FiQA derivative; do not edit the CSV by hand. |
| Check repository boundaries | `check_project.py` | Verifies the submission inventory, public-file policy, and documentation links. |
| Inspect UCL GPU availability | `ucl_gpu_status.py` | Read-only SSH status probe; requires UCL access. |
| Run the benchmark TUI on Windows | `run_tui.ps1` | Convenience launcher for the root Python package. |

## Historical experiment tooling

### Collection and preparation

| Workflow | Entry points | Guide |
| --- | --- | --- |
| LSEG collection cleanup and merging | [`build_lseg_clean_csvs.py`](build_lseg_clean_csvs.py), [`merge_lseg_headline_collections.py`](merge_lseg_headline_collections.py) | [LSEG pipeline](../docs/lseg_ollama_pipeline.md) |
| Materiality sample and story bodies | [`prepare_lseg_materiality_pilot.py`](prepare_lseg_materiality_pilot.py), [`fetch_lseg_materiality_bodies.py`](fetch_lseg_materiality_bodies.py) | [Research archive](../docs/research_archive.md) |
| Strategy price panel | [`build_strategy_price_panel.py`](build_strategy_price_panel.py) | [Strategy-research pipeline](../docs/strategy_research_pipeline.md) |
| FNSPID source retrieval and audit | [`download_fnspid.py`](download_fnspid.py), [`audit_fnspid_news.py`](audit_fnspid_news.py) | [Data and rights](../docs/data_and_rights.md) |

### Scoring and remote execution

| Workflow | Entry points | Guide |
| --- | --- | --- |
| Historical LSEG baseline study | [`run_lseg_headline_study.py`](run_lseg_headline_study.py) | [Baseline design](../docs/sentiment_trading_baseline.md) |
| OpenRouter validation and LSEG scoring | [`run_openrouter_gemma4_validation.py`](run_openrouter_gemma4_validation.py), [`run_openrouter_lseg_scoring.py`](run_openrouter_lseg_scoring.py) | [Validation](../docs/openrouter_gemma4_validation.md) and [authorisation record](../docs/lseg_external_processing_authorisation.md) |
| UCL local-model execution | [`ucl_run_lseg_baselines.sh`](ucl_run_lseg_baselines.sh), [`ucl_label_headlines.sh`](ucl_label_headlines.sh), [`ucl_ollama_server.sh`](ucl_ollama_server.sh) | [UCL GPU labelling](../docs/ucl_gpu_labeling.md) |
| Backward-replication status and launch | [`audit_lseg_backward_pipeline_status.py`](audit_lseg_backward_pipeline_status.py), [`watch_and_launch_lseg_finbert_ucl.py`](watch_and_launch_lseg_finbert_ucl.py) | [Backward-replication runbook](../docs/lseg_backward_replication_runbook.md) |

### Reports and derived views

| Output | Entry point | Guide |
| --- | --- | --- |
| Tail-risk factorial comparison | [`build_fnsipid_tail_risk_factorial.py`](build_fnsipid_tail_risk_factorial.py) | [Variant comparison](../docs/fnspid_tail_risk_variant_comparison.md) |
| Frozen strategy dashboard | [`build_recent_news_dashboard.py`](build_recent_news_dashboard.py) | [Mid-cap result](../docs/recent_news_midcap_finbert_strategy.md) |
| Week 6 trade explorer | [`build_week6_trade_dashboard.py`](build_week6_trade_dashboard.py) | [Trade-explorer guide](../docs/week6_trade_explorer.md) |
| Week 7 receipt-aligned P/L | [`build_week7_receipt_pnl.py`](build_week7_receipt_pnl.py) | [Receipt-aligned guide](../docs/week7_receipt_aligned_pnl.md) |

These are historical entry points, not a suggested sequence to rerun. Scripts that
fetch data, score text, or launch remote jobs may use external services, credentials,
licensed inputs, substantial compute, or a recorded authorisation flag. Read the linked
protocol and the script's `--help` output before running one.

Generated outputs belong in the ignored data, result, report, or experiment-output
locations already defined by the project. Preserve existing manifests and run folders;
the research tools generally create a new artifact or refuse to overwrite one.
