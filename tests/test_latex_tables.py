from pathlib import Path

from sentiment_benchmark.latex_tables import (
    agreement_table,
    generate_latex_tables,
    latex_escape,
    leaderboard_table,
    mcnemar_table,
    operational_table,
    per_class_table,
    sensitivity_table_latex,
)
from sentiment_benchmark.prompt_sensitivity import SensitivityResult, VariantResult


def _metric_record(model_id: str, accuracy: float, macro_f1: float) -> dict:
    return {
        "model_id": model_id,
        "scope": "primary",
        "row_count": 90,
        "accuracy": accuracy,
        "balanced_accuracy": accuracy - 0.02,
        "mcc": accuracy - 0.1,
        "macro_f1": macro_f1,
        "weighted_f1": accuracy,
        "per_class": {
            "positive": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 30.0},
            "negative": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 30.0},
            "neutral": {"precision": 0.5, "recall": 0.4, "f1": 0.45, "support": 30.0},
        },
        "invalid_output_count": 1,
        "api_error_count": 0,
    }


def _statistics() -> dict:
    return {
        "scope": "primary",
        "seed": 42,
        "per_model": {
            "openai/gpt-4o-mini": {
                "accuracy_ci": {"point": 0.9, "lower": 0.83, "upper": 0.95, "confidence": 0.95},
                "macro_f1_ci": {"point": 0.85, "lower": 0.78, "upper": 0.91, "confidence": 0.95},
                "n": 90,
            },
            "baseline/tfidf_logreg": {
                "accuracy_ci": {"point": 0.6, "lower": 0.5, "upper": 0.7, "confidence": 0.95},
                "macro_f1_ci": {"point": 0.55, "lower": 0.45, "upper": 0.65, "confidence": 0.95},
                "n": 90,
            },
        },
        "pairwise_mcnemar": [
            {
                "model_a": "baseline/tfidf_logreg",
                "model_b": "openai/gpt-4o-mini",
                "n": 90,
                "n_discordant": 30,
                "b_a_correct_b_wrong": 3,
                "c_a_wrong_b_correct": 27,
                "statistic": 17.6,
                "p_value": 0.00002,
                "method": "chi2_continuity",
            }
        ],
        "agreement": {
            "n_units": 90,
            "n_units_all_raters": 90,
            "n_raters": 2,
            "raters": ["baseline/tfidf_logreg", "openai/gpt-4o-mini"],
            "observed_agreement": 0.71,
            "fleiss_kappa": 0.55,
            "krippendorff_alpha": 0.56,
            "pairwise_cohen_kappa": [
                {"rater_a": "baseline/tfidf_logreg", "rater_b": "openai/gpt-4o-mini", "kappa": 0.55, "n": 90}
            ],
        },
        "operational": {
            "note": "test",
            "per_model": {
                "openai/gpt-4o-mini": {
                    "n_rows": 90,
                    "latency_ms": {"n": 90, "mean": 450.0, "p50": 400.0, "p95": 900.0},
                    "cost": {"n_rows_with_cost": 90, "total_usd": 0.0135, "usd_per_1k_rows": 0.15},
                    "tokens": {"total_prompt": 4500, "total_completion": 90, "mean_total_per_row": 51.0},
                    "invalid_count": 1,
                    "invalid_rate": 1 / 90,
                    "api_error_count": 0,
                    "api_error_rate": 0.0,
                },
                "baseline/tfidf_logreg": {
                    "n_rows": 90,
                    "latency_ms": {"n": 0, "mean": None, "p50": None, "p95": None},
                    "cost": {"n_rows_with_cost": 0, "total_usd": None, "usd_per_1k_rows": None},
                    "tokens": {"total_prompt": 0, "total_completion": 0, "mean_total_per_row": None},
                    "invalid_count": 0,
                    "invalid_rate": 0.0,
                    "api_error_count": 0,
                    "api_error_rate": 0.0,
                },
            },
        },
    }


def _records() -> list[dict]:
    return [
        _metric_record("openai/gpt-4o-mini", 0.9, 0.85),
        _metric_record("baseline/tfidf_logreg", 0.6, 0.55),
    ]


def test_latex_escape_handles_special_characters() -> None:
    assert latex_escape("baseline/tfidf_logreg") == r"baseline/tfidf\_logreg"
    assert latex_escape("50% & more $") == r"50\% \& more \$"
    assert latex_escape(r"a\b") == r"a\textbackslash{}b"


def test_leaderboard_sorts_bolds_best_and_includes_ci() -> None:
    table = leaderboard_table(_records(), _statistics(), run_id=12)
    assert table is not None
    assert "\\toprule" in table and "\\bottomrule" in table
    # Best (and only best) values are bold, with the CI outside the bold point.
    assert r"\textbf{0.900} [0.830, 0.950]" in table
    assert r"\textbf{0.850} [0.780, 0.910]" in table
    assert r"\textbf{0.600}" not in table
    # Sorted by macro-F1: the LLM row comes before the baseline row.
    assert table.index("gpt-4o-mini") < table.index(r"tfidf\_logreg")
    assert r"\label{tab:leaderboard-run12}" in table


def test_leaderboard_without_ci_falls_back_to_points() -> None:
    table = leaderboard_table(_records(), {"per_model": {}}, run_id=3)
    assert table is not None
    assert r"\textbf{0.900}" in table
    assert "[0.830, 0.950]" not in table


def test_per_class_table_blocks_per_model() -> None:
    table = per_class_table(_records(), run_id=12)
    assert table is not None
    assert table.count("positive") == 2
    assert "\\addlinespace" in table
    assert "& 30 \\\\" in table


def test_mcnemar_table_marks_significance() -> None:
    table = mcnemar_table(_statistics(), run_id=12)
    assert table is not None
    assert "$<$0.001$^{***}$" in table
    assert "uncorrected for multiple comparisons" in table


def test_agreement_table_includes_pairwise_kappa() -> None:
    table = agreement_table(_statistics(), run_id=12)
    assert table is not None
    assert "Fleiss' $\\kappa$ & 0.550" in table
    assert "($n$ = 90)" in table


def test_operational_table_handles_missing_cost_and_latency() -> None:
    table = operational_table(_statistics(), run_id=12)
    assert table is not None
    gpt_line = next(line for line in table.splitlines() if "gpt-4o-mini" in line)
    assert "0.0135" in gpt_line and "400" in gpt_line and "900" in gpt_line
    baseline_line = next(line for line in table.splitlines() if "tfidf" in line)
    # Cost total, cost/1k rows, p50, p95, and tokens/row are all missing for the baseline.
    assert baseline_line.count("--") == 5


def test_tables_return_none_without_data() -> None:
    assert leaderboard_table([], {}, run_id=1) is None
    assert mcnemar_table({"pairwise_mcnemar": []}, run_id=1) is None
    assert agreement_table({}, run_id=1) is None
    assert operational_table({}, run_id=1) is None


def test_sensitivity_table_latex_lists_variants_and_summary() -> None:
    result = SensitivityResult(
        model_id="openai/gpt-4o-mini",
        scope="primary",
        metric="macro_f1",
        variants=[
            VariantResult(run_id=3, prompt_id="base__order-pos-neg-neu", value=0.84),
            VariantResult(run_id=4, prompt_id="base__paraphrase-0", value=0.81),
        ],
        mean=0.825,
        std=0.015,
        minimum=0.81,
        maximum=0.84,
        spread=0.03,
        cv=0.0182,
    )
    table = sensitivity_table_latex(result)
    assert r"base\_\_order-pos-neg-neu & 3 & 0.8400" in table
    assert r"\multicolumn{2}{l}{Spread (max $-$ min)} & 0.0300" in table
    assert "booktabs" in table


def test_generate_latex_tables_writes_files_with_provenance(tmp_path: Path) -> None:
    metadata = {
        "package_version": "0.1.0",
        "exported_at": "2026-06-10T00:00:00+00:00",
        "dataset_path": "Data/data.csv",
        "dataset_sha256": "3d86ea04e694471479b7473a84c005a545",
        "prompt_hash": "b1271fc052ea2409",
        "mode": "pilot",
    }
    paths = generate_latex_tables(_records(), _statistics(), metadata, 12, tmp_path)

    names = {path.name for path in paths}
    assert names == {
        "leaderboard.tex",
        "per_class.tex",
        "mcnemar.tex",
        "agreement.tex",
        "operational.tex",
        "tables.tex",
    }
    leaderboard = (tmp_path / "leaderboard.tex").read_text(encoding="utf-8")
    assert "sha256 3d86ea04e694" in leaderboard
    assert "do not edit by hand" in leaderboard
    master = (tmp_path / "tables.tex").read_text(encoding="utf-8")
    assert "\\input{leaderboard.tex}" in master
    assert "\\input{operational.tex}" in master


def test_generate_latex_tables_empty_data_writes_nothing(tmp_path: Path) -> None:
    assert generate_latex_tables([], {}, {}, 1, tmp_path / "tables") == []
    assert not (tmp_path / "tables").exists()
