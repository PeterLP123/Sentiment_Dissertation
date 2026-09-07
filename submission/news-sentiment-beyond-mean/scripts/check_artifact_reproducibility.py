"""Regenerate manuscript artifacts with a fresh font cache and compare bytes."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "manuscript" / "artifacts"
FIGURE_STEMS = (
    "fig_aggregation_family",
    "fig_break_even_cost_gap",
    "fig_coefficient_drift",
    "fig_cross_source_coefficient_power",
    "fig_double_sort",
    "fig_estimate_stability",
    "fig_lseg_matched_control_value",
    "fig_prompt_gate",
    "fig_risk_regime_stability",
    "fig_timing_placebo",
)
EXPECTED_ARTIFACTS = (
    "derived_metrics.csv",
    "headline_numbers.json",
    *(f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")),
    "tab_aggregation_family.tex",
    "tab_headline_estimates.tex",
    "tab_hypothesis_verdicts.tex",
    "tab_outcome_leg_decomposition.tex",
    "tab_outcome_label_sensitivities.tex",
    "tab_prompt_selection.tex",
    "tab_robustness_diagnostics.tex",
    "validation_report.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    missing_reference = [name for name in EXPECTED_ARTIFACTS if not (REFERENCE_DIR / name).is_file()]
    if missing_reference:
        print("Missing reference artifacts:", *missing_reference, sep="\n- ", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="news-sentiment-artifact-repro-") as temp_dir:
        temp_root = Path(temp_dir)
        generated_dir = temp_root / "artifacts"
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(temp_root / "matplotlib")
        environment["PYTHONHASHSEED"] = "0"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.generate_dissertation_artifacts",
                "--output-dir",
                str(generated_dir),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout, end="", file=sys.stderr)
            print(result.stderr, end="", file=sys.stderr)
            return result.returncode
        if "findfont:" in result.stderr:
            print("Font fallback occurred during clean-cache regeneration:", file=sys.stderr)
            print(result.stderr, end="", file=sys.stderr)
            return 1

        missing_generated = [name for name in EXPECTED_ARTIFACTS if not (generated_dir / name).is_file()]
        if missing_generated:
            print("Generator omitted expected artifacts:", *missing_generated, sep="\n- ", file=sys.stderr)
            return 1

        mismatches: list[tuple[str, str, str]] = []
        for name in EXPECTED_ARTIFACTS:
            reference_hash = sha256(REFERENCE_DIR / name)
            generated_hash = sha256(generated_dir / name)
            if reference_hash != generated_hash:
                mismatches.append((name, reference_hash, generated_hash))
        if mismatches:
            print("Artifact byte comparison failed:", file=sys.stderr)
            for name, reference_hash, generated_hash in mismatches:
                print(f"- {name}: reference={reference_hash} generated={generated_hash}", file=sys.stderr)
            return 1

    print(
        "Artifact reproducibility: PASS "
        f"({len(EXPECTED_ARTIFACTS)} files, including {len(FIGURE_STEMS) * 2} figure files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
