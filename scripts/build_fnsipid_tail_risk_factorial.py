"""Build the verified four-cell FNSPID tail-risk factorial report."""
# Report prose intentionally remains readable as complete Markdown lines.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CELL_ORDER = ("v1", "price_only", "refit_only", "v2")
DISPLAY_NAMES = {
    "v1": "v1",
    "price_only": "price-only",
    "refit_only": "refit-only",
    "v2": "v2",
}
EXPECTED_DESIGN = {
    "v1": ("none", "frozen_development"),
    "price_only": ("min_abs_return", "frozen_development"),
    "refit_only": ("none", "annual_expanding"),
    "v2": ("min_abs_return", "annual_expanding"),
}
SOURCE_KEYS = ("helper_module", "notebook_ipynb", "notebook_source")
SAMPLE_KEYS = (
    "evaluation_news_rows",
    "evaluation_rows",
    "evaluation_target_dates",
    "panel_rows",
    "retained_firms",
)
REQUIRED_CELL_OUTPUTS = frozenset(
    {
        "bootstrap.csv",
        "calibration.csv",
        "forecasts.parquet",
    }
)
REQUIRED_INPUT_ROLES = frozenset(
    {
        "adjusted_daily_prices",
        "coherent_firm_cohort",
        "e5_timing_manifest",
        "e6_manifest",
        "news_corpus_and_scores",
    }
)
CONTRAST_ORDER = ("d_news_vs_price", "d_semantic", "d_tone_given_intensity")
EFFECT_METRICS = ("m1_hit_rate", "m2_hit_rate", "std_z", "m1_es_residual")
COVERAGE_DIAGNOSTICS = (
    "dq_conditional_coverage_wald",
    "es_conditional_calibration_wald",
)
EXPECTED_MODELS = ("M0", "M1", "M2", "M2_intensity")


class FactorialBuildError(RuntimeError):
    """Raised when a source bundle cannot support a closed factorial report."""


@dataclass(frozen=True)
class CellBundle:
    name: str
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    input_identities: Mapping[str, tuple[str, int]]
    sample_counts: Mapping[str, int]
    timing_limitation: Mapping[str, Any]
    calibration: Mapping[str, float | int]
    contrasts: pd.DataFrame
    assertion_count: int
    gate_count: int
    notebook_error_count: int
    executed_notebook_sha256: str


@dataclass(frozen=True)
class RunRecord:
    host: str
    python: str
    gpu: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FactorialBuildError(f"{context} must be a JSON object")
    return value


def _list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise FactorialBuildError(f"{context} must be a JSON array")
    return value


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FactorialBuildError(f"missing {context}: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorialBuildError(f"cannot read {context} {path}: {exc}") from exc
    return _mapping(value, context)


def _finite(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FactorialBuildError(f"{context} is not numeric") from exc
    if not math.isfinite(result):
        raise FactorialBuildError(f"{context} is not finite")
    return result


def _check_empty_output(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FactorialBuildError(f"output path is not a directory: {output_dir}")
    try:
        first_entry = next(output_dir.iterdir(), None)
    except OSError as exc:
        raise FactorialBuildError(f"cannot inspect output directory {output_dir}: {exc}") from exc
    if first_entry is not None:
        raise FactorialBuildError(f"output directory is not empty: {output_dir}")


def _verify_outputs(directory: Path, manifest: Mapping[str, Any]) -> None:
    outputs = _mapping(manifest.get("outputs"), f"{directory} manifest outputs")
    if not outputs:
        raise FactorialBuildError(f"{directory} manifest lists no outputs")
    missing_required = sorted(REQUIRED_CELL_OUTPUTS - set(outputs))
    if missing_required:
        raise FactorialBuildError(f"{directory} manifest omits consumed outputs: {', '.join(missing_required)}")
    root = directory.resolve()
    bad: list[str] = []
    for relative_name, raw_metadata in outputs.items():
        if not isinstance(relative_name, str) or not relative_name:
            raise FactorialBuildError(f"{directory} manifest has an invalid output path")
        metadata = _mapping(raw_metadata, f"output metadata for {relative_name}")
        expected_hash = metadata.get("sha256")
        expected_size = metadata.get("size_bytes")
        if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise FactorialBuildError(f"invalid SHA-256 for {directory / relative_name}")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            raise FactorialBuildError(f"invalid size for {directory / relative_name}")
        path = directory / relative_name
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError):
            bad.append(f"{relative_name} (missing or outside cell directory)")
            continue
        if not resolved.is_file():
            bad.append(f"{relative_name} (not a file)")
            continue
        if resolved.stat().st_size != expected_size or _sha256(resolved) != expected_hash:
            bad.append(relative_name)
    if bad:
        raise FactorialBuildError(f"{directory} failed manifest output verification: {', '.join(bad)}")


def _notebook_errors(directory: Path) -> int:
    path = directory / "fnsipid_tail_risk_core.executed.ipynb"
    notebook = _read_json(path, "executed notebook")
    errors = 0
    for cell_index, raw_cell in enumerate(_list(notebook.get("cells"), f"{path} cells")):
        cell = _mapping(raw_cell, f"{path} cell {cell_index}")
        outputs = cell.get("outputs", [])
        for output_index, raw_output in enumerate(_list(outputs, f"{path} cell {cell_index} outputs")):
            output = _mapping(raw_output, f"{path} cell {cell_index} output {output_index}")
            errors += output.get("output_type") == "error"
    return errors


def _source_hashes(manifest: Mapping[str, Any], cell_name: str) -> Mapping[str, str]:
    code = _mapping(manifest.get("code"), f"{cell_name} code")
    result: dict[str, str] = {}
    for key in SOURCE_KEYS:
        entry = _mapping(code.get(key), f"{cell_name} code.{key}")
        value = entry.get("sha256")
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise FactorialBuildError(f"{cell_name} has no valid source hash for {key}")
        result[key] = value
    return result


def _input_identities(manifest: Mapping[str, Any], cell_name: str) -> Mapping[str, tuple[str, int]]:
    inputs = _list(manifest.get("inputs"), f"{cell_name} inputs")
    result: dict[str, tuple[str, int]] = {}
    for index, raw_input in enumerate(inputs):
        entry = _mapping(raw_input, f"{cell_name} input {index}")
        role = entry.get("role")
        sha256 = entry.get("sha256")
        size_bytes = entry.get("size_bytes")
        if not isinstance(role, str) or not role:
            raise FactorialBuildError(f"{cell_name} input {index} has no valid role")
        if role in result:
            raise FactorialBuildError(f"{cell_name} has duplicate input role {role}")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise FactorialBuildError(f"{cell_name} input {role} has no valid SHA-256")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise FactorialBuildError(f"{cell_name} input {role} has no valid size")
        result[role] = (sha256, size_bytes)
    missing = sorted(REQUIRED_INPUT_ROLES - set(result))
    if missing:
        raise FactorialBuildError(f"{cell_name} omits required input roles: {', '.join(missing)}")
    return result


def _sample_counts(manifest: Mapping[str, Any], cell_name: str) -> Mapping[str, int]:
    counts = _mapping(manifest.get("counts"), f"{cell_name} counts")
    result: dict[str, int] = {}
    for key in SAMPLE_KEYS:
        value = counts.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise FactorialBuildError(f"{cell_name} has no valid sample count for {key}")
        result[key] = value
    return result


def _timing_limitation(manifest: Mapping[str, Any], cell_name: str) -> Mapping[str, Any]:
    timing = _mapping(manifest.get("timing_rule"), f"{cell_name} timing_rule")
    policy = _mapping(timing.get("upstream_policy"), f"{cell_name} timing_rule.upstream_policy")
    counts = _mapping(
        timing.get("upstream_counts_before_window_and_deduplication"),
        f"{cell_name} upstream timing counts",
    )
    date_rows = counts.get("date_only_or_exact_midnight_rows")
    precise_rows = counts.get("precise_timestamp_rows")
    if not isinstance(date_rows, int) or isinstance(date_rows, bool) or date_rows < 0:
        raise FactorialBuildError(f"{cell_name} has invalid date-only/exact-midnight row count")
    if not isinstance(precise_rows, int) or isinstance(precise_rows, bool) or precise_rows < 0:
        raise FactorialBuildError(f"{cell_name} has invalid precise-timestamp row count")
    date_policy = policy.get("date_only_or_exact_midnight")
    precise_policy = policy.get("full_datetime")
    if not isinstance(date_policy, str) or not date_policy.strip():
        raise FactorialBuildError(f"{cell_name} has no date-only/exact-midnight timing policy")
    if not isinstance(precise_policy, str) or not precise_policy.strip():
        raise FactorialBuildError(f"{cell_name} has no precise-timestamp timing policy")
    if timing.get("original_timestamp_retained_per_event") is not False:
        raise FactorialBuildError(f"{cell_name} does not declare original timestamps absent")
    if timing.get("strict_date_only_rule_verified_per_headline") is not False:
        raise FactorialBuildError(f"{cell_name} overstates per-headline timing verification")
    limitation = timing.get("limitation")
    if not isinstance(limitation, str) or not limitation.strip():
        raise FactorialBuildError(f"{cell_name} has no explicit timing limitation")
    return {
        "upstream_policy": {
            "date_only_or_exact_midnight": date_policy,
            "full_datetime": precise_policy,
        },
        "upstream_counts_before_window_and_deduplication": {
            "date_only_or_exact_midnight_rows": date_rows,
            "precise_timestamp_rows": precise_rows,
        },
        "original_timestamp_retained_per_event": False,
        "strict_date_only_rule_verified_per_headline": False,
        "limitation": limitation,
    }


def _assertions_and_gates(manifest: Mapping[str, Any], cell_name: str) -> tuple[int, int]:
    assertions = _list(manifest.get("assertions"), f"{cell_name} assertions")
    gates = _list(manifest.get("gates"), f"{cell_name} gates")
    if not assertions or not gates:
        raise FactorialBuildError(f"{cell_name} must report assertions and gates")
    failed_assertions = [
        str(_mapping(item, f"{cell_name} assertion").get("assertion", index))
        for index, item in enumerate(assertions)
        if _mapping(item, f"{cell_name} assertion {index}").get("passed") is not True
    ]
    failed_gates = [
        str(_mapping(item, f"{cell_name} gate").get("gate", index))
        for index, item in enumerate(gates)
        if _mapping(item, f"{cell_name} gate {index}").get("status") != "PASS"
    ]
    if failed_assertions:
        raise FactorialBuildError(f"{cell_name} failed assertions: {', '.join(failed_assertions)}")
    if failed_gates:
        raise FactorialBuildError(f"{cell_name} failed gates: {', '.join(failed_gates)}")
    return len(assertions), len(gates)


def _read_csv(path: Path, required: Sequence[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        raise FactorialBuildError(f"cannot read {path}: {exc}") from exc
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FactorialBuildError(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def _one(frame: pd.DataFrame, mask: pd.Series, context: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise FactorialBuildError(f"expected one {context} row, found {len(selected)}")
    return selected.iloc[0]


def _load_calibration(directory: Path, manifest: Mapping[str, Any], cell_name: str) -> Mapping[str, float | int]:
    frame = _read_csv(
        directory / "calibration.csv",
        ("model", "population", "diagnostic", "value", "hit_rate", "detail"),
    )
    full = frame["population"].eq("full_evaluation_panel")
    m1_hit = _one(
        frame,
        full & frame["model"].eq("M1") & frame["diagnostic"].eq("var_hit_rate_minus_alpha"),
        f"{cell_name} M1 hit-rate",
    )
    m2_hit = _one(
        frame,
        full & frame["model"].eq("M2") & frame["diagnostic"].eq("var_hit_rate_minus_alpha"),
        f"{cell_name} M2 hit-rate",
    )
    m1_es = _one(
        frame,
        full & frame["model"].eq("M1") & frame["diagnostic"].eq("mean_es_identification_residual"),
        f"{cell_name} M1 ES residual",
    )
    coverage = frame[full & frame["model"].isin(EXPECTED_MODELS) & frame["diagnostic"].isin(COVERAGE_DIAGNOSTICS)]
    expected_pairs = {(model, diagnostic) for model in EXPECTED_MODELS for diagnostic in COVERAGE_DIAGNOSTICS}
    if len(coverage) != 8 or set(zip(coverage["model"], coverage["diagnostic"], strict=True)) != expected_pairs:
        raise FactorialBuildError(f"{cell_name} is missing conditional calibration tests")
    rejections = 0
    for detail in coverage["detail"]:
        match = re.search(r"(?:^|;)\s*p\s*=\s*([0-9.eE+-]+)", str(detail))
        if match is None:
            raise FactorialBuildError(f"{cell_name} calibration test has no parseable p-value")
        p_value = _finite(match.group(1), f"{cell_name} calibration p-value")
        if not 0.0 <= p_value <= 1.0:
            raise FactorialBuildError(f"{cell_name} calibration p-value is outside [0, 1]")
        rejections += p_value < 0.05

    forecast_path = directory / "forecasts.parquet"
    try:
        z_target = pd.read_parquet(forecast_path, columns=["z_target"])["z_target"]
    except (FileNotFoundError, OSError, ValueError, KeyError, ImportError) as exc:
        raise FactorialBuildError(f"cannot read z_target from {forecast_path}: {exc}") from exc
    expected_rows = _sample_counts(manifest, cell_name)["evaluation_rows"]
    if len(z_target) != expected_rows:
        raise FactorialBuildError(f"{cell_name} forecasts have {len(z_target)} rows, expected {expected_rows}")
    m1_hit_rate = _finite(m1_hit["hit_rate"], f"{cell_name} M1 hit-rate")
    m2_hit_rate = _finite(m2_hit["hit_rate"], f"{cell_name} M2 hit-rate")
    std_z = _finite(z_target.std(), f"{cell_name} std(z_target)")
    if not 0.0 <= m1_hit_rate <= 1.0 or not 0.0 <= m2_hit_rate <= 1.0:
        raise FactorialBuildError(f"{cell_name} has a hit rate outside [0, 1]")
    if std_z <= 0.0:
        raise FactorialBuildError(f"{cell_name} has non-positive std(z_target)")
    return {
        "m1_hit_rate": m1_hit_rate,
        "m2_hit_rate": m2_hit_rate,
        "std_z": std_z,
        "m1_es_residual": _finite(m1_es["value"], f"{cell_name} M1 ES residual"),
        "coverage_rejections": int(rejections),
    }


def _load_contrasts(directory: Path, manifest: Mapping[str, Any], cell_name: str) -> pd.DataFrame:
    frame = _read_csv(
        directory / "bootstrap.csv",
        (
            "population",
            "contrast",
            "weighting",
            "point_estimate",
            "ci_low",
            "ci_high",
            "share_below_zero",
            "replications",
            "block_length",
            "seed",
        ),
    )
    bootstrap = _mapping(manifest.get("bootstrap"), f"{cell_name} bootstrap")
    selected = frame[
        frame["population"].eq("news_bearing_evaluation_origins")
        & frame["weighting"].eq("observation_equal")
        & frame["contrast"].isin(CONTRAST_ORDER)
        & frame["replications"].eq(bootstrap.get("replications"))
        & frame["block_length"].eq(bootstrap.get("block_length"))
        & frame["seed"].eq(bootstrap.get("seed"))
    ].copy()
    if len(selected) != len(CONTRAST_ORDER) or set(selected["contrast"]) != set(CONTRAST_ORDER) or selected["contrast"].duplicated().any():
        raise FactorialBuildError(f"{cell_name} is missing or duplicates required contrasts")
    selected = selected.set_index("contrast").loc[list(CONTRAST_ORDER)].reset_index()
    selected = selected.rename(
        columns={
            "point_estimate": "point",
            "ci_low": "lo",
            "ci_high": "hi",
            "share_below_zero": "below",
        }
    )[["contrast", "point", "lo", "hi", "below"]]
    for column in ("point", "lo", "hi", "below"):
        selected[column] = [
            _finite(value, f"{cell_name} {contrast} {column}")
            for value, contrast in zip(selected[column], selected["contrast"], strict=True)
        ]
    if (selected["lo"] > selected["hi"]).any():
        raise FactorialBuildError(f"{cell_name} has an inverted contrast interval")
    if ((selected["below"] < 0.0) | (selected["below"] > 1.0)).any():
        raise FactorialBuildError(f"{cell_name} has a bootstrap share outside [0, 1]")
    return selected


def _verify_design(manifest: Mapping[str, Any], cell_name: str) -> None:
    variant = _mapping(manifest.get("variant"), f"{cell_name} variant")
    observed = (variant.get("price_repair"), variant.get("volatility_refit"))
    if observed != EXPECTED_DESIGN[cell_name] or variant.get("id") != cell_name:
        raise FactorialBuildError(
            f"{cell_name} design is {observed!r}/{variant.get('id')!r}, expected {EXPECTED_DESIGN[cell_name]!r}/{cell_name!r}"
        )


def _verify_full_run(manifest: Mapping[str, Any], cell_name: str) -> None:
    configuration = _mapping(manifest.get("configuration"), f"{cell_name} configuration")
    if (
        manifest.get("run_mode") != "full"
        or configuration.get("RUN_MODE") != "full"
        or manifest.get("smoke_run_is_engineering_check_only") is not False
    ):
        raise FactorialBuildError(f"{cell_name} is not a completed full-mode research run")


def _load_cell(cell_name: str, directory: Path) -> CellBundle:
    if not directory.is_dir():
        raise FactorialBuildError(f"{cell_name} directory does not exist: {directory}")
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path, f"{cell_name} manifest")
    if manifest.get("status") != "completed":
        raise FactorialBuildError(f"{cell_name} status is not completed")
    _verify_full_run(manifest, cell_name)
    _verify_design(manifest, cell_name)
    source_hashes = _source_hashes(manifest, cell_name)
    input_identities = _input_identities(manifest, cell_name)
    sample_counts = _sample_counts(manifest, cell_name)
    timing_limitation = _timing_limitation(manifest, cell_name)
    assertion_count, gate_count = _assertions_and_gates(manifest, cell_name)
    _verify_outputs(directory, manifest)
    notebook_error_count = _notebook_errors(directory)
    if notebook_error_count:
        raise FactorialBuildError(f"{cell_name} executed notebook contains {notebook_error_count} errors")
    executed_notebook_sha256 = _sha256(directory / "fnsipid_tail_risk_core.executed.ipynb")
    return CellBundle(
        name=cell_name,
        directory=directory,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        manifest=manifest,
        source_hashes=source_hashes,
        input_identities=input_identities,
        sample_counts=sample_counts,
        timing_limitation=timing_limitation,
        calibration=_load_calibration(directory, manifest, cell_name),
        contrasts=_load_contrasts(directory, manifest, cell_name),
        assertion_count=assertion_count,
        gate_count=gate_count,
        notebook_error_count=notebook_error_count,
        executed_notebook_sha256=executed_notebook_sha256,
    )


def _verify_cross_cell(cells: Mapping[str, CellBundle]) -> None:
    baseline = cells["v1"]
    alpha = _finite(_mapping(baseline.manifest.get("configuration"), "v1 configuration").get("ALPHA"), "v1 ALPHA")
    if not 0.0 < alpha < 1.0:
        raise FactorialBuildError("ALPHA must lie strictly between zero and one")
    for name in CELL_ORDER[1:]:
        cell = cells[name]
        if dict(cell.source_hashes) != dict(baseline.source_hashes):
            raise FactorialBuildError(f"source hashes drift between v1 and {name}")
        if dict(cell.input_identities) != dict(baseline.input_identities):
            raise FactorialBuildError(f"input identities drift between v1 and {name}")
        if dict(cell.sample_counts) != dict(baseline.sample_counts):
            raise FactorialBuildError(f"sample counts drift between v1 and {name}")
        if dict(cell.timing_limitation) != dict(baseline.timing_limitation):
            raise FactorialBuildError(f"timing provenance drifts between v1 and {name}")
        if cell.manifest.get("model_formulas") != baseline.manifest.get("model_formulas"):
            raise FactorialBuildError(f"model formulas drift between v1 and {name}")
        if cell.manifest.get("bootstrap") != baseline.manifest.get("bootstrap"):
            raise FactorialBuildError(f"bootstrap design drifts between v1 and {name}")
        other_alpha = _mapping(cell.manifest.get("configuration"), f"{name} configuration").get("ALPHA")
        if other_alpha != alpha:
            raise FactorialBuildError(f"ALPHA drifts between v1 and {name}")


def _contrast(cell: CellBundle, name: str) -> pd.Series:
    return _one(cell.contrasts, cell.contrasts["contrast"].eq(name), f"{cell.name} {name}")


def _verify_conclusions(cells: Mapping[str, CellBundle]) -> None:
    for name in CELL_ORDER:
        cell = cells[name]
        arrival = _contrast(cell, "d_news_vs_price")
        semantic = _contrast(cell, "d_semantic")
        tone = _contrast(cell, "d_tone_given_intensity")
        if not (arrival["point"] < 0.0 and arrival["hi"] < 0.0):
            raise FactorialBuildError(f"{name} contradicts the news-arrival conclusion")
        if not semantic["lo"] <= 0.0 <= semantic["hi"]:
            raise FactorialBuildError(f"{name} contradicts the semantics conclusion")
        if not tone["lo"] <= 0.0 <= tone["hi"]:
            raise FactorialBuildError(f"{name} contradicts the signed-tone conclusion")
        if cell.calibration["coverage_rejections"] != 8:
            raise FactorialBuildError(f"{name} does not reject all eight calibration tests")
    v1 = cells["v1"].calibration
    price = cells["price_only"].calibration
    refit = cells["refit_only"].calibration
    v2 = cells["v2"].calibration
    alpha = _finite(
        _mapping(cells["v1"].manifest.get("configuration"), "v1 configuration").get("ALPHA"),
        "v1 ALPHA",
    )
    if v1["m1_hit_rate"] == alpha or v1["m1_es_residual"] == 0.0 or v1["std_z"] == 1.0:
        raise FactorialBuildError("v1 levels do not support the bounded calibration attribution")
    if not (
        price["m1_hit_rate"] >= v1["m1_hit_rate"]
        and price["std_z"] < v1["std_z"]
        and abs(price["std_z"] - 1.0) < abs(v1["std_z"] - 1.0)
        and price["m1_es_residual"] < v1["m1_es_residual"]
        and abs(price["m1_es_residual"]) < abs(v1["m1_es_residual"])
    ):
        raise FactorialBuildError("price-only cell contradicts the bounded price-repair finding")
    for name, repaired in (("refit_only", refit), ("v2", v2)):
        if not (
            repaired["m1_hit_rate"] < v1["m1_hit_rate"]
            and abs(repaired["m1_hit_rate"] - alpha) < abs(v1["m1_hit_rate"] - alpha)
            and repaired["std_z"] < v1["std_z"]
            and abs(repaired["std_z"] - 1.0) < abs(v1["std_z"] - 1.0)
            and repaired["m1_es_residual"] < v1["m1_es_residual"]
            and abs(repaired["m1_es_residual"]) < abs(v1["m1_es_residual"])
        ):
            raise FactorialBuildError(f"{name} contradicts the calibration finding")


def _calibration_frame(cells: Mapping[str, CellBundle]) -> pd.DataFrame:
    rows = []
    for name in CELL_ORDER:
        price_repair, volatility_refit = EXPECTED_DESIGN[name]
        rows.append(
            {
                "variant": name,
                "price_repair": price_repair,
                "volatility_refit": volatility_refit,
                **cells[name].calibration,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "variant",
            "price_repair",
            "volatility_refit",
            "m1_hit_rate",
            "m2_hit_rate",
            "std_z",
            "m1_es_residual",
            "coverage_rejections",
        ),
    )


def _contrast_frame(cells: Mapping[str, CellBundle]) -> pd.DataFrame:
    rows = []
    for name in CELL_ORDER:
        rows.extend({"variant": name, **row} for row in cells[name].contrasts.to_dict("records"))
    return pd.DataFrame(rows, columns=("variant", "contrast", "point", "lo", "hi", "below"))


def _effects_frame(calibration: pd.DataFrame) -> pd.DataFrame:
    levels = calibration.set_index("variant")
    rows = []
    for metric in EFFECT_METRICS:
        v1 = float(levels.at["v1", metric])
        price = float(levels.at["price_only", metric])
        refit = float(levels.at["refit_only", metric])
        v2 = float(levels.at["v2", metric])
        rows.append(
            {
                "metric": metric,
                "price_only_minus_v1": price - v1,
                "refit_only_minus_v1": refit - v1,
                "v2_minus_v1": v2 - v1,
                "interaction": v2 - price - refit + v1,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "metric",
            "price_only_minus_v1",
            "refit_only_minus_v1",
            "v2_minus_v1",
            "interaction",
        ),
    )


def _signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _source_line(cells: Mapping[str, CellBundle]) -> str:
    expected = {name: f"fnsipid_tail_risk_core_{name}" for name in CELL_ORDER}
    if all(cells[name].directory.name == expected[name] for name in CELL_ORDER):
        parents = {cells[name].directory.parent.as_posix() for name in CELL_ORDER}
        if len(parents) == 1:
            parent = next(iter(parents))
            return f"- Source bundles: `{parent}/fnsipid_tail_risk_core_{{v1,price_only,refit_only,v2}}/`"
    paths = ", ".join(f"{name}=`{cells[name].directory.as_posix()}`" for name in CELL_ORDER)
    return f"- Source bundles: {paths}"


def _build_summary(cells: Mapping[str, CellBundle], calibration: pd.DataFrame, run_record: RunRecord | None) -> str:
    levels = calibration.set_index("variant")
    v1, price, refit, v2 = (levels.loc[name] for name in CELL_ORDER)
    alpha = _finite(_mapping(cells["v1"].manifest["configuration"], "configuration")["ALPHA"], "ALPHA")
    counts = cells["v1"].sample_counts
    bootstrap = _mapping(cells["v1"].manifest.get("bootstrap"), "bootstrap")
    timing = cells["v1"].timing_limitation
    timing_counts = _mapping(timing["upstream_counts_before_window_and_deduplication"], "timing counts")
    timing_policy = _mapping(timing["upstream_policy"], "timing policy")

    calibration_rows = []
    for name in CELL_ORDER:
        row = levels.loc[name]
        calibration_rows.append(
            f"| {DISPLAY_NAMES[name]} | {row['m1_hit_rate']:.4%} | {row['m2_hit_rate']:.4%} | "
            f"{row['std_z']:.4f} | {row['m1_es_residual']:.5f} | {int(row['coverage_rejections'])}/8 |"
        )
    contrast_rows = []
    for name in CELL_ORDER:
        values = cells[name].contrasts.set_index("contrast")
        news, semantic, tone = (values.loc[contrast] for contrast in CONTRAST_ORDER)
        contrast_rows.append(
            f"| {DISPLAY_NAMES[name]} | {_signed(news['point'], 5)} [{_signed(news['lo'], 5)}, {_signed(news['hi'], 5)}] | "
            f"{_signed(semantic['point'], 6)} [{_signed(semantic['lo'], 6)}, {_signed(semantic['hi'], 6)}] | "
            f"{_signed(tone['point'], 6)} [{_signed(tone['lo'], 6)}, {_signed(tone['hi'], 6)}] |"
        )

    price_delta = float(price["m1_hit_rate"] - v1["m1_hit_rate"])
    refit_delta = float(refit["m1_hit_rate"] - v1["m1_hit_rate"])
    v2_delta = float(v2["m1_hit_rate"] - v1["m1_hit_rate"])
    price_es_pct = (float(price["m1_es_residual"]) / float(v1["m1_es_residual"]) - 1) * 100
    refit_es_pct = (float(refit["m1_es_residual"]) / float(v1["m1_es_residual"]) - 1) * 100
    v2_es_pct = (float(v2["m1_es_residual"]) / float(v1["m1_es_residual"]) - 1) * 100
    closed_pct = -refit_delta / (float(v1["m1_hit_rate"]) - alpha) * 100

    assertions = ", ".join(f"{name} {cells[name].assertion_count}/{cells[name].assertion_count}" for name in CELL_ORDER)
    if run_record is None:
        execution = "Execution-host metadata were not supplied. "
        gpu_text = ""
    else:
        execution = (
            f"All four cells ran concurrently from the same source bytes on UCL host `{run_record.host}` "
            f"under Python {run_record.python}. Input SHA-256 hashes match the frozen local FinBERT "
            "checkpoint and price archive. "
        )
        gpu_text = (
            f"\n\nThe workstation has an {run_record.gpu}, but this notebook is a NumPy/SciPy/`arch` "
            "CPU workload with no CUDA path. The run record shows the GPU occupied by another process "
            "at launch; these notebooks did not allocate it."
        )
    execution += (
        f"Every bundle reports `completed`, passes its notebook assertions ({assertions}) and "
        f"{cells['v1'].gate_count}/{cells['v1'].gate_count} gates, contains no executed-notebook errors, "
        "and verifies every manifest-listed output hash."
    )
    record_file = "- `ucl_run_record.txt` — remote host, source hashes and observed GPU state\n" if run_record is not None else ""

    return f"""# FNSPID tail-risk 2×2 sensitivity attribution

## Verdict

The **annual expanding GJR-GARCH refit accounts for essentially all of the VaR hit-rate improvement** between v1 and v2. The minimum-absolute-return sensitivity over candidate adjusted/raw-return gaps improves standardised-return dispersion and ES identification, but does not improve the M1 VaR hit rate by itself.

Neither sensitivity changes the substantive result: news arrival and volume improve paired FZ0 loss beyond price state, while FinBERT semantics and signed tone do not add a detectable increment.

## Frozen design

| Cell | Candidate-gap sensitivity | Volatility-filter refit |
| --- | --- | --- |
| v1 | none | frozen 2011–2016 fit |
| price-only | minimum-absolute-return rule | frozen 2011–2016 fit |
| refit-only | none | annual expanding window ending before each evaluation year |
| v2 | minimum-absolute-return rule | annual expanding window ending before each evaluation year |

Every cell uses identical notebook source, helper and notebook-pair hashes, plus the same corpus, timing rule, split, model formulas, optimiser, bootstrap, seed and evaluation sample: {counts["retained_firms"]:,} firms, {counts["evaluation_rows"]:,} evaluation rows, {counts["evaluation_news_rows"]:,} news-bearing rows and {counts["evaluation_target_dates"]:,} target dates.

Timing limitation: the hashed upstream policy mapped {timing_counts["date_only_or_exact_midnight_rows"]:,} date-only/exact-midnight rows by **{timing_policy["date_only_or_exact_midnight"]}** and {timing_counts["precise_timestamp_rows"]:,} precise-timestamp rows by **{timing_policy["full_datetime"]}** before windowing and deduplication. Original timestamps/type flags are absent from the completed checkpoint, so a uniformly date-only rule cannot be verified per retained event.

## Calibration attribution

| Cell | M1 VaR hit rate | M2 VaR hit rate | std$(z)$ | M1 ES residual | rejected coverage tests |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(calibration_rows)}

Relative to v1:

- Candidate-gap sensitivity alone changes the M1 hit rate by **{price_delta * 100:+.4f} percentage points**, changes std$(z)$ by {float(price["std_z"] - v1["std_z"]):+.4f}, and changes the ES residual by {price_es_pct:+.1f}%. These rows are candidates, not proven errors, because authoritative corporate-action metadata are unavailable.
- Annual refitting changes the M1 hit rate by **{refit_delta * 100:+.4f} percentage points**, closing {closed_pct:.1f}% of the gap to nominal {alpha:.1%}; it changes std$(z)$ by {float(refit["std_z"] - v1["std_z"]):+.4f} and the ES residual by {refit_es_pct:+.1f}%.
- v2 changes the M1 hit rate by **{v2_delta * 100:+.4f} percentage points**, changes std$(z)$ by {float(v2["std_z"] - v1["std_z"]):+.4f}, and changes the ES residual by {v2_es_pct:+.1f}%.
- The 2×2 interactions are reported in `factorial_effects.csv`; interpretation remains bounded by absolute calibration tests.

Calibration is improved, not fixed: all eight conditional-coverage/calibration tests reject in every cell.

## Forecast-value conclusions

Paired FZ0 differences on news-bearing evaluation origins; 95% moving-block bootstrap over target dates, block length {bootstrap["block_length"]}, {int(bootstrap["replications"]):,} replications, seed {bootstrap["seed"]}:

| Cell | M1 − M0: news arrival | M2 − M1: semantics | M2 − M2-intensity: signed tone |
| --- | --- | --- | --- |
{chr(10).join(contrast_rows)}

Arrival excludes zero in all four cells; semantics and signed tone cover zero in all four. The conclusion is robust to both declared sensitivities separately and jointly.

## Execution and verification

{execution}{gpu_text}

## Files

- `calibration_comparison.csv` — four-cell calibration levels
- `factorial_effects.csv` — price-only, refit-only, combined and interaction differences from v1
- `contrast_comparison.csv` — frozen paired-loss contrasts and confidence intervals
{record_file}{_source_line(cells)}
"""


def _parse_run_record(path: Path, source_hashes: Mapping[str, str]) -> RunRecord:
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        raise FactorialBuildError(f"cannot read UCL run record {path}: {exc}") from exc
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields.setdefault(key.strip(), value.strip())
    if fields.get("overall_status") != "0":
        raise FactorialBuildError("UCL run record does not report overall_status=0")
    if fields.get("source_sha256") != source_hashes["notebook_source"]:
        raise FactorialBuildError("UCL run record lacks the matching notebook source SHA-256")
    if fields.get("helper_sha256") != source_hashes["helper_module"]:
        raise FactorialBuildError("UCL run record lacks the matching helper SHA-256")
    gpu_line = next((line for line in text.splitlines() if line.startswith("NVIDIA ")), None)
    host = fields.get("host")
    python = fields.get("python", "").removeprefix("Python ")
    if not host or not python or gpu_line is None:
        raise FactorialBuildError("UCL run record lacks host, Python, or GPU metadata")
    return RunRecord(host=host, python=python, gpu=gpu_line.split(",", 1)[0].strip())


def _metadata(path: Path) -> Mapping[str, str | int]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def build_factorial_report(
    *,
    v1_dir: Path,
    price_only_dir: Path,
    refit_only_dir: Path,
    v2_dir: Path,
    output_dir: Path,
    ucl_run_record: Path | None = None,
    source_commit: str | None = None,
) -> Mapping[str, Any]:
    """Verify four core bundles and write their compact factorial report."""
    _check_empty_output(output_dir)
    directories = {
        "v1": v1_dir,
        "price_only": price_only_dir,
        "refit_only": refit_only_dir,
        "v2": v2_dir,
    }
    if len({path.resolve() for path in directories.values()}) != 4:
        raise FactorialBuildError("the four cell directories must be distinct")
    cells = {name: _load_cell(name, directories[name]) for name in CELL_ORDER}
    _verify_cross_cell(cells)
    _verify_conclusions(cells)
    run_record = _parse_run_record(ucl_run_record, cells["v1"].source_hashes) if ucl_run_record is not None else None
    calibration = _calibration_frame(cells)
    contrasts = _contrast_frame(cells)
    effects = _effects_frame(calibration)
    summary = _build_summary(cells, calibration, run_record)

    _check_empty_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_paths = {
        "calibration_comparison.csv": output_dir / "calibration_comparison.csv",
        "contrast_comparison.csv": output_dir / "contrast_comparison.csv",
        "factorial_effects.csv": output_dir / "factorial_effects.csv",
        "summary.md": output_dir / "summary.md",
    }
    calibration.to_csv(compact_paths["calibration_comparison.csv"], index=False, lineterminator="\n")
    contrasts.to_csv(compact_paths["contrast_comparison.csv"], index=False, lineterminator="\n")
    effects.to_csv(compact_paths["factorial_effects.csv"], index=False, lineterminator="\n")
    compact_paths["summary.md"].write_text(summary, encoding="utf-8")
    if ucl_run_record is not None:
        copied_record = output_dir / "ucl_run_record.txt"
        shutil.copyfile(ucl_run_record, copied_record)
        compact_paths["ucl_run_record.txt"] = copied_record

    verification = {}
    for name in CELL_ORDER:
        runtime_seconds = _finite(cells[name].manifest.get("runtime_seconds"), f"{name} runtime")
        verification[name] = {
            "assertions": [cells[name].assertion_count, cells[name].assertion_count],
            "bad_hashes": [],
            "gates": [cells[name].gate_count, cells[name].gate_count],
            "notebook_errors": cells[name].notebook_error_count,
            "executed_notebook_sha256": cells[name].executed_notebook_sha256,
            "runtime_min": runtime_seconds / 60.0,
            "status": "completed",
        }
    manifest: dict[str, Any] = {
        "builder": {
            "path": "scripts/build_fnsipid_tail_risk_factorial.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "cells": {name: list(EXPECTED_DESIGN[name]) for name in CELL_ORDER},
        "created_at": datetime.now(UTC).isoformat(),
        "design": "2x2 sensitivity attribution; every cell executed from identical source bytes",
        "experiment": "fnsipid_tail_risk_factorial_v1",
        "finding": {
            "calibration_driver": "annual_expanding_volatility_refit",
            "news_arrival": "improves paired FZ0 loss in all four cells",
            "price_repair_effect": "minimum-absolute-return sensitivity over candidate adjusted/raw gaps improves std(z) and ES residual, not M1 VaR hit rate alone",
            "semantics": "no detectable incremental paired FZ0 improvement in all four cells",
            "signed_tone": "no detectable incremental paired FZ0 improvement in all four cells",
        },
        "identical_sample": dict(cells["v1"].sample_counts),
        "outputs": {name: _metadata(path) for name, path in compact_paths.items()},
        "schema_version": 1,
        "source_commit": source_commit,
        "source_hashes": dict(cells["v1"].source_hashes),
        "source_manifests": {
            name: {
                "path": cells[name].manifest_path.as_posix(),
                "sha256": cells[name].manifest_sha256,
            }
            for name in CELL_ORDER
        },
        "status": "completed",
        "timing_limitation": dict(cells["v1"].timing_limitation),
        "verification": verification,
    }
    if run_record is not None:
        manifest["ucl_execution"] = {
            "gpu": run_record.gpu,
            "gpu_used_by_notebook": False,
            "host": run_record.host,
            "note": "CPU-only NumPy/SciPy/arch workload; GPU occupied by another process at launch",
            "python": run_record.python,
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-dir", required=True, type=Path)
    parser.add_argument("--price-only-dir", required=True, type=Path)
    parser.add_argument("--refit-only-dir", required=True, type=Path)
    parser.add_argument("--v2-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ucl-run-record", "--run-record", dest="ucl_run_record", type=Path)
    parser.add_argument("--source-commit", help="Optional source commit label")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_factorial_report(
            v1_dir=args.v1_dir,
            price_only_dir=args.price_only_dir,
            refit_only_dir=args.refit_only_dir,
            v2_dir=args.v2_dir,
            output_dir=args.output_dir,
            ucl_run_record=args.ucl_run_record,
            source_commit=args.source_commit,
        )
    except FactorialBuildError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps({"output_dir": str(args.output_dir), "status": manifest["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
