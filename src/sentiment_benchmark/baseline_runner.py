"""Orchestrate non-LLM baselines and persist them like any other benchmark run.

Baselines are written to the same SQLite store as LLM runs (same ``runs``,
``responses`` and ``metrics`` tables) so they appear in the TUI runs list and
can be exported and compared with identical tooling. A sentinel prompt is
recorded to satisfy the schema; baselines do not use prompts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .baselines import DEFAULT_BASELINES, predict_baseline
from .constants import ALLOWED_LABELS, DEFAULT_DATASET_PATH, DEFAULT_DB_PATH, DEFAULT_PILOT_PER_CLASS, DEFAULT_SEED
from .dataset import load_dataset, select_rows
from .metrics import evaluate_responses
from .models import LLMResponseRecord, PromptConfig, RunConfig, RunMode
from .prompts import make_prompt
from .storage import BenchmarkStore

ProgressCallback = Callable[[str], None]

BASELINE_MODEL_PREFIX = "baseline/"


def baseline_prompt() -> PromptConfig:
    """Sentinel prompt recorded for baseline runs (baselines use no prompt)."""
    return make_prompt(
        prompt_id="baseline",
        system_prompt="Non-LLM baseline classifier; no prompt is used.",
        user_template="{sentence}",
        output_mode="label_only",
    )


@dataclass(frozen=True)
class BaselineRunSummary:
    run_id: int
    selected_row_count: int
    baseline_count: int
    status: str = "completed"


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def run_baselines(
    baselines: list[str] | None = None,
    *,
    mode: RunMode = "pilot",
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    db_path: str = str(DEFAULT_DB_PATH),
    sample_per_class: int = DEFAULT_PILOT_PER_CLASS,
    seed: int = DEFAULT_SEED,
    folds: int = 5,
    match_run_id: int | None = None,
    callback: ProgressCallback | None = None,
) -> BaselineRunSummary:
    names = list(baselines) if baselines else list(DEFAULT_BASELINES)
    store = BenchmarkStore(db_path)
    store.initialize()

    rows = load_dataset(dataset_path)
    store.upsert_dataset(rows, dataset_path)
    prompt = baseline_prompt()
    store.save_prompt(prompt)

    if match_run_id is not None:
        selected_numbers = set(store.get_run_selected_rows(match_run_id))
        selected_rows = [row for row in rows if row.row_number in selected_numbers]
    else:
        selected_rows = select_rows(rows, mode, sample_per_class, seed)

    model_ids = [f"{BASELINE_MODEL_PREFIX}{name}" for name in names]
    config = RunConfig(
        models=model_ids,
        prompt=prompt,
        mode=mode,
        dataset_path=dataset_path,
        db_path=db_path,
        base_url="local-baseline",
        sample_per_class=sample_per_class,
        seed=seed,
        temperature=0.0,
        max_completion_tokens=0,
        concurrency=1,
        retries=0,
    )
    run_id = store.create_run(config, [row.row_number for row in selected_rows])

    for name in names:
        model_id = f"{BASELINE_MODEL_PREFIX}{name}"
        store.upsert_run_model(run_id, model_id, "running")
        _notify(callback, f"Running baseline {name} on {len(selected_rows)} rows")
        predictions = predict_baseline(name, selected_rows, folds=folds, seed=seed)
        for row, label in zip(selected_rows, predictions, strict=True):
            valid = label in ALLOWED_LABELS
            record = LLMResponseRecord(
                row_number=row.row_number,
                model_id=model_id,
                prompt_hash=prompt.prompt_hash,
                raw_content=str(label) if label is not None else None,
                normalized_label=label if valid else None,
                parse_status="valid" if valid else "invalid",
                status="success",
            )
            store.save_response(run_id, record)

        responses = store.fetch_responses(run_id, model_id)
        primary = evaluate_responses(selected_rows, responses, model_id=model_id, scope="primary")
        audit = evaluate_responses(selected_rows, responses, model_id=model_id, scope="all")
        store.save_metrics(run_id, primary)
        store.save_metrics(run_id, audit)
        store.upsert_run_model(run_id, model_id, "completed")
        _notify(callback, f"Completed baseline {name}: primary accuracy={primary.accuracy:.4f}")

    store.mark_run_complete(run_id, status="completed")
    return BaselineRunSummary(
        run_id=run_id,
        selected_row_count=len(selected_rows),
        baseline_count=len(names),
    )
