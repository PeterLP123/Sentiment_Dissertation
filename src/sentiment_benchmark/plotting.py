"""Publication-ready figures for a benchmark run.

Figures are generated from already-parsed metric/statistics structures (not the
database) so the plotting code stays decoupled and easy to test. matplotlib is an
optional dependency: if it is not installed, :func:`generate_figures` returns an
empty list instead of raising, mirroring the optional-baseline pattern.

Figures produced (when the underlying data is present):

* ``confusion_<model>.png`` — per-model confusion-matrix heatmap.
* ``per_class_f1_<model>.png`` — per-model precision/recall/F1 bars by class.
* ``accuracy_ci_forest.png`` — accuracy point + bootstrap CI across models.
* ``leaderboard.png`` — models ranked by accuracy and macro-F1.
* ``pareto_quality_cost.png`` — macro-F1 vs provider cost with the Pareto frontier.
* ``pareto_quality_latency.png`` — macro-F1 vs p95 latency with the Pareto frontier.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .constants import ALLOWED_LABELS

# Predicted columns include the two pseudo-labels the evaluator emits.
_PRED_LABELS = (*ALLOWED_LABELS, "__invalid__", "__error__")
_PRED_DISPLAY = (*ALLOWED_LABELS, "invalid", "error")


def matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(" ", "_")


def _metrics_for_scope(metric_records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    rows = [record for record in metric_records if record.get("scope") == scope]
    rows.sort(key=lambda record: (-float(record.get("accuracy", 0.0)), str(record.get("model_id"))))
    return rows


def _plot_confusion(plt: Any, record: dict[str, Any], output_dir: Path) -> Path | None:
    matrix = record.get("confusion_matrix") or {}
    if not matrix:
        return None
    grid = [[int((matrix.get(actual) or {}).get(pred, 0) or 0) for pred in _PRED_LABELS] for actual in ALLOWED_LABELS]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    image = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(_PRED_DISPLAY)), labels=_PRED_DISPLAY)
    ax.set_yticks(range(len(ALLOWED_LABELS)), labels=list(ALLOWED_LABELS))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {record['model_id']} ({record['scope']})")
    largest = max((value for row in grid for value in row), default=0)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if largest and value > largest / 2 else "black",
            )
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    path = output_dir / f"confusion_{_safe_name(record['model_id'])}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_per_class(plt: Any, record: dict[str, Any], output_dir: Path) -> Path | None:
    per_class = record.get("per_class") or {}
    if not per_class:
        return None
    classes = [label for label in ALLOWED_LABELS if label in per_class]
    if not classes:
        return None
    metrics = ("precision", "recall", "f1")
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for offset, metric in enumerate(metrics):
        positions = [index + (offset - 1) * width for index in range(len(classes))]
        values = [float(per_class[label].get(metric, 0.0)) for label in classes]
        ax.bar(positions, values, width=width, label=metric.title())
    ax.set_xticks(range(len(classes)), labels=classes)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class scores — {record['model_id']} ({record['scope']})")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"per_class_f1_{_safe_name(record['model_id'])}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_accuracy_ci_forest(plt: Any, statistics: dict[str, Any], output_dir: Path) -> Path | None:
    per_model = statistics.get("per_model") or {}
    if not per_model:
        return None
    items = []
    for model_id, model_stats in per_model.items():
        ci = model_stats.get("accuracy_ci") or {}
        if "point" not in ci:
            continue
        items.append((str(model_id), float(ci["point"]), float(ci.get("lower", ci["point"])), float(ci.get("upper", ci["point"]))))
    if not items:
        return None
    items.sort(key=lambda item: item[1])
    labels = [item[0] for item in items]
    points = [item[1] for item in items]
    lowers = [item[1] - item[2] for item in items]
    uppers = [item[3] - item[1] for item in items]

    confidence = 95
    sample = next(iter(per_model.values())).get("accuracy_ci", {})
    if isinstance(sample.get("confidence"), (int, float)):
        confidence = round(float(sample["confidence"]) * 100)

    fig, ax = plt.subplots(figsize=(7.0, 0.6 * len(items) + 1.5))
    ax.errorbar(points, range(len(items)), xerr=[lowers, uppers], fmt="o", capsize=4, color="#1f77b4")
    ax.set_yticks(range(len(items)), labels=labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Accuracy with {confidence}% bootstrap CI ({statistics.get('scope', 'primary')} scope)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = output_dir / "accuracy_ci_forest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices of the points on the (minimize x, maximize y) Pareto frontier.

    Returned in ascending-x order. A point is on the frontier when no other point
    has both lower-or-equal x and strictly higher y.
    """
    order = sorted(range(len(points)), key=lambda index: (points[index][0], -points[index][1]))
    frontier: list[int] = []
    best_y = float("-inf")
    for index in order:
        if points[index][1] > best_y:
            frontier.append(index)
            best_y = points[index][1]
    return frontier


def _operational_items(
    scope_records: list[dict[str, Any]],
    operational: dict[str, Any],
    x_keys: tuple[str, str],
) -> list[tuple[str, float, float]]:
    """Join per-model macro-F1 with one operational quantity as (model, x, macro_f1)."""
    section, key = x_keys
    items: list[tuple[str, float, float]] = []
    for record in scope_records:
        model_id = str(record["model_id"])
        value = ((operational.get(model_id) or {}).get(section) or {}).get(key)
        if isinstance(value, (int, float)):
            items.append((model_id, float(value), float(record.get("macro_f1", 0.0))))
    return items


def _plot_pareto(
    plt: Any,
    items: list[tuple[str, float, float]],
    x_label: str,
    title: str,
    filename: str,
    output_dir: Path,
) -> Path | None:
    """Quality-vs-resource scatter with the Pareto frontier highlighted.

    Skipped with fewer than two models because a single point has no frontier
    worth publishing.
    """
    if len(items) < 2:
        return None
    points = [(x, y) for _, x, y in items]
    frontier = pareto_frontier(points)
    frontier_set = set(frontier)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    xs = [x for _, x, _ in items]
    for index, (model_id, x, y) in enumerate(items):
        on_frontier = index in frontier_set
        ax.scatter(
            x,
            y,
            s=70 if on_frontier else 45,
            color="#d62728" if on_frontier else "#1f77b4",
            zorder=3,
            label=None,
        )
        ax.annotate(
            model_id,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    frontier_xs = [points[index][0] for index in frontier]
    frontier_ys = [points[index][1] for index in frontier]
    ax.step(frontier_xs, frontier_ys, where="post", color="#d62728", linestyle="--", alpha=0.7, zorder=2,
            label="Pareto frontier (red points)")

    # Costs commonly span orders of magnitude; switch to log when linear would
    # squash the cheap models into the axis.
    if min(xs) > 0 and max(xs) / min(xs) > 25:
        ax.set_xscale("log")
        x_label = f"{x_label} (log scale)"
    ax.margins(x=0.15)  # keep model-name annotations inside the axes
    ax.set_xlabel(x_label)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_leaderboard(plt: Any, scope_records: list[dict[str, Any]], scope: str, output_dir: Path) -> Path | None:
    if not scope_records:
        return None
    labels = [str(record["model_id"]) for record in scope_records]
    accuracy = [float(record.get("accuracy", 0.0)) for record in scope_records]
    macro_f1 = [float(record.get("macro_f1", 0.0)) for record in scope_records]
    width = 0.4

    fig, ax = plt.subplots(figsize=(7.0, 0.6 * len(labels) + 2.0))
    positions = range(len(labels))
    ax.barh([p + width / 2 for p in positions], accuracy, height=width, label="Accuracy", color="#1f77b4")
    ax.barh([p - width / 2 for p in positions], macro_f1, height=width, label="Macro F1", color="#ff7f0e")
    ax.set_yticks(list(positions), labels=labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title(f"Model leaderboard ({scope} scope)")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "leaderboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_figures(
    metric_records: list[dict[str, Any]],
    statistics: dict[str, Any],
    output_dir: str | Path,
    scope: str = "primary",
) -> list[Path]:
    """Write figures for a run and return the paths created.

    Returns an empty list if matplotlib is unavailable or there is no data to plot.
    """
    if not matplotlib_available():
        return []

    import matplotlib

    matplotlib.use("Agg")  # headless: no GUI backend, safe outside the main thread
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    scope_records = _metrics_for_scope(metric_records, scope)
    if not scope_records and not (statistics.get("per_model") or {}):
        return []

    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    leaderboard = _plot_leaderboard(plt, scope_records, scope, destination)
    if leaderboard is not None:
        paths.append(leaderboard)

    forest = _plot_accuracy_ci_forest(plt, statistics, destination)
    if forest is not None:
        paths.append(forest)

    operational = (statistics.get("operational") or {}).get("per_model") or {}
    if operational:
        cost_items = _operational_items(scope_records, operational, ("cost", "usd_per_1k_rows"))
        cost_items = [(model, x, y) for model, x, y in cost_items if x > 0]
        cost_plot = _plot_pareto(
            plt,
            cost_items,
            "Provider cost (USD per 1,000 rows)",
            f"Macro-F1 vs cost — Pareto frontier ({scope} scope)",
            "pareto_quality_cost.png",
            destination,
        )
        if cost_plot is not None:
            paths.append(cost_plot)

        latency_items = _operational_items(scope_records, operational, ("latency_ms", "p95"))
        latency_plot = _plot_pareto(
            plt,
            latency_items,
            "Latency p95 (ms)",
            f"Macro-F1 vs latency — Pareto frontier ({scope} scope)",
            "pareto_quality_latency.png",
            destination,
        )
        if latency_plot is not None:
            paths.append(latency_plot)

    for record in scope_records:
        confusion = _plot_confusion(plt, record, destination)
        if confusion is not None:
            paths.append(confusion)
        per_class = _plot_per_class(plt, record, destination)
        if per_class is not None:
            paths.append(per_class)

    return paths
