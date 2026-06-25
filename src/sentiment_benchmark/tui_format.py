"""Pure Rich-text formatting helpers and table-sort metadata for the TUI.

Kept dependency-free of the app classes so both ``tui`` and the feature mixins
can import these without a circular import.
"""

from __future__ import annotations

import json
import re

from rich.text import Text

_MONITOR_FIELD = re.compile(r"\b(run|model|row|status|label)=([^\s]+)")

# Status keyword -> colour, used to tint table cells so runs can be scanned at a glance.
_STATUS_COLORS = {
    "completed": "green",
    "done": "green",
    "success": "green",
    "running": "yellow",
    "queued": "grey58",
    "cancelled": "red",
    "failed": "red",
    "error": "red",
    "interrupted": "orange1",
}


def _num(value: object, style: str | None = None) -> Text:
    """A right-aligned numeric cell so magnitudes line up down a column."""
    return Text(str(value), style=style or "", justify="right")


def _accuracy_text(value: float, *, bold: bool = False) -> Text:
    """Colour an accuracy/F1 score on a traffic-light scale."""
    if value >= 0.8:
        color = "green"
    elif value >= 0.6:
        color = "yellow"
    else:
        color = "red"
    return Text(f"{value:.4f}", style=f"bold {color}" if bold else color, justify="right")


def _status_text(status: str) -> Text:
    """Colour a status label (first word drives the colour, e.g. 'done (acc ...)')."""
    key = status.split(" ", 1)[0].lower()
    return Text(status, style=_STATUS_COLORS.get(key, "white"))


def _count_text(value: int) -> Text:
    """Red when there is something to worry about (errors/invalids), dim otherwise."""
    return Text(str(value), style="red" if value > 0 else "grey58", justify="right")


def _latency_text(value: float | None) -> Text:
    if not isinstance(value, (int, float)):
        return Text("-", style="grey58", justify="right")
    return Text(f"{value:.0f} ms", justify="right")


def _tokens_text(value: int | None) -> Text:
    if not isinstance(value, (int, float)) or value == 0:
        return Text("-", style="grey58", justify="right")
    return Text(f"{int(value):,}", justify="right")


def _cost_text(value: float | None) -> Text:
    if not isinstance(value, (int, float)):
        return Text("-", style="grey58", justify="right")
    return Text(f"${value:.4f}", style="cyan", justify="right")


def _monitor_field_style(key: str, value: str) -> str:
    if key == "model":
        return "bold cyan"
    if key in {"run", "row"}:
        return "cyan"
    if key == "status":
        return _STATUS_COLORS.get(value.lower(), "white")
    if key == "label":
        return {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow",
            "-": "grey58",
        }.get(value.lower(), "white")
    return "white"


def _monitor_text(message: str) -> Text:
    text = Text()
    cursor = 0
    for match in _MONITOR_FIELD.finditer(message):
        text.append(message[cursor : match.start()])
        key, value = match.groups()
        text.append(f"{key}=", style="yellow")
        text.append(value, style=_monitor_field_style(key, value))
        cursor = match.end()
    text.append(message[cursor:])
    return text


# Sort metadata for the clickable Results tables. Each column index maps to
# (header label, ascending-on-first-click, key extractor). Numeric/date columns
# default to descending (most recent / highest first); text columns to ascending.
_RUNS_SORT_COLUMNS: dict[int, tuple] = {
    0: ("ID", False, lambda run: int(run["id"])),
    1: ("Created", False, lambda run: run["created_at"] or ""),
    2: ("Mode", True, lambda run: run["mode"] or ""),
    3: ("Status", True, lambda run: run["status"] or ""),
    4: ("Machine", True, lambda run: (run["machine_label"] or run["machine_id"] or "").lower()),
    5: ("Models", False, lambda run: len(json.loads(run["models_json"]) if run["models_json"] else [])),
}
_LEADERBOARD_SORT_COLUMNS: dict[int, tuple] = {
    0: ("Model", True, "model_id"),
    1: ("Best Acc", False, "accuracy"),
    2: ("Macro F1", False, "macro_f1"),
    3: ("Best run", False, "run_id"),
    4: ("Runs", False, "runs"),
    5: ("Rows", False, "row_count"),
}
_RUNS_HELP_BASE = "Press Enter on a row to load metrics for that run. Click a column header to sort."
_LEADERBOARD_HELP_BASE = (
    "Best accuracy each model has reached across all stored runs at the chosen scope. "
    "Macro F1, best run id, and run count are for that best run. Click a column header to sort."
)
