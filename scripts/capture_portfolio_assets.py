"""Capture the real TUI and render a paced replay of an offline demo transcript.

Run with the locked development and figures extras. The GIF preserves command
output but uses fixed reading times, not measured execution latency.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, DataTable, TabbedContent

from sentiment_benchmark.exporter import export_run
from sentiment_benchmark.metrics import evaluate_responses
from sentiment_benchmark.models import DatasetRow, LLMResponseRecord, RunConfig
from sentiment_benchmark.storage import BenchmarkStore
from sentiment_benchmark.tui import SentimentBenchmarkApp

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"


def seed_synthetic_results(app: SentimentBenchmarkApp) -> int:
    """Compute and export metrics from 12 invented labels and canned predictions."""
    model = "synthetic/fixture-v1"
    labels = ("positive", "negative", "neutral")
    rows = []
    responses = []
    for class_index, label in enumerate(labels):
        for item in range(4):
            number = len(rows) + 1
            rows.append(DatasetRow(number, f"Synthetic {label} example {item + 1}.", label, False, False, 1))
            prediction = label if item < 3 else labels[(class_index + 1) % len(labels)]
            responses.append(LLMResponseRecord(
                row_number=number, model_id=model, prompt_hash=app.prompt.prompt_hash,
                raw_content=prediction, normalized_label=prediction, parse_status="valid", status="success",
            ))
    store = BenchmarkStore(app.db_path)
    store.initialize()
    store.upsert_dataset(rows, "synthetic-inline-fixture")
    store.save_prompt(app.prompt)
    config = RunConfig(
        models=[model], prompt=app.prompt, mode="pilot", dataset_path="synthetic-inline-fixture",
        db_path=str(app.db_path), base_url="fixture://no-inference", sample_per_class=4,
    )
    run_id = store.create_run(config, [row.row_number for row in rows])
    for response in responses:
        store.save_response(run_id, response)
    result = evaluate_responses(rows, [asdict(response) for response in responses], model)
    if result.row_count != 12 or result.accuracy != 0.75 or abs(result.macro_f1 - 0.75) > 1e-12:
        raise ValueError("Synthetic preview metrics differ from the declared fixture")
    store.save_metrics(run_id, result)
    store.mark_run_complete(run_id)
    exported = export_run(app.db_path, run_id, output_dir="results/synthetic-preview-export")
    if not exported or not all(path.is_file() for path in exported):
        raise ValueError("Synthetic preview export is incomplete")
    print(f"Synthetic preview: 12 rows, macro-F1 0.75; {len(exported)} export files verified. No model inference.")
    return run_id


async def capture_tui() -> None:
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="portfolio-tui-") as directory:
        work = Path(directory)
        shutil.copytree(ROOT / "configs", work / "configs")
        dataset = Path("Data/derived/labeled/financial_sentiment_v2.csv")
        (work / dataset).parent.mkdir(parents=True)
        shutil.copyfile(ROOT / dataset, work / dataset)
        try:
            os.chdir(work)
            # No inherited credentials, saved sessions, queues or real run database.
            with patch.dict(os.environ, {
                "SENTIMENT_BENCH_AUTO_FETCH_MODELS": "0", "SENTIMENT_BENCH_MACHINE_LABEL": "synthetic-preview",
            }, clear=True):
                app = SentimentBenchmarkApp(
                    session_path=Path("session.json"),
                    queue_path=Path("queue.json"),
                    db_path=Path("benchmark.sqlite"),
                )
                async with app.run_test(size=(120, 42)) as pilot:
                    await pilot.pause(3)
                    screenshot = app.export_screenshot(title="Sentiment benchmark | offline preview")
                    (ASSETS / "benchmark-tui.svg").write_text(
                        "\n".join(line.rstrip() for line in screenshot.splitlines()) + "\n", encoding="utf-8"
                    )
                run_id = seed_synthetic_results(app)
                app = SentimentBenchmarkApp(
                    session_path=Path("results-session.json"), queue_path=Path("results-queue.json"),
                    db_path=Path("benchmark.sqlite"),
                )
                app.title = "SYNTHETIC FIXTURE | No model inference"
                async with app.run_test(size=(120, 48)) as pilot:
                    await pilot.pause(0.5)
                    app.query_one(TabbedContent).active = "results-tab"
                    app._load_metrics_for(run_id)
                    metrics = app.query_one("#metrics-table", DataTable)
                    app.on_data_table_row_selected(DataTable.RowSelected(metrics, 0, next(iter(metrics.rows))))
                    app.query_one("#perclass-section", Collapsible).collapsed = True
                    app.query_one("#confusion-section", Collapsible).collapsed = False
                    await pilot.pause(1)
                    app.set_focus(None)
                    app.query_one("#results-scroll", VerticalScroll).scroll_home(animate=False)
                    await pilot.pause(0.2)
                    screenshot = app.export_screenshot(title="SYNTHETIC FIXTURE | 12 invented examples | no model inference")
                    (ASSETS / "benchmark-results.svg").write_text(
                        "\n".join(line.rstrip() for line in screenshot.splitlines()) + "\n", encoding="utf-8"
                    )
        finally:
            os.chdir(previous)


def render_replay(demo: Path) -> None:
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), 16)
    transcripts = [(demo / f"step-{index}.txt").read_text(encoding="utf-8") for index in range(1, 4)]
    titles = ["1 / 3   Validate the public benchmark", "2 / 3   Inspect synthetic inputs", "3 / 3   Write the fixture report"]
    frames = []
    for title, transcript in zip(titles, transcripts, strict=True):
        frame = Image.new("RGB", (1160, 770), "#10202c")
        draw = ImageDraw.Draw(frame)
        draw.text((24, 18), title, font=font, fill="#75d4bd")
        draw.text((24, 48), "SYNTHETIC STRATEGY DEMO | captured output | playback timing shortened", font=font, fill="#b8cbd5")
        lines = transcript.splitlines()
        for row, line in enumerate(lines):
            if len(line) > 113 or row >= 32:
                raise ValueError("Transcript exceeds replay frame; adjust layout rather than truncating output")
            draw.text((24, 92 + row * 20), line, font=font, fill="#e2ecf2")
        frames.append(frame)
    frames[0].save(ASSETS / "offline-demo.gif", save_all=True, append_images=frames[1:], duration=[5000, 9000, 5000], loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_directory", type=Path, help="Output directory printed by scripts/portfolio_demo.py")
    args = parser.parse_args()
    demo = args.demo_directory.resolve()
    ASSETS.mkdir(parents=True, exist_ok=True)
    asyncio.run(capture_tui())
    render_replay(demo)
    print("Captured docs/assets/benchmark-tui.svg, benchmark-results.svg and offline-demo.gif")


if __name__ == "__main__":
    main()
