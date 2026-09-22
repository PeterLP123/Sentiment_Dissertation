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
from pathlib import Path
from unittest.mock import patch

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from sentiment_benchmark.tui import SentimentBenchmarkApp

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"


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
            with patch.dict(os.environ, {"SENTIMENT_BENCH_AUTO_FETCH_MODELS": "0"}, clear=True):
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
    print("Captured docs/assets/benchmark-tui.svg and offline-demo.gif")


if __name__ == "__main__":
    main()
