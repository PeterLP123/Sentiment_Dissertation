"""Run the synthetic portfolio walkthrough in a fresh, ignored output directory."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(ROOT)
    output_root = ROOT / "results" / "portfolio_demo"
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
    relative = output.relative_to(ROOT).as_posix()
    config = (ROOT / "configs/strategy_research/smoke.toml").read_text(encoding="utf-8")
    config = config.replace('derived_root = "Data/derived/strategy_research"', f'derived_root = "{relative}/derived"')
    config = config.replace('results_root = "results/strategy_research"', f'results_root = "{relative}/results"')
    parsed = tomllib.loads(config)
    if (
        parsed["run"]["mode"] != "smoke"
        or parsed["data"]["source"] != "synthetic"
        or parsed["scoring"]["provider"] != "fixture"
        or parsed["outputs"]["derived_root"] != f"{relative}/derived"
        or parsed["outputs"]["results_root"] != f"{relative}/results"
    ):
        raise SystemExit("Demo requires synthetic fixtures and fresh isolated output roots.")
    config_path = output / "demo.toml"
    config_path.write_text(config, encoding="utf-8")
    commands = [
        ["validate-data"],
        ["strategy", "run", "--config", str(config_path.relative_to(ROOT)), "--dry-run"],
        ["strategy", "run", "--config", str(config_path.relative_to(ROOT))],
    ]
    print("Synthetic fixture demonstration. No model calls or research evaluation.\n", flush=True)
    for index, args in enumerate(commands, start=1):
        command = "sentiment-bench " + " ".join(args)
        print(f"$ {command}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "sentiment_benchmark.cli", *args],
            capture_output=True,
            text=True,
            env={**os.environ, "NO_COLOR": "1", "COLUMNS": "110"},
        )
        transcript = f"$ {command}\n{result.stdout}{result.stderr}"
        (output / f"step-{index}.txt").write_text(transcript, encoding="utf-8")
        print(result.stdout + result.stderr, end="", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    print(f"\nDemo outputs and transcripts: {relative}")


if __name__ == "__main__":
    main()
