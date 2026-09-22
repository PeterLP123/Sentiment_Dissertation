# Presentation assets

`social-preview.png` is the 1280 × 640 GitHub share card, with an editable SVG
counterpart. It is prepared for a future public launch. Its title and sample facts
come from the dissertation.

The main README uses the empirical coefficient figure from
`submission/news-sentiment-beyond-mean/manuscript/artifacts/`.

Rebuild the share card from the repository root:

```bash
uv run --locked --extra figures python scripts/build_social_preview.py
```

`research-workflow.svg` diagrams the earlier research process.

## Software preview

`benchmark-tui.svg` is a capture of the actual Textual app at 120 × 42 cells,
using the public benchmark and an empty temporary database. The capture script
copies configuration and the public dataset into a temporary working directory,
clears inherited environment variables and disables model auto-fetch. It does not
load the user's `.env`, saved session, queue or run database.

`offline-demo.gif` renders the three captured CLI transcripts from the synthetic
demo as a looping replay. Output is preserved; playback uses fixed reading times
(5, 9 and 5 seconds), not measured run durations. The demo guide provides a static
text alternative. These are generated presentation assets, separate from the
frozen manuscript figures.

To regenerate them from a fresh run:

```bash
uv sync --locked --extra dev --extra figures
uv run --locked --extra dev --extra figures python scripts/portfolio_demo.py
# Replace RUN_DIRECTORY with the directory printed by the previous command.
uv run --locked --extra dev --extra figures python scripts/capture_portfolio_assets.py RUN_DIRECTORY
```

Source: [demo runner](../../scripts/portfolio_demo.py) and
[capture script](../../scripts/capture_portfolio_assets.py).
