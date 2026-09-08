# Presentation assets

`project-banner.svg` is the README masthead. `social-preview.png` is the 1280 × 640
GitHub share card; its SVG counterpart is also included. The title and sample facts
come from the submitted dissertation. These assets are presentation artwork, separate
from the empirical figures under `submission/news-sentiment-beyond-mean/manuscript/artifacts/`.

Rebuild the share card from the repository root:

```bash
uv run --locked --extra figures python scripts/build_social_preview.py
```

The other SVG files illustrate the earlier benchmark CLI, TUI and research workflow.
