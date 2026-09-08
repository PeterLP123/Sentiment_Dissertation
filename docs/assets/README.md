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
