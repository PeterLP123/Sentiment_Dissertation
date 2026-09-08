"""Render the repository share card from the submitted study's headline facts.

Run with the root project's locked ``figures`` extra. This is presentation artwork;
the manuscript's empirical figures and their reproduction checks are separate.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
NAVY = "#102c3d"
WHITE = "#ffffff"
MUTED = "#b8cbd5"
TEAL = "#75d4bd"


def main() -> None:
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "negative-story-threshold"}):
        fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=NAVY)
        ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 1280), ylim=(0, 640))
        ax.set_axis_off()

        def text(x: float, y: float, value: str, size: float, color: str = WHITE, weight: str = "normal") -> None:
            ax.text(x, y, value, fontsize=size, color=color, weight=weight, va="baseline")

        ax.plot((64, 99), (577, 577), color=TEAL, lw=3, solid_capstyle="round")
        text(119, 569, "UCL  /  MSc Computational Finance  /  2026", 13, MUTED)
        text(60, 471, "A Hard Negative-Story", 44, weight="bold")
        text(60, 400, "Threshold", 44, weight="bold")
        text(64, 337, "Training-Period Association,", 21, MUTED)
        text(64, 298, "Temporal Non-Replication and Economic Limits", 21, MUTED)
        ax.plot((64, 1216), (252, 252), color="#345364", lw=1)

        for x, value, label in (
            (64, "715,546", "NEWS-BEARING FIRM-DAYS"),
            (531, "570", "PRICED FIRMS"),
            (905, "2011–2023", "PRIMARY SAMPLE"),
        ):
            text(x, 172, value, 31, TEAL, "bold")
            text(x, 137, label, 11, MUTED)

        ax.plot((64, 1216), (94, 94), color="#345364", lw=1)
        text(64, 45, "Peter Prendergast", 14)
        text(631, 46, "NEWS SENTIMENT  /  TIMING  /  REPLICATION", 11, MUTED)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT / "social-preview.png", dpi=100)
        svg_path = OUTPUT / "social-preview.svg"
        fig.savefig(svg_path, metadata={"Date": None})
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)
    print("Wrote docs/assets/social-preview.{png,svg} (1280 × 640)")


if __name__ == "__main__":
    main()
