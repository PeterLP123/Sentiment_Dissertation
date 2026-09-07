"""Convert a path-only SVG (the UCL wordmark) to a vector PDF for pdfLaTeX.

pdfLaTeX cannot include SVG. This writes filled PDF paths so
``make manuscript`` can keep ``manuscript/ucl_logo.svg`` as the source mark.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

TOKEN_RE = re.compile(
    r"[MmLlHhVvCcSsQqTtAaZz]|[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?"
)
COMMANDS = set("MmLlHhVvCcSsQqTtAaZz")
ARG_COUNTS = {
    "M": 2,
    "L": 2,
    "H": 1,
    "V": 1,
    "C": 6,
    "S": 4,
    "Q": 4,
    "T": 2,
    "A": 7,
    "Z": 0,
}
MOVETO = "m"
LINETO = "l"
CURVE4 = "c"
CLOSE = "h"


def local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    raw = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if raw:
        parts = [float(part) for part in re.split(r"[,\s]+", raw.strip()) if part]
        if len(parts) != 4:
            raise ValueError(f"unrecognised SVG viewBox: {raw!r}")
        return parts[0], parts[1], parts[2], parts[3]
    width = float(root.attrib["width"])
    height = float(root.attrib["height"])
    return 0.0, 0.0, width, height


def tokenize_path(d: str) -> list[str]:
    return TOKEN_RE.findall(d.replace(",", " "))


def _floats(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def parse_svg_path(d: str) -> list[tuple[str, list[tuple[float, float]]]]:
    """Return PDF-like path ops: move, line, cubic, close."""
    tokens = tokenize_path(d)
    ops: list[tuple[str, list[tuple[float, float]]]] = []
    i = 0
    command = ""
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    last_control: tuple[float, float] | None = None

    while i < len(tokens):
        token = tokens[i]
        if token in COMMANDS:
            command = token
            i += 1
            if command in "Zz":
                ops.append((CLOSE, []))
                current = start
                last_control = None
            continue
        if not command:
            raise ValueError(f"SVG path data starts with a number: {d[:40]!r}")

        family = command.upper()
        count = ARG_COUNTS[family]
        args = _floats(tokens[i : i + count])
        if len(args) != count:
            raise ValueError(f"SVG command {command} expected {count} arguments")
        i += count
        relative = command.islower()
        x0, y0 = current

        if family == "M":
            x, y = args
            if relative:
                x, y = x0 + x, y0 + y
            current = start = (x, y)
            ops.append((MOVETO, [current]))
            command = "l" if relative else "L"
            last_control = None
            continue

        if family == "L":
            x, y = args
            if relative:
                x, y = x0 + x, y0 + y
            current = (x, y)
            ops.append((LINETO, [current]))
        elif family == "H":
            x = x0 + args[0] if relative else args[0]
            current = (x, y0)
            ops.append((LINETO, [current]))
        elif family == "V":
            y = y0 + args[0] if relative else args[0]
            current = (x0, y)
            ops.append((LINETO, [current]))
        elif family == "C":
            x1, y1, x2, y2, x, y = args
            if relative:
                x1, y1, x2, y2, x, y = (
                    x0 + x1,
                    y0 + y1,
                    x0 + x2,
                    y0 + y2,
                    x0 + x,
                    y0 + y,
                )
            current = (x, y)
            last_control = (x2, y2)
            ops.append((CURVE4, [(x1, y1), (x2, y2), current]))
        elif family == "S":
            x2, y2, x, y = args
            if relative:
                x2, y2, x, y = x0 + x2, y0 + y2, x0 + x, y0 + y
            if last_control is None:
                x1, y1 = current
            else:
                x1, y1 = (2 * x0 - last_control[0], 2 * y0 - last_control[1])
            current = (x, y)
            last_control = (x2, y2)
            ops.append((CURVE4, [(x1, y1), (x2, y2), current]))
        elif family == "Q":
            x1, y1, x, y = args
            if relative:
                x1, y1, x, y = x0 + x1, y0 + y1, x0 + x, y0 + y
            current = _quad_to_cubic(ops, current, (x1, y1), (x, y))
            last_control = (x1, y1)
        elif family == "T":
            x, y = args
            if relative:
                x, y = x0 + x, y0 + y
            if last_control is None:
                q_control = current
            else:
                q_control = (2 * x0 - last_control[0], 2 * y0 - last_control[1])
            current = _quad_to_cubic(ops, current, q_control, (x, y))
            last_control = q_control
        elif family == "A":
            raise ValueError("SVG arc commands are not supported")
        else:
            raise ValueError(f"unsupported SVG command {command}")

        if family not in {"C", "S", "Q", "T"}:
            last_control = None

    return ops


def _quad_to_cubic(
    ops: list[tuple[str, list[tuple[float, float]]]],
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    c1 = (start[0] + 2.0 * (control[0] - start[0]) / 3.0, start[1] + 2.0 * (control[1] - start[1]) / 3.0)
    c2 = (end[0] + 2.0 * (control[0] - end[0]) / 3.0, end[1] + 2.0 * (control[1] - end[1]) / 3.0)
    ops.append((CURVE4, [c1, c2, end]))
    return end


def parse_fill(colour: str) -> tuple[float, float, float]:
    colour = colour.strip()
    if colour.startswith("#") and len(colour) == 7:
        red, green, blue = (int(colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        return (red, green, blue)
    if colour.lower() in {"black", "#000", "#000000"}:
        return (0.0, 0.0, 0.0)
    raise ValueError(f"unsupported SVG fill {colour!r}")


def path_to_pdf_ops(
    ops: list[tuple[str, list[tuple[float, float]]]],
    min_x: float,
    min_y: float,
    height: float,
) -> str:
    def xy(point: tuple[float, float]) -> str:
        x = point[0] - min_x
        y = height - (point[1] - min_y)
        return f"{x:.3f} {y:.3f}"

    parts: list[str] = []
    for op, points in ops:
        if op == CLOSE:
            parts.append("h")
        elif op == MOVETO:
            parts.append(f"{xy(points[0])} m")
        elif op == LINETO:
            parts.append(f"{xy(points[0])} l")
        elif op == CURVE4:
            parts.append(f"{xy(points[0])} {xy(points[1])} {xy(points[2])} c")
        else:
            raise ValueError(f"unknown path op {op}")
    return " ".join(parts)


def build_pdf(width: float, height: float, content: str) -> bytes:
    stream = content.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.3f} {height:.3f}] "
            f"/Contents 4 0 R /Resources << >> >>"
        ).encode("ascii"),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    root = ET.parse(svg_path).getroot()
    if local_tag(root.tag) != "svg":
        raise ValueError(f"{svg_path} is not an SVG document")
    min_x, min_y, width, height = parse_viewbox(root)
    chunks = ["q"]
    for element in root.iter():
        if local_tag(element.tag) != "path":
            continue
        d = element.attrib.get("d")
        if not d:
            continue
        fill = element.attrib.get("fill", "#000000")
        if fill.lower() in {"none", "transparent"}:
            continue
        red, green, blue = parse_fill(fill)
        ops = parse_svg_path(d)
        if not ops:
            continue
        chunks.append(f"{red:.3f} {green:.3f} {blue:.3f} rg")
        chunks.append(path_to_pdf_ops(ops, min_x, min_y, height))
        chunks.append("f")
    chunks.append("Q")
    if len(chunks) == 2:
        raise ValueError(f"no drawable paths in {svg_path}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(build_pdf(width, height, "\n".join(chunks) + "\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", type=Path)
    parser.add_argument("pdf", type=Path, nargs="?")
    args = parser.parse_args()
    pdf = args.pdf if args.pdf is not None else args.svg.with_suffix(".pdf")
    svg_to_pdf(args.svg, pdf)


if __name__ == "__main__":
    main()
