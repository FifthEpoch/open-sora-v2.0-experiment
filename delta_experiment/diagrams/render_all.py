#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from .diagram_lib import Style, new_figure, build_baseline_detailed
from .overlays import overlay_delta_a, overlay_delta_b, overlay_delta_c


def _render_one(out_dir: Path, stem: str, title: str, overlay_fn):
    style = Style()
    fig, ax = new_figure(style)
    anchors = build_baseline_detailed(ax, style, title=title)
    overlay_fn(ax, style, anchors)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return pdf_path, svg_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=str,
        default="delta_experiment/diagrams/out",
        help="Output directory for rendered diagrams (PDF+SVG).",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional: render only one of {a,b,c}.",
    )
    p.add_argument(
        "--log-path",
        type=str,
        default="delta_experiment/diagrams/out/render_log.txt",
        help="Write a render log here (useful when stdout/stderr is not visible).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    only = args.only.strip().lower()
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(msg.rstrip() + "\n")

    _log("=== δ-TTA diagram render ===")
    _log(f"out_dir={out_dir}")
    _log(f"only={only!r}")
    _log("")

    try:
        rendered = []
        if only in ("", "a"):
            rendered.append(
                _render_one(
                    out_dir,
                    stem="delta_a",
                    title="Option A: global δ added to conditioning vec (Open-Sora v2.0 δ-TTA)",
                    overlay_fn=overlay_delta_a,
                )
            )
        if only in ("", "b"):
            rendered.append(
                _render_one(
                    out_dir,
                    stem="delta_b",
                    title="Option B: grouped per-layer δ offsets into modulation (Open-Sora v2.0 δ-TTA)",
                    overlay_fn=overlay_delta_b,
                )
            )
        if only in ("", "c"):
            rendered.append(
                _render_one(
                    out_dir,
                    stem="delta_c",
                    title="Option C: δ_out as denoiser output correction (Open-Sora v2.0 δ-TTA)",
                    overlay_fn=overlay_delta_c,
                )
            )

        for pdf_path, svg_path in rendered:
            _log(f"Wrote: {pdf_path}")
            _log(f"Wrote: {svg_path}")
            print(f"Wrote: {pdf_path}")
            print(f"Wrote: {svg_path}")
    except Exception:
        tb = traceback.format_exc()
        _log("ERROR during render:")
        _log(tb)
        raise


if __name__ == "__main__":
    main()

