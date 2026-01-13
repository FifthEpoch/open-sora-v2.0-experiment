#!/usr/bin/env python3
"""
Generate a local plot comparing Open-Sora v1.3 vs v2.0 baseline performance for video continuation.

This script is intentionally self-contained: it uses the baseline metrics values you provided in chat.

Outputs (by default):
  - delta_experiment/plots/v13_vs_v20_baseline_bar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict


V13_BASELINE: Dict[str, float] = {
    "psnr": 8.90851309905017,
    "ssim": 0.3067392551397064,
    "lpips": 0.7185019291477439,
}

V20_BASELINE: Dict[str, float] = {
    "psnr": 15.494961792163876,
    "ssim": 0.49427606615343406,
    "lpips": 0.41047981577459725,
}


def fmt(x: float, ndigits: int = 4) -> str:
    return f"{x:.{ndigits}f}"


def print_table(v13: Dict[str, float], v20: Dict[str, float]) -> None:
    d_psnr = v20["psnr"] - v13["psnr"]
    d_ssim = v20["ssim"] - v13["ssim"]
    d_lpips = v20["lpips"] - v13["lpips"]

    print("")
    print("Baseline comparison (v1.3 vs v2.0)")
    print("-" * 72)
    print(f"{'Metric':<10} {'v1.3':>12} {'v2.0':>12} {'Δ(v2-v1)':>14} {'Direction':>12}")
    print("-" * 72)
    print(f"{'PSNR':<10} {fmt(v13['psnr']):>12} {fmt(v20['psnr']):>12} {fmt(d_psnr):>14} {'higher↑':>12}")
    print(f"{'SSIM':<10} {fmt(v13['ssim']):>12} {fmt(v20['ssim']):>12} {fmt(d_ssim):>14} {'higher↑':>12}")
    print(f"{'LPIPS':<10} {fmt(v13['lpips']):>12} {fmt(v20['lpips']):>12} {fmt(d_lpips):>14} {'lower↓':>12}")
    print("-" * 72)
    print("")


def save_plot(out_path: Path, v13: Dict[str, float], v20: Dict[str, float], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate the plot. "
            "Activate the same conda env you use for evaluation (it typically includes matplotlib)."
        ) from e

    metrics = ["psnr", "ssim", "lpips"]
    labels = ["PSNR", "SSIM", "LPIPS"]
    v13_vals = [v13[m] for m in metrics]
    v20_vals = [v20[m] for m in metrics]

    x = list(range(len(metrics)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], v13_vals, width=width, label="v1.3 baseline")
    ax.bar([i + width / 2 for i in x], v20_vals, width=width, label="v2.0 baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=str,
        default="delta_experiment/plots/v13_vs_v20_baseline_bar.png",
        help="Where to write the plot PNG (relative to repo root is fine).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Video Continuation Baseline — Open-Sora v1.3 vs v2.0",
    )
    args = p.parse_args()

    print_table(V13_BASELINE, V20_BASELINE)
    save_plot(Path(args.output), V13_BASELINE, V20_BASELINE, title=args.title)
    print(f"Wrote plot: {args.output}")


if __name__ == "__main__":
    main()

