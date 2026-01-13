#!/usr/bin/env python3
"""
Generate a local plot comparing Open-Sora v1.3 vs v2.0 baseline performance for video continuation.

This script is intentionally self-contained: it uses the baseline metrics values you provided in chat.

Outputs (by default):
  - delta_experiment/plots/v13_vs_v20_baseline_psnr.png
  - delta_experiment/plots/v13_vs_v20_baseline_ssim.png
  - delta_experiment/plots/v13_vs_v20_baseline_lpips.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
from typing import Dict


V13_BASELINE: Dict[str, float] = {
    "psnr": 8.90851309905017,
    "ssim": 0.3067392551397064,
    "lpips": 0.7185019291477439,
}

# v1.3 best "LoRA/TTA" run metrics (from the hp sweep summary snippet you provided):
# option2_20steps_5e5:
#   avg_psnr_finetuned=10.98881351066431
#   avg_ssim_finetuned=0.2640347222153441
#   avg_lpips_finetuned=0.6463043822182549
V13_TTA_BEST: Dict[str, float] = {
    "psnr": 10.98881351066431,
    "ssim": 0.2640347222153441,
    "lpips": 0.6463043822182549,
}

V20_BASELINE: Dict[str, float] = {
    "psnr": 15.494961792163876,
    "ssim": 0.49427606615343406,
    "lpips": 0.41047981577459725,
}

# v2.0 best LoRA/TTA run metrics (from lora_experiment/results/evaluation/summary.json you pasted)
V20_TTA_BEST: Dict[str, float] = {
    "psnr": 15.676040090409934,
    "ssim": 0.5071047728639196,
    "lpips": 0.3954454526491463,
}

V13_COLOR = "#4985B6"
V20_COLOR = "#0037FF"

BAR_WIDTH = 0.22  # thinner bars; bars touch within a version group (dx == width)
GROUP_GAP = 0.18  # visual gap between v1.3 and v2.0 groups (in x-axis units)
X_TICK_FONTSIZE = 8
TITLE_WRAP_CHARS = 38


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


def save_single_metric_plot(
    out_path: Path,
    metric_key: str,
    metric_label: str,
    v13: Dict[str, float],
    v20: Dict[str, float],
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate the plot. "
            "Activate the same conda env you use for evaluation (it typically includes matplotlib)."
        ) from e

    v13_val = v13[metric_key]
    v20_val = v20[metric_key]

    fig, ax = plt.subplots(figsize=(4.4, 4))
    # We now plot four bars: baseline + best TTA/LoRA for each version.
    v13_tta = V13_TTA_BEST[metric_key]
    v20_tta = V20_TTA_BEST[metric_key]

    # Grouped bars:
    # - v1.3 baseline and v1.3 TTA touch (no gap)
    # - v2.0 baseline and v2.0 TTA touch (no gap)
    # - keep a visible gap between the v1.3 group and v2.0 group
    start_v13 = 0.0
    start_v20 = start_v13 + 2 * BAR_WIDTH + GROUP_GAP

    xs = [start_v13, start_v13 + BAR_WIDTH, start_v20, start_v20 + BAR_WIDTH]
    values = [v13_val, v13_tta, v20_val, v20_tta]
    colors = [V13_COLOR, V13_COLOR, V20_COLOR, V20_COLOR]
    alphas = [0.55, 1.0, 0.55, 1.0]

    bars = ax.bar(xs, values, width=BAR_WIDTH, color=colors)
    for b, a in zip(bars, alphas):
        b.set_alpha(a)
    ax.set_title(textwrap.fill(title, width=TITLE_WRAP_CHARS))
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["v1.3\nbaseline", "v1.3\nTTA", "v2.0\nbaseline", "v2.0\nTTA"],
        fontsize=X_TICK_FONTSIZE,
    )
    # Tighten x-limits so the plot doesn't leave a large empty center gap.
    pad = BAR_WIDTH * 0.85
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=V13_COLOR, alpha=0.55),
            plt.Rectangle((0, 0), 1, 1, color=V13_COLOR, alpha=1.0),
            plt.Rectangle((0, 0), 1, 1, color=V20_COLOR, alpha=0.55),
            plt.Rectangle((0, 0), 1, 1, color=V20_COLOR, alpha=1.0),
        ],
        labels=["v1.3 baseline", "v1.3 TTA", "v2.0 baseline", "v2.0 TTA"],
        frameon=False,
        fontsize=9,
        loc="best",
    )

    # Metric-specific y scaling.
    if metric_key == "psnr":
        ymax = max(v13_val, v20_val) + 1.0
        ymin = max(0.0, min(v13_val, v20_val) - 1.0)
        ax.set_ylim(ymin, ymax)
    else:
        # SSIM/LPIPS are in [0,1]; keep a consistent scale.
        ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=str,
        default="delta_experiment/plots",
        help="Directory to write plots into (relative to repo root is fine).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Video Continuation Baseline — Open-Sora v1.3 vs v2.0",
    )
    args = p.parse_args()

    print_table(V13_BASELINE, V20_BASELINE)
    # Also print the TTA/LoRA best runs for reference.
    print("Best TTA/LoRA (as provided in logs):")
    print(f"  v1.3 best TTA: PSNR={fmt(V13_TTA_BEST['psnr'])}, SSIM={fmt(V13_TTA_BEST['ssim'])}, LPIPS={fmt(V13_TTA_BEST['lpips'])}")
    print(f"  v2.0 best TTA: PSNR={fmt(V20_TTA_BEST['psnr'])}, SSIM={fmt(V20_TTA_BEST['ssim'])}, LPIPS={fmt(V20_TTA_BEST['lpips'])}")
    print("")
    out_dir = Path(args.output_dir)
    save_single_metric_plot(
        out_path=out_dir / "v13_vs_v20_baseline_psnr.png",
        metric_key="psnr",
        metric_label="PSNR (dB)",
        v13=V13_BASELINE,
        v20=V20_BASELINE,
        title=f"{args.title} — PSNR",
    )
    save_single_metric_plot(
        out_path=out_dir / "v13_vs_v20_baseline_ssim.png",
        metric_key="ssim",
        metric_label="SSIM",
        v13=V13_BASELINE,
        v20=V20_BASELINE,
        title=f"{args.title} — SSIM",
    )
    save_single_metric_plot(
        out_path=out_dir / "v13_vs_v20_baseline_lpips.png",
        metric_key="lpips",
        metric_label="LPIPS (lower is better)",
        v13=V13_BASELINE,
        v20=V20_BASELINE,
        title=f"{args.title} — LPIPS",
    )
    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()

