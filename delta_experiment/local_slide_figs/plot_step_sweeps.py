#!/usr/bin/env python3
"""
Plot step sweep results for LoRA and delta A/B.
Full-TTA data is loaded but omitted from plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_series(d: dict, key: str) -> tuple[list[int], list[float]]:
    steps = sorted([int(k) for k in d.keys()])
    vals = [d[str(s)][key] for s in steps]
    return steps, vals


def plot_metric(steps, series, labels, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = ["#4985B6", "#2E7D32", "#7C4DFF"]
    for i, (vals, label) in enumerate(zip(series, labels)):
        ax.plot(steps, vals, marker="o", linewidth=2, color=colors[i], label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.set_xticks(steps)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-json", required=True, type=str)
    parser.add_argument("--lora-json", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    args = parser.parse_args()

    delta = load_json(Path(args.delta_json))
    lora = load_json(Path(args.lora_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps, lora_psnr = extract_series(lora["lora"], "psnr")
    _, lora_ssim = extract_series(lora["lora"], "ssim")
    _, lora_lpips = extract_series(lora["lora"], "lpips")
    _, lora_train = extract_series(lora["lora"], "train_s")

    steps_a, a_psnr = extract_series(delta["delta_a"], "psnr")
    _, a_ssim = extract_series(delta["delta_a"], "ssim")
    _, a_lpips = extract_series(delta["delta_a"], "lpips")
    _, a_train = extract_series(delta["delta_a"], "train_s")

    steps_b, b_psnr = extract_series(delta["delta_b"], "psnr")
    _, b_ssim = extract_series(delta["delta_b"], "ssim")
    _, b_lpips = extract_series(delta["delta_b"], "lpips")
    _, b_train = extract_series(delta["delta_b"], "train_s")

    assert steps == steps_a == steps_b

    plot_metric(
        steps,
        [lora_psnr, a_psnr, b_psnr],
        ["LoRA", "Global-δ (A)", "Multi-δ (B)"],
        "PSNR",
        out_dir / "sweep_psnr.png",
    )
    plot_metric(
        steps,
        [lora_ssim, a_ssim, b_ssim],
        ["LoRA", "Global-δ (A)", "Multi-δ (B)"],
        "SSIM",
        out_dir / "sweep_ssim.png",
    )
    plot_metric(
        steps,
        [lora_lpips, a_lpips, b_lpips],
        ["LoRA", "Global-δ (A)", "Multi-δ (B)"],
        "LPIPS",
        out_dir / "sweep_lpips.png",
    )
    plot_metric(
        steps,
        [lora_train, a_train, b_train],
        ["LoRA", "Global-δ (A)", "Multi-δ (B)"],
        "Train time per video (s)",
        out_dir / "sweep_train_time.png",
    )


if __name__ == "__main__":
    main()
