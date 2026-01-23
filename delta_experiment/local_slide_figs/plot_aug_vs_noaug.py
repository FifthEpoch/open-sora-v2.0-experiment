#!/usr/bin/env python3
"""
Plot augmentation vs. no-augmentation comparisons for PSNR/SSIM/LPIPS.
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


def plot_metric(metric: str, labels: list[str], no_aug: list[float], aug: list[float], out_path: Path) -> None:
    xs = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(xs - width / 2, no_aug, width=width, label="No Aug", color="#4985B6")
    ax.bar(xs + width / 2, aug, width=width, label="Aug", color="#2E7D32")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{metric.upper()} (Aug vs. No Aug)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, frameon=False)

    # Tight y-range to make small changes visible
    vals = no_aug + aug
    vmin = min(vals)
    vmax = max(vals)
    pad = (vmax - vmin) * 0.25 if vmax > vmin else 0.01
    ax.set_ylim(vmin - pad, vmax + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_delta(metric: str, labels: list[str], deltas: list[float], out_path: Path) -> None:
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(xs, deltas, width=0.45, color="#7C4DFF")
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"Δ{metric.upper()} (Aug - No Aug)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-json", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    args = parser.parse_args()

    data = load_json(Path(args.in_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["baseline", "lora", "delta_a", "delta_b"]
    labels = ["Baseline", "LoRA", "Global-δ", "Multi-δ"]

    for metric in ["psnr", "ssim", "lpips"]:
        no_aug = [data["no_aug"][m][metric] for m in methods]
        aug = [data["aug"][m][metric] for m in methods]
        plot_metric(
            metric,
            labels,
            no_aug,
            aug,
            out_dir / f"aug_vs_noaug_{metric}.png",
        )
        deltas = [a - b for a, b in zip(aug, no_aug)]
        plot_delta(
            metric,
            labels,
            deltas,
            out_dir / f"aug_delta_{metric}.png",
        )


if __name__ == "__main__":
    main()
