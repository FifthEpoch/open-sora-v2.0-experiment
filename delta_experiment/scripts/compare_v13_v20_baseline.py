#!/usr/bin/env python3
"""
Compare Open-Sora v1.3 vs v2.0 baseline performance (video continuation).

Inputs:
  - v1.3: naive_experiment hp sweep summary JSON (list of dicts), containing:
      avg_psnr_baseline, avg_ssim_baseline, avg_lpips_baseline
    Example:
      Open-Sora-1.3/naive_experiment/results/hp_sweep/hp_sweep_summary.json

  - v2.0: evaluation summary.json produced by lora_experiment/scripts/evaluate.py, containing:
      baseline_psnr_mean, baseline_ssim_mean, baseline_lpips_mean
    Example:
      Open-Sora-2.0/lora_experiment/results/evaluation/summary.json

This script prints a terminal table and can optionally write a simple bar plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def extract_v13_baseline(hp_sweep_summary_json: Path) -> Dict[str, float]:
    data = load_json(hp_sweep_summary_json)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty JSON list in v1.3 summary: {hp_sweep_summary_json}")

    # Baseline metrics should be identical across entries; we take the first and sanity-check.
    first = data[0]
    psnr = _safe_float(first.get("avg_psnr_baseline"))
    ssim = _safe_float(first.get("avg_ssim_baseline"))
    lpips = _safe_float(first.get("avg_lpips_baseline"))
    if psnr is None or ssim is None or lpips is None:
        raise ValueError(
            "Missing v1.3 baseline keys. Expected avg_psnr_baseline/avg_ssim_baseline/avg_lpips_baseline."
        )

    # Light sanity-check across all entries (ignore missing).
    for i, r in enumerate(data[:50]):  # cap work
        p = _safe_float(r.get("avg_psnr_baseline"))
        s = _safe_float(r.get("avg_ssim_baseline"))
        l = _safe_float(r.get("avg_lpips_baseline"))
        if p is None or s is None or l is None:
            continue
        if abs(p - psnr) > 1e-6 or abs(s - ssim) > 1e-6 or abs(l - lpips) > 1e-6:
            raise ValueError(
                f"v1.3 baseline metrics differ across sweep entries (row {i}). "
                f"Got ({p},{s},{l}) vs first ({psnr},{ssim},{lpips})."
            )

    return {"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips)}


def extract_v20_baseline(eval_summary_json: Path) -> Dict[str, float]:
    d = load_json(eval_summary_json)
    if not isinstance(d, dict):
        raise ValueError(f"Expected a JSON object in v2.0 summary: {eval_summary_json}")

    psnr = _safe_float(d.get("baseline_psnr_mean"))
    ssim = _safe_float(d.get("baseline_ssim_mean"))
    lpips = _safe_float(d.get("baseline_lpips_mean"))
    if psnr is None or ssim is None or lpips is None:
        raise ValueError(
            "Missing v2.0 baseline keys. Expected baseline_psnr_mean/baseline_ssim_mean/baseline_lpips_mean."
        )
    return {"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips)}


def fmt(x: Optional[float], ndigits: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{ndigits}f}"


def print_table(v13: Dict[str, float], v20: Dict[str, float]) -> None:
    # absolute deltas: v2 - v1 (LPIPS: lower is better, but still a delta)
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    p.add_argument("--v13-hp-sweep-summary", type=str, required=True)
    p.add_argument("--v20-eval-summary", type=str, required=True)
    p.add_argument("--plot-path", type=str, default=None, help="If set, save a bar plot PNG to this path.")
    p.add_argument(
        "--title",
        type=str,
        default="UCF101 Video Continuation — Baseline: Open-Sora v1.3 vs v2.0",
    )
    args = p.parse_args()

    v13 = extract_v13_baseline(Path(args.v13_hp_sweep_summary))
    v20 = extract_v20_baseline(Path(args.v20_eval_summary))
    print_table(v13, v20)

    if args.plot_path:
        save_plot(Path(args.plot_path), v13, v20, title=args.title)
        print(f"Wrote plot: {args.plot_path}")


if __name__ == "__main__":
    main()

