#!/usr/bin/env python3
"""
Summarize 50-video step sweep evaluations for delta A/B into a comparison JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def format_metric(metric: str, mean: float, std: float) -> str:
    if metric == "psnr":
        return f"{mean:.2f} ± {std:.2f}"
    return f"{mean:.4f} ± {std:.4f}"


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_eval_dirs(eval_root: Path, prefix: str) -> list[Path]:
    pattern = re.compile(rf"^{prefix}_steps(\d+)_50videos$")
    out = []
    for p in eval_root.iterdir():
        if p.is_dir() and pattern.match(p.name):
            out.append(p)
    return sorted(out, key=lambda p: int(re.search(r"steps(\d+)", p.name).group(1)))


def summary_to_row(summary: dict, name: str) -> dict:
    return {
        "name": name,
        "psnr": format_metric("psnr", summary["lora_psnr_mean"], summary["lora_psnr_std"]),
        "ssim": format_metric("ssim", summary["lora_ssim_mean"], summary["lora_ssim_std"]),
        "lpips": format_metric("lpips", summary["lora_lpips_mean"], summary["lora_lpips_std"]),
    }


def baseline_row(summary: dict) -> dict:
    return {
        "name": "baseline",
        "psnr": format_metric("psnr", summary["baseline_psnr_mean"], summary["baseline_psnr_std"]),
        "ssim": format_metric("ssim", summary["baseline_ssim_mean"], summary["baseline_ssim_std"]),
        "lpips": format_metric("lpips", summary["baseline_lpips_mean"], summary["baseline_lpips_std"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=str,
        required=True,
        help="Path to delta_experiment/results/evaluation",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output JSON path",
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    out_path = Path(args.output_json)

    delta_a_dirs = find_eval_dirs(eval_root, "deltaA")
    delta_b_dirs = find_eval_dirs(eval_root, "deltaB")

    rows = []
    baseline = None

    for p in delta_a_dirs:
        summary_path = p / "baseline_vs_delta" / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_summary(summary_path)
        if baseline is None:
            baseline = baseline_row(summary)
        steps = re.search(r"steps(\d+)", p.name).group(1)
        rows.append(summary_to_row(summary, f"delta_A_steps{steps}"))

    for p in delta_b_dirs:
        summary_path = p / "baseline_vs_delta" / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_summary(summary_path)
        if baseline is None:
            baseline = baseline_row(summary)
        steps = re.search(r"steps(\d+)", p.name).group(1)
        rows.append(summary_to_row(summary, f"delta_B_steps{steps}"))

    result = {
        "eval_root": str(eval_root),
        "rows": ([baseline] if baseline else []) + rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
