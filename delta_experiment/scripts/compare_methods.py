#!/usr/bin/env python3
"""
Aggregate evaluation outputs into a single comparison table.

Inputs are the evaluation output directories created by:
- lora_experiment/scripts/evaluate.py (baseline vs some method)
- delta_experiment/scripts/evaluate_delta.py (baseline_vs_delta and bestlora_vs_delta)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    p = path / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with open(p) as f:
        return json.load(f)


def fmt(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Best-LoRA vs δ methods")
    parser.add_argument("--baseline_eval_dir", type=str, required=True, help="Evaluation dir containing baseline metrics (summary.json)")
    parser.add_argument("--bestlora_eval_dir", type=str, required=True, help="Evaluation dir containing best LoRA metrics (summary.json)")
    parser.add_argument("--delta_a_eval_dir", type=str, required=False)
    parser.add_argument("--delta_b_eval_dir", type=str, required=False)
    parser.add_argument("--delta_c_eval_dir", type=str, required=False)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    baseline = load_summary(Path(args.baseline_eval_dir))
    bestlora = load_summary(Path(args.bestlora_eval_dir))

    rows = []

    def row_from(summary: dict, name: str, side: str) -> dict:
        """
        side: 'baseline' -> use baseline_* metrics, 'lora' -> use lora_* metrics.
        """
        prefix = "baseline_" if side == "baseline" else "lora_"
        return {
            "name": name,
            "psnr": fmt(summary.get(f"{prefix}psnr_mean"), summary.get(f"{prefix}psnr_std"), digits=2),
            "ssim": fmt(summary.get(f"{prefix}ssim_mean"), summary.get(f"{prefix}ssim_std"), digits=4),
            "lpips": fmt(summary.get(f"{prefix}lpips_mean"), summary.get(f"{prefix}lpips_std"), digits=4),
        }

    # baseline_eval_dir is a baseline_vs_delta summary -> baseline_* refers to true baseline
    rows.append(row_from(baseline, "baseline", side="baseline"))
    # bestlora_eval_dir is a bestlora_vs_delta summary -> baseline_* refers to best LoRA
    rows.append(row_from(bestlora, "best_lora", side="baseline"))

    for key, label in [
        ("delta_a_eval_dir", "delta_A"),
        ("delta_b_eval_dir", "delta_B"),
        ("delta_c_eval_dir", "delta_C"),
    ]:
        p = getattr(args, key)
        if p:
            rows.append(row_from(load_summary(Path(p)), label, side="lora"))

    out = {
        "baseline_eval_dir": args.baseline_eval_dir,
        "bestlora_eval_dir": args.bestlora_eval_dir,
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Print a small table to stdout
    print("name\tpsnr\tssim\tlpips")
    for r in rows:
        print(f"{r['name']}\t{r['psnr']}\t{r['ssim']}\t{r['lpips']}")


if __name__ == "__main__":
    main()


