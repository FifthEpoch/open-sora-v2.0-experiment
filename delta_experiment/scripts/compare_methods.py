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


def load_timing_summary(run_dir: Path) -> dict | None:
    """
    Load run-level timing summary produced by experiment_timing.write_timing_files().
    """
    p = run_dir / "timing_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt_seconds(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}s"


def timing_mean_seconds(timing: dict | None, key: str) -> float | None:
    """
    key: 'total_s' or a phase key like 'generate' / 'tta_train'
    """
    if timing is None:
        return None
    if key == "total_s":
        return (timing.get("total_s") or {}).get("mean_s")
    return ((timing.get("phases_s") or {}).get(key) or {}).get("mean_s")


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Best-LoRA vs δ methods")
    parser.add_argument("--baseline_eval_dir", type=str, required=True, help="Evaluation dir containing baseline metrics (summary.json)")
    parser.add_argument("--bestlora_eval_dir", type=str, required=True, help="Evaluation dir containing best LoRA metrics (summary.json)")
    parser.add_argument(
        "--bestlora_side",
        type=str,
        choices=["baseline", "lora"],
        default="baseline",
        help=(
            "Which side of bestlora_eval_dir to report as 'best_lora'. "
            "Use 'baseline' for legacy delta_experiment evaluate_delta outputs (best LoRA stored under baseline_*), "
            "or 'lora' when bestlora_eval_dir comes from evaluate.py (best LoRA stored under lora_*)."
        ),
    )
    parser.add_argument("--delta_a_eval_dir", type=str, required=False)
    parser.add_argument("--delta_b_eval_dir", type=str, required=False)
    parser.add_argument("--delta_c_eval_dir", type=str, required=False)
    parser.add_argument("--baseline_run_dir", type=str, required=False, help="Method run dir containing timing_summary.json (baseline generation outputs)")
    parser.add_argument("--bestlora_run_dir", type=str, required=False, help="Method run dir containing timing_summary.json (best LoRA outputs)")
    parser.add_argument("--delta_a_run_dir", type=str, required=False, help="Method run dir containing timing_summary.json (delta-A outputs)")
    parser.add_argument("--delta_b_run_dir", type=str, required=False, help="Method run dir containing timing_summary.json (delta-B outputs)")
    parser.add_argument("--delta_c_run_dir", type=str, required=False, help="Method run dir containing timing_summary.json (delta-C outputs)")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    baseline = load_summary(Path(args.baseline_eval_dir))
    bestlora = load_summary(Path(args.bestlora_eval_dir))

    baseline_timing = load_timing_summary(Path(args.baseline_run_dir)) if args.baseline_run_dir else None
    bestlora_timing = load_timing_summary(Path(args.bestlora_run_dir)) if args.bestlora_run_dir else None
    delta_a_timing = load_timing_summary(Path(args.delta_a_run_dir)) if args.delta_a_run_dir else None
    delta_b_timing = load_timing_summary(Path(args.delta_b_run_dir)) if args.delta_b_run_dir else None
    delta_c_timing = load_timing_summary(Path(args.delta_c_run_dir)) if args.delta_c_run_dir else None

    rows = []

    def row_from(summary: dict, name: str, side: str, timing: dict | None) -> dict:
        """
        side: 'baseline' -> use baseline_* metrics, 'lora' -> use lora_* metrics.
        """
        prefix = "baseline_" if side == "baseline" else "lora_"
        return {
            "name": name,
            "psnr": fmt(summary.get(f"{prefix}psnr_mean"), summary.get(f"{prefix}psnr_std"), digits=2),
            "ssim": fmt(summary.get(f"{prefix}ssim_mean"), summary.get(f"{prefix}ssim_std"), digits=4),
            "lpips": fmt(summary.get(f"{prefix}lpips_mean"), summary.get(f"{prefix}lpips_std"), digits=4),
            "time_total": fmt_seconds(timing_mean_seconds(timing, "total_s")),
            "time_tta_train": fmt_seconds(timing_mean_seconds(timing, "tta_train")),
            "time_generate": fmt_seconds(timing_mean_seconds(timing, "generate")),
        }

    # baseline_eval_dir: baseline_* refers to true baseline
    rows.append(row_from(baseline, "baseline", side="baseline", timing=baseline_timing))
    # bestlora_eval_dir: configurable which side is "best_lora" (legacy vs benchmark usage)
    rows.append(row_from(bestlora, "best_lora", side=args.bestlora_side, timing=bestlora_timing))

    for key, label in [
        ("delta_a_eval_dir", "delta_A"),
        ("delta_b_eval_dir", "delta_B"),
        ("delta_c_eval_dir", "delta_C"),
    ]:
        p = getattr(args, key)
        if p:
            timing = None
            if label == "delta_A":
                timing = delta_a_timing
            elif label == "delta_B":
                timing = delta_b_timing
            elif label == "delta_C":
                timing = delta_c_timing
            rows.append(row_from(load_summary(Path(p)), label, side="lora", timing=timing))

    out = {
        "baseline_eval_dir": args.baseline_eval_dir,
        "bestlora_eval_dir": args.bestlora_eval_dir,
        "baseline_run_dir": args.baseline_run_dir,
        "bestlora_run_dir": args.bestlora_run_dir,
        "delta_a_run_dir": args.delta_a_run_dir,
        "delta_b_run_dir": args.delta_b_run_dir,
        "delta_c_run_dir": args.delta_c_run_dir,
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Print a small table to stdout
    print("name\tpsnr\tssim\tlpips\ttotal_time\ttrain_time\tgen_time")
    for r in rows:
        print(
            f"{r['name']}\t{r['psnr']}\t{r['ssim']}\t{r['lpips']}\t"
            f"{r['time_total']}\t{r['time_tta_train']}\t{r['time_generate']}"
        )


if __name__ == "__main__":
    main()


