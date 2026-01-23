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


def find_run_dir(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def summary_to_row(summary: dict, name: str, timing: dict | None = None) -> dict:
    row = {
        "name": name,
        "psnr": format_metric("psnr", summary["lora_psnr_mean"], summary["lora_psnr_std"]),
        "ssim": format_metric("ssim", summary["lora_ssim_mean"], summary["lora_ssim_std"]),
        "lpips": format_metric("lpips", summary["lora_lpips_mean"], summary["lora_lpips_std"]),
    }
    if timing:
        row["train_time_mean_s"] = timing.get("train_time_mean_s")
        row["gen_time_mean_s"] = timing.get("gen_time_mean_s")
    return row


def baseline_row(summary: dict) -> dict:
    return {
        "name": "baseline",
        "psnr": format_metric("psnr", summary["baseline_psnr_mean"], summary["baseline_psnr_std"]),
        "ssim": format_metric("ssim", summary["baseline_ssim_mean"], summary["baseline_ssim_std"]),
        "lpips": format_metric("lpips", summary["baseline_lpips_mean"], summary["baseline_lpips_std"]),
    }


def load_timing(run_dir: Path) -> dict | None:
    timing_path = run_dir / "timing_summary.json"
    if timing_path.exists():
        with open(timing_path) as f:
            data = json.load(f)
        phases = data.get("phases_s", {})
        return {
            "train_time_mean_s": (phases.get("tta_train") or {}).get("mean_s"),
            "gen_time_mean_s": (phases.get("generate") or {}).get("mean_s"),
        }

    # Fallback: compute from per-video jsonl
    per_video = run_dir / "timing_per_video.jsonl"
    if not per_video.exists():
        return None
    train_vals = []
    gen_vals = []
    with open(per_video) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("success", False):
                continue
            phases = rec.get("phases_s") or {}
            if "tta_train" in phases:
                train_vals.append(float(phases["tta_train"]))
            if "generate" in phases:
                gen_vals.append(float(phases["generate"]))
    if not train_vals and not gen_vals:
        return None
    return {
        "train_time_mean_s": sum(train_vals) / len(train_vals) if train_vals else None,
        "gen_time_mean_s": sum(gen_vals) / len(gen_vals) if gen_vals else None,
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
    parser.add_argument(
        "--include-100",
        action="store_true",
        help="Include 100-video runs (baseline/LoRA/delta A/B) with timing if available.",
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
        timing = load_timing(eval_root.parent / "sweep_delta_a_steps" / f"steps{steps}_50videos")
        rows.append(summary_to_row(summary, f"delta_A_steps{steps}", timing))

    for p in delta_b_dirs:
        summary_path = p / "baseline_vs_delta" / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_summary(summary_path)
        if baseline is None:
            baseline = baseline_row(summary)
        steps = re.search(r"steps(\d+)", p.name).group(1)
        timing = load_timing(eval_root.parent / "sweep_delta_b_steps" / f"steps{steps}_50videos")
        rows.append(summary_to_row(summary, f"delta_B_steps{steps}", timing))

    if args.include_100:
        project_root = eval_root.parents[2]
        base_eval = eval_root / "comparison.json"
        if base_eval.exists():
            try:
                base_cmp = load_summary(base_eval)
                for row in base_cmp.get("rows", []):
                    if row.get("name") in ("baseline", "best_lora", "delta_A", "delta_B"):
                        name = row["name"]
                        timing = None
                        if name == "baseline":
                            run_dir = find_run_dir(
                                [
                                    project_root / "lora_experiment" / "results" / "baseline",
                                ]
                            )
                        elif name == "best_lora":
                            run_dir = find_run_dir(
                                [
                                    project_root / "lora_experiment" / "results" / "lora_r8_lr2e-4_20steps_aug",
                                    project_root / "lora_experiment" / "results" / "lora_r8_lr2e-4_20steps",
                                ]
                            )
                        elif name == "delta_A":
                            run_dir = find_run_dir(
                                [
                                    eval_root.parent / "delta_a_global_aug",
                                    eval_root.parent / "delta_a_global",
                                ]
                            )
                        elif name == "delta_B":
                            run_dir = find_run_dir(
                                [
                                    eval_root.parent / "delta_b_grouped_aug",
                                    eval_root.parent / "delta_b_grouped",
                                ]
                            )
                        else:
                            run_dir = None
                        if run_dir:
                            timing = load_timing(run_dir)
                        rows.append(
                            {
                                "name": f"{name}_100videos",
                                "psnr": row["psnr"],
                                "ssim": row["ssim"],
                                "lpips": row["lpips"],
                                "train_time_mean_s": timing.get("train_time_mean_s") if timing else None,
                                "gen_time_mean_s": timing.get("gen_time_mean_s") if timing else None,
                            }
                        )
            except Exception:
                pass

    result = {
        "eval_root": str(eval_root),
        "rows": ([baseline] if baseline else []) + rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
