#!/usr/bin/env python3
"""
Summarize many evaluation runs into a single table (CSV/JSON).

This script is intended for sweeps (LoRA LR sweep, delta-B sweep, etc.) where each
run directory contains:
  - eval_baseline/summary.json  (from lora_experiment/scripts/evaluate.py)
  - config.json                (saved by the run script)
  - timing_summary.json        (saved by experiment_timing.py; optional)

Example usage:
  python lora_experiment/scripts/summarize_evals.py \
    --runs-glob "delta_experiment/results/delta_b_sweep/*" \
    --output-csv delta_experiment/results/delta_b_sweep_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def safe_get(d: dict[str, Any] | None, key: str) -> Any:
    if not d:
        return None
    return d.get(key)


def timing_mean_seconds(timing: dict[str, Any] | None, key: str) -> float | None:
    """
    key: 'total_s' or phase name (e.g., 'tta_train', 'generate')
    """
    if not timing:
        return None
    if key == "total_s":
        return (timing.get("total_s") or {}).get("mean_s")
    return ((timing.get("phases_s") or {}).get(key) or {}).get("mean_s")


def flatten_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """
    Pull out the most relevant hyperparameters from config.json, across LoRA and delta runs.
    """
    if not cfg:
        return {}
    keys = [
        # common
        "seed",
        "max_videos",
        "stratified",
        "dtype",
        "inference_steps",
        "guidance",
        "guidance_img",
        # LoRA
        "lora_rank",
        "lora_alpha",
        "learning_rate",
        "num_steps",
        "warmup_steps",
        "target_mlp",
        # delta
        "delta_steps",
        "delta_lr",
        "delta_l2",
        "weight_decay",
        "max_grad_norm",
        "groups_double",
        "groups_single",
        "reference_results_json",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if k in cfg:
            out[k] = cfg.get(k)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval_baseline/summary.json across many run dirs")
    parser.add_argument(
        "--runs-glob",
        type=str,
        required=True,
        help="Glob that expands to run directories (e.g., 'delta_experiment/results/delta_b_sweep/*')",
    )
    parser.add_argument(
        "--eval-subdir",
        type=str,
        default="eval_baseline",
        help="Subdir under each run dir where evaluation outputs live (default: eval_baseline)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="psnr_improvement_mean",
        help="Field in summary.json to sort descending by (default: psnr_improvement_mean)",
    )
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    matches = glob.glob(args.runs_glob)
    run_dirs = sorted([Path(m) for m in matches])
    if not run_dirs:
        cwd = os.getcwd()
        raise FileNotFoundError(
            f"No run directories matched: {args.runs_glob}\n"
            f"cwd: {cwd}\n"
            f"Tip: run `ls {args.runs_glob}` to confirm the pattern, or try a broader glob like "
            f"`--runs-glob \"delta_experiment/results/delta_b_sweep/*\"`."
        )

    rows: list[dict[str, Any]] = []
    missing_eval: list[str] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue

        summary = load_json(run_dir / args.eval_subdir / "summary.json")
        if not summary:
            # Skip runs that haven't been evaluated yet.
            missing_eval.append(run_dir.name)
            continue

        cfg = load_json(run_dir / "config.json")
        timing = load_json(run_dir / "timing_summary.json")

        row: dict[str, Any] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            "eval_dir": str((run_dir / args.eval_subdir).resolve()),
            # evaluation metrics (means)
            "num_videos_evaluated": summary.get("num_videos_evaluated"),
            "psnr_improvement_mean": summary.get("psnr_improvement_mean"),
            "ssim_improvement_mean": summary.get("ssim_improvement_mean"),
            "lpips_improvement_mean": summary.get("lpips_improvement_mean"),
            "baseline_psnr_mean": summary.get("baseline_psnr_mean"),
            "lora_psnr_mean": summary.get("lora_psnr_mean"),
            "baseline_ssim_mean": summary.get("baseline_ssim_mean"),
            "lora_ssim_mean": summary.get("lora_ssim_mean"),
            "baseline_lpips_mean": summary.get("baseline_lpips_mean"),
            "lora_lpips_mean": summary.get("lora_lpips_mean"),
            # compute cost
            "time_total_mean_s": timing_mean_seconds(timing, "total_s"),
            "time_tta_train_mean_s": timing_mean_seconds(timing, "tta_train"),
            "time_generate_mean_s": timing_mean_seconds(timing, "generate"),
        }

        row.update(flatten_config(cfg))
        rows.append(row)

    # Sort descending by chosen metric (missing values last)
    sort_key = args.sort_by

    def key_fn(r: dict[str, Any]) -> tuple[int, float]:
        v = r.get(sort_key)
        if v is None:
            return (1, 0.0)
        try:
            return (0, float(v))
        except Exception:
            return (1, 0.0)

    rows.sort(key=key_fn, reverse=True)

    out_csv = Path(args.output_csv)
    write_csv(out_csv, rows)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(
                {
                    "runs_glob": args.runs_glob,
                    "eval_subdir": args.eval_subdir,
                    "sorted_by": sort_key,
                    "num_rows": len(rows),
                    "rows": rows,
                },
                f,
                indent=2,
            )

    print(f"Found {len(run_dirs)} run dirs for pattern: {args.runs_glob}")
    print(f"Wrote {len(rows)} evaluated rows to {out_csv}")
    if missing_eval:
        preview = ", ".join(missing_eval[:10])
        more = "" if len(missing_eval) <= 10 else f" (+{len(missing_eval) - 10} more)"
        print(f"Skipped {len(missing_eval)} runs missing {args.eval_subdir}/summary.json: {preview}{more}")
    if args.output_json:
        print(f"Wrote JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()

