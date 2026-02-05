#!/usr/bin/env python3
"""
Aggregate outcomes across multiple run directories.

Supports:
  - evaluation outputs: summary.json (from lora_experiment/scripts/evaluate.py)
  - run outputs: results.json, metrics_summary.json, timing_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def extract_eval_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in ("psnr", "ssim", "lpips"):
        # Prefer method-specific entries (baseline_*, lora_*, delta_*)
        for prefix in ("baseline_", "lora_", "delta_", "full_", ""):
            k = f"{prefix}{key}_mean"
            if k in summary:
                row[f"{prefix}{key}_mean"] = summary.get(k)
                std_k = f"{prefix}{key}_std"
                if std_k in summary:
                    row[f"{prefix}{key}_std"] = summary.get(std_k)
    row["num_videos_evaluated"] = summary.get("num_videos_evaluated")
    return row


def aggregate_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [r for r in results if r.get("success", False)]
    row: dict[str, Any] = {
        "success_count": len(successes),
        "total_count": len(results),
    }
    best_psnrs = [r.get("best_psnr") for r in successes if r.get("best_psnr") is not None]
    row["best_psnr_mean"] = mean([float(v) for v in best_psnrs]) if best_psnrs else None
    train_times = [r.get("train_time") for r in successes if isinstance(r.get("train_time"), (int, float))]
    gen_times = [r.get("gen_time") for r in successes if isinstance(r.get("gen_time"), (int, float))]
    row["avg_train_time_s"] = mean([float(v) for v in train_times]) if train_times else None
    row["avg_gen_time_s"] = mean([float(v) for v in gen_times]) if gen_times else None
    return row


def collect_run_metrics(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run": run_dir.name,
        "path": str(run_dir),
    }

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        row.update(extract_eval_summary(summary))

    metrics_summary_path = run_dir / "metrics_summary.json"
    if metrics_summary_path.exists():
        row.update(load_json(metrics_summary_path))

    results_path = run_dir / "results.json"
    if results_path.exists():
        results = load_json(results_path)
        if isinstance(results, list):
            row.update(aggregate_from_results(results))

    timing_summary_path = run_dir / "timing_summary.json"
    if timing_summary_path.exists():
        row.update(load_json(timing_summary_path))

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics from multiple run directories.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories to aggregate")
    parser.add_argument("--output", type=str, default="aggregate_summary",
                        help="Output basename (writes .json and .csv)")
    args = parser.parse_args()

    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    rows = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        rows.append(collect_run_metrics(run_dir))

    out_base = Path(args.output).expanduser().resolve()
    out_json = out_base.with_suffix(".json")
    out_csv = out_base.with_suffix(".csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    write_csv(out_csv, rows)

    print(f"Wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
