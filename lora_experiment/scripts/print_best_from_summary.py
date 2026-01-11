#!/usr/bin/env python3
"""
Print best-performing hyperparameters from a sweep summary JSON produced by summarize_evals.py.

Example:
  python lora_experiment/scripts/print_best_from_summary.py \
    --summary-json delta_experiment/results/delta_b_sweep_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def pick_best(rows: list[dict[str, Any]], key: str, higher_is_better: bool = True) -> dict[str, Any] | None:
    best = None
    best_v = None
    for r in rows:
        v = as_float(r.get(key))
        if v is None:
            continue
        if best is None:
            best, best_v = r, v
            continue
        if higher_is_better:
            if v > best_v:
                best, best_v = r, v
        else:
            if v < best_v:
                best, best_v = r, v
    return best


def fmt_row(r: dict[str, Any], metric_key: str) -> str:
    run = r.get("run_name", "N/A")
    metric = r.get(metric_key)
    cfg_bits = []
    for k in ["delta_steps", "delta_lr", "delta_l2", "groups_double", "groups_single", "learning_rate", "num_steps", "lora_rank"]:
        if k in r and r.get(k) is not None:
            cfg_bits.append(f"{k}={r.get(k)}")
    cfg = ", ".join(cfg_bits) if cfg_bits else "(no config fields found)"
    n = r.get("num_videos_evaluated")
    return f"{run} | {metric_key}={metric} | n={n} | {cfg}"


def top_k(rows: list[dict[str, Any]], key: str, k: int, higher_is_better: bool = True) -> list[dict[str, Any]]:
    scored = []
    for r in rows:
        v = as_float(r.get(key))
        if v is None:
            continue
        scored.append((v, r))
    scored.sort(key=lambda t: t[0], reverse=higher_is_better)
    return [r for _, r in scored[:k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print best configs from sweep summary JSON")
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5, help="Print top-K runs per metric (default: 5)")
    args = parser.parse_args()

    data = load_json(Path(args.summary_json))
    rows = data.get("rows") or []
    if not rows:
        print(f"No rows found in {args.summary_json}.")
        print("This usually means eval outputs (eval_baseline/summary.json) are missing for the runs.")
        return

    metrics = [
        ("psnr_improvement_mean", True),
        ("ssim_improvement_mean", True),
        ("lpips_improvement_mean", True),
    ]

    print(f"Loaded {len(rows)} evaluated runs from: {args.summary_json}")
    print("")

    for metric_key, hib in metrics:
        best = pick_best(rows, metric_key, higher_is_better=hib)
        print(f"=== BEST by {metric_key} ({'higher' if hib else 'lower'} is better) ===")
        if best is None:
            print("No valid values found.")
            print("")
            continue
        print(fmt_row(best, metric_key))
        print("")

        if args.top_k and args.top_k > 1:
            print(f"Top {args.top_k} by {metric_key}:")
            for r in top_k(rows, metric_key, k=args.top_k, higher_is_better=hib):
                print("  - " + fmt_row(r, metric_key))
            print("")


if __name__ == "__main__":
    main()

