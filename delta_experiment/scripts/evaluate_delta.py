#!/usr/bin/env python3
"""
Evaluate δ-TTA outputs against ground truth and compare to baseline and best-LoRA.

This script wraps the existing evaluation logic in lora_experiment/scripts/evaluate.py
so metrics stay consistent (PSNR/SSIM/LPIPS on frames 33-64).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Evaluate δ-TTA results")
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--best-lora-dir", type=str, required=True)
    parser.add_argument("--delta-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--eval-frame-count", type=int, default=None)
    parser.add_argument("--eval-frame-stride", type=int, default=1)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--best-of", type=int, default=1)
    args = parser.parse_args()

    root = project_root()
    sys.path.insert(0, str(root))

    # Import existing evaluation code
    from lora_experiment.scripts.evaluate import run_evaluation

    # Evaluate baseline vs delta
    class Obj:
        pass

    # 1) Baseline vs delta
    a = Obj()
    a.gt_dir = args.gt_dir
    a.baseline_dir = args.baseline_dir
    a.lora_dir = args.delta_dir
    a.output_dir = str(Path(args.output_dir) / "baseline_vs_delta")
    a.max_videos = args.max_videos
    a.eval_frame_count = args.eval_frame_count
    a.eval_frame_stride = args.eval_frame_stride
    a.include_context = args.include_context
    a.best_of = args.best_of
    run_evaluation(a)

    # 2) Best LoRA vs delta (treat best LoRA as "baseline" in a separate run)
    b = Obj()
    b.gt_dir = args.gt_dir
    b.baseline_dir = args.best_lora_dir
    b.lora_dir = args.delta_dir
    b.output_dir = str(Path(args.output_dir) / "bestlora_vs_delta")
    b.max_videos = args.max_videos
    b.eval_frame_count = args.eval_frame_count
    b.eval_frame_stride = args.eval_frame_stride
    b.include_context = args.include_context
    b.best_of = args.best_of
    run_evaluation(b)


if __name__ == "__main__":
    main()


