#!/usr/bin/env python3
"""
Plot Open-Sora v1.3 vs v2.0 UCF101 video-continuation metrics.

This script is designed to work with:
- v1.3 naive experiment metrics:
    Open-Sora-1.3/naive_experiment/scripts/results/RUN_NAME/metrics.json
  (a JSON list with per-video "baseline" and "finetuned" metrics + timing)

- v2.0 evaluator outputs:
    Open-Sora-2.0/lora_experiment/scripts/evaluate.py
  which produces:
    <eval_output_dir>/summary.json
    <eval_output_dir>/detailed_metrics.csv

Example:
  python delta_experiment/scripts/plot_ucf101_v13_vs_v20.py \
    --v13-metrics /abs/path/Open-Sora-1.3/naive_experiment/scripts/results/100steps_5e5/metrics.json \
    --v20-eval-dir /abs/path/Open-Sora-2.0/lora_experiment/results/evaluation \
    --output-dir /abs/path/Open-Sora-2.0/delta_experiment/plots/ucf101_v13_vs_v20
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RunLabel:
    version: str
    method: str

    def pretty(self) -> str:
        return f"{self.version} • {self.method}"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return float(str(x))
    except Exception:
        return None


def load_v13_naive_metrics(metrics_json: Path) -> Tuple[list[dict[str, Any]], Dict[str, float]]:
    """
    Returns:
      - per_video_df: one row per video, with baseline_* and finetuned_* columns (psnr/ssim/lpips + timing)
      - summary: aggregate means/stds
    """
    with metrics_json.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in v1.3 metrics file, got: {type(data)}")

    rows: list[dict[str, Any]] = []
    for item in data:
        baseline = item.get("baseline") or {}
        finetuned = item.get("finetuned") or {}

        rows.append(
            {
                "video_idx": item.get("video_idx"),
                "video_name": Path(str(item.get("original_path", ""))).stem or None,
                "baseline_psnr": _safe_float(baseline.get("psnr")),
                "baseline_ssim": _safe_float(baseline.get("ssim")),
                "baseline_lpips": _safe_float(baseline.get("lpips")),
                "finetuned_psnr": _safe_float(finetuned.get("psnr")),
                "finetuned_ssim": _safe_float(finetuned.get("ssim")),
                "finetuned_lpips": _safe_float(finetuned.get("lpips")),
                "baseline_inference_time_sec": _safe_float(item.get("baseline_inference_time_sec")),
                "finetune_time_sec": _safe_float(item.get("finetune_time_sec")),
                "finetuned_inference_time_sec": _safe_float(item.get("finetuned_inference_time_sec")),
            }
        )

    summary: Dict[str, float] = {"num_videos": int(len(rows))}
    for col in [
        "baseline_psnr",
        "baseline_ssim",
        "baseline_lpips",
        "finetuned_psnr",
        "finetuned_ssim",
        "finetuned_lpips",
        "baseline_inference_time_sec",
        "finetune_time_sec",
        "finetuned_inference_time_sec",
    ]:
        vals = [_safe_float(r.get(col)) for r in rows]
        valid = np.array([v for v in vals if v is not None], dtype=float)
        if valid.size == 0:
            continue
        summary[f"{col}_mean"] = float(np.mean(valid))
        summary[f"{col}_std"] = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
        summary[f"{col}_median"] = float(np.median(valid))

    return rows, summary


def load_v20_eval_dir(eval_dir: Path) -> Tuple[list[dict[str, Any]], Dict[str, Any]]:
    """
    Loads v2.0 evaluator outputs.

    Returns:
      - detailed_metrics_df (may be empty if not found)
      - summary.json dict (may be empty if not found)
    """
    detailed_csv = eval_dir / "detailed_metrics.csv"
    summary_json = eval_dir / "summary.json"

    detailed_rows: list[dict[str, Any]] = []
    if detailed_csv.exists():
        with detailed_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned: dict[str, Any] = {}
                for k, v in row.items():
                    if v is None or v == "":
                        cleaned[k] = None
                        continue
                    fv = _safe_float(v)
                    cleaned[k] = fv if fv is not None else v
                detailed_rows.append(cleaned)

    summary: Dict[str, Any] = {}
    if summary_json.exists():
        with summary_json.open("r") as f:
            summary = json.load(f)

    return detailed_rows, summary


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    return plt


def plot_metric_bars(
    out_path: Path,
    series: Dict[RunLabel, Dict[str, float]],
    metrics: Tuple[str, ...] = ("psnr", "ssim", "lpips"),
    title: str = "UCF101 Video Continuation — Summary Metrics",
):
    plt = _import_matplotlib()

    labels = [k.pretty() for k in series.keys()]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, m in enumerate(metrics):
        vals = [series[k].get(m, np.nan) for k in series.keys()]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width=width, label=m.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_boxplots(
    out_path: Path,
    rows: list[dict[str, Any]],
    value_cols: Tuple[str, ...],
    title: str,
):
    """
    rows format:
      - each row: {"label": "...", <value_cols...>: float}
    """
    plt = _import_matplotlib()

    # Simple faceted boxplot (one subplot per metric)
    metrics = list(value_cols)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        labels = sorted({r["label"] for r in rows if r.get("label") is not None})
        data = []
        for l in labels:
            vals = [_safe_float(r.get(metric)) for r in rows if r.get("label") == l]
            vals = [v for v in vals if v is not None]
            data.append(np.array(vals, dtype=float))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(metric.replace("_", " ").upper())
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v13-metrics", type=str, required=True, help="Path to v1.3 naive_experiment metrics.json")
    p.add_argument("--v20-eval-dir", type=str, default=None, help="Path to v2.0 evaluation output directory")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for plots")
    p.add_argument("--title", type=str, default="UCF101 Video Continuation — Open-Sora v1.3 vs v2.0")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v13_metrics_path = Path(args.v13_metrics)
    v13_rows, v13_summary = load_v13_naive_metrics(v13_metrics_path)

    # Build summary series (means)
    series: Dict[RunLabel, Dict[str, float]] = {
        RunLabel("v1.3", "baseline"): {
            "psnr": float(v13_summary.get("baseline_psnr_mean", np.nan)),
            "ssim": float(v13_summary.get("baseline_ssim_mean", np.nan)),
            "lpips": float(v13_summary.get("baseline_lpips_mean", np.nan)),
        },
        RunLabel("v1.3", "finetuned"): {
            "psnr": float(v13_summary.get("finetuned_psnr_mean", np.nan)),
            "ssim": float(v13_summary.get("finetuned_ssim_mean", np.nan)),
            "lpips": float(v13_summary.get("finetuned_lpips_mean", np.nan)),
        },
    }

    # Optional v2.0
    v20_detailed: list[dict[str, Any]] = []
    v20_summary: Dict[str, Any] = {}
    if args.v20_eval_dir:
        v20_detailed, v20_summary = load_v20_eval_dir(Path(args.v20_eval_dir))
        if v20_summary:
            if "baseline_psnr_mean" in v20_summary:
                series[RunLabel("v2.0", "baseline")] = {
                    "psnr": float(v20_summary.get("baseline_psnr_mean", np.nan)),
                    "ssim": float(v20_summary.get("baseline_ssim_mean", np.nan)),
                    "lpips": float(v20_summary.get("baseline_lpips_mean", np.nan)),
                }
            if "lora_psnr_mean" in v20_summary:
                series[RunLabel("v2.0", "tta/lora")] = {
                    "psnr": float(v20_summary.get("lora_psnr_mean", np.nan)),
                    "ssim": float(v20_summary.get("lora_ssim_mean", np.nan)),
                    "lpips": float(v20_summary.get("lora_lpips_mean", np.nan)),
                }

    # Write combined JSON for provenance
    combined = {
        "v13_metrics": str(v13_metrics_path),
        "v20_eval_dir": str(args.v20_eval_dir) if args.v20_eval_dir else None,
        "v13_summary": v13_summary,
        "v20_summary": v20_summary,
    }
    (out_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2))

    # Plot summary bars
    plot_metric_bars(
        out_path=out_dir / "summary_metrics_bar.png",
        series=series,
        title=args.title,
    )

    # Plot per-video boxplots where available
    rows: list[dict[str, Any]] = []
    for r in v13_rows:
        rows.append(
            {
                "label": RunLabel("v1.3", "baseline").pretty(),
                "psnr": _safe_float(r.get("baseline_psnr")),
                "ssim": _safe_float(r.get("baseline_ssim")),
                "lpips": _safe_float(r.get("baseline_lpips")),
            }
        )
        rows.append(
            {
                "label": RunLabel("v1.3", "finetuned").pretty(),
                "psnr": _safe_float(r.get("finetuned_psnr")),
                "ssim": _safe_float(r.get("finetuned_ssim")),
                "lpips": _safe_float(r.get("finetuned_lpips")),
            }
        )

    if v20_detailed:
        # v2.0 detailed metrics columns are baseline_psnr, lora_psnr, etc.
        for r in v20_detailed:
            if "baseline_psnr" in r:
                rows.append(
                    {
                        "label": RunLabel("v2.0", "baseline").pretty(),
                        "psnr": _safe_float(r.get("baseline_psnr")),
                        "ssim": _safe_float(r.get("baseline_ssim")),
                        "lpips": _safe_float(r.get("baseline_lpips")),
                    }
                )
            if "lora_psnr" in r:
                rows.append(
                    {
                        "label": RunLabel("v2.0", "tta/lora").pretty(),
                        "psnr": _safe_float(r.get("lora_psnr")),
                        "ssim": _safe_float(r.get("lora_ssim")),
                        "lpips": _safe_float(r.get("lora_lpips")),
                    }
                )

    plot_metric_boxplots(
        out_path=out_dir / "per_video_boxplots.png",
        rows=rows,
        value_cols=("psnr", "ssim", "lpips"),
        title=f"{args.title} — Per-video distributions",
    )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()

