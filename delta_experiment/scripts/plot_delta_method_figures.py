#!/usr/bin/env python3
"""
Generate presentation-ready plots for δ-based TTA experiments.

Inputs:
  - One "main" comparison JSON produced by delta_experiment/scripts/compare_methods.py
    (e.g., delta_experiment/results/benchmark_comparison.json)
  - Optional group-sweep comparison JSONs for δ-B group-count study
    (e.g., benchmark_comparison_g2_2.json, benchmark_comparison_g4_4.json, benchmark_comparison_g8_8.json)

Outputs:
  - Standard 4-method bar charts (baseline / LoRA / global-δ / multi-δ) for PSNR/SSIM/LPIPS
  - δ-B group-count study plots: Δmetric vs (2,2)/(4,4)/(8,8)
  - Parameter-efficiency plots: Δmetric vs trainable params (log-x)
  - Compute-efficiency plots: ΔLPIPS vs train_s (Pareto-style scatter)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_mean(s: Any) -> Optional[float]:
    """
    Accept either a float, or a string like:
      - "15.49 ± 3.42"
      - "0.4943 ± 0.1620"
      - "N/A"
    Returns the mean as float, else None.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    if "N/A" in s or s.strip().lower() in {"na", "n/a"}:
        return None
    m = _FLOAT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_seconds(s: Any) -> Optional[float]:
    """
    Accept either float, or "25.6s", or "N/A".
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    if "N/A" in s or s.strip().lower() in {"na", "n/a"}:
        return None
    m = _FLOAT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


@dataclass
class MethodRow:
    name: str
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]
    time_total_s: Optional[float]
    time_train_s: Optional[float]
    time_gen_s: Optional[float]


def load_comparison_rows(path: Path) -> Dict[str, MethodRow]:
    d = load_json(path)
    if not isinstance(d, dict) or "rows" not in d:
        raise ValueError(f"Expected compare_methods output JSON with 'rows': {path}")
    rows = d["rows"]
    if not isinstance(rows, list):
        raise ValueError(f"Expected rows list: {path}")

    out: Dict[str, MethodRow] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name"))
        out[name] = MethodRow(
            name=name,
            psnr=parse_mean(r.get("psnr")),
            ssim=parse_mean(r.get("ssim")),
            lpips=parse_mean(r.get("lpips")),
            time_total_s=parse_seconds(r.get("time_total")),
            time_train_s=parse_seconds(r.get("time_tta_train")),
            time_gen_s=parse_seconds(r.get("time_generate")),
        )
    return out


def delta_vs_baseline(baseline: MethodRow, m: MethodRow) -> Dict[str, Optional[float]]:
    """
    Improvements:
      - PSNR: higher better => m - baseline
      - SSIM: higher better => m - baseline
      - LPIPS: lower better => baseline - m
    """
    d_psnr = None if baseline.psnr is None or m.psnr is None else (m.psnr - baseline.psnr)
    d_ssim = None if baseline.ssim is None or m.ssim is None else (m.ssim - baseline.ssim)
    d_lpips = None if baseline.lpips is None or m.lpips is None else (baseline.lpips - m.lpips)
    return {"psnr": d_psnr, "ssim": d_ssim, "lpips": d_lpips}


def ensure_out_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(path: Path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def plot_metric_bars(
    out_dir: Path,
    metric: str,
    rows: Dict[str, MethodRow],
    title: str,
    order: List[str],
    colors: Dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = []
    labels = []
    for k in order:
        r = rows.get(k)
        if not r:
            continue
        v = getattr(r, metric)
        if v is None:
            continue
        values.append(v)
        labels.append(k)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = list(range(len(values)))
    bar_colors = [colors.get(lbl, "#666666") for lbl in labels]
    ax.bar(xs, values, color=bar_colors, width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11, wrap=True)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel(metric.upper())
    fig.tight_layout()
    save_fig(out_dir / f"metric_bar_{metric}.png", fig)
    plt.close(fig)


def plot_group_sweep_delta(
    out_dir: Path,
    metric: str,
    group_paths: List[Path],
    group_labels: List[str],
    lora_key: str = "delta_B",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(group_paths) != len(group_labels):
        raise ValueError("group_paths and group_labels must have same length")

    deltas = []
    for p in group_paths:
        rows = load_comparison_rows(p)
        baseline = rows["baseline"]
        m = rows[lora_key]
        d = delta_vs_baseline(baseline, m)[metric]
        deltas.append(d if d is not None else float("nan"))

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    xs = list(range(len(group_labels)))
    ax.bar(xs, deltas, color="#4C9F70", width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_title(f"δ-B group count sweep (20 steps) — Δ{metric.upper()} vs baseline", fontsize=11, wrap=True)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel(f"Δ{metric.upper()} (LPIPS is baseline−method)" if metric == "lpips" else f"Δ{metric.upper()}")
    fig.tight_layout()
    save_fig(out_dir / f"group_sweep_delta_{metric}.png", fig)
    plt.close(fig)


def plot_params_vs_delta(
    out_dir: Path,
    metric: str,
    baseline: MethodRow,
    methods: List[Tuple[str, MethodRow, int]],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = []
    ys = []
    labels = []
    for name, row, params in methods:
        d = delta_vs_baseline(baseline, row)[metric]
        if d is None:
            continue
        xs.append(params)
        ys.append(d)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.scatter(xs, ys)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel(f"Δ{metric.upper()} (LPIPS is baseline−method)" if metric == "lpips" else f"Δ{metric.upper()}")
    ax.set_title(title, fontsize=11, wrap=True)
    fig.tight_layout()
    save_fig(out_dir / f"params_vs_delta_{metric}.png", fig)
    plt.close(fig)


def plot_train_time_vs_delta_lpips(
    out_dir: Path,
    baseline: MethodRow,
    methods: List[Tuple[str, MethodRow]],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = []
    ys = []
    labels = []
    for name, row in methods:
        if row.time_train_s is None:
            continue
        d_lpips = delta_vs_baseline(baseline, row)["lpips"]
        if d_lpips is None:
            continue
        xs.append(row.time_train_s)
        ys.append(d_lpips)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.scatter(xs, ys)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Training time per video (s)")
    ax.set_ylabel("ΔLPIPS (baseline−method)")
    ax.set_title(title, fontsize=11, wrap=True)
    fig.tight_layout()
    save_fig(out_dir / "train_time_vs_delta_lpips.png", fig)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--main-comparison-json", type=str, required=True, help="benchmark_comparison*.json from compare_methods.py")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--group-comparison-jsons",
        type=str,
        nargs="*",
        default=[],
        help="Optional: comparison JSONs for δ-B groups, e.g. g2_2, g4_4, g8_8 (same config except groups).",
    )
    p.add_argument(
        "--group-labels",
        type=str,
        nargs="*",
        default=[],
        help="Optional: labels for group-comparison-jsons, e.g. '(2,2)' '(4,4)' '(8,8)'.",
    )
    p.add_argument("--lora-trainable-params", type=int, default=5_200_000, help="LoRA trainable param count (for plots).")
    args = p.parse_args()

    out_dir = ensure_out_dir(Path(args.out_dir))
    main_rows = load_comparison_rows(Path(args.main_comparison_json))

    # Standard method order + colors
    order = ["baseline", "best_lora", "delta_A", "delta_B"]
    colors = {
        "baseline": "#9AA0A6",  # gray
        "best_lora": "#0037FF",  # v2.0 blue
        "delta_A": "#4985B6",  # muted blue
        "delta_B": "#4C9F70",  # green
    }

    plot_metric_bars(
        out_dir,
        metric="psnr",
        rows=main_rows,
        title="PSNR — Baseline vs LoRA vs Global-δ vs Multi-δ",
        order=order,
        colors=colors,
    )
    plot_metric_bars(
        out_dir,
        metric="ssim",
        rows=main_rows,
        title="SSIM — Baseline vs LoRA vs Global-δ vs Multi-δ",
        order=order,
        colors=colors,
    )
    plot_metric_bars(
        out_dir,
        metric="lpips",
        rows=main_rows,
        title="LPIPS — Baseline vs LoRA vs Global-δ vs Multi-δ",
        order=order,
        colors=colors,
    )

    # Group sweep plots (optional)
    if args.group_comparison_jsons:
        gps = [Path(x) for x in args.group_comparison_jsons]
        glabels = args.group_labels if args.group_labels else [p.stem for p in gps]
        for metric in ["psnr", "ssim", "lpips"]:
            plot_group_sweep_delta(out_dir, metric=metric, group_paths=gps, group_labels=glabels)

    # Parameter efficiency plots (main comparison)
    baseline = main_rows["baseline"]
    lora = main_rows.get("best_lora")
    da = main_rows.get("delta_A")
    db = main_rows.get("delta_B")

    def params_delta_a() -> int:
        return 3072

    def params_delta_b(groups_double: int = 4, groups_single: int = 4) -> int:
        return 3072 * (groups_double + groups_single + 1)

    methods_for_params: List[Tuple[str, MethodRow, int]] = []
    if lora:
        methods_for_params.append(("LoRA (20 steps)", lora, int(args.lora_trainable_params)))
    if da:
        methods_for_params.append(("Global-δ", da, params_delta_a()))
    if db:
        # We assume main comparison's δ-B is the multi-δ (4,4) baseline; adjust if needed.
        methods_for_params.append(("Multi-δ (4,4)", db, params_delta_b(4, 4)))

    for metric in ["psnr", "ssim", "lpips"]:
        plot_params_vs_delta(
            out_dir,
            metric=metric,
            baseline=baseline,
            methods=methods_for_params,
            title=f"Parameter efficiency — Δ{metric.upper()} vs trainable parameters (log-x)",
        )

    # Compute-efficiency (train time vs ΔLPIPS)
    methods_for_time: List[Tuple[str, MethodRow]] = []
    if lora:
        methods_for_time.append(("LoRA (20 steps)", lora))
    if da:
        methods_for_time.append(("Global-δ", da))
    if db:
        methods_for_time.append(("Multi-δ (4,4)", db))
    plot_train_time_vs_delta_lpips(
        out_dir,
        baseline=baseline,
        methods=methods_for_time,
        title="Compute efficiency — training time vs LPIPS improvement",
    )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()

