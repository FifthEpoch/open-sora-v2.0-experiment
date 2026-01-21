#!/usr/bin/env python3

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def lpips_improvement(baseline_lpips: float, method_lpips: float) -> float:
    # lower is better => improvement is baseline - method
    return baseline_lpips - method_lpips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    with open(args.in_json, "r") as f:
        d = json.load(f)

    base = d["baseline"]
    methods = d["methods"]

    # --- Standard 4-method bar charts: baseline / LoRA / global-δ / multi-δ ---
    order = ["Baseline"] + [m["name"] for m in methods]
    colors = {
        "Baseline": "#9AA0A6",
        "LoRA (20 steps)": "#0037FF",
        "Global-δ (20 steps)": "#4985B6",
        "Multi-δ (20 steps, g4_4, lr=3e-3, l2=3e-4)": "#4C9F70",
    }

    def plot_metric(metric_key: str, title: str, ylabel: str) -> None:
        vals = [base[metric_key]] + [m[metric_key] for m in methods]
        # Narrower figure + tighter x-limits to reduce visible gaps between bars.
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        # Pack bars closer by using a smaller step between x positions.
        xs = [i * 0.78 for i in range(len(vals))]
        ax.bar(
            xs,
            vals,
            color=[colors.get(x, "#666") for x in order],
            width=0.55,
        )
        # Tighten x-axis padding (matplotlib defaults add extra whitespace).
        ax.set_xlim(xs[0] - 0.32, xs[-1] + 0.32)
        # Set a non-misleading y-scale:
        # Use baseline mean ± k*std as the primary range to avoid overstating small deltas,
        # then expand if any bar falls outside that range.
        std_key = metric_key.replace("_mean", "_std")
        base_mean = float(base[metric_key])
        base_std = float(base.get(std_key, 0.0) or 0.0)

        # k controls how "zoomed" we are. Larger k = less exaggeration.
        # Sweet spot for slides: show separation without looking like a huge effect.
        k = 0.20
        ymin = base_mean - k * base_std
        ymax = base_mean + k * base_std

        # Ensure the axis includes all bars.
        ymin = min(ymin, min(vals))
        ymax = max(ymax, max(vals))

        # Moderate padding for readability.
        yrange = max(1e-9, ymax - ymin)
        pad = 0.10 * yrange
        ymin -= pad
        ymax += pad

        # Clamp to sensible lower bounds for bounded metrics.
        if metric_key.startswith(("ssim", "lpips")):
            ymin = max(0.0, ymin)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"bar_{metric_key}.png"), dpi=200)
        plt.close(fig)

    plot_metric("psnr_mean", "PSNR — Baseline vs LoRA vs Global-δ vs Multi-δ", "PSNR")
    plot_metric("ssim_mean", "SSIM — Baseline vs LoRA vs Global-δ vs Multi-δ", "SSIM")
    plot_metric("lpips_mean", "LPIPS — Baseline vs LoRA vs Global-δ vs Multi-δ", "LPIPS")

    # --- δ-B group sweep: Δ metrics vs groups (2,2)/(4,4)/(8,8) ---
    gs = d["group_sweep"]["points"]
    labels = [p["label"] for p in gs]

    def plot_group_delta(metric_key: str, title: str, is_lpips: bool = False) -> None:
        if is_lpips:
            deltas = [lpips_improvement(base["lpips_mean"], p["lpips_mean"]) for p in gs]
            ylab = "ΔLPIPS (baseline − method)"
        else:
            deltas = [p[metric_key] - base[metric_key] for p in gs]
            ylab = f"Δ{metric_key.split('_')[0].upper()}"
        # Narrower figure + tighter x-limits to reduce whitespace.
        fig, ax = plt.subplots(figsize=(4.9, 3.4))
        xs = [i * 0.72 for i in range(len(deltas))]
        # More distinguishable (but still harmonious) greens for (2,2)/(4,4)/(8,8).
        shades = ["#2E7D32", "#43A047", "#81C784"]
        bar_colors = [shades[i % len(shades)] for i in range(len(deltas))]
        ax.bar(xs, deltas, width=0.45, color=bar_colors)
        ax.set_xlim(xs[0] - 0.28, xs[-1] + 0.28)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"group_delta_{metric_key}.png"), dpi=200)
        plt.close(fig)

    plot_group_delta("psnr_mean", "δ-B group sweep (20 steps) — ΔPSNR vs baseline")
    plot_group_delta("ssim_mean", "δ-B group sweep (20 steps) — ΔSSIM vs baseline")
    plot_group_delta("lpips_mean", "δ-B group sweep (20 steps) — ΔLPIPS vs baseline", is_lpips=True)

    # --- Performance per parameter (log-x) ---
    def plot_params_eff(metric_key: str, title: str, is_lpips: bool = False) -> None:
        xs, ys, names = [], [], []
        for m in methods:
            if m.get("params") is None:
                continue
            xs.append(m["params"])
            if is_lpips:
                ys.append(lpips_improvement(base["lpips_mean"], m["lpips_mean"]))
                ylab = "ΔLPIPS (baseline − method)"
            else:
                ys.append(m[metric_key] - base[metric_key])
                ylab = f"Δ{metric_key.split('_')[0].upper()}"
            names.append(m["name"])

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.scatter(xs, ys)
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Trainable params (log)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"params_eff_{metric_key}.png"), dpi=200)
        plt.close(fig)

    plot_params_eff("psnr_mean", "Parameter efficiency — ΔPSNR vs trainable params")
    plot_params_eff("ssim_mean", "Parameter efficiency — ΔSSIM vs trainable params")
    plot_params_eff("lpips_mean", "Parameter efficiency — ΔLPIPS vs trainable params", is_lpips=True)

    # --- Compute vs quality (train time vs ΔLPIPS) ---
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for m in methods:
        if m.get("train_s") is None:
            continue
        y = lpips_improvement(base["lpips_mean"], m["lpips_mean"])
        ax.scatter([m["train_s"]], [y])
        ax.annotate(m["name"], (m["train_s"], y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Training time per video (s)")
    ax.set_ylabel("ΔLPIPS (baseline − method)")
    ax.set_title("Compute efficiency — training time vs LPIPS improvement", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "train_time_vs_delta_lpips.png"), dpi=200)
    plt.close(fig)

    print(f"Wrote plots to: {args.out_dir}")


if __name__ == "__main__":
    main()

