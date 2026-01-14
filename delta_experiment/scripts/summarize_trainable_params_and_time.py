#!/usr/bin/env python3
"""
Summarize trainable parameter counts and timing (training vs inference) for:
- LoRA TTA runs
- δ Option A runs
- δ Option B runs

This script is designed to be run on the cluster *after* runs finish, because it
reads each run directory's:
- config.json
- timing_summary.json

It explicitly reports:
- TRAINING time from phase "tta_train"
- INFERENCE time from phase "generate"

It intentionally does NOT include:
- model load/unload time (not recorded in our PhaseTimer phases)
- encoding/saving overhead (reported separately if present)

Usage:
  python delta_experiment/scripts/summarize_trainable_params_and_time.py \
    --runs \
      lora_experiment/results/lora_r8_lr2e-4_20steps \
      delta_experiment/results/delta_a \
      delta_experiment/results/delta_b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _phase_mean_s(timing_summary: Dict[str, Any], phase: str) -> Optional[float]:
    return _get(timing_summary, f"phases_s.{phase}.mean_s")


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.3f}"


def _infer_hidden_size(config: Dict[str, Any]) -> Optional[int]:
    # Most of our runs use the 256px config with hidden_size=3072, but keep this flexible.
    # We store it in some configs; if not, return None.
    for key in ["hidden_size", "model.hidden_size", "cfg.model.hidden_size"]:
        v = _get(config, key)
        if isinstance(v, int):
            return v
    return None


def trainable_params_delta_a(config: Dict[str, Any]) -> Optional[int]:
    # Option A: one vec-space delta (size = hidden_size)
    hs = _infer_hidden_size(config)
    return int(hs) if hs is not None else None


def trainable_params_delta_b(config: Dict[str, Any]) -> Optional[int]:
    # Option B: grouped vec-space deltas + delta_final (also vec-space).
    # By implementation in run_delta_b.py:
    #   delta_double_groups: groups_double vectors (hidden_size each)
    #   delta_single_groups: groups_single vectors (hidden_size each)
    #   delta_final: 1 vector (hidden_size)
    hs = _infer_hidden_size(config)
    if hs is None:
        return None
    gd = _get(config, "groups_double") or _get(config, "args.groups_double") or _get(config, "extra.groups_double")
    gs = _get(config, "groups_single") or _get(config, "args.groups_single") or _get(config, "extra.groups_single")
    if not isinstance(gd, int) or not isinstance(gs, int):
        # try reading from config.json written by our scripts (flat keys)
        gd = config.get("groups_double")
        gs = config.get("groups_single")
    if not isinstance(gd, int) or not isinstance(gs, int):
        return None
    return int(hs) * (int(gd) + int(gs) + 1)


def trainable_params_lora(run_dir: Path) -> Optional[int]:
    """
    Best-effort LoRA trainable parameter count without loading the base model:
    - If the run saved per-video LoRA weights (lora_weights/*.pt), load the first and sum numel.
    - Otherwise, return None (we can extend to 'load model and count' if desired).
    """
    lw_dir = run_dir / "lora_weights"
    if lw_dir.exists():
        pts = sorted(lw_dir.glob("*.pt"))
        if pts:
            sd = _load_json  # type: ignore
    # Actually these .pt are torch state dicts; load via torch if available.
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    if lw_dir.exists() and torch is not None:
        pts = sorted(lw_dir.glob("*.pt"))
        if pts:
            state = torch.load(str(pts[0]), map_location="cpu")
            if isinstance(state, dict):
                return int(sum(int(v.numel()) for v in state.values() if hasattr(v, "numel")))
    return None


def classify_run(config: Dict[str, Any], run_dir: Path) -> str:
    # Our scripts write config.json differently between LoRA and delta; infer from file paths/keys.
    if (run_dir / "lora_weights").exists() or _get(config, "type") == "lora_tta":
        return "lora"
    # delta scripts typically store method in metrics_summary.json; if present use it.
    ms = run_dir / "metrics_summary.json"
    if ms.exists():
        d = _load_json(ms)
        method = d.get("method")
        if method == "delta_b_grouped_per_layer":
            return "delta_b"
        if method == "delta_a_global_vec":
            return "delta_a"
    # fallback heuristic from directory name
    name = run_dir.name.lower()
    if "delta_b" in name:
        return "delta_b"
    if "delta_a" in name:
        return "delta_a"
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="Run directories to summarize.")
    args = p.parse_args()

    rows = []
    for rd in [Path(x) for x in args.runs]:
        config_path = rd / "config.json"
        timing_path = rd / "timing_summary.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json: {config_path}")
        if not timing_path.exists():
            raise FileNotFoundError(f"Missing timing_summary.json: {timing_path}")

        config = _load_json(config_path)
        timing = _load_json(timing_path)

        kind = classify_run(config if isinstance(config, dict) else {}, rd)
        train_s = _phase_mean_s(timing, "tta_train")
        infer_s = _phase_mean_s(timing, "generate")
        encode_s = _phase_mean_s(timing, "encode_video")
        embed_s = _phase_mean_s(timing, "embed_text")

        trainable = None
        if kind == "delta_a":
            trainable = trainable_params_delta_a(config)
        elif kind == "delta_b":
            trainable = trainable_params_delta_b(config)
        elif kind == "lora":
            trainable = trainable_params_lora(rd)

        rows.append(
            {
                "run": str(rd),
                "kind": kind,
                "trainable_params": trainable,
                "train_mean_s": train_s,
                "infer_mean_s": infer_s,
                "encode_mean_s": encode_s,
                "embed_mean_s": embed_s,
            }
        )

    # Print table
    print("")
    print("Trainable params + timing (mean seconds per video)")
    print("-" * 120)
    print(f"{'kind':<10} {'trainable_params':>16} {'train_s':>10} {'infer_s':>10} {'encode_s':>10} {'embed_s':>10}  run")
    print("-" * 120)
    for r in rows:
        tp = r["trainable_params"]
        tp_s = f"{tp:,}" if isinstance(tp, int) else "NA"
        print(
            f"{r['kind']:<10} {tp_s:>16} {_fmt(r['train_mean_s']):>10} {_fmt(r['infer_mean_s']):>10} "
            f"{_fmt(r['encode_mean_s']):>10} {_fmt(r['embed_mean_s']):>10}  {r['run']}"
        )
    print("-" * 120)
    print("")
    print("Notes:")
    print("- train_s uses phase 'tta_train' (excludes model load/unload).")
    print("- infer_s uses phase 'generate'.")
    print("- encode_s/embed_s are shown separately; save_video is not shown here (can be added if you want).")
    print("- LoRA trainable_params is computed from the first saved lora_weights/*.pt if present; otherwise NA.")
    print("")


if __name__ == "__main__":
    main()

