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


DEFAULT_HIDDEN_SIZE = 3072  # Open-Sora v2.0 256px config (vec-space dim)
DEFAULT_MLP_RATIO = 4.0
DEFAULT_DEPTH_DOUBLE = 19
DEFAULT_DEPTH_SINGLE = 38
DEFAULT_FUSED_QKV = False  # matches configs/diffusion/inference/256px.py in this repo


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


def _infer_hidden_size(config: Dict[str, Any]) -> int:
    """
    Infer vec-space dimension used by δ (hidden_size).

    Many run config.json files don't persist hidden_size; in that case we fall back to
    the Open-Sora v2.0 256px default (3072).
    """
    for key in ["hidden_size", "model.hidden_size", "cfg.model.hidden_size"]:
        v = _get(config, key)
        if isinstance(v, int):
            return v
    return DEFAULT_HIDDEN_SIZE


def trainable_params_delta_a(config: Dict[str, Any]) -> int:
    # Option A: one vec-space delta (size = hidden_size)
    return int(_infer_hidden_size(config))


def trainable_params_delta_b(config: Dict[str, Any]) -> Optional[int]:
    # Option B: grouped vec-space deltas + delta_final (also vec-space).
    # By implementation in run_delta_b.py:
    #   delta_double_groups: groups_double vectors (hidden_size each)
    #   delta_single_groups: groups_single vectors (hidden_size each)
    #   delta_final: 1 vector (hidden_size)
    hs = int(_infer_hidden_size(config))
    gd = _get(config, "groups_double") or _get(config, "args.groups_double") or _get(config, "extra.groups_double")
    gs = _get(config, "groups_single") or _get(config, "args.groups_single") or _get(config, "extra.groups_single")
    if not isinstance(gd, int) or not isinstance(gs, int):
        # try reading from config.json written by our scripts (flat keys)
        gd = config.get("groups_double")
        gs = config.get("groups_single")
    if not isinstance(gd, int) or not isinstance(gs, int):
        return None
    return hs * (int(gd) + int(gs) + 1)


def trainable_params_lora(run_dir: Path) -> Optional[int]:
    """
    Best-effort LoRA trainable parameter count without loading the base model:
    - If the run saved per-video LoRA weights (lora_weights/*.pt), load the first and sum numel.
    - Otherwise, return None (we can extend to 'load model and count' if desired).
    """
    lw_dir = run_dir / "lora_weights"
    # Actually these .pt are torch state dicts; load via torch if available.
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    if lw_dir.exists() and torch is not None:
        pts = sorted(lw_dir.glob("*.pt"))
        if pts:
            state = torch.load(str(pts[0]), map_location="cpu")
            if isinstance(state, dict):
                return int(sum(int(v.numel()) for v in state.values() if hasattr(v, "numel")))
    return None


def _load_mmengine_config_256px() -> Dict[str, Any]:
    """
    Best-effort load of the canonical 256px inference config to read model depth/hidden_size.
    If mmengine isn't available (e.g., local lint), fall back to constants.
    """
    cfg = {
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "mlp_ratio": DEFAULT_MLP_RATIO,
        "depth": DEFAULT_DEPTH_DOUBLE,
        "depth_single_blocks": DEFAULT_DEPTH_SINGLE,
        "fused_qkv": DEFAULT_FUSED_QKV,
    }
    try:
        from mmengine.config import Config  # type: ignore

        repo_root = Path(__file__).resolve().parents[2]
        cfg_path = repo_root / "configs" / "diffusion" / "inference" / "256px.py"
        if cfg_path.exists():
            c = Config.fromfile(str(cfg_path))
            model = getattr(c, "model", None)
            if isinstance(model, dict):
                for k in ["hidden_size", "mlp_ratio", "depth", "depth_single_blocks", "fused_qkv"]:
                    v = model.get(k)
                    if isinstance(v, (int, float, bool)):
                        cfg[k] = v
    except Exception:
        pass
    return cfg


def trainable_params_lora_analytic(run_config: Dict[str, Any]) -> Optional[int]:
    """
    Analytic LoRA parameter count matching *this repo's* LoRA injection implementation.

    This is used when lora_weights/*.pt are not present (older runs or saving disabled).
    It uses model architecture fields from configs/diffusion/inference/256px.py (best-effort).
    """
    r = _get(run_config, "lora.rank") or run_config.get("lora_rank") or run_config.get("rank")
    if not isinstance(r, int) or r <= 0:
        return None

    target_mlp = bool(_get(run_config, "lora.target_mlp") or run_config.get("target_mlp") or False)

    mcfg = _load_mmengine_config_256px()
    d = int(mcfg["hidden_size"])
    mlp_ratio = float(mcfg["mlp_ratio"])
    depth_double = int(mcfg["depth"])
    depth_single = int(mcfg["depth_single_blocks"])
    fused_qkv = bool(mcfg["fused_qkv"])

    mlp_hidden = int(d * mlp_ratio)

    # --- Double-stream blocks ---
    # LoRA injection in lora_experiment/lora_layers.py:
    # - inject_lora_into_self_attention(... target_modules=["qkv","proj"])
    # - For fused_qkv=False attention, current implementation only injects "proj" (q_proj/k_proj/v_proj are NOT targeted).
    # - Always two attentions per double block: img_attn and txt_attn.
    #
    # Attention params per attention:
    #   - fused_qkv=True: LoRAFusedQKV adds Q/K/V adapters: 3 * (r*d + d*r) = 6*r*d
    #                    + proj LoRALinear: 2*r*d
    #                    => 8*r*d
    #   - fused_qkv=False: proj only => 2*r*d
    if fused_qkv:
        attn_per = 8 * r * d
    else:
        attn_per = 2 * r * d
    double_attn_total = depth_double * (2 * attn_per)  # img + txt

    # Optional MLP LoRA in double blocks when --target-mlp is enabled:
    # - img_mlp: [Linear(d->mlp_hidden), GELU, Linear(mlp_hidden->d)]
    # - txt_mlp: same
    # Each Linear adds r*(in+out).
    double_mlp_total = 0
    if target_mlp:
        per_stream_mlp = r * (d + mlp_hidden) + r * (mlp_hidden + d)  # two linears
        double_mlp_total = depth_double * 2 * per_stream_mlp  # img + txt

    # --- Single-stream blocks ---
    # lora_layers.py wraps:
    # - fused_qkv=True: linear1 and linear2
    # - fused_qkv=False: q_proj, k_proj, v_mlp, linear2
    single_total = 0
    if fused_qkv:
        # linear1: d -> (3d + mlp_hidden)
        single_total += depth_single * (r * (d + (3 * d + mlp_hidden)))
        # linear2: (d + mlp_hidden) -> d
        single_total += depth_single * (r * ((d + mlp_hidden) + d))
    else:
        # q_proj: d -> d
        single_total += depth_single * (r * (d + d))
        # k_proj: d -> d
        single_total += depth_single * (r * (d + d))
        # v_mlp: d -> (d + mlp_hidden)
        single_total += depth_single * (r * (d + (d + mlp_hidden)))
        # linear2: (d + mlp_hidden) -> d
        single_total += depth_single * (r * ((d + mlp_hidden) + d))

    return int(double_attn_total + double_mlp_total + single_total)


def trainable_params_lora_by_loading_model(run_dir: Path) -> Optional[int]:
    """
    Exact LoRA trainable parameter count by loading the base model and injecting LoRA,
    then using lora_experiment/lora_layers.py counting.

    This is slower than reading lora_weights/*.pt, but works even if those weights were not saved.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "configs" / "diffusion" / "inference" / "256px.py"
    try:
        import sys

        import torch  # type: ignore
        from mmengine.config import Config  # type: ignore
        from opensora.utils.misc import to_torch_dtype
        from opensora.utils.sampling import prepare_models

        # LoRA helpers (repo local). Note: lora_experiment/ is not a package (no __init__.py),
        # so we import it the same way run_lora_tta.py does: add the directory to sys.path.
        sys.path.insert(0, str(repo_root / "lora_experiment"))
        import lora_layers  # type: ignore
    except Exception:
        return None

    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    exp_cfg = _load_json(config_path)
    if not isinstance(exp_cfg, dict):
        return None

    # Extract LoRA settings from our config.json format
    lora_rank = _get(exp_cfg, "lora.rank") or exp_cfg.get("lora_rank") or exp_cfg.get("rank")
    lora_alpha = _get(exp_cfg, "lora.alpha") or exp_cfg.get("lora_alpha") or exp_cfg.get("alpha")
    target_mlp = bool(_get(exp_cfg, "lora.target_mlp") or exp_cfg.get("target_mlp") or False)
    dtype_str = exp_cfg.get("dtype", "bf16")

    if not isinstance(lora_rank, int):
        return None
    if not isinstance(lora_alpha, int):
        # alpha can be float in some runs; still fine
        if isinstance(lora_alpha, (int, float)):
            lora_alpha = int(lora_alpha)
        else:
            lora_alpha = lora_rank * 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = to_torch_dtype(dtype_str)
    cfg = Config.fromfile(str(cfg_path))
    model, *_ = prepare_models(cfg, device, dtype, offload_model=False)

    lora_layers.inject_lora_into_mmdit(
        model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=0.0,
        target_modules=["qkv", "proj"],
        target_blocks="all",
        target_mlp=target_mlp,
    )
    counts = lora_layers.count_lora_parameters(model)
    # counts["lora"] is number of LoRA parameters
    return int(counts.get("lora")) if isinstance(counts, dict) and isinstance(counts.get("lora"), int) else None


def load_timing_summary_flexible(run_dir: Path) -> Dict[str, Any]:
    """
    Backward-compatible timing loader:
    - Prefer timing_summary.json (new format)
    - Else if timing_per_video.jsonl exists, compute summary using experiment_timing.summarize
    - Else return {} (caller can fall back to metrics_summary.json or results.json)
    """
    timing_path = run_dir / "timing_summary.json"
    if timing_path.exists():
        d = _load_json(timing_path)
        return d if isinstance(d, dict) else {}

    per_video = run_dir / "timing_per_video.jsonl"
    if per_video.exists():
        try:
            # Local import (repo root module)
            from experiment_timing import summarize  # type: ignore
        except Exception:
            return {}

        records = []
        with per_video.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        out = summarize(records)
        return out if isinstance(out, dict) else {}

    return {}


def load_train_gen_times_fallback(run_dir: Path) -> Dict[str, Optional[float]]:
    """
    Backward-compatible loader for older runs that only wrote metrics_summary.json or results.json.
    Returns mean seconds per video for training and inference, if available.
    """
    # 1) metrics_summary.json (our scripts often include avg_train_time / avg_gen_time)
    ms = run_dir / "metrics_summary.json"
    if ms.exists():
        d = _load_json(ms)
        if isinstance(d, dict):
            # Common keys across scripts
            for train_k, gen_k in [
                ("avg_train_time", "avg_gen_time"),
                ("train_time_mean", "gen_time_mean"),
            ]:
                tr = d.get(train_k)
                ge = d.get(gen_k)
                if isinstance(tr, (int, float)) or isinstance(ge, (int, float)):
                    return {
                        "train_mean_s": float(tr) if isinstance(tr, (int, float)) else None,
                        "infer_mean_s": float(ge) if isinstance(ge, (int, float)) else None,
                    }

    # 2) results.json (per-video entries with train_time/gen_time)
    rj = run_dir / "results.json"
    if rj.exists():
        d = _load_json(rj)
        if isinstance(d, list) and d:
            ok = [it for it in d if isinstance(it, dict) and it.get("success", False)]
            if ok:
                trains = [float(it["train_time"]) for it in ok if isinstance(it.get("train_time"), (int, float))]
                gens = [float(it["gen_time"]) for it in ok if isinstance(it.get("gen_time"), (int, float))]
                train_mean = sum(trains) / len(trains) if trains else None
                gen_mean = sum(gens) / len(gens) if gens else None
                return {"train_mean_s": train_mean, "infer_mean_s": gen_mean}

    return {"train_mean_s": None, "infer_mean_s": None}


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
    p.add_argument(
        "--compute-lora-by-loading-model",
        action="store_true",
        help="If LoRA params cannot be inferred from lora_weights/*.pt, load model+inject LoRA to count params (slower).",
    )
    args = p.parse_args()

    rows = []
    for rd in [Path(x) for x in args.runs]:
        config_path = rd / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json: {config_path}")

        config = _load_json(config_path)
        timing = load_timing_summary_flexible(rd)

        kind = classify_run(config if isinstance(config, dict) else {}, rd)
        train_s = _phase_mean_s(timing, "tta_train") if timing else None
        infer_s = _phase_mean_s(timing, "generate") if timing else None
        encode_s = _phase_mean_s(timing, "encode_video") if timing else None
        embed_s = _phase_mean_s(timing, "embed_text") if timing else None

        # Backward compatibility: if we still don't have train/gen from timing, try metrics_summary.json/results.json
        if train_s is None and infer_s is None:
            fb = load_train_gen_times_fallback(rd)
            train_s = fb.get("train_mean_s")
            infer_s = fb.get("infer_mean_s")

        trainable = None
        if kind == "delta_a":
            trainable = trainable_params_delta_a(config)
        elif kind == "delta_b":
            trainable = trainable_params_delta_b(config)
        elif kind == "lora":
            trainable = trainable_params_lora(rd)
            if trainable is None:
                # Analytic fallback when lora_weights/*.pt are not present.
                trainable = trainable_params_lora_analytic(config if isinstance(config, dict) else {})
            if trainable is None and args.compute_lora_by_loading_model:
                trainable = trainable_params_lora_by_loading_model(rd)

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

