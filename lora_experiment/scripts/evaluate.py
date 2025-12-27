#!/usr/bin/env python3
"""
Evaluate video continuation quality for Open-Sora v2.0 TTA experiments.

This script computes metrics comparing generated continuations with ground truth:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Usage:
    python evaluate.py --generated <path> --ground-truth <path> --output <path>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Try to import metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage not available, some metrics will be skipped")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, LPIPS metric will be skipped")


def load_video_frames(video_path: str) -> np.ndarray:
    """Load video and return frames as numpy array [T, H, W, C]."""
    import av
    
    container = av.open(video_path)
    frames = []
    
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
    
    container.close()
    
    return np.stack(frames, axis=0)


def compute_psnr(generated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute average PSNR across frames."""
    if not SKIMAGE_AVAILABLE:
        return float('nan')
    
    psnr_values = []
    for i in range(min(len(generated), len(ground_truth))):
        val = psnr(ground_truth[i], generated[i], data_range=255)
        psnr_values.append(val)
    
    return np.mean(psnr_values)


def compute_ssim(generated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute average SSIM across frames."""
    if not SKIMAGE_AVAILABLE:
        return float('nan')
    
    ssim_values = []
    for i in range(min(len(generated), len(ground_truth))):
        val = ssim(ground_truth[i], generated[i], multichannel=True, channel_axis=2, data_range=255)
        ssim_values.append(val)
    
    return np.mean(ssim_values)


def compute_lpips(generated: np.ndarray, ground_truth: np.ndarray, lpips_model) -> float:
    """Compute average LPIPS across frames."""
    if not LPIPS_AVAILABLE or lpips_model is None:
        return float('nan')
    
    device = next(lpips_model.parameters()).device
    
    lpips_values = []
    for i in range(min(len(generated), len(ground_truth))):
        # Convert to tensor and normalize to [-1, 1]
        gen_tensor = torch.from_numpy(generated[i]).permute(2, 0, 1).float() / 127.5 - 1
        gt_tensor = torch.from_numpy(ground_truth[i]).permute(2, 0, 1).float() / 127.5 - 1
        
        gen_tensor = gen_tensor.unsqueeze(0).to(device)
        gt_tensor = gt_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            val = lpips_model(gen_tensor, gt_tensor).item()
        
        lpips_values.append(val)
    
    return np.mean(lpips_values)


def evaluate_video(
    generated_path: str,
    ground_truth_path: str,
    lpips_model=None,
    conditioning_frames: int = 33,
) -> dict:
    """Evaluate a single video pair."""
    
    # Load videos
    generated = load_video_frames(generated_path)
    ground_truth = load_video_frames(ground_truth_path)
    
    # Only compare continuation frames (skip conditioning)
    gen_continuation = generated[conditioning_frames:]
    gt_continuation = ground_truth[conditioning_frames:]
    
    # Ensure same length
    min_len = min(len(gen_continuation), len(gt_continuation))
    gen_continuation = gen_continuation[:min_len]
    gt_continuation = gt_continuation[:min_len]
    
    # Compute metrics
    metrics = {
        "psnr": compute_psnr(gen_continuation, gt_continuation),
        "ssim": compute_ssim(gen_continuation, gt_continuation),
        "lpips": compute_lpips(gen_continuation, gt_continuation, lpips_model),
        "num_frames_compared": min_len,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate video continuation quality")
    
    parser.add_argument("--generated-dir", type=str, required=True,
                        help="Directory with generated videos")
    parser.add_argument("--ground-truth-dir", type=str, required=True,
                        help="Directory with ground truth videos")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file for metrics (JSON)")
    parser.add_argument("--conditioning-frames", type=int, default=33,
                        help="Number of conditioning frames to skip")
    parser.add_argument("--use-lpips", action="store_true",
                        help="Compute LPIPS metric (requires GPU)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Video Continuation Quality Evaluation")
    print("=" * 70)
    print(f"Generated videos: {args.generated_dir}")
    print(f"Ground truth videos: {args.ground_truth_dir}")
    print(f"Conditioning frames: {args.conditioning_frames}")
    print("=" * 70)
    
    # Setup LPIPS model
    lpips_model = None
    if args.use_lpips and LPIPS_AVAILABLE:
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            lpips_model = lpips_model.cuda()
    
    # Find video pairs
    gen_dir = Path(args.generated_dir)
    gt_dir = Path(args.ground_truth_dir)
    
    gen_videos = list(gen_dir.rglob("*.mp4"))
    print(f"Found {len(gen_videos)} generated videos")
    
    # Evaluate each video
    all_metrics = []
    
    for gen_path in tqdm(gen_videos, desc="Evaluating"):
        # Find corresponding ground truth
        rel_path = gen_path.relative_to(gen_dir)
        gt_path = gt_dir / rel_path
        
        if not gt_path.exists():
            print(f"Warning: Ground truth not found for {rel_path}")
            continue
        
        try:
            metrics = evaluate_video(
                str(gen_path),
                str(gt_path),
                lpips_model,
                args.conditioning_frames,
            )
            metrics["video"] = str(rel_path)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error evaluating {rel_path}: {e}")
            continue
    
    # Compute summary statistics
    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)
        
        summary = {
            "num_videos": len(all_metrics),
            "psnr": {
                "mean": df["psnr"].mean(),
                "std": df["psnr"].std(),
                "min": df["psnr"].min(),
                "max": df["psnr"].max(),
            },
            "ssim": {
                "mean": df["ssim"].mean(),
                "std": df["ssim"].std(),
                "min": df["ssim"].min(),
                "max": df["ssim"].max(),
            },
        }
        
        if args.use_lpips:
            summary["lpips"] = {
                "mean": df["lpips"].mean(),
                "std": df["lpips"].std(),
                "min": df["lpips"].min(),
                "max": df["lpips"].max(),
            }
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Also save per-video metrics
        per_video_path = output_path.with_suffix(".csv")
        df.to_csv(per_video_path, index=False)
        
        print()
        print("=" * 70)
        print("Evaluation Summary")
        print("=" * 70)
        print(f"Videos evaluated: {len(all_metrics)}")
        print(f"PSNR: {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f}")
        print(f"SSIM: {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
        if args.use_lpips:
            print(f"LPIPS: {summary['lpips']['mean']:.4f} ± {summary['lpips']['std']:.4f}")
        print()
        print(f"Results saved to: {output_path}")
        print(f"Per-video metrics: {per_video_path}")
        print("=" * 70)
    else:
        print("No videos were evaluated successfully.")


if __name__ == "__main__":
    main()

