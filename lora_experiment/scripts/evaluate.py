#!/usr/bin/env python3
"""
Evaluate generated videos against ground truth for Open-Sora v2.0 TTA experiments.

This script computes various metrics comparing generated videos to ground truth:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Temporal consistency metrics

Usage:
    python evaluate.py \
        --baseline-dir lora_experiment/results/baseline \
        --lora-dir lora_experiment/results/lora_r16_lr2e4_100steps \
        --gt-dir lora_experiment/data/ucf101_processed \
        --output-dir lora_experiment/results/evaluation
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_video(video_path: str, max_frames: int = None) -> np.ndarray:
    """Load video as numpy array [T, H, W, C]."""
    import av
    
    container = av.open(video_path)
    frames = []
    
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
        if max_frames and len(frames) >= max_frames:
            break
    
    container.close()
    
    return np.stack(frames, axis=0)


def resize_video(video: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize video frames to target resolution.
    
    Args:
        video: Video array [T, H, W, C]
        target_height: Target height
        target_width: Target width
    
    Returns:
        Resized video array [T, target_height, target_width, C]
    """
    import cv2
    
    T, H, W, C = video.shape
    if H == target_height and W == target_width:
        return video
    
    resized_frames = []
    for t in range(T):
        frame = cv2.resize(video[t], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_frames.append(frame)
    
    return np.stack(resized_frames, axis=0)


def match_video_shapes(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """Resize videos to match shapes for comparison.
    
    Resizes the prediction to match the ground truth dimensions.
    
    Args:
        pred: Predicted video [T1, H1, W1, C]
        gt: Ground truth video [T2, H2, W2, C]
    
    Returns:
        Tuple of (resized_pred, gt) with matching spatial dimensions
    """
    # Match frame count
    min_frames = min(pred.shape[0], gt.shape[0])
    pred = pred[:min_frames]
    gt = gt[:min_frames]
    
    # Match spatial dimensions (resize pred to match gt)
    if pred.shape[1:3] != gt.shape[1:3]:
        pred = resize_video(pred, gt.shape[1], gt.shape[2])
    
    return pred, gt


def slice_eval_frames(
    video: np.ndarray,
    cond_frames: int,
    include_context: bool,
    eval_frame_count: int | None,
    eval_frame_stride: int,
    gt_start: int = 0,
) -> np.ndarray:
    """Slice video to evaluation window with optional stride."""
    if include_context:
        sliced = video
    else:
        sliced = video[gt_start:, :, :, :]
    if eval_frame_stride > 1:
        sliced = sliced[::eval_frame_stride]
    if eval_frame_count is not None:
        sliced = sliced[:eval_frame_count]
    return sliced


def collect_sample_paths(item: dict, best_of: int) -> list[str]:
    """Collect candidate sample paths for best-of evaluation."""
    paths: list[str] = []
    if not item:
        return paths
    if "output_paths" in item and isinstance(item["output_paths"], list):
        paths.extend([p for p in item["output_paths"] if isinstance(p, str)])
    if "output_path" in item and isinstance(item["output_path"], str):
        paths.append(item["output_path"])
    # Look for numbered sample variants if requested
    if best_of > 1 and paths:
        base_path = Path(paths[0])
        for idx in range(best_of):
            candidate = base_path.with_name(f"{base_path.stem}_sample{idx}.mp4")
            if candidate.exists():
                paths.append(str(candidate))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped[: max(1, best_of)]


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute PSNR between two videos."""
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute average SSIM between two videos."""
    from skimage.metrics import structural_similarity as ssim
    
    ssim_values = []
    T = min(pred.shape[0], gt.shape[0])
    
    for t in range(T):
        # Compute SSIM for each channel and average
        ssim_val = ssim(pred[t], gt[t], channel_axis=2, data_range=255)
        ssim_values.append(ssim_val)
    
    return float(np.mean(ssim_values))


def compute_lpips(pred: np.ndarray, gt: np.ndarray, lpips_model=None) -> float:
    """Compute average LPIPS between two videos."""
    if lpips_model is None:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex')
            lpips_model = lpips_model.cuda() if torch.cuda.is_available() else lpips_model
        except ImportError:
            print("LPIPS not available, skipping...")
            return None
    
    lpips_values = []
    T = min(pred.shape[0], gt.shape[0])
    device = next(lpips_model.parameters()).device
    
    for t in range(T):
        # Convert to tensor [1, C, H, W] in range [-1, 1]
        pred_t = torch.from_numpy(pred[t]).permute(2, 0, 1).float() / 127.5 - 1.0
        gt_t = torch.from_numpy(gt[t]).permute(2, 0, 1).float() / 127.5 - 1.0
        
        pred_t = pred_t.unsqueeze(0).to(device)
        gt_t = gt_t.unsqueeze(0).to(device)
        
        with torch.no_grad():
            lpips_val = lpips_model(pred_t, gt_t).item()
        lpips_values.append(lpips_val)
    
    return float(np.mean(lpips_values))


def compute_temporal_consistency(video: np.ndarray) -> float:
    """Compute temporal consistency as average frame-to-frame PSNR."""
    T = video.shape[0]
    if T < 2:
        return float('inf')
    
    psnr_values = []
    for t in range(1, T):
        psnr = compute_psnr(video[t], video[t-1])
        if psnr != float('inf'):
            psnr_values.append(psnr)
    
    return float(np.mean(psnr_values)) if psnr_values else float('inf')


def run_evaluation(args):
    """Run evaluation comparing baseline and LoRA results."""
    
    print("=" * 70)
    print("Evaluation for Open-Sora v2.0 TTA Experiments")
    print("=" * 70)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    gt_metadata = pd.read_csv(Path(args.gt_dir) / "metadata.csv")
    
    # Load baseline results if available
    baseline_results = None
    if args.baseline_dir:
        baseline_results_path = Path(args.baseline_dir) / "results.json"
        if baseline_results_path.exists():
            with open(baseline_results_path) as f:
                baseline_results = json.load(f)
            print(f"Loaded {len(baseline_results)} baseline results")
    
    # Load LoRA results
    lora_results = None
    if args.lora_dir:
        lora_results_path = Path(args.lora_dir) / "results.json"
        if lora_results_path.exists():
            with open(lora_results_path) as f:
                lora_results = json.load(f)
            print(f"Loaded {len(lora_results)} LoRA results")
    
    # Initialize LPIPS model
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            lpips_model = lpips_model.cuda()
    except ImportError:
        print("Warning: LPIPS not available")
    
    # Compute metrics
    all_metrics = []
    
    # Determine which videos to evaluate
    if lora_results:
        video_list = [r for r in lora_results if r.get("success", False)]
    elif baseline_results:
        video_list = [r for r in baseline_results if r.get("success", False)]
    else:
        # Fall back to metadata
        video_list = [{"video_name": Path(row['path']).stem, "input_path": row['path']} 
                      for _, row in gt_metadata.iterrows()]
    
    if args.max_videos:
        video_list = video_list[:args.max_videos]
    
    print(f"\nEvaluating {len(video_list)} videos...")
    
    # Evaluation params
    cond_frames = 2  # current experiments use 2 conditioning frames
    gt_start = args.gt_start
    gt_length = args.gt_frames
    eval_frame_count = args.eval_frame_count
    eval_frame_stride = max(1, args.eval_frame_stride)
    include_context = bool(args.include_context)
    best_of = max(1, args.best_of)
    
    for item in tqdm(video_list, desc="Evaluating"):
        video_name = item["video_name"]
        
        # Find GT video
        gt_row = gt_metadata[gt_metadata['path'].str.contains(video_name)]
        if len(gt_row) == 0:
            print(f"Warning: Ground truth not found for {video_name}")
            continue
        gt_path = gt_row.iloc[0]['path']
        
        metrics = {"video_name": video_name}
        
        try:
            # Load ground truth (generation frames: 34-65, i.e., indices 33-64)
            gt_video = load_video(gt_path)
            gt_gen_frames = gt_video[gt_start:gt_start + gt_length, :, :, :]
            
            # Evaluate baseline
            if baseline_results:
                baseline_item = next((r for r in baseline_results 
                                      if r.get("video_name") == video_name and r.get("success")), None)
                if baseline_item:
                    sample_paths = collect_sample_paths(baseline_item, best_of)
                    if len(sample_paths) == 1 and best_of > 1:
                        print(f"Warning: baseline best-of requested but only one sample for {video_name}")
                    best_psnr = None
                    best_ssim = None
                    best_lpips = None
                    best_temporal = None
                    for baseline_path in sample_paths:
                        if not os.path.exists(baseline_path):
                            continue
                        baseline_video = load_video(baseline_path)
                        baseline_clip = slice_eval_frames(
                            baseline_video,
                            cond_frames=cond_frames,
                            include_context=include_context,
                            eval_frame_count=eval_frame_count,
                            eval_frame_stride=eval_frame_stride,
                            gt_start=cond_frames,
                        )
                        gt_clip = slice_eval_frames(
                            gt_video,
                            cond_frames=cond_frames,
                            include_context=include_context,
                            eval_frame_count=eval_frame_count,
                            eval_frame_stride=eval_frame_stride,
                            gt_start=gt_start,
                        )

                        # Match shapes (resize baseline to GT resolution for fair comparison)
                        baseline_clip, gt_for_baseline = match_video_shapes(baseline_clip, gt_clip)

                        psnr = compute_psnr(baseline_clip, gt_for_baseline)
                        ssim = compute_ssim(baseline_clip, gt_for_baseline)
                        lpips = compute_lpips(baseline_clip, gt_for_baseline, lpips_model) if lpips_model else None
                        temporal = compute_temporal_consistency(baseline_clip)

                        best_psnr = psnr if best_psnr is None else max(best_psnr, psnr)
                        best_ssim = ssim if best_ssim is None else max(best_ssim, ssim)
                        if lpips is not None:
                            best_lpips = lpips if best_lpips is None else min(best_lpips, lpips)
                        best_temporal = temporal if best_temporal is None else max(best_temporal, temporal)

                    if best_psnr is not None:
                        metrics["baseline_psnr"] = best_psnr
                        metrics["baseline_ssim"] = best_ssim
                        if best_lpips is not None:
                            metrics["baseline_lpips"] = best_lpips
                        metrics["baseline_temporal"] = best_temporal
            
            # Evaluate LoRA
            if lora_results:
                lora_item = next((r for r in lora_results 
                                  if r.get("video_name") == video_name and r.get("success")), None)
                if lora_item:
                    sample_paths = collect_sample_paths(lora_item, best_of)
                    if len(sample_paths) == 1 and best_of > 1:
                        print(f"Warning: lora best-of requested but only one sample for {video_name}")
                    best_psnr = None
                    best_ssim = None
                    best_lpips = None
                    best_temporal = None
                    for lora_path in sample_paths:
                        if not os.path.exists(lora_path):
                            continue
                        lora_video = load_video(lora_path)
                        lora_clip = slice_eval_frames(
                            lora_video,
                            cond_frames=cond_frames,
                            include_context=include_context,
                            eval_frame_count=eval_frame_count,
                            eval_frame_stride=eval_frame_stride,
                            gt_start=cond_frames,
                        )
                        gt_clip = slice_eval_frames(
                            gt_video,
                            cond_frames=cond_frames,
                            include_context=include_context,
                            eval_frame_count=eval_frame_count,
                            eval_frame_stride=eval_frame_stride,
                            gt_start=gt_start,
                        )

                        # Match shapes (resize lora to GT resolution for fair comparison)
                        lora_clip, gt_for_lora = match_video_shapes(lora_clip, gt_clip)

                        psnr = compute_psnr(lora_clip, gt_for_lora)
                        ssim = compute_ssim(lora_clip, gt_for_lora)
                        lpips = compute_lpips(lora_clip, gt_for_lora, lpips_model) if lpips_model else None
                        temporal = compute_temporal_consistency(lora_clip)

                        best_psnr = psnr if best_psnr is None else max(best_psnr, psnr)
                        best_ssim = ssim if best_ssim is None else max(best_ssim, ssim)
                        if lpips is not None:
                            best_lpips = lpips if best_lpips is None else min(best_lpips, lpips)
                        best_temporal = temporal if best_temporal is None else max(best_temporal, temporal)

                    if best_psnr is not None:
                        metrics["lora_psnr"] = best_psnr
                        metrics["lora_ssim"] = best_ssim
                        if best_lpips is not None:
                            metrics["lora_lpips"] = best_lpips
                        metrics["lora_temporal"] = best_temporal

                        # Add training info
                        metrics["train_time"] = lora_item.get("train_time")
                        metrics["final_loss"] = lora_item.get("final_loss")
            
            # Compute improvement metrics
            if "baseline_psnr" in metrics and "lora_psnr" in metrics:
                metrics["psnr_improvement"] = metrics["lora_psnr"] - metrics["baseline_psnr"]
            if "baseline_ssim" in metrics and "lora_ssim" in metrics:
                metrics["ssim_improvement"] = metrics["lora_ssim"] - metrics["baseline_ssim"]
            if "baseline_lpips" in metrics and "lora_lpips" in metrics:
                # Lower LPIPS is better, so negative difference is improvement
                metrics["lpips_improvement"] = metrics["baseline_lpips"] - metrics["lora_lpips"]
            
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error evaluating {video_name}: {e}")
            continue
    
    # Save detailed results
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "detailed_metrics.csv", index=False)
    
    # Compute summary statistics
    summary = {}
    
    for col in metrics_df.columns:
        if col == "video_name":
            continue
        if metrics_df[col].dtype in [np.float64, np.int64]:
            valid = metrics_df[col].dropna()
            if len(valid) > 0:
                summary[f"{col}_mean"] = float(valid.mean())
                summary[f"{col}_std"] = float(valid.std())
                summary[f"{col}_median"] = float(valid.median())
    
    summary["num_videos_evaluated"] = len(all_metrics)
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Videos evaluated: {len(all_metrics)}")
    
    if "baseline_psnr_mean" in summary:
        print(f"\nBaseline metrics:")
        print(f"  PSNR: {summary.get('baseline_psnr_mean', 'N/A'):.2f} ± {summary.get('baseline_psnr_std', 0):.2f}")
        print(f"  SSIM: {summary.get('baseline_ssim_mean', 'N/A'):.4f} ± {summary.get('baseline_ssim_std', 0):.4f}")
        if "baseline_lpips_mean" in summary:
            print(f"  LPIPS: {summary.get('baseline_lpips_mean', 'N/A'):.4f} ± {summary.get('baseline_lpips_std', 0):.4f}")
    
    if "lora_psnr_mean" in summary:
        print(f"\nLoRA TTA metrics:")
        print(f"  PSNR: {summary.get('lora_psnr_mean', 'N/A'):.2f} ± {summary.get('lora_psnr_std', 0):.2f}")
        print(f"  SSIM: {summary.get('lora_ssim_mean', 'N/A'):.4f} ± {summary.get('lora_ssim_std', 0):.4f}")
        if "lora_lpips_mean" in summary:
            print(f"  LPIPS: {summary.get('lora_lpips_mean', 'N/A'):.4f} ± {summary.get('lora_lpips_std', 0):.4f}")
    
    if "psnr_improvement_mean" in summary:
        print(f"\nImprovement (LoRA - Baseline):")
        print(f"  PSNR: {summary.get('psnr_improvement_mean', 'N/A'):+.2f}")
        print(f"  SSIM: {summary.get('ssim_improvement_mean', 'N/A'):+.4f}")
        if "lpips_improvement_mean" in summary:
            print(f"  LPIPS: {summary.get('lpips_improvement_mean', 'N/A'):+.4f} (positive = better)")
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTA results")
    
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Directory with baseline results")
    parser.add_argument("--lora-dir", type=str, default=None,
                        help="Directory with LoRA TTA results")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory with ground truth videos and metadata.csv")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum number of videos to evaluate")
    parser.add_argument("--eval-frame-count", type=int, default=None,
                        help="Number of frames to evaluate after slicing/stride")
    parser.add_argument("--eval-frame-stride", type=int, default=1,
                        help="Stride when sampling frames for evaluation")
    parser.add_argument("--include-context", action="store_true",
                        help="Include conditioning frames in evaluation clip")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Best-of-N sampling for metrics when multiple samples exist")
    parser.add_argument("--gt-start", type=int, default=10,
                        help="Start index for GT evaluation frames")
    parser.add_argument("--gt-frames", type=int, default=16,
                        help="Number of GT frames for evaluation")
    
    args = parser.parse_args()
    
    if not args.baseline_dir and not args.lora_dir:
        raise ValueError("At least one of --baseline-dir or --lora-dir must be provided")
    
    run_evaluation(args)


if __name__ == "__main__":
    main()
