#!/usr/bin/env python3
"""
Generate baseline video continuations (without TTA) for Open-Sora v2.0.

This script generates video continuations using the vanilla Open-Sora v2.0 model
without any LoRA fine-tuning. These serve as the baseline for comparison.

Usage:
    python generate_baseline.py \
        --data-dir lora_experiment/data/ucf101_processed \
        --output-dir lora_experiment/results/baseline \
        --max-videos 100
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from colossalai.utils import set_seed
from opensora.registry import DATASETS, build_module
from opensora.utils.config import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.utils.sampling import (
    SamplingOption,
    prepare_api,
    prepare_models,
    sanitize_sampling_option,
)
from opensora.datasets.utils import save_sample

# Shared timing utilities (repo root)
from experiment_timing import PhaseTimer, TimingRecord, now_s, write_timing_files


def stratified_sample(df: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Stratified sampling: select n_samples proportionally from each class.
    
    Args:
        df: DataFrame with 'class' column
        n_samples: Total number of samples to select
        seed: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame with stratified selection
    """
    np.random.seed(seed)
    
    classes = df['class'].unique()
    n_classes = len(classes)
    
    # Calculate samples per class (at least 1 per class if possible)
    base_per_class = max(1, n_samples // n_classes)
    remainder = n_samples - (base_per_class * n_classes)
    
    sampled_dfs = []
    for i, cls in enumerate(sorted(classes)):
        class_df = df[df['class'] == cls]
        # Add 1 extra sample to first 'remainder' classes
        n_for_class = base_per_class + (1 if i < remainder else 0)
        n_for_class = min(n_for_class, len(class_df))  # Don't exceed available
        
        if n_for_class > 0:
            sampled = class_df.sample(n=n_for_class, random_state=seed)
            sampled_dfs.append(sampled)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we still need more samples (some classes had fewer than needed)
    if len(result) < n_samples:
        remaining = df[~df.index.isin(result.index)]
        extra_needed = n_samples - len(result)
        if len(remaining) >= extra_needed:
            extra = remaining.sample(n=extra_needed, random_state=seed)
            result = pd.concat([result, extra], ignore_index=True)
    
    return result.head(n_samples)  # Ensure exact count


def load_video_for_conditioning(
    video_path: str,
    num_frames: int = 33,
    target_height: int | None = None,
    target_width: int | None = None,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load video frames for v2v_head conditioning and return as a tensor [-1, 1]."""
    import av
    import cv2
    
    container = av.open(video_path)
    frames = []
    
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        if target_height is not None and target_width is not None:
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frames.append(img)
        if len(frames) >= num_frames:
            break
    
    container.close()
    
    if len(frames) < num_frames:
        # Pad with last frame if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    # [T, H, W, C] -> [C, T, H, W]
    frames = np.stack(frames[:num_frames], axis=0)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    pixel_frames = (frames * 2 - 1).unsqueeze(0).to(device, dtype)  # [1, C, T, H, W]
    return pixel_frames


def save_video(
    video_tensor: torch.Tensor,
    output_path: str,
    fps: int = 24,
    target_height: int | None = None,
    target_width: int | None = None,
):
    """
    Save video tensor to mp4 using higher quality settings to avoid blockiness.
    """
    import cv2
    import imageio

    # video_tensor: [B, C, T, H, W] or [C, T, H, W]
    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]

    # [C, T, H, W] -> [T, H, W, C]
    video = video_tensor.permute(1, 2, 3, 0)

    # Clamp and convert to uint8
    video = ((video + 1) / 2).clamp(0, 1)  # [-1, 1] -> [0, 1]
    video = (video * 255).to(torch.uint8).cpu().numpy()

    # Optional resize to match conditioning resolution (e.g., 256x464)
    if target_height is not None and target_width is not None:
        resized_frames = []
        for frame in video:
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)
        video = resized_frames

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(
        output_path,
        video,
        fps=fps,
        codec="libx264",
        quality=9,
        bitrate="4M",
        macro_block_size=None,
    )


def run_baseline_generation(args):
    """Main baseline generation loop."""
    
    print("=" * 70)
    print("Baseline Generation for Open-Sora v2.0")
    print("=" * 70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = to_torch_dtype(args.dtype)
    set_seed(args.seed)
    
    # Load metadata
    metadata_path = Path(args.data_dir) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    print(f"Found {len(df)} videos in dataset")
    
    if args.max_videos:
        if args.stratified and 'class' in df.columns:
            # Stratified sampling: select proportionally from each class
            df = stratified_sample(df, args.max_videos, seed=args.seed)
            print(f"Stratified sample: {args.max_videos} videos from {df['class'].nunique()} classes")
        else:
            df = df.head(args.max_videos)
            print(f"Processing first {args.max_videos} videos")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # Load checkpoint if resuming
    checkpoint_path = output_dir / "checkpoint.json"
    start_idx = 0
    results = []
    
    if checkpoint_path.exists() and not args.restart:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_idx = checkpoint.get("next_idx", 0)
        results = checkpoint.get("results", [])
        print(f"Resuming from video {start_idx}")
    
    # Save config
    config = {
        "type": "baseline",
        "num_frames": 65,
        "conditioning_frames": 33,
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "guidance_img": args.guidance_img,
        "seed": args.seed,
        "dtype": args.dtype,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Diffusion steps: {args.num_steps}")
    print(f"Guidance: {args.guidance}, Img guidance: {args.guidance_img}")
    print()
    
    # Build config for model loading
    # We need to load models similar to how inference.py does it
    from mmengine.config import Config
    
    # Load the 256px inference config
    cfg_path = PROJECT_ROOT / "configs" / "diffusion" / "inference" / "256px.py"
    cfg = Config.fromfile(str(cfg_path))
    
    # Build models
    print("Loading models...")
    model_load_start = time.time()
    
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=False
    )
    
    model_load_time = time.time() - model_load_start
    print(f"Models loaded in {model_load_time:.1f}s")
    
    # Prepare API function
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)
    
    # Sampling options
    sampling_option = SamplingOption(
        resolution="256px",
        aspect_ratio="16:9",
        num_frames=65,
        num_steps=args.num_steps,
        shift=True,
        temporal_reduction=4,
        is_causal_vae=True,
        guidance=args.guidance,
        guidance_img=args.guidance_img,
        text_osci=True,
        image_osci=True,
        scale_temporal_osci=True,
        method="i2v",
        seed=args.seed,
    )
    sampling_option = sanitize_sampling_option(sampling_option)
    
    # Process videos
    print(f"\nProcessing {len(df) - start_idx} videos...")
    
    total_time = 0
    success_count = 0
    fail_count = 0
    timing_records: list[dict] = []
    
    for idx in tqdm(range(start_idx, len(df)), desc="Generating baselines"):
        row = df.iloc[idx]
        video_path = row['path']
        caption = row.get('caption', row.get('class', 'a video'))
        video_name = Path(video_path).stem
        
        try:
            phases: dict[str, float] = {}
            pt = PhaseTimer(phases)
            total_start = now_s()
            
            # Load conditioning frames for stitching later
            pixel_frames = load_video_for_conditioning(
                video_path, num_frames=33, target_height=256, target_width=464,
                device=device, dtype=dtype
            )

            # Generate using the API function
            with pt.phase("generate"):
                with torch.inference_mode():
                    output = api_fn(
                        sampling_option,
                        cond_type="v2v_head",
                        text=[caption],
                        ref=[video_path],
                        seed=args.seed + idx,
                        channel=cfg.model.get("in_channels", 64),
                    )
            
            gen_time = phases.get("generate", 0.0)
            total_time += gen_time
            
            # Save output video
            output_path = videos_dir / f"{video_name}_baseline.mp4"
            with pt.phase("save_video"):
                # Stitch original conditioning frames back into the output
                output[:, :, :33, :, :] = pixel_frames.to(output.device, output.dtype)

                save_video(
                    output,
                    str(output_path),
                    fps=24,
                    target_height=256,
                    target_width=464,
                )

            total_s = now_s() - total_start
            
            # Record result
            results.append({
                "idx": idx,
                "video_name": video_name,
                "input_path": video_path,
                "output_path": str(output_path),
                "caption": caption,
                "generation_time": gen_time,
                "success": True,
            })

            timing_records.append(
                TimingRecord(
                    idx=idx,
                    video_name=video_name,
                    success=True,
                    phases_s=phases,
                    total_s=total_s,
                    extra={"caption": caption},
                ).to_dict()
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {video_name}: {e}")
            results.append({
                "idx": idx,
                "video_name": video_name,
                "input_path": video_path,
                "error": str(e),
                "success": False,
            })
            timing_records.append(
                TimingRecord(
                    idx=idx,
                    video_name=video_name,
                    success=False,
                    phases_s={},
                    total_s=0.0,
                    extra={"error": str(e)},
                ).to_dict()
            )
            fail_count += 1
        
        # Save checkpoint every 10 videos
        if (idx + 1) % 10 == 0:
            checkpoint = {
                "next_idx": idx + 1,
                "results": results,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # Save final results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-video timing + run-level timing summary
    write_timing_files(output_dir, timing_records)
    
    # Summary
    print("\n" + "=" * 70)
    print("Baseline Generation Complete!")
    print("=" * 70)
    print(f"Successful: {success_count}/{len(df) - start_idx}")
    print(f"Failed: {fail_count}/{len(df) - start_idx}")
    if success_count > 0:
        print(f"Average generation time: {total_time / success_count:.1f}s")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline video continuations")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with preprocessed videos and metadata.csv")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for baseline results")
    
    # Generation arguments
    parser.add_argument("--num-steps", type=int, default=25,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Text guidance scale")
    parser.add_argument("--guidance-img", type=float, default=3.0,
                        help="Image guidance scale")
    
    # Other arguments
    parser.add_argument("--dtype", type=str, default="bf16",
                        help="Data type (bf16 or fp16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum number of videos to process")
    parser.add_argument("--stratified", action="store_true",
                        help="Use stratified sampling across classes (requires 'class' column in metadata)")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from beginning (ignore checkpoint)")
    
    args = parser.parse_args()
    
    run_baseline_generation(args)


if __name__ == "__main__":
    main()
