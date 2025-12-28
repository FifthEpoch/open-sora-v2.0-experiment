#!/usr/bin/env python3
"""
Preprocess UCF-101 videos for Open-Sora v2.0 LoRA TTA experiments.

This script:
1. Reads videos from the UCF-101 dataset (from v1.3 repo or raw)
2. Resizes to v2.0 compatible resolution (256px or 768px)
3. Ensures videos have at least 65 frames (33 conditioning + 32 generation)
4. Saves preprocessed videos with metadata

Key differences from v1.3 preprocessing:
- Target frames: 65 (vs 49 in v1.3)
- Conditioning frames: 33 (vs 22 in v1.3)
- Resolution: 256px 16:9 or 768px 16:9
"""

import argparse
import os
from pathlib import Path
import av
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2


# Resolution presets for Open-Sora v2.0
RESOLUTION_PRESETS = {
    "256px": {
        "16:9": (256, 455),   # height x width (256 * 16/9 ≈ 455)
        "9:16": (455, 256),
        "1:1": (256, 256),
    },
    "768px": {
        "16:9": (768, 1365),  # height x width
        "9:16": (1365, 768),
        "1:1": (768, 768),
    },
}


def get_target_size(resolution="256px", aspect_ratio="16:9"):
    """Get target (height, width) for given resolution and aspect ratio."""
    if resolution not in RESOLUTION_PRESETS:
        raise ValueError(f"Unknown resolution: {resolution}")
    if aspect_ratio not in RESOLUTION_PRESETS[resolution]:
        raise ValueError(f"Unknown aspect ratio: {aspect_ratio}")
    return RESOLUTION_PRESETS[resolution][aspect_ratio]


def center_crop_resize(frame, target_height, target_width):
    """
    Center crop and resize frame to target dimensions.
    Maintains aspect ratio during crop, then resizes.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Frame must be numpy array, got {type(frame)}")
    
    if len(frame.shape) != 3:
        raise ValueError(f"Frame must have 3 dimensions (H, W, C), got shape {frame.shape}")
    
    # Ensure frame is contiguous and proper dtype
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    frame = np.copy(frame)
    h, w = frame.shape[:2]
    
    # Calculate aspect ratios
    target_aspect = target_width / target_height
    current_aspect = w / h
    
    # Crop to target aspect ratio
    if current_aspect > target_aspect:
        # Video is wider, crop width
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        frame = frame[:, start_x:start_x + new_w]
    elif current_aspect < target_aspect:
        # Video is taller, crop height
        new_h = int(w / target_aspect)
        start_y = (h - new_h) // 2
        frame = frame[start_y:start_y + new_h, :]
    
    frame = np.ascontiguousarray(frame)
    
    # Resize to target dimensions
    try:
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    except Exception:
        # Fallback to PIL if cv2 fails
        from PIL import Image
        img_pil = Image.fromarray(frame)
        img_pil = img_pil.resize((target_width, target_height), Image.LANCZOS)
        frame = np.array(img_pil)
    
    return frame


def resample_video(frames, original_fps, target_fps=24):
    """Resample video to target fps using linear interpolation."""
    if abs(original_fps - target_fps) < 0.1:
        return frames
    
    num_frames = len(frames)
    original_duration = num_frames / original_fps
    target_num_frames = int(original_duration * target_fps)
    
    if target_num_frames == 0:
        return frames
    
    # Create interpolation indices
    target_indices = np.linspace(0, num_frames - 1, target_num_frames)
    
    resampled_frames = []
    for idx in target_indices:
        lower_idx = int(np.floor(idx))
        upper_idx = min(int(np.ceil(idx)), num_frames - 1)
        alpha = idx - lower_idx
        
        if lower_idx == upper_idx:
            frame = frames[lower_idx]
        else:
            frame = ((1 - alpha) * frames[lower_idx] + alpha * frames[upper_idx]).astype(np.uint8)
        
        resampled_frames.append(frame)
    
    return resampled_frames


def uniform_sample_frames(frames, n):
    """Uniformly sample n frames from video."""
    if len(frames) < n:
        return None
    
    if len(frames) == n:
        return frames
    
    indices = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in indices]


def read_video(video_path):
    """Read video and return frames and fps."""
    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)
        
        frames = []
        for frame in container.decode(video=0):
            try:
                img = frame.to_ndarray(format='rgb24')
                if not isinstance(img, np.ndarray) or len(img.shape) != 3:
                    continue
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                frames.append(img)
            except Exception:
                continue
        
        container.close()
        
        if len(frames) == 0:
            return None, None
        
        return frames, fps
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None, None


def write_video(frames, output_path, fps=24):
    """Write frames to video file using imageio (more reliable than PyAV for encoding)."""
    import imageio
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert list to numpy array if needed
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)
    
    # Use imageio with ffmpeg backend for reliable encoding
    imageio.mimwrite(
        str(output_path), 
        frames, 
        fps=fps, 
        codec='libx264',
        output_params=['-crf', '18', '-pix_fmt', 'yuv420p']
    )


def parse_ucf101_filename(filename):
    """Parse UCF-101 filename to extract class name."""
    if not filename.startswith('v_'):
        return None
    
    name_without_prefix = filename[2:]
    name_without_ext = name_without_prefix.rsplit('.', 1)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) < 3:
        return None
    
    class_name = '_'.join(parts[:-2])
    return class_name


def process_video(video_path, output_base, target_fps=24, target_frames=65, 
                  target_height=256, target_width=455, verbose=False):
    """
    Process a single video for v2.0 TTA experiments.
    
    Returns: (success, output_path, num_frames, caption, skip_reason) or (False, None, None, None, reason)
    """
    try:
        frames, fps = read_video(video_path)
        if frames is None:
            return False, None, None, None, "read_failed"
        if len(frames) == 0:
            return False, None, None, None, "no_frames"
        
        original_frame_count = len(frames)
        if verbose:
            print(f"  Read {original_frame_count} frames at {fps:.1f} fps from {video_path.name}")
        
        # Center crop and resize each frame
        processed_frames = []
        for frame in frames:
            try:
                if not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
                    continue
                processed_frame = center_crop_resize(frame, target_height, target_width)
                processed_frames.append(processed_frame)
            except Exception:
                continue
        
        if len(processed_frames) == 0:
            return False, None, None, None, "processing_failed"
        
        # Resample to target fps
        resampled_frames = resample_video(processed_frames, fps, target_fps)
        
        if verbose:
            print(f"  After resampling to {target_fps}fps: {len(resampled_frames)} frames")
        
        # Check if we have enough frames
        if len(resampled_frames) < target_frames:
            return False, None, None, None, f"too_short:{len(resampled_frames)}"
        
        # Uniformly sample to target frames
        sampled_frames = uniform_sample_frames(resampled_frames, target_frames)
        if sampled_frames is None:
            return False, None, None, None, "sampling_failed"
        
        # Determine output path
        relative_path = video_path.relative_to(video_path.parents[1])
        output_path = output_base / relative_path.parent / f"{video_path.stem}.mp4"
        
        # Write video
        write_video(sampled_frames, output_path, target_fps)
        
        # Get caption from filename
        class_name = parse_ucf101_filename(video_path.name)
        if class_name is None:
            class_name = video_path.parent.name
        
        caption = ''.join([' ' + c.lower() if c.isupper() else c for c in class_name]).strip()
        caption = caption.replace('_', ' ')
        
        return True, output_path, target_frames, caption, None
    
    except Exception as e:
        print(f"  Error processing {video_path.name}: {e}")
        return False, None, None, None, f"exception:{e}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess UCF-101 for Open-Sora v2.0 TTA")
    parser.add_argument("--input-dir", type=str, 
                        default="/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org",
                        help="Input directory containing UCF-101 videos")
    parser.add_argument("--output-dir", type=str, 
                        default="/scratch/wc3013/open-sora-v2.0-experiment/lora_experiment/data/ucf101_processed",
                        help="Output directory for preprocessed videos")
    parser.add_argument("--fps", type=int, default=24,
                        help="Target frame rate")
    parser.add_argument("--frames", type=int, default=65,
                        help="Target number of frames (33 conditioning + 32 generation)")
    parser.add_argument("--resolution", type=str, default="256px",
                        choices=["256px", "768px"],
                        help="Target resolution")
    parser.add_argument("--aspect-ratio", type=str, default="16:9",
                        choices=["16:9", "9:16", "1:1"],
                        help="Target aspect ratio")
    parser.add_argument("--conditioning-frames", type=int, default=33,
                        help="Number of conditioning frames (for v2v_head mode)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum number of videos to process (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for all videos")
    
    args = parser.parse_args()
    
    # Get target dimensions
    target_height, target_width = get_target_size(args.resolution, args.aspect_ratio)
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        print("\nPlease ensure UCF-101 is downloaded in the v1.3 repo:")
        print("  /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org")
        return
    
    print("=" * 70)
    print("UCF-101 Preprocessing for Open-Sora v2.0 TTA")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target resolution: {args.resolution} ({target_width}×{target_height})")
    print(f"Target aspect ratio: {args.aspect_ratio}")
    print(f"Target FPS: {args.fps}")
    print(f"Target frames: {args.frames}")
    print(f"Conditioning frames: {args.conditioning_frames}")
    print(f"Generation frames: {args.frames - args.conditioning_frames}")
    print("=" * 70)
    
    # Find all videos
    video_files = list(input_dir.rglob("*.avi")) + list(input_dir.rglob("*.mp4"))
    print(f"\nFound {len(video_files)} videos to process")
    
    if len(video_files) == 0:
        print("❌ No videos found!")
        return
    
    if args.max_videos:
        video_files = video_files[:args.max_videos]
        print(f"Processing first {args.max_videos} videos only")
    
    # Enable verbose mode for first few videos to debug (or all if --verbose)
    verbose_count = len(video_files) if args.verbose else 5
    
    # Process videos
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    successful = 0
    skip_reasons = {}
    
    for idx, video_path in enumerate(tqdm(video_files, desc="Processing videos")):
        verbose = idx < verbose_count
        
        success, output_path, num_frames, caption, skip_reason = process_video(
            video_path, output_dir, args.fps, args.frames, 
            target_height, target_width, verbose=verbose
        )
        
        if success:
            metadata.append({
                'path': str(output_path),
                'relative_path': str(output_path.relative_to(output_dir)),
                'num_frames': num_frames,
                'conditioning_frames': args.conditioning_frames,
                'generation_frames': num_frames - args.conditioning_frames,
                'height': target_height,
                'width': target_width,
                'fps': args.fps,
                'resolution': args.resolution,
                'aspect_ratio': args.aspect_ratio,
                'text': caption,
                'class': video_path.parent.name,
            })
            successful += 1
        else:
            # Track skip reasons
            reason_key = skip_reason.split(':')[0] if skip_reason else "unknown"
            skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1
    
    # Save metadata
    csv_path = output_dir / "metadata.csv"
    df = pd.DataFrame(metadata)
    df.to_csv(csv_path, index=False)
    
    total_skipped = sum(skip_reasons.values())
    
    print("\n" + "=" * 70)
    print("✓ Preprocessing complete!")
    print("=" * 70)
    print(f"Successful: {successful} videos")
    print(f"Skipped: {total_skipped} videos")
    if skip_reasons:
        print("  Skip breakdown:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata CSV: {csv_path}")
    print()
    if successful > 0:
        print("Next steps:")
        print("  1. Run baseline generation: sbatch scripts/generate_baseline.sbatch")
        print("  2. Run LoRA TTA: sbatch scripts/run_lora_tta.sbatch")
        print()
        print("✓ Ready for TTA experiments!")
    else:
        print("⚠️  No videos were processed successfully!")
        print("Check the error messages above for details.")


if __name__ == "__main__":
    main()

