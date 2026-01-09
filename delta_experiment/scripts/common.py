#!/usr/bin/env python3
"""
Common utilities for δ-TTA experiments.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def stratified_sample(df: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Stratified sampling: select n_samples proportionally from each class.

    If there are >= n_samples unique classes, this effectively selects ~1 per class.
    """
    rng = np.random.default_rng(seed)
    if "class" not in df.columns:
        return df.head(n_samples)

    classes = sorted(df["class"].unique().tolist())
    n_classes = len(classes)
    if n_classes == 0:
        return df.head(n_samples)

    base_per_class = max(1, n_samples // n_classes)
    remainder = n_samples - (base_per_class * n_classes)

    sampled_rows = []
    for i, cls in enumerate(classes):
        class_df = df[df["class"] == cls]
        n_for_class = base_per_class + (1 if i < remainder else 0)
        n_for_class = min(n_for_class, len(class_df))
        if n_for_class <= 0:
            continue
        sampled_idx = rng.choice(class_df.index.to_numpy(), size=n_for_class, replace=False)
        sampled_rows.append(df.loc[sampled_idx])

    if not sampled_rows:
        return df.head(n_samples)

    out = pd.concat(sampled_rows, axis=0)

    # Fill to exact n_samples if some classes were too small.
    if len(out) < n_samples:
        remaining = df.loc[~df.index.isin(out.index)]
        extra_needed = n_samples - len(out)
        if len(remaining) > 0:
            extra_idx = rng.choice(remaining.index.to_numpy(), size=min(extra_needed, len(remaining)), replace=False)
            out = pd.concat([out, df.loc[extra_idx]], axis=0)

    return out.head(n_samples).reset_index(drop=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


@dataclass
class VideoSelectionConfig:
    max_videos: int = 100
    stratified: bool = True
    seed: int = 42


def select_videos(metadata_csv: Path, cfg: VideoSelectionConfig) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if cfg.stratified and "class" in df.columns and cfg.max_videos is not None:
        df = stratified_sample(df, cfg.max_videos, seed=cfg.seed)
    elif cfg.max_videos is not None:
        df = df.head(cfg.max_videos)
    return df


def project_root() -> Path:
    # delta_experiment/scripts/common.py -> delta_experiment/scripts -> delta_experiment -> repo root
    return Path(__file__).resolve().parents[2]


def set_env_for_cluster(project_root: Path) -> None:
    # Keep in sync with sbatch scripts; safe no-op locally.
    os.environ.setdefault("PYTHONPATH", f"{project_root}:{os.environ.get('PYTHONPATH','')}")


def save_video(
    video_tensor: torch.Tensor,
    output_path: str,
    fps: int = 24,
    target_height: int | None = None,
    target_width: int | None = None,
):
    """
    Save video tensor to mp4 using higher quality settings to avoid blockiness.

    This is CPU-side encoding; VRAM usage is not materially affected.
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


def load_video_for_training(
    video_path: str,
    model_ae,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a video and return (latents, pixel_frames) for the first num_frames.
    """
    import av

    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
        if len(frames) >= num_frames:
            break
    container.close()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    # Pad by repeating last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = np.stack(frames[:num_frames], axis=0)  # [T, H, W, C]
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

    pixel_frames = frames * 2 - 1  # [-1, 1]
    pixel_frames = pixel_frames.unsqueeze(0).to(device, dtype)  # [1, C, T, H, W]

    with torch.no_grad():
        latents = model_ae.encode(pixel_frames)

    return latents, pixel_frames


