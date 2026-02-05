#!/usr/bin/env python3
"""
Common utilities for δ-TTA experiments.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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


def load_video_name_list_from_results(results_json: Path, max_videos: int | None = None) -> list[str]:
    """
    Load a stable video_name list from a prior experiment's results.json.

    This is used to ensure *exactly the same* 100 videos across baselines and methods.
    """
    with open(results_json) as f:
        items = json.load(f)
    names = [it["video_name"] for it in items if it.get("success", False) and "video_name" in it]
    if max_videos is not None:
        names = names[:max_videos]
    return names


def select_videos(metadata_csv: Path, cfg: VideoSelectionConfig, reference_results_json: Path | None = None) -> pd.DataFrame:
    """
    Select videos either by (a) reproducing stratified sampling, or (b) following a reference results.json list.

    Using a reference results.json is the safest way to guarantee identical video sets across experiments.
    """
    df = pd.read_csv(metadata_csv)

    if reference_results_json is not None:
        names = load_video_name_list_from_results(reference_results_json, max_videos=cfg.max_videos)
        # Map by stem of path
        df = df.copy()
        df["video_name"] = df["path"].apply(lambda p: Path(p).stem)
        lookup = df.set_index("video_name", drop=False)
        rows = []
        for n in names:
            if n in lookup.index:
                rows.append(lookup.loc[n])
        if rows:
            out = pd.DataFrame(rows)
            # If lookup.loc returns a Series for unique index, normalize
            if isinstance(out.iloc[0], pd.Series):
                pass
            return out.reset_index(drop=True)

    # Fallback to local stratified/head selection
    if cfg.stratified and "class" in df.columns and cfg.max_videos is not None:
        df = stratified_sample(df, cfg.max_videos, seed=cfg.seed)
    elif cfg.max_videos is not None:
        df = df.head(cfg.max_videos)
    return df.reset_index(drop=True)


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
    target_height: int | None = None,
    target_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a video and return (latents, pixel_frames) for the first num_frames.
    If target_height/width are provided, pixels are resized before encoding.
    """
    import av
    import cv2

    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        if target_height is not None and target_width is not None:
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
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


def load_video_for_eval(
    video_path: str,
    num_frames: int,
    target_height: int | None = None,
    target_width: int | None = None,
) -> torch.Tensor:
    """
    Load a video and return pixel frames for evaluation (CPU tensor).
    """
    import av
    import cv2

    container = av.open(video_path)
    frames = []

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        if target_height is not None and target_width is not None:
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frames.append(img)
        if len(frames) >= num_frames:
            break

    container.close()

    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])

    frames = np.stack(frames[:num_frames], axis=0)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    pixel_frames = frames * 2 - 1
    return pixel_frames.unsqueeze(0)


def compute_psnr_tensor(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR between two tensors in [-1, 1] range."""
    pred_u8 = ((pred + 1) / 2).clamp(0, 1) * 255.0
    gt_u8 = ((gt + 1) / 2).clamp(0, 1) * 255.0
    mse = torch.mean((pred_u8 - gt_u8) ** 2)
    if mse.item() == 0:
        return float("inf")
    return float(20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse))


def make_img_ids_from_time_ids(
    time_ids: torch.Tensor,
    h_patches: int,
    w_patches: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build img_ids using explicit time indices (supports temporal augmentations).
    Returns shape [B, T*H*W, 3].
    """
    time_ids = time_ids.to(device=device, dtype=dtype)
    t_len = int(time_ids.shape[0])
    img_ids = torch.zeros(t_len, h_patches, w_patches, 3, device=device, dtype=dtype)
    img_ids[..., 0] = time_ids[:, None, None]
    img_ids[..., 1] = torch.arange(h_patches, device=device, dtype=dtype)[None, :, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device, dtype=dtype)[None, None, :]
    img_ids = img_ids.reshape(t_len * h_patches * w_patches, 3).unsqueeze(0)
    return img_ids.repeat(batch, 1, 1)


def _rotate_clip(pixel_frames: torch.Tensor, degrees: float) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    from torchvision.transforms.functional import InterpolationMode

    # pixel_frames: [1, C, T, H, W]
    clip = pixel_frames[0].permute(1, 0, 2, 3)  # [T, C, H, W]
    rotated = torch.stack(
        [
            TF.rotate(
                frame,
                degrees,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.0,
            )
            for frame in clip
        ],
        dim=0,
    )
    return rotated.permute(1, 0, 2, 3).unsqueeze(0)


def build_augmented_pixel_variants(
    pixel_frames: torch.Tensor,
    *,
    enable_flip: bool = False,
    rotate_deg: float = 0.0,
    speed_factors: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Build augmented pixel variants from a conditioning clip.

    Returns list of dicts with keys: pixel_frames, time_ids, name.
    """
    variants: list[dict[str, Any]] = []
    device = pixel_frames.device
    t_len = int(pixel_frames.shape[2])
    base_time = torch.arange(t_len, device=device, dtype=torch.long)

    variants.append({"pixel_frames": pixel_frames, "time_ids": base_time, "name": "orig"})

    if enable_flip:
        variants.append(
            {
                "pixel_frames": pixel_frames.flip(dims=[4]),
                "time_ids": base_time,
                "name": "flip_h",
            }
        )

    if rotate_deg and rotate_deg > 0:
        for deg in (-rotate_deg, rotate_deg):
            variants.append(
                {
                    "pixel_frames": _rotate_clip(pixel_frames, deg),
                    "time_ids": base_time,
                    "name": f"rotate_{deg:+.1f}",
                }
            )

    if speed_factors:
        for factor in speed_factors:
            if factor == 1.0:
                continue
            if factor > 1.0:
                stride = max(2, int(round(factor)))
                idx = torch.arange(0, t_len, step=stride, device=device)
                variants.append(
                    {
                        "pixel_frames": pixel_frames[:, :, idx, :, :],
                        "time_ids": idx,
                        "name": f"speed_{stride}x",
                    }
                )
            elif factor < 1.0:
                repeat = max(2, int(round(1.0 / factor)))
                idx = torch.arange(t_len, device=device).repeat_interleave(repeat)[:t_len]
                variants.append(
                    {
                        "pixel_frames": pixel_frames[:, :, idx, :, :],
                        "time_ids": idx,
                        "name": f"slow_{repeat}x",
                    }
                )

    return variants


def build_augmented_latent_variants(
    *,
    pixel_frames: torch.Tensor,
    base_latents: torch.Tensor,
    model_ae,
    enable_flip: bool = False,
    rotate_deg: float = 0.0,
    speed_factors: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Build augmented latent variants from a conditioning clip.

    Returns list of dicts with keys: latents, time_ids, name.
    """
    pixel_variants = build_augmented_pixel_variants(
        pixel_frames,
        enable_flip=enable_flip,
        rotate_deg=rotate_deg,
        speed_factors=speed_factors,
    )
    variants: list[dict[str, Any]] = []
    for item in pixel_variants:
        if item["name"] == "orig":
            latents = base_latents
        else:
            with torch.no_grad():
                latents = model_ae.encode(item["pixel_frames"])
        variants.append(
            {
                "latents": latents,
                "time_ids": item["time_ids"],
                "name": item["name"],
            }
        )
    return variants
