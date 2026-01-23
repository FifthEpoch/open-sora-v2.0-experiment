#!/usr/bin/env python3
"""
Full-model Test-Time Adaptation (TTA) for Open-Sora v2.0.

This script fine-tunes ALL model parameters on conditioning frames per video,
then generates continuations. It resets weights per video to measure an
upper-bound for single-video TTA.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Shared timing utilities (repo root)
from experiment_timing import PhaseTimer, TimingRecord, now_s, write_timing_files

from colossalai.utils import set_seed
from mmengine.config import Config

from opensora.utils.misc import to_torch_dtype
from opensora.utils.sampling import (
    SamplingOption,
    prepare_api,
    prepare_models,
    sanitize_sampling_option,
)

# Reuse δ-experiment selection + augmentation helpers
from delta_experiment.scripts.common import (
    VideoSelectionConfig,
    select_videos,
    build_augmented_latent_variants,
    make_img_ids_from_time_ids,
)


def stratified_sample(df: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    classes = df["class"].unique()
    n_classes = len(classes)
    base_per_class = max(1, n_samples // n_classes)
    remainder = n_samples - (base_per_class * n_classes)
    sampled_dfs = []
    for i, cls in enumerate(sorted(classes)):
        class_df = df[df["class"] == cls]
        n_for_class = base_per_class + (1 if i < remainder else 0)
        n_for_class = min(n_for_class, len(class_df))
        if n_for_class > 0:
            sampled = class_df.sample(n=n_for_class, random_state=seed)
            sampled_dfs.append(sampled)
    result = pd.concat(sampled_dfs, ignore_index=True)
    if len(result) < n_samples:
        remaining = df[~df.index.isin(result.index)]
        extra_needed = n_samples - len(result)
        if len(remaining) >= extra_needed:
            extra = remaining.sample(n=extra_needed, random_state=seed)
            result = pd.concat([result, extra], ignore_index=True)
    return result.head(n_samples)


def parse_speed_factors(raw: str) -> list[float]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [float(p) for p in parts if p]


def resize_pixel_frames_to_output(pixel_frames: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    if pixel_frames.shape[-2:] == output.shape[-2:]:
        return pixel_frames
    b, c, t, h, w = pixel_frames.shape
    target_h, target_w = output.shape[-2], output.shape[-1]
    frames = pixel_frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    frames = F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)
    frames = frames.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)
    return frames


def save_video(
    video_tensor: torch.Tensor,
    output_path: str,
    fps: int = 24,
    target_height: int | None = None,
    target_width: int | None = None,
):
    import imageio
    import cv2

    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]
    video = video_tensor.permute(1, 2, 3, 0)
    video = ((video + 1) / 2).clamp(0, 1)
    video = (video * 255).to(torch.uint8).cpu().numpy()
    if target_height is not None and target_width is not None:
        resized_frames = []
        for frame in video:
            resized = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
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
    pixel_frames = pixel_frames.unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        latents = model_ae.encode(pixel_frames)
    return latents, pixel_frames


def finetune_full_on_conditioning(
    model,
    latents: torch.Tensor,
    text_embeds: dict,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    latents_variants: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[list, float]:
    from einops import repeat
    from opensora.utils.sampling import pack

    model.train()

    lr = config.get("learning_rate", 2e-4)
    num_steps = config.get("num_steps", 20)
    warmup_steps = config.get("warmup_steps", 5)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    sigma_min = 1e-5
    patch_size = 2

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=lr,
        betas=config.get("betas", (0.9, 0.999)),
        weight_decay=config.get("weight_decay", 0.01),
        eps=1e-8,
    )

    if latents_variants is None:
        base_time = torch.arange(latents.shape[2], device=device, dtype=torch.long)
        latents_variants = [{"latents": latents, "time_ids": base_time}]

    variant_cache: list[dict[str, torch.Tensor]] = []
    for variant in latents_variants:
        v_latents = variant["latents"]
        time_ids = variant["time_ids"]
        B, C, T, H, W = v_latents.shape
        h_patches = H // patch_size
        w_patches = W // patch_size
        img_ids = make_img_ids_from_time_ids(time_ids, h_patches, w_patches, B, device, dtype)
        latents_packed = pack(v_latents, patch_size=patch_size)
        masks = torch.ones(B, 1, T, H, W, device=device, dtype=dtype)
        cond = torch.cat((masks, v_latents), dim=1)
        cond_packed = pack(cond, patch_size=patch_size)
        variant_cache.append(
            {
                "latents": v_latents,
                "latents_packed": latents_packed,
                "cond_packed": cond_packed,
                "img_ids": img_ids,
            }
        )

    B = variant_cache[0]["latents"].shape[0]
    txt_ids = torch.zeros(B, text_embeds["txt"].shape[1], 3, device=device, dtype=dtype)
    guidance_vec = torch.ones(B, device=device, dtype=dtype)

    losses = []
    train_start = time.time()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        if step < warmup_steps:
            warmup_lr = lr * (step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        variant = variant_cache[torch.randint(0, len(variant_cache), (1,), device=device).item()]
        v_latents = variant["latents"]
        latents_packed = variant["latents_packed"]
        cond_packed = variant["cond_packed"]
        img_ids = variant["img_ids"]

        t = torch.rand(B, device=device, dtype=dtype)
        noise = torch.randn_like(v_latents)
        noise_packed = pack(noise, patch_size=patch_size)

        t_rev = 1 - t
        t_expand = t_rev[:, None, None]
        z_t_packed = t_expand * latents_packed + (1 - (1 - sigma_min) * t_expand) * noise_packed
        v_target = (1 - sigma_min) * noise_packed - latents_packed

        v_pred = model(
            img=z_t_packed,
            img_ids=img_ids,
            txt=text_embeds["txt"],
            txt_ids=txt_ids,
            timesteps=t.to(dtype),
            y_vec=text_embeds["vec"],
            cond=cond_packed,
            guidance=guidance_vec,
        )

        loss = F.mse_loss(v_pred.float(), v_target.float())
        loss_value = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        losses.append(loss_value)

        del loss, v_pred, z_t_packed, v_target, noise, noise_packed, t, t_rev, t_expand
        if step % 10 == 0:
            torch.cuda.empty_cache()

    train_time = time.time() - train_start
    model.eval()
    del txt_ids, guidance_vec
    variant_cache.clear()
    torch.cuda.empty_cache()

    return losses, train_time


def reset_full_model_weights(model, base_state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        model.load_state_dict(base_state, strict=True)


def run_full_tta(args):
    print("=" * 70)
    print("Full-Model Test-Time Adaptation for Open-Sora v2.0")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = to_torch_dtype(args.dtype)
    set_seed(args.seed)

    metadata_path = Path(args.data_dir) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    if args.reference_results_json:
        ref = Path(args.reference_results_json)
        cfg_sel = VideoSelectionConfig(
            max_videos=int(args.max_videos) if args.max_videos else 10**9,
            stratified=bool(args.stratified),
            seed=int(args.seed),
        )
        df = select_videos(metadata_path, cfg_sel, reference_results_json=ref)
        print(f"Selected {len(df)} videos from reference list: {ref}")
    else:
        df = pd.read_csv(metadata_path)
        print(f"Found {len(df)} videos in dataset")
        if args.max_videos:
            if args.stratified and "class" in df.columns:
                df = stratified_sample(df, args.max_videos, seed=args.seed)
                print(f"Stratified sample: {args.max_videos} videos from {df['class'].nunique()} classes")
            else:
                df = df.head(args.max_videos)
                print(f"Processing first {args.max_videos} videos")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    train_config = {
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "betas": (0.9, 0.999),
    }

    exp_config = {
        "type": "full_tta",
        "training": train_config,
        "inference": {
            "num_frames": 65,
            "conditioning_frames": 33,
            "num_steps": args.inference_steps,
            "guidance": args.guidance,
            "guidance_img": args.guidance_img,
        },
        "seed": args.seed,
        "max_videos": args.max_videos,
        "stratified": args.stratified,
        "reference_results_json": args.reference_results_json,
        "dtype": args.dtype,
        "augmentation": {
            "enabled": args.aug_enabled,
            "flip": args.aug_flip,
            "rotate_deg": args.aug_rotate_deg,
            "speed_factors": parse_speed_factors(args.aug_speed_factors),
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    print("Loading models...")
    cfg_path = PROJECT_ROOT / "configs" / "diffusion" / "inference" / "256px.py"
    cfg = Config.fromfile(str(cfg_path))
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=False
    )
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

    # Ensure all params are trainable
    for p in model.parameters():
        p.requires_grad = True

    # Save initial weights for per-video reset
    base_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    sampling_option = SamplingOption(
        resolution="256px",
        aspect_ratio="16:9",
        num_frames=65,
        num_steps=args.inference_steps,
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

    results = []
    timing_records: list[dict] = []
    total_train_time = 0.0
    total_gen_time = 0.0

    for idx in tqdm(range(len(df)), desc="Full TTA"):
        row = df.iloc[idx]
        video_path = row["path"]
        caption = row.get("caption", row.get("class", "a video"))
        video_name = Path(video_path).stem

        try:
            phases: dict[str, float] = {}
            pt = PhaseTimer(phases)
            total_start = now_s()

            with pt.phase("encode_video"):
                latents, pixel_frames = load_video_for_training(
                    video_path, model_ae, 33, device, dtype,
                    target_height=256, target_width=464
                )

            with pt.phase("embed_text"):
                with torch.no_grad():
                    txt_embed = model_t5(caption)
                    vec_embed = model_clip(caption)
            text_embeds = {"txt": txt_embed, "vec": vec_embed}

            latents_variants = None
            if args.aug_enabled:
                speed_factors = parse_speed_factors(args.aug_speed_factors)
                latents_variants = build_augmented_latent_variants(
                    pixel_frames=pixel_frames,
                    base_latents=latents,
                    model_ae=model_ae,
                    enable_flip=args.aug_flip,
                    rotate_deg=args.aug_rotate_deg,
                    speed_factors=speed_factors,
                )

            with pt.phase("tta_train"):
                reset_full_model_weights(model, base_state)
                losses, train_time = finetune_full_on_conditioning(
                    model=model,
                    latents=latents,
                    text_embeds=text_embeds,
                    config=train_config,
                    device=device,
                    dtype=dtype,
                    latents_variants=latents_variants,
                )
            total_train_time += train_time

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
            total_gen_time += gen_time

            output_path = videos_dir / f"{video_name}_full.mp4"
            with pt.phase("save_video"):
                stitched = resize_pixel_frames_to_output(pixel_frames, output)
                output = output.clone()
                output[:, :, :33, :, :] = stitched.to(output.device, output.dtype)
                save_video(output, str(output_path), fps=24, target_height=256, target_width=464)

            total_s = now_s() - total_start
            results.append(
                {
                    "idx": idx,
                    "video_name": video_name,
                    "input_path": video_path,
                    "output_path": str(output_path),
                    "caption": caption,
                    "train_time": train_time,
                    "gen_time": gen_time,
                    "total_time": train_time + gen_time,
                    "final_loss": losses[-1] if losses else None,
                    "avg_loss": sum(losses) / len(losses) if losses else None,
                    "success": True,
                }
            )
            timing_records.append(
                TimingRecord(
                    idx=idx,
                    video_name=video_name,
                    success=True,
                    phases_s=phases,
                    total_s=total_s,
                    extra={
                        "caption": caption,
                        "learning_rate": args.learning_rate,
                        "num_steps": args.num_steps,
                    },
                ).to_dict()
            )

            del latents, txt_embed, vec_embed, output, losses
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            results.append(
                {
                    "idx": idx,
                    "video_name": video_name,
                    "input_path": video_path,
                    "error": str(e),
                    "success": False,
                }
            )
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
            torch.cuda.empty_cache()

        # Save checkpoint after each video
        checkpoint = {"next_idx": idx + 1, "results": results}
        with open(output_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        write_timing_files(output_dir, timing_records)

    print("Done.")
    print(f"Avg train time: {total_train_time / max(1, len(df)):.2f}s")
    print(f"Avg gen time: {total_gen_time / max(1, len(df)):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Full-model TTA for Open-Sora v2.0")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--guidance-img", type=float, default=3.0)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--reference-results-json", type=str, default=None)
    parser.add_argument("--aug-enabled", action="store_true")
    parser.add_argument("--aug-flip", action="store_true")
    parser.add_argument("--aug-rotate-deg", type=float, default=0.0)
    parser.add_argument("--aug-speed-factors", type=str, default="")
    args = parser.parse_args()
    run_full_tta(args)


if __name__ == "__main__":
    main()
