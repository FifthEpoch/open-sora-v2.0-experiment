#!/usr/bin/env python3
"""
LoRA Test-Time Adaptation (TTA) for Open-Sora v2.0

This script performs test-time adaptation by fine-tuning LoRA adapters on
conditioning frames for each video, then generating continuations.

Key features:
- Fine-tunes ONLY on conditioning frames (no ground truth in training)
- Uses v2v_head mode (33 conditioning frames)
- Resets LoRA weights between videos
- Checkpoints progress for resumability

Usage:
    python run_lora_tta.py \
        --data-dir lora_experiment/data/ucf101_processed \
        --output-dir lora_experiment/results/lora_r16_lr2e4_100steps \
        --lora-rank 16 \
        --learning-rate 2e-4 \
        --num-steps 100
"""

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

from opensora.registry import MODELS, build_module
from opensora.utils.misc import to_torch_dtype
from opensora.utils.sampling import (
    SamplingOption,
    prepare_api,
    prepare_models,
    sanitize_sampling_option,
)

# Import LoRA utilities
sys.path.insert(0, str(PROJECT_ROOT / "lora_experiment"))
from lora_layers import (
    inject_lora_into_mmdit,
    get_lora_parameters,
    count_lora_parameters,
    save_lora_weights,
    reset_lora_weights,
)

# Reuse δ-experiment selection helper so we can exactly match a reference results.json video list/order.
from delta_experiment.scripts.common import (
    VideoSelectionConfig,
    select_videos,
    build_augmented_latent_variants,
    make_img_ids_from_time_ids,
)


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


def parse_speed_factors(raw: str) -> list[float]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [float(p) for p in parts if p]


def save_video(
    video_tensor: torch.Tensor,
    output_path: str,
    fps: int = 24,
    target_height: int | None = None,
    target_width: int | None = None,
):
    """Save video tensor to file with optional upscaling and higher encode quality."""
    import imageio
    import cv2
    
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
            resized = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            resized_frames.append(resized)
        video = resized_frames
    
    # Save with higher quality / bitrate to reduce blockiness
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
    Load and encode video for training.
    
    Args:
        video_path: Path to video file
        model_ae: VAE model for encoding
        num_frames: Number of conditioning frames to use
        device: Device to use
        dtype: Data type
        target_height: Optional target height for resizing
        target_width: Optional target width for resizing
    
    Returns:
        Tuple of (latents, pixel_frames) where latents are VAE-encoded
    """
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
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    # Convert to tensor [T, H, W, C] -> [C, T, H, W]
    frames = np.stack(frames[:num_frames], axis=0)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    
    # Normalize to [-1, 1]
    pixel_frames = frames * 2 - 1
    
    # Add batch dimension and encode with VAE
    pixel_frames = pixel_frames.unsqueeze(0).to(device, dtype)  # [1, C, T, H, W]
    
    with torch.no_grad():
        latents = model_ae.encode(pixel_frames)
    
    return latents, pixel_frames


def finetune_lora_on_conditioning(
    model,
    latents: torch.Tensor,
    text_embeds: dict,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    latents_variants: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[list, float]:
    """
    Fine-tune LoRA adapters on conditioning frames only.
    
    This is the core TTA step: we fine-tune on the observed conditioning
    frames to adapt the model to this specific video before generation.
    
    Args:
        model: The diffusion model with LoRA adapters
        latents: VAE-encoded conditioning frames [1, C, T, H, W]
        text_embeds: Pre-computed text embeddings dict
        config: Training configuration
        device: Device
        dtype: Data type
    
    Returns:
        Tuple of (loss_history, training_time)
    """
    from einops import rearrange, repeat
    from opensora.utils.sampling import pack
    
    model.train()
    
    # Training config
    lr = config.get("learning_rate", 2e-4)
    num_steps = config.get("num_steps", 100)
    warmup_steps = config.get("warmup_steps", 5)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    sigma_min = 1e-5
    patch_size = 2
    
    # Get LoRA parameters only
    lora_params = get_lora_parameters(model)
    if not lora_params:
        raise ValueError("No LoRA parameters found. Make sure LoRA was injected into the model.")
    
    # Create optimizer
    optimizer = AdamW(
        lora_params,
        lr=lr,
        betas=config.get("betas", (0.9, 0.999)),
        weight_decay=config.get("weight_decay", 0.01),
        eps=1e-8,
    )
    
    # Prepare latent variants (allows temporal augmentations)
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
        cond = torch.cat((masks, v_latents), dim=1)  # [B, C+1, T, H, W]
        cond_packed = pack(cond, patch_size=patch_size)

        variant_cache.append(
            {
                "latents": v_latents,
                "latents_packed": latents_packed,
                "cond_packed": cond_packed,
                "img_ids": img_ids,
            }
        )

    # Pre-compute txt_ids (all zeros for text positional embeddings)
    B = variant_cache[0]["latents"].shape[0]
    txt_ids = torch.zeros(B, text_embeds["txt"].shape[1], 3, device=device, dtype=dtype)

    # Pre-compute guidance vector (no guidance during training, set to 1.0)
    guidance_vec = torch.ones(B, device=device, dtype=dtype)
    
    losses = []
    train_start = time.time()
    
    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        # Learning rate warmup
        if step < warmup_steps:
            warmup_lr = lr * (step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Sample random timestep using flow matching: t ~ U(0, 1)
        t = torch.rand(B, device=device, dtype=dtype)
        
        variant = variant_cache[torch.randint(0, len(variant_cache), (1,), device=device).item()]
        v_latents = variant["latents"]
        latents_packed = variant["latents_packed"]
        cond_packed = variant["cond_packed"]
        img_ids = variant["img_ids"]

        # Sample noise in latent space
        noise = torch.randn_like(v_latents)
        noise_packed = pack(noise, patch_size=patch_size)
        
        # Flow matching interpolation (following Open-Sora v2.0 convention)
        t_rev = 1 - t
        t_expand = t_rev[:, None, None]
        z_t_packed = t_expand * latents_packed + (1 - (1 - sigma_min) * t_expand) * noise_packed
        
        # Target velocity: v = (1 - sigma_min) * noise - latents
        v_target = (1 - sigma_min) * noise_packed - latents_packed
        
        # Forward pass through model
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
        
        # Compute MSE loss in packed format
        loss = F.mse_loss(v_pred.float(), v_target.float())
        
        # Store loss value before backward (detached)
        loss_value = loss.item()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Store loss
        losses.append(loss_value)
        
        # Explicit memory cleanup every step
        del loss, v_pred, z_t_packed, v_target, noise, noise_packed, t, t_rev, t_expand
        
        # Periodic cache clearing
        if step % 10 == 0:
            torch.cuda.empty_cache()
    
    train_time = time.time() - train_start
    model.eval()
    
    # Final cleanup
    del txt_ids, guidance_vec
    variant_cache.clear()
    torch.cuda.empty_cache()
    
    return losses, train_time


def run_tta_experiment(args):
    """Main TTA experiment loop."""
    
    print("=" * 70)
    print("LoRA Test-Time Adaptation for Open-Sora v2.0")
    print("=" * 70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = to_torch_dtype(args.dtype)
    set_seed(args.seed)
    
    # Load/select videos (optionally reuse a reference results.json list for exact fairness)
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
    lora_weights_dir = output_dir / "lora_weights"
    lora_weights_dir.mkdir(exist_ok=True)
    
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
    
    # Training config
    train_config = {
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "betas": (0.9, 0.999),
    }
    
    # Save experiment config
    exp_config = {
        "type": "lora_tta",
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "target_mlp": args.target_mlp,
        },
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
    
    print(f"\nLoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Training: lr={args.learning_rate}, steps={args.num_steps}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load config and models
    print("Loading models...")
    model_load_start = time.time()
    
    cfg_path = PROJECT_ROOT / "configs" / "diffusion" / "inference" / "256px.py"
    cfg = Config.fromfile(str(cfg_path))
    
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=False
    )
    
    model_load_time = time.time() - model_load_start
    print(f"Models loaded in {model_load_time:.1f}s")
    
    # Inject LoRA into the model
    print(f"\nInjecting LoRA layers (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_modules = inject_lora_into_mmdit(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        target_modules=["qkv", "proj"],
        target_blocks="all",
        target_mlp=args.target_mlp,
    )
    
    # Count parameters
    param_counts = count_lora_parameters(model)
    print(f"Parameter counts:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  LoRA trainable: {param_counts['lora']:,}")
    print(f"  Trainable %: {param_counts['trainable_pct']:.4f}%")
    
    # Prepare API function for inference
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)
    
    # Sampling options for generation
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
    
    # Process videos
    print(f"\nProcessing {len(df) - start_idx} videos...")
    
    total_train_time = 0
    total_gen_time = 0
    success_count = 0
    fail_count = 0
    timing_records: list[dict] = []
    
    for idx in tqdm(range(start_idx, len(df)), desc="LoRA TTA"):
        row = df.iloc[idx]
        video_path = row['path']
        caption = row.get('caption', row.get('class', 'a video'))
        video_name = Path(video_path).stem
        
        try:
            phases: dict[str, float] = {}
            pt = PhaseTimer(phases)
            total_start = now_s()
            
            # Load and encode conditioning frames
            with pt.phase("encode_video"):
                latents, pixel_frames = load_video_for_training(
                    video_path, model_ae, 33, device, dtype,
                    target_height=256, target_width=464
                )
            
            # Encode text
            with pt.phase("embed_text"):
                with torch.no_grad():
                    txt_embed = model_t5(caption)
                    vec_embed = model_clip(caption)
            
            text_embeds = {
                "txt": txt_embed,
                "vec": vec_embed,
            }

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
            
            # Fine-tune LoRA on conditioning frames
            with pt.phase("tta_train"):
                # Reset LoRA weights before each video (explicitly pass device)
                reset_lora_weights(model, device=device)

                losses, train_time = finetune_lora_on_conditioning(
                    model=model,
                    latents=latents,
                    text_embeds=text_embeds,
                    config=train_config,
                    device=device,
                    dtype=dtype,
                    latents_variants=latents_variants,
                )
            total_train_time += train_time
            
            # Generate continuation with adapted model
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
            
            # Save output video, upscaling to conditioning resolution (256x464) with higher quality encoding
            output_path = videos_dir / f"{video_name}_lora.mp4"
            with pt.phase("save_video"):
                # Stitch original conditioning frames back into the output
                # output: [1, C, T, H, W], pixel_frames: [1, C, 33, H, W]
                # Ensure we match resolution if needed (pixel_frames is already resized)
                output[:, :, :33, :, :] = pixel_frames.to(output.device, output.dtype)

                save_video(
                    output,
                    str(output_path),
                    fps=24,
                    target_height=256,
                    target_width=464,
                )
            
            # Optionally save LoRA weights
            if args.save_lora_weights:
                with pt.phase("save_lora_weights"):
                    lora_path = lora_weights_dir / f"{video_name}_lora.pt"
                    save_lora_weights(model, str(lora_path))
            
            # Record result
            total_s = now_s() - total_start
            result = {
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
            results.append(result)
            timing_records.append(
                TimingRecord(
                    idx=idx,
                    video_name=video_name,
                    success=True,
                    phases_s=phases,
                    total_s=total_s,
                    extra={
                        "caption": caption,
                        "lora_rank": args.lora_rank,
                        "learning_rate": args.learning_rate,
                        "num_steps": args.num_steps,
                    },
                ).to_dict()
            )
            success_count += 1
            
            # Cleanup after successful processing
            del latents, text_embeds, txt_embed, vec_embed, output, losses
            
        except Exception as e:
            import traceback
            print(f"\nError processing {video_name}: {e}")
            traceback.print_exc()
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
        
        # Aggressive memory cleanup after each video
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA operations complete
    
    # Save final results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-video timing + run-level timing summary
    write_timing_files(output_dir, timing_records)
    
    # Compute metrics summary
    successful_results = [r for r in results if r.get("success", False)]
    if successful_results:
        avg_train_time = sum(r["train_time"] for r in successful_results) / len(successful_results)
        avg_gen_time = sum(r["gen_time"] for r in successful_results) / len(successful_results)
        avg_total_time = sum(r["total_time"] for r in successful_results) / len(successful_results)
        avg_final_loss = sum(r["final_loss"] for r in successful_results if r["final_loss"]) / len(successful_results)
        
        summary = {
            "num_videos": len(df),
            "successful": success_count,
            "failed": fail_count,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
            "avg_train_time": avg_train_time,
            "avg_gen_time": avg_gen_time,
            "avg_total_time": avg_total_time,
            "avg_final_loss": avg_final_loss,
            "total_train_time": total_train_time,
            "total_gen_time": total_gen_time,
        }
        
        with open(output_dir / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("LoRA TTA Complete!")
    print("=" * 70)
    print(f"Successful: {success_count}/{len(df) - start_idx}")
    print(f"Failed: {fail_count}/{len(df) - start_idx}")
    if success_count > 0:
        print(f"Average training time: {avg_train_time:.1f}s")
        print(f"Average generation time: {avg_gen_time:.1f}s")
        print(f"Average total time: {avg_total_time:.1f}s")
        print(f"Average final loss: {avg_final_loss:.4f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LoRA TTA for Open-Sora v2.0")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with preprocessed videos and metadata.csv")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    
    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--target-mlp", action="store_true",
                        help="Also apply LoRA to MLP layers")
    
    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate for LoRA fine-tuning")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of fine-tuning steps per video")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Number of warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    # Augmentation arguments (TTA only)
    parser.add_argument("--aug-enabled", action="store_true",
                        help="Enable data augmentation during TTA")
    parser.add_argument("--aug-flip", action="store_true",
                        help="Enable horizontal flip augmentation")
    parser.add_argument("--aug-rotate-deg", type=float, default=0.0,
                        help="Max rotation degrees (applies ±deg)")
    parser.add_argument("--aug-speed-factors", type=str, default="",
                        help="Comma-separated speed factors (e.g., 0.5,2.0)")
    
    # Inference arguments
    parser.add_argument("--inference-steps", type=int, default=25,
                        help="Number of diffusion steps for generation")
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
    parser.add_argument("--save-lora-weights", action="store_true",
                        help="Save LoRA weights for each video")
    parser.add_argument(
        "--reference-results-json",
        type=str,
        default=None,
        help="Optional: path to results.json to reuse the exact same video list/order (recommended).",
    )
    
    args = parser.parse_args()
    
    run_tta_experiment(args)


if __name__ == "__main__":
    main()
