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
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from opensora.registry import MODELS, build_module
from opensora.utils.config import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.datasets.utils import save_sample


def load_video_frames(video_path: str, num_frames: int = 65) -> torch.Tensor:
    """Load video and return frames as tensor [C, T, H, W]."""
    import av
    
    container = av.open(video_path)
    frames = []
    
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
        if len(frames) >= num_frames:
            break
    
    container.close()
    
    if len(frames) < num_frames:
        raise ValueError(f"Video has only {len(frames)} frames, need {num_frames}")
    
    # Convert to tensor [T, H, W, C] -> [C, T, H, W]
    frames = np.stack(frames[:num_frames], axis=0)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    
    # Normalize to [-1, 1]
    frames = frames * 2 - 1
    
    return frames


def extract_conditioning_frames(video_tensor: torch.Tensor, num_cond: int = 33) -> torch.Tensor:
    """Extract first num_cond frames for conditioning [C, T, H, W]."""
    return video_tensor[:, :num_cond, :, :]


def setup_lora_model(model, lora_config: dict):
    """Add LoRA adapters to model."""
    config = LoraConfig(
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        target_modules=lora_config.get("target_modules", ["to_q", "to_k", "to_v", "to_out"]),
        lora_dropout=lora_config.get("dropout", 0.0),
        bias="none",
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    return model


def finetune_on_conditioning(
    model,
    model_ae,
    conditioning_frames: torch.Tensor,
    text_embeds: dict,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Fine-tune LoRA adapters on conditioning frames only.
    
    Args:
        model: The main diffusion model with LoRA
        model_ae: The VAE for encoding
        conditioning_frames: [C, T, H, W] tensor of conditioning frames
        text_embeds: Pre-computed text embeddings
        config: Training config
        device: Device to use
        dtype: Data type
    
    Returns:
        Average loss over training
    """
    model.train()
    
    # Training config
    lr = config.get("learning_rate", 2e-4)
    num_steps = config.get("num_steps", 100)
    
    # Setup optimizer (only LoRA parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=config.get("weight_decay", 0.01),
        betas=config.get("betas", (0.9, 0.999)),
    )
    
    # Encode conditioning frames to latent
    with torch.no_grad():
        # Add batch dimension [1, C, T, H, W]
        x = conditioning_frames.unsqueeze(0).to(device, dtype)
        z = model_ae.encode(x)  # [1, C_latent, T_latent, H_latent, W_latent]
    
    total_loss = 0.0
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Sample timestep
        t = torch.rand(1, device=device, dtype=dtype)
        
        # Sample noise
        noise = torch.randn_like(z)
        
        # Create noisy latent (simple linear interpolation for flow matching)
        sigma_min = 1e-5
        z_t = (1 - (1 - sigma_min) * t) * noise + t * z
        
        # Target velocity
        v_target = (1 - sigma_min) * z - noise
        
        # Forward pass
        # Note: This is a simplified version; actual implementation may need
        # to handle the full model interface
        try:
            v_pred = model(
                z_t,
                timesteps=t.expand(z_t.shape[0]),
                **text_embeds,
            )
            
            # Compute loss
            loss = F.mse_loss(v_pred.float(), v_target.float())
        except Exception as e:
            print(f"  Warning: Forward pass failed at step {step}: {e}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, config.get("max_grad_norm", 1.0))
        
        optimizer.step()
        
        total_loss += loss.item()
    
    model.eval()
    
    return total_loss / num_steps if num_steps > 0 else 0.0


def generate_continuation(
    model,
    model_ae,
    model_t5,
    model_clip,
    conditioning_frames: torch.Tensor,
    text: str,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate video continuation using v2v_head mode.
    
    Returns:
        Generated video tensor [C, T, H, W]
    """
    from opensora.utils.sampling import prepare_api, SamplingOption
    from opensora.utils.inference import collect_references_batch, prepare_inference_condition
    
    model.eval()
    
    # This is a placeholder - the actual implementation needs to use
    # the full inference pipeline from opensora
    # For now, we'll use the inference API
    
    sampling_option = SamplingOption(
        resolution=config.get("resolution", "256px"),
        aspect_ratio=config.get("aspect_ratio", "16:9"),
        num_frames=config.get("num_frames", 65),
        num_steps=config.get("num_steps", 25),
        guidance=config.get("guidance", 7.5),
        guidance_img=config.get("guidance_img", 3.0),
        shift=True,
        temporal_reduction=4,
        is_causal_vae=True,
        text_osci=True,
        image_osci=True,
        scale_temporal_osci=True,
        method="i2v",
    )
    
    # Prepare API function
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, {})
    
    # Generate
    with torch.no_grad():
        output = api_fn(
            sampling_option,
            cond_type="v2v_head",
            text=[text],
            ref=[conditioning_frames],
            seed=config.get("seed", 42),
            channel=config.get("in_channels", 64),
        )
    
    return output


def run_tta_experiment(args):
    """Main TTA experiment loop."""
    
    print("=" * 70)
    print("LoRA Test-Time Adaptation for Open-Sora v2.0")
    print("=" * 70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = to_torch_dtype(args.dtype)
    
    # Load metadata
    metadata_path = Path(args.data_dir) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    print(f"Found {len(df)} videos in dataset")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # LoRA config
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "target_modules": ["to_q", "to_k", "to_v", "to_out"],
        "dropout": 0.0,
    }
    
    # Training config
    train_config = {
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    }
    
    # Inference config
    infer_config = {
        "resolution": "256px",
        "aspect_ratio": "16:9",
        "num_frames": 65,
        "num_steps": 25,
        "guidance": 7.5,
        "guidance_img": 3.0,
        "seed": args.seed,
        "in_channels": 64,
    }
    
    print(f"\nLoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Training config: lr={args.learning_rate}, steps={args.num_steps}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Note: Full model loading would happen here
    # For now, we provide the structure - actual implementation needs
    # integration with the opensora model loading code
    
    print("=" * 70)
    print("Note: This is a template script.")
    print("Full implementation requires integration with opensora's model loading.")
    print("=" * 70)
    print()
    print("To complete the implementation:")
    print("1. Load models using opensora's build_module")
    print("2. Add LoRA adapters using peft")
    print("3. For each video:")
    print("   a. Load conditioning frames (1-33)")
    print("   b. Fine-tune LoRA on conditioning frames")
    print("   c. Generate continuation using v2v_head")
    print("   d. Save output video")
    print("   e. Reset LoRA weights")
    print("4. Compute metrics on generated vs ground truth")
    
    # Save experiment config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "lora": lora_config,
            "training": train_config,
            "inference": infer_config,
            "args": vars(args),
        }, f, indent=2)
    
    print(f"\nConfig saved to {config_path}")


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
    
    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate for LoRA fine-tuning")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of fine-tuning steps per video")
    
    # Other arguments
    parser.add_argument("--dtype", type=str, default="bf16",
                        help="Data type (bf16 or fp16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum number of videos to process")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from beginning (ignore checkpoint)")
    
    args = parser.parse_args()
    
    run_tta_experiment(args)


if __name__ == "__main__":
    main()

