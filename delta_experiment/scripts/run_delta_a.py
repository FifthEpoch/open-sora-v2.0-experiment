#!/usr/bin/env python3
"""
Option A: δ as an additive vector to the denoiser’s global conditioning embedding (vec).

This is the most SLOT-like adaptation: a single sample-specific vector δ is optimized
at test time using only the conditioning frames, and is discarded after each video.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from mmengine.config import Config
from colossalai.utils import set_seed

from opensora.utils.misc import to_torch_dtype
from opensora.utils.sampling import SamplingOption, prepare_api, prepare_models, sanitize_sampling_option, pack

from delta_experiment.scripts.common import VideoSelectionConfig, project_root, select_videos, ensure_dir, write_json, save_video, load_video_for_training
from delta_experiment.scripts.delta_modules import DeltaAWrapper, make_delta_global


def optimize_delta_global(
    model,
    delta_wrapper: DeltaAWrapper,
    delta: torch.nn.Parameter,
    latents: torch.Tensor,
    text_embeds: dict,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int,
    lr: float,
    warmup_steps: int,
    max_grad_norm: float,
    weight_decay: float,
    delta_l2: float,
) -> tuple[list[float], float]:
    """
    Optimize δ for one video using flow-matching MSE on conditioning latents.
    """
    from einops import repeat

    model.train()

    # Enable δ injection
    delta_wrapper.enable(delta)

    # Only δ is trainable
    optimizer = AdamW([delta], lr=lr, weight_decay=weight_decay)

    B, C, T, H, W = latents.shape
    patch_size = 2
    sigma_min = 1e-5

    # ids
    h_patches = H // patch_size
    w_patches = W // patch_size
    img_ids = torch.zeros(T, h_patches, w_patches, 3, device=device, dtype=dtype)
    img_ids[..., 0] = img_ids[..., 0] + torch.arange(T, device=device)[:, None, None]
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h_patches, device=device)[None, :, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w_patches, device=device)[None, None, :]
    img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=B)

    txt_ids = torch.zeros(B, text_embeds["txt"].shape[1], 3, device=device, dtype=dtype)
    guidance_vec = torch.ones(B, device=device, dtype=dtype)

    latents_packed = pack(latents, patch_size=patch_size)

    masks = torch.ones(B, 1, T, H, W, device=device, dtype=dtype)
    cond = torch.cat((masks, latents), dim=1)
    cond_packed = pack(cond, patch_size=patch_size)

    losses: list[float] = []
    t0 = time.time()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        if step < warmup_steps and warmup_steps > 0:
            warmup_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        t = torch.rand(B, device=device, dtype=dtype)
        noise = torch.randn_like(latents)
        noise_packed = pack(noise, patch_size=patch_size)

        t_rev = 1 - t
        t_expand = t_rev[:, None, None]
        z_t = t_expand * latents_packed + (1 - (1 - sigma_min) * t_expand) * noise_packed

        v_target = (1 - sigma_min) * noise_packed - latents_packed

        v_pred = model(
            img=z_t,
            img_ids=img_ids,
            txt=text_embeds["txt"],
            txt_ids=txt_ids,
            timesteps=t.to(dtype),
            y_vec=text_embeds["vec"],
            cond=cond_packed,
            guidance=guidance_vec,
        )

        loss = F.mse_loss(v_pred.float(), v_target.float())
        if delta_l2 > 0:
            loss = loss + delta_l2 * (delta.float().pow(2).mean())

        loss_value = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta], max_grad_norm)
        optimizer.step()

        losses.append(loss_value)

        del loss, v_pred, z_t, v_target, noise, noise_packed, t, t_rev, t_expand
        if step % 10 == 0:
            torch.cuda.empty_cache()

    train_time = time.time() - t0
    model.eval()

    # Disable δ injection (keeps model clean for next video)
    delta_wrapper.disable()

    del latents_packed, cond_packed, masks, cond, img_ids, txt_ids, guidance_vec
    torch.cuda.empty_cache()

    return losses, train_time


def main():
    parser = argparse.ArgumentParser(description="δ-TTA Option A (global δ vec)")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--delta-steps", type=int, default=20)
    parser.add_argument("--delta-lr", type=float, default=5e-3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--delta-l2", type=float, default=0.0)

    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--guidance-img", type=float, default=3.0)
    parser.add_argument("--dtype", type=str, default="bf16")

    args = parser.parse_args()

    root = project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = to_torch_dtype(args.dtype)
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    videos_dir = ensure_dir(out_dir / "videos")
    write_json(out_dir / "config.json", vars(args))

    df = select_videos(data_dir / "metadata.csv", VideoSelectionConfig(max_videos=args.max_videos, stratified=args.stratified, seed=args.seed))

    # Load models
    cfg_path = root / "configs" / "diffusion" / "inference" / "256px.py"
    cfg = Config.fromfile(str(cfg_path))
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(cfg, device, dtype, offload_model=False)

    # Freeze model params
    for p in model.parameters():
        p.requires_grad = False

    # Prepare inference API
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)
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

    delta_wrapper = DeltaAWrapper(model)
    hidden_size = getattr(model, "hidden_size", 3072)

    results = []
    total_train = 0.0
    total_gen = 0.0

    for idx in tqdm(range(len(df)), desc="delta-A"):
        row = df.iloc[idx]
        video_path = row["path"]
        caption = row.get("caption", row.get("class", "a video"))
        video_name = Path(video_path).stem

        try:
            # Create a fresh δ for this video
            delta = make_delta_global(hidden_size, device=device, dtype=dtype)

            latents, _ = load_video_for_training(video_path, model_ae, 33, device, dtype)
            with torch.no_grad():
                txt_embed = model_t5(caption)
                vec_embed = model_clip(caption)
            text_embeds = {"txt": txt_embed, "vec": vec_embed}

            losses, train_time = optimize_delta_global(
                model=model,
                delta_wrapper=delta_wrapper,
                delta=delta,
                latents=latents,
                text_embeds=text_embeds,
                device=device,
                dtype=dtype,
                num_steps=args.delta_steps,
                lr=args.delta_lr,
                warmup_steps=args.warmup_steps,
                max_grad_norm=args.max_grad_norm,
                weight_decay=args.weight_decay,
                delta_l2=args.delta_l2,
            )
            total_train += train_time

            # Apply δ during generation by enabling wrapper (no grad)
            delta_wrapper.enable(delta)
            gen_start = time.time()
            with torch.inference_mode():
                output = api_fn(
                    sampling_option,
                    cond_type="v2v_head",
                    text=[caption],
                    ref=[video_path],
                    seed=args.seed + idx,
                    channel=cfg.model.get("in_channels", 64),
                )
            gen_time = time.time() - gen_start
            total_gen += gen_time
            delta_wrapper.disable()

            output_path = videos_dir / f"{video_name}_deltaA.mp4"
            save_video(output, str(output_path), fps=24, target_height=256, target_width=464)

            results.append({
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
            })

            del latents, txt_embed, vec_embed, output, losses, delta

        except Exception as e:
            results.append({
                "idx": idx,
                "video_name": video_name,
                "input_path": video_path,
                "error": str(e),
                "success": False,
            })

        gc.collect()
        torch.cuda.empty_cache()

    write_json(out_dir / "results.json", results)
    write_json(out_dir / "metrics_summary.json", {
        "method": "delta_a_global_vec",
        "num_videos": len(df),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "delta_steps": args.delta_steps,
        "delta_lr": args.delta_lr,
        "total_train_time": total_train,
        "total_gen_time": total_gen,
    })


if __name__ == "__main__":
    main()


