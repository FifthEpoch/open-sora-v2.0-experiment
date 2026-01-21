#!/usr/bin/env python3
"""
Option C: δ as a correction on the denoiser output.

We keep the denoiser unchanged, but add a learned constant residual δ_out to every
token prediction of the denoiser (velocity/noise prediction in packed latent space).

This is intentionally minimal: δ_out has dimension equal to the model output channel
(typically 64), and is optimized per video using conditioning-only flow-matching MSE.
"""

from __future__ import annotations

import argparse
import gc
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

from delta_experiment.scripts.common import (
    VideoSelectionConfig,
    project_root,
    select_videos,
    ensure_dir,
    write_json,
    save_video,
    load_video_for_training,
    build_augmented_latent_variants,
    make_img_ids_from_time_ids,
)
from delta_experiment.scripts.delta_modules import apply_output_delta, make_delta_out

# Shared timing utilities (repo root)
from experiment_timing import PhaseTimer, TimingRecord, now_s, write_timing_files


def parse_speed_factors(raw: str) -> list[float]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [float(p) for p in parts if p]


def optimize_delta_out(
    model,
    delta_out: torch.nn.Parameter,
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
    latents_variants: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[list[float], float]:
    """
    Optimize δ_out for one video (conditioning-only).
    """
    model.train()
    optimizer = AdamW([delta_out], lr=lr, weight_decay=weight_decay)

    patch_size = 2
    sigma_min = 1e-5

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

    losses: list[float] = []
    t0 = time.time()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        if step < warmup_steps and warmup_steps > 0:
            warmup_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

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
        z_t = t_expand * latents_packed + (1 - (1 - sigma_min) * t_expand) * noise_packed

        v_target = (1 - sigma_min) * noise_packed - latents_packed

        v_pred_raw = model(
            img=z_t,
            img_ids=img_ids,
            txt=text_embeds["txt"],
            txt_ids=txt_ids,
            timesteps=t.to(dtype),
            y_vec=text_embeds["vec"],
            cond=cond_packed,
            guidance=guidance_vec,
        )
        v_pred = apply_output_delta(v_pred_raw, delta_out)

        loss = F.mse_loss(v_pred.float(), v_target.float())
        if delta_l2 > 0:
            loss = loss + delta_l2 * (delta_out.float().pow(2).mean())

        loss_value = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_out], max_grad_norm)
        optimizer.step()

        losses.append(loss_value)

        del loss, v_pred, v_pred_raw, z_t, v_target, noise, noise_packed, t, t_rev, t_expand
        if step % 10 == 0:
            torch.cuda.empty_cache()

    train_time = time.time() - t0
    model.eval()

    del txt_ids, guidance_vec
    variant_cache.clear()
    torch.cuda.empty_cache()

    return losses, train_time


def main():
    parser = argparse.ArgumentParser(description="δ-TTA Option C (output correction δ)")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-results-json", type=str, default=None,
                        help="Optional: path to results.json to reuse the exact same video list/order (recommended).")

    parser.add_argument("--delta-steps", type=int, default=20)
    parser.add_argument("--delta-lr", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--delta-l2", type=float, default=0.0)
    parser.add_argument("--aug-enabled", action="store_true")
    parser.add_argument("--aug-flip", action="store_true")
    parser.add_argument("--aug-rotate-deg", type=float, default=0.0)
    parser.add_argument("--aug-speed-factors", type=str, default="")

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

    ref = Path(args.reference_results_json) if args.reference_results_json else None
    df = select_videos(
        data_dir / "metadata.csv",
        VideoSelectionConfig(max_videos=args.max_videos, stratified=args.stratified, seed=args.seed),
        reference_results_json=ref,
    )

    cfg_path = root / "configs" / "diffusion" / "inference" / "256px.py"
    cfg = Config.fromfile(str(cfg_path))
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(cfg, device, dtype, offload_model=False)

    for p in model.parameters():
        p.requires_grad = False

    # Determine output dim (out_channels)
    out_dim = getattr(model, "out_channels", getattr(model, "in_channels", 64))

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

    results = []
    total_train = 0.0
    total_gen = 0.0
    timing_records: list[dict] = []

    for idx in tqdm(range(len(df)), desc="delta-C"):
        row = df.iloc[idx]
        video_path = row["path"]
        caption = row.get("caption", row.get("class", "a video"))
        video_name = Path(video_path).stem

        try:
            phases: dict[str, float] = {}
            pt = PhaseTimer(phases)
            total_start = now_s()

            delta_out = make_delta_out(out_dim, device=device, dtype=dtype)

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
                losses, train_time = optimize_delta_out(
                    model=model,
                    delta_out=delta_out,
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
                    latents_variants=latents_variants,
                )
            total_train += train_time

            # Patch model.forward so api_fn sees corrected outputs during denoising.
            # Restore reliably even if sampling throws.
            orig_forward = model.forward

            def patched_forward(**kwargs):
                v_raw = orig_forward(**kwargs)
                return apply_output_delta(v_raw, delta_out)

            with pt.phase("generate"):
                model.forward = patched_forward
                try:
                    with torch.inference_mode():
                        output = api_fn(
                            sampling_option,
                            cond_type="v2v_head",
                            text=[caption],
                            ref=[video_path],
                            seed=args.seed + idx,
                            channel=cfg.model.get("in_channels", 64),
                        )
                finally:
                    model.forward = orig_forward

            gen_time = phases.get("generate", 0.0)
            total_gen += gen_time

            output_path = videos_dir / f"{video_name}_deltaC.mp4"
            with pt.phase("save_video"):
                # Stitch original conditioning frames back into the output
                output[:, :, :33, :, :] = pixel_frames.to(output.device, output.dtype)

                save_video(output, str(output_path), fps=24, target_height=256, target_width=464)

            total_s = now_s() - total_start

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
            timing_records.append(
                TimingRecord(
                    idx=idx,
                    video_name=video_name,
                    success=True,
                    phases_s=phases,
                    total_s=total_s,
                    extra={
                        "caption": caption,
                        "delta_steps": args.delta_steps,
                        "delta_lr": args.delta_lr,
                        "delta_l2": args.delta_l2,
                    },
                ).to_dict()
            )

            del latents, txt_embed, vec_embed, output, losses, delta_out

        except Exception as e:
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

        gc.collect()
        torch.cuda.empty_cache()

    write_json(out_dir / "results.json", results)
    write_timing_files(out_dir, timing_records)
    write_json(out_dir / "metrics_summary.json", {
        "method": "delta_c_output_correction",
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


