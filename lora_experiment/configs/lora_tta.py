# LoRA Test-Time Adaptation Configuration for Open-Sora v2.0
# ============================================================
# This config is used for fine-tuning LoRA adapters on conditioning frames
# at test time to improve video continuation quality.

# ============================================================
# Experiment Settings
# ============================================================
experiment_name = "lora_tta"
seed = 42
dtype = "bf16"

# ============================================================
# Data Settings
# ============================================================
# Conditioning frames: first 2 frames
# Total frames needed: 18 (2 conditioning + 16 to generate)
conditioning_frames = 2
total_frames = 18
fps = 24
resolution = "256px"  # or "768px"
aspect_ratio = "16:9"

# Dataset paths (will be set at runtime)
data_dir = "lora_experiment/data/ucf101_processed"
output_dir = "lora_experiment/results"

# ============================================================
# LoRA Settings
# ============================================================
lora = dict(
    # Layers to apply LoRA
    target_modules=["to_q", "to_k", "to_v", "to_out"],  # Attention projections
    
    # LoRA hyperparameters
    rank=16,           # LoRA rank (options: 4, 8, 16, 32)
    alpha=32,          # LoRA alpha (typically 2x rank)
    dropout=0.0,       # LoRA dropout
    
    # Optional: also adapt MLP layers
    include_mlp=False,
    mlp_modules=["mlp.fc1", "mlp.fc2"],
)

# ============================================================
# Training Settings (for TTA fine-tuning)
# ============================================================
training = dict(
    # Only use conditioning frames for fine-tuning (NO ground truth)
    use_conditioning_only=True,
    
    # Optimizer
    optimizer="adamw",
    learning_rate=2e-4,      # Options: 1e-4, 2e-4, 5e-4
    weight_decay=0.01,
    betas=(0.9, 0.999),
    
    # Training duration
    num_steps=100,           # Options: 20, 50, 100
    warmup_steps=5,
    
    # Gradient settings
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    
    # Memory optimization
    gradient_checkpointing=True,
)

# ============================================================
# Inference Settings
# ============================================================
inference = dict(
    cond_type="i2v_head2",   # Use first 2 frames as conditioning
    num_frames=18,           # Total output frames
    num_steps=25,            # Diffusion steps
    guidance=7.5,            # Text guidance scale
    guidance_img=3.0,        # Image guidance scale
    shift=True,
    
    # Oscillation guidance (from official config)
    text_osci=True,
    image_osci=True,
    scale_temporal_osci=True,
)

# ============================================================
# Model Paths (relative to project root)
# ============================================================
model = dict(
    type="flux",
    from_pretrained="./ckpts/Open_Sora_v2.safetensors",
    # Architecture params from 256px.py
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    cond_embed=True,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
)

ae = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)

t5 = dict(
    type="text_embedder",
    from_pretrained="./ckpts/google/t5-v1_1-xxl",
    max_length=512,
    shardformer=True,
)

clip = dict(
    type="text_embedder",
    from_pretrained="./ckpts/openai/clip-vit-large-patch14",
    max_length=77,
)

# ============================================================
# Evaluation Settings
# ============================================================
evaluation = dict(
    metrics=["psnr", "ssim", "lpips"],
    compare_with_baseline=True,
    save_videos=True,
    save_frames=False,
)

