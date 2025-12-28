# LoRA Test-Time Adaptation for Open-Sora v2.0

This directory contains the implementation of LoRA Test-Time Adaptation (TTA) for Open-Sora v2.0 video generation models.

## Overview

The goal is to improve video continuation quality by fine-tuning lightweight LoRA adapters on conditioning frames at test time, before generating the continuation. This approach:

- **Fine-tunes only on conditioning frames** (no ground truth needed during training)
- **Uses v2v_head mode** (33 conditioning frames → 32 generated frames)
- **Resets LoRA weights between videos** (each video gets its own adaptation)
- **Supports checkpointing** for resumable experiments

## Directory Structure

```
lora_experiment/
├── README.md                  # This file
├── configs/
│   └── lora_tta.py            # Configuration for TTA experiments
├── lora_layers.py             # LoRA layer implementations for MMDiT
├── scripts/
│   ├── preprocess_ucf101.py   # Preprocess UCF-101 for v2.0
│   ├── preprocess_ucf101.sbatch
│   ├── generate_baseline.py   # Generate baseline (no TTA)
│   ├── generate_baseline.sbatch
│   ├── run_lora_tta.py        # Main LoRA TTA script
│   ├── run_lora_tta.sbatch
│   ├── test_v2v_ucf101.sbatch # Test v2v_head with preprocessed videos
│   ├── evaluate.py            # Compute metrics
│   └── evaluate.sbatch
├── data/
│   └── ucf101_processed/      # Preprocessed dataset (created by preprocess script)
│       ├── <class>/
│       │   └── *.mp4
│       └── metadata.csv
└── results/
    ├── baseline/              # Baseline generation results
    ├── lora_r16_lr2e4_100steps/  # LoRA TTA results
    └── evaluation/            # Evaluation metrics
```

## Prerequisites

1. **Preprocessed UCF-101 dataset**: Run the preprocessing script first
2. **Open-Sora v2.0 environment**: Must have the opensora20 conda environment set up
3. **Model checkpoints**: Downloaded to `ckpts/` directory

## Quick Start

### 1. Preprocess UCF-101

```bash
# Submit preprocessing job
sbatch lora_experiment/scripts/preprocess_ucf101.sbatch
```

This creates 65-frame videos at 256px resolution suitable for v2v_head conditioning.

### 2. Verify v2v_head Works

```bash
# Test v2v_head inference with a few videos
sbatch lora_experiment/scripts/test_v2v_ucf101.sbatch
```

### 3. Generate Baselines

```bash
# Generate continuations without TTA
sbatch lora_experiment/scripts/generate_baseline.sbatch
```

### 4. Run LoRA TTA

```bash
# Run with default configuration (rank=16, lr=2e-4, steps=100)
sbatch lora_experiment/scripts/run_lora_tta.sbatch

# Or customize with environment variables:
LORA_RANK=8 LEARNING_RATE=1e-4 NUM_STEPS=50 \
    sbatch lora_experiment/scripts/run_lora_tta.sbatch
```

### 5. Evaluate Results

```bash
# Compare baseline vs LoRA TTA
sbatch lora_experiment/scripts/evaluate.sbatch
```

## Configuration Options

### LoRA Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `LORA_RANK` | 16 | 4, 8, 16, 32 | LoRA rank (lower = fewer parameters) |
| `LORA_ALPHA` | 32 | 8, 16, 32, 64 | LoRA alpha (typically 2x rank) |
| `TARGET_MLP` | false | true, false | Also adapt MLP layers |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 2e-4 | Learning rate for LoRA fine-tuning |
| `NUM_STEPS` | 100 | Number of fine-tuning steps per video |
| `WARMUP_STEPS` | 5 | Learning rate warmup steps |

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INFERENCE_STEPS` | 25 | Diffusion sampling steps |
| `GUIDANCE` | 7.5 | Text guidance scale |
| `GUIDANCE_IMG` | 3.0 | Image guidance scale |

## Hyperparameter Search

To run experiments with different configurations:

```bash
# Example: rank=8, lr=1e-4, steps=50
LORA_RANK=8 LEARNING_RATE=1e-4 NUM_STEPS=50 \
LORA_OUTPUT_DIR=/path/to/results/lora_r8_lr1e4_50steps \
    sbatch lora_experiment/scripts/run_lora_tta.sbatch

# Example: rank=16, lr=2e-4, steps=100, with MLP
LORA_RANK=16 LEARNING_RATE=2e-4 NUM_STEPS=100 TARGET_MLP=true \
LORA_OUTPUT_DIR=/path/to/results/lora_r16_lr2e4_100steps_mlp \
    sbatch lora_experiment/scripts/run_lora_tta.sbatch
```

## Metrics

The evaluation script computes:

- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)
- **Temporal Consistency**: Frame-to-frame stability

## Architecture

### MMDiT LoRA Targets

The LoRA layers are injected into:

1. **Double Stream Blocks** (depth=19):
   - `img_attn.qkv` / `img_attn.proj`
   - `txt_attn.qkv` / `txt_attn.proj`
   - Optionally: `img_mlp`, `txt_mlp`

2. **Single Stream Blocks** (depth=38):
   - `linear1` (fused QKV + MLP input)
   - `linear2` (output projection)

### Training Process

For each video:
1. Load conditioning frames (1-33)
2. Encode with VAE → latent space
3. Fine-tune LoRA on conditioning frames using flow matching loss
4. Generate continuation using adapted model
5. Reset LoRA weights for next video

## Troubleshooting

### Out of Memory

- Reduce `LORA_RANK` (e.g., 8 instead of 16)
- Disable `TARGET_MLP`
- Reduce `NUM_STEPS`

### Slow Training

- Reduce `NUM_STEPS` (try 50 or 20)
- Increase `LEARNING_RATE` (try 5e-4)

### Poor Quality

- Increase `NUM_STEPS` (try 200)
- Increase `LORA_RANK` (try 32)
- Try different `LEARNING_RATE` values

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Open-Sora v2.0](https://github.com/hpcaitech/Open-Sora)
- [SLOT: Self-supervised Learning on Target](https://arxiv.org/abs/2309.15254) (inspiration for TTA approach)
