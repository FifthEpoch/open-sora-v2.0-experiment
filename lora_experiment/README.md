# LoRA Test-Time Adaptation (TTA) Experiment for Open-Sora v2.0

This directory contains the implementation of LoRA-based Test-Time Adaptation for video continuation using Open-Sora v2.0.

## Overview

**Goal:** Improve video continuation quality by fine-tuning LoRA adapters on conditioning frames at test time.

**Key Difference from v1.3:**
- v2.0 uses `v2v_head` mode which conditions on **33 frames** (vs 22 frames in v1.3)
- v2.0 uses a FLUX-based architecture (vs STDiT3 in v1.3)
- v2.0 produces higher quality outputs at 256px and 768px resolutions

## Experiment Setup

### Data Flow
```
Input Video (65+ frames)
    │
    ├─→ Conditioning Frames (1-33) ──→ LoRA Fine-tuning ──→ Adapted Model
    │                                        │
    └─→ Ground Truth (34-65) ←───────── Generate Continuation
                                              │
                                              ▼
                                    Compare & Evaluate
```

### Key Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Conditioning frames | 33 | First 33 frames (9 latent frames) |
| Total frames | 65 | 33 conditioning + 32 generated |
| Resolution | 256px | Can upgrade to 768px |
| VAE compression | 4:1 | Temporal compression ratio |
| LoRA rank | 8, 16, 32 | Hyperparameter to tune |
| Learning rate | 1e-4, 2e-4 | Hyperparameter to tune |
| Fine-tuning steps | 20, 50, 100 | Hyperparameter to tune |

## Directory Structure

```
lora_experiment/
├── README.md                    # This file
├── configs/
│   └── lora_tta.py             # LoRA TTA configuration
├── scripts/
│   ├── preprocess_ucf101.py    # Preprocess UCF-101 for v2.0
│   ├── preprocess_ucf101.sbatch
│   ├── run_lora_tta.py         # Main TTA training script
│   ├── run_lora_tta.sbatch     # SLURM job for TTA
│   ├── generate_baseline.py    # Generate baseline (no TTA)
│   └── evaluate.py             # Compute metrics
├── results/                    # Experiment outputs
│   └── [experiment_name]/
│       ├── videos/
│       ├── metrics/
│       └── logs/
└── data/                       # Symlink to preprocessed UCF-101
```

## Usage

### 1. Preprocess UCF-101 Dataset

```bash
# From cluster, uses UCF-101 from v1.3 repo
sbatch lora_experiment/scripts/preprocess_ucf101.sbatch
```

This will:
- Load videos from `/scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org`
- Resize to v2.0 compatible resolution (256px or 768px)
- Ensure videos have at least 65 frames
- Save to `lora_experiment/data/ucf101_processed/`

### 2. Run LoRA TTA Experiment

```bash
sbatch lora_experiment/scripts/run_lora_tta.sbatch
```

### 3. Generate Baseline (No TTA)

```bash
sbatch lora_experiment/scripts/generate_baseline.sbatch
```

### 4. Evaluate Results

```bash
python lora_experiment/scripts/evaluate.py \
    --baseline results/baseline \
    --tta results/rank16_lr2e4_100steps
```

## TTA Training Details

### What Gets Fine-tuned
- LoRA adapters on attention layers (Q, K, V, O projections)
- Optional: MLP layers

### Training Objective
- **Only conditioning frames are used** (frames 1-33)
- No ground truth frames in training
- Loss: Reconstruction/diffusion loss on conditioning frames

### Inference
1. Load base Open-Sora v2.0 model
2. Apply LoRA adapters (fine-tuned on this video's conditioning frames)
3. Run v2v_head inference with conditioning frames
4. Generate continuation (frames 34-65)
5. Compare with ground truth for evaluation

## Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Temporal Consistency**: Frame-to-frame coherence

