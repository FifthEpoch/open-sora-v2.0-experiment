# Environment Setup Guide for Open-Sora v2.0

This directory contains scripts to set up the environment for running Open-Sora v2.0 on an HPC cluster with SLURM.

## Quick Start

```bash
# 1. Clone/copy Open-Sora v2.0 to scratch
cd /scratch/wc3013
git clone https://github.com/hpcaitech/Open-Sora Open-Sora-v2
cd Open-Sora-v2

# 2. Create environment (submit as SLURM job)
sbatch env_setup/01_setup_environment.sbatch

# 3. Wait for completion, then download checkpoints
sbatch env_setup/download_checkpoints/download_checkpoints.sbatch

# 4. Test inference
sbatch env_setup/02_test_inference.sbatch
```

## Execution Order

Follow these scripts **in order**:

### 0. Scratch Environment Variables (Auto-loaded)
**File:** `00_set_scratch_env.sh`
- Redirects all installations, caches, and temp files to `/scratch/wc3013`
- Prevents filling up `/home` directory
- **Note:** Automatically sourced by all other scripts

### 1. Create Conda Environment
**File:** `01_setup_environment.sbatch`

Creates the `opensora20` conda environment with all required packages:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | Required |
| PyTorch | 2.4.0+cu121 | CUDA 12.1 |
| torchvision | 0.19.0 | |
| xformers | 0.0.27.post2 | Attention optimization |
| ColossalAI | >=0.4.4 | Distributed training |
| liger-kernel | 0.5.2 | Kernel optimizations |
| flash-attn | latest | Flash attention |
| PyAV | 13.x | Video I/O |

**Run:**
```bash
cd /scratch/wc3013/Open-Sora-v2
sbatch env_setup/01_setup_environment.sbatch

# Monitor progress
tail -f env_setup/slurm_log/setup_env_*.out
```

### 2. Download Model Checkpoints
**Directory:** `download_checkpoints/`

Downloads required models from Hugging Face:

| Model | Size | Description |
|-------|------|-------------|
| Open-Sora v2 | ~22GB | 11B MMDiT model |
| Hunyuan VAE | ~335MB | Video encoder/decoder |
| T5-v1.1-XXL | ~22GB | Text encoder |
| CLIP ViT-L/14 | ~600MB | Text/image encoder |

**Run:**
```bash
sbatch env_setup/download_checkpoints/download_checkpoints.sbatch

# Monitor progress
tail -f env_setup/slurm_log/download_ckpts_*.out
```

### 3. Test Inference
**File:** `02_test_inference.sbatch`

Runs a quick inference test to verify the setup:
- Generates a 33-frame video at 256px resolution
- Uses single H200 GPU
- Expected runtime: ~60 seconds

**Run:**
```bash
sbatch env_setup/02_test_inference.sbatch

# Check output
ls test_samples/
```

## Key Differences from Open-Sora v1.3

| Aspect | v1.3 | v2.0 |
|--------|------|------|
| Model | STDiT3-XL/2 (~1B) | MMDiT (~11B) |
| PyTorch | 2.2.2 | 2.4.0 |
| xformers | 0.0.25.post1 | 0.0.27.post2 |
| VAE | Open-Sora VAE | Hunyuan VAE |
| Architecture | STDiT | Flux-style MMDiT |
| Memory (256px) | ~30GB | ~52GB |
| New deps | bitsandbytes | liger-kernel |

## Memory Requirements

| Task | Resolution | Frames | GPU Memory | Min GPUs |
|------|------------|--------|------------|----------|
| Inference | 256px | 129 | ~52GB | 1 |
| Inference | 768px | 129 | ~60GB | 8 (with SP) |
| Training | 256px | 1 (image) | ~80GB | 1 |
| Training | 256px | 33 | ~120GB | 2 (recommended) |
| Training | 256px | 129 | ~140GB+ | 4+ |

## Video Continuation (I2V/V2V)

Open-Sora v2.0 supports video continuation with these modes:

```bash
# Image-to-video (first frame)
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/256px.py \
    --cond_type i2v_head \
    --ref path/to/image.png \
    --prompt "your prompt"

# Video extension (first half conditioning)
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/256px.py \
    --cond_type v2v_head_half \
    --ref path/to/video.mp4 \
    --prompt "your prompt"
```

## Troubleshooting

### Environment Creation Fails

Check the error log:
```bash
cat env_setup/slurm_log/setup_env_*.err
```

Common issues:
- Disk space: Ensure >100GB free in `/scratch/wc3013`
- Network: HuggingFace downloads may timeout, retry the job

### Checkpoint Download Fails

```bash
# Resume interrupted download
sbatch env_setup/download_checkpoints/download_checkpoints.sbatch
```

### Out of Memory During Inference

For 768px resolution, use multi-GPU with sequence parallelism:
```bash
torchrun --nproc_per_node 8 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/768px.py \
    --prompt "your prompt"
```

### flash-attn Build Fails

This is optional. If it fails, inference will still work but may be slower:
```bash
# Check if flash-attn is installed
python -c "import flash_attn; print(flash_attn.__version__)"

# If not, inference will use xformers instead (still fast)
```

## Directory Structure After Setup

```
/scratch/wc3013/
├── conda-envs/
│   └── opensora20/           # Conda environment
├── Open-Sora-v2/
│   ├── ckpts/                # Model checkpoints
│   │   ├── Open_Sora_v2.safetensors
│   │   ├── hunyuan_vae.safetensors
│   │   ├── google/t5-v1_1-xxl/
│   │   └── openai/clip-vit-large-patch14/
│   ├── env_setup/            # This directory
│   │   ├── slurm_log/        # SLURM job logs
│   │   └── download_checkpoints/
│   ├── configs/              # Configuration files
│   ├── scripts/              # Training/inference scripts
│   ├── opensora/             # Source code
│   └── test_samples/         # Test outputs
└── hf-cache/                 # HuggingFace cache
```

## Using the Environment in Other Scripts

Add this to your SLURM scripts:
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --gres=gpu:1
# ... other SLURM options ...

# Setup environment
source /scratch/wc3013/Open-Sora-v2/env_setup/00_set_scratch_env.sh
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/wc3013/conda-envs/opensora20

cd /scratch/wc3013/Open-Sora-v2

# Your commands here
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py ...
```

