# Open-Sora v2.0 Checkpoint Download

This directory contains scripts to download the required model checkpoints for Open-Sora v2.0.

## Required Checkpoints

| Model | Source | Size | Description |
|-------|--------|------|-------------|
| **Open-Sora v2** | [hpcai-tech/Open-Sora-v2](https://huggingface.co/hpcai-tech/Open-Sora-v2) | ~22GB | 11B parameter MMDiT model |
| **Hunyuan VAE** | Included in Open-Sora-v2 | ~335MB | Video VAE for encoding/decoding |
| **T5-v1.1-XXL** | [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | ~22GB | Text encoder (11B params) |
| **CLIP ViT-L/14** | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | ~600MB | Text/image encoder |

**Total download size: ~45GB**

## Download Instructions

### Option 1: SLURM Batch Job (Recommended)

```bash
cd /scratch/wc3013/Open-Sora-v2
sbatch env_setup/download_checkpoints/download_checkpoints.sbatch
```

### Option 2: Manual Download

```bash
# Activate environment
conda activate /scratch/wc3013/conda-envs/opensora20

# Create checkpoint directory
mkdir -p ckpts
cd ckpts

# Download Open-Sora v2 (includes VAE)
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir .

# Download T5-XXL
mkdir -p google
huggingface-cli download google/t5-v1_1-xxl --local-dir google/t5-v1_1-xxl

# Download CLIP
mkdir -p openai
huggingface-cli download openai/clip-vit-large-patch14 --local-dir openai/clip-vit-large-patch14
```

## Expected Directory Structure

After download, the checkpoint directory should look like:

```
ckpts/
├── Open_Sora_v2.safetensors      # Main 11B model
├── hunyuan_vae.safetensors       # Video VAE
├── google/
│   └── t5-v1_1-xxl/              # T5 text encoder
│       ├── config.json
│       ├── model.safetensors
│       └── ...
└── openai/
    └── clip-vit-large-patch14/   # CLIP encoder
        ├── config.json
        ├── model.safetensors
        └── ...
```

## Optional: Flux Model for T2I2V Pipeline

For the text-to-image-to-video pipeline (T2I2V), you can optionally download the Flux model:

```bash
# Download Flux-dev (for T2I step)
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ckpts/flux1-dev

# Or use the pre-converted version
huggingface-cli download hpcai-tech/flux1-dev-fused-rope --local-dir ckpts/flux1-dev-fused-rope
```

## Troubleshooting

### Download Interrupted

If download is interrupted, simply re-run the script. `huggingface-cli` will resume from where it left off.

### Disk Space Issues

Ensure you have at least 60GB free in `/scratch/wc3013` before starting the download.

```bash
df -h /scratch/wc3013
```

### Authentication Required

Some models may require Hugging Face authentication:

```bash
huggingface-cli login
```


