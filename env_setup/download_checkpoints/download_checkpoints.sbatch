#!/bin/bash
#SBATCH --job-name=download_opensora20_ckpts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=env_setup/slurm_log/download_ckpts_%j.out
#SBATCH --error=env_setup/slurm_log/download_ckpts_%j.err

# ==============================================================================
# Download Open-Sora v2.0 Model Checkpoints
# ==============================================================================
# Downloads the following models from Hugging Face:
#   1. Open-Sora v2 (11B MMDiT model) - hpcai-tech/Open-Sora-v2
#   2. Hunyuan VAE - included in Open-Sora-v2
#   3. T5-XXL text encoder - google/t5-v1_1-xxl
#   4. CLIP ViT-L/14 - openai/clip-vit-large-patch14
#
# Total download size: ~50-60GB
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "Open-Sora v2.0 Checkpoint Download"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================================================="
echo ""

# ==============================================================================
# Configuration
# ==============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v2.0-experiment"
CKPT_DIR="${PROJECT_ROOT}/ckpts"
ENV_PATH="${SCRATCH_BASE}/conda-envs/opensora20"

# ==============================================================================
# Setup Environment
# ==============================================================================
echo "Setting up environment..."
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

# Load conda
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Activate environment
conda activate "${ENV_PATH}"

echo "Environment activated: ${CONDA_PREFIX}"
echo ""

# Create checkpoint directory
mkdir -p "${CKPT_DIR}"
cd "${PROJECT_ROOT}"

# ==============================================================================
# Download Open-Sora v2 (Main Model + VAE)
# ==============================================================================
echo "=============================================================================="
echo "Step 1: Downloading Open-Sora v2 (11B model + Hunyuan VAE)"
echo "=============================================================================="
echo "Source: https://huggingface.co/hpcai-tech/Open-Sora-v2"
echo "Destination: ${CKPT_DIR}"
echo ""

# Check if already downloaded
if [ -f "${CKPT_DIR}/Open_Sora_v2.safetensors" ] && [ -f "${CKPT_DIR}/hunyuan_vae.safetensors" ]; then
    echo "Open-Sora v2 checkpoints already exist, skipping download..."
    echo "  - ${CKPT_DIR}/Open_Sora_v2.safetensors"
    echo "  - ${CKPT_DIR}/hunyuan_vae.safetensors"
else
    echo "Downloading Open-Sora v2 from Hugging Face..."
    huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir "${CKPT_DIR}"
    echo "Open-Sora v2 download complete!"
fi
echo ""

# ==============================================================================
# Download T5-XXL Text Encoder
# ==============================================================================
echo "=============================================================================="
echo "Step 2: Downloading T5-v1.1-XXL Text Encoder"
echo "=============================================================================="
echo "Source: https://huggingface.co/google/t5-v1_1-xxl"
echo "Destination: ${CKPT_DIR}/google/t5-v1_1-xxl"
echo ""

T5_DIR="${CKPT_DIR}/google/t5-v1_1-xxl"
if [ -d "${T5_DIR}" ] && [ "$(ls -A ${T5_DIR} 2>/dev/null)" ]; then
    echo "T5-XXL already exists, skipping download..."
else
    echo "Downloading T5-v1.1-XXL..."
    mkdir -p "${CKPT_DIR}/google"
    huggingface-cli download google/t5-v1_1-xxl --local-dir "${T5_DIR}"
    echo "T5-XXL download complete!"
fi
echo ""

# ==============================================================================
# Download CLIP ViT-L/14
# ==============================================================================
echo "=============================================================================="
echo "Step 3: Downloading CLIP ViT-L/14"
echo "=============================================================================="
echo "Source: https://huggingface.co/openai/clip-vit-large-patch14"
echo "Destination: ${CKPT_DIR}/openai/clip-vit-large-patch14"
echo ""

CLIP_DIR="${CKPT_DIR}/openai/clip-vit-large-patch14"
if [ -d "${CLIP_DIR}" ] && [ "$(ls -A ${CLIP_DIR} 2>/dev/null)" ]; then
    echo "CLIP ViT-L/14 already exists, skipping download..."
else
    echo "Downloading CLIP ViT-L/14..."
    mkdir -p "${CKPT_DIR}/openai"
    huggingface-cli download openai/clip-vit-large-patch14 --local-dir "${CLIP_DIR}"
    echo "CLIP download complete!"
fi
echo ""

# ==============================================================================
# Verify Downloads
# ==============================================================================
echo "=============================================================================="
echo "Step 4: Verifying Downloads"
echo "=============================================================================="

echo "Checking required files..."

REQUIRED_FILES=(
    "${CKPT_DIR}/Open_Sora_v2.safetensors"
    "${CKPT_DIR}/hunyuan_vae.safetensors"
    "${CKPT_DIR}/google/t5-v1_1-xxl/config.json"
    "${CKPT_DIR}/openai/clip-vit-large-patch14/config.json"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "  [OK] $file ($SIZE)"
    else
        echo "  [MISSING] $file"
        ALL_PRESENT=false
    fi
done

echo ""

if [ "$ALL_PRESENT" = true ]; then
    echo "All checkpoint files verified!"
else
    echo "WARNING: Some files are missing. Please check the download logs."
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Download Complete!"
echo "=============================================================================="
echo ""
echo "Checkpoint directory: ${CKPT_DIR}"
echo ""
echo "Directory structure:"
ls -la "${CKPT_DIR}/"
echo ""
echo "Disk usage:"
du -sh "${CKPT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Test inference:"
echo "     sbatch env_setup/02_test_inference.sbatch"
echo ""
echo "End time: $(date)"
echo "=============================================================================="

