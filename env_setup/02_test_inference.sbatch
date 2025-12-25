#!/bin/bash
#SBATCH --job-name=test_opensora20_infer
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --output=env_setup/slurm_log/test_infer_%j.out
#SBATCH --error=env_setup/slurm_log/test_infer_%j.err

# ==============================================================================
# Test Open-Sora v2.0 Inference
# ==============================================================================
# This script tests that the environment is correctly set up by running a simple
# text-to-video inference at 256px resolution.
#
# Expected:
#   - GPU memory usage: ~52GB
#   - Inference time: ~60s
#   - Output: samples/video_256px/*.mp4
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "Open-Sora v2.0 Inference Test"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "=============================================================================="
echo ""

# ==============================================================================
# Configuration
# ==============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/Open-Sora-v2"
ENV_PATH="${SCRATCH_BASE}/conda-envs/opensora20"
OUTPUT_DIR="${PROJECT_ROOT}/test_samples"

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

cd "${PROJECT_ROOT}"

# ==============================================================================
# Verify Checkpoints
# ==============================================================================
echo "=============================================================================="
echo "Step 1: Verifying Checkpoints"
echo "=============================================================================="

REQUIRED_FILES=(
    "ckpts/Open_Sora_v2.safetensors"
    "ckpts/hunyuan_vae.safetensors"
    "ckpts/google/t5-v1_1-xxl/config.json"
    "ckpts/openai/clip-vit-large-patch14/config.json"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  [OK] $file"
    else
        echo "  [MISSING] $file"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "ERROR: Missing checkpoint files!"
    echo "Please run: sbatch env_setup/download_checkpoints/download_checkpoints.sbatch"
    exit 1
fi

echo ""
echo "All checkpoints verified!"
echo ""

# ==============================================================================
# Run Text-to-Video Inference Test
# ==============================================================================
echo "=============================================================================="
echo "Step 2: Running T2V Inference Test (256px)"
echo "=============================================================================="
echo "Prompt: 'A cat walking on grass'"
echo "Resolution: 256px"
echo "Frames: 33"
echo "Aspect ratio: 16:9"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Record start time
START_TIME=$(date +%s)

# Run inference with reduced frames for quick test
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/256px.py \
    --save-dir "${OUTPUT_DIR}" \
    --prompt "A cat walking on grass" \
    --sampling_option.num_frames 33 \
    --sampling_option.num_steps 25 \
    --seed 42

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Inference completed in ${ELAPSED} seconds"
echo ""

# ==============================================================================
# Check Output
# ==============================================================================
echo "=============================================================================="
echo "Step 3: Checking Output"
echo "=============================================================================="

# Find generated video
GENERATED_VIDEO=$(find "${OUTPUT_DIR}" -name "*.mp4" -type f -mmin -5 | head -1)

if [ -n "$GENERATED_VIDEO" ]; then
    echo "Generated video: ${GENERATED_VIDEO}"
    echo "File size: $(du -h "$GENERATED_VIDEO" | cut -f1)"
    echo ""
    echo "SUCCESS! Inference test passed!"
else
    echo "WARNING: No video file found in output directory"
    echo "Contents of ${OUTPUT_DIR}:"
    ls -la "${OUTPUT_DIR}/" || true
    find "${OUTPUT_DIR}" -type f || true
fi

# ==============================================================================
# GPU Memory Usage
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 4: GPU Memory Usage"
echo "=============================================================================="
nvidia-smi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Test Complete!"
echo "=============================================================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Inference time: ${ELAPSED} seconds"
echo ""
echo "If the test passed, your environment is ready for Open-Sora v2.0!"
echo ""
echo "Example inference commands:"
echo ""
echo "  # Text-to-video (256px)"
echo "  torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \\"
echo "      configs/diffusion/inference/256px.py \\"
echo "      --prompt 'your prompt here'"
echo ""
echo "  # Image-to-video (256px)"
echo "  torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \\"
echo "      configs/diffusion/inference/256px.py \\"
echo "      --cond_type i2v_head \\"
echo "      --ref path/to/image.png \\"
echo "      --prompt 'your prompt here'"
echo ""
echo "End time: $(date)"
echo "=============================================================================="

