#!/bin/bash
#SBATCH --job-name=test_v2v_opensora20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm_log/test_v2v_%j.out
#SBATCH --error=slurm_log/test_v2v_%j.err

# ==============================================================================
# Test Open-Sora v2.0 Video-to-Video (V2V) Inference
# ==============================================================================
# This script tests video-to-video continuation capability.
# It uses v2v_head_half mode which conditions on the first half of a video
# and generates a continuation.
#
# Modes available in Open-Sora v2.0:
#   - t2v: text-to-video
#   - i2v_head: image-to-video (image at head)
#   - i2v_tail: image-to-video (image at tail)
#   - i2v_loop: connect images
#   - v2v_head_half: video extension with first half as condition
#   - v2v_tail_half: video extension with second half as condition
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "Open-Sora v2.0 V2V Inference Test"
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
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v2.0-experiment"
ENV_PATH="${SCRATCH_BASE}/conda-envs/opensora20"
OUTPUT_DIR="${PROJECT_ROOT}/test_samples"

# ==============================================================================
# Setup Environment
# ==============================================================================
echo "Setting up environment..."
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

# Load conda - but don't let the module override our environment's PATH
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Activate environment
conda activate "${ENV_PATH}"

# CRITICAL: Ensure conda environment's bin comes FIRST in PATH
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "Environment activated: ${CONDA_PREFIX}"
echo ""

cd "${PROJECT_ROOT}"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Verify python
echo "Python check: $(which python) - $(python --version)"
if [[ "$(which python)" != "${CONDA_PREFIX}/bin/python" ]]; then
    echo "ERROR: Wrong Python!"
    exit 1
fi

# Install opensora
pip install -e . --no-deps 2>&1 | tail -2

echo ""

# ==============================================================================
# Prepare Test Video
# ==============================================================================
echo "=============================================================================="
echo "Step 1: Preparing Conditioning Video"
echo "=============================================================================="

# Check if we have the t2v output from previous test
T2V_VIDEO="${OUTPUT_DIR}/video_256px/prompt_0000.mp4"
if [ -f "$T2V_VIDEO" ]; then
    echo "Using T2V output as conditioning video: ${T2V_VIDEO}"
    COND_VIDEO="${T2V_VIDEO}"
else
    echo "T2V video not found at ${T2V_VIDEO}"
    echo "Please run 02_test_inference.sbatch first to generate a test video."
    echo ""
    echo "Alternatively, testing with sample image (i2v mode)..."
    
    # Fall back to i2v test with the sample image
    SAMPLE_IMAGE="${PROJECT_ROOT}/assets/texts/i2v.png"
    if [ -f "$SAMPLE_IMAGE" ]; then
        echo "Using sample image for I2V test: ${SAMPLE_IMAGE}"
        
        # Run I2V test instead
        echo ""
        echo "=============================================================================="
        echo "Running I2V Test (image-to-video)"
        echo "=============================================================================="
        
        MASTER_PORT=$((29500 + RANDOM % 1000))
        echo "Using MASTER_PORT: ${MASTER_PORT}"
        
        START_TIME=$(date +%s)
        
        torchrun --nproc_per_node 1 --standalone --master_port ${MASTER_PORT} scripts/diffusion/inference.py \
            configs/diffusion/inference/256px.py \
            --save-dir "${OUTPUT_DIR}" \
            --cond_type i2v_head \
            --ref "${SAMPLE_IMAGE}" \
            --prompt "A plump pig wallows in a muddy pond on a rustic farm" \
            --sampling_option.num_frames 33 \
            --sampling_option.num_steps 25 \
            --seed 42
        
        END_TIME=$(date +%s)
        echo ""
        echo "I2V test completed in $((END_TIME - START_TIME)) seconds"
        
        # Check output
        I2V_OUTPUT=$(find "${OUTPUT_DIR}" -name "*.mp4" -type f -mmin -5 | head -1)
        if [ -n "$I2V_OUTPUT" ]; then
            echo "Generated video: ${I2V_OUTPUT}"
            echo "File size: $(du -h "$I2V_OUTPUT" | cut -f1)"
            echo "SUCCESS! I2V test passed!"
        fi
        
        exit 0
    else
        echo "ERROR: No sample image found either."
        exit 1
    fi
fi

echo ""

# ==============================================================================
# Run V2V Inference Test
# ==============================================================================
echo "=============================================================================="
echo "Step 2: Running V2V Inference Test (v2v_head_half)"
echo "=============================================================================="
echo "Conditioning: First half of ${COND_VIDEO}"
echo "Prompt: Continuing the scene with natural motion"
echo "Resolution: 256px"
echo "Frames: 33"
echo ""

MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT: ${MASTER_PORT}"

START_TIME=$(date +%s)

# Run V2V with first-half conditioning
torchrun --nproc_per_node 1 --standalone --master_port ${MASTER_PORT} scripts/diffusion/inference.py \
    configs/diffusion/inference/256px.py \
    --save-dir "${OUTPUT_DIR}/v2v_test" \
    --cond_type v2v_head_half \
    --ref "${COND_VIDEO}" \
    --prompt "The scene continues with natural motion" \
    --sampling_option.num_frames 33 \
    --sampling_option.num_steps 25 \
    --seed 42

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "V2V inference completed in ${ELAPSED} seconds"
echo ""

# ==============================================================================
# Check Output
# ==============================================================================
echo "=============================================================================="
echo "Step 3: Checking Output"
echo "=============================================================================="

V2V_OUTPUT=$(find "${OUTPUT_DIR}/v2v_test" -name "*.mp4" -type f -mmin -10 | head -1)

if [ -n "$V2V_OUTPUT" ]; then
    echo "Generated video: ${V2V_OUTPUT}"
    echo "File size: $(du -h "$V2V_OUTPUT" | cut -f1)"
    echo ""
    echo "SUCCESS! V2V test passed!"
else
    echo "WARNING: No video found in v2v_test output directory"
    echo "Contents of ${OUTPUT_DIR}/v2v_test:"
    ls -la "${OUTPUT_DIR}/v2v_test/" 2>/dev/null || echo "Directory doesn't exist"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================================================="
echo "V2V Test Complete!"
echo "=============================================================================="
echo ""
echo "Conditioning video: ${COND_VIDEO}"
echo "Output directory: ${OUTPUT_DIR}/v2v_test"
echo "Inference time: ${ELAPSED} seconds"
echo ""
echo "Note: Open-Sora v2.0 v2v_head_half mode conditions on the first half"
echo "of the input video and generates a continuation."
echo ""
echo "For your TTA experiments with 22-frame conditioning (similar to v1.3),"
echo "you may need to:"
echo "  1. Pre-process videos to extract first 22 frames as conditioning"
echo "  2. Use i2v_head with the 22nd frame as the condition image"
echo "  3. Or use v2v_head_half with a video cropped to your desired length"
echo ""
echo "End time: $(date)"
echo "=============================================================================="

