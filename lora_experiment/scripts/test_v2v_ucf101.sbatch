#!/bin/bash
#SBATCH --job-name=test_v2v_ucf101
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/logs/test_v2v_%j.out
#SBATCH --error=lora_experiment/logs/test_v2v_%j.err

# ==============================================================================
# Test v2v_head Inference with Preprocessed UCF-101 Videos
# ==============================================================================
# This script verifies that v2v_head inference works with the preprocessed
# UCF-101 videos before running full TTA experiments.
#
# It tests a few sample videos to ensure:
# - Video loading works correctly
# - v2v_head conditioning (33 frames) works
# - Output video generation succeeds
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "Testing v2v_head with Preprocessed UCF-101 Videos"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "=============================================================================="

# Configuration
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v2.0-experiment"
ENV_PATH="${SCRATCH_BASE}/conda-envs/opensora20"
DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/test_v2v_output"

# Number of test videos
NUM_TEST_VIDEOS=3

# Create directories
mkdir -p "${PROJECT_ROOT}/lora_experiment/logs"
mkdir -p "${OUTPUT_DIR}"

# Setup environment
echo ""
echo "Setting up environment..."
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

conda activate "${ENV_PATH}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Environment activated: ${CONDA_PREFIX}"
echo "Python: $(which python) - $(python --version)"

cd "${PROJECT_ROOT}"
pip install -e . --no-deps 2>&1 | tail -2

# Verify data exists
if [ ! -f "${DATA_DIR}/metadata.csv" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run: sbatch lora_experiment/scripts/preprocess_ucf101.sbatch"
    exit 1
fi

NUM_VIDEOS=$(tail -n +2 "${DATA_DIR}/metadata.csv" | wc -l)
echo ""
echo "Found ${NUM_VIDEOS} preprocessed videos"
echo "Testing with first ${NUM_TEST_VIDEOS} videos..."
echo ""

# Use random port
MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT: ${MASTER_PORT}"
echo ""

# Test each video
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in $(seq 1 ${NUM_TEST_VIDEOS}); do
    echo "=============================================================================="
    echo "Test Video ${i}/${NUM_TEST_VIDEOS}"
    echo "=============================================================================="
    
    # Get video path from metadata (skip header, get nth line)
    LINE=$((i + 1))
    VIDEO_PATH=$(sed -n "${LINE}p" "${DATA_DIR}/metadata.csv" | cut -d',' -f1)
    CAPTION=$(sed -n "${LINE}p" "${DATA_DIR}/metadata.csv" | cut -d',' -f11)
    
    echo "Video: ${VIDEO_PATH}"
    echo "Caption: ${CAPTION}"
    echo ""
    
    if [ ! -f "${VIDEO_PATH}" ]; then
        echo "ERROR: Video file not found: ${VIDEO_PATH}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # Create output subdirectory for this test
    TEST_OUTPUT="${OUTPUT_DIR}/test_${i}"
    mkdir -p "${TEST_OUTPUT}"
    
    START_TIME=$(date +%s)
    
    # Run v2v_head inference
    echo "Running v2v_head inference..."
    torchrun --nproc_per_node 1 --standalone --master_port ${MASTER_PORT} \
        scripts/diffusion/inference.py \
        configs/diffusion/inference/256px.py \
        --cond_type v2v_head \
        --ref "${VIDEO_PATH}" \
        --prompt "${CAPTION}" \
        --save-dir "${TEST_OUTPUT}" \
        --sampling_option.num_frames 65 \
        --sampling_option.num_steps 25 \
        --seed 42
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    # Check if output was generated
    OUTPUT_VIDEO=$(find "${TEST_OUTPUT}" -name "*.mp4" -type f -mmin -5 | head -1)
    
    if [ -n "$OUTPUT_VIDEO" ]; then
        echo ""
        echo "✓ Success! Generated: ${OUTPUT_VIDEO}"
        echo "  File size: $(du -h "$OUTPUT_VIDEO" | cut -f1)"
        echo "  Time: ${ELAPSED} seconds"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "✗ Failed: No output video generated"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo ""
    
    # Increment port for next run
    MASTER_PORT=$((MASTER_PORT + 1))
done

# Summary
echo "=============================================================================="
echo "Test Summary"
echo "=============================================================================="
echo "Successful: ${SUCCESS_COUNT}/${NUM_TEST_VIDEOS}"
echo "Failed: ${FAIL_COUNT}/${NUM_TEST_VIDEOS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

if [ ${SUCCESS_COUNT} -eq ${NUM_TEST_VIDEOS} ]; then
    echo "✓ All tests passed! v2v_head is working correctly."
    echo ""
    echo "You can now run full experiments:"
    echo "  sbatch lora_experiment/scripts/generate_baseline.sbatch"
    echo "  sbatch lora_experiment/scripts/run_lora_tta.sbatch"
else
    echo "⚠️  Some tests failed. Check the error logs above."
fi

echo ""
echo "End time: $(date)"
echo "=============================================================================="

