#!/bin/bash
#SBATCH --job-name=eval_tta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/logs/eval_%j.out
#SBATCH --error=lora_experiment/logs/eval_%j.err

# ==============================================================================
# Evaluate TTA Results for Open-Sora v2.0
# ==============================================================================
# This script computes metrics comparing baseline and LoRA TTA results
# against ground truth.
# ==============================================================================

echo "=============================================================================="
echo "Evaluation for Open-Sora v2.0 TTA Experiments"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================================================="

# ==============================================================================
# Environment Setup
# ==============================================================================

# IMPORTANT: Use absolute path - SLURM copies scripts to temp location
PROJECT_ROOT="/scratch/wc3013/open-sora-v2.0-experiment"

# Verify project root exists
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found at ${PROJECT_ROOT}"
    echo "Please update PROJECT_ROOT in this script to your actual project location"
    exit 1
fi

# Source the scratch environment setup
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

# Load modules
module load anaconda3/2025.06

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate opensora20
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "Environment activated: ${CONDA_PREFIX}"
echo "Python: $(which python) - $(python --version)"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Install additional dependencies for evaluation
pip install lpips scikit-image --quiet

# Create logs directory
mkdir -p "${PROJECT_ROOT}/lora_experiment/logs"

# ==============================================================================
# Configuration
# ==============================================================================

GT_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
BASELINE_DIR="${PROJECT_ROOT}/lora_experiment/results/baseline"
LORA_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_r16_lr2e-4_20steps"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/evaluation"

# Override with environment variables if set
GT_DIR="${EVAL_GT_DIR:-$GT_DIR}"
BASELINE_DIR="${EVAL_BASELINE_DIR:-$BASELINE_DIR}"
LORA_DIR="${EVAL_LORA_DIR:-$LORA_DIR}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$OUTPUT_DIR}"
MAX_VIDEOS="${EVAL_MAX_VIDEOS:-}"
EVAL_FRAME_COUNT="${EVAL_FRAME_COUNT:-}"
EVAL_FRAME_STRIDE="${EVAL_FRAME_STRIDE:-1}"
EVAL_INCLUDE_CONTEXT="${EVAL_INCLUDE_CONTEXT:-false}"
EVAL_BEST_OF="${EVAL_BEST_OF:-1}"

echo ""
echo "Configuration:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Ground truth directory: ${GT_DIR}"
echo "  Baseline directory: ${BASELINE_DIR}"
echo "  LoRA directory: ${LORA_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Max videos: ${MAX_VIDEOS:-all}"
echo "  Eval frame count: ${EVAL_FRAME_COUNT:-all}"
echo "  Eval frame stride: ${EVAL_FRAME_STRIDE}"
echo "  Include context: ${EVAL_INCLUDE_CONTEXT}"
echo "  Best-of: ${EVAL_BEST_OF}"
echo ""

# ==============================================================================
# Run Evaluation
# ==============================================================================

CMD="python ${PROJECT_ROOT}/lora_experiment/scripts/evaluate.py \
    --gt-dir ${GT_DIR} \
    --output-dir ${OUTPUT_DIR}"

if [ -d "${BASELINE_DIR}" ]; then
    CMD="${CMD} --baseline-dir ${BASELINE_DIR}"
else
    echo "Warning: Baseline directory not found, skipping baseline evaluation"
fi

if [ -d "${LORA_DIR}" ]; then
    CMD="${CMD} --lora-dir ${LORA_DIR}"
else
    echo "Warning: LoRA directory not found, skipping LoRA evaluation"
fi

if [ -n "${MAX_VIDEOS}" ]; then
    CMD="${CMD} --max-videos ${MAX_VIDEOS}"
fi
if [ -n "${EVAL_FRAME_COUNT}" ]; then
    CMD="${CMD} --eval-frame-count ${EVAL_FRAME_COUNT}"
fi
if [ -n "${EVAL_FRAME_STRIDE}" ]; then
    CMD="${CMD} --eval-frame-stride ${EVAL_FRAME_STRIDE}"
fi
if [ "${EVAL_INCLUDE_CONTEXT}" = "true" ]; then
    CMD="${CMD} --include-context"
fi
if [ -n "${EVAL_BEST_OF}" ]; then
    CMD="${CMD} --best-of ${EVAL_BEST_OF}"
fi

echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=============================================================================="
echo "Evaluation Complete!"
echo "=============================================================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="
