#!/bin/bash
#SBATCH --job-name=eval_tta
#SBATCH --output=lora_experiment/logs/eval_%j.out
#SBATCH --error=lora_experiment/logs/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

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

# Source the scratch environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

# Load modules
module load anaconda3/2025.06

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
LORA_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_r16_lr2e4_100steps"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/evaluation"

# Override with environment variables if set
GT_DIR="${EVAL_GT_DIR:-$GT_DIR}"
BASELINE_DIR="${EVAL_BASELINE_DIR:-$BASELINE_DIR}"
LORA_DIR="${EVAL_LORA_DIR:-$LORA_DIR}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$OUTPUT_DIR}"
MAX_VIDEOS="${EVAL_MAX_VIDEOS:-}"

echo ""
echo "Configuration:"
echo "  Ground truth directory: ${GT_DIR}"
echo "  Baseline directory: ${BASELINE_DIR}"
echo "  LoRA directory: ${LORA_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Max videos: ${MAX_VIDEOS:-all}"
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

