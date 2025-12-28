#!/bin/bash
#SBATCH --job-name=baseline_gen
#SBATCH --output=lora_experiment/logs/baseline_%j.out
#SBATCH --error=lora_experiment/logs/baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# ==============================================================================
# Generate Baseline Video Continuations for Open-Sora v2.0
# ==============================================================================
# This script generates video continuations using the vanilla Open-Sora v2.0
# model without any LoRA fine-tuning. These serve as the baseline for comparison.
# ==============================================================================

echo "=============================================================================="
echo "Baseline Generation for Open-Sora v2.0"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
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

# Reinstall opensora in editable mode
cd "${PROJECT_ROOT}"
pip uninstall opensora -y 2>/dev/null || true
pip install -e . --quiet

# Create logs directory
mkdir -p "${PROJECT_ROOT}/lora_experiment/logs"

# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/baseline"

# Override with environment variables if set
DATA_DIR="${BASELINE_DATA_DIR:-$DATA_DIR}"
OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-$OUTPUT_DIR}"

NUM_STEPS="${BASELINE_NUM_STEPS:-25}"
GUIDANCE="${BASELINE_GUIDANCE:-7.5}"
GUIDANCE_IMG="${BASELINE_GUIDANCE_IMG:-3.0}"
MAX_VIDEOS="${BASELINE_MAX_VIDEOS:-}"
SEED="${BASELINE_SEED:-42}"

echo ""
echo "Configuration:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Data directory: ${DATA_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Diffusion steps: ${NUM_STEPS}"
echo "  Guidance: ${GUIDANCE}"
echo "  Image guidance: ${GUIDANCE_IMG}"
echo "  Max videos: ${MAX_VIDEOS:-all}"
echo "  Seed: ${SEED}"
echo ""

# ==============================================================================
# Run Baseline Generation
# ==============================================================================

CMD="python ${PROJECT_ROOT}/lora_experiment/scripts/generate_baseline.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --num-steps ${NUM_STEPS} \
    --guidance ${GUIDANCE} \
    --guidance-img ${GUIDANCE_IMG} \
    --seed ${SEED}"

if [ -n "${MAX_VIDEOS}" ]; then
    CMD="${CMD} --max-videos ${MAX_VIDEOS}"
fi

echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=============================================================================="
echo "Baseline Generation Complete!"
echo "=============================================================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="
