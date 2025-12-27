#!/bin/bash
#SBATCH --job-name=lora_tta_v2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=lora_experiment/logs/tta_%j.out
#SBATCH --error=lora_experiment/logs/tta_%j.err

# ==============================================================================
# LoRA Test-Time Adaptation Experiment
# ==============================================================================
# This script runs LoRA TTA on UCF-101 videos.
# It fine-tunes LoRA adapters on conditioning frames for each video,
# then generates continuations.
#
# Estimated runtime: ~2 min per video (100 steps + inference)
# For 1000 videos: ~33 hours
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "LoRA TTA Experiment for Open-Sora v2.0"
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

# Experiment settings
DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/rank${LORA_RANK:-16}_lr${LEARNING_RATE:-2e4}_${NUM_STEPS:-100}steps"

# LoRA hyperparameters (can be overridden via environment variables)
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_STEPS="${NUM_STEPS:-100}"

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

# Check data
if [ ! -f "${DATA_DIR}/metadata.csv" ]; then
    echo "ERROR: Dataset not preprocessed!"
    echo "Please run: sbatch lora_experiment/scripts/preprocess_ucf101.sbatch"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "Experiment Configuration"
echo "=============================================================================="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "LoRA rank: ${LORA_RANK}"
echo "LoRA alpha: ${LORA_ALPHA}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Fine-tuning steps: ${NUM_STEPS}"
echo "=============================================================================="
echo ""

# Use random port for distributed
MASTER_PORT=$((29500 + RANDOM % 1000))
export MASTER_PORT

# Run TTA experiment
python lora_experiment/scripts/run_lora_tta.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --learning-rate ${LEARNING_RATE} \
    --num-steps ${NUM_STEPS} \
    --dtype bf16 \
    --seed 42

echo ""
echo "=============================================================================="
echo "Experiment Complete!"
echo "=============================================================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

