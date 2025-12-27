#!/bin/bash
#SBATCH --job-name=baseline_v2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=lora_experiment/logs/baseline_%j.out
#SBATCH --error=lora_experiment/logs/baseline_%j.err

# ==============================================================================
# Generate Baseline Video Continuations (No TTA)
# ==============================================================================
# This script generates video continuations using vanilla Open-Sora v2.0
# without any LoRA fine-tuning. These serve as the baseline for comparison.
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "Baseline Generation for Open-Sora v2.0"
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
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/baseline"

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
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Use random port for distributed
MASTER_PORT=$((29500 + RANDOM % 1000))
export MASTER_PORT

# Run baseline generation
python lora_experiment/scripts/generate_baseline.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --dtype bf16 \
    --seed 42

echo ""
echo "=============================================================================="
echo "Baseline Generation Complete!"
echo "=============================================================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

