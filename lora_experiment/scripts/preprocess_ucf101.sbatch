#!/bin/bash
#SBATCH --job-name=preprocess_ucf101_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=lora_experiment/logs/preprocess_%j.out
#SBATCH --error=lora_experiment/logs/preprocess_%j.err

# ==============================================================================
# Preprocess UCF-101 for Open-Sora v2.0 TTA Experiments
# ==============================================================================
# This script preprocesses UCF-101 videos for TTA experiments:
# - Resizes to 256px resolution (16:9 aspect ratio)
# - Ensures 65 frames per video (33 conditioning + 32 generation)
# - Generates metadata CSV for training
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "UCF-101 Preprocessing for Open-Sora v2.0"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================================================="

# Configuration
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v2.0-experiment"
V13_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
ENV_PATH="${SCRATCH_BASE}/conda-envs/opensora20"

# UCF-101 source (from v1.3 repo)
UCF101_SOURCE="${V13_ROOT}/env_setup/download_ucf101/ucf101_org"

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"

# Create log directory
mkdir -p "${PROJECT_ROOT}/lora_experiment/logs"

# Setup environment
echo ""
echo "Setting up environment..."
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

conda activate "${ENV_PATH}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "Environment activated: ${CONDA_PREFIX}"
echo "Python: $(which python) - $(python --version)"

cd "${PROJECT_ROOT}"

# Check if UCF-101 source exists
if [ ! -d "${UCF101_SOURCE}" ]; then
    echo "ERROR: UCF-101 source not found at ${UCF101_SOURCE}"
    echo ""
    echo "Please download UCF-101 first in the v1.3 repo:"
    echo "  cd ${V13_ROOT}/env_setup/download_ucf101"
    echo "  sbatch preprocess_ucf101.sbatch"
    exit 1
fi

echo ""
echo "UCF-101 source: ${UCF101_SOURCE}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run preprocessing
python lora_experiment/scripts/preprocess_ucf101.py \
    --input-dir "${UCF101_SOURCE}" \
    --output-dir "${OUTPUT_DIR}" \
    --fps 24 \
    --frames 65 \
    --resolution "256px" \
    --aspect-ratio "16:9" \
    --conditioning-frames 33

echo ""
echo "=============================================================================="
echo "Preprocessing Complete!"
echo "=============================================================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

