#!/bin/bash
#SBATCH --job-name=lora_tta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/logs/lora_tta_%j.out
#SBATCH --error=lora_experiment/logs/lora_tta_%j.err

# ==============================================================================
# LoRA Test-Time Adaptation for Open-Sora v2.0
# ==============================================================================
# This script performs test-time adaptation by fine-tuning LoRA adapters on
# conditioning frames for each video, then generating continuations.
# ==============================================================================

echo "=============================================================================="
echo "LoRA Test-Time Adaptation for Open-Sora v2.0"
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

# LoRA hyperparameters - can be overridden by environment variables
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"  # Best results at 2e-4 with early stopping
NUM_STEPS="${NUM_STEPS:-20}"  # 20 steps showed best results, more causes overfitting
WARMUP_STEPS="${WARMUP_STEPS:-5}"  # Proportional to 50 steps
TARGET_MLP="${TARGET_MLP:-false}"

# Inference parameters
INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
GUIDANCE="${GUIDANCE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"

# Other parameters
MAX_VIDEOS="${MAX_VIDEOS:-100}"  # Default to 100 videos for TTA experiments
STRATIFIED="${STRATIFIED:-true}"  # Use stratified sampling across classes
SEED="${SEED:-42}"
SAVE_LORA_WEIGHTS="${SAVE_LORA_WEIGHTS:-false}"
RESTART="${LORA_RESTART:-true}"  # Default to restart (fresh start)

# Construct output directory name
OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_r${LORA_RANK}_lr${LEARNING_RATE}_${NUM_STEPS}steps"
OUTPUT_DIR="${LORA_OUTPUT_DIR:-$OUTPUT_DIR}"
DATA_DIR="${LORA_DATA_DIR:-$DATA_DIR}"

echo ""
echo "Configuration:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Data directory: ${DATA_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  LoRA alpha: ${LORA_ALPHA}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Training steps: ${NUM_STEPS}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo "  Target MLP: ${TARGET_MLP}"
echo "  Inference steps: ${INFERENCE_STEPS}"
echo "  Guidance: ${GUIDANCE}"
echo "  Image guidance: ${GUIDANCE_IMG}"
echo "  Max videos: ${MAX_VIDEOS:-all}"
echo "  Stratified sampling: ${STRATIFIED}"
echo "  Restart (fresh start): ${RESTART}"
echo "  Seed: ${SEED}"
echo "  Save LoRA weights: ${SAVE_LORA_WEIGHTS}"
echo ""

# ==============================================================================
# Run LoRA TTA
# ==============================================================================

CMD="python ${PROJECT_ROOT}/lora_experiment/scripts/run_lora_tta.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --learning-rate ${LEARNING_RATE} \
    --num-steps ${NUM_STEPS} \
    --warmup-steps ${WARMUP_STEPS} \
    --inference-steps ${INFERENCE_STEPS} \
    --guidance ${GUIDANCE} \
    --guidance-img ${GUIDANCE_IMG} \
    --seed ${SEED}"

if [ "${TARGET_MLP}" = "true" ]; then
    CMD="${CMD} --target-mlp"
fi

if [ "${SAVE_LORA_WEIGHTS}" = "true" ]; then
    CMD="${CMD} --save-lora-weights"
fi

if [ -n "${MAX_VIDEOS}" ]; then
    CMD="${CMD} --max-videos ${MAX_VIDEOS}"
fi

if [ "${STRATIFIED}" = "true" ]; then
    CMD="${CMD} --stratified"
fi

if [ "${RESTART}" = "true" ]; then
    CMD="${CMD} --restart"
fi

echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=============================================================================="
echo "LoRA TTA Complete!"
echo "=============================================================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="
