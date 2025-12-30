#!/bin/bash
#SBATCH --job-name=lora_tta_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --output=lora_experiment/logs/lora_tta_4gpu_%j.out
#SBATCH --error=lora_experiment/logs/lora_tta_4gpu_%j.err

# ==============================================================================
# LoRA TTA with Multi-GPU for Open-Sora v2.0
# ==============================================================================
# This script requests 4 H200 GPUs for training.
#
# Current implementation:
#   - Uses first GPU for training (CUDA:0)
#   - Other GPUs reserved but not actively used by TTA script
#
# For full distributed training with sequence parallelism, the TTA script
# would need to be modified to use ColossalAI's hybrid plugin:
#   torchrun --nproc_per_node=4 scripts/train.py --plugin hybrid --sp_size 4
#
# Reference: https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_main_commands/
# ==============================================================================

echo "=============================================================================="
echo "LoRA TTA with Multi-GPU Support"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Allocated GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Start time: $(date)"
echo "=============================================================================="

# ==============================================================================
# Environment Setup
# ==============================================================================

PROJECT_ROOT="/scratch/wc3013/open-sora-v2.0-experiment"

if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found at ${PROJECT_ROOT}"
    exit 1
fi

source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate opensora20
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "Environment activated: ${CONDA_PREFIX}"
echo "Python: $(which python) - $(python --version)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
pip uninstall opensora -y 2>/dev/null || true
pip install -e . --quiet

mkdir -p "${PROJECT_ROOT}/lora_experiment/logs"

# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_STEPS="${NUM_STEPS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
TARGET_MLP="${TARGET_MLP:-false}"

INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
GUIDANCE="${GUIDANCE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"

MAX_VIDEOS="${MAX_VIDEOS:-100}"
STRATIFIED="${STRATIFIED:-true}"
SEED="${SEED:-42}"
SAVE_LORA_WEIGHTS="${SAVE_LORA_WEIGHTS:-false}"
RESTART="${LORA_RESTART:-true}"

OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_r${LORA_RANK}_lr${LEARNING_RATE}_${NUM_STEPS}steps"
OUTPUT_DIR="${LORA_OUTPUT_DIR:-$OUTPUT_DIR}"
DATA_DIR="${LORA_DATA_DIR:-$DATA_DIR}"

echo ""
echo "Configuration:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Data directory: ${DATA_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Training steps: ${NUM_STEPS}"
echo "  Max videos: ${MAX_VIDEOS:-all}"
echo ""

# ==============================================================================
# GPU Memory Check
# ==============================================================================

echo "GPU Memory Status Before Training:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

# ==============================================================================
# Run LoRA TTA
# ==============================================================================

# Note: Current TTA script uses single GPU. For true multi-GPU distributed
# training with sequence parallelism, use:
#   torchrun --nproc_per_node=4 --master_port=$((29500 + RANDOM % 1000)) \
#       ${PROJECT_ROOT}/lora_experiment/scripts/run_lora_tta_distributed.py ...
#
# The distributed version would require ColossalAI hybrid plugin setup.

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
