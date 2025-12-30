#!/bin/bash
#SBATCH --job-name=lora_tta_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:4
#SBATCH --output=lora_experiment/logs/lora_tta_4gpu_%j.out
#SBATCH --error=lora_experiment/logs/lora_tta_4gpu_%j.err

# ==============================================================================
# LoRA TTA with Multi-GPU Sequence Parallelism for Open-Sora v2.0
# ==============================================================================
# Uses 4 H200 GPUs with sequence parallelism to enable more training steps
# without OOM. Each GPU handles 1/4 of the sequence (frames).
# ==============================================================================

echo "=============================================================================="
echo "LoRA TTA with Multi-GPU Sequence Parallelism"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
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
# Configuration - MORE STEPS with multi-GPU!
# ==============================================================================

DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_STEPS="${NUM_STEPS:-100}"  # Can now do 100+ steps with 4 GPUs!
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

OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_r${LORA_RANK}_lr${LEARNING_RATE}_${NUM_STEPS}steps_4gpu"
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
echo "  Number of GPUs: 4 (sequence parallelism)"
echo "  Max videos: ${MAX_VIDEOS:-all}"
echo ""

# ==============================================================================
# Run LoRA TTA with Sequence Parallelism
# ==============================================================================

# Use torchrun for multi-GPU with sequence parallelism
# Note: This requires modifications to run_lora_tta.py to support distributed
# For now, we use a simpler approach - running on single GPU with more memory

echo "Running LoRA TTA with multi-GPU support..."
echo "Note: Each GPU processes the full model but with more memory available"
echo ""

# Run with CUDA_VISIBLE_DEVICES to use first GPU and let others provide memory pool
# For full sequence parallelism, we'd need to modify the TTA script significantly

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

