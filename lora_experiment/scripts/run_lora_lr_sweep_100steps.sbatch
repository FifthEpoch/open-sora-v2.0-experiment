#!/bin/bash
#SBATCH --job-name=lora_lr_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-3
#SBATCH --output=lora_experiment/logs/lora_lr_sweep_%A_%a.out
#SBATCH --error=lora_experiment/logs/lora_lr_sweep_%A_%a.err

# ==============================================================================
# LoRA LR Sweep (100 steps) for Open-Sora v2.0
# ==============================================================================
# Runs 4 learning rates as a SLURM array on the same 100 stratified videos.
#
# Submit with (example):
#   sbatch --account=torch_pr_36_mren lora_experiment/scripts/run_lora_lr_sweep_100steps.sbatch
#
# Optional overrides at submit time:
#   LORA_RANK=8 LORA_ALPHA=32 sbatch ...
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "LoRA LR Sweep (100 steps) for Open-Sora v2.0"
echo "=============================================================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Array: ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
echo "Start time: $(date)"
echo "=============================================================================="

# IMPORTANT: update to your scratch checkout
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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
pip uninstall opensora -y 2>/dev/null || true
pip install -e . --quiet

mkdir -p "${PROJECT_ROOT}/lora_experiment/logs"

DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"

# LR sweep values (array index 0-3)
LRS=("1e-4" "5e-5" "2e-5" "1e-5")
LEARNING_RATE="${LRS[${SLURM_ARRAY_TASK_ID}]}"

# Keep these consistent across the sweep (override-able)
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
NUM_STEPS="${NUM_STEPS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
TARGET_MLP="${TARGET_MLP:-false}"

INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
GUIDANCE="${GUIDANCE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"

MAX_VIDEOS="${MAX_VIDEOS:-100}"
STRATIFIED="${STRATIFIED:-true}"
SEED="${SEED:-42}"
RESTART="${LORA_RESTART:-true}"

OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_r${LORA_RANK}_lr${LEARNING_RATE}_${NUM_STEPS}steps"

echo ""
echo "Configuration:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  LoRA rank/alpha: ${LORA_RANK}/${LORA_ALPHA}"
echo "  LR: ${LEARNING_RATE}"
echo "  Steps: ${NUM_STEPS} (warmup ${WARMUP_STEPS})"
echo "  Max videos: ${MAX_VIDEOS} (stratified=${STRATIFIED}, seed=${SEED})"
echo ""

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
  --seed ${SEED} \
  --max-videos ${MAX_VIDEOS}"

if [ "${STRATIFIED}" = "true" ]; then
  CMD="${CMD} --stratified"
fi

if [ "${TARGET_MLP}" = "true" ]; then
  CMD="${CMD} --target-mlp"
fi

if [ "${RESTART}" = "true" ]; then
  CMD="${CMD} --restart"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=============================================================================="
echo "Done: LoRA LR Sweep"
echo "Output: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

