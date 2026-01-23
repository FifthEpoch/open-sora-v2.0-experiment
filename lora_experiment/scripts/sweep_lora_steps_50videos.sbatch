#!/bin/bash
#SBATCH --job-name=lora_steps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/logs/lora_steps_%A_%a.out
#SBATCH --error=lora_experiment/logs/lora_steps_%A_%a.err

set -euo pipefail

echo "=============================================================================="
echo "LoRA Step Sweep (50 videos + eval + cleanup)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array ID: ${SLURM_ARRAY_TASK_ID:-na}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
echo "Start time: $(date)"
echo "=============================================================================="

PROJECT_ROOT="/scratch/wc3013/open-sora-v2.0-experiment"
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found at ${PROJECT_ROOT}"
    exit 1
fi

source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"
module load anaconda3/2025.06
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
eval "$(conda shell.bash hook)"
conda activate opensora20
set -u
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
pip install -e . --quiet
pip install lpips scikit-image --quiet

mkdir -p "${PROJECT_ROOT}/lora_experiment/logs"

DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
BASELINE_DIR="${PROJECT_ROOT}/lora_experiment/results/baseline"

STEPS_LIST=(10 20 30 50 100)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_STEPS="${STEPS_LIST[$IDX]}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
GUIDANCE="${GUIDANCE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"

MAX_VIDEOS="${MAX_VIDEOS:-50}"
STRATIFIED="${STRATIFIED:-true}"
SEED="${SEED:-42}"

# Augmentation parameters (TTA only)
AUG_ENABLED="${AUG_ENABLED:-true}"
AUG_FLIP="${AUG_FLIP:-true}"
AUG_ROTATE_DEG="${AUG_ROTATE_DEG:-5}"
AUG_SPEED_FACTORS="${AUG_SPEED_FACTORS:-0.5,2.0}"

OUT_DIR="${PROJECT_ROOT}/lora_experiment/results/sweep_lora_steps/steps${NUM_STEPS}_50videos"
EVAL_DIR="${PROJECT_ROOT}/lora_experiment/results/evaluation/lora_steps${NUM_STEPS}_50videos"

CMD="python ${PROJECT_ROOT}/lora_experiment/scripts/run_lora_tta.py \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUT_DIR} \
  --reference-results-json ${BASELINE_DIR}/results.json \
  --lora-rank ${LORA_RANK} \
  --lora-alpha ${LORA_ALPHA} \
  --learning-rate ${LEARNING_RATE} \
  --num-steps ${NUM_STEPS} \
  --warmup-steps ${WARMUP_STEPS} \
  --max-grad-norm ${MAX_GRAD_NORM} \
  --weight-decay ${WEIGHT_DECAY} \
  --inference-steps ${INFERENCE_STEPS} \
  --guidance ${GUIDANCE} \
  --guidance-img ${GUIDANCE_IMG} \
  --max-videos ${MAX_VIDEOS} \
  --seed ${SEED}"

if [ "${STRATIFIED}" = "true" ]; then
    CMD="${CMD} --stratified"
fi
if [ "${AUG_ENABLED}" = "true" ]; then
    CMD="${CMD} --aug-enabled"
fi
if [ "${AUG_FLIP}" = "true" ]; then
    CMD="${CMD} --aug-flip"
fi
if [ -n "${AUG_ROTATE_DEG}" ]; then
    CMD="${CMD} --aug-rotate-deg ${AUG_ROTATE_DEG}"
fi
if [ -n "${AUG_SPEED_FACTORS}" ]; then
    CMD="${CMD} --aug-speed-factors ${AUG_SPEED_FACTORS}"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "Evaluating LoRA (steps=${NUM_STEPS})..."
python ${PROJECT_ROOT}/lora_experiment/scripts/evaluate.py \
  --gt-dir "${DATA_DIR}" \
  --baseline-dir "${BASELINE_DIR}" \
  --lora-dir "${OUT_DIR}" \
  --output-dir "${EVAL_DIR}"

echo "Cleaning videos to save storage..."
rm -rf "${OUT_DIR}/videos" || true

echo "=============================================================================="
echo "Done: LoRA step sweep (steps=${NUM_STEPS})"
echo "Output: ${OUT_DIR}"
echo "Eval: ${EVAL_DIR}"
echo "End time: $(date)"
echo "=============================================================================="
