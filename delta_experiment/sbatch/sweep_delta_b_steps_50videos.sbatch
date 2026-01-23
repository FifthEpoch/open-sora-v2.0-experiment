#!/bin/bash
#SBATCH --job-name=deltaB_steps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/logs/deltaB_steps_%A_%a.out
#SBATCH --error=delta_experiment/logs/deltaB_steps_%A_%a.err

set -euo pipefail

echo "=============================================================================="
echo "δ-TTA Option B Step Sweep (50 videos + eval + cleanup)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array ID: ${SLURM_ARRAY_TASK_ID:-na}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
echo "Start time: $(date)"
echo "=============================================================================="

PROJECT_ROOT="/scratch/wc3013/open-sora-v2.0-experiment"
if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "ERROR: Project root not found: ${PROJECT_ROOT}"
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

mkdir -p "${PROJECT_ROOT}/delta_experiment/logs"

DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
BASELINE_DIR="${PROJECT_ROOT}/lora_experiment/results/baseline"
BEST_LORA_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_r8_lr2e-4_20steps_aug"

STEPS_LIST=(10 20 30 50 100)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
DELTA_STEPS="${STEPS_LIST[$IDX]}"

DELTA_LR="${DELTA_LR:-3e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
GROUPS_DOUBLE="${GROUPS_DOUBLE:-4}"
GROUPS_SINGLE="${GROUPS_SINGLE:-4}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
STRATIFIED="${STRATIFIED:-true}"
SEED="${SEED:-42}"

# Augmentation parameters (TTA only)
AUG_ENABLED="${AUG_ENABLED:-true}"
AUG_FLIP="${AUG_FLIP:-true}"
AUG_ROTATE_DEG="${AUG_ROTATE_DEG:-5}"
AUG_SPEED_FACTORS="${AUG_SPEED_FACTORS:-0.5,2.0}"

OUT_DIR="${PROJECT_ROOT}/delta_experiment/results/sweep_delta_b_steps/steps${DELTA_STEPS}_50videos"
EVAL_DIR="${PROJECT_ROOT}/delta_experiment/results/evaluation/deltaB_steps${DELTA_STEPS}_50videos"

CMD="python ${PROJECT_ROOT}/delta_experiment/scripts/run_delta_b.py \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUT_DIR} \
  --reference-results-json ${BASELINE_DIR}/results.json \
  --delta-steps ${DELTA_STEPS} \
  --delta-lr ${DELTA_LR} \
  --warmup-steps ${WARMUP_STEPS} \
  --groups-double ${GROUPS_DOUBLE} \
  --groups-single ${GROUPS_SINGLE} \
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

echo "Evaluating δ-B (steps=${DELTA_STEPS})..."
python ${PROJECT_ROOT}/delta_experiment/scripts/evaluate_delta.py \
  --gt-dir "${DATA_DIR}" \
  --baseline-dir "${BASELINE_DIR}" \
  --best-lora-dir "${BEST_LORA_DIR}" \
  --delta-dir "${OUT_DIR}" \
  --output-dir "${EVAL_DIR}"

echo "Cleaning videos to save storage..."
rm -rf "${OUT_DIR}/videos" || true

echo "=============================================================================="
echo "Done: δ-B step sweep (steps=${DELTA_STEPS})"
echo "Output: ${OUT_DIR}"
echo "Eval: ${EVAL_DIR}"
echo "End time: $(date)"
echo "=============================================================================="
