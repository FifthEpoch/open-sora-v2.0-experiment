#!/bin/bash
#SBATCH --job-name=deltaA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/logs/deltaA_%j.out
#SBATCH --error=delta_experiment/logs/deltaA_%j.err

set -euo pipefail

echo "=============================================================================="
echo "δ-TTA Option A (global vec δ)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
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
source "${PROJECT_ROOT}/env_setup/11_pexels_400k_env.sh"
module load anaconda3/2025.06
# Some site conda activate hooks assume certain vars exist; avoid nounset crash.
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
eval "$(conda shell.bash hook)"
conda activate opensora20
set -u
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
pip uninstall opensora -y 2>/dev/null || true
pip install -e . --quiet

mkdir -p "${PROJECT_ROOT}/delta_experiment/logs"

DATA_DIR="${PEXELS_100_DIR}"
OUT_DIR="${PROJECT_ROOT}/delta_experiment/results/delta_a_global"

# Hyperparams (override by env if desired)
DELTA_STEPS="${DELTA_STEPS:-20}"
DELTA_LR="${DELTA_LR:-5e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"

MAX_VIDEOS="${MAX_VIDEOS:-100}"
STRATIFIED="${STRATIFIED:-true}"
SEED="${SEED:-42}"

# Augmentation parameters (TTA only)
AUG_ENABLED="${AUG_ENABLED:-false}"
AUG_FLIP="${AUG_FLIP:-true}"
AUG_ROTATE_DEG="${AUG_ROTATE_DEG:-5}"
AUG_SPEED_FACTORS="${AUG_SPEED_FACTORS:-0.5,2.0}"

CMD="python ${PROJECT_ROOT}/delta_experiment/scripts/run_delta_a.py \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUT_DIR} \
  --reference-results-json ${PROJECT_ROOT}/lora_experiment/results/baseline/results.json \
  --delta-steps ${DELTA_STEPS} \
  --delta-lr ${DELTA_LR} \
  --warmup-steps ${WARMUP_STEPS} \
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

echo "=============================================================================="
echo "Done: δ-TTA Option A"
echo "Output: ${OUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="


