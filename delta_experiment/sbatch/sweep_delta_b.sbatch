#!/bin/bash
#SBATCH --job-name=deltaB_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-35
#SBATCH --output=delta_experiment/logs/deltaB_sweep_%A_%a.out
#SBATCH --error=delta_experiment/logs/deltaB_sweep_%A_%a.err

# ==============================================================================
# δ-B Hyperparameter Sweep (steps, lr, l2)
# ==============================================================================
# Grid:
#   delta_steps ∈ {20, 50, 100}
#   delta_lr    ∈ {3e-3, 1e-3, 5e-4}
#   delta_l2    ∈ {0, 1e-6, 1e-5, 1e-4}
#
# Total jobs: 3 * 3 * 4 = 36 (SLURM array 0-35).
#
# Submit with (example):
#   sbatch --account=torch_pr_36_mren delta_experiment/sbatch/sweep_delta_b.sbatch
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "δ-B Hyperparameter Sweep"
echo "=============================================================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Array: ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-N/A}"
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
pip uninstall opensora -y 2>/dev/null || true
pip install -e . --quiet

mkdir -p "${PROJECT_ROOT}/delta_experiment/logs"

DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
REF_JSON="${PROJECT_ROOT}/lora_experiment/results/baseline/results.json"

STEPS_ARR=(20 50 100)
LR_ARR=("3e-3" "1e-3" "5e-4")
L2_ARR=("0" "1e-6" "1e-5" "1e-4")

N_LR=${#LR_ARR[@]}
N_L2=${#L2_ARR[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
STEP_IDX=$(( TASK_ID / (N_LR * N_L2) ))
REM=$(( TASK_ID % (N_LR * N_L2) ))
LR_IDX=$(( REM / N_L2 ))
L2_IDX=$(( REM % N_L2 ))

DELTA_STEPS="${STEPS_ARR[$STEP_IDX]}"
DELTA_LR="${LR_ARR[$LR_IDX]}"
DELTA_L2="${L2_ARR[$L2_IDX]}"

WARMUP_STEPS="${WARMUP_STEPS:-3}"
GROUPS_DOUBLE="${GROUPS_DOUBLE:-4}"
GROUPS_SINGLE="${GROUPS_SINGLE:-4}"

MAX_VIDEOS="${MAX_VIDEOS:-100}"
STRATIFIED="${STRATIFIED:-true}"
SEED="${SEED:-42}"

OUT_DIR="${PROJECT_ROOT}/delta_experiment/results/delta_b_sweep/steps${DELTA_STEPS}_lr${DELTA_LR}_l2${DELTA_L2}"

echo ""
echo "Configuration:"
echo "  Output directory: ${OUT_DIR}"
echo "  delta_steps: ${DELTA_STEPS}"
echo "  delta_lr: ${DELTA_LR}"
echo "  delta_l2: ${DELTA_L2}"
echo "  groups_double/single: ${GROUPS_DOUBLE}/${GROUPS_SINGLE}"
echo "  Max videos: ${MAX_VIDEOS} (stratified=${STRATIFIED}, seed=${SEED})"
echo ""

CMD="python ${PROJECT_ROOT}/delta_experiment/scripts/run_delta_b.py \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUT_DIR} \
  --reference-results-json ${REF_JSON} \
  --delta-steps ${DELTA_STEPS} \
  --delta-lr ${DELTA_LR} \
  --delta-l2 ${DELTA_L2} \
  --warmup-steps ${WARMUP_STEPS} \
  --groups-double ${GROUPS_DOUBLE} \
  --groups-single ${GROUPS_SINGLE} \
  --max-videos ${MAX_VIDEOS} \
  --seed ${SEED}"

if [ "${STRATIFIED}" = "true" ]; then
  CMD="${CMD} --stratified"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=============================================================================="
echo "Done: δ-B Sweep"
echo "Output: ${OUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

