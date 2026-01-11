#!/bin/bash
#SBATCH --job-name=eval_deltaB_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-35
#SBATCH --output=delta_experiment/logs/eval_deltaB_sweep_%A_%a.out
#SBATCH --error=delta_experiment/logs/eval_deltaB_sweep_%A_%a.err

# ==============================================================================
# Evaluate δ-B sweep runs (baseline vs run) into RUN_DIR/eval_baseline
# ==============================================================================
# Assumes each sweep run lives under:
#   delta_experiment/results/delta_b_sweep/steps*_lr*_l2*
#
# This is a SLURM array with 36 tasks to match the sweep grid.
#
# Submit:
#   sbatch --account=torch_pr_36_mren delta_experiment/sbatch/evaluate_sweep_delta_b.sbatch
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "Evaluate δ-B sweep runs"
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

BASELINE_DIR="${PROJECT_ROOT}/lora_experiment/results/baseline"
GT_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
SWEEP_ROOT="${PROJECT_ROOT}/delta_experiment/results/delta_b_sweep"

if [ ! -d "${SWEEP_ROOT}" ]; then
  echo "ERROR: sweep root not found: ${SWEEP_ROOT}"
  exit 1
fi

# Collect run dirs deterministically
mapfile -t RUN_DIRS < <(ls -d ${SWEEP_ROOT}/steps*_lr*_l2* 2>/dev/null | sort)
N_RUNS=${#RUN_DIRS[@]}
if [ "${N_RUNS}" -eq 0 ]; then
  echo "ERROR: No run dirs found under ${SWEEP_ROOT} (expected steps*_lr*_l2*)"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "${TASK_ID}" -ge "${N_RUNS}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} but only N_RUNS=${N_RUNS} dirs found."
  echo "Either adjust #SBATCH --array or fix sweep outputs."
  exit 1
fi

RUN_DIR="${RUN_DIRS[$TASK_ID]}"
OUT_DIR="${RUN_DIR}/eval_baseline"

echo ""
echo "Evaluating run:"
echo "  RUN_DIR: ${RUN_DIR}"
echo "  OUT_DIR: ${OUT_DIR}"
echo ""

CMD="python ${PROJECT_ROOT}/lora_experiment/scripts/evaluate.py \
  --baseline-dir ${BASELINE_DIR} \
  --lora-dir ${RUN_DIR} \
  --gt-dir ${GT_DIR} \
  --output-dir ${OUT_DIR}"

echo "Running: ${CMD}"
eval ${CMD}

echo "=============================================================================="
echo "Done: eval δ-B sweep"
echo "OUT_DIR: ${OUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

