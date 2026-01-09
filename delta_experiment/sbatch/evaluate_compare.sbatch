#!/bin/bash
#SBATCH --job-name=delta_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/logs/delta_eval_%j.out
#SBATCH --error=delta_experiment/logs/delta_eval_%j.err

set -euo pipefail

echo "=============================================================================="
echo "δ-TTA Evaluation + Comparison"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================================================="

PROJECT_ROOT="/scratch/wc3013/open-sora-v2.0-experiment"
if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "ERROR: Project root not found: ${PROJECT_ROOT}"
  exit 1
fi

source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate opensora20
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
pip install lpips scikit-image --quiet

mkdir -p "${PROJECT_ROOT}/delta_experiment/logs"

GT_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
BASELINE_DIR="${PROJECT_ROOT}/lora_experiment/results/baseline"
BEST_LORA_DIR="${PROJECT_ROOT}/lora_experiment/results/lora_r8_lr2e-4_20steps"

DELTA_A_DIR="${PROJECT_ROOT}/delta_experiment/results/delta_a_global"
DELTA_B_DIR="${PROJECT_ROOT}/delta_experiment/results/delta_b_grouped"
DELTA_C_DIR="${PROJECT_ROOT}/delta_experiment/results/delta_c_output"

OUT_ROOT="${PROJECT_ROOT}/delta_experiment/results/evaluation"
mkdir -p "${OUT_ROOT}"

echo "Evaluating δ-A..."
python ${PROJECT_ROOT}/delta_experiment/scripts/evaluate_delta.py \
  --gt-dir "${GT_DIR}" \
  --baseline-dir "${BASELINE_DIR}" \
  --best-lora-dir "${BEST_LORA_DIR}" \
  --delta-dir "${DELTA_A_DIR}" \
  --output-dir "${OUT_ROOT}/deltaA"

echo "Evaluating δ-B..."
python ${PROJECT_ROOT}/delta_experiment/scripts/evaluate_delta.py \
  --gt-dir "${GT_DIR}" \
  --baseline-dir "${BASELINE_DIR}" \
  --best-lora-dir "${BEST_LORA_DIR}" \
  --delta-dir "${DELTA_B_DIR}" \
  --output-dir "${OUT_ROOT}/deltaB"

echo "Evaluating δ-C..."
python ${PROJECT_ROOT}/delta_experiment/scripts/evaluate_delta.py \
  --gt-dir "${GT_DIR}" \
  --baseline-dir "${BASELINE_DIR}" \
  --best-lora-dir "${BEST_LORA_DIR}" \
  --delta-dir "${DELTA_C_DIR}" \
  --output-dir "${OUT_ROOT}/deltaC"

echo "Writing comparison summary..."
python ${PROJECT_ROOT}/delta_experiment/scripts/compare_methods.py \
  --baseline_eval_dir "${OUT_ROOT}/deltaA/baseline_vs_delta" \
  --bestlora_eval_dir "${OUT_ROOT}/deltaA/bestlora_vs_delta" \
  --delta_a_eval_dir "${OUT_ROOT}/deltaA/baseline_vs_delta" \
  --delta_b_eval_dir "${OUT_ROOT}/deltaB/baseline_vs_delta" \
  --delta_c_eval_dir "${OUT_ROOT}/deltaC/baseline_vs_delta" \
  --output "${OUT_ROOT}/comparison.json"

echo "=============================================================================="
echo "Done: δ-TTA Evaluation + Comparison"
echo "Output: ${OUT_ROOT}"
echo "End time: $(date)"
echo "=============================================================================="


