#!/bin/bash
#SBATCH --job-name=bench_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/logs/bench_eval_%j.out
#SBATCH --error=delta_experiment/logs/bench_eval_%j.err

set -euo pipefail

echo "=============================================================================="
echo "Benchmark: Evaluate LoRA / δ-A / δ-B (PSNR/SSIM/LPIPS) + aggregate timing"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
echo "Start time: $(date)"
echo "=============================================================================="

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/open-sora-v2.0-experiment}"
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

GT_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
BASELINE_DIR="${PROJECT_ROOT}/lora_experiment/results/baseline"

LORA_DIR="${PROJECT_ROOT}/lora_experiment/results/benchmark_lora_r8_lr2e-4_20steps_100videos"
DELTA_A_DIR="${PROJECT_ROOT}/delta_experiment/results/benchmark_delta_a_global_100videos"
DELTA_B_DIR="${PROJECT_ROOT}/delta_experiment/results/benchmark_delta_b_steps50_lr3e-3_l21e-4_100videos"

echo "Evaluating LoRA benchmark..."
python ${PROJECT_ROOT}/lora_experiment/scripts/evaluate.py \
  --baseline-dir ${BASELINE_DIR} \
  --lora-dir ${LORA_DIR} \
  --gt-dir ${GT_DIR} \
  --output-dir ${LORA_DIR}/eval_baseline

echo "Evaluating δ-A benchmark..."
python ${PROJECT_ROOT}/lora_experiment/scripts/evaluate.py \
  --baseline-dir ${BASELINE_DIR} \
  --lora-dir ${DELTA_A_DIR} \
  --gt-dir ${GT_DIR} \
  --output-dir ${DELTA_A_DIR}/eval_baseline

echo "Evaluating δ-B benchmark..."
python ${PROJECT_ROOT}/lora_experiment/scripts/evaluate.py \
  --baseline-dir ${BASELINE_DIR} \
  --lora-dir ${DELTA_B_DIR} \
  --gt-dir ${GT_DIR} \
  --output-dir ${DELTA_B_DIR}/eval_baseline

echo "Aggregation: benchmark comparison JSON..."
python ${PROJECT_ROOT}/delta_experiment/scripts/compare_methods.py \
  --baseline_eval_dir ${LORA_DIR}/eval_baseline \
  --bestlora_eval_dir ${LORA_DIR}/eval_baseline \
  --bestlora_side lora \
  --delta_a_eval_dir ${DELTA_A_DIR}/eval_baseline \
  --delta_b_eval_dir ${DELTA_B_DIR}/eval_baseline \
  --baseline_run_dir ${BASELINE_DIR} \
  --bestlora_run_dir ${LORA_DIR} \
  --delta_a_run_dir ${DELTA_A_DIR} \
  --delta_b_run_dir ${DELTA_B_DIR} \
  --output ${PROJECT_ROOT}/delta_experiment/results/benchmark_comparison.json

echo "=============================================================================="
echo "Done: Benchmark evaluation"
echo "End time: $(date)"
echo "=============================================================================="

