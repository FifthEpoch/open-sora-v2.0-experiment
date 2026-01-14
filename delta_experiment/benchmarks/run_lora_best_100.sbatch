#!/bin/bash
#SBATCH --job-name=bench_lora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/logs/bench_lora_%j.out
#SBATCH --error=delta_experiment/logs/bench_lora_%j.err

set -euo pipefail

echo "=============================================================================="
echo "Benchmark: LoRA best (100 stratified videos, full timing phases)"
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

DATA_DIR="${PROJECT_ROOT}/lora_experiment/data/ucf101_processed"
BASELINE_RESULTS_JSON="${PROJECT_ROOT}/lora_experiment/results/baseline/results.json"
OUT_DIR="${PROJECT_ROOT}/lora_experiment/results/benchmark_lora_r8_lr2e-4_20steps_100videos"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_STEPS="${NUM_STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
GUIDANCE="${GUIDANCE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"
DTYPE="${DTYPE:-bf16}"

MAX_VIDEOS="${MAX_VIDEOS:-100}"
SEED="${SEED:-42}"

CMD="python ${PROJECT_ROOT}/lora_experiment/scripts/run_lora_tta.py \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUT_DIR} \
  --lora-rank ${LORA_RANK} \
  --lora-alpha ${LORA_ALPHA} \
  --learning-rate ${LEARNING_RATE} \
  --num-steps ${NUM_STEPS} \
  --warmup-steps ${WARMUP_STEPS} \
  --inference-steps ${INFERENCE_STEPS} \
  --guidance ${GUIDANCE} \
  --guidance-img ${GUIDANCE_IMG} \
  --dtype ${DTYPE} \
  --max-videos ${MAX_VIDEOS} \
  --seed ${SEED} \
  --stratified \
  --restart \
  --reference-results-json ${BASELINE_RESULTS_JSON}"

echo "Running: ${CMD}"
eval ${CMD}

echo "=============================================================================="
echo "Done: Benchmark LoRA"
echo "Output: ${OUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

