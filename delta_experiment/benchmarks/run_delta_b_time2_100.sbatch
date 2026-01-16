#!/bin/bash
#SBATCH --job-name=bench_dB_time2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/logs/bench_deltaB_time2_%j.out
#SBATCH --error=delta_experiment/logs/bench_deltaB_time2_%j.err

set -euo pipefail

echo "=============================================================================="
echo "Benchmark: δ-B (time2) (100 stratified videos, full timing phases)"
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

DELTA_STEPS="${DELTA_STEPS:-20}"
DELTA_LR="${DELTA_LR:-3e-3}"
DELTA_L2="${DELTA_L2:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
GROUPS_DOUBLE="${GROUPS_DOUBLE:-4}"
GROUPS_SINGLE="${GROUPS_SINGLE:-4}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/delta_experiment/results/benchmark_delta_b_time2_steps${DELTA_STEPS}_lr${DELTA_LR}_l2${DELTA_L2}_g${GROUPS_DOUBLE}_${GROUPS_SINGLE}_100videos}"

INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
GUIDANCE="${GUIDANCE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"
DTYPE="${DTYPE:-bf16}"

MAX_VIDEOS="${MAX_VIDEOS:-100}"
SEED="${SEED:-42}"

CMD="python ${PROJECT_ROOT}/delta_experiment/scripts/run_delta_b_time2.py \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUT_DIR} \
  --reference-results-json ${BASELINE_RESULTS_JSON} \
  --delta-steps ${DELTA_STEPS} \
  --delta-lr ${DELTA_LR} \
  --delta-l2 ${DELTA_L2} \
  --warmup-steps ${WARMUP_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --max-grad-norm ${MAX_GRAD_NORM} \
  --groups-double ${GROUPS_DOUBLE} \
  --groups-single ${GROUPS_SINGLE} \
  --inference-steps ${INFERENCE_STEPS} \
  --guidance ${GUIDANCE} \
  --guidance-img ${GUIDANCE_IMG} \
  --dtype ${DTYPE} \
  --max-videos ${MAX_VIDEOS} \
  --seed ${SEED} \
  --stratified"

echo "Running: ${CMD}"
eval ${CMD}

echo "=============================================================================="
echo "Done: Benchmark δ-B time2"
echo "Output: ${OUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

