#!/bin/bash
#SBATCH --job-name=bench_deltaA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/logs/bench_deltaA_%j.out
#SBATCH --error=delta_experiment/logs/bench_deltaA_%j.err

set -euo pipefail

echo "=============================================================================="
echo "Benchmark: δ-A (global vec) (100 stratified videos, full timing phases)"
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
OUT_DIR="${PROJECT_ROOT}/delta_experiment/results/benchmark_delta_a_global_100videos"

# Reuse the exact δ-A hyperparams from the prior run config.json.
CFG_SRC="${CFG_SRC:-${PROJECT_ROOT}/delta_experiment/results/delta_a_global/config.json}"
if [ ! -f "${CFG_SRC}" ]; then
  echo "ERROR: δ-A source config not found: ${CFG_SRC}"
  echo "Set CFG_SRC to an existing delta_a_global/config.json path."
  exit 1
fi

DELTA_STEPS="$(python - <<'PY'
import json, os
cfg=json.load(open(os.environ["CFG_SRC"]))
print(cfg.get("delta_steps", 20))
PY
)"
DELTA_LR="$(python - <<'PY'
import json, os
cfg=json.load(open(os.environ["CFG_SRC"]))
print(cfg.get("delta_lr", 5e-3))
PY
)"
DELTA_L2="$(python - <<'PY'
import json, os
cfg=json.load(open(os.environ["CFG_SRC"]))
print(cfg.get("delta_l2", 0.0))
PY
)"
WARMUP_STEPS="$(python - <<'PY'
import json, os
cfg=json.load(open(os.environ["CFG_SRC"]))
print(cfg.get("warmup_steps", 3))
PY
)"
WEIGHT_DECAY="$(python - <<'PY'
import json, os
cfg=json.load(open(os.environ["CFG_SRC"]))
print(cfg.get("weight_decay", 0.0))
PY
)"
MAX_GRAD_NORM="$(python - <<'PY'
import json, os
cfg=json.load(open(os.environ["CFG_SRC"]))
print(cfg.get("max_grad_norm", 1.0))
PY
)"

INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
GUIDANCE="${GUIDANCE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"
DTYPE="${DTYPE:-bf16}"

MAX_VIDEOS="${MAX_VIDEOS:-100}"
SEED="${SEED:-42}"

CMD="python ${PROJECT_ROOT}/delta_experiment/scripts/run_delta_a.py \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUT_DIR} \
  --reference-results-json ${BASELINE_RESULTS_JSON} \
  --delta-steps ${DELTA_STEPS} \
  --delta-lr ${DELTA_LR} \
  --delta-l2 ${DELTA_L2} \
  --warmup-steps ${WARMUP_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --max-grad-norm ${MAX_GRAD_NORM} \
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
echo "Done: Benchmark δ-A"
echo "Output: ${OUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="

