#!/bin/bash
#SBATCH --job-name=full_tta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/logs/full_tta_%j.out
#SBATCH --error=lora_experiment/logs/full_tta_%j.err

set -euo pipefail

echo "=============================================================================="
echo "Full-Model Test-Time Adaptation for Open-Sora v2.0"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
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
source "${PROJECT_ROOT}/env_setup/11_pexels_400k_env.sh"
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

mkdir -p "${PROJECT_ROOT}/lora_experiment/logs"

DATA_DIR="${PEXELS_100_DIR}"

# Full-model hyperparameters (override via env if desired)
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_STEPS="${NUM_STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# Inference parameters
INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
GUIDANCE="${GUIDANCE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"

# Other parameters
MAX_VIDEOS="${MAX_VIDEOS:-100}"
STRATIFIED="${STRATIFIED:-true}"
SEED="${SEED:-42}"

# Augmentation parameters (TTA only)
AUG_ENABLED="${AUG_ENABLED:-false}"
AUG_FLIP="${AUG_FLIP:-true}"
AUG_ROTATE_DEG="${AUG_ROTATE_DEG:-5}"
AUG_SPEED_FACTORS="${AUG_SPEED_FACTORS:-0.5,2.0}"

OUTPUT_DIR="${PROJECT_ROOT}/lora_experiment/results/full_tta_lr${LEARNING_RATE}_${NUM_STEPS}steps"
OUTPUT_DIR="${FULL_TTA_OUTPUT_DIR:-$OUTPUT_DIR}"
DATA_DIR="${FULL_TTA_DATA_DIR:-$DATA_DIR}"

echo ""
echo "Configuration:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Training steps: ${NUM_STEPS}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo "  Max grad norm: ${MAX_GRAD_NORM}"
echo "  Inference steps: ${INFERENCE_STEPS}"
echo "  Guidance: ${GUIDANCE}"
echo "  Image guidance: ${GUIDANCE_IMG}"
echo "  Max videos: ${MAX_VIDEOS}"
echo "  Stratified: ${STRATIFIED}"
echo "  Seed: ${SEED}"
echo "  Aug enabled: ${AUG_ENABLED}"
echo "  Aug flip: ${AUG_FLIP}"
echo "  Aug rotate deg: ${AUG_ROTATE_DEG}"
echo "  Aug speed factors: ${AUG_SPEED_FACTORS}"
echo ""

CMD="python ${PROJECT_ROOT}/lora_experiment/scripts/run_full_tta.py \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --learning-rate ${LEARNING_RATE} \
  --num-steps ${NUM_STEPS} \
  --warmup-steps ${WARMUP_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --max-grad-norm ${MAX_GRAD_NORM} \
  --inference-steps ${INFERENCE_STEPS} \
  --guidance ${GUIDANCE} \
  --guidance-img ${GUIDANCE_IMG} \
  --seed ${SEED} \
  --reference-results-json ${PROJECT_ROOT}/lora_experiment/results/baseline/results.json"

if [ -n "${MAX_VIDEOS}" ]; then
    CMD="${CMD} --max-videos ${MAX_VIDEOS}"
fi

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

echo ""
echo "=============================================================================="
echo "Full-model TTA Complete!"
echo "=============================================================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="
