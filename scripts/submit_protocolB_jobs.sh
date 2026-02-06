#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/scratch/wc3013/open-sora-v2.0-experiment"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_100}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
MEMORY="${MEMORY:-256G}"
GPU_REQ="${GPU_REQ:-gpu:h200:1}"
MAX_VIDEOS="${MAX_VIDEOS:-100}"
STRATIFIED="${STRATIFIED:-true}"
SEED="${SEED:-42}"
BEST_OF="${BEST_OF:-5}"

AUG_ENABLED="${AUG_ENABLED:-true}"
AUG_FLIP="${AUG_FLIP:-true}"
AUG_ROTATE_DEG="${AUG_ROTATE_DEG:-10}"
AUG_ROTATE_RANDOM_MIN="${AUG_ROTATE_RANDOM_MIN:-5}"
AUG_ROTATE_RANDOM_MAX="${AUG_ROTATE_RANDOM_MAX:-15}"
AUG_ROTATE_RANDOM_COUNT="${AUG_ROTATE_RANDOM_COUNT:-2}"
AUG_ROTATE_RANDOM_STEP="${AUG_ROTATE_RANDOM_STEP:-1}"
AUG_ROTATE_ZOOM="${AUG_ROTATE_ZOOM:-true}"
AUG_SPEED_FACTORS="${AUG_SPEED_FACTORS:-0.5,2.0}"

submit() {
  local export_kv="$1"
  local script="$2"
  sbatch --account="${ACCOUNT}" --gres="${GPU_REQ}" --mem="${MEMORY}" --time="${TIME_LIMIT}" \
    --export="${export_kv}" \
    "${script}"
}

# -----------------------
# Baselines
# -----------------------
submit "BASELINE_DATA_DIR=${DATA_DIR},BASELINE_OUTPUT_DIR=${PROJECT_ROOT}/lora_experiment/results/baseline_c2,BASELINE_MAX_VIDEOS=${MAX_VIDEOS},BASELINE_STRATIFIED=${STRATIFIED},BASELINE_SEED=${SEED},BASELINE_RESTART=true,BASELINE_COND_FRAMES=2,BASELINE_COND_START=8,BASELINE_GT_FRAMES=16,BASELINE_GT_START=10" \
  "${PROJECT_ROOT}/lora_experiment/scripts/generate_baseline.sbatch"

submit "BASELINE_DATA_DIR=${DATA_DIR},BASELINE_OUTPUT_DIR=${PROJECT_ROOT}/lora_experiment/results/baseline_c10,BASELINE_MAX_VIDEOS=${MAX_VIDEOS},BASELINE_STRATIFIED=${STRATIFIED},BASELINE_SEED=${SEED},BASELINE_RESTART=true,BASELINE_COND_FRAMES=10,BASELINE_COND_START=0,BASELINE_GT_FRAMES=16,BASELINE_GT_START=10" \
  "${PROJECT_ROOT}/lora_experiment/scripts/generate_baseline.sbatch"

# -----------------------
# TTA configs
# -----------------------
export COMMON_TTA_ENV="DATA_DIR=${DATA_DIR},MAX_VIDEOS=${MAX_VIDEOS},STRATIFIED=${STRATIFIED},SEED=${SEED},BEST_OF=${BEST_OF},AUG_ENABLED=${AUG_ENABLED},AUG_FLIP=${AUG_FLIP},AUG_ROTATE_DEG=${AUG_ROTATE_DEG},AUG_ROTATE_RANDOM_MIN=${AUG_ROTATE_RANDOM_MIN},AUG_ROTATE_RANDOM_MAX=${AUG_ROTATE_RANDOM_MAX},AUG_ROTATE_RANDOM_COUNT=${AUG_ROTATE_RANDOM_COUNT},AUG_ROTATE_RANDOM_STEP=${AUG_ROTATE_RANDOM_STEP},AUG_ROTATE_ZOOM=${AUG_ROTATE_ZOOM},AUG_SPEED_FACTORS=${AUG_SPEED_FACTORS}"

# (c) TTA10, cond2
TTA10_C2="TTA_TRAIN_FRAMES=10,TTA_TRAIN_START=0,COND_FRAMES=2,COND_START=8,GT_FRAMES=16,GT_START=10"
# (d) TTA10, cond10
TTA10_C10="TTA_TRAIN_FRAMES=10,TTA_TRAIN_START=0,COND_FRAMES=10,COND_START=0,GT_FRAMES=16,GT_START=10"
# (e) TTA2, cond2
TTA2_C2="TTA_TRAIN_FRAMES=2,TTA_TRAIN_START=0,COND_FRAMES=2,COND_START=0,GT_FRAMES=16,GT_START=10"

# -----------------------
# Full TTA (3 jobs)
# -----------------------
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/lora_experiment/results/full_tta_t10_c2,RESTART=true,${TTA10_C2}" \
  "${PROJECT_ROOT}/lora_experiment/scripts/run_full_tta.sbatch"
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/lora_experiment/results/full_tta_t10_c10,RESTART=true,${TTA10_C10}" \
  "${PROJECT_ROOT}/lora_experiment/scripts/run_full_tta.sbatch"
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/lora_experiment/results/full_tta_t2_c2,RESTART=true,${TTA2_C2}" \
  "${PROJECT_ROOT}/lora_experiment/scripts/run_full_tta.sbatch"

# -----------------------
# LoRA r=8 TTA (3 jobs)
# -----------------------
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/lora_experiment/results/lora_r8_t10_c2,LORA_RANK=8,RESTART=true,${TTA10_C2}" \
  "${PROJECT_ROOT}/lora_experiment/scripts/run_lora_tta.sbatch"
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/lora_experiment/results/lora_r8_t10_c10,LORA_RANK=8,RESTART=true,${TTA10_C10}" \
  "${PROJECT_ROOT}/lora_experiment/scripts/run_lora_tta.sbatch"
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/lora_experiment/results/lora_r8_t2_c2,LORA_RANK=8,RESTART=true,${TTA2_C2}" \
  "${PROJECT_ROOT}/lora_experiment/scripts/run_lora_tta.sbatch"

# -----------------------
# Delta A (3 jobs)
# -----------------------
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/delta_experiment/results/deltaA_t10_c2,${TTA10_C2}" \
  "${PROJECT_ROOT}/delta_experiment/sbatch/run_delta_a.sbatch"
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/delta_experiment/results/deltaA_t10_c10,${TTA10_C10}" \
  "${PROJECT_ROOT}/delta_experiment/sbatch/run_delta_a.sbatch"
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/delta_experiment/results/deltaA_t2_c2,${TTA2_C2}" \
  "${PROJECT_ROOT}/delta_experiment/sbatch/run_delta_a.sbatch"

# -----------------------
# Delta B (3 jobs)
# -----------------------
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/delta_experiment/results/deltaB_t10_c2,${TTA10_C2}" \
  "${PROJECT_ROOT}/delta_experiment/sbatch/run_delta_b.sbatch"
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/delta_experiment/results/deltaB_t10_c10,${TTA10_C10}" \
  "${PROJECT_ROOT}/delta_experiment/sbatch/run_delta_b.sbatch"
submit "${COMMON_TTA_ENV},OUT_DIR=${PROJECT_ROOT}/delta_experiment/results/deltaB_t2_c2,${TTA2_C2}" \
  "${PROJECT_ROOT}/delta_experiment/sbatch/run_delta_b.sbatch"

echo "Submitted Protocol-B baseline + TTA jobs."
