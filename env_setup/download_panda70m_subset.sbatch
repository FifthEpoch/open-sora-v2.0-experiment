#!/bin/bash
#SBATCH --job-name=panda100_dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --output=env_setup/slurm_log/panda100_%j.out
#SBATCH --error=env_setup/slurm_log/panda100_%j.err

set -euo pipefail

echo "=============================================================================="
echo "Download Panda-70M 100-video subset + build manifest"
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
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
eval "$(conda shell.bash hook)"
conda activate opensora20
set -u
export PATH="${CONDA_PREFIX}/bin:${PATH}"

source "${PROJECT_ROOT}/env_setup/12_panda70m_env.sh"

mkdir -p "${PANDA_100_DIR}"
cd "${PROJECT_ROOT}"

pip install requests pandarallel yt-dlp --quiet
conda install -c conda-forge nodejs ffmpeg av -y --quiet
pip install gdown --quiet

META_PATH="${PANDA_META_PATH:-}"
META_URL="${PANDA_META_URL:-}"

if [ -z "${META_PATH}" ]; then
  if [ -n "${META_URL}" ]; then
    META_PATH="${PANDA_100_DIR}/panda70m_metadata.jsonl"
    echo "Downloading metadata from ${META_URL}"
    if echo "${META_URL}" | grep -q "drive.google.com"; then
      gdown --fuzzy "${META_URL}" -O "${META_PATH}"
    else
      curl -L "${META_URL}" -o "${META_PATH}"
    fi
  else
    echo "ERROR: Set PANDA_META_PATH or PANDA_META_URL to a Panda-70M metadata file."
    exit 1
  fi
fi

python scripts/cnv/download_panda70m_subset.py \
  --meta-path "${META_PATH}" \
  --out-dir "${PANDA_100_DIR}" \
  --num-videos "${NUM_VIDEOS:-100}" \
  --seed "${SEED:-42}" \
  --min-duration "${MIN_DURATION:-4}" \
  --max-duration "${MAX_DURATION:-60}" \
  --min-bytes "${MIN_BYTES:-1000000}" \
  --resize-height "${RESIZE_HEIGHT:-256}" \
  --resize-width "${RESIZE_WIDTH:-464}"

echo "Building Open-Sora manifest..."
python scripts/cnv/meta.py \
  --input "${PANDA_100_METADATA}" \
  --output "${PANDA_100_CSV}" \
  --num_workers "${NUM_WORKERS:-0}"

echo "=============================================================================="
echo "Done: Panda-70M 100-video subset"
echo "Dataset: ${PANDA_100_DIR}"
echo "Manifest: ${PANDA_100_CSV}"
echo "End time: $(date)"
echo "=============================================================================="
