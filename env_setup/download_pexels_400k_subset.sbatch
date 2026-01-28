#!/bin/bash
#SBATCH --job-name=pexels100_dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --output=env_setup/slurm_log/pexels100_%j.out
#SBATCH --error=env_setup/slurm_log/pexels100_%j.err

set -euo pipefail

echo "=============================================================================="
echo "Download Pexels-400k 100-video subset + build manifest"
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

source "${PROJECT_ROOT}/env_setup/11_pexels_400k_env.sh"

mkdir -p "${PEXELS_100_DIR}"
cd "${PROJECT_ROOT}"

pip install datasets requests pandarallel --quiet

python scripts/cnv/download_pexels400k_subset.py \
  --out-dir "${PEXELS_100_DIR}" \
  --num-videos "${NUM_VIDEOS:-100}" \
  --seed "${SEED:-42}" \
  --min-duration "${MIN_DURATION:-4}" \
  --max-duration "${MAX_DURATION:-60}"

echo "Building Open-Sora manifest..."
python scripts/cnv/meta.py \
  --input "${PEXELS_100_METADATA}" \
  --output "${PEXELS_100_CSV}" \
  --num_workers "${NUM_WORKERS:-0}"

echo "=============================================================================="
echo "Done: Pexels-400k 100-video subset"
echo "Dataset: ${PEXELS_100_DIR}"
echo "Manifest: ${PEXELS_100_CSV}"
echo "End time: $(date)"
echo "=============================================================================="
