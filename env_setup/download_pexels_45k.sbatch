#!/bin/bash
#SBATCH --job-name=pexels45k_dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=env_setup/slurm_log/pexels45k_%j.out
#SBATCH --error=env_setup/slurm_log/pexels45k_%j.err

set -euo pipefail

echo "=============================================================================="
echo "Download Open-Sora Pexels 45K dataset"
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

# Load dataset env paths
source "${PROJECT_ROOT}/env_setup/10_pexels_45k_env.sh"

echo "Dataset dir: ${PEXELS_45K_DIR}"
echo "Manifest csv: ${PEXELS_45K_CSV}"

mkdir -p "${PEXELS_45K_DIR}"
cd "${PEXELS_45K_DIR}"

# Download from HF (requires huggingface-cli login if private)
if ! command -v huggingface-cli >/dev/null 2>&1; then
  pip install "huggingface_hub[cli]" --quiet
fi

echo "Downloading dataset..."
huggingface-cli download --repo-type dataset hpcai-tech/open-sora-pexels-45k --local-dir "${PEXELS_45K_DIR}"

echo "Extracting shards..."
if ls "${PEXELS_45K_DIR}"/tar/pexels_45k.tar.* >/dev/null 2>&1; then
  cat "${PEXELS_45K_DIR}"/tar/pexels_45k.tar.* > "${PEXELS_45K_DIR}"/pexels_45k.tar
  tar -xvf "${PEXELS_45K_DIR}"/pexels_45k.tar -C "${PEXELS_45K_DIR}"
fi

if [ -f "${PEXELS_45K_CSV}" ]; then
  echo "Found manifest: ${PEXELS_45K_CSV}"
else
  echo "ERROR: Missing ${PEXELS_45K_CSV}"
  exit 1
fi

echo "=============================================================================="
echo "Done: Pexels 45K download"
echo "End time: $(date)"
echo "=============================================================================="
