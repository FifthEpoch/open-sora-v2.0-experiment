#!/bin/bash
#SBATCH --job-name=setup_opensora20_env
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=slurm_log/setup_env_%j.out
#SBATCH --error=slurm_log/setup_env_%j.err

# ==============================================================================
# Comprehensive Open-Sora v2.0 Environment Setup
# ==============================================================================
# This script creates and configures the complete environment for Open-Sora 2.0
# 
# Key differences from v1.3:
#   - PyTorch 2.4.0 (upgraded from 2.2.2)
#   - xformers 0.0.27.post2 (upgraded from 0.0.25.post1)
#   - ColossalAI 0.4.4+ (upgraded)
#   - New dependency: liger-kernel 0.5.2
#   - av 13.1.0 (upgraded from 12.x)
#   - No bitsandbytes (not needed for v2.0)
#
# Safe to run multiple times - will detect existing environment and verify/fix
# ==============================================================================

set -euo pipefail

# Prevent user site-packages from interfering with conda environment
export PYTHONNOUSERSITE=1

echo "=============================================================================="
echo "Open-Sora v2.0 Environment Setup"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "=============================================================================="
echo ""

# ==============================================================================
# Configuration
# ==============================================================================
SCRATCH_BASE="/scratch/wc3013"
ENV_PATH="${SCRATCH_BASE}/conda-envs/opensora20"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v2.0-experiment"

# Expected versions
EXPECTED_PYTHON="3.10"
EXPECTED_TORCH="2.4.0+cu121"
EXPECTED_XFORMERS="0.0.27.post2"
EXPECTED_COLOSSALAI="0.4.4"

# ==============================================================================
# Setup Scratch Environment Variables
# ==============================================================================
echo "Setting up scratch environment variables..."
source "${PROJECT_ROOT}/env_setup/00_set_scratch_env.sh"

# Create slurm_log directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/env_setup/slurm_log"

# ==============================================================================
# Load Conda
# ==============================================================================
echo ""
echo "Loading conda module..."
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Accept Conda Terms of Service (required for non-interactive use)
echo ""
echo "Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda tos accept --override-channels --channel conda-forge || true

# Configure conda to use scratch directories
conda config --prepend envs_dirs "${SCRATCH_BASE}/conda-envs" 2>/dev/null || true
conda config --prepend pkgs_dirs "${SCRATCH_BASE}/conda-pkgs" 2>/dev/null || true

# ==============================================================================
# Check/Create Environment
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 1: Environment Detection"
echo "=============================================================================="

if [ -d "${ENV_PATH}" ]; then
    echo "Environment exists at: ${ENV_PATH}"
    echo "  Will verify and fix packages if needed..."
    ENV_EXISTS=true
else
    echo "Environment not found at: ${ENV_PATH}"
    echo "  Will create new environment..."
    ENV_EXISTS=false
fi

if [ "$ENV_EXISTS" = false ]; then
    echo ""
    echo "Creating new conda environment with Python 3.10..."
    conda create -y -p "${ENV_PATH}" python=3.10
    echo "Environment created!"
fi

# Activate environment
echo ""
echo "Activating environment..."
conda activate "${ENV_PATH}"

# Verify activation
if [ "$CONDA_PREFIX" != "${ENV_PATH}" ]; then
    echo "ERROR: Failed to activate environment!"
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    echo "Expected: ${ENV_PATH}"
    exit 1
fi

# Explicitly ensure conda env's bin is first in PATH
export PATH="${ENV_PATH}/bin:${PATH}"

echo "Environment activated"
echo "  Python: $(python --version)"
echo "  Location: ${CONDA_PREFIX}"
echo "  Python executable: $(which python)"

# Critical verification: ensure we're using the environment's Python
PYTHON_PATH=$(which python)
if [[ ! "$PYTHON_PATH" =~ ^${ENV_PATH} ]]; then
    echo "ERROR: Python executable is not from the conda environment!"
    echo "  Expected: ${ENV_PATH}/bin/python"
    echo "  Got: ${PYTHON_PATH}"
    exit 1
fi
echo "Using environment's Python: ${PYTHON_PATH}"

# ==============================================================================
# Check Python Version
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 2: Python Version Check"
echo "=============================================================================="

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Current Python version: ${PYTHON_VERSION}"

if [[ ! "$PYTHON_VERSION" =~ ^3\.1[0-9] ]]; then
    echo "Python 3.10+ required (found ${PYTHON_VERSION})"
    echo "Upgrading Python to 3.10..."
    conda install -y python=3.10
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "New Python version: ${PYTHON_VERSION}"
fi

echo "Python version OK: ${PYTHON_VERSION}"

# Upgrade pip
python -m pip install -U pip setuptools wheel

# ==============================================================================
# Install Core ML Stack (PyTorch 2.4.0 + xformers)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 3: Core ML Stack Installation"
echo "=============================================================================="
echo "Installing PyTorch 2.4.0 + xformers 0.0.27.post2..."
echo ""

# Step 3.1: Install NumPy first (v2.0 can use NumPy 2.x, but 1.x is safer)
echo "Step 3.1: Installing NumPy..."
pip install 'numpy>=1.26'

# Step 3.2: Install PyTorch 2.4.0 with CUDA 12.1
echo ""
echo "Step 3.2: Installing PyTorch 2.4.0..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 \
    torchvision==0.19.0

# Step 3.3: Install xformers
echo ""
echo "Step 3.3: Installing xformers 0.0.27.post2..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
    xformers==0.0.27.post2

echo ""
echo "Verifying core ML stack..."
python -c "
import numpy as np
import torch
import torchvision
import xformers

print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'xformers: {xformers.__version__}')

# Verify versions
assert '2.4.0' in torch.__version__, f'PyTorch must be 2.4.0, got {torch.__version__}'
assert xformers.__version__ == '0.0.27.post2', f'xformers must be 0.0.27.post2, got {xformers.__version__}'

print('Core ML stack versions verified!')
"

echo "Core ML stack installed and verified"

# ==============================================================================
# Install Flash Attention (Optional - may fail on some systems)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 4: Flash Attention Installation (Optional)"
echo "=============================================================================="
echo "Installing flash-attn build dependencies..."
echo ""

# Install build dependencies for flash-attn
pip install psutil packaging ninja

echo ""
echo "Attempting to install flash-attn (this may take 10-20 minutes to compile)..."
echo "Note: flash-attn is optional - inference will still work with xformers if this fails"
echo ""

# Try to install flash-attn, but don't fail the script if it doesn't work
if pip install flash-attn --no-build-isolation; then
    echo ""
    echo "Verifying flash-attn..."
    python -c "
import flash_attn
print(f'flash-attn: {flash_attn.__version__}')
print('flash-attn installed successfully!')
"
    echo "flash-attn installed and verified"
else
    echo ""
    echo "WARNING: flash-attn installation failed (this is OK)"
    echo "The model will use xformers for attention instead"
    echo "You can try installing flash-attn manually later if needed"
fi

# ==============================================================================
# Install PyAV via conda
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 5: PyAV Installation"
echo "=============================================================================="
echo "Installing PyAV via conda for better ffmpeg handling..."
echo ""

# Note: av==13.1.0 is specified in requirements.txt
# conda-forge usually has the latest version
conda install -y av -c conda-forge

echo ""
echo "Verifying PyAV..."
python -c "
import av
print(f'PyAV: {av.__version__}')
print('PyAV installed successfully!')
"

echo "PyAV installed and verified"

# ==============================================================================
# Install ColossalAI and liger-kernel
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 6: ColossalAI and liger-kernel Installation"
echo "=============================================================================="
echo "Installing ColossalAI >= 0.4.4 and liger-kernel..."
echo ""

pip install 'colossalai>=0.4.4'
pip install 'liger-kernel==0.5.2'

echo ""
echo "Verifying ColossalAI and liger-kernel..."
python -c "
import colossalai
import liger_kernel
print(f'ColossalAI: {colossalai.__version__}')
# liger_kernel doesn't expose __version__, just verify import works
print('liger-kernel: imported successfully (no __version__ attribute)')
print('ColossalAI and liger-kernel installed successfully!')
"

echo "ColossalAI and liger-kernel installed and verified"

# ==============================================================================
# Install Open-Sora v2.0 Requirements
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 7: Open-Sora v2.0 Requirements"
echo "=============================================================================="

cd "${PROJECT_ROOT}"

echo "Installing Open-Sora v2.0 requirements..."
pip install -r requirements.txt

echo ""
echo "Installing Open-Sora v2.0 in development mode..."
pip install -v -e .

echo "Open-Sora v2.0 requirements installed"

# ==============================================================================
# Install Additional Training Requirements
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 8: Additional Training Requirements"
echo "=============================================================================="

# TensorNVMe is optional - requires system libraries (liburing, libaio)
echo "Attempting to install TensorNVMe for async checkpoint saving..."
echo "Note: TensorNVMe is optional - inference will work without it"
echo ""

if pip install git+https://github.com/hpcaitech/TensorNVMe.git 2>/dev/null; then
    echo "TensorNVMe installed successfully"
else
    echo "WARNING: TensorNVMe installation failed (requires liburing/libaio system libs)"
    echo "This is optional - training/inference will still work without it"
fi

echo ""
echo "Installing additional packages..."
pip install pandarallel huggingface_hub[cli] datasets

echo "Additional training requirements installed"

# ==============================================================================
# Install Evaluation Requirements (LPIPS, etc.)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 9: Evaluation Requirements"
echo "=============================================================================="
echo "Installing evaluation packages..."
echo ""

pip install lpips scikit-image scikit-learn decord imageio

echo "Evaluation requirements installed"

# ==============================================================================
# Final Verification
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 10: Final Verification"
echo "=============================================================================="
echo "Running comprehensive verification..."
echo ""

python -c "
import sys
import numpy as np
import torch
import torchvision
import xformers
import colossalai
import liger_kernel
import av
import pandas as pd

print('=' * 70)
print('ENVIRONMENT VERIFICATION')
print('=' * 70)
print()
print(f'Python: {sys.version}')
print(f'  Location: {sys.executable}')
print()

# Check critical versions
print('Critical Package Versions:')
print(f'  NumPy: {np.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  xformers: {xformers.__version__}')
print(f'  ColossalAI: {colossalai.__version__}')
print(f'  liger-kernel: imported successfully')  # no __version__ attribute
print(f'  PyAV: {av.__version__}')
print(f'  Pandas: {pd.__version__}')
print()

# Check flash-attn
try:
    import flash_attn
    print(f'  flash-attn: {flash_attn.__version__}')
except ImportError:
    print('  flash-attn: NOT INSTALLED (optional)')

print()

# Check GPU
print('GPU Check:')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  GPU name: {torch.cuda.get_device_name(0)}')
print()

# Verify assertions
print('Version Assertions:')
errors = []

if '2.4.0' not in torch.__version__:
    errors.append(f'  PyTorch must be 2.4.0, got {torch.__version__}')
else:
    print(f'  PyTorch 2.4.0: {torch.__version__}')

if xformers.__version__ != '0.0.27.post2':
    errors.append(f'  xformers must be 0.0.27.post2, got {xformers.__version__}')
else:
    print(f'  xformers 0.0.27.post2: {xformers.__version__}')

# Check ColossalAI version >= 0.4.4
from packaging import version
if version.parse(colossalai.__version__) < version.parse('0.4.4'):
    errors.append(f'  ColossalAI must be >= 0.4.4, got {colossalai.__version__}')
else:
    print(f'  ColossalAI >= 0.4.4: {colossalai.__version__}')

if not sys.version_info >= (3, 10):
    errors.append(f'  Python must be 3.10+, got {sys.version_info.major}.{sys.version_info.minor}')
else:
    print(f'  Python 3.10+: {sys.version_info.major}.{sys.version_info.minor}')

print()

if errors:
    print('ERRORS FOUND:')
    for error in errors:
        print(error)
    sys.exit(1)
else:
    print('All version assertions passed!')

# Test opensora import
print()
print('Testing opensora import...')
sys.path.insert(0, '${PROJECT_ROOT}')
try:
    import opensora
    print(f'  opensora imports from: {opensora.__file__}')
except Exception as e:
    print(f'  Warning: opensora import failed: {e}')
    print(f'  Run: pip install -e . in the project root')

print()
print('=' * 70)
print('ENVIRONMENT SETUP COMPLETE AND VERIFIED!')
print('=' * 70)
"

VERIFICATION_EXIT=$?

if [ $VERIFICATION_EXIT -ne 0 ]; then
    echo ""
    echo "Verification failed! Please check errors above."
    exit 1
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Setup Complete!"
echo "=============================================================================="
echo "Environment location: ${ENV_PATH}"
echo "Project root: ${PROJECT_ROOT}"
echo ""
echo "To use this environment in future jobs:"
echo "  conda activate ${ENV_PATH}"
echo ""
echo "Next steps:"
echo "  1. Download model checkpoints:"
echo "     sbatch env_setup/download_checkpoints/download_checkpoints.sbatch"
echo ""
echo "  2. Test inference:"
echo "     sbatch env_setup/02_test_inference.sbatch"
echo ""
echo "End time: $(date)"
echo "=============================================================================="

