#!/bin/bash
# Environment setup script to redirect all installations and caches to /scratch
# This prevents hitting /home directory quotas on clusters

set -eo pipefail

# Base scratch directory
SCRATCH_BASE="/scratch/wc3013"

# Create organized directory structure under scratch
mkdir -p "${SCRATCH_BASE}"/{conda-envs,conda-pkgs,py-cache/{pip,torch/extensions,models,triton,torch-inductor,colossalaJit,python},hf-cache/{datasets,hub,transformers},tmp,wandb,tensorboard,wheels}

# Set conda environment and package directories
export CONDA_ENVS_DIRS="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"

# Set cache directories for various tools
export XDG_CACHE_HOME="${SCRATCH_BASE}/py-cache"

# PyTorch cache directories
export TORCH_HOME="${SCRATCH_BASE}/py-cache/models"
export TORCH_EXTENSIONS_DIR="${SCRATCH_BASE}/py-cache/torch/extensions"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/py-cache/torch-inductor"

# Triton (used by CUDA kernels in PyTorch)
export TRITON_CACHE_DIR="${SCRATCH_BASE}/py-cache/triton"

# ColossalAI extensions cache
export COLOSSALAI_EXTENSIONS_JIT_CACHE_PATH="${SCRATCH_BASE}/py-cache/colossalaJit"

# Hugging Face caches
export HF_HOME="${SCRATCH_BASE}/hf-cache"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/hf-cache/datasets"
export HF_HUB_CACHE="${SCRATCH_BASE}/hf-cache/hub"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/hf-cache/transformers"

# Pip cache
export PIP_CACHE_DIR="${SCRATCH_BASE}/py-cache/pip"
# Disable pip cache to avoid cross-device link errors
export PIP_NO_CACHE_DIR=1

# Temporary directory for compilations
export TMPDIR="${SCRATCH_BASE}/tmp"
export TMP="${SCRATCH_BASE}/tmp"
export TEMP="${SCRATCH_BASE}/tmp"

# Additional Python caches
export PYTHONPYCACHEPREFIX="${SCRATCH_BASE}/py-cache/python"

# WANDB (Weights & Biases) cache
export WANDB_CACHE_DIR="${SCRATCH_BASE}/wandb"

# TensorBoard logs
export TENSORBOARD_LOGDIR="${SCRATCH_BASE}/tensorboard"

# Tell user what was set
echo "=================================="
echo "Environment Variables Set"
echo "=================================="
echo "Base directory: ${SCRATCH_BASE}"
echo ""
echo "CONDA_ENVS_DIRS: ${CONDA_ENVS_DIRS}"
echo "CONDA_PKGS_DIRS: ${CONDA_PKGS_DIRS}"
echo "XDG_CACHE_HOME: ${XDG_CACHE_HOME}"
echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"
echo "TMPDIR: ${TMPDIR}"
echo "=================================="
echo ""
echo "All installations and caches will be redirected to /scratch"
echo "This script should be sourced before running setup scripts:"
echo "  source env_setup/00_set_scratch_env.sh"

