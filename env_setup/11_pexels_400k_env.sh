#!/bin/bash
# Shared dataset paths for Pexels-400k 100-video subset.

export PEXELS_100_DIR="${PEXELS_100_DIR:-/scratch/wc3013/open-sora-v2.0-experiment/datasets/pexels_100}"
export PEXELS_100_METADATA="${PEXELS_100_METADATA:-${PEXELS_100_DIR}/metadata.csv}"
export PEXELS_100_CSV="${PEXELS_100_CSV:-${PEXELS_100_DIR}/pexels_100_necessary.csv}"
