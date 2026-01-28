#!/bin/bash
# Shared dataset paths for Panda-70M 100-video subset.

export PANDA_100_DIR="${PANDA_100_DIR:-/scratch/wc3013/open-sora-v2.0-experiment/datasets/panda_100}"
export PANDA_100_METADATA="${PANDA_100_METADATA:-${PANDA_100_DIR}/metadata.csv}"
export PANDA_100_CSV="${PANDA_100_CSV:-${PANDA_100_DIR}/panda_100_necessary.csv}"
