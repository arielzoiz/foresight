#!/bin/sh
# One-time setup: a conda env with vLLM, isolated under FORESIGHT_ROOT so it
# never touches the 2 GB $HOME quota. Safe to re-run; it recreates the env.
#
# Usage:
#   deploy/tau-slurm/setup/install_vllm_env.sh [--root DIR]
#
# Run this on the login node -- it needs outbound internet for conda/pip.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
. "$SCRIPT_DIR/../lib/common.sh"

foresight_resolve_root "$@"

ENV_PREFIX="$FORESIGHT_ROOT/conda-envs/foresight-vllm"
PIP_CACHE_DIR="$FORESIGHT_ROOT/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

echo "foresight: creating conda env at $ENV_PREFIX ..." >&2
conda create -y -p "$ENV_PREFIX" python=3.12

echo "foresight: installing vllm (this can take a few minutes) ..." >&2
conda run -p "$ENV_PREFIX" pip install --cache-dir "$PIP_CACHE_DIR" vllm

echo "foresight: done. Activate with:" >&2
echo "  conda activate $ENV_PREFIX" >&2
