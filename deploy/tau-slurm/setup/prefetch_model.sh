#!/bin/sh
# Download one model's weights into HF_HOME before any GPU job needs them.
# Run from the login node: compute nodes do have outbound internet (verified),
# but spending GPU allocation on a multi-GB download is wasteful, and a
# preempted job restarts the transfer from scratch.
#
# Models are named by HF id only, never by path -- serve_vllm.sbatch resolves
# the same id through HF's standard cache layout
# ($HF_HOME/hub/models--<org>--<name>/snapshots/...), so this script and that
# one never need to agree on a path of our own.
#
# Usage:
#   deploy/tau-slurm/setup/prefetch_model.sh [--root DIR] <hf-model-id>
# Example:
#   deploy/tau-slurm/setup/prefetch_model.sh Qwen/Qwen2.5-Coder-1.5B-Instruct
#
# Requires install_vllm_env.sh to have been run first -- it provides the
# huggingface_hub CLI used here, so weights and the code that serves them
# come from the same, already-verified environment.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
. "$SCRIPT_DIR/../lib/common.sh"

model_id=""
root_arg=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --root)
            root_arg="--root $2"
            shift 2
            ;;
        --root=*)
            root_arg="$1"
            shift
            ;;
        -*)
            echo "foresight: unknown option $1" >&2
            exit 1
            ;;
        *)
            if [ -n "$model_id" ]; then
                echo "foresight: unexpected extra argument: $1" >&2
                exit 1
            fi
            model_id="$1"
            shift
            ;;
    esac
done

if [ -z "$model_id" ]; then
    echo "usage: $0 [--root DIR] <hf-model-id>" >&2
    echo "example: $0 Qwen/Qwen2.5-Coder-1.5B-Instruct" >&2
    exit 1
fi

# shellcheck disable=SC2086
foresight_resolve_root $root_arg

env_prefix="$FORESIGHT_ROOT/conda-envs/foresight-vllm"
if [ ! -x "$env_prefix/bin/python" ]; then
    echo "foresight: no env at $env_prefix -- run install_vllm_env.sh first" >&2
    exit 1
fi

echo "foresight: downloading $model_id into HF_HOME=$HF_HOME ..." >&2
if conda run -p "$env_prefix" sh -c 'command -v hf' >/dev/null 2>&1; then
    conda run -p "$env_prefix" hf download "$model_id"
else
    conda run -p "$env_prefix" huggingface-cli download "$model_id"
fi

snap_dir="$HF_HOME/hub/models--$(echo "$model_id" | sed 's#/#--#')/snapshots"
if [ -d "$snap_dir" ] && [ -n "$(ls -A "$snap_dir" 2>/dev/null)" ]; then
    echo "foresight: snapshot ready under $snap_dir" >&2
else
    echo "foresight: WARNING -- expected snapshot dir not found or empty: $snap_dir" >&2
    exit 1
fi

echo "foresight: HF_HOME usage:" >&2
df -h "$HF_HOME" | tail -1 >&2
