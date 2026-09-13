#!/bin/sh
#SBATCH --job-name=foresight-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1
#SBATCH --time=180
#SBATCH --partition=killable
# NOT studentkillable: its Titan Xp (Pascal) GPUs cannot run the CUDA-13
# build `pip install vllm` currently resolves -- confirmed by direct
# testing, see launch.sh's header comment for the fuller story.
#
# Serves ONE model with vLLM and publishes its endpoint as a file, once the
# server actually answers -- not once the job starts running. Submitted by
# launch.sh, which overrides --partition/--account/--gpus/--time/--output/
# --error on the sbatch command line (those win over the defaults above) and
# passes everything else via --export. Never submit this directly without
# reading deploy/tau-slurm/README.md first.
#
# Required environment (set by launch.sh via --export):
#   FORESIGHT_ROOT        working root (see lib/common.sh)
#   FORESIGHT_RUN_DIR      this run's directory; endpoint file goes here
#   FORESIGHT_ROLE          "target" or "aux" -- names the endpoint file
#   FORESIGHT_MODEL_ID     HF model id to serve (e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct)
#   FORESIGHT_ALIAS        --served-model-name alias(es); a single name in
#                           --split mode (e.g. "target-upstream"), or a
#                           space-separated pair in --single mode
#                           (e.g. "target-upstream aux-upstream") so one
#                           loaded model answers to both
#   FORESIGHT_PORT         port to bind
# Optional:
#   FORESIGHT_MAX_MODEL_LEN  passed to --max-model-len (default 32768)
#   FORESIGHT_TOOL_CALL_PARSER  passed to --tool-call-parser (default hermes,
#                                the format Qwen2.5-Instruct's chat template
#                                uses; set "" to disable tool calling entirely)
#   HF_HOME                  inherited if already set; else derived from FORESIGHT_ROOT

set -eu

: "${FORESIGHT_ROOT:?FORESIGHT_ROOT must be exported before submitting this job}"
: "${FORESIGHT_RUN_DIR:?FORESIGHT_RUN_DIR must be set}"
: "${FORESIGHT_ROLE:?FORESIGHT_ROLE must be set (target|aux)}"
: "${FORESIGHT_MODEL_ID:?FORESIGHT_MODEL_ID must be set (an HF model id)}"
: "${FORESIGHT_ALIAS:?FORESIGHT_ALIAS must be set (the --served-model-name alias)}"
: "${FORESIGHT_PORT:?FORESIGHT_PORT must be set}"
FORESIGHT_MAX_MODEL_LEN="${FORESIGHT_MAX_MODEL_LEN:-32768}"
FORESIGHT_TOOL_CALL_PARSER="${FORESIGHT_TOOL_CALL_PARSER-hermes}"
HF_HOME="${HF_HOME:-$FORESIGHT_ROOT/.cache/huggingface}"
export HF_HOME

case "$FORESIGHT_ROLE" in
    target|aux) ;;
    *)
        echo "foresight: FORESIGHT_ROLE must be target or aux, got: $FORESIGHT_ROLE" >&2
        exit 1
        ;;
esac

env_prefix="$FORESIGHT_ROOT/conda-envs/foresight-vllm"
if [ ! -x "$env_prefix/bin/python" ]; then
    echo "foresight: no env at $env_prefix -- run install_vllm_env.sh first" >&2
    exit 1
fi

# Some cluster nodes (observed: the DGX-1 rack, tesla_v100) ship a system
# libstdc++ too old for vllm's dependency chain (ImportError: CXXABI_1.3.15
# not found, pulled in via libicui18n through sqlite3). The conda env already
# carries a newer one as a side effect of its own dependencies -- point the
# loader at it explicitly rather than relying on whichever system one a given
# node happens to have.
export LD_LIBRARY_PATH="$env_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# flashinfer's JIT-compiled sampling kernel hits two more problems on this
# cluster: (1) the system /usr/local/cuda's nvcc is too old for a flag the
# JIT wrapper passes (--compress-mode=size); (2) even pointed at the newer
# nvcc pip installs alongside vllm (site-packages/nvidia/cuNN/), the JIT
# wrapper's own include-path construction truncates "python3.12" to
# "python3.1" (an upstream bug, not fixable from here) and fails with "CUDA
# compiler and CUDA toolkit headers are incompatible". Simplest fix: skip the
# JIT path entirely -- it is a performance optimization, not a correctness
# requirement, and moot for a single-GPU dense model with no tensor
# parallelism (the flag only affects the sampler; MoE/allreduce kernels this
# does not touch aren't relevant here either).
export VLLM_USE_FLASHINFER_SAMPLER=0

# Same story for nvcc: flashinfer JIT-compiles kernels at warm-up
# with flags (--compress-mode=size) that the system /usr/local/cuda's older
# nvcc (12.2 here) doesn't understand. pip pulls its own matching nvcc in as
# an nvidia-cuda-nvcc wheel (site-packages/nvidia/cuNN/) -- point CUDA_HOME
# and PATH at that instead of letting the JIT compiler fall through to the
# system one. Globbed rather than hardcoded to a CUDA major version, since
# that's whatever pip resolved at install_vllm_env.sh time, not a constant.
# Kept as a defensive fallback alongside VLLM_USE_FLASHINFER_SAMPLER=0 above,
# in case some other flashinfer JIT path still triggers.
cuda_pkg_home="$(find "$env_prefix"/lib/python*/site-packages/nvidia/cu[0-9]* -maxdepth 0 2>/dev/null | head -1)"
if [ -n "$cuda_pkg_home" ]; then
    export CUDA_HOME="$cuda_pkg_home"
    export PATH="$cuda_pkg_home/bin:$PATH"
    export LD_LIBRARY_PATH="$cuda_pkg_home/lib:$LD_LIBRARY_PATH"
fi

endpoint_file="$FORESIGHT_RUN_DIR/${FORESIGHT_ROLE}.endpoint"
rm -f "$endpoint_file"

vllm_pid=""
cleanup() {
    rm -f "$endpoint_file"
    [ -n "$vllm_pid" ] && kill "$vllm_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "foresight: [$FORESIGHT_ROLE] node=$(hostname) serving $FORESIGHT_MODEL_ID" \
     "as $FORESIGHT_ALIAS on port $FORESIGHT_PORT (HF_HOME=$HF_HOME)" >&2

# Without --enable-auto-tool-choice/--tool-call-parser, vLLM 400s on ANY
# request carrying a tools array -- which every message from a real agent
# harness does (it always advertises its bash tool). Observed directly: an
# opencode session pointed at a server started without these flags retried
# the same 400 forever, 1215 requests in 45 minutes, never making progress.
tool_call_args=""
if [ -n "$FORESIGHT_TOOL_CALL_PARSER" ]; then
    tool_call_args="--enable-auto-tool-choice --tool-call-parser $FORESIGHT_TOOL_CALL_PARSER"
fi

# FORESIGHT_ALIAS and tool_call_args are deliberately NOT quoted below:
# FORESIGHT_ALIAS may be "target-upstream aux-upstream" -- two space-separated
# names in --single mode -- so that vLLM's --served-model-name (which takes
# one or more values) registers both aliases against the one loaded model.
# tool_call_args is either empty (nothing to pass) or two flags.
# shellcheck disable=SC2086
conda run --no-capture-output -p "$env_prefix" \
    vllm serve "$FORESIGHT_MODEL_ID" \
        --host 0.0.0.0 \
        --port "$FORESIGHT_PORT" \
        --served-model-name $FORESIGHT_ALIAS \
        --max-model-len "$FORESIGHT_MAX_MODEL_LEN" \
        $tool_call_args \
    &
vllm_pid=$!

# Poll our own port rather than trusting "the process started" -- weight
# loading on a cold cache can take minutes, and a request sent too early is a
# connection-refused, not a useful error.
ready=0
i=0
while [ "$i" -lt 720 ]; do   # up to 60 min
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
        echo "foresight: [$FORESIGHT_ROLE] vllm exited before becoming ready" >&2
        exit 1
    fi
    if curl -fsS -m 5 "http://127.0.0.1:$FORESIGHT_PORT/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 5
    i=$((i + 1))
done

if [ "$ready" -ne 1 ]; then
    echo "foresight: [$FORESIGHT_ROLE] vllm did not become ready in time" >&2
    exit 1
fi

echo "http://$(hostname):$FORESIGHT_PORT/v1" > "$endpoint_file"
echo "foresight: [$FORESIGHT_ROLE] ready, endpoint published to $endpoint_file" >&2

wait "$vllm_pid"
