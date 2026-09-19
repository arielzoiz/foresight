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
#   FORESIGHT_TENSOR_PARALLEL  passed to --tensor-parallel-size (default 1).
#                                launch.sh defaults it to --gpus, which is almost
#                                always what you want: without it, asking Slurm
#                                for N GPUs gets you N GPUs and vLLM quietly uses
#                                ONE of them, so a model that needs the combined
#                                VRAM OOMs while N-1 cards sit idle. Must divide
#                                the model's attention-head count -- powers of two
#                                are the safe choice.
#   FORESIGHT_TOOL_CALL_PARSER  passed to --tool-call-parser. Default "auto":
#                                read the checkpoint's own chat template and pick
#                                the matching parser (see detect_tool_call_parser
#                                below). An explicit name overrides detection;
#                                "" disables tool calling entirely.
#   FORESIGHT_ENFORCE_EAGER   "1" passes --enforce-eager, skipping CUDA-graph
#                                capture (and the torch.compile pass ahead of
#                                it) entirely. Trades throughput for startup
#                                reliability -- see "CUDA-graph capture can
#                                legitimately take longer than the readiness
#                                wait" below. Default unset (graphs on).
#   FORESIGHT_READY_TIMEOUT_S  how long to wait for vllm's own /v1/models to
#                                answer before giving up, in seconds (default
#                                3600 = 60 min). Large models doing real
#                                torch.compile + CUDA-graph capture across many
#                                batch sizes can genuinely need longer than
#                                that -- this was previously a hardcoded cap,
#                                which killed a real (if slow) H200 run at 88%
#                                through graph capture. Raise this rather than
#                                assume a slow run is a hung one.
#   HF_HOME                  inherited if already set; else derived from FORESIGHT_ROOT
#   VLLM_CACHE_ROOT           inherited if already set; else derived from
#                                FORESIGHT_ROOT. vLLM's own default
#                                (~/.cache/vllm) is the small, per-node-local
#                                $HOME -- torch.compile/CUDA-graph artifacts
#                                written there never persist across a
#                                different node assignment (which is nearly
#                                every run), so the compile cache silently
#                                never helps and nothing ever benefits from a
#                                previous run's work. Redirecting it here is
#                                the same fix HF_HOME already gets, and it's
#                                what makes a second run against the same
#                                checkpoint fast.

set -eu

: "${FORESIGHT_ROOT:?FORESIGHT_ROOT must be exported before submitting this job}"
: "${FORESIGHT_RUN_DIR:?FORESIGHT_RUN_DIR must be set}"
: "${FORESIGHT_ROLE:?FORESIGHT_ROLE must be set (target|aux)}"
: "${FORESIGHT_MODEL_ID:?FORESIGHT_MODEL_ID must be set (an HF model id)}"
: "${FORESIGHT_ALIAS:?FORESIGHT_ALIAS must be set (the --served-model-name alias)}"
: "${FORESIGHT_PORT:?FORESIGHT_PORT must be set}"
FORESIGHT_MAX_MODEL_LEN="${FORESIGHT_MAX_MODEL_LEN:-32768}"
FORESIGHT_TENSOR_PARALLEL="${FORESIGHT_TENSOR_PARALLEL:-1}"
FORESIGHT_TOOL_CALL_PARSER="${FORESIGHT_TOOL_CALL_PARSER-auto}"
FORESIGHT_ENFORCE_EAGER="${FORESIGHT_ENFORCE_EAGER:-}"
FORESIGHT_READY_TIMEOUT_S="${FORESIGHT_READY_TIMEOUT_S:-3600}"
HF_HOME="${HF_HOME:-$FORESIGHT_ROOT/.cache/huggingface}"
export HF_HOME
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$FORESIGHT_ROOT/.cache/vllm}"
export VLLM_CACHE_ROOT

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
# requirement. The flag only affects the SAMPLER, so it costs some throughput
# and changes no output; it is orthogonal to the model's architecture (it
# applies equally to the dense 7B and to the 30B-A3B MoE) and to whether
# tensor parallelism is in use.
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

# GPU preflight. A node can sit in Slurm's `mix` state, with no drain reason,
# while one of its GPUs has fallen off the bus -- and Slurm will happily
# allocate you the dead one. Directly observed on n-102 (gpu-h100-killable):
# all eight /dev/nvidia* present, cgroup fine, but bare `nvidia-smi -L` printed
#
#     Unable to determine the device handle for gpu 0000:18:00.0: Unknown Error
#
# for two of the eight, and the job was handed CUDA_VISIBLE_DEVICES=7, one of
# them. vLLM then died with `NVMLError_Unknown` from inside `import vllm` --
# CudaPlatform.log_warnings() enumerates EVERY physical device at import time,
# so one sick GPU anywhere on the node aborts the job even if yours is healthy.
#
# That traceback points at pynvml and looks like a vLLM/driver version problem.
# It is neither. Checking here turns 30+ minutes of queue and weight loading,
# ending in a confusing stack trace, into a five-second failure that names the
# node -- so you can resubmit elsewhere (or --exclude it) immediately.
if command -v nvidia-smi >/dev/null 2>&1; then
    smi_out="$(nvidia-smi -L 2>&1 || true)"
    if echo "$smi_out" | grep -q "Unable to determine the device handle"; then
        echo "foresight: [$FORESIGHT_ROLE] ABORT -- $(hostname) has an unhealthy GPU." >&2
        echo "foresight: nvidia-smi -L cannot enumerate every device on this node:" >&2
        echo "$smi_out" | sed 's/^/foresight:   /' >&2
        echo "foresight: vLLM enumerates ALL physical devices at import and will" >&2
        echo "foresight: die with NVMLError_Unknown regardless of which GPU is ours." >&2
        echo "foresight: resubmit, or pass --exclude $(hostname) to launch.sh." >&2
        exit 1
    fi
    echo "foresight: [$FORESIGHT_ROLE] GPU preflight OK on $(hostname)" \
         "(CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
fi

# CUDA compiler / toolkit-header version preflight. pip resolves
# nvidia-cuda-nvcc and nvidia-cuda-runtime as independent wheels, not a
# matched bundle, so a plain `pip install vllm` can land them on different
# CUDA point releases within the same 13.x line. cccl's cuda_toolkit.h
# asserts these are equal at major.minor granularity; a mismatch is invisible
# until a JIT-compiled kernel (DeepGEMM, flashinfer's fused MoE path) is
# actually built -- which happens well after a checkpoint has already loaded.
# Confirmed directly on a B200 node serving Qwen3-Coder-30B/Next-FP8,
# 2026-09-18: "CUDA compiler and CUDA toolkit headers are incompatible,
# please check your include paths", ~20 minutes and a full checkpoint load
# into the failure both times. install_vllm_env.sh now pins these together
# at install time; this catches a hand-modified or pre-existing env before
# paying that cost again.
nvcc_line="$(conda run -p "$env_prefix" python -c \
    "import importlib.metadata as m; print('.'.join(m.version('nvidia-cuda-nvcc').split('.')[:2]))" \
    2>/dev/null || true)"
rt_line="$(conda run -p "$env_prefix" python -c \
    "import importlib.metadata as m; print('.'.join(m.version('nvidia-cuda-runtime').split('.')[:2]))" \
    2>/dev/null || true)"
if [ -n "$nvcc_line" ] && [ -n "$rt_line" ] && [ "$nvcc_line" != "$rt_line" ]; then
    echo "foresight: [$FORESIGHT_ROLE] ABORT -- nvidia-cuda-nvcc (${nvcc_line}.x) and" >&2
    echo "foresight: nvidia-cuda-runtime (${rt_line}.x) are on different CUDA point" >&2
    echo "foresight: releases. Any JIT-compiled kernel (DeepGEMM, flashinfer's fused" >&2
    echo "foresight: MoE path) will fail with 'CUDA compiler and CUDA toolkit headers" >&2
    echo "foresight: are incompatible' -- usually after a full checkpoint load, not" >&2
    echo "foresight: before. Fix:" >&2
    echo "foresight:   conda run -p $env_prefix pip install --no-deps 'nvidia-cuda-runtime==${nvcc_line}.*'" >&2
    exit 1
fi

# CUDA library layout preflight. pip's nvidia-cuXX wheels ship libraries
# under plain lib/, not the lib64/ a traditional system CUDA install uses,
# and ship only the versioned .so (no unversioned libNAME.so for a bare
# `-lNAME` to resolve against) -- confirmed across every library in the
# package, not just cudart: fixing cudart alone only revealed nvrtc as the
# next one flashinfer's JIT linker needed (Qwen3-Coder-Next-FP8, H200,
# 2026-09-19). A tool built for the traditional layout fails with "cannot
# find -lNAME: No such file or directory", downstream of (and masked by) the
# version-mismatch check above until that one is fixed -- and each library
# is only discovered once something actually links against it, well after a
# full checkpoint load. install_vllm_env.sh now creates lib64 plus an
# unversioned symlink for every versioned .so at install time; this checks
# for the same before paying for a full checkpoint load again.
missing_unversioned=""
if [ -n "$cuda_pkg_home" ]; then
    for versioned in "$cuda_pkg_home"/lib/lib*.so.*; do
        [ -e "$versioned" ] || continue
        unversioned="$(basename "$versioned" | sed 's/\.so\..*/.so/')"
        [ -e "$cuda_pkg_home/lib/$unversioned" ] || missing_unversioned="$missing_unversioned $unversioned"
    done
fi
if [ -n "$cuda_pkg_home" ] && { [ ! -e "$cuda_pkg_home/lib64" ] || [ -n "$missing_unversioned" ]; }; then
    echo "foresight: [$FORESIGHT_ROLE] ABORT -- $cuda_pkg_home is missing the" >&2
    echo "foresight: lib64 -> lib compatibility symlink and/or unversioned .so" >&2
    echo "foresight: symlinks for:${missing_unversioned:- (lib64 only)}" >&2
    echo "foresight: Any JIT-compiled kernel's link step (observed: flashinfer's" >&2
    echo "foresight: fused MoE path) will fail with 'cannot find -lNAME' after a" >&2
    echo "foresight: full checkpoint load, not before. Fix: re-run" >&2
    echo "foresight:   deploy/tau-slurm/setup/install_vllm_env.sh" >&2
    exit 1
fi

# -- which tool-call parser this checkpoint needs --------------------------
#
# The checkpoint already knows. Its chat template is what emits tool calls, so
# grepping the template is a direct reading of the wire format rather than a
# guess from the model's name -- and it removes a per-run decision that fails
# SILENTLY when got wrong: a mismatched parser returns HTTP 200 with
# tool_calls=null, which is indistinguishable from a model that cannot call
# tools at all.
#
# `auto` (the default) detects; an explicit name always wins; empty disables
# tool calling entirely.
#
# Order matters and is not arbitrary: Qwen3-Coder's template contains BOTH
# `<function=` and `<tool_call>`, because it wraps the XML in the same tag
# Qwen2.5 wraps JSON in. Testing `<function=` first is what separates them.
#
# Both files are searched because the template lives in different places
# depending on the checkpoint -- Qwen3-Coder ships chat_template.jinja, while
# Qwen2.5-Coder-7B has no such file and carries the template inside
# tokenizer_config.json.
detect_tool_call_parser() {
    _d=""
    if [ -d "$FORESIGHT_MODEL_ID" ]; then
        _d="$FORESIGHT_MODEL_ID"
    else
        # HF's own cache layout: models--<org>--<name>/snapshots/<rev>/
        _cache="$HF_HOME/hub/models--$(echo "$FORESIGHT_MODEL_ID" | sed 's#/#--#g')"
        _d="$(ls -dt "$_cache"/snapshots/*/ 2>/dev/null | head -1)"
    fi
    [ -n "$_d" ] || return 1

    for _f in "$_d/chat_template.jinja" "$_d/tokenizer_config.json"; do
        [ -f "$_f" ] || continue
        if grep -q '<function=' "$_f" 2>/dev/null; then echo qwen3_coder; return 0; fi
        if grep -q '<tool_call>' "$_f" 2>/dev/null; then echo hermes; return 0; fi
        if grep -q '\[TOOL_CALLS\]' "$_f" 2>/dev/null; then echo mistral; return 0; fi
        if grep -q 'python_tag' "$_f" 2>/dev/null; then echo llama3_json; return 0; fi
    done
    return 1
}

if [ "$FORESIGHT_TOOL_CALL_PARSER" = "auto" ]; then
    if _detected="$(detect_tool_call_parser)" && [ -n "$_detected" ]; then
        FORESIGHT_TOOL_CALL_PARSER="$_detected"
        echo "foresight: [$FORESIGHT_ROLE] detected tool-call parser: $_detected" >&2
    else
        # Not fatal, and deliberately not a guess: serving with the WRONG parser
        # fails silently, while serving with none fails loudly on the first
        # request carrying a tools array -- which is the better failure.
        FORESIGHT_TOOL_CALL_PARSER=""
        echo "foresight: [$FORESIGHT_ROLE] WARNING -- could not detect a tool-call" \
             "parser for $FORESIGHT_MODEL_ID; serving WITHOUT tool calling." >&2
        echo "foresight: pass --tool-call-parser NAME to launch.sh if this model" \
             "needs one (a real agent harness always sends a tools array)." >&2
    fi
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
# Echoed because a wrong parser is otherwise invisible: it produces 200s with
# tool_calls=null, not errors. This line is the first thing to check when a run
# comes back ungrounded.
echo "foresight: [$FORESIGHT_ROLE] tool-call parser: ${FORESIGHT_TOOL_CALL_PARSER:-(disabled)}" >&2
echo "foresight: [$FORESIGHT_ROLE] tensor-parallel-size: $FORESIGHT_TENSOR_PARALLEL" >&2

# Without --enable-auto-tool-choice/--tool-call-parser, vLLM 400s on ANY
# request carrying a tools array -- which every message from a real agent
# harness does (it always advertises its bash tool). Observed directly: an
# opencode session pointed at a server started without these flags retried
# the same 400 forever, 1215 requests in 45 minutes, never making progress.
tool_call_args=""
if [ -n "$FORESIGHT_TOOL_CALL_PARSER" ]; then
    tool_call_args="--enable-auto-tool-choice --tool-call-parser $FORESIGHT_TOOL_CALL_PARSER"
fi

# CUDA-graph capture can legitimately take longer than the readiness wait.
# Observed directly, gpu-h200-killable, Qwen3-Coder-30B-A3B-Instruct-FP8:
# capture reached 45/51 (88%) with per-item cost climbing (27s -> 80s ->
# 93s as batch size grew) before the readiness loop's old hardcoded cap
# killed the job -- genuine slow progress on a cold compile cache (see
# VLLM_CACHE_ROOT above), not a hang. --enforce-eager skips torch.compile
# and CUDA-graph capture entirely, trading throughput for a startup that
# cannot get stuck here -- the safer choice for a large model's first run
# or an unattended one. Once VLLM_CACHE_ROOT is warm and
# FORESIGHT_READY_TIMEOUT_S comfortably covers a cold compile, this can be
# dropped for the throughput graphs give back.
eager_args=""
if [ "$FORESIGHT_ENFORCE_EAGER" = "1" ]; then
    eager_args="--enforce-eager"
    echo "foresight: [$FORESIGHT_ROLE] --enforce-eager: skipping CUDA-graph capture" >&2
fi

# FORESIGHT_ALIAS, tool_call_args and eager_args are deliberately NOT quoted
# below: FORESIGHT_ALIAS may be "target-upstream aux-upstream" -- two
# space-separated names in --single mode -- so that vLLM's
# --served-model-name (which takes one or more values) registers both
# aliases against the one loaded model. tool_call_args/eager_args are each
# either empty (nothing to pass) or the relevant flag(s).
# shellcheck disable=SC2086
conda run --no-capture-output -p "$env_prefix" \
    vllm serve "$FORESIGHT_MODEL_ID" \
        --host 0.0.0.0 \
        --port "$FORESIGHT_PORT" \
        --served-model-name $FORESIGHT_ALIAS \
        --max-model-len "$FORESIGHT_MAX_MODEL_LEN" \
        --tensor-parallel-size "$FORESIGHT_TENSOR_PARALLEL" \
        $tool_call_args \
        $eager_args \
    &
vllm_pid=$!

# Poll our own port rather than trusting "the process started" -- weight
# loading on a cold cache can take minutes, and a request sent too early is a
# connection-refused, not a useful error. FORESIGHT_READY_TIMEOUT_S, not a
# hardcoded cap -- see its header comment for why a fixed 60 min cut off a
# genuinely-progressing H200 run at 88% through CUDA-graph capture.
ready=0
i=0
iters=$((FORESIGHT_READY_TIMEOUT_S / 5))
while [ "$i" -lt "$iters" ]; do
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
