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

# pip resolves nvidia-cuda-nvcc (the compiler) and nvidia-cuda-runtime (its
# headers) as independent wheels, not a matched bundle -- so a plain `pip
# install vllm` can silently land them on different CUDA point releases
# within the same 13.x line (observed: nvcc 13.4.59 against runtime 13.0.96).
# cccl's cuda_toolkit.h asserts these are equal at major.minor granularity;
# a mismatch is invisible until a JIT-compiled kernel (DeepGEMM, flashinfer's
# fused MoE path) actually gets built, which happens well after a checkpoint
# has already loaded: "CUDA compiler and CUDA toolkit headers are
# incompatible, please check your include paths". Confirmed directly on a
# B200 node serving Qwen3-Coder-30B/Next-FP8, 2026-09-18 -- ~20 minutes and a
# full checkpoint load into the failure both times. Force them back onto the
# same line explicitly, right after install, rather than discovering this per
# job.
nvcc_line="$(conda run -p "$ENV_PREFIX" python -c \
    "import importlib.metadata as m; print('.'.join(m.version('nvidia-cuda-nvcc').split('.')[:2]))")"
echo "foresight: pinning nvidia-cuda-runtime to the ${nvcc_line}.x line to match nvcc ..." >&2
conda run -p "$ENV_PREFIX" pip install --cache-dir "$PIP_CACHE_DIR" --no-deps \
    "nvidia-cuda-runtime==${nvcc_line}.*"

# Every nvidia-cuXX pip wheel (cu13's cudart, cublas, cufft, curand,
# cusolver, cusparse, nvrtc, nvJitLink, ... -- confirmed: ALL of them, not
# just cudart) ships its libraries under a plain lib/, never the lib64/ a
# traditional system CUDA toolkit install uses, and ships only the versioned
# .so (libcudart.so.13, libnvrtc.so.13, ...) -- no unversioned .so symlink
# for a bare `-lNAME` to resolve against. Tools written against the
# traditional layout (observed: flashinfer's JIT linker step) fail with
# "cannot find -lNAME: No such file or directory" -- distinct from, and
# downstream of, the nvcc/runtime version-mismatch fixed above: this masked
# it entirely until that fix let the build reach the link step, and even
# then, fixing one library only exposed the *next* one flashinfer happened
# to link against next (cudart, then nvrtc -- confirmed on B200 and H200
# respectively, serving Qwen3-Coder-30B-A3B-Instruct-FP8 and
# Qwen3-Coder-Next-FP8, 2026-09-19). Not GPU-architecture-specific -- this
# is pip packaging, not driver/hardware -- so fix every library up front
# rather than chasing them one at a time. libcuda.so (the driver stub
# `-lcuda` needs) is unaffected: it comes from the node's real NVIDIA driver
# install on the default system linker path, not from pip.
cuda_pkg_dir="$(find "$ENV_PREFIX"/lib/python*/site-packages/nvidia/cu[0-9]* -maxdepth 0)"
echo "foresight: linking lib64 -> lib under $cuda_pkg_dir ..." >&2
ln -sfn lib "$cuda_pkg_dir/lib64"
echo "foresight: linking every libNAME.so -> libNAME.so.<version> under $cuda_pkg_dir/lib ..." >&2
for versioned in "$cuda_pkg_dir"/lib/lib*.so.*; do
    [ -e "$versioned" ] || continue
    base="$(basename "$versioned")"
    unversioned="${base%%.so.*}.so"
    ln -sf "$base" "$cuda_pkg_dir/lib/$unversioned"
done

echo "foresight: done. Activate with:" >&2
echo "  conda activate $ENV_PREFIX" >&2
