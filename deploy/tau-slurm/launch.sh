#!/bin/sh
# Orchestrator: submits one or two vLLM jobs plus the foresight proxy job,
# wired together by a run directory full of endpoint files (see
# deploy/tau-slurm/README.md for the full picture). This is the only entry
# point teammates should use -- serve_vllm.sbatch and foresight.sbatch are
# not meant to be submitted by hand.
#
# Usage:
#   launch.sh --single <hf-model-id> [options]
#   launch.sh --split <target-hf-id> <aux-hf-id> [options]
#
# Options:
#   --adapter NAME          local (default) or generic.
#                            local   -- milestone 2: a REAL opencode aux agent
#                                       explores --workspace, manifest guard
#                                       armed, trace written to the run dir.
#                                       Requires --workspace and a prior
#                                       setup/install_opencode.sh.
#                            generic -- milestone 1: aux is one chat call over
#                                       the request body and sees no code. The
#                                       cheap bring-up rung; use it to prove the
#                                       topology before blaming the agent.
#   --workspace DIR         repo the aux agent explores, read-only. REQUIRED with
#                            --adapter local. The guard DETECTS writes rather
#                            than preventing them, so give it a scratch
#                            checkout, never work you care about.
#   --fake                  NOT the experimental path. Skip the vLLM jobs entirely
#                            and serve both roles from tools/fake_upstream.py
#                            inside the proxy job -- no GPU, no queue wait, ~2
#                            minutes. Everything the AUX AGENT can get wrong
#                            (opencode provider config, OPENCODE_HOME isolation,
#                            argv substitution, cwd, guard, answer stream, trace)
#                            is independent of whether a real model answers, so
#                            this separates "the plumbing is broken" from "the
#                            model is broken". Run it when a real run misbehaves.
#   --root DIR              working root (see lib/common.sh); default $FORESIGHT_ROOT or $WORK
#   --partition NAME        Slurm partition for the vLLM job(s) (default: killable)
#   --account NAME          Slurm account for the vLLM job(s), if needed (default: none --
#                            most accounts need one for killable; run --help to check yours)
#   --foresight-partition N partition for the proxy job (default: cpu-killable)
#   --foresight-account N   Slurm account for the proxy job, if needed (default: none)
#   --gpus N                GPUs per vLLM job (default: 1)
#   --max-model-len N       vLLM --max-model-len (default: 32768)
#   --py-env NAME_OR_PATH   conda env with foresight's own deps (default: foresight)
#   --exclude LIST          Slurm --exclude for the vLLM job(s), comma-separated
#                            node names (default: rack-bgw-dgx1,rack-gww-dgx1,
#                            rack-omerl-g01 -- see below). Pass "" to disable.
#   --constraint FEATURE    Slurm --constraint for the vLLM job(s), i.e. which GPU
#                            model to demand (default: none -- take what the
#                            scheduler gives). `sinfo -o "%N %f"` lists the
#                            features. Prefer this over growing --exclude: the
#                            recurring failure is a node whose NVIDIA driver is
#                            older than the CUDA build pip installed for vLLM
#                            ("The NVIDIA driver on your system is too old"), and
#                            naming the hardware you want is more durable than
#                            enumerating the hardware you don't. Measured drivers:
#                            rtx_3090/a6000 595.84, a5000 580.17, h100 610.57 all
#                            work; l40s 535.18 and tesla_v100 are too old.
#                            Constrain to dodge those, NOT to chase speed: n-301
#                            and n-306 are both rtx_3090 and gave the fastest and
#                            the worst (hung) startups of the same evening.
#   -h, --help              print this and your own account/partition access
#
# Default partition is killable, not studentkillable, even though
# studentkillable needs no --account at all (the broadest-access option).
# studentkillable's hardware is exclusively Titan Xp (Pascal, compute
# capability 6.1), and `pip install vllm` currently resolves a CUDA-13 build
# that Pascal cannot run at all -- confirmed by direct testing, not a
# theoretical concern. killable's pool (a5000/a6000/l40s/rtx_3090/rtx_2080/
# v100/quadro) is Volta-or-newer throughout and works. The cost: most
# accounts need --account=gpu-research (or your own) for killable, which
# studentkillable never required -- run --help to check yours.
#
# The vLLM and proxy jobs commonly need DIFFERENT accounts on top of that:
# e.g. on this cluster cpu-killable also needs --account=gpu-research --
# that is why --account and --foresight-account are two separate flags, not
# one shared --account.
#
# The default --exclude list is 3 individually-managed legacy nodes (as
# opposed to the centrally-imaged n-*/t-* pool), observed to fail vLLM
# startup two different ways on rack-bgw-dgx1: a system libstdc++ too old for
# vllm's dependency chain, and separately a GPU driver capped at CUDA 12.4
# while pip installs vLLM builds compiled for CUDA 13 (vLLM's own releases
# offer no older variant than cu128, so pinning an older CUDA build is not an
# available fix here -- excluding the node is). rack-gww-dgx1 and
# rack-omerl-g01 share the same naming/provisioning pattern and are excluded
# defensively, not individually confirmed bad.
#
# The model in the --adapter local examples is a 7B, not the 1.5B used for
# topology bring-up, and that is not arbitrary: an aux AGENT has to emit tool
# calls in the shape opencode expects, and Qwen2.5-Coder-1.5B-Instruct was
# observed not to manage it (README §2, "Model tool-call quality"). The 1.5B is
# still the right choice for --adapter generic, whose aux call carries no tools
# at all.
#
# Examples:
#   # full pipeline, one model backing both roles, real aux agent
#   launch.sh --single Qwen/Qwen2.5-Coder-7B-Instruct \
#       --workspace $WORK/workspaces/tinyrepo \
#       --account gpu-research --foresight-account gpu-research
#
#   # topology only, no aux agent -- the rung to fall back to when the above breaks
#   launch.sh --single Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter generic \
#       --account gpu-research --foresight-account gpu-research
#
#   # two servers, so aux's many calls stop competing with the target's held-open one
#   launch.sh --split Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct \
#       --workspace $WORK/workspaces/tinyrepo \
#       --account gpu-research --foresight-account gpu-research

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"

usage() {
    sed -n '2,98p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "Your Slurm access (sacctmgr -P -i show user -s \$USER):"
    sacctmgr -P -i show user -s "$USER" 2>&1 || echo "  (sacctmgr failed -- are you on the login node?)"
}

mode=""
target_model=""
aux_model=""
root_arg=""
partition="killable"
account=""
foresight_partition="cpu-killable"
foresight_account=""
gpus="1"
max_model_len="32768"
py_env="foresight"
exclude_nodes="rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01"
constraint=""
adapter="local"
workspace=""
fake=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --single)
            mode="single"
            target_model="$2"
            aux_model="$2"
            shift 2
            ;;
        --split)
            mode="split"
            target_model="$2"
            aux_model="$3"
            shift 3
            ;;
        --root)
            root_arg="--root $2"
            shift 2
            ;;
        --partition)
            partition="$2"
            shift 2
            ;;
        --account)
            account="$2"
            shift 2
            ;;
        --foresight-partition)
            foresight_partition="$2"
            shift 2
            ;;
        --foresight-account)
            foresight_account="$2"
            shift 2
            ;;
        --gpus)
            gpus="$2"
            shift 2
            ;;
        --max-model-len)
            max_model_len="$2"
            shift 2
            ;;
        --py-env)
            py_env="$2"
            shift 2
            ;;
        --exclude)
            exclude_nodes="$2"
            shift 2
            ;;
        --constraint)
            constraint="$2"
            shift 2
            ;;
        --adapter)
            adapter="$2"
            shift 2
            ;;
        --workspace)
            workspace="$2"
            shift 2
            ;;
        --fake)
            fake=1
            mode="fake"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "foresight: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$mode" ]; then
    echo "foresight: one of --single, --split or --fake is required" >&2
    usage >&2
    exit 1
fi

# Validated here, before anything is submitted: the same mistake caught inside
# foresight.sbatch would already have two vLLM jobs sitting in the queue holding
# GPUs against a run that cannot start.
case "$adapter" in
    local)
        if [ -z "$workspace" ]; then
            echo "foresight: --adapter local needs --workspace DIR (the repo the aux agent explores)." >&2
            echo "foresight: for a topology-only run with no aux agent, pass --adapter generic." >&2
            exit 1
        fi
        if [ ! -d "$workspace" ]; then
            echo "foresight: --workspace is not a directory: $workspace" >&2
            exit 1
        fi
        workspace="$(cd "$workspace" && pwd -P)"
        ;;
    generic)
        if [ -n "$workspace" ]; then
            echo "foresight: --workspace is meaningless with --adapter generic (aux sees only the request body)." >&2
            exit 1
        fi
        ;;
    *)
        echo "foresight: unknown --adapter: $adapter (want local or generic)" >&2
        exit 1
        ;;
esac

# shellcheck disable=SC2086
foresight_resolve_root $root_arg

# Also before submitting: a missing harness is a one-time setup step the user
# has simply not run yet, and it should not cost them a queue wait to find out.
if [ "$adapter" = "local" ] && [ ! -x "$(foresight_agent_dir)/bin/opencode" ]; then
    echo "foresight: --adapter local needs the opencode wrapper at $(foresight_agent_dir)/bin/opencode" >&2
    echo "foresight: run deploy/tau-slurm/setup/install_opencode.sh (login node -- it needs internet)." >&2
    exit 1
fi

run_dir="$(foresight_new_run_dir)"
echo "foresight: run directory: $run_dir" >&2

account_opt=""
[ -n "$account" ] && account_opt="--account=$account"
foresight_account_opt=""
[ -n "$foresight_account" ] && foresight_account_opt="--account=$foresight_account"
exclude_opt=""
[ -n "$exclude_nodes" ] && exclude_opt="--exclude=$exclude_nodes"
constraint_opt=""
[ -n "$constraint" ] && constraint_opt="--constraint=$constraint"

submit_vllm() {
    # submit_vllm ROLE MODEL_ID ALIAS PORT
    role="$1"
    model_id="$2"
    alias="$3"
    port="$4"
    # shellcheck disable=SC2086
    sbatch \
        --job-name="foresight-vllm-$role" \
        --output="$run_dir/logs/vllm-$role-%j.out" \
        --error="$run_dir/logs/vllm-$role-%j.err" \
        --partition="$partition" \
        $account_opt \
        $exclude_opt \
        $constraint_opt \
        --gpus="$gpus" \
        --export=ALL,FORESIGHT_ROOT="$FORESIGHT_ROOT",HF_HOME="$HF_HOME",FORESIGHT_RUN_DIR="$run_dir",FORESIGHT_ROLE="$role",FORESIGHT_MODEL_ID="$model_id",FORESIGHT_ALIAS="$alias",FORESIGHT_PORT="$port",FORESIGHT_MAX_MODEL_LEN="$max_model_len" \
        --parsable \
        "$SCRIPT_DIR/serve_vllm.sbatch"
}

if [ "$fake" -eq 1 ]; then
    echo "foresight: --fake: no vLLM jobs; fake_upstream runs inside the proxy job" >&2
    vllm_job_target=""
elif [ "$mode" = "single" ]; then
    echo "foresight: submitting one vLLM job serving $target_model as both roles ..." >&2
    vllm_job_target="$(submit_vllm target "$target_model" "target-upstream aux-upstream" 8001)"
    echo "foresight: vLLM job (single, both roles): $vllm_job_target" >&2
else
    echo "foresight: submitting two vLLM jobs (target=$target_model, aux=$aux_model) ..." >&2
    vllm_job_target="$(submit_vllm target "$target_model" "target-upstream" 8001)"
    vllm_job_aux="$(submit_vllm aux "$aux_model" "aux-upstream" 8002)"
    echo "foresight: vLLM job (target): $vllm_job_target" >&2
    echo "foresight: vLLM job (aux):    $vllm_job_aux" >&2
fi

# The proxy waits on the vLLM jobs' liveness, not on a stopwatch -- see
# foresight_wait_for_file in lib/common.sh.
vllm_jobs="${vllm_job_target:-}${vllm_job_aux:+ $vllm_job_aux}"

# shellcheck disable=SC2086
foresight_job="$(sbatch \
    --job-name=foresight-proxy \
    --output="$run_dir/logs/proxy-%j.out" \
    --error="$run_dir/logs/proxy-%j.err" \
    --partition="$foresight_partition" \
    $foresight_account_opt \
    --export=ALL,FORESIGHT_ROOT="$FORESIGHT_ROOT",HF_HOME="$HF_HOME",FORESIGHT_RUN_DIR="$run_dir",REPO_DIR="$REPO_DIR",FORESIGHT_PY_ENV="$py_env",FORESIGHT_ADAPTER="$adapter",FORESIGHT_WORKSPACE="$workspace",FORESIGHT_MAX_MODEL_LEN="$max_model_len",FORESIGHT_VLLM_JOBS="$vllm_jobs",FORESIGHT_FAKE_UPSTREAM="$fake" \
    --parsable \
    "$SCRIPT_DIR/foresight.sbatch")"
echo "foresight: proxy job: $foresight_job" >&2

if [ "$mode" = "single" ] && [ "$fake" -eq 0 ]; then
    # One vLLM instance backs both roles: mirror its endpoint into aux.endpoint
    # too, once it publishes target.endpoint, so foresight.sbatch's wait-for-
    # both loop is satisfied without a second GPU job.
    (
        target_ep="$run_dir/target.endpoint"
        aux_ep="$run_dir/aux.endpoint"
        i=0
        while [ ! -s "$target_ep" ] && [ "$i" -lt 720 ]; do
            sleep 5
            i=$((i + 1))
        done
        if [ -s "$target_ep" ]; then
            cp "$target_ep" "$aux_ep"
        fi
    ) &
fi

cat <<EOF >&2

foresight: adapter:         $adapter${workspace:+  (aux workspace: $workspace)}
foresight: run directory:   $run_dir
foresight: watch progress:  tail -f $run_dir/logs/*.out
foresight: once ready:      cat $run_dir/foresight.endpoint
foresight: trace:           $run_dir/trace.jsonl
foresight: cancel this run: scancel ${vllm_job_target:-} ${vllm_job_aux:-} $foresight_job
EOF
