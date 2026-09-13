# Shared helpers for deploy/tau-slurm scripts. Not part of the foresight library --
# source this from sbatch scripts and launch.sh, never from anything under foresight/.
#
# Resolves one thing the whole deployment hangs off: FORESIGHT_ROOT. See
# deploy/tau-slurm/README.md for the precedence rules and why $WORK cannot be
# assumed to exist for every user of this cluster.

set -eu

# foresight_resolve_root [--root DIR]
#
# Precedence: --root  >  $FORESIGHT_ROOT  >  $WORK  >  abort.
# Sets and exports FORESIGHT_ROOT. Aborts if the resolved root, or HF_HOME
# (existing or derived), falls under the real $HOME -- that filesystem has a
# 2 GB quota on this cluster and is a different mount from the working area.
foresight_resolve_root() {
    _fr_arg_root=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --root)
                _fr_arg_root="$2"
                shift 2
                ;;
            --root=*)
                _fr_arg_root="${1#--root=}"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    if [ -n "$_fr_arg_root" ]; then
        FORESIGHT_ROOT="$_fr_arg_root"
    elif [ -n "${FORESIGHT_ROOT:-}" ]; then
        : # already set in the environment
    elif [ -n "${WORK:-}" ]; then
        FORESIGHT_ROOT="$WORK"
    else
        cat >&2 <<'EOF'
foresight: cannot resolve a working root.

None of --root, $FORESIGHT_ROOT or $WORK is set. This deployment refuses to
guess, because the only unset fallback (writing under $HOME) fills a 2 GB
quota on this cluster's home filesystem.

Fix: either
  export FORESIGHT_ROOT=<your-working-area>
or pass --root <your-working-area> to this script.

Your working area is your lab's shared storage -- NOT $HOME, which has a small
quota on this cluster and is a different filesystem. If you are unsure which
path that is, ask whoever administers your group's storage.
EOF
        exit 1
    fi

    # Resolve to an absolute, real path so the $HOME-prefix check below cannot
    # be fooled by a relative path or a symlink hop.
    if [ ! -d "$FORESIGHT_ROOT" ]; then
        mkdir -p "$FORESIGHT_ROOT"
    fi
    FORESIGHT_ROOT="$(cd "$FORESIGHT_ROOT" && pwd -P)"

    _fr_real_home="$(cd "$HOME" && pwd -P)"
    case "$FORESIGHT_ROOT" in
        "$_fr_real_home"|"$_fr_real_home"/*)
            echo "foresight: refusing to use FORESIGHT_ROOT=$FORESIGHT_ROOT -- it is" \
                 "under \$HOME ($_fr_real_home), whose quota is 2 GB on this cluster." >&2
            echo "Point it at your lab's shared storage instead." >&2
            exit 1
            ;;
    esac

    export FORESIGHT_ROOT

    # HF_HOME: respect an existing value untouched; otherwise derive from the
    # root. Either way, guard it the same way.
    HF_HOME="${HF_HOME:-$FORESIGHT_ROOT/.cache/huggingface}"
    mkdir -p "$HF_HOME"
    HF_HOME="$(cd "$HF_HOME" && pwd -P)"
    case "$HF_HOME" in
        "$_fr_real_home"|"$_fr_real_home"/*)
            echo "foresight: refusing HF_HOME=$HF_HOME -- it is under \$HOME" \
                 "($_fr_real_home), whose quota is 2 GB on this cluster." >&2
            exit 1
            ;;
    esac
    export HF_HOME
}

# foresight_runs_dir
#
# Echoes $FORESIGHT_ROOT/foresight-runs -- the parent of every per-run
# directory. Call foresight_resolve_root first.
foresight_runs_dir() {
    echo "$FORESIGHT_ROOT/foresight-runs"
}

# foresight_new_run_dir
#
# Creates and echoes a fresh $FORESIGHT_ROOT/foresight-runs/<user>-<stamp>/
# directory with a logs/ subdirectory. One call per launch.sh invocation.
foresight_new_run_dir() {
    _fr_stamp="$(date +%Y%m%d-%H%M%S)"
    _fr_run="$(foresight_runs_dir)/${USER}-${_fr_stamp}"
    mkdir -p "$_fr_run/logs"
    echo "$_fr_run"
}

# foresight_agent_dir
#
# Echoes $FORESIGHT_ROOT/agent -- where install_opencode.sh installs node,
# opencode-ai and the wrapper. Call foresight_resolve_root first.
foresight_agent_dir() {
    echo "$FORESIGHT_ROOT/agent"
}

# foresight_wait_for_file PATH TIMEOUT_S [PRODUCER_JOBIDS]
#
# Polls for PATH to appear (written by serve_vllm.sbatch once its vLLM answers
# 200 on /v1/models). Covers "job still pending in the queue" for free --
# no squeue parsing. Exits non-zero after TIMEOUT_S with a clear message.
#
# PRODUCER_JOBIDS turns the deadline from a guess into a fact. A fixed wall
# clock has to answer an unanswerable question -- how long is too long? -- and
# gets it wrong in both directions: too short kills a healthy run (observed:
# the proxy gave up at exactly 1800s while vLLM was still reading a 14 GB
# checkpoint off NFS and came up fine minutes later), too long leaves the proxy
# idling for an hour against a job that died in its first second.
#
# The producing job's own liveness answers it exactly. While any producer is
# PENDING or RUNNING the endpoint may still arrive, so keep waiting; once every
# producer has left the queue without writing the file, it never will, so fail
# immediately instead of serving out the clock. TIMEOUT_S stays as a backstop
# for the case where the producer is alive but hung.
foresight_wait_for_file() {
    _fr_path="$1"
    _fr_timeout="$2"
    _fr_jobs="${3:-}"
    _fr_waited=0
    while [ ! -s "$_fr_path" ]; do
        if [ -n "$_fr_jobs" ]; then
            _fr_live=0
            for _fr_j in $_fr_jobs; do
                if squeue -j "$_fr_j" -h -o "%T" 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; then
                    _fr_live=1
                    break
                fi
            done
            if [ "$_fr_live" -eq 0 ]; then
                echo "foresight: producer job(s) [$_fr_jobs] left the queue without publishing" \
                     "$_fr_path -- it is never going to appear. Check their logs." >&2
                return 1
            fi
        fi
        if [ "$_fr_waited" -ge "$_fr_timeout" ]; then
            echo "foresight: timed out after ${_fr_timeout}s waiting for $_fr_path" \
                 "(producer job(s) [${_fr_jobs:-unknown}] still alive but not ready --" \
                 "check their logs; raise FORESIGHT_WAIT_TIMEOUT if this is just a slow load)" >&2
            return 1
        fi
        sleep 5
        _fr_waited=$((_fr_waited + 5))
    done
}
