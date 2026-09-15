#!/bin/sh
#SBATCH --job-name=foresight-proxy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000
#SBATCH --time=180
#SBATCH --partition=cpu-killable
#
# Waits for both vLLM endpoints, renders a config for this run, and runs the
# foresight proxy. No GPU -- this is the "caller side" host, matching where
# SWE-CI/mini will eventually run. Submitted by launch.sh, which overrides
# --partition/--account/--output/--error and passes everything else via
# --export.
#
# Required environment (set by launch.sh via --export):
#   FORESIGHT_ROOT      working root (see lib/common.sh)
#   FORESIGHT_RUN_DIR   this run's directory; also where target/aux.endpoint appear
#   REPO_DIR            path to the foresight repo checkout (so this script,
#                        run from an arbitrary cwd, can find configs/ and the
#                        foresight package)
# Optional:
#   FORESIGHT_ADAPTER      "local" (default) or "generic". Selects the template
#                           and decides whether an aux agent runs at all:
#                             local   -- configs/vllm-local.yaml.tmpl, a real
#                                        opencode subprocess over FORESIGHT_WORKSPACE
#                             generic -- configs/vllm.yaml.tmpl, aux is one chat
#                                        call over the request body (V1-V3)
#   FORESIGHT_WORKSPACE    repo the aux agent explores. REQUIRED when
#                           FORESIGHT_ADAPTER=local; unused otherwise.
#   FORESIGHT_PY_ENV       conda env with foresight's own deps installed --
#                           a bare name (conda run -n) or an absolute path
#                           (conda run -p). Default: "foresight", matching the
#                           env name in the main README's own setup section.
#   FORESIGHT_MAX_MODEL_LEN the vLLM job's --max-model-len (default 32768).
#                            Declared to the aux agent as its context limit;
#                            must match, or opencode misjudges its own budget.
#   FORESIGHT_AUX_MAX_OUTPUT max_tokens the aux agent asks for (default 4096).
#                            Must leave room under FORESIGHT_MAX_MODEL_LEN for a
#                            prompt that grows every turn -- see the note by the
#                            opencode_home.sh call below.
#   FORESIGHT_PORT          port foresight listens on. Unset (the default) claims a\n#                            free one derived from the job id -- two proxy jobs can\n#                            share a node, and a fixed port collides. Set it only\n#                            when a caller needs a predictable address.
#   FORESIGHT_WAIT_TIMEOUT  backstop seconds to wait for each vLLM endpoint
#                           (default 7200). Rarely the thing that fires: the
#                           real check is FORESIGHT_VLLM_JOBS below.
#   FORESIGHT_VLLM_JOBS     space-separated vLLM job ids. While any is still in
#                           the queue the endpoint may yet appear, so keep
#                           waiting; once all have left without publishing it,
#                           fail at once rather than serve out the clock.

set -eu

: "${FORESIGHT_ROOT:?FORESIGHT_ROOT must be exported before submitting this job}"
: "${FORESIGHT_RUN_DIR:?FORESIGHT_RUN_DIR must be set}"
: "${REPO_DIR:?REPO_DIR must be set to the foresight repo checkout}"
FORESIGHT_PY_ENV="${FORESIGHT_PY_ENV:-foresight}"
# Port selection, not a constant. Two proxy jobs can land on the SAME node --
# cpu-killable packs them -- and a fixed 8000 then collides. Observed: the
# second job failed with EADDRINUSE, but only AFTER its readiness probe curled
# 127.0.0.1:8000/health, got the FIRST job's server, concluded it was ready, and
# published an endpoint pointing at someone else's proxy. A wrong endpoint is
# worse than a failed job, so the port is claimed before anything trusts it.
#
# Start from a job-id-derived offset so two jobs rarely even contend, then
# probe upward for one that actually binds.
if [ -z "${FORESIGHT_PORT:-}" ]; then
    FORESIGHT_PORT=$(python3 - "$@" <<'PYPORT'
import socket, os
base = 8000 + (int(os.environ.get("SLURM_JOB_ID", "0")) % 900)
for port in range(base, base + 100):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        continue
    finally:
        s.close()
    print(port)
    break
else:
    raise SystemExit("foresight: no free port in range")
PYPORT
    ) || { echo "foresight: could not claim a port" >&2; exit 1; }
fi
echo "foresight: using port $FORESIGHT_PORT" >&2
FORESIGHT_WAIT_TIMEOUT="${FORESIGHT_WAIT_TIMEOUT:-7200}"
FORESIGHT_VLLM_JOBS="${FORESIGHT_VLLM_JOBS:-}"
FORESIGHT_ADAPTER="${FORESIGHT_ADAPTER:-local}"
FORESIGHT_FAKE_UPSTREAM="${FORESIGHT_FAKE_UPSTREAM:-0}"
FORESIGHT_FAKE_PORT="${FORESIGHT_FAKE_PORT:-8011}"
FORESIGHT_MAX_MODEL_LEN="${FORESIGHT_MAX_MODEL_LEN:-32768}"
FORESIGHT_AUX_MAX_OUTPUT="${FORESIGHT_AUX_MAX_OUTPUT:-4096}"

. "$REPO_DIR/deploy/tau-slurm/lib/common.sh"
foresight_resolve_root --root "$FORESIGHT_ROOT"

# conda run takes -p for a path and -n for a name. Computed once, here, because
# both the fake_upstream launch and the server launch need it -- an inline
# `echo -n` would have printed nothing at all, -n being echo's own flag.
if echo "$FORESIGHT_PY_ENV" | grep -q /; then
    py_env_flag="-p"
else
    py_env_flag="-n"
fi

fake_pid=""
if [ "$FORESIGHT_FAKE_UPSTREAM" = "1" ]; then
    # No vLLM, no GPU, no endpoint files: the "model" runs in this job. Bodies
    # are recorded so the exact request the target received can be read back
    # byte for byte, which is the one thing the trace cannot show.
    echo "foresight: starting fake_upstream on port $FORESIGHT_FAKE_PORT" >&2
    conda run --no-capture-output "$py_env_flag" "$FORESIGHT_PY_ENV" \
        python "$REPO_DIR/tools/fake_upstream.py" \
        --port "$FORESIGHT_FAKE_PORT" \
        --record "$FORESIGHT_RUN_DIR/upstream-bodies.jsonl" &
    fake_pid=$!
    i=0
    until curl -fsS -m 3 -X POST "http://127.0.0.1:$FORESIGHT_FAKE_PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"fake-aux","messages":[{"role":"user","content":"ping"}]}' >/dev/null 2>&1; do
        i=$((i + 1))
        if [ "$i" -ge 30 ] || ! kill -0 "$fake_pid" 2>/dev/null; then
            echo "foresight: fake_upstream never became ready" >&2
            exit 1
        fi
        sleep 2
    done
    target_base_url="http://127.0.0.1:$FORESIGHT_FAKE_PORT/v1"
    aux_base_url="$target_base_url"
    echo "foresight: fake_upstream ready at $target_base_url" >&2
else
    target_endpoint_file="$FORESIGHT_RUN_DIR/target.endpoint"
    aux_endpoint_file="$FORESIGHT_RUN_DIR/aux.endpoint"

    echo "foresight: node=$(hostname) waiting for $target_endpoint_file ..." >&2
    foresight_wait_for_file "$target_endpoint_file" "$FORESIGHT_WAIT_TIMEOUT" "$FORESIGHT_VLLM_JOBS"
    echo "foresight: node=$(hostname) waiting for $aux_endpoint_file ..." >&2
    foresight_wait_for_file "$aux_endpoint_file" "$FORESIGHT_WAIT_TIMEOUT" "$FORESIGHT_VLLM_JOBS"

    target_base_url="$(cat "$target_endpoint_file")"
    aux_base_url="$(cat "$aux_endpoint_file")"
fi
echo "foresight: target=$target_base_url aux=$aux_base_url" >&2

config_file="$FORESIGHT_RUN_DIR/config.yaml"
trace_file="$FORESIGHT_RUN_DIR/trace.jsonl"

# The aux agent's opencode home. Node-local, and deliberately NOT under the run
# directory: opencode materializes a node_modules tree on first use in a fresh
# home, measured at >6 min onto this cluster's NFS versus <20s on /tmp. Scoped
# by job id so two concurrent runs on one node cannot share a session DB.
aux_home="${FORESIGHT_AUX_HOME:-/tmp/foresight-aux-home-${SLURM_JOB_ID:-$$}}"

case "$FORESIGHT_ADAPTER" in
    generic)
        # V1-V3: no aux agent, so no workspace, no guard, no opencode home.
        sed \
            -e "s#@TARGET_BASE_URL@#$target_base_url#" \
            -e "s#@AUX_BASE_URL@#$aux_base_url#" \
            "$REPO_DIR/configs/vllm.yaml.tmpl" > "$config_file"
        ;;
    local)
        : "${FORESIGHT_WORKSPACE:?FORESIGHT_WORKSPACE must be set when FORESIGHT_ADAPTER=local (launch.sh --workspace)}"
        if [ "$FORESIGHT_FAKE_UPSTREAM" = "1" ]; then
            tmpl="$REPO_DIR/configs/fake.yaml.tmpl"
        else
            tmpl="$REPO_DIR/configs/vllm-local.yaml.tmpl"
        fi
        if [ ! -d "$FORESIGHT_WORKSPACE" ]; then
            echo "foresight: workspace is not a directory: $FORESIGHT_WORKSPACE" >&2
            exit 1
        fi
        agent_bin="$(foresight_agent_dir)/bin/opencode"
        if [ ! -x "$agent_bin" ]; then
            echo "foresight: opencode wrapper not found: $agent_bin" >&2
            echo "foresight: run deploy/tau-slurm/setup/install_opencode.sh first" >&2
            exit 1
        fi

        # Point aux back at foresight, not at vLLM: its calls then re-enter as
        # aux-model, bypass the pipeline, and land in the same trace file. This
        # is setup only -- foresight execs the opencode wrapper itself, so no
        # machinery of ours sits on the aux path.
        #
        # --context MUST match the vLLM job's --max-model-len, and --output must
        # leave room under it for a prompt that grows every turn. opencode
        # otherwise assumes a 32000-token output budget for a model it does not
        # recognise, which against --max-model-len 32768 rejects every request
        # before the prompt is even considered -- and then retries forever. That
        # is why max_model_len is exported to this job at all.
        "$REPO_DIR/tools/opencode_home.sh" \
            --home "$aux_home" \
            --model aux-model \
            --base-url "http://127.0.0.1:$FORESIGHT_PORT/v1" \
            --context "$FORESIGHT_MAX_MODEL_LEN" \
            --output "$FORESIGHT_AUX_MAX_OUTPUT"

        sed \
            -e "s#@TARGET_BASE_URL@#$target_base_url#" \
            -e "s#@AUX_BASE_URL@#$aux_base_url#" \
            -e "s#@WORKSPACE@#$FORESIGHT_WORKSPACE#" \
            -e "s#@AGENT_BIN@#$agent_bin#" \
            -e "s#@AUX_HOME@#$aux_home#" \
            -e "s#@TRACE_PATH@#$trace_file#" \
            -e "s#@FAKE_BASE_URL@#$target_base_url#" \
            "$tmpl" > "$config_file"
        ;;
    *)
        echo "foresight: unknown FORESIGHT_ADAPTER: $FORESIGHT_ADAPTER (want local or generic)" >&2
        exit 1
        ;;
esac
echo "foresight: rendered $config_file (adapter=$FORESIGHT_ADAPTER)" >&2

# -- aux transcript collection ------------------------------------------
#
# trace.jsonl already shows HOW MANY model calls the aux agent made (one
# role:"aux" row each) and what it finally answered. What it cannot show is
# what happened INSIDE a turn: whether the model emitted a tool call, in what
# wire format, whether the server's --tool-call-parser recognised it, and what
# the tool returned. Two rows means "no tool round-trip happened" but never
# says why -- and that distinction is the whole question when evaluating a new
# checkpoint's agentic ability.
#
# opencode already records all of it. Its session DB holds one step-start/
# step-finish pair per model call plus a type:"tool" part per invocation,
# carrying the tool name, its input and its output; `opencode export` renders
# one session as JSON. So nothing here needs to log wire bytes -- and a
# proxy-side log would be strictly worse anyway, because it would show what the
# model SAID without showing how opencode INTERPRETED it, which is the half
# that failed on Qwen2.5-Coder-7B.
#
# The gap was never recording, it was retrieval: the home lives on the compute
# node's /tmp (deliberately -- see the note by aux_home above), killable nodes
# reclaim it, and nothing copied it to shared storage. Hence this.
#
# ON EXIT ONLY, never on a timer. An earlier version re-exported every 120s to
# allow watching a run in progress. That is actively unsafe: exporting means
# running `opencode` against the SAME OPENCODE_HOME the aux agent uses, and two
# opencode processes sharing one home deadlock its SQLite --
# "Error: Unexpected error / database is locked" -- which surfaces as a 502
# aux_failure and kills the caller's request. With adapter.timeout_s at 1800s, a
# single aux run could overlap a dozen ticks. The loop risked breaking the very
# runs it was meant to document, and the mid-run visibility it bought went
# unused. To inspect a live run, attach to the job instead
# (`srun --overlap --jobid=<proxy-job>`), which touches nothing.
#
# The trap covers preemption and scancel, since Slurm sends SIGTERM before
# SIGKILL. A hard SIGKILL loses the transcripts; that is the accepted trade for
# not corrupting live runs.
#
# Two rules learned the hard way, both load-bearing:
#   * ALWAYS pass an explicit session id. `opencode export` with no argument
#     blocks forever waiting on input.
#   * ALWAYS wrap in `timeout`. A killed opencode leaves a lock behind that
#     makes every subsequent opencode command in that home hang.
aux_collect=0
[ "$FORESIGHT_ADAPTER" = "local" ] && aux_collect=1
aux_out_dir="$FORESIGHT_RUN_DIR/aux-agent"

collect_aux_sessions() {
    [ "$aux_collect" = "1" ] || return 0
    [ -d "$aux_home/.local/share/opencode" ] || return 0
    mkdir -p "$aux_out_dir" 2>/dev/null || return 0

    # opencode's own log: cheap, and the only place a harness-level error
    # (provider misconfiguration, plugin failure) is visible at all.
    cp -f "$aux_home/.local/share/opencode/log/opencode.log" \
          "$aux_out_dir/opencode.log" 2>/dev/null || true

    # Sessions are scoped by project, so this must run with cwd = the workspace
    # the aux agent explored, exactly as LocalAdapter spawns it.
    ids="$(cd "$FORESIGHT_WORKSPACE" 2>/dev/null && \
           OPENCODE_HOME="$aux_home" timeout 120 "$agent_bin" session list 2>/dev/null \
           | grep -oE 'ses_[A-Za-z0-9]{20,}')" || true
    for sid in $ids; do
        (cd "$FORESIGHT_WORKSPACE" 2>/dev/null && \
         OPENCODE_HOME="$aux_home" timeout 120 "$agent_bin" export "$sid" \
             > "$aux_out_dir/$sid.json.part" 2>/dev/null) || true
        if [ ! -s "$aux_out_dir/$sid.json.part" ]; then
            rm -f "$aux_out_dir/$sid.json.part"
            continue
        fi
        # Keep only sessions belonging to THIS run's workspace.
        #
        # OPENCODE_HOME does not isolate opencode's session database the way
        # this deployment assumes: measured directly, sessions from every run --
        # different job, different node, different workspace -- accumulate in one
        # shared DB, and `opencode session list` returns all of them. Exporting
        # unfiltered puts another experiment's transcripts into this run's
        # evidence, which is worse than having none.
        #
        # The session's own `directory` field is the reliable discriminator,
        # since LocalAdapter always spawns the agent with cwd = the workspace.
        if grep -q "\"directory\"[[:space:]]*:[[:space:]]*\"$FORESIGHT_WORKSPACE\"" \
                "$aux_out_dir/$sid.json.part" 2>/dev/null; then
            mv -f "$aux_out_dir/$sid.json.part" "$aux_out_dir/$sid.json"
        else
            rm -f "$aux_out_dir/$sid.json.part"
        fi
    done
}

foresight_endpoint_file="$FORESIGHT_RUN_DIR/foresight.endpoint"
rm -f "$foresight_endpoint_file"

server_pid=""
cleanup() {
    rm -f "$foresight_endpoint_file"
    # The server is stopped FIRST, so no aux agent can still be mid-run holding
    # opencode's SQLite when the export below opens it. Ordering is the whole
    # reason this is safe -- see the note by collect_aux_sessions.
    [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null || true
    [ -n "$fake_pid" ] && kill "$fake_pid" 2>/dev/null || true
    # Then capture the transcripts, before the node takes /tmp back.
    collect_aux_sessions || true
}
trap cleanup EXIT INT TERM

if [ "$aux_collect" = "1" ]; then
    mkdir -p "$aux_out_dir"
    echo "foresight: aux transcripts -> $aux_out_dir (written when the job ends)" >&2
fi


cd "$REPO_DIR"
conda run --no-capture-output "$py_env_flag" "$FORESIGHT_PY_ENV" \
    python -m foresight.server --config "$config_file" --host 0.0.0.0 --port "$FORESIGHT_PORT" &
server_pid=$!

# Same pattern as serve_vllm.sbatch: publish the endpoint only once /health
# actually answers, not once the process starts.
ready=0
i=0
while [ "$i" -lt 60 ]; do   # up to 5 min -- foresight itself starts fast
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "foresight: proxy exited before becoming ready" >&2
        exit 1
    fi
    # /health alone is not proof of identity -- a neighbouring proxy on the same
    # port answers it identically. The liveness check above (kill -0) plus a
    # port we exclusively bound is what makes this ours.
    if curl -fsS -m 5 "http://127.0.0.1:$FORESIGHT_PORT/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 5
    i=$((i + 1))
done

if [ "$ready" -ne 1 ]; then
    echo "foresight: proxy did not become ready in time" >&2
    exit 1
fi

echo "http://$(hostname):$FORESIGHT_PORT/v1" > "$foresight_endpoint_file"
echo "foresight: ready, endpoint published to $foresight_endpoint_file" >&2

wait "$server_pid"
