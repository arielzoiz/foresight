#!/bin/sh
# Copy one LOCAL (non-Slurm) run's evidence into results/ in the repo.
#
# tools/collect_run.sh does this for a Slurm run directory (config.yaml,
# trace.jsonl and aux-agent/ all under one $run_dir, plus sacct/vLLM specifics
# that only exist on that cluster). A run against a local model server (Ollama,
# llama.cpp, LM Studio -- anything OpenAI-compatible on localhost) has no run
# directory and no sacct: the config lives wherever you put it, the trace lives
# wherever adapter.trace.path points, and aux's session state lives under
# whatever --aux-home you gave foresight's adapter.env.HOME. This script takes
# those as separate arguments instead of assuming a shared parent directory,
# and replaces collect_run.sh's Slurm/vLLM-specific files with a local
# equivalent of "how was this run resourced": the serving stack's own version,
# what it reports about the model (params, quantization, context, declared
# capabilities), and this host's hardware -- the things that decide whether a
# result reproduces on someone else's machine.
#
# What it takes, and what it deliberately leaves behind: same as
# collect_run.sh's config.yaml / trace.jsonl / aux-agent/ / prompts.txt /
# summary.txt. No vllm.txt (nothing to collect locally); run-settings.txt is
# reworked, see above.
#
# Usage:
#   tools/collect_local_run.sh --config PATH --trace PATH --workspace DIR \
#       --aux-home DIR --label NAME [--ollama-url URL]
#   tools/collect_local_run.sh --help
#
# Example (matching the local-model smoke test in README's M2):
#   tools/collect_local_run.sh \
#       --config /tmp/local-qwen3.yaml \
#       --trace /tmp/foresight-scratch-trace.jsonl \
#       --workspace /tmp/foresight-scratch \
#       --aux-home /tmp/foresight-aux-home \
#       --label qwen3-8b-ollama-local
#
# Then write results/<label>/findings.md by hand -- what the run showed, and
# what it does NOT show -- and commit the directory. That file is the only
# part no script can produce.

set -eu

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

config="" trace="" workspace="" aux_home="" label="" ollama_url="http://127.0.0.1:11434"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)     config="$2"; shift 2 ;;
        --trace)      trace="$2"; shift 2 ;;
        --workspace)  workspace="$2"; shift 2 ;;
        --aux-home)   aux_home="$2"; shift 2 ;;
        --label)      label="$2"; shift 2 ;;
        --ollama-url) ollama_url="$2"; shift 2 ;;
        *)
            echo "collect_local_run: unknown argument: $1 (see --help)" >&2
            exit 1
            ;;
    esac
done

if [ -z "$config" ] || [ -z "$trace" ] || [ -z "$workspace" ] || [ -z "$aux_home" ] || [ -z "$label" ]; then
    echo "usage: $0 --config PATH --trace PATH --workspace DIR --aux-home DIR --label NAME (see --help)" >&2
    exit 1
fi

# A label becomes a directory name under results/. Rejected rather than
# sanitised: silently rewriting someone's label would put the output somewhere
# they did not ask for and then report success.
case "$label" in
    */*|.*|"")
        echo "collect_local_run: label must be a plain directory name (no '/', no leading '.'): $label" >&2
        exit 1
        ;;
esac

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=""
    echo "collect_local_run: WARNING -- no python found; prompts.txt and summary.txt will be skipped" >&2
fi

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
out="$repo_dir/results/$label"
mkdir -p "$out"

# GNU coreutils' `timeout` is not on PATH by default on macOS (not even as
# `gtimeout`, unless coreutils was brew-installed) -- and going without it is
# not an option: a killed opencode leaves a lock behind that makes every
# subsequent opencode command in that home hang, per foresight.sbatch's own
# comment. Falls back to a plain background-and-kill wrapper, portable to any
# POSIX sh.
run_with_timeout() {
    dur="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "$dur" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$dur" "$@"
    else
        "$@" &
        cmd_pid=$!
        ( sleep "$dur"; kill -TERM "$cmd_pid" 2>/dev/null ) &
        watcher_pid=$!
        wait "$cmd_pid" 2>/dev/null
        status=$?
        kill "$watcher_pid" 2>/dev/null
        wait "$watcher_pid" 2>/dev/null
        return "$status"
    fi
}

# -- config and trace -------------------------------------------------------

if [ -f "$config" ]; then
    cp -f "$config" "$out/config.yaml"
else
    echo "collect_local_run: WARNING -- no config file at $config" >&2
fi

if [ -f "$trace" ]; then
    cp -f "$trace" "$out/trace.jsonl"
else
    echo "collect_local_run: WARNING -- no trace file at $trace" >&2
fi

# -- aux session exports ------------------------------------------------------
# Same two rules as foresight.sbatch's collect_aux_sessions, both load-bearing:
#   * ALWAYS pass an explicit session id -- `opencode export` with none blocks
#     forever waiting on input.
#   * Filter by the session's own `directory` field, not by "everything opencode
#     knows about" -- OPENCODE_HOME does not isolate the session database
#     (measured on Slurm, and nothing here suggests it isolates it locally
#     either), so an unfiltered export can pull in another run's transcripts.
agent_bin="$(command -v opencode || true)"
aux_out_dir="$out/aux-agent"
if [ -n "$agent_bin" ] && [ -d "$aux_home/.local/share/opencode" ]; then
    mkdir -p "$aux_out_dir"
    cp -f "$aux_home/.local/share/opencode/log/opencode.log" \
          "$aux_out_dir/opencode.log" 2>/dev/null || true

    ids="$(cd "$workspace" 2>/dev/null && \
           OPENCODE_HOME="$aux_home" run_with_timeout 120 "$agent_bin" session list 2>/dev/null \
           | grep -oE 'ses_[A-Za-z0-9]{20,}')" || true
    for sid in $ids; do
        (cd "$workspace" 2>/dev/null && \
         OPENCODE_HOME="$aux_home" run_with_timeout 120 "$agent_bin" export "$sid" \
             > "$aux_out_dir/$sid.json.part" 2>/dev/null) || true
        if [ ! -s "$aux_out_dir/$sid.json.part" ]; then
            rm -f "$aux_out_dir/$sid.json.part"
            continue
        fi
        if grep -q "\"directory\"[[:space:]]*:[[:space:]]*\"$workspace\"" \
                "$aux_out_dir/$sid.json.part" 2>/dev/null; then
            mv -f "$aux_out_dir/$sid.json.part" "$aux_out_dir/$sid.json"
        else
            rm -f "$aux_out_dir/$sid.json.part"
        fi
    done
    [ -n "$(ls "$aux_out_dir" 2>/dev/null)" ] || rmdir "$aux_out_dir" 2>/dev/null || true
else
    echo "collect_local_run: WARNING -- no opencode home at $aux_home (generic adapter, or aux never ran)" >&2
fi

# -- local run-settings: what would have to match to reproduce this ---------
# The Slurm equivalent is sacct (partition, node, GPUs, time limit). Locally
# there is no scheduler, so the things that actually decide whether this
# result reproduces are: which serving stack, which exact model tag and
# quantization, and what hardware it ran on -- a 30B that OOMs on 16GB unified
# memory is a different experiment from the same config on a workstation.
{
    echo "# Run settings for $label (local, non-Slurm)"
    echo "# Recovered from the local Ollama server and this host; see config.yaml"
    echo "# for the pipeline's own settings."
    echo
    echo "## foresight"
    echo "  commit=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "  dirty=$(test -n "$(git -C "$repo_dir" status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo "  collected=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "## host"
    if command -v sw_vers >/dev/null 2>&1; then
        echo "  os=$(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"
    else
        echo "  os=$(uname -sr)"
    fi
    echo "  arch=$(uname -m)"
    if command -v sysctl >/dev/null 2>&1; then
        echo "  cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || echo unknown)"
        mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
        [ "$mem_bytes" -gt 0 ] 2>/dev/null && echo "  memory_gb=$((mem_bytes / 1073741824))"
    fi
    echo
    echo "## ollama"
    if command -v ollama >/dev/null 2>&1; then
        echo "  version=$(ollama --version 2>/dev/null | head -1)"
        echo "  url=$ollama_url"
        # Every model the config names (best-effort text search -- config.yaml
        # is Jinja/YAML and this is a startup sanity check, not a parser).
        # [[:space:]]/[^[:space:]], not \s/\S -- BSD sed (macOS default) treats
        # \s as a literal "s", not whitespace, and silently fails to strip the
        # prefix, which then gets word-split by the `for` loop below into
        # garbage entries. Measured directly, not assumed.
        models="$(command grep -oE '^[[:space:]]*model:[[:space:]]*[^[:space:]]+' "$config" 2>/dev/null \
                  | sed -E 's/^[[:space:]]*model:[[:space:]]*//' | sort -u)" || true
        for m in $models; do
            echo "  -- ollama show $m --"
            ollama show "$m" 2>/dev/null | sed 's/^/    /' || echo "    (not available locally)"
        done
    else
        echo "  (ollama not found on PATH)"
    fi
} > "$out/run-settings.txt" 2>/dev/null || true

# -- the prompts, which ARE the experimental condition -----------------------
# Identical to collect_run.sh's extraction: config.yaml's shape does not change
# between a Slurm run and a local one.
[ -n "$PY" ] && "$PY" - "$out" "$out" <<'PYPROMPTS' 2>/dev/null || true
import json, pathlib, sys

run, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
parts = []

task = None
trace = run / "trace.jsonl"
if trace.exists():
    for line in trace.open(encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("role") == "target" and row.get("prompt_in"):
            if task is None or len(row["prompt_in"]) > len(task):
                task = row["prompt_in"]
if task:
    parts.append("=== TASK PROMPT (what the target agent was asked to do) ===\n\n" + task)

cfg = run / "config.yaml"
if cfg.exists():
    text = cfg.read_text(encoding="utf-8", errors="replace")
    for key, label in (("aux_prompt:", "AUX PROMPT (what the aux agent was asked)"),
                       ("  template:", "BUILDER TEMPLATE (how aux's answer is injected)")):
        start = text.find("\n" + key)
        if start == -1:
            continue
        block, indent = [], None
        for line in text[start + 1:].splitlines()[1:]:
            if not line.strip():
                block.append("")
                continue
            lead = len(line) - len(line.lstrip())
            if indent is None:
                indent = lead
            if lead < indent:
                break
            block.append(line[indent:])
        parts.append(f"=== {label} ===\n\n" + "\n".join(block).rstrip())

if parts:
    (out / "prompts.txt").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
PYPROMPTS

# -- aux transcript verdict ---------------------------------------------------
if [ -n "$PY" ] && [ -d "$aux_out_dir" ] && [ -n "$(ls "$aux_out_dir"/*.json 2>/dev/null)" ]; then
    "$PY" "$repo_dir/tools/aux_transcript.py" "$aux_out_dir"/*.json \
        > "$out/summary.txt" 2>&1 || true
fi

echo "collect_local_run: wrote $out"
ls -la "$out"
cat >&2 <<EOF

collect_local_run: next steps
  1. write $out/findings.md -- what the run showed, and what it does NOT show
  2. skim the files above: they carry absolute paths and the task prompt
  3. git add results/$label && git commit
EOF
