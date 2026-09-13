#!/bin/sh
# Write an opencode HOME that points at a given OpenAI-compatible base URL.
#
# NOT a harness launcher -- it launches nothing. It performs the *setup* half
# of SWE-CI's opencode integration (setup_opencode,
# SWE-CI/src/swe_ci/benchmark/agents/opencode.py:16-67): the two files opencode
# reads at startup to learn about a custom provider. Running opencode is then a
# plain `opencode run --model custom/<model>`, which is what belongs in
# LocalAdapter's `adapter.agent_cmd` -- argv naming a real harness, exactly as
# the design's "we define no tools" constraint requires.
#
# Splitting setup from invocation is what keeps the measured path clean. If the
# aux command were a script of ours that configured-then-ran, every aux run
# would go through machinery we invented; instead the config is written once,
# out of band, and foresight execs the harness itself.
#
# Usage:
#   tools/opencode_home.sh --home DIR --model NAME --base-url URL
#                          [--context N] [--output N]
#
# --output is not cosmetic. For a model it does not recognise, opencode defaults
# the output limit to 32000 and sends it as `max_tokens` on every request. Served
# by vLLM with the usual --max-model-len 32768, that leaves 768 tokens for the
# prompt, and opencode's own system prompt is larger than that -- so every single
# request 400s with "maximum context length is 32768 tokens... you requested
# 32000 output tokens". opencode retries forever: 2901 rejected requests before
# the run was killed at 25 minutes, no progress, GPU allocation burning throughout.
#
# It reads exactly nothing about the endpoint it is pointed at, so the limits
# have to be declared here. Keep --context equal to vLLM's --max-model-len (it is
# what opencode uses to decide when to compact a session) and --output a modest
# fraction of it, leaving room for a prompt that grows every turn.
#
# Then either:
#   OPENCODE_HOME=DIR opencode run --model custom/NAME "prompt"
# or, in a foresight config:
#   adapter:
#     agent_cmd: ["/path/to/agent/bin/opencode", "run", "--model", "custom/NAME", "{prompt}"]
#     env: {OPENCODE_HOME: DIR}
#
# --home MUST be on local disk, not NFS. On its first run in a fresh home
# opencode materializes its plugin dependencies -- a node_modules tree of
# several thousand small files -- which took over 6 minutes onto this cluster's
# NFS versus under 20 seconds on node-local /tmp, measured directly. Give aux
# and target DIFFERENT homes: opencode keeps its session SQLite under
# $HOME/.local/share/opencode, and sharing it corrupts both state and (under
# SWE-CI) the benchmark's own token accounting.

set -eu

oc_home=""
model=""
base_url=""
context="32768"
output="4096"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --home)     oc_home="$2"; shift 2 ;;
        --model)    model="$2"; shift 2 ;;
        --base-url) base_url="$2"; shift 2 ;;
        --context)  context="$2"; shift 2 ;;
        --output)   output="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "foresight: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$oc_home" ] || [ -z "$model" ] || [ -z "$base_url" ]; then
    echo "usage: $0 --home DIR --model NAME --base-url URL" >&2
    exit 1
fi

mkdir -p "$oc_home/.local/share/opencode" "$oc_home/.config/opencode"

# Mirrors setup_opencode() -- same two files, same shape, with the "custom"
# provider pointed at foresight instead of at a model endpoint directly. That
# indirection is deliberate (design, §Server): aux traffic re-enters foresight
# as `aux-model`, bypasses the pipeline, and so lands in the same trace file
# and the same Backend layer as target traffic.
cat > "$oc_home/.local/share/opencode/auth.json" <<EOF
{
    "custom": {
        "type": "api",
        "key": "dummy"
    }
}
EOF

cat > "$oc_home/.config/opencode/opencode.json" <<EOF
{
    "\$schema": "https://opencode.ai/config.json",
    "permission": "allow",
    "provider": {
        "custom": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "custom",
            "options": {
                "baseURL": "$base_url"
            },
            "models": {
                "$model": {
                    "name": "$model",
                    "limit": {
                        "context": $context,
                        "output": $output
                    }
                }
            }
        }
    }
}
EOF

echo "foresight: opencode home ready: $oc_home (model=$model base_url=$base_url" \
     "context=$context output=$output)" >&2
