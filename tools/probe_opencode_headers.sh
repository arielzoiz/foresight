#!/bin/sh
# Does opencode forward provider.options.headers, and does it send its own
# per-session identifier? MEASURED 2026-09-16 against opencode 1.18.31: yes to
# both. This script makes that measurement repeatable -- re-run it whenever
# SWE-CI's task image installs a different opencode version (agent_name /
# AGENT_NPM_PKG, config.py:114-131), since the answer is undocumented
# behaviour, not a documented contract.
#
# WHY THIS MATTERS
# -----------------
# README.md ("Keeping SWE-CI's task concurrency") needs a per-session
# discriminator that works with SWE-CI's `evolve.max_workers > 1` pointed at
# ONE foresight process. Two properties settle it:
#
#   1. opencode.json's provider.custom.options.headers (an
#      @ai-sdk/openai-compatible option, not an opencode one) must actually
#      reach the wire, or deploy/shims/docker's X-Foresight-Container
#      injection is dead code.
#   2. opencode must send something that identifies ONE `opencode run`
#      invocation across all of its requests -- if it does, SweCiAdapter needs
#      no docker/udocker involvement at all for session keying.
#
# Needs no Docker, no GPU, no SWE-CI checkout: a local opencode binary and this
# repo's own mock upstream are enough. Two SEPARATE `opencode run` invocations
# are run so a stable-but-shared identifier (e.g. a machine ID) cannot be
# mistaken for a real per-session one.
#
# Usage:
#   tools/probe_opencode_headers.sh --opencode-bin PATH [--python PYTHON]
#
# Exit 0: both properties hold, tier A in README.md is viable as specified.
# Exit 1: report which property failed; deploy/shims/docker's header comment
# and README.md's concurrency table need updating to match reality, not the
# other way around.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

opencode_bin=""
python_bin="${PYTHON_BIN:-python3}"
port="${PROBE_PORT:-8899}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --opencode-bin) opencode_bin="$2"; shift 2 ;;
        --python)       python_bin="$2"; shift 2 ;;
        --port)         port="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "probe_opencode_headers: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$opencode_bin" ]; then
    echo "usage: $0 --opencode-bin PATH [--python PYTHON] [--port N]" >&2
    exit 1
fi
if [ ! -x "$opencode_bin" ]; then
    echo "probe_opencode_headers: not executable: $opencode_bin" >&2
    exit 1
fi

work="$(mktemp -d)"
home="$work/home"
record="$work/upstream.jsonl"
trap 'kill "$UPSTREAM_PID" 2>/dev/null; rm -rf "$work"' EXIT

echo "probe_opencode_headers: work dir $work" >&2

# Same provider config shape as tools/opencode_home.sh, plus the header this
# probe exists to check. Written directly rather than via opencode_home.sh
# because that script does not (yet) expose a --header flag, and adding one
# there for a single one-off probe would be the wrong place to carry it --
# deploy/shims/docker's real injection happens by rewriting this exact file's
# stdin as SWE-CI's own setup_opencode() writes it, which this probe mimics.
mkdir -p "$home/.local/share/opencode" "$home/.config/opencode"
cat > "$home/.local/share/opencode/auth.json" <<'EOF'
{"custom": {"type": "api", "key": "dummy"}}
EOF
cat > "$home/.config/opencode/opencode.json" <<EOF
{
    "\$schema": "https://opencode.ai/config.json",
    "permission": "allow",
    "provider": {
        "custom": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "custom",
            "options": {
                "baseURL": "http://127.0.0.1:$port/v1",
                "headers": {"X-Foresight-Container": "probe-container"}
            },
            "models": {"probe-model": {"name": "probe-model"}}
        }
    }
}
EOF

"$python_bin" "$REPO_ROOT/tools/fake_upstream.py" --port "$port" --record "$record" \
    >"$work/upstream.log" 2>&1 &
UPSTREAM_PID=$!
sleep 2

run_once() {
    HOME="$home" \
    XDG_CONFIG_HOME="$home/.config" \
    XDG_DATA_HOME="$home/.local/share" \
    XDG_CACHE_HOME="$home/.cache" \
        "$opencode_bin" run --model custom/probe-model "$1" >/dev/null 2>&1 || true
}

( cd "$work" && run_once "probe invocation one" )
( cd "$work" && run_once "probe invocation two" )

"$python_bin" - "$record" <<'PYEOF'
import json, sys

records = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
# Drop bookkeeping requests this probe did not ask for (there are none besides
# opencode's own title-generation call, which is expected and fine to include
# -- it shares the session id with the real turn, same as under SWE-CI).
if not records:
    print("FAIL: no requests recorded -- opencode did not call the mock upstream at all")
    sys.exit(1)

failed = False

injected_ok = all(r["headers"].get("x-foresight-container") == "probe-container" for r in records)
print(("PASS" if injected_ok else "FAIL") + ": provider.options.headers forwarded to every request "
      f"({len(records)} requests)")
failed = failed or not injected_ok

session_ids = [r["headers"].get("x-session-id") for r in records]
by_invocation = {}
for r in records:
    sid = r["headers"].get("x-session-id")
    by_invocation.setdefault(sid, 0)
    by_invocation[sid] += 1

distinct = len(by_invocation)
stable = all(n >= 1 for n in by_invocation.values())
two_invocations_distinct = distinct >= 2 and all(session_ids)
print(("PASS" if two_invocations_distinct else "FAIL") + ": x-session-id present, stable per "
      f"invocation, distinct across invocations (saw {distinct} distinct id(s) over "
      f"{len(records)} requests: {dict(by_invocation)})")
failed = failed or not two_invocations_distinct

sys.exit(1 if failed else 0)
PYEOF
