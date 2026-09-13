#!/bin/sh
# A STAND-IN FOR THE CALLER. NOT PART OF THE LIBRARY, NOT FOR MEASURED RUNS.
#
# foresight never launches the target agent -- that is the benchmark's job. This
# script pretends to be one: an opencode session pointed at foresight from the
# outside, the role SWE-CI or mini will fill for real. It exists so the Slurm
# topology can be exercised end to end before a benchmark is wired up.
#
# It does NOT stand in for the aux agent. That side is real as of milestone 2:
# LocalAdapter spawns opencode itself, configured by tools/opencode_home.sh,
# which this script also calls rather than duplicating -- so the probe cannot
# drift from what the real aux path sees.
#
# So: fine for hand-debugging and for the V4 latency measurement, never part of
# an experimental run -- a measured result needs a real harness as the caller.
#
# Mirrors: SWE-CI/src/swe_ci/benchmark/agents/opencode.py:148-167 (call_opencode);
# the :16-67 setup_opencode half now lives in tools/opencode_home.sh
#
# Usage:
#   tools/opencode_probe.sh --home DIR --model NAME --base-url URL --prompt TEXT [--opencode-bin PATH] [--root DIR]
#
# --home MUST be on local disk, not NFS-mounted storage (i.e. not under
# $FORESIGHT_ROOT/foresight-runs/...). On its first run in any fresh home,
# opencode materializes its own plugin dependencies -- a node_modules tree of
# several thousand small files -- and writing that onto NFS took over 6
# minutes in direct testing; the identical install on local disk (e.g. /tmp
# on the node actually running this script) took under 20 seconds. Every
# distinct --home pays this cost once, so a run dir per session is the worst
# case for this specific failure mode.
#
# Example (against a foresight proxy running on :8000):
#   tools/opencode_probe.sh \
#       --home /tmp/opencode-probe-home \
#       --model target-model \
#       --base-url http://localhost:8000/v1 \
#       --prompt "Fix the bug in parse_date()"

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
. "$SCRIPT_DIR/../deploy/tau-slurm/lib/common.sh"

probe_home=""
model=""
base_url=""
prompt=""
opencode_bin=""
root_arg=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --home)
            probe_home="$2"
            shift 2
            ;;
        --model)
            model="$2"
            shift 2
            ;;
        --base-url)
            base_url="$2"
            shift 2
            ;;
        --prompt)
            prompt="$2"
            shift 2
            ;;
        --opencode-bin)
            opencode_bin="$2"
            shift 2
            ;;
        --root)
            root_arg="--root $2"
            shift 2
            ;;
        *)
            echo "foresight: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$probe_home" ] || [ -z "$model" ] || [ -z "$base_url" ] || [ -z "$prompt" ]; then
    echo "usage: $0 --home DIR --model NAME --base-url URL --prompt TEXT [--opencode-bin PATH] [--root DIR]" >&2
    exit 1
fi

if [ -z "$opencode_bin" ]; then
    # shellcheck disable=SC2086
    foresight_resolve_root $root_arg
    opencode_bin="$(foresight_agent_dir)/bin/opencode"
fi

if [ ! -x "$opencode_bin" ]; then
    echo "foresight: opencode binary not found or not executable: $opencode_bin" >&2
    echo "foresight: run deploy/tau-slurm/setup/install_opencode.sh first" >&2
    exit 1
fi

# The provider config opencode reads at startup. Shared with the real aux path
# (foresight.sbatch renders the same two files via the same script), so the
# probe cannot drift from what LocalAdapter's harness actually sees -- which was
# the point of the probe.
"$SCRIPT_DIR/opencode_home.sh" --home "$probe_home" --model "$model" --base-url "$base_url"

echo "foresight: [probe] home=$probe_home model=$model base_url=$base_url" >&2

started="$(date +%s)"
# Mirrors call_opencode() in opencode.py:148-167.
# set +e around this call: under set -e, a non-zero exit from the command
# substitution would abort the script on this line, before exit_code=$? ever
# runs -- exactly the failure this script exists to report, not hide.
set +e
output="$(OPENCODE_HOME="$probe_home" "$opencode_bin" run --model "custom/$model" "$prompt" 2>&1)"
exit_code=$?
set -e
elapsed=$(($(date +%s) - started))

echo "----- opencode output -----"
echo "$output"
echo "----------------------------"
echo "foresight: [probe] exit_code=$exit_code elapsed_s=$elapsed" >&2

exit "$exit_code"
