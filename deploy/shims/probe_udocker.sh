#!/bin/bash
# Does udocker support the ONE pattern milestones 3-4 require?
#
# The pattern: start a long-lived container, then `exec` into it repeatedly
# from separate processes. SWE-CI does it (run_container + call_cli_agent +
# copy_*), mini does it, and SweCiAdapter adds a second `exec` of its own into
# the same container. udocker is daemonless and may not support it at all --
# the design plan calls this the thing to test FIRST, because a failure here
# blocks M3 regardless of adapter quality.
#
# This script answers that question and nothing else. It needs no GPU, no
# dataset and no SWE-CI checkout; one small public image is enough.
#
#     UDOCKER_DIR=/scratch/$USER/udocker ./probe_udocker.sh
#
# Exit 0 means the shim approach is viable and milestone 3 can proceed to a
# real one-task run. Any other exit means it is not, and the fallback options
# are in the design plan's risk 1 (singularity for mini, sb-cli for scoring).

set -u

UDOCKER="${UDOCKER:-udocker}"
IMAGE="${PROBE_IMAGE:-python:3.11-slim}"
NAME="foresight-probe-$$"

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILED=1; }
step() { printf '\n== %s\n' "$1"; }

FAILED=0
cleanup() {
  [ -n "${RUN_PID:-}" ] && kill "$RUN_PID" 2>/dev/null
  "$UDOCKER" rm "$NAME" >/dev/null 2>&1
}
trap cleanup EXIT

step "0. udocker is present and reports a version"
if command -v "$UDOCKER" >/dev/null 2>&1; then
  pass "$($UDOCKER --version 2>&1 | head -1)"
else
  fail "$UDOCKER not found; set UDOCKER=/path/to/udocker"
  exit 1
fi
printf '  UDOCKER_DIR=%s\n' "${UDOCKER_DIR:-$HOME/.udocker (default -- put this on scratch)}"

step "1. pull and create ($IMAGE)"
if "$UDOCKER" pull "$IMAGE" >/dev/null 2>&1; then
  pass "pull"
else
  fail "pull -- no network, or no registry access from this node"
fi
if "$UDOCKER" create --name="$NAME" "$IMAGE" >/dev/null 2>&1; then
  pass "create"
else
  fail "create"
  exit 1
fi

step "2. a foreground run works at all"
if OUT=$("$UDOCKER" run "$NAME" echo hello 2>&1); then
  case "$OUT" in *hello*) pass "run echo" ;; *) fail "run echo -> $OUT" ;; esac
else
  fail "run echo -> $OUT"
fi

step "3. THE question: a backgrounded run stays alive"
# What SWE-CI's `docker run -d ... tail -f /dev/null` becomes.
nohup "$UDOCKER" run "$NAME" tail -f /dev/null >/tmp/$NAME.log 2>&1 &
RUN_PID=$!
sleep 3
if kill -0 "$RUN_PID" 2>/dev/null; then
  pass "background run survived 3s (pid $RUN_PID)"
else
  fail "background run died; see /tmp/$NAME.log"
  sed 's/^/        /' /tmp/$NAME.log | head -5
fi

step "4. THE other question: exec into that live container, twice"
# Twice, from separate processes, because the target harness and the aux
# harness both do it -- concurrently, in the real thing.
if OUT=$("$UDOCKER" exec "$NAME" sh -c 'echo first' 2>&1); then
  case "$OUT" in *first*) pass "exec #1" ;; *) fail "exec #1 -> $OUT" ;; esac
else
  fail "exec #1 -> $OUT"
fi
if OUT=$("$UDOCKER" exec "$NAME" sh -c 'echo second' 2>&1); then
  case "$OUT" in *second*) pass "exec #2" ;; *) fail "exec #2 -> $OUT" ;; esac
else
  fail "exec #2 -> $OUT"
fi

step "5. writes made by one exec are visible to the next"
# The whole premise of aux reading the code the target is editing.
"$UDOCKER" exec "$NAME" sh -c 'echo marker > /tmp/probe-marker' >/dev/null 2>&1
if OUT=$("$UDOCKER" exec "$NAME" cat /tmp/probe-marker 2>&1); then
  case "$OUT" in *marker*) pass "shared writable layer" ;; *) fail "not shared -> $OUT" ;; esac
else
  fail "could not read the marker back -> $OUT"
fi

step "6. -w and -e reach the process (the HOME override depends on this)"
if OUT=$("$UDOCKER" exec --workdir=/tmp --env=HOME=/tmp/aux-home "$NAME" \
         sh -c 'echo $PWD $HOME' 2>&1); then
  case "$OUT" in
    *"/tmp /tmp/aux-home"*) pass "-w and -e" ;;
    *) fail "-w/-e not honoured -> $OUT" ;;
  esac
else
  fail "-w/-e -> $OUT"
fi

step "7. does the container have its own IP? (resolution depends on it)"
HOST_IP=$(hostname -i 2>/dev/null | awk '{print $1; exit}')
GUEST_IP=$("$UDOCKER" exec "$NAME" sh -c 'hostname -i 2>/dev/null' 2>&1 | awk '{print $1; exit}')
printf '  host: %s\n  container: %s\n' "${HOST_IP:-?}" "${GUEST_IP:-?}"
if [ -n "$GUEST_IP" ] && [ "$GUEST_IP" != "$HOST_IP" ]; then
  pass "per-container address -- client-IP resolution can work"
else
  printf '  NOTE  no distinct address: the container shares the host namespace.\n'
  printf '        client-IP -> container resolution is IMPOSSIBLE here, and the\n'
  printf '        adapter falls back to "the sole running container", which is\n'
  printf '        exact ONLY at evolve.max_workers = 1. Set it to 1.\n'
fi

step "8. udocker has no cp: how would a host tree get in?"
ROOT=$("$UDOCKER" inspect -p "$NAME" 2>/dev/null)
if [ -n "$ROOT" ] && [ -d "$ROOT" ]; then
  pass "container root is a host directory: $ROOT"
  printf '        SWE-CI calls `docker cp` ~8 times per epoch, so the shim needs\n'
  printf '        this path to implement it. Confirm the layout before writing it.\n'
else
  fail "inspect -p gave no usable root; `docker cp` cannot be emulated"
fi

printf '\n'
if [ "$FAILED" -eq 0 ]; then
  echo "RESULT: viable. Steps 3-5 are the ones that matter, and they passed."
  echo "Next: implement `cp` in the shim (step 8), then a one-task SWE-CI run"
  echo "with max_epoch = 1 and evolve.max_workers = 1."
else
  echo "RESULT: NOT viable as-is. If step 3, 4 or 5 failed, the long-lived"
  echo "container + repeated exec pattern is unavailable and milestone 3 is"
  echo "blocked -- see risk 1 in foresight-design-plan.md for the alternatives."
fi
exit "$FAILED"
