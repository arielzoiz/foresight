# SweCiAdapter against real Docker — first real-Docker run, two bugs found and fixed

**Run:** local (no Slurm), 2026-09-16/17, Apple M1 Pro / 16GB, Docker Desktop
29.8.0
**Task:** `15r10nk__inline-snapshot__3bb05d__e2b9b2` (real SWE-CI task data,
trimmed from the full 100-task `metadata/default.csv` to this one row --
see `metadata/default_all.csv` in the SWE-CI checkout for the full list)
**Models:** `tools/fake_upstream.py` for both target and aux -- deliberately,
per README's own "Verifying `SweCiAdapter` against real Docker" checklist:
this isolates the Docker/exec/bootstrap mechanics from model quality
**foresight commit:** `52c16d5` (includes both fixes below)

## Verdict

**The mechanics work, and didn't before this run found two real bugs.**
`init_tasks()` (real `docker build`/`run`/`exec`/`cp`, no LLM involved at all)
passed cleanly on the first try. `run_tasks()` (the actual evolve loop,
`SweCiAdapter` in the loop) initially failed 100% of the time; after the two
fixes below, all 10 architect attempts (SWE-CI's own `max_try`) completed
with **zero** `aux_failure` or resolution errors across 40 trace rows --
every one correctly resolved a container, bootstrapped aux's provider,
ran the aux harness, checked the guard, and enhanced the prompt. Containers
and images were all cleaned up afterward (`docker ps -a` / `docker images`
empty except the base task image). The architect phase still exhausted its
retries and the task ended `failed` overall -- expected and correct, since
`fake_upstream`'s canned replies cannot produce a real `requirement.xml` for
SWE-CI's own success check to find. That failure is SWE-CI judging a fake
model correctly, not an infrastructure problem.

## Bug 1 — the stdin bootstrap silently wrote empty files

`_bootstrap_provider` writes aux's `auth.json`/`opencode.json` by piping JSON
into `docker exec ... sh -c "cat > file"`. `_exec_argv` never added `-i`.
Without it, `docker exec` never attaches the child's stdin to the caller's --
measured directly, not assumed: `cat` saw immediate EOF and `auth.json` came
out as a real, exit-0, zero-byte file inside the container:

```
$ docker exec bea0ebdecc04 ls -la /tmp/aux-home/.local/share/opencode/auth.json
-rw-r--r-- 1 root root 0 ... auth.json
```

The bootstrap call itself reported success (exit 0) — nothing in
`_bootstrap_provider` treats a short/empty write as a failure, since the
`cat` command genuinely does exit 0 on empty stdin. The actual failure
surfaced one step later and one layer removed: aux's own `opencode run`
inside the container failed its first request with opencode's own generic
`UnknownError: Unexpected server error. Check server logs for details.`,
which is what a provider with no working credentials looks like from the
caller's side. Nothing pointed at the bootstrap step being the cause; it was
found by manually `docker exec`ing into the live container and reading the
file directly.

**Fix:** `_exec_argv` gained a `stdin: bool` parameter that adds `-i`;
`_bootstrap_provider` passes `stdin=True`. `tools/fake_docker.py` pipes stdin
through unconditionally (it doesn't model this Docker behavior), so this
could not have been caught by the existing 176 tests — a new test asserts
the flag directly on `_exec_argv`'s output instead.

## Bug 2 — a new session could inherit a dead container's ID

After fixing bug 1, aux's first attempt in a task succeeded, but the *next*
session (SWE-CI recreates a container per architect/programmer retry) failed
every `docker exec` with `Error response from daemon: No such container:
<id>`. `session_key()` only calls `_resolve()` (which refreshes the
IP-keyed container cache) on its IP-based path; when a session header is
present — which is *always*, since opencode reliably sends `x-session-id`
(measured in PR #5 / `tools/probe_opencode_headers.sh`) — `session_key()`
returns immediately and never touches the cache at all. `run_aux()` then
read `self._resolved[ip]` on trust, assuming (per its own comment)
"session_key already tried". Docker's default bridge reuses the lowest free
address, so the new session's container landed on the exact IP the previous,
now-removed one had — handing `run_aux` a container ID that no longer
existed, every time.

**Fix:** each cache entry is now tagged with the *session identity* (header
value or IP, computed with no subprocess call — `_session_identity`) that
produced it. `run_aux` only trusts a cache hit whose stored identity matches
the current request's; a mismatch forces a fresh resolve. This preserves the
existing, intentional behavior that a container dying *mid*-session (the
same session that resolved it) still fails loudly through a real `docker
exec`, rather than being silently swallowed into a body-only degrade — two
existing tests specifically assert that, and both still pass. A new test
recreates a container at a reused IP under two different session headers and
asserts the second session resolves to the new container.

## The platform-specific finding this all sits on top of

Docker Desktop for Mac does not route `172.17.0.1` (the bridge gateway,
correct on real Linux Docker) to the host. Measured directly: `curl` from
inside a container to `172.17.0.1` fails; to `host.docker.internal`
succeeds. `configs/swe_ci.yaml`'s `foresight_base_url` and SWE-CI's own
`base_url` both need `host.docker.internal` on this platform. Recorded in
`config.yaml` in this directory and in the SWE-CI checkout's
`config_local.toml`, not changed in `configs/swe_ci.yaml`'s own default
(`172.17.0.1` remains correct for the primary real-Linux-Docker target).

## What this run does not show

- Nothing about a real model's behavior in this loop — see
  `results/qwen3-8b-ollama-local/` and `results/qwen3-14b-ollama-local/` for
  that, against `LocalAdapter`, not yet against `SweCiAdapter`.
- Whether the fixes hold under real concurrency (`evolve.max_workers > 1`,
  multiple containers with genuinely distinct IPs at once) — this run used
  `max_workers = 1` throughout.
- `iflow` as the harness — only `opencode` was exercised.

## Reproducing

Environment notes specific to this Mac, not generic instructions: `docker`
was only on `PATH` after prefixing it with
`/Applications/Docker.app/Contents/Resources/bin` (no CLI symlink installed
in `/usr/local/bin`); SWE-CI's own `config.py` calls a Linux-only
`get_docker_storage_disk()` whenever `docker.storage_disk` is empty, which
raises on macOS — see the SWE-CI checkout's `config_local.toml` for the
non-empty placeholder plus `read_bps`/`write_bps` left empty to avoid ever
needing that value for real.

```sh
# In the foresight checkout:
python tools/fake_upstream.py --port 8011 --record /tmp/upstream.jsonl &
python -m foresight.server --config results/swe-ci-mechanics-m1-docker/config.yaml

# In the SWE-CI checkout (config_local.toml as above, metadata/default.csv
# trimmed to one task, that task's data/ already downloaded):
PYTHONPATH=src python -c "
from swe_ci.benchmark import init_tasks, run_tasks
init_tasks(); run_tasks()
"
```
