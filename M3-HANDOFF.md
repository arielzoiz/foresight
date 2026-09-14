# M3 (`SweCiAdapter`) — handoff for validation

Status: **work in progress, not working, not committed.** Two new untracked files.
Treat this as a draft to be reviewed and largely rewritten, not as an
implementation to be smoke-tested. The defects below were found by inspection
after writing; none of this has ever been executed.

Read `foresight-design-plan.md` §"`SweCiAdapter` specifics" and `README.md`
"What's next" first — they are the spec this is measured against.

## What was produced

| Path | State |
|---|---|
| `foresight/adapters/swe_ci.py` | draft `SweCiAdapter`; parses, does not work (see D1–D4) |
| `deploy/shims/docker` | `docker`→`udocker` translation shim, executable, `UDOCKER_EXEC` is a placeholder path |

Nothing else was touched. `git status`: both files untracked, no commits made.

## Intended design (unchanged from the plan)

1. SWE-CI containers sit on Docker's default bridge, so each has its own
   `172.17.0.x`, which foresight sees as the client IP.
2. `_resolve_container_id(client_ip)` walks `docker ps -q` + `docker inspect`
   and matches `NetworkSettings.IPAddress`.
3. `session_key` returns the **container ID**, not a prompt hash — SWE-CI's
   prompts are byte-identical across all 100 tasks.
4. `run_aux` runs `docker exec -e HOME=/tmp/aux-home <cid> <agent_cmd>`, then
   reads the answer back with `docker exec <cid> cat <answer_file>`.
5. `phase` classifies architect vs programmer for the trace only.

## Defects to fix (found, not fixed)

**D1 — `run_aux` returns `None`.** The methods were appended in separate edits
and the line numbers drifted: `phase` was inserted *into the middle of*
`run_aux`. As the file now stands, `run_aux` ends at the `# TODO: Verify
workspace guard` comment (line ~174) with no `return`, and the
`return AuxResult(...)` block plus a stray `return container_id` sit **after**
`phase`'s own `return`, at method-body indentation — unreachable dead code
referencing names (`text`, `full_cmd`, `duration`) that do not exist in that
scope. `ast.parse` passes; the code cannot work. Reorder so `run_aux` ends with
its `AuxResult` and `phase` follows it, and delete the trailing
`return container_id`.

**D2 — `__init__` signature is incompatible with the ABC.** `CallerAdapter.__init__`
is keyword-only (`*, aux_backend, aux_spec, aux_prompt, options=None,
guard_name=None`) and `config.py` constructs adapters that way. The draft
declares `__init__(self, options, *args, **kwargs)` and forwards `options`
positionally, so construction raises `TypeError`. Follow `LocalAdapter`:
`def __init__(self, **kwargs)` then `super().__init__(**kwargs)`.

**D3 — the adapter is not registered.** `foresight/adapters/__init__.py` still
lists only `generic` and `local`. `adapter.name: swe_ci` therefore fails at
startup with `unknown adapter 'swe_ci'`. Add the import and the `ADAPTERS`
entry (the registry docstring anticipates exactly this: "Milestones 3-4 each
add one import and one entry").

**D4 — blocking `subprocess.run` inside `async def`, which deadlocks by
construction.** This is the most serious defect and it is not merely a
performance issue. The aux agent runs *inside the container* and is pointed
back at foresight as `aux-model`; its own model calls must be served by this
same event loop while `run_aux` is awaiting it. `subprocess.run` blocks the
loop for the entire multi-minute agent run, so those `aux-model` requests can
never be answered — aux hangs, then the timeout fires, every time. Use
`asyncio.create_subprocess_exec` + `asyncio.wait_for` as `LocalAdapter` does
(`local.py::_spawn`), and `asyncio.to_thread` for the guard's runner.

## Design gaps beyond the four defects

- **`agent_cmd` placeholder substitution is wrong.** The draft renders each argv
  element through Jinja, but `AdapterOptions.agent_cmd` and `configs/local.yaml`
  document single-brace placeholders (`{prompt}`, `{answer_file}`,
  `{workspace}`) substituted with `str.replace` — deliberately, "so braces in
  prompts and code survive". Jinja leaves `{prompt}` untouched. Copy
  `LocalAdapter::_argv`.
- **No `ManifestGuard` wiring**, only two `TODO` comments and a hardcoded
  `"workspace_guard": "TODO"` in provenance. `guards.py` states the guard is
  mandatory for M3: an aux edit under `/app/code` is copied out when the epoch
  ends and silently becomes part of the measured result. The guard needs a
  `docker exec`-based runner (the Protocol takes a runner precisely so the
  adapter supplies "how do I reach this workspace"), and a detected mismatch
  should raise `GuardViolation`.
- **The IP→container-ID cache is unbounded and never invalidated.** Docker
  recycles `172.17.0.x` across epochs, so caching on IP forever re-uses epoch
  1's container ID for epoch 2 — which defeats the entire reason the design
  keys on container ID. Re-verify at each session start, or drop the cache.
- **`session_key` shells out and raises `AuxFailure`.** It is called on *every*
  request (the builder re-application path), not only session starts, so every
  request pays `docker ps` + N× `docker inspect`, and a resolution failure
  turns a mid-session request into a 502.
- **No body-only fallback.** The plan requires degrading to `_body_only_aux`
  with that recorded in provenance, so those instances can be *excluded* at
  analysis time. Note the plan's own caveat: for SWE-CI such a call has no
  task-specific content, so it keeps the run alive but is not a valid condition.
- **No `agent_name` startup check.** The plan requires reading SWE-CI's
  `agent_name` (opencode vs iflow) and failing loudly on mismatch, since the
  task image ships exactly one harness.
- **`phase` detection is a guess.** It greps the prompt for
  "architect"/"programmer"; SWE-CI renders both from one Jinja template with
  `role`/`mode` as the only variables. Verify against
  `SWE-CI/src/swe_ci/benchmark/prompt.jinja2` before trusting it.
- **`workspace_in_container = "/app"` is hardcoded**; should be config.
- **No config, no tests.** There is no `configs/swe_ci.yaml` and no
  `tests/test_swe_ci_adapter.py`. Every other adapter has both;
  `tests/test_local_adapter.py` is the model to copy (it fakes the harness with
  `tools/fake_agent.py`).

## The `udocker` question — unresolved, and it gates the milestone

The Slurm cluster (TAU) has no Docker and no root, so the plan was a `docker`
shim on `PATH` forwarding to `udocker`, keeping SWE-CI's repo unmodified
(`SWE-CI/src/swe_ci/benchmark/utils/docker.py` hardcodes the literal
`"docker"`). Two things a validator must check before any of the above matters:

1. **`udocker` is daemonless and its `exec` semantics differ from Docker's.**
   The design plan flags this as the thing to test *first*, because SWE-CI
   starts a long-lived container and then `docker exec`s into it repeatedly.
   If `udocker` cannot hold a container running across separate `exec` calls,
   the shim approach fails and M3 is blocked regardless of adapter quality.
2. **`docker inspect` output compatibility.** The adapter's whole container
   resolution depends on `NetworkSettings.IPAddress` being present in
   `docker inspect` JSON. The current shim does **not** implement `inspect` at
   all — it maps `container inspect` to `udocker ps | grep` and returns
   grep output, not JSON. `_resolve_container_id` will therefore raise
   `JSONDecodeError`, be caught, and return `None` for every container.
   Worse, udocker containers are not on a Docker bridge network and may have
   **no per-container IP at all**, in which case client-IP→container resolution
   is not just unimplemented but *impossible*, and the session key needs a
   different source entirely.

The shim as written is a sketch: `build`, `run`, `exec`, `cp`, `rm`, `rmi` are
forwarded verbatim; `image inspect` and `container inspect` are approximated
with `grep`; unknown verbs fall through to `udocker` with a warning. Flag
translation (e.g. `-d`, `--network`) is not handled. `UDOCKER_EXEC` at line 6 is
`/path/to/your/udocker` and must be edited on the cluster.

## Suggested validation order

1. **Do not test the adapter yet.** Settle the `udocker` question above first;
   it may invalidate the container-ID session-keying design.
2. Set up the env — `pytest` currently cannot even collect here
   (`ModuleNotFoundError: No module named 'pydantic'`; `python` is not on PATH,
   `python3` is). `conda activate foresight`, then `pytest tests/ -q` to confirm
   M1/M2 still pass — they are untouched, so any failure is environmental.
3. Fix D1–D4, register the adapter, then write `tests/test_swe_ci_adapter.py`
   with a faked `docker` on `PATH` (no runtime needed) covering: IP→ID
   resolution, container-ID session keying, `docker exec` argv shape including
   `HOME=/tmp/aux-home`, timeout, empty-answer → `AuxFailure`, and guard
   violation → `GuardViolation`.
4. Only then attempt a real one-task run (`max_epoch = 1`, smallest splitting),
   per Verification step 5 in the design plan.

