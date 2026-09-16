# foresight

A caller-agnostic prompt-enhancement proxy for *"Enhancing LLM Code
Maintainability via Future-Task Prompting."*

To its caller, foresight **is** a language model: it speaks the OpenAI
chat-completions protocol, so a harness configured with
`base_url = http://host:8000/v1` cannot distinguish it from vLLM or a hosted
API. Inside one request it consults an auxiliary model about plausible future
work, rewrites the task prompt to carry that context, and forwards it to the
target model. The control arm is the same pipeline with a `passthrough`
builder, so both arms traverse identical code and differ only in the prompt.

The full design lives at `foresight-design-plan.md`.

**M1** was the naive vertical slice: `GenericAdapter` (aux = one chat call over
the request body, no agent, no workspace, no container), a real aux-injecting
`template` builder, and a mock upstream.

**M2** makes aux a real agent. `LocalAdapter` spawns a harness as a subprocess
with `cwd` set to a configured repo, lets it explore the actual code, and reads
back what it wrote -- with `ManifestGuard` failing the request if aux modified
anything. Still no GPU, no Docker, no dataset.

### General shape

```
 CALLER SIDE                    FORESIGHT                      MODEL ENDPOINTS
┌──────────────────┐        ┌───────────────────────┐
│  target agent    │──(1)──▶│ server                │        ┌──────────────┐
│  (harness)       │        │   │                   │──(7)──▶│ target model │
│  owns tools,     │◀─(8)───│   ▼                   │◀───────│              │
│  edits workspace │        │ CallerAdapter         │        └──────────────┘
└────────┬─────────┘        │   (2) session start?  │
         │                  │   (3) run_aux ────┐   │        ┌──────────────┐
         │ reads/writes     │   (6) builder     │   │──(5)──▶│  aux model   │
         ▼                  └───────────────────┼───┘◀───────│              │
   ┌──────────────┐                             │            └──────────────┘
   │  workspace   │◀────(4) read-only──── ┌─────▼──────┐
   └──────────────┘                       │ aux agent  │
                                          └────────────┘
```

1. harness sends a request
2. adapter: is this a session start?
3. if yes, `run_aux` spawns an aux agent over the workspace
4. aux explores, read-only
5. aux's model calls return to foresight as `aux-model`, bypass the pipeline, exit via `Backend`
6. builder rewrites the task prompt, on **every** request of the session, using the cached result
7. forward to target model
8. relay reply.

See `foresight-design-plan.md` for the per-caller flow diagrams (SWE-CI, mini-SWE-agent, local use).

## Setup

```bash
conda create -y -n foresight python=3.13
conda activate foresight
pip install -r requirements.txt
```

## Run it (local test setup)

The commands below stand up foresight against `fake_upstream.py`, a mock
model server, purely so you can exercise the pipeline locally. This is for
testing only -- for a real run, point foresight at an actual benchmark or
agent harness on one side and a real model endpoint on the other, instead of
`fake_upstream.py`.

Two processes: foresight itself, and the mock upstream standing in for both
models.

```bash
# terminal 1
python -m foresight.server --config configs/naive.yaml

# terminal 2
python tools/fake_upstream.py --port 8001 --record /tmp/upstream.jsonl
```

Then, from a third terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "target-model",
        "messages": [{"role": "user", "content": "Fix the bug in parse_date()"}]
      }'
```

Diff `/tmp/upstream.jsonl` against a run with `configs/control.yaml` -- the
only difference in what the target model receives should be the prompt.

### Traces

Each config writes one JSON record per request to `traces/<arm>.jsonl` (set
`trace.path: null` to disable). This is the experiment's own record; the two
files answer different questions:

| | Records | Use it to |
|---|---|---|
| `traces/naive.jsonl` | foresight's view: prompt in/out, aux text and provenance, session key, timings, token counts | analyse a run |
| `/tmp/upstream.jsonl` | the upstream's view: whole request bodies | confirm a body byte for byte |

A record carries `prompt_in` and `prompt_out` -- the task prompt before and
after the builder -- which is what makes the re-application invariant checkable:
`aux` fires once per session (only the session-start row has `timings.aux_s`),
yet every row of that session is `enhanced` with identical `prompt_out`.

`is_agent_turn: false` marks a request a harness made for its own bookkeeping
rather than as a turn of the agent session -- e.g. opencode's title-generation
request, which carries no `tools`. Those rows were forwarded byte-identical to
what the control arm would have sent: `session_key` is empty and `aux` is
`null`, because no key was ever resolved. Exclude them from per-session row
counts; they belong to no session. See "Design decisions" below.

```bash
python -c "
import json
for l in open('traces/naive.jsonl'):
    r = json.loads(l)
    print(r['session_key'], r['is_session_start'], r['enhanced'],
          r['prompt_in_chars'], '->', r['prompt_out_chars'], r['timings'])
"
```

Two fields are `null` on streamed requests, and neither is a shortcut:
`upstream_status`, because `Backend.stream` yields bytes and never exposes the
status; and `usage.upstream`, unless the caller set
`stream_options: {"include_usage": true}`. foresight will not inject that option
-- raw passthrough is the guarantee the whole design rests on -- so when the
caller does ask, the counts are recovered by watching the bytes go past.

### Configs

| Config | Arm | Builder |
|---|---|---|
| `configs/naive.yaml` | treatment | `template` -- injects aux's future-task context |
| `configs/control.yaml` | control | `passthrough` -- forwards unchanged |
| `configs/canary.yaml` | diagnostic, not a condition | a trivially visible silly prefix |
| `configs/ollama.yaml` | treatment, real model | `naive.yaml` pointed at a local Ollama instead of the mock |
| `configs/local.yaml` | treatment, **M2** | aux is a real agent over a real repo, `guard: manifest` |

The local configs (`local.yaml`, `local-fake.yaml`, `vllm-local.yaml.tmpl`,
`fake.yaml.tmpl`) also set `adapter.max_concurrent_aux: 1`: two `opencode run`
processes sharing one on-disk session database deadlock each other. See
"Design decisions" below.

`canary.yaml` answers a question no upstream inspection can, once a real
harness sits between us and the visible output: *did the enhancement reach the
model, or did it merely get sent?* Point a real harness at it and grep its
transcript for "what day is it" -- reach for this when the enhanced arm
(`naive.yaml`) behaves indistinguishably from the control, to rule out "never
arrived" before concluding "no effect", and to catch a silent revert of the
re-application logic mid-session.

## Tests

```bash
pytest tests/ -q
```

Needs neither GPU nor Docker, and no agent harness -- `tools/fake_agent.py`
stands in for one, the same way `fake_upstream.py` stands in for a model.

Covers builder rewriting (both arms), session keying, the
aux-fires-once-per-session invariant, the `aux-model` bypass, config validation,
`AuxFailure` handling, trace record shape and its failure policy, the aux quality
gate, `LocalAdapter`'s subprocess contract (argv substitution, cwd, timeouts,
stale answers), `ManifestGuard` against real directories, an in-process
end-to-end run against `fake_upstream` (including `stream: true`, streaming
token-usage recovery, and tool-call passthrough), the non-agent-turn gate
(`tests/test_agent_turn.py`), the global aux concurrency limit, and
`SweCiAdapter` against `tools/fake_docker.py` (`tests/test_swe_ci_adapter.py`)
-- IP/header session keying, the `docker exec` argv, the HOME-wrapper bypass,
provider bootstrap, the guard, and the body-only degradation.

### Verifying `SweCiAdapter` against real Docker

`pytest` above never touches a real container. What it cannot prove:

1. **The `docker exec` argv actually works** -- `_exec_argv`'s flags
   (`-u`, `-w`, `-e`) against a real `docker` binary, not `fake_docker.py`'s
   host-subprocess stand-in.
2. **The HOME-wrapper bypass holds against the real image.** Build (or pull)
   an actual SWE-CI task image, start a container from it, and confirm
   `docker exec -u root <cid> /opt/agent/npm-global/bin/opencode --version`
   runs without touching `/opt/agent/home`.
3. **Provider bootstrap produces a config opencode actually accepts.** After
   one real `run_aux` call, `docker exec <cid> cat /tmp/aux-home/.config/opencode/opencode.json`
   should be valid, and a manual `opencode run` inside the container using it
   should reach `foresight_base_url` and get a reply.
4. **One real SWE-CI task, end to end** -- smallest splitting, `max_epoch = 1`,
   pointed at `configs/swe_ci.yaml`. Confirm: no instance falls back to
   `valid_condition: false`, the guard reports `intact`, `phase` classifies
   architect vs. programmer correctly, and SWE-CI's own `iteration.jsonl`
   shows non-`None`, plausible token counts and `execution_time` for the
   target -- the point of the HOME-wrapper fix; a shared-HOME regression
   would corrupt exactly these fields, silently.

Fix whatever step 1-3 finds before attempting step 4 -- a real task run costs
significantly more time to fail than any of the three isolated checks.

## Design decisions specific to this milestone

See `foresight-design-plan.md` for the full reasoning. In short:

- **The proxied request body is never parsed into a Pydantic model.** Doing so
  would silently drop any field we did not declare, which breaks the
  "indistinguishable from vLLM" requirement. Config, `ModelSpec`, and `AuxResult`
  are Pydantic; the wire body stays an opaque `dict`.
- **Aux failure fails the request, loudly** (`AuxFailure` -> `502`). No fallback,
  no degraded path. Silently degrading to an unenhanced prompt would move an
  instance into the control arm while the config still says treatment.
- **Every request that reached a model is traced, and a failed trace write fails
  the request** (`TraceFailure` -> `500`). A run whose traces silently stopped
  produces results nobody can interpret afterwards, which is the same failure
  mode `AuxFailure` guards against one layer out. Most causes are caught at
  startup instead: an unwritable `trace.path` is a `ConfigError` before the port
  is bound. Two paths do not honour it -- a request that is already failing
  (the 502 must not become a 500 and hide the cause), and a stream, whose status
  is on the wire before the record is written.
- **The session store is not a cache.** It holds the one thing that must outlive
  a request: aux's finished output, so the builder can re-inject it into every
  later request of the same target session. See the docstring on
  `foresight.pipeline.SessionStore` for the full argument.
- **Requests that are not agent turns are forwarded untouched.** opencode opens
  every session with a title-generation request ("Generate a title for this
  conversation:") that carries no `tools`, shortly before the real agent turn,
  which does. Left unhandled this costs a whole aux agent run naming a chat
  and, because the two hash to different session keys and take different
  per-key locks, races it into the real aux run -- measured as `database is
  locked` against opencode's shared session database (M2, see
  `results/qwen2.5-coder-7b/findings.md`). `CallerAdapter.is_agent_turn`
  (`adapter.require_tools`, tri-state, default on) checks this before
  `session_key` is even resolved, so a skipped request never reaches
  aux *or* the builder -- skipping only aux and letting the builder run
  would still enhance the title request with an empty future-task block. A
  caller whose config never advertises tools (e.g. mini-swe-agent's
  text-based model classes) sets `require_tools: false`; foresight warns on
  stderr, and `/health` reports `enhanced`/`skipped_not_agent` counters, if a
  run goes quiet for too long without enhancing anything, since that would
  otherwise silently be the control arm.

## What's next

Milestones 1, 2 and 3 have been implemented. See `foresight-design-plan.md` for
more info on the milestones.
> note: Update this section as further milestones are implemented.

**The M2 flow runs on Slurm.** A real opencode aux agent over a real small repo, with
both `target-model` and `aux-model` served by vLLM on a GPU node -- aux runs once
per session, the enhanced prompt is re-applied to every request of that session,
and the workspace guard comes back clean. See `deploy/tau-slurm/`.

**Aux is not grounded yet.** With `Qwen/Qwen2.5-Coder-7B-Instruct` the aux agent never
ran a single tool. Under the current prompt it does not even attempt one — it answers
in prose, inventing JavaScript filenames for a Python repo. (Under an earlier
tool-forcing prompt it did attempt a call, but emitted a fenced JSON block that no
parser recognises.) Either way aux answers from the task text without reading the
code, which makes its output weaker evidence than the design intends. See
`results/qwen2.5-coder-7b/`.

A larger model is the obvious next step, and `Qwen/Qwen3-Coder-30B-A3B-Instruct` is
downloaded, but the bf16 attempt never loaded — see
`results/qwen3-coder-30b-bf16-FAILED/`. The FP8 variant on a single H100/H200 is the
retry to prefer.

**Two things to get right before blaming a checkpoint for that**, both learned the
expensive way and both documented in `deploy/tau-slurm/README.md`:

- **The `--tool-call-parser` must match the checkpoint.** Qwen2.5 wraps *JSON* in
  `<tool_call>`; Qwen3-Coder wraps *XML* in the same tag. The wrong parser returns
  HTTP 200 with `tool_calls: null` and no error anywhere, which looks exactly like a
  model that cannot call tools.
- **Read the aux agent's own transcript, not just the trace.** `$RUN/aux-agent/ses_*.json`
  is opencode's full session export — every tool call with its input and output.
  `python tools/aux_transcript.py "$RUN/aux-agent/ses_*.json"` summarises it and says
  whether the run was grounded. The trace's `role: "aux"` row count tells you *that*
  no tool ran; only the transcript tells you *why*.
- **Never guess a parser on a GPU.** `tools/probe_tool_parsers.py` replays a captured
  model output through every registered vLLM parser offline — no GPU, no weights,
  seconds instead of a queue wait per guess. Measured with it: what the 7B emits
  (a fenced JSON block holding a *single* object) parses under no parser at all, and
  the fence is not the obstacle — `xlam` accepts fences but demands an array.

- **M3 -- `SweCiAdapter`. Implemented; verified only against `tools/fake_docker.py`,
  not yet against a real Docker install.** Resolves a container from the client
  IP (or a session header, see below), bootstraps aux's own opencode provider
  config, `docker exec`s a second harness into the live task container, and
  reads the answer back out through the workspace guard. `configs/swe_ci.yaml`
  targets real Docker; this Slurm cluster does not currently have one --
  `udocker` (the cluster's only container route) was tested directly, not
  assumed, and fails what M3 needs regardless of the adapter's own
  correctness: no `exec` verb, and writes made by `udocker run` do not
  survive a separate, later invocation of the same container, in any of its
  execution modes. Real Docker cannot be installed without root either. None
  of this affects running `SweCiAdapter` anywhere real Docker already works
  (a local machine, most CI runners) -- see `foresight-design-plan.md` risk 1
  for the measurement and the two options for this cluster specifically (an
  admin ticket, or pivoting to M4).
- **M4 -- mini-SWE-agent.** A config file pointing mini at foresight, then
  `MiniAdapter` (option B: aux explores a throwaway container from the
  per-instance image). Independent of M3, not sequenced after it, and worth
  prioritising on this cluster specifically: mini's environment backend is
  pluggable and includes `bubblewrap`, not yet tested here.

`ManifestGuard` is implemented and required from M2 on. It is not a security
boundary -- nothing prevents a write, and a determined agent could restore an
mtime. It exists so contamination is loud instead of silent.

## For the M3 / SWE-CI PR

Four things the M2 runs established that land on `SweCiAdapter`. Details in
`results/qwen2.5-coder-7b/findings.md` and `deploy/tau-slurm/README.md`. Two of
them (2 and 3) are fixed in this PR, for every caller, before `SweCiAdapter`
exists -- not as four fixes after it. SWE-CI's target agent is invoked exactly
the way M2's aux was (`opencode run --model ... "<prompt>"`,
`SWE-CI/src/swe_ci/benchmark/agents/opencode.py`), so it issues the same
title-generation request and would hit the same bug.

1. ~~Session keying must be the container ID, not the prompt hash.~~ **Done.**
   SWE-CI's prompts come from a static template with only `role` and `mode` as
   variables, so every architect call across all tasks and epochs is
   byte-identical -- the ABC's default prompt-hash key would collapse them
   all into one session and run aux once, ever. Container ID is one-to-one
   with a session -- SWE-CI builds a fresh container per *phase* (`run.py`:
   architect and programmer each get their own
   `run_container`/`remove_container`, twice per epoch, plus a third for
   pytest that never contacts foresight), so keying on it also separates
   architect from programmer for free. The container *name* is reused across
   a whole task (`uuid.uuid4().hex[:16]` generated once in `_run_locked`), so
   `SweCiAdapter.session_key` keys on the container **ID** from
   `docker inspect`, never the name -- with opencode's own `x-session-id`
   header preferred first where present (see "Keeping SWE-CI's task
   concurrency" below), which needs neither.
2. ~~Fix the title-generation request first.~~ **Done.** opencode opens every
   session with a second request whose prompt is `Generate a title for this
   conversation:` and no `tools`. `CallerAdapter.is_agent_turn` now checks this
   on every request, before `session_key` is even resolved, and the pipeline
   skips aux *and* the builder for it -- see "Design decisions" above. This
   matters more than it would have under prompt-hash keying: under container-ID
   keying the title request and the real turn share a key, so without the gate
   whichever wins the race would seed (or, worse, consume) the cached aux
   result for the whole session.
3. ~~Concurrent aux runs are handled only where keys collide.~~ **Removed at
   the source, plus a configurable backstop.** The M2 `database is locked`
   failure was two aux runs against opencode's title request and the real turn
   racing each other; (2) removes it, because the title request no longer
   starts an aux run at all. That is not the same as "concurrency is now
   safe" -- two *genuinely distinct* sessions (two epochs, architect and
   programmer, SWE-CI's `max_workers = 16`) still run concurrently by design.
   `adapter.max_concurrent_aux` caps aux runs in flight for callers whose aux
   subprocess shares on-disk state (`LocalAdapter`/opencode); it is 0
   (unlimited) by default and must **not** be set for `SweCiAdapter`, since
   each of its aux runs is isolated in its own container and serialising 16
   workers behind one 3600s aux call would be self-inflicted.
4. ~~`HOME` isolation.~~ **Done, and the diagnosis needed a correction along
   the way -- the requirement turned out to be two separate fixes, not one.**
   `OPENCODE_HOME` is not an
   opencode variable -- it is this project's own wrapper convention
   (`deploy/tau-slurm/setup/install_opencode.sh: export HOME="${OPENCODE_HOME:-$BASE/home}"`).
   So the M2 measurement was not "the wrong variable was set": `HOME` *was* set
   per run, and `deploy/tau-slurm/README.md` records a split -- opencode's
   config files followed it, but the session SQLite lived on shared storage
   regardless. Investigating `SweCiAdapter`'s draft (`origin/m3-swe-ci-adapter`)
   surfaced why that split happens under SWE-CI specifically, and a second gap
   the design plan did not previously name:

   - **D-A -- `docker exec -e HOME=...` is a no-op against SWE-CI's own image.**
     `Dockerfile.opencode` generates `/usr/local/bin/opencode` as a wrapper that
     *unconditionally* does `export HOME="$BASE/home"` (plus `XDG_CACHE_HOME`,
     `NPM_CONFIG_PREFIX`, `PATH`), with no `${OPENCODE_HOME:-...}` escape hatch
     -- unlike this project's own wrapper. So `docker exec -e HOME=/tmp/aux-home
     ... opencode` still lands aux in `/opt/agent/home`, sharing the target's
     `opencode.db`. `read_usage` (`opencode.py:71-106`) sums `SELECT data FROM
     message` across *every* message with no session filter, and reads
     `SELECT time_created, time_updated FROM session LIMIT 1` with **no
     `ORDER BY`** -- so a DB genuinely shared with aux corrupts both SWE-CI's
     `execution_time` and its token counts, non-deterministically and
     silently. `configs/swe_ci.yaml` points `agent_cmd` at the real binary
     directly, `/opt/agent/npm-global/bin/opencode`, and sets the whole
     environment itself (`HOME`, `XDG_CACHE_HOME`, `NPM_CONFIG_PREFIX`,
     `npm_config_cache`, `PATH=/opt/agent/node/bin:/opt/agent/npm-global/bin:$PATH`)
     -- no code change needed, since `SweCiAdapter._exec_argv` already forwards
     every `adapter.env` entry as its own `-e`. Image permissions
     (`chmod -R 777 /opt/agent`) make this work under any uid.
   - **D-B -- aux needs its own provider config, or it cannot call anything.**
     `SweCiAdapter._bootstrap_provider` replicates `setup_opencode`
     (`SWE-CI/src/swe_ci/benchmark/agents/opencode.py:16-67`) under aux's own
     home before the first aux spawn -- `auth.json` + `opencode.json` pointing
     at foresight as `aux-model` (`adapter.foresight_base_url`), written the
     same way SWE-CI writes them (`docker exec -i -u root <name> sh -c
     "mkdir -p D && cat > D/F"` with the JSON on stdin). Without this, a
     fresh `HOME` has no provider config at all and aux fails on its very
     first request, not silently but not obviously either -- it looks like
     any other `AuxFailure`.

   Both are implemented and covered by `tests/test_swe_ci_adapter.py` against
   `fake_docker.py`; both still need a real Docker to verify for real --
   available on a local machine, not on this Slurm cluster as configured (see
   the M3 bullet above, "Verifying `SweCiAdapter` against real Docker", and
   `foresight-design-plan.md` risk 1).

## Keeping SWE-CI's task concurrency

SWE-CI's own `evolve.max_workers` (`config.toml`, default 16) is what runs
100 tasks in a practical amount of time. Getting `SweCiAdapter` to work with
it -- rather than forcing `max_workers = 1` -- needed settling which
container a request belongs to, and which SWE-CI *session* it belongs to.

**The design plan's proposal -- one foresight process per `run_container`/
`remove_container` call -- is rejected.** It needs a caller-repo change:
`base_url` is a process-global read at `SWE-CI/src/swe_ci/benchmark/agents/
opencode.py:47`, and giving each call its own value means threading a
parameter through `tools.py::call_cli_agent`, `opencode.py::call_opencode`/
`setup_opencode`, and both `run.py` call sites. Small (~4 lines), but still
SWE-CI's repo, which binding constraint 1 forbids.

**Container identity: client-IP -> `docker inspect` already works under real
Docker**, exactly as the original design plan describes (SWE-CI's containers
sit on Docker's default bridge, each with its own address). No change needed
there.

**Session identity: opencode sends one for free, which simplifies this
further.** Measured with `tools/probe_opencode_headers.sh` (no Docker, no
GPU -- run it against the task image's actual opencode/iFlow binary before
relying on this): opencode sends `x-session-id` on every request, stable
across one `opencode run` invocation and distinct across invocations, with
zero configuration. `SweCiAdapter.session_key` reads it directly when
present (`adapter.session_header_names`, checked in order), falling back to
`client_ip` + `docker inspect` otherwise -- removing a blocking subprocess
pair from the hot path in the common case. (The title-generation request
shares the id with the real turn, so `is_agent_turn` gating -- item 2 above
-- remains load-bearing here exactly as it is under container-ID keying.)

Together: `evolve.max_workers > 1` runs against one foresight process under
real Docker with no sharding needed. `max_workers` itself should still be
set from measured throughput rather than the default of 16 -- every session
now costs two agent runs (aux plus target) against whatever is serving
`target-model`/`aux-model`, which matters more, not less, when that's a
CPU-served local model rather than a GPU endpoint.
