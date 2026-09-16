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
(`tests/test_agent_turn.py`), and the global aux concurrency limit.

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

Milestones 1 and 2 have been implemented. See `foresight-design-plan.md` for
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

- **M3 -- `SweCiAdapter`.** Container resolution, `docker exec` aux,
  container-ID session keying. Gated on a container runtime being reachable --
  test `udocker` first, since that decides whether M3 and M4 are possible at all.
- **M4 -- mini-SWE-agent.** A config file pointing mini at foresight, then
  `MiniAdapter` (option B: aux explores a throwaway container from the
  per-instance image). Independent of M3, not sequenced after it.

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

1. **Session keying must be the container ID, not the prompt hash.** Still M3
   work. SWE-CI's prompts come from a static template with only `role` and
   `mode` as variables, so every architect call across all tasks and epochs is
   byte-identical. Under the current default they collapse into one session and
   aux runs once, ever. Container ID is one-to-one with a session -- SWE-CI
   builds a fresh container per *phase* (`run.py`: architect and programmer
   each get their own `run_container`/`remove_container`, twice per epoch), not
   per epoch as earlier phrasing here said, so keying on it also separates
   architect from programmer for free. The container *name* is reused across a
   whole task (`uuid.uuid4().hex[:16]` generated once in `_run_locked`), so
   `SweCiAdapter` must key on the container **ID** from `docker inspect`, never
   the name.
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
4. **`HOME` isolation: the diagnosis needed a correction, the requirement still
   needs verifying.** `OPENCODE_HOME` is not an opencode variable -- it is
   this project's own wrapper convention
   (`deploy/tau-slurm/setup/install_opencode.sh: export HOME="${OPENCODE_HOME:-$BASE/home}"`).
   So the M2 measurement was not "the wrong variable was set": `HOME` *was* set
   per run, and `deploy/tau-slurm/README.md` records a split -- opencode's
   config files followed it, but the session SQLite lived on shared storage
   regardless. Two concrete things for `SweCiAdapter` to get right, which the
   design plan does not currently say: it must replicate `setup_opencode`
   (`SWE-CI/src/swe_ci/benchmark/agents/opencode.py:16-67`) under aux's own
   home -- `auth.json` + `opencode.json` pointing at foresight as
   `aux-model` -- or aux has no provider config and cannot call anything; and
   `read_usage` (`opencode.py:71-106`) sums `SELECT data FROM message` across
   *every* message and reads `SELECT time_created, time_updated FROM session
   LIMIT 1` with **no `ORDER BY`**, so a DB shared with aux would corrupt
   SWE-CI's own `execution_time` as well as its token counts,
   non-deterministically, and silently (`read_usage` returns all-`None` when
   the DB is simply missing). Needs a container runtime to verify -- gated on
   risk 1, `udocker` first.

### Open question before implementing `SweCiAdapter`: one process per call, not one shared process

Item 1 above (container-ID keying) exists to answer "which of 16 concurrent
sessions did this request belong to?" -- a question that only needs answering
because SWE-CI's design, as sketched in `foresight-design-plan.md`, has all 16
`ProcessPoolExecutor` workers (`run.py`, `CONFIG.evolve.max_workers`) point at
one shared `base_url` and therefore one shared foresight process.

**The alternative worth deciding first: one foresight process per call**,
started and stopped alongside each `run_container`/`remove_container` pair
SWE-CI already does (`run.py`, once per architect call and once per programmer
call -- up to ~40 times per task). Under this model:

- Session identity is trivial -- one process, one session, no client-IP or
  `docker inspect` resolution to build. `SweCiAdapter` may not need
  container-ID keying, possibly not even a `session_key` override, at all.
- Architect and programmer are isolated automatically, as a side effect of
  matching container lifecycle, not as a separate design decision -- they
  already get separate containers per call, so they'd get separate foresight
  processes too.
- The whole concurrency-correctness surface this PR hardened (per-key locks,
  `get_if_created_after` dedupe, `max_concurrent_aux`) becomes moot for
  SWE-CI specifically: no two calls ever share a `Pipeline` or `SessionStore`
  to race over.

Costs to weigh against that: SWE-CI's `config.toml` has one global `base_url`
for the whole run (`CONFIG.base_url`, read once), so this needs a per-call
`base_url` on the SWE-CI side -- real code there, not just a foresight config
change. Traces fragment into many files (merge as a post-processing step).
More process churn than one shared process, but bounded by the container
churn SWE-CI already pays for, and foresight itself is a lightweight
CPU-only async process with no heavy init -- none of this needs its own GPU
Slurm allocation; all instances can run under one CPU job, same as the single
shared process does today, pointed at the same GPU-backed vLLM endpoints.

Decide this **before** writing `SweCiAdapter`: it changes what the adapter
needs to be, not just how it's configured.
