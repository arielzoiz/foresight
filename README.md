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
stale answers), `ManifestGuard` against real directories, and an in-process
end-to-end run against `fake_upstream` (including `stream: true`, streaming
token-usage recovery, and tool-call passthrough).

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
`results/qwen2.5-coder-7b/findings.md` and `deploy/tau-slurm/README.md`.

1. **Session keying must be the container ID, not the prompt hash.** SWE-CI's
   prompts come from a static template with only `role` and `mode` as variables,
   so every architect call across all tasks and epochs is byte-identical. Under
   the current default they collapse into one session and aux runs once, ever.
   Container ID is one-to-one with a session, since SWE-CI builds a fresh
   container per epoch -- which is also what gives each epoch its own aux run.
2. **Fix the title-generation request first.** opencode opens a session with a
   second request whose prompt is `Generate a title for this conversation:`, and
   foresight treats it as a task prompt. Once keying is by container ID both
   requests share a key, so whichever wins the race seeds the cached aux result
   for the whole epoch -- possibly with future tasks invented for naming a chat.
   Non-deterministic, and it silently corrupts the condition.
3. **Concurrent aux runs are handled only where keys collide.**
   `Pipeline._resolve_aux` takes a per-key lock on session start and
   `get_if_created_after` collapses two concurrent starts into one aux run. The
   M2 `database is locked` failure happened because the two requests hashed to
   *different* keys and so took different locks; container-ID keying makes them
   match, so this should stop recurring under `SweCiAdapter` — on a path no test
   exercises yet. **It stays live everywhere else:** `LocalAdapter` and any other
   prompt-hash keying still give the two requests different keys, so the
   deadlock can recur on further local runs.
4. **`HOME` isolation does not work -- verify before relying on it.** Measured
   on M2: `OPENCODE_HOME` does not isolate opencode's session database, which is
   shared across jobs, nodes and workspaces. The design's `SweCiAdapter` notes
   assume a separate `HOME` keeps aux out of SWE-CI's own token accounting
   (which reads `opencode.db`). That assumption needs checking, not inheriting.

**2 and 3 are one bug, and one rule fixes both for every caller.** They are both
consequences of treating opencode's title-generation request as an agent turn. It
is distinguishable without any container: a real agent turn always advertises the
harness's tools, and that request carries none — measured, `tool_count` 10 versus
0. So: **do not start an aux run for a session-start request with no `tools`
array.** The spurious run disappears at its source, which leaves exactly one
session start per harness session, which in turn removes the concurrency — under
`LocalAdapter` as much as `SweCiAdapter`, rather than relying on keys colliding.

That leaves 1 as the only genuinely caller-specific piece (session identity, which
needs the container ID), and 4 as a thing to verify. Worth doing as one change
before SWE-CI rather than four fixes after it. Keep the rule configurable: a
non-agent caller sending no tools would otherwise silently never be enhanced.
