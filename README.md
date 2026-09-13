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
This milestone (M1) is the naive vertical slice: `GenericAdapter` (aux = one
chat call over the request body, no agent, no workspace, no container), a real
aux-injecting `template` builder, and a mock upstream. No GPU, no Docker, no
dataset.

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

Needs neither GPU nor Docker. Covers builder rewriting (both arms), session
keying, the aux-fires-once-per-session invariant, the `aux-model` bypass, config
validation, `AuxFailure` handling, trace record shape and its failure policy,
and an in-process end-to-end run against `fake_upstream` (including
`stream: true`, streaming token-usage recovery, and tool-call passthrough).

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

## What's next (not in this milestone)

Milestone 1 has been implemented. See `foresight-design-plan.md` for more
info on the milestones.
> note: Update this section as further milestones are implemented.

- **M2 -- `LocalAdapter`.** A real aux agent exploring a real repo, with
  `WorkspaceGuard` implemented and the future-task `template` builder doing real
  work. No Docker, no Slurm, no dataset -- the fastest path to the whole
  hypothesis working end to end.
- **M3 -- `SweCiAdapter`.** Container resolution, `docker exec` aux,
  container-ID session keying.
- **M4 -- mini-SWE-agent.** A config file pointing mini at foresight, then
  `MiniAdapter` (option B: aux explores a throwaway container from the
  per-instance image).

`foresight/guards.py` documents why `WorkspaceGuard` ships as a Protocol with no
implementation in M1, and why M2/M3 cannot skip it.
