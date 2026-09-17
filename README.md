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

See `deploy/local-ollama/README.md` for the reproducible setup.

**Current state.** The `docker exec` argv, the HOME-wrapper bypass, and provider bootstrap --
are confirmed against real Docker Desktop (M1 Mac).

**Step 4 -- one real task, end to end, with a real model -- is still open**,
for two separate reasons:

- **Memory.** Docker's VM + a real task container + a loaded local model
  OOM'd this 16GB machine twice (`qwen3:14b`, then `qwen3:8b`) before either
  produced a single trace row.
- **No model tested is actually good enough as aux**, which is the real
  remaining gap, not an infrastructure one:

  | Model | Result |
  |---|---|
  | `qwen2.5-coder:7b` / `:14b` | can't emit well-formed tool calls at all via Ollama |
  | `qwen3:8b` | tool-call format is fine, but 90% of aux invocations produced empty output outright |
  | `qwen3:14b` | stopped failing outright, but still couldn't find the one file in a two-file toy repo after 7.5 minutes of searching -- grounded activity, ungrounded result |
  | `qwen3-coder:30b` | doesn't fit my 16GB RAM M1 Macbook -- 18.56GB minimum, exceeds this machine's 16GB entirely |

  (One aux hang hit while debugging `qwen3:8b` was never root-caused; see `results/qwen3-8b-ollama-local/findings.md`.)

**Next step: try `qwen3-coder:30b`** -- the model class this project's own
results already flagged as the one worth trying, and no smaller model
cleared the grounding bar. It won't fit on this machine, so point
`models.target`/`models.aux` at a model server running elsewhere
(`base_url` is just a URL) rather than loading it locally alongside Docker.

## Design decisions specific to this milestone

See `foresight-design-plan.md` for the full reasoning. In short:

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

- **M3 -- `SweCiAdapter`. Implemented; current state, model findings and next
  steps in "Verifying `SweCiAdapter` against real Docker" above.** Resolves a
  container from the client IP (or a session header, see below), bootstraps
  aux's own opencode provider config, `docker exec`s a second harness into
  the live task container, and reads the answer back out through the
  workspace guard. `configs/swe_ci.yaml` targets real Docker; this Slurm
  cluster does not currently have one (or does it? I wan't able to find docker installed - AZ),
  and cannot get one without an admin ticket -- see `foresight-design-plan.md` risk 1 for the measurement.
- **Run `SweCiAdapter` against real Docker, locally, with a bigger model.**
  Try `qwen3-coder:30b` -- the smaller models tested so far (`qwen2.5-coder`,
  `qwen3:8b`, `qwen3:14b`) were not sufficient as aux for a real task, see
  the model table above.
- **Check again for Docker on the Slurm cluster.** Not found on node
  `c-006-tau-slurm` at time of writing (AZ). If it turns out to exist there or
  elsewhere on the cluster, run the SWE-CI benchmark against
  `SweCiAdapter` on Slurm with a 30B+ model and report results -- the
  cluster's GPUs make a bigger model practical in a way this Mac cannot.
- **If Docker never turns up on Slurm, run foresight locally with Docker
  while serving the model(s) from Slurm as jobs instead.** Point
  `models.target`/`models.aux` at a vLLM job's node (needs VPN access to
  the cluster from off-campus). Slurm compute nodes have outbound
  internet, confirmed directly in `deploy/tau-slurm/README.md` -- but
  this direction is the reverse (this Mac calling *in* to a job's node),
  which has not been checked and should not be assumed to work the same
  way; confirm inbound requests actually reach the job before relying on
  this.
- **M4 -- mini-SWE-agent.** A config file pointing mini at foresight, then
  `MiniAdapter` (option B: aux explores a throwaway container from the
  per-instance image). Independent of M3, not sequenced after it, and worth
  prioritising on this cluster specifically: mini's environment backend is
  pluggable and includes `bubblewrap`, not yet tested here.
- **M5 (if time permits) -- an adapter for enhancing local coding agents'
  prompts.** Alongside the aux agent, also run a target agent locally, and
  use foresight to enhance the first prompt the target agent gets.

`ManifestGuard` is implemented and required from M2 on. It is not a security
boundary -- nothing prevents a write, and a determined agent could restore an
mtime. It exists so contamination is loud instead of silent.

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
shares the id with the real turn, so `is_agent_turn` gating -- see "Design
decisions specific to this milestone" above -- remains load-bearing here
exactly as it is under container-ID keying.)

Together: `evolve.max_workers > 1` runs against one foresight process under
real Docker with no sharding needed. `max_workers` itself should still be
set from measured throughput rather than the default of 16 -- every session
now costs two agent runs (aux plus target) against whatever is serving
`target-model`/`aux-model`, which matters more, not less, when that's a
CPU-served local model rather than a GPU endpoint.
