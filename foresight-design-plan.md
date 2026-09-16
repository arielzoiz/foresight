# Design: `foresight` — a caller-agnostic prompt-enhancement proxy

## Context

The project ("Enhancing LLM Code Maintainability via Future-Task Prompting") tests whether forcing a coding agent to consider plausible *future* tasks produces more maintainable code.
The primary measurement instrument is SWE-CI (`~/repos/SWE-CI`); mini-SWE-agent (`~/repos/mini-swe-agent`) is the secondary one. Both must run unmodified.

**On SWE-bench vs mini-SWE-agent.** They are the same benchmark tasks, by the same Princeton & Stanford team;
mini is the *agentic* runner for them — a real agent loop with a `bash` tool exploring and editing a repo in a container,
the same shape as SWE-CI. SWE-bench's own `run_api.py` is the older single-completion path with no agent at all,
which cannot test a hypothesis about agents. **From here on this plan refers only to mini.**

mini does inference only: it writes `preds.json` = `{instance_id, model_name_or_path, model_patch}` and stops.
Deciding whether a patch *resolved* an issue means running that instance's tests, which is a separate step — either **`sb-cli`** (cloud-hosted, free for research,
needs no local container runtime) or `python -m swebench.harness.run_evaluation` from the `~/repos/SWE-bench` checkout (local, needs one).
mini's own docs present exactly these two options (`docs/usage/swebench.md:110-130`).

`foresight` is a component that, to its caller, **is a language model**: it speaks the OpenAI chat-completions protocol,
so a harness configured with `base_url = http://host:8000/v1` cannot distinguish it from vLLM or a hosted API.
Inside that single request/response it may run a whole auxiliary agent, rewrite the prompt, and forward to a different model — all invisibly.

It consults an auxiliary agent about plausible future work, rewrites the task prompt to carry that context, and forwards it to the target model.
The control condition is the same pipeline with a `passthrough` builder, so both arms of the experiment traverse identical code and differ only in the prompt.

---

## Vocabulary

| Term | Meaning |
|---|---|
| **request** | One `POST /v1/chat/completions` arriving at `foresight`. The only unit we observe. |
| **agent session** | One harness invocation (`opencode run …`): a sequence of requests over a growing message list. Its *first* request carries the task prompt; later ones carry tool results. |
| **task / instance** | The caller's unit of work: one SWE-CI task, one mini instance, one user request. |
| **epoch** | SWE-CI only: architect session → programmer session → pytest, up to `max_epoch` (default 20) per task. |

| Term | Meaning |
|---|---|
| **target agent** | The harness — the caller's, or the benchmark's. Runs a loop, owns tools, edits files. **We never launch it.** |
| **target model** | The LLM endpoint the harness asks for completions. `foresight` calls it. |
| **aux model** | The LLM endpoint `foresight` consults before rewriting the prompt. **Always present.** Served as `aux-model`. |
| **aux agent** | A second harness instance, launched by **us**, read-only. **Optional** — only when `run_aux` needs to explore a workspace. Absent under `GenericAdapter` and mini option A, where `run_aux` is a single chat call. |

`foresight` sits between agent and model on both sides. It is **not an agent**: no tools, no loop, never edits files.

A SWE-CI task therefore spans up to 40 agent sessions (2 per epoch × 20 epochs), each contributing exactly one aux run and one enhanceable prompt,
followed by however many tool round-trips the agent needs.

---

## Binding constraints

1. **No changes to the caller's repo.** Each benchmark is driven only through its own documented configuration.
2. **We define no tools.** Aux reaches the codebase through a real agent harness, never through machinery we invent.
   *Consequence for SWE-CI:* the task image ships exactly one harness — `Dockerfile.opencode` installs whichever `AGENT_NPM_PKG` was selected,
   chosen from `agent_name` in `config.py:114-131` — so aux must use that one.
   Anything else means editing the image. Read `agent_name` at startup and fail loudly on a mismatch; this is misconfiguration detection, not methodology.
3. **One agent at a time.** The aux agent runs first, read-only; only once the enhanced prompt is built does the target agent proceed.
4. **Generic by construction.** Adding caller #4 means writing one file, not editing the pipeline.

---

## How callers feed context to the model

**Every caller reaches the codebase through an agent harness.** No caller defines tools — the harness owns them.
SWE-CI writes a config containing only `baseUrl` / `apiKey` / `modelName` (`opencode.py:16-67`, `iflow.py:11-31`) and shells out:

```python
"opencode", "run", "--model", f"custom/{CONFIG.model_name}", prompt   # opencode.py:161
```

The harness owns the system prompt, the tool definitions, the agent loop and conversation state.
**What differs between callers is not how the target gets context — it always goes through a harness — but *how aux reaches the same code*.**
SWE-CI: `docker exec` into the live container.
Local: the configured path. mini: a throwaway container from the per-instance image (option B below).
In every case aux ends up reading the real codebase; only the plumbing differs, and that is what `run_aux` exists to absorb.

**SWE-CI's task setup:** `initialize.py:48-54` checks out `current_sha` and `target_sha`, strips tests from `current`, copies **target's** tests in,
pytest-runs both in Docker and diffs the reports; `tools.py:107-146` writes `current/non-passed/summary.jsonl` (one line per test passing at target but failing at current) plus per-test tracebacks. 
Each epoch (`run.py:101-108`, `run.py:127-130`) starts a fresh container and copies `current/code` + `current/non-passed` (architect) or `current/code` + `requirement.xml` (programmer) into `/app`.

**The consequence that drives the design:** SWE-CI's prompts come from a static Jinja template with only `role` and `mode` as variables (`run.py:78-79`).
Every architect call, for all 100 tasks, is byte-identical. The request body alone carries no task identity and no task content — which is precisely why aux needs the harness, not the prompt.

**SWE-CI has no "issue" (unlike SWE-bench).** What substitutes for one is per-phase and changes every epoch:

| Phase | Its "issue" | Changes per epoch? |
|---|---|---|
| Architect | `/app/non-passed/summary.jsonl` + tracebacks | **Yes** — shrinks as tests pass |
| Programmer | `/app/requirement.xml`, written by the architect this epoch | **Yes** — rewritten each epoch |

So aux must run **every session**. Caching across epochs is not merely stale, it is wrong: epoch 10's task is a different task from epoch 1's.
(Generating once from the epoch-0 gap would be coherent but would change the experiment into "hand the agent a roadmap of the actual remaining work" — a different claim from the proposal's.)

---

## Flow diagrams

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

1. harness sends a request · 2. adapter: is this a session start? · 3. if yes, `run_aux` spawns an aux agent over the workspace ·
4. aux explores, read-only · 5. aux's model calls return to foresight as `aux-model`, bypass the pipeline, exit via `Backend` ·
6. builder rewrites the task prompt, on **every** request of the session, using the cached result · 7. forward to target model · 8. relay reply.

### SWE-CI

```
HOST — run.py, per epoch: architect session, then programmer session
┌─ task container (fresh per session) ──────────────────┐
│  /app/code   /app/non-passed   /app/requirement.xml   │
│                                                       │
│  opencode #1  ── the target agent ──┐                 │
│  opencode #2  ── the aux agent ─────┼──┐              │
│     HOME=/tmp/aux-home, read-only   │  │              │
└─────────────────────────────────────┼──┼──────────────┘
                                 (1)  │  │ (4) aux-model
                                      ▼  │
                        ┌─────────────────▼─────┐      ┌──────────────┐
                        │ FORESIGHT             │─(5)─▶│  aux model   │
                        │  SweCiAdapter         │◀─────└──────────────┘
                        │   (2) session start?  │      ┌──────────────┐
                        │   (3) docker exec ────┼─(6)─▶│ target model │
                        │   (7) builder         │◀─────└──────────────┘
                        └───────────────────────┘
```

Both harnesses run in the **same** container, sequentially — `opencode #1` is blocked on our HTTP response while `opencode #2` explores.
The container ID scopes the cached aux result, so epoch 2 never reuses epoch 1's.

### mini-SWE-agent

```
HOST
┌─ mini-SWE-agent (target agent) ─┐
│   runs on the HOST              │
│   creates its own container     │
└───────────┬─────────────────────┘
            │ docker exec  (the bash tool)
            ▼
     ┌─ instance container (/testbed) ─┐        ┌─ OPTION B only ──────────────┐
     │  the repo at the base commit    │        │ throwaway container from     │
     └─────────────────────────────────┘        │ {{image}} — aux explores here │
                                                └───────────▲──────────────────┘
            │ (1) request — from the HOST, no container IP  │
            ▼                                               │
   ┌──────────────────────────────────┐      ┌──────────────┴┐
   │ FORESIGHT                        │─(4)─▶│ target model  │
   │   (2) session start?             │◀─────└───────────────┘
   │   (3) run_aux:                   │      ┌───────────────┐
   │       A — one chat call (issue)  │─(3)─▶│  aux model    │
   │       B — spawn container, explore│◀─────└───────────────┘
   │   (4) builder                    │
   └──────────────────────────────────┘
```

**Two supported options — decide at implementation time.** Both are viable; A is free, B is stronger. What is verified either way:

- `LitellmModel._query` calls `litellm.completion(model=…, messages=…, tools=[BASH_TOOL], **model_kwargs)` (`litellm_model.py:66-72`),
   so pointing it at foresight is one config file (see Wiring).
- OpenAI tool calling, a single `bash` tool, `parallel_tool_calls: true` — raw passthrough handles it.
- The task prompt is the first user message: `instance_template` wraps `{{task}}`, the PR
  description (`config/benchmarks/swebench.yaml`). Session-start detection works, and the issue
  text is unique per instance, so the default `hash(system + first_user)` is a valid session key.

#### Option A — `GenericAdapter`, aux sees the issue text only

Zero new code. mini runs on the host and creates its own container
(`DockerEnvironment._start_container`), so requests arrive with no container IP to resolve, and `instance_id` never reaches `litellm.completion`.
Aux therefore gets one chat call over the request body — which is exactly `GenericAdapter`, with every ABC default already correct.

*Cost:* asymmetric with SWE-CI. Aux reasons about future tasks from the issue text without ever seeing the code, so the mini condition is weaker evidence than the SWE-CI one,
and the write-up must say so rather than presenting them as a matched pair.

#### Option B — `MiniAdapter`, aux explores the real repo

`get_sb_environment` sets `env_config["image"] = get_swebench_docker_image_name(instance)` (`swebench.py:82-86`);
`image` is a `DockerEnvironmentConfig` field, and `env.get_template_vars()` returns `self.config.model_dump()` —
so **`{{image}}` is available in the `instance_template`, which lives in our config file, not mini's repo.**
Embedding it gives foresight an explicit per-instance identifier (stronger than SWE-CI's client-IP heuristic),
from which `MiniAdapter` starts a throwaway container from that image, lets aux explore `/testbed`, and discards it.

*Cost:* one small adapter, one extra container per session, and a marker line in our prompt template —
which slightly perturbs the prompt the target model sees, so the same template must be used in **both** experimental arms.

*Bonus:* aux gets its own container, so it is **mechanically** isolated from the target's workspace rather than relying on the prompt-level read-only instruction.
No `WorkspaceGuard` needed here.

**B is the design; A is a bring-up step.** The hypothesis is that anticipating future work produces code that makes that work easier —
which aux cannot judge from an issue description alone. Option A is worth running once to prove the mini integration end to end (config, routing, traces) before adding the container machinery,
but a reported mini result should use B. The choice is contained in one adapter and one config file.

Note `sb-cli` does **not** help here: it evaluates finished predictions, it does not expose a workspace.
mini's `swerex_modal` environment runs containers in the cloud, which could matter if no local runtime is available — but aux would then need Modal access too,
adding a dependency rather than removing one.

**Symmetry with SWE-CI under option B, and the one justified difference:**

| | How aux reaches the repo | Why |
|---|---|---|
| SWE-CI | `docker exec` into the **live** container | code mutates every epoch — aux must see *current* state |
| mini | `docker run` a **fresh** container from `{{image}}` | repo is pristine at session start, so a clean copy is equivalent |

SWE-CI cannot use the fresh-container approach: its code mutates each epoch, and the current state
exists only inside the live container.

### Local user

```
   YOU
    │  "Add rate limiting to the API client"
    ▼
┌──────────────────────┐
│ your agent           │  opencode / Aider / Cursor — edits your repo
│ base_url=:8000       │  (not Claude Code — wrong inbound protocol)
└──────────┬───────────┘
           │ (1)
           ▼
┌────────────────────────────────────┐        ┌──────────────┐
│ FORESIGHT — LocalAdapter           │──(5)──▶│ target model │
│  (2) session start?                │◀───────└──────────────┘
│  (3) spawn aux agent, cwd = repo ──┼──┐     ┌──────────────┐
│      WorkspaceGuard before/after   │  │(4)─▶│  aux model   │
│  (4) builder                       │◀─┘     └──────────────┘
└────────────────────────────────────┘
           │
           ▼   your repo — aux reads only; guard aborts on any diff
```

No container, so the aux agent runs directly on the filesystem — the one case where the read-only guarantee protects something that cannot be thrown away.
This is also the **fastest end-to-end path**: no Docker, no Slurm, no dataset.

---

## What each caller requires

| Caller | Docker | Target harness | Aux sees | Aux mechanism | Session key | Adapter |
|---|---|---|---|---|---|---|
| **SWE-CI** | required | opencode / iFlow, **in** container | the repo | `docker exec` into the same container | container ID | `SweCiAdapter` |
| **mini** — option A | pluggable — docker / singularity / udocker / bubblewrap | mini, **on the host** | the issue text only | one chat call | prompt hash | `GenericAdapter` |
| **mini** — option B | same, ×2 containers | mini, **on the host** | the repo | fresh container from `{{image}}` | prompt hash | `MiniAdapter` |
| **Local user** | not required | yours, host-side | the repo | host subprocess, `cwd` = repo | workspace + prompt hash | `LocalAdapter` |
| **Generic / tests** | not required | none | request body | one chat call | prompt hash | `GenericAdapter` |

All three intended configurations give aux the codebase — mini via option B, which is the design;
option A (issue text only) is a bring-up step, not a reportable condition. Local use
needs no container at all. SWE-CI hardcodes `docker`; mini's runtime is configurable and its
scoring can run in the cloud via `sb-cli` — see risk 1.

---

## The library

```
foresight/
  __init__.py
  context.py        # InboundRequest (body + transport facts), RequestContext
  llm.py            # ModelSpec
  backends.py       # Backend ABC + OpenAICompatBackend
  guards.py         # WorkspaceGuard protocol — M1: contract + TODO only, no implementation
  adapters/
    __init__.py     # registry
    base.py         # CallerAdapter ABC, AuxResult
    generic.py      # GenericAdapter — body-only aux call
    local.py        # LocalAdapter — host subprocess over a local repo
    swe_ci.py       # SweCiAdapter — docker exec into the task container
  stages.py         # Stage ABC + AuxStage (delegates to adapter.run_aux)
  builders.py       # Builder ABC + passthrough (control) + template (treatment)
  pipeline.py       # Pipeline + per-session aux cache
  trace.py          # JSONL trace writer
  config.py         # YAML -> objects; injects adapter, backends, guard at startup
  server.py         # OpenAI-compatible endpoint
```

### `CallerAdapter` — one file per caller

Everything caller-specific lives here. **No `if caller == …` anywhere else.**

```python
class CallerAdapter(ABC):
    name: str

    # concrete defaults — most callers never override these
    def is_agent_turn(self, req: InboundRequest) -> bool: ...      # caller's tools advertised?
    def is_session_start(self, req: InboundRequest) -> bool: ...   # no assistant message present
    def session_key(self, req: InboundRequest) -> str: ...         # caller-specific scoping
    def phase(self, req: InboundRequest) -> str | None: ...        # trace metadata only

    @abstractmethod
    def run_aux(self, req: InboundRequest, aux: ModelSpec) -> AuxResult: ...
```

One abstract method; the rest are defaults. `InboundRequest` wraps the request body **plus
transport facts** (client IP, URL path, headers) — the latter is what makes SWE-CI resolvable at
all. `AuxResult` carries the aux output plus provenance (how it was obtained, token counts,
duration, guard verdict) for the trace.

`is_agent_turn` is checked first, before `session_key` is resolved: a harness's own bookkeeping
call (opencode issues one to generate a conversation title) looks like a session start but
carries no `tools`, and treating it as a task prompt wastes a whole aux run and can race the real
one. Default: a request is an agent turn iff it advertises a non-empty `tools` array (every
supported caller's real turns do; the OpenAI protocol has a harness resend them every request).
Configurable per adapter class (`require_tools_default`) and per deployment
(`adapter.require_tools`) for a caller that legitimately never sends tools. See `README.md`,
"Design decisions specific to this milestone", for the measurement this was built against.

Three implementations cover every caller:

| Implementation | `run_aux` | `session_key` | Serves |
|---|---|---|---|
| **`SweCiAdapter`** | resolve the container from the client IP, `docker exec` a second harness into it | container ID | SWE-CI |
| **`LocalAdapter`** | spawn a harness as a host subprocess with `cwd` = the configured repo | workspace path + prompt hash | local use |
| **`GenericAdapter`** | one chat call over the request body | prompt hash (ABC default) | **mini (option A)**, tests |
| **`MiniAdapter`** *(optional)* | parse `{{image}}` from the prompt, `docker run` a throwaway container, explore, discard | prompt hash | **mini (option B)** |

**mini has two supported shapes** — see its flow diagram. Option A costs no code and gives aux the
issue text only; option B costs one small adapter plus a marker line in our prompt template and
gives aux the real repo, matching SWE-CI. Both use the ABC's default session handling. Decide when
implementing; nothing else in the system changes.

**`SweCiAdapter` specifics:**

1. SWE-CI containers use Docker's default bridge (`docker.py:117-120`, no `--network` flag), so
   each has its own `172.17.0.x`, which the proxy sees as the client address.
2. `docker exec -e HOME=/tmp/aux-home` a second harness against that container. **The `HOME`
   override is mandatory but not sufficient by itself** — see `README.md`, "For the M3 / SWE-CI
   PR", item 4. opencode keeps its session DB under `$HOME/.local/share/opencode`, and SWE-CI reads
   `opencode.db` back out to extract token usage (`opencode.py:71-106`). Sharing `$HOME` would
   corrupt the benchmark's own accounting. Two things beyond the env var: `SweCiAdapter` must
   replicate `setup_opencode` (`agents/opencode.py:16-67`) under aux's own home — `auth.json` +
   `opencode.json` pointing at foresight as `aux-model` — or aux has no provider config and cannot
   call anything; and `read_usage` (`opencode.py:71-106`) sums `SELECT data FROM message` across
   *every* message in the DB and reads `SELECT time_created, time_updated FROM session LIMIT 1`
   with **no `ORDER BY`**, so a DB genuinely shared with aux corrupts `execution_time` as well as
   token counts, non-deterministically and silently (`read_usage` returns all-`None` when the DB
   is simply absent, not when it is merely wrong).
3. Have the aux agent write its answer to a file under `/tmp` and read it back — mirroring
   SWE-CI's own `requirement.xml` pattern (`run.py:108`), and more reliable than scraping stdout.
   `/tmp` is never copied out of the container.
4. Key the cached `AuxResult` on the **container ID** (`docker inspect`), which is unique per
   container *instance*. SWE-CI creates a fresh container per *phase* — `run_locked` calls
   `run_container`/`remove_container` separately for architect and programmer, twice per epoch, all
   under one *name* generated once per task (`uuid.uuid4().hex[:16]` in `_run_locked`) — so
   container ID and session are one-to-one (and separate architect from programmer for free) only
   if keyed on the ID from `docker inspect`; the container *name* is not unique per session and
   must not be used.

**Fallback, and why it is not a valid condition.** If the container cannot be resolved,
`run_aux` degrades to a single body-only aux call and records that in the provenance. For SWE-CI
the request body is byte-identical across all 100 tasks, so such a call has no task-specific
content to work from. It keeps the run alive and makes the failure visible; **instances that fall
back must be excluded from results, never averaged in.**

Each adapter module also **owns its caller's idiosyncrasies**, documented in its docstring.
Already gathered for `adapters/swe_ci.py`:

- SWE-CI accepts exactly two harnesses — `func_map = {"iflow": …, "opencode": …}`
  (`tools.py:176-179`). **"Qwen Code" cannot be the agent without modifying the repo.** Run
  opencode or iFlow CLI driven by a locally served Qwen model; the proposal's wording needs
  adjusting to match.
- `tdd` copies the target commit's tests into the current tree (`initialize.py:54`), so the
  programmer can already read the future tests; `rdd` deletes `/app/code/tests` first
  (`run.py:129`). `rdd` is the cleaner setting for the hypothesis; `tdd` vs `rdd` is a free
  ablation.
- ANC scores sustained correctness across iterations and penalises regressions asymmetrically, so
  conservative and bold agents are not comparable on pass counts alone.
- **Architect and programmer are treated identically** — same pipeline, same aux prompt, twice per
  epoch. `phase` is recorded but never branches control flow. Expect the effect to differ between
  them, though: the architect prompt explicitly constrains scope ("1 to 5 items", "focus on the
  core contradictions"), so future-task prompting pushes against its own instructions and may
  cause scope creep, which ANC punishes. The programmer has no such tension. Because `phase` is in
  the traces, this can be tested by splitting the results afterwards at zero extra cost.

### `Backend` — which provider a model call goes to

Inbound is always OpenAI chat-completions; the callers decide that, not us. **Outbound** varies
by provider, and absorbing that variation is this layer's whole job.

```python
class Backend(ABC):
    async def complete(self, spec: ModelSpec, body: dict) -> dict: ...
    async def stream(self, spec: ModelSpec, body: dict) -> AsyncIterator[bytes]: ...
```

- **`OpenAICompatBackend`** — the default. vLLM, Ollama, any OpenAI-shaped endpoint, and Claude via the compatibility layer. Raw passthrough, zero translation.
- **`LiteLLMBackend`** — added when a provider's wire format demands it (Gemini, Bedrock). Already in the environment: mini itself is litellm-based (`litellm_model.py`).
- **`AnthropicBackend`** — only if Claude becomes a real condition *and* caching cost binds.

**Pick a backend by what the provider's wire format demands, not by how many providers it could
theoretically cover.** Making `LiteLLMBackend` the default would put a translation layer in front
of vLLM, which needs none — and raw SSE passthrough is what keeps tool calls intact on the main
experimental path.

Attached to a `ModelSpec`, **not** to a role. *Provider* choice is therefore symmetric — target
and aux may each be Qwen, Claude, or anything else — even though *agent launching* is not: the
caller launches the target agent, we launch only aux's. Because aux traffic is routed back through
foresight (below), the `Backend` layer serves aux too; that symmetry would not hold if aux called
its model directly.

| Target | Backend | Config |
|---|---|---|
| Qwen on local vLLM | `OpenAICompatBackend` | `base_url` → the Slurm node |
| Qwen3-Max / -plus | `OpenAICompatBackend` | DashScope OpenAI-compatible endpoint |
| Claude | `OpenAICompatBackend` | `https://api.anthropic.com/v1/` |

**Claude via the compatibility layer.** Anthropic publishes an OpenAI-SDK compatibility layer at
`https://api.anthropic.com/v1/`, whose support table lists everything an agent harness needs as
fully supported: `stream`, `stream_options`, `tools[].function.{name,description,parameters}`,
assistant `tool_calls`, `tool`-role messages with `tool_call_id`, `parallel_tool_calls`. Three
caveats:

1. **No prompt caching.** An agent re-sends a growing context every request, so this is a real
   cost multiplier. Routing through litellm to the native API does *not* fix it — caching needs an
   explicit `cache_control` breakpoint (or the top-level auto-caching flag), and the harness emits
   neither. The fix is our proxy injecting the marker, i.e. `AnthropicBackend`.
2. **`strict` is ignored**, so tool-call JSON is not guaranteed to match the declared schema.
3. Anthropic scopes it as *"intended to test and compare model capabilities… not a long-term or
   production-ready solution."*

Also ignored (harmless): `response_format`, `seed`, `logprobs`, `presence_penalty`,
`frequency_penalty`. System messages are hoisted and concatenated into one leading system message
— worth remembering when reading traces.

**Caching is not a risk on the open-weight path:** vLLM does automatic prefix caching server-side
with no protocol markers.

### Pipeline

`stages → builder`. **Two decisions per request, not one:**

- **Run the aux step?** Only when `adapter.is_session_start(req)`. It is the expensive part — a whole agent run — and its output does not change mid-session,
   so the `AuxResult` is cached under `adapter.session_key(req)`.
- **Apply the builder?** On *every* request of the session, rewriting the task-prompt message wherever it appears in `messages`, using the cached result.

**The second is required for correctness, not an optimisation** — and it is the single easiest thing to get wrong.
The harness keeps its own message list and never learns that we rewrote anything: it resends the **original** prompt on every subsequent request.
Enhancing only the first request would put the enhanced text in front of the model exactly once, leaving the assistant's own prior turn replying to text no longer in context.
For SWE-CI's architect — which reads files for dozens of requests and writes `requirement.xml` only at the end — the instruction would be long gone by the time it mattered.

Because the rewrite is idempotent, the conversation the model sees is byte-identical to what it would have seen had the harness been handed the enhanced prompt in the first place.
Re-applying is not "enhancing every request"; it is what stops a single enhancement from silently reverting.

The store holds **in-flight session state, not a long-lived cache**: a session start always overwrites any existing entry under that key.
Entries expire on a TTL. Because SWE-CI keys on container ID, concurrent sessions never collide and successive epochs never alias.

- **Stages** enrich a `RequestContext`, writing into `ctx.extras`. The principal one delegates to `adapter.run_aux`.
- **Builders** are the only component permitted to rewrite `messages`, so **each experimental arm
  is a builder**: `passthrough` (control — forwards unchanged) and `template` (treatment — the
  real future-task prompt). Switching arms is one config value, and both traverse identical code,
  so the only difference between them is the prompt. Two implementations are the honest minimum;
  a position ablation (inject into the system message instead of the user message) would be a
  third, if anyone commits to running it.

### `WorkspaceGuard` — detect aux contamination

Aux is read-only **by prompt instruction only**; nothing enforces it. The guard detects violations
rather than preventing them.

**A `Protocol`, not an ABC, and not a separate package.** Only one implementation is planned, so
there is no contract to enforce across a hierarchy — the Protocol documents the shape and lets
tests pass a stub. What varies is not the algorithm but *where the workspace lives* (inside a
container for SWE-CI, on the host for local), so the guard takes an "execute this command in the
workspace" callable from the adapter, which already owns that capability.

```python
class WorkspaceGuard(Protocol):
    def snapshot(self) -> list[str]: ...          # sorted manifest of the tree
    def verify(self, before: list[str]) -> list[str]: ...   # [] == intact, else changed paths
```

**`ManifestGuard`** is the single implementation: `find <root> -type f -printf '%s %T@ %p\n' |
sort`. Catches modifications, additions and deletions. Keeping the manifest rather than hashing it
means `verify` can *name* the changed files — which is why no separate `GitGuard` is needed (and
`GitGuard` could not serve SWE-CI anyway: `initialize.py:52` strips `.git*` from the workspace).
Opting out is `guard: null` in config, not a `NoopGuard` class. One extra command before and after
aux — negligible against a multi-minute agent run.

**Where it is needed:** SWE-CI (aux runs in the live container; an edit under `/app/code` is
copied out at `run.py:135` and becomes part of the measured result) and local use (aux runs on a
real repo — highest stakes, nothing to throw away). **Not** needed for mini option B, where aux
gets its own throwaway container and physically cannot reach the target's, nor for any path with
no aux agent at all.

**The policy matters more than the detector.** On mismatch, write `workspace_intact: false` into
the trace and **exclude that instance from results**. Same pattern as the body-only fallback:
record a validity flag, filter at analysis time. Silent contamination averaged into results is the
failure mode worth engineering against.

**Milestone 1 ships the Protocol and a TODO, no implementation.** `GenericAdapter` launches no aux
agent, so there is no workspace to contaminate. The file exists to mark the seam and record, for
whoever builds milestones 2–3, why it is mandatory there.

### Server

`POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.

**Routing is by served model name**, decided before any pipeline logic runs:

```
model == "target-model"  →  full pipeline  →  Backend  →  target model
model == "aux-model"     →  bypass         →  Backend  →  aux model
```

Aux agents are pointed at foresight rather than directly at the aux model. This costs one local
hop and buys three things: the `Backend` layer serves aux as well as target, aux token counts
land in our accounting for free, and every model call in the experiment appears in one trace file.
The bypass is one branch at the top of the handler, with a test asserting it never enhances.

Streaming is relayed as raw SSE bytes, never reassembled — re-serialising deltas reliably breaks
tool calls. Tool definitions in the request body are forwarded untouched.

Every request appends a JSONL trace: prompt in, aux output and provenance, guard verdict, prompt
out, timings, token counts. Without it an enhancement effect is indistinguishable from a plumbing
bug.

### Shells

The proxy is the only shell we build. A second, **agent shell** — a CLI where the user talks to
foresight directly and foresight launches *both* agents — is possible for local use, because
nobody else owns the harness there. It is impossible for SWE-CI, which launches `opencode run`
itself and offers only `base_url` / `model_name` as configuration. Keeping the core shell-agnostic
costs nothing and leaves that option open.

---

## Wiring, with no repo changes

**SWE-CI** — `config.toml` only:
```toml
base_url   = "http://172.17.0.1:8000/v1"   # docker0 gateway, reachable from the bridge network
api_key    = "dummy"
model_name = "target-model"
agent_name = "opencode"
mode       = "rdd"
max_epoch  = 5                              # cost lever; see risk 3
```

**Local** — point your agent at `http://localhost:8000/v1`, model `target-model`; set the
workspace path in foresight's config.

**mini-SWE-agent** — one config file passed with `-c`, no repo changes:
```yaml
model:
  model_name: "openai/target-model"
  model_kwargs:
    api_base: "http://localhost:8000/v1"
    api_key: "dummy"
    drop_params: true

# Option B only: expose the per-instance image so MiniAdapter can give aux the repo.
# Must be present in BOTH arms of the experiment, since it alters the prompt.
agent:
  instance_template: |
    <foresight workspace_image="{{image}}"/>
    … the rest of mini's own swebench.yaml template, unchanged …
```

**Keys.** Moot on the open-weight path — vLLM takes a dummy key. When a hosted model is added,
`ModelSpec.api_key_env` reads from the environment; commit a `.env.example` with empty values and
gitignore the real `.env`. No further hardening: SWE-CI already writes the key into a config file
inside the container itself (`opencode.py:20-24`).

**Container egress.** With a hosted aux model, the task container needs outbound internet.
SWE-CI's containers are unrestricted by default; a firewalled cluster would break this.

---

## Accepted imperfections

The "one model" illusion is not perfect, and these are accepted:

1. **Latency.** Session-start requests take dramatically longer. Nothing in the protocol reveals
   why — but the caller's HTTP client may time out. See risk 7.
2. **Token accounting.** SWE-CI reads opencode's SQLite DB (`opencode.py:71-106`), which is
   *meant* to count only target traffic, because aux runs under an isolated `HOME`. Unverified
   under `SweCiAdapter` -- the equivalent M2 measurement (host-side `OPENCODE_HOME`, not a
   container `HOME`) found opencode's config files follow the home override but its session
   database does not (`deploy/tau-slurm/README.md`), so this needs checking with a container
   runtime rather than inherited from that host result. See the `SweCiAdapter` specifics above.
   If it holds, SWE-CI's reported usage understates true cost; `AuxResult` carries aux token
   counts either way, so the write-up can report both.
3. **Wall-clock.** SWE-CI records `execution_time` in `iteration.jsonl`. The enhanced condition
   will show longer times. It does not feed ANC, but it is in the output.

And one conditional: **nothing aux does enters the caller's recorded data, provided aux writes
nothing under `/app`.** Aux's harness state is isolated under `HOME=/tmp/aux-home` and its output
goes to `/tmp`, neither of which is copied out. But that proviso is guaranteed only by the prompt
— hence `WorkspaceGuard`.

---

## Starting state

The working directory holds only `Final Project Proposal.pdf`. A scaffold written before this design settled has been **deleted deliberately**,
so the implementation follows the design rather than the design being bent around code that predates it.

---

## Milestones

**Milestone 1 — the naive version, no GPU and no Docker.**

1. Receive the request; `GenericAdapter` is injected.
2. `AuxStage` calls `adapter.run_aux`. `GenericAdapter` has no workspace access, so aux can only
   confirm the *prompt* — have it echo back a distinctive detail of what it received, so the check
   is meaningful rather than merely agreeable. No guard: with no aux agent there is no workspace
   to contaminate (`guard: null`).
3. The `template` builder prepends `"first answer what day is it, then\n"` to the task prompt —
   i.e. `template: "first answer what day is it, then\n{{prompt}}"`. Deliberately silly, and
   trivially visible in the target's reply, which is the point: it proves the rewrite reached the
   model. `passthrough` ships alongside it as the control arm.
4. Forward to the target model; relay the reply.
5. Append a trace record.

Plus `tools/fake_upstream.py` (a mock OpenAI server), `configs/naive.yaml`, `tests/`, `README.md`,
`requirements.txt`, and `guards.py` carrying the Protocol plus a TODO — no implementation, but the
seam marked and the reason recorded for milestones 2–3.

**Milestone 2 — `LocalAdapter`.** A real aux agent exploring a real repo, with `WorkspaceGuard` and the real future-task `template` builder.
**No Docker, no Slurm, no dataset** — the fastest path to seeing the whole hypothesis work end to end, and it de-risks everything SWE-CI needs.

**Milestone 3 — `SweCiAdapter`.** Container resolution, `docker exec` aux, container-ID session keying. Gated on Docker (risk 1).

**Milestone 4 — mini-SWE-agent.** A mini config file pointing at foresight, then `MiniAdapter` (option B):
`{{image}}` in our `instance_template`, a throwaway container per session, aux explores the real repo — matching SWE-CI's condition.

Run option A (`GenericAdapter`, aux sees the issue text only) **once as a bring-up step**, to prove config, routing and traces before adding container machinery.
It is not a reportable condition: aux cannot judge what makes future work easier without seeing the code.

Start with A for a cheap result; move to B if the mini condition needs to carry evidential weight. Score with `sb-cli` (cloud).

**Independent of milestone 3, not sequenced after it.** mini's container runtime is configurable and its scoring is cloud-hosted, so it may be reachable while SWE-CI is still blocked (risk 1).
If the udocker test succeeds for mini but not for SWE-CI, do this milestone first.

---

## Verification

1. **Unit** — `pytest tests/`: builder message rewriting (both arms — `passthrough` must leave
   `messages` byte-identical); session keying (same key across a session, distinct keys for
   concurrent sessions with identical prompts); aux firing exactly once per session; `aux-model`
   bypass never enhancing; adapter selection; trace shape. *From milestone 2:* guard manifest
   detecting add / modify / delete.
2. **End-to-end, no GPU** — `tools/fake_upstream.py` as both models; `curl` a completion. Assert
   the prefix reached the target, the aux reply was captured, a trace row was written. Repeat with
   `stream: true` for SSE passthrough.
3. **Protocol conformance** — replay a captured opencode request (system + user + tool
   definitions) and confirm the tool definitions survive untouched.
4. **Local, real model** — vLLM on a Slurm GPU node (use the `slurm-jobs` skill). Confirm a
   multi-request session completes, aux ran **once**, the enhanced prompt is present in *every*
   outbound request of the session, and the guard reports the repo untouched.
5. **SWE-CI, one task** — smallest splitting, `max_epoch = 1`. Confirm traces classify architect
   and programmer correctly, no instance fell back to the body-only path, SWE-CI's own token
   accounting is intact (proving `HOME` isolation), and the workspace guard passed.
6. **Provider portability** (only if a Claude target is wanted) — point a `ModelSpec` at
   `https://api.anthropic.com/v1/` via `OpenAICompatBackend` and replay the step-3 request.

Steps 1–3 need neither GPU nor Docker.

---

## Open risks

1. **No container runtime on `c-006`** — `docker not found`, no `/var/run/docker.sock`, no
   `singularity`/`apptainer`. Blocks milestones 3–4, not 1–2. But there are **three concrete leads,
   none requiring a repo change**, and they should be tried before treating this as a blocker:

   - **`udocker` is installed** (`/usr/local/bin/udocker`) and is the cluster's sanctioned
     container route (see the `slurm-jobs` skill). mini reads `MSWEA_DOCKER_EXECUTABLE`
     (`environments/docker.py`), so pointing it at udocker is one env var. SWE-CI hardcodes the
     literal `"docker"` in `docker.py`, but a `docker` shim earlier on `PATH` that forwards to
     udocker is an *environment* change, not a repo change — constraint 1 holds.
     Caveat: mini and SWE-CI both start a long-lived container and then `docker exec` into it;
     udocker is daemonless and may not support that pattern. **Test this first** — it decides
     whether milestones 3–4 are reachable at all.
   - **mini's environment is pluggable** — `_ENVIRONMENT_MAPPING` offers `singularity`,
     `bubblewrap`, `local`, `swerex_modal` alongside `docker`. Singularity/Apptainer is absent on
     the login node but is common on HPC compute nodes; check via Slurm before concluding
     otherwise. SWE-CI has no such option.
   - **Scoring needs no local runtime at all**: `sb-cli` evaluates in the cloud.

   Net: **mini may be unblocked even if SWE-CI is not**, which is a reason to keep milestone 4
   independent of milestone 3 rather than sequenced after it.
2. **Two `SweCiAdapter` mechanisms are unverified**, both because Docker is missing: client-IP →
   container resolution, and whether a second `docker exec` harness behaves cleanly against a
   container whose first harness is mid-request. The `HOME` override should isolate the state
   directory, but that is reasoning, not measurement -- and the nearest measurement available
   (M2, host-side `OPENCODE_HOME`) found the session database is *not* isolated by it, only
   opencode's config files are; whether a container's own passwd/uid boundary changes that is the
   open question. Either failure degrades to the body-only aux call — which is not a valid
   condition.
3. **Aux-as-agent cost is the dominant unknown.** Up to **40 aux agent runs per SWE-CI task**
   (2 per epoch × 20 epochs), so 100 tasks × 2 conditions × seeds is thousands of agent runs.
   Levers that do not change what is measured: lower `max_epoch`, and a smaller task subset.
   Caching aux across epochs is **not** a lever — it changes the experiment (see "SWE-CI has no
   issue"). Measure one task end to end before committing.
4. ~~mini-SWE-agent unverified~~ — **resolved.** Cloned to `~/repos/mini-swe-agent` and checked:
   litellm-based, `api_base` via `model_kwargs`, OpenAI tool calling, task prompt in the first
   user message. Runs host-side, so aux is limited to the issue text; no adapter needed.
5. **Floor effect and run-to-run variance.** The comparison is paired — same model, same tasks,
   enhancement vs. the `passthrough` no-op through identical code — so absolute scores need not
   match the published leaderboard. Two risks pairing does not remove:
   - *Floor effect.* Every model on SWE-CI's leaderboard (`docs/result.png`) is a hosted frontier
     API; the Qwen entries are the proprietary `-Max` / `-plus` tiers, not open weights. On the
     EvoScore(γ=1) axis the weakest entry, `QWen3-coder-plus`, scores ≈ 0.24. If a local 30B fails
     tasks for reasons unrelated to prompt quality, both conditions sit near zero.
   - *Variance.* Agent trajectories are stochastic. The effect must exceed that spread.

   **Before building the enhancement layer, run baseline tasks across seeds and measure both.**
6. **Served model not yet chosen**, pending the GPUs obtainable on Slurm (use the `slurm-jobs`
   skill). Candidates: Qwen3-Coder-30B-A3B (MoE, ~3B active, ~60 GB bf16), Devstral-Small-24B, or
   a smaller Qwen2.5-Coder for the model-size axis. Feeds risk 5.
7. **Caller-side HTTP timeout.** The target agent's request stays open for the entire aux run —
   potentially minutes. SWE-CI gives the whole `docker exec` 3600s (`evolve.architect.timeout`),
   but the harness's own HTTP client timeout is not under our control. Verify early; it is cheap
   to hit and expensive to discover late.

*Not* a risk: disk. The volume has 6.6 TB free and the account's quota is 1 TB with 73 GB used, so
SWE-CI's ~50 GB dataset fits comfortably.

---

## Future: Claude Code as a caller

Claude Code speaks Anthropic's native protocol, not OpenAI chat-completions, so it cannot be a caller today.
Support would need an inbound `POST /v1/messages` route plus edge translation:
request (system is a top-level parameter, content blocks, different tool schema), response, and the SSE event vocabulary — the last being the fiddly part.
**Nothing in the pipeline, builders or adapters changes**, because `InboundRequest` is already the seam.
If the target is also Claude, the outbound side needs no conversion at all, making that the easiest combination to build first.
