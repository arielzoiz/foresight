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

## Run it for real: models served by vLLM on Slurm

### At a glance (the first SWE-CI A/B)

**Server side, on Slurm** (whoever runs the models)
- Two independent vLLM servers, one job each, same checkpoint (`Qwen3-Coder-30B-A3B-Instruct-FP8`), same
  node, one B200 GPU each, no tensor parallelism, `--max-model-len 262144`,
  `--tool-call-parser qwen3_coder`, no auth, 12 h jobs.
- `target` (`:8001`, served name `target-model`) is what SWE-CI calls directly in the control arm.
  `aux` (`:8002`, `aux-model`) is used only by foresight's aux agent in the treatment arm; SWE-CI never
  calls it. foresight does not run here.

**Caller side, your machine over the TAU VPN**
- SWE-CI, Docker and foresight. Control arm: SWE-CI to target. Treatment arm: SWE-CI to foresight to
  target, with aux exploring the task container. Run the arms one after the other, `max_workers = 1`.
- After the run the caller collects the results (`tools/collect_swe_ci_ab.py`). The server side then adds
  what only it knows (vLLM flags, sampling defaults, partition, job ids, logs) to `server-side/` in the
  same results folder.

foresight is a light HTTP proxy (no GPU). Only the two model endpoints, `target` and `aux`,
need a GPU, and they are ordinary OpenAI-compatible vLLM servers, so they can run as Slurm jobs
(`deploy/tau-slurm/`, a site deployment, not part of the library). Where foresight itself runs
depends on the caller:

| Caller | foresight runs on | Why |
|---|---|---|
| a harness on the cluster, no Docker (`LocalAdapter`) | a Slurm CPU job (`deploy/tau-slurm/launch.sh`) | starts models, proxy and aux agent together |
| SWE-CI (needs Docker) | **your machine**, models on Slurm, over the TAU VPN | there is no Docker on the cluster, and SWE-CI's containers must reach foresight, so it has to sit where Docker sits |

Models on Slurm, foresight on your machine (`deploy/local-swe-ci-slurm-models/README.md` has the
full walkthrough and the exact `sbatch` line):

1. **Once, on the login node:** `deploy/tau-slurm/setup/install_vllm_env.sh`, then
   `deploy/tau-slurm/setup/prefetch_model.sh <hf-model-id>` (weights are fetched before a GPU job
   needs them, so a preempted job never restarts the transfer).
2. **One vLLM job per role**, submitting `deploy/tau-slurm/serve_vllm.sbatch` directly (not
   `launch.sh`, which would also start its own foresight job). Per job, via `--export`:

   | Variable | Meaning |
   |---|---|
   | `FORESIGHT_ROLE` | `target` or `aux` (names the endpoint file) |
   | `FORESIGHT_MODEL_ID` | Hugging Face model id |
   | `FORESIGHT_ALIAS` | served model name; two space-separated names let one loaded model answer as both roles |
   | `FORESIGHT_PORT` | port to bind, one per job when the roles use separate jobs |
   | `FORESIGHT_TOOL_CALL_PARSER` | e.g. `qwen3_coder`; the harness needs working tool calls (default `auto`) |
   | `FORESIGHT_MAX_MODEL_LEN` | context length (default 32768; SWE-CI prompts need far more, 262144 for the model above) |

   Cold start takes 15-50 min; the job writes `<run dir>/<role>.endpoint` once vLLM answers.
   Request a `--time` longer than the whole run. The jobs do not stop themselves: `scancel` them.

   Choices that mattered (as reported by the server side, 2026-09-20):
   - **Checkpoint:** pick one that loads fast from NFS. `Qwen3-Coder-30B-A3B-Instruct-FP8` (31 GB, one
     GPU, no tensor parallelism) loads in about 10-35 min; `Qwen3-Coder-Next-FP8` once took 4.5 h.
     `--max-model-len 262144` is this model's native context, so no rope scaling.
   - **Partition:** the shared `*-killable` pools can preempt a job mid-experiment. `gpu-b200` is
     non-killable and cs_dcor-owned, so the two jobs cannot be preempted; the tradeoff is that submitting
     there forcibly requeues other users' lower-tier jobs on that node to free the GPUs.
   - **Hardware:** only B200 and H200 are confirmed to run this vLLM build (CUDA-13 wheels). a6000 runs
     out of KV-cache memory on one GPU, l40s and older cards have drivers too old for CUDA 13, and both
     H100 nodes are broken or blocked.
   - **Wall clock:** the first A/B used 12 h jobs; the run itself took about 3.5 h after the load.
3. **Check reachability from the machine running foresight**, and use the node's fully qualified
   name (measured: the short hostname did not resolve off-cluster, the FQDN did):

   ```bash
   nc -zv <node-fqdn> <port>
   curl http://<node-fqdn>:<port>/v1/models
   docker run --rm curlimages/curl -s -m 10 http://<node-fqdn>:<port>/v1/models   # from a container too
   ```

   The container check matters because SWE-CI's containers, not the host, make the calls: to the target
   directly in the control arm, to foresight in the treatment arm.
4. **Point foresight at them** (`configs/swe_ci.yaml` is the template). `served_name` is what the
   caller addresses foresight as; `model` is what the upstream calls the model. Leave out
   `api_key_env` for an endpoint without auth. Both roles may share one URL.

   ```yaml
   models:
     target: {served_name: target-model, backend: openai_compat, model: target-model,
              base_url: "http://<node-fqdn>:8001/v1", timeout_s: 3600}
     aux:    {served_name: aux-model,    backend: openai_compat, model: aux-model,
              base_url: "http://<node-fqdn>:8002/v1", timeout_s: 1800}
   ```
5. `python -m foresight.server --config <your.yaml>`; the caller's `base_url` is this server.

### Running with SWE-CI

Everything but the models runs on one machine: Docker, the SWE-CI clone (its own venv), and
foresight. SWE-CI's repo stays unmodified; the only change on its side is `base_url`.

One-time setup: Docker Desktop running and the VPN on; in the SWE-CI clone
`conda create -p .venv python=3.11 && .venv/bin/pip install -r requirements.txt`; the `foresight` env from
"Setup" above.

1. **Pick the tasks.** A task is a row of the benchmark csv `metadata/<splitting>.csv` (1-based data
   rows). Back up `default.csv`, write the header plus your rows into it, and fetch only those
   tasks with `swe_ci.download.download_hf_folder(CONFIG.hf_repo_id, f"data/{task_id}",
   CONFIG.save_root_dir, hf_token)` (`PYTHONPATH=src`). Do not run `python -m swe_ci.download`: it
   overwrites the trimmed csv and fetches the whole ~50 GB dataset.
2. **Two SWE-CI configs that differ only in `experiment_name`, `base_url` and `model_name`.**
   *Control* points at the target endpoint (`api_key = "dummy"`); *foresight* points at
   `http://<address containers can reach>:<foresight port>/v1` with `model_name = "target-model"`.
   Everything else must match: `agent_name = "opencode"`, `mode`, `splitting`, `evolve.max_epoch`,
   `max_try`, and `max_workers = 1` (see below). On macOS with Docker Desktop, containers reach the
   host at `host.docker.internal`, and SWE-CI needs `docker.storage_disk = "local-docker-desktop"`,
   `read_bps = write_bps = ""` and a small `memory`/`memory_reservation` (`2048mb`/`1024mb`); on
   Linux the bridge gateway `172.17.0.1` is used.
3. **A foresight config for the run:** copy `configs/swe_ci.yaml` and set the model URLs,
   `adapter.foresight_base_url` (the address from step 2), `adapter.swe_ci_config` (absolute path
   of the foresight SWE-CI config), a unique `trace.path`, `trace.log_replies: true` to keep what the
   target replied, and `adapter.aux_export_dir` to keep what the aux agent actually did (its own
   opencode sessions, exported from the container before SWE-CI deletes it).
4. **Run the arms one after the other, never together:**

   ```bash
   # from the SWE-CI dir, control arm
   PYTHONPATH=src nohup .venv/bin/python -u -m swe_ci.evaluate --config_file config_control.toml > control.log 2>&1 &

   # foresight arm: start the proxy, check it from the host AND from a container, then run
   python -m foresight.server --config configs/<your>.yaml
   curl http://localhost:<port>/v1/models
   docker run --rm curlimages/curl -s http://host.docker.internal:<port>/v1/models
   PYTHONPATH=src nohup .venv/bin/python -u -m swe_ci.evaluate --config_file config_foresight.toml > foresight.log 2>&1 &
   ```

   Keep `evolve.max_workers = 1`. foresight finds the calling container by IP, or failing that as
   "the only running container" (`resolved_by: sole_container`, what Docker Desktop gives). With
   several containers running and no IP match it silently degrades to a body-only aux call
   (`provenance.fallback: body_only`), which is not the condition under test; exclude those runs.
   Under Linux Docker, see "Keeping SWE-CI's task concurrency" below.

   Watch for:
   - Container exits with code 137 are normal teardown after each pytest run, not an out-of-memory kill.
   - `Cannot connect to API ... Max attempts reached` means an endpoint died. Stop, and do not retry
     against a dead endpoint.
   - A failed experiment folder must be renamed or removed before re-running under the same name, because
     the same name resumes from its checkpoints.
   - `timeout` does not exist on macOS, and zsh errors on an unmatched glob; use `find`/`ls` in scripts.
5. **Collect the results** into `results/`, one command for both arms:

   ```bash
   python tools/collect_swe_ci_ab.py --swe-ci-dir ../SWE-CI --metadata-csv ../SWE-CI/metadata/<full>.csv \
       --arm control=<control experiment>:config_control.toml \
       --arm foresight=<foresight experiment>:config_foresight.toml \
       --trace foresight=traces/<foresight experiment>.jsonl \
       --aux-export-dir foresight=<aux_export_dir> --foresight-config configs/<your>.yaml \
       --stdout-log control=control.log --stdout-log foresight=foresight.log \
       --model-url target=http://<node-fqdn>:8001/v1 --model-url aux=http://<node-fqdn>:8002/v1
   ```

   It writes `RUN.md` (results, timing, cost, aux provenance, incidents), `manifest.json` (rows,
   each task's commits, settings and whether the arms' settings matched), `metrics.csv` (per epoch:
   gap, lines changed, maintainability index, pylint note), per-epoch failing tests (and the failing
   tests after the last epoch, in `final/`), `requirement.xml` and code diffs, the aux runs paired
   with their base task, the console output of each `swe_ci.evaluate` process, and plots. Check
   that no aux run has a `fallback`.
6. **The server side adds its part**, after the caller side has collected: into
   `results/<label>/server-side/`, following the checklist the collector left in `TEMPLATE.md` (models and
   vLLM version, launch flags, the sampling defaults the servers applied, hardware, job ids and times,
   events such as preemptions or OOMs, other load on the node, and the vLLM logs). Sampling defaults
   change results and are visible only on this side. The collector never overwrites that folder, so a
   re-collection is safe. Then commit the results folder.

### Naming conventions for A/B tests

Start every name with `ab-test`, then `swe-ci`, the rows and the epoch cap. Rows are 1-based data
rows of the **full** `metadata/<splitting>.csv` (not a trimmed copy): `6-10`, or `1-2_5` for a
non-contiguous set. The epoch cap is `evolve.max_epoch`.

| What | Pattern | Example |
|---|---|---|
| results folder | `results/ab-test__swe-ci__rows-<rows>__ep<epochs>` | `ab-test__swe-ci__rows-6-10__ep20` |
| SWE-CI `experiment_name` | `ab-test__swe-ci__<arm>__rows-<rows>__ep<epochs>`, `<arm>` is `control` or `foresight` | `ab-test__swe-ci__foresight__rows-6-10__ep20` |
| foresight trace | `traces/<experiment_name>.jsonl` | `traces/ab-test__swe-ci__foresight__rows-6-10__ep20.jsonl` |
| aux export dir | `traces/<experiment_name>__aux-agent/` | |

`collect_swe_ci_ab.py` derives the results folder name from the data (`--label` overrides it). SWE-CI
resumes any experiment whose name already exists, so a repeat of the same rows needs a new name:
append `__run2` to the experiment names and to the results folder.

**Merging batches later** (say rows 6-10 and rows 11-15 at 20 epochs each): task ids are unique, so
the `data/<arm>/<task_id>/` folders of two batches can be combined by copying. Only merge batches
with the same epoch cap, arms, models and settings (compare `settings` in each `manifest.json`).
EvoScore is averaged over `max_epoch` epochs, and SWE-CI pads a task that stopped early with its
last value, so recompute it over all merged tasks from the `iteration.jsonl` files with one epoch
cap; averaging two batches' averages is only right when they have the same number of tasks. There is
no merge tool yet.

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

By default a record holds prompts and token counts, not what the target *said*. Set
`trace.log_replies: true` to also keep the target's reply on each row as `reply`: its text and every tool
call (`name`, `arguments`), the finish reason, and `truncated` if it was cut at `trace.reply_max_chars`
(default 50,000). Replies are small next to prompts (about 1 MB per arm for a 25-epoch SWE-CI run), and
what the agent later feeds back to the model, the tool outputs, is not recorded. Aux replies are never
logged here: the aux agent's own record is `adapter.aux_export_dir`. Streamed and non-streamed replies use
the same shape, and the relayed bytes are untouched either way.

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

**Update:** step 4 is done. A real SWE-CI A/B, five tasks and five epochs against
`Qwen3-Coder-30B-A3B-Instruct-FP8` served by vLLM on Slurm, is in
`results/ab-test__swe-ci__rows-1-5__ep5/`. What follows is the earlier, local-model attempt.

**Step 4 -- one real task, end to end, with a real model -- was open**,
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
- **Done: `SweCiAdapter` against real Docker, with a 30B model served on Slurm**
  (`results/ab-test__swe-ci__rows-1-5__ep5/`; inbound requests from a laptop to a job's node
  work over the VPN). Originally: try `qwen3-coder:30b` -- the smaller models tested so far (`qwen2.5-coder`,
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
