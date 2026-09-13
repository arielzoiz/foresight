# foresight on the TAU CS Slurm cluster

**This is one site's deployment, not part of the `foresight` library.** The library
(`foresight/`) is cluster-agnostic and knows nothing about Slurm; everything under this
directory exists only to run it — plus a real vLLM and a real opencode agent — on the
TAU CS cluster (`slurm-client.cs.tau.ac.il`). If you're on a different cluster, this
directory is a template, not a dependency.

It stands up three things, wired together by a run directory full of small files:

```
serve_vllm.sbatch (target)  ─┐
serve_vllm.sbatch (aux)     ─┼─► foresight.sbatch ──► the proxy, ready to be curled
   (--single: just one)     ─┘
```

In `--single` mode one vLLM instance backs both `target-model` and `aux-model`. In
`--split` mode each gets its own vLLM job — potentially different checkpoints.
`launch.sh` is the only entry point; the two `.sbatch` files are not meant to be
submitted by hand.

## 1. One-time setup

Run these from the **login node** (they need outbound internet):

```sh
export FORESIGHT_ROOT=<your-working-area>   # only if you don't already have $WORK set
deploy/tau-slurm/setup/install_vllm_env.sh
deploy/tau-slurm/setup/install_opencode.sh
deploy/tau-slurm/setup/prefetch_model.sh Qwen/Qwen2.5-Coder-1.5B-Instruct
```

- `FORESIGHT_ROOT` is resolved once, the same way everywhere in this deployment:
  `--root` flag → `$FORESIGHT_ROOT` → `$WORK` → **abort**. Nothing here falls back to
  `$HOME` silently — see §4.
- `install_vllm_env.sh` creates a conda env at `$FORESIGHT_ROOT/conda-envs/foresight-vllm`.
- `install_opencode.sh` installs a pinned node + `opencode-ai` under
  `$FORESIGHT_ROOT/agent/`, and writes a wrapper at `$FORESIGHT_ROOT/agent/bin/opencode`.
- `prefetch_model.sh <hf-id>` downloads one model's weights before any GPU job needs
  them — do this on the login node, not on a compute node, so a preempted job never
  restarts a multi-GB transfer. Requires `install_vllm_env.sh` to have run first (it
  provides the `hf`/`huggingface_hub` CLI used here). Run it once per model you want to
  serve.
- **You also need the base `foresight` package itself installed** (its own
  `README.md`, `## Setup`) in a conda env — by default named `foresight`, which is what
  `launch.sh --py-env` expects.

All of these are idempotent — safe to re-run.

## 2. Running

```sh
# the full pipeline: one model backing both roles, a REAL aux agent over a repo
deploy/tau-slurm/launch.sh --single Qwen/Qwen2.5-Coder-7B-Instruct \
    --workspace $WORK/workspaces/tinyrepo \
    --account gpu-research --foresight-account gpu-research

# two models, two jobs -- checkpoints may differ
deploy/tau-slurm/launch.sh --split Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct \
    --workspace $WORK/workspaces/tinyrepo \
    --account gpu-research --foresight-account gpu-research
```

Full flag list: `deploy/tau-slurm/launch.sh --help` — it also prints your own
`sacctmgr -P -i show user -s $USER` output, so you can see which
partition/account pairs you actually have.

### `--adapter`: which of the two aux mechanisms runs

| | `--adapter local` (default) | `--adapter generic` |
|---|---|---|
| Aux is | a real opencode subprocess, `cwd` = `--workspace` | one chat call over the request body |
| Aux sees | the actual code | the task text only |
| Needs | `--workspace`, `install_opencode.sh` | nothing extra |
| Guard | `manifest`, armed | `null` — nothing to contaminate |
| Template | `configs/vllm-local.yaml.tmpl` | `configs/vllm.yaml.tmpl` |
| Milestone | 2 | 1 |

`local` is the design; `generic` is a bring-up rung. When a `local` run
misbehaves, re-run it as `generic`: if that works, the fault is in the agent
(model tool-calling, opencode config, workspace) rather than in the topology.

**`--workspace` must be a scratch checkout.** The guard *detects* writes, it
does not prevent them — aux is read-only by prompt instruction alone. A detected
write fails the request with `workspace_contaminated` rather than quietly
producing a contaminated measurement, but the write has already happened. Clone
something disposable:

```sh
git clone <repo> $WORK/workspaces/tinyrepo
```

Keep it modest in size: the guard runs `find -type f -printf` over the whole
tree twice per session, and on NFS that is not free.

**Why the default partition is `killable`, not the account-free `studentkillable`.**
`studentkillable`'s hardware is exclusively Titan Xp (Pascal, compute capability 6.1).
Confirmed by direct testing, not a theoretical concern: `pip install vllm` currently
resolves a build compiled for CUDA 13, which Pascal cannot run at all —
`vllm serve` crashes at GPU-init with `RuntimeError: CUDA unknown error`. `killable`'s
pool (a5000/a6000/l40s/rtx_3090/rtx_2080/v100/quadro) is Volta-or-newer throughout and
works. The cost is that `killable` needs `--account` for most accounts, which
`studentkillable` never required — check `sacctmgr` (above) for yours.

**The vLLM job(s) and the proxy job commonly need *different* accounts on top of
that.** On this cluster, `cpu-killable` (the default proxy partition) needs
`--account=gpu-research`, independent of whichever account the vLLM job needs. That's
why `--account` and `--foresight-account` are two separate flags, not one shared flag —
if you get `Invalid account or account/partition combination specified` from `sbatch`,
this is almost always why.

**`--exclude`, default on.** Three individually-managed legacy nodes
(`rack-bgw-dgx1`, `rack-gww-dgx1`, `rack-omerl-g01` — as opposed to the
centrally-imaged `n-*`/`t-*` pool) are excluded from vLLM job scheduling by default.
`rack-bgw-dgx1` was directly confirmed to fail vLLM startup two different ways: a
system `libstdc++` too old for vllm's dependency chain, and separately a GPU driver
capped at CUDA 12.4 while pip installs a CUDA-13 build (vLLM ships no older
pre-built variant than cu128, so there is no "just use an older CUDA build" fix here).
The other two share the same naming/provisioning pattern and are excluded
defensively, not individually confirmed. Override with `--exclude ""` if you want to
try them, or `--exclude other,nodes` to adjust the list.

**Tool calling is enabled by default, with no flag needed.** `serve_vllm.sbatch`
always passes `--enable-auto-tool-choice --tool-call-parser hermes` (the format
Qwen2.5-Instruct's chat template emits). Without this, vLLM 400s on *any* request
carrying a `tools` array — which every message from a real agent harness does, since
it always advertises its own tools. Confirmed directly: pointed at a server started
without these flags, an opencode session retried the same 400 forever — 1215 requests
in 45 minutes, never making progress, silently burning the GPU allocation the whole
time. Override with `FORESIGHT_TOOL_CALL_PARSER=<name>` before calling `launch.sh`
for a non-Qwen model needing a different parser (vLLM ships dozens — see its own
`--tool-call-parser` docs or `vllm/tool_parsers/__init__.py` in the installed
package), or `FORESIGHT_TOOL_CALL_PARSER=""` to disable tool calling entirely — but
expect the same retry-storm failure mode against any real agent harness if you do,
unless the model never receives a `tools` array at all (true for `GenericAdapter`'s
aux call, not for a real target or aux **agent**).

**The `max_tokens` collision — read this before blaming the model.** An earlier
version of this README attributed the 400 storm on real opencode traffic to model
tool-call quality. That was wrong, and the real cause is worth knowing because it
looks identical from the outside.

For a model it does not recognise — which any `custom` provider is — opencode
defaults its output limit to **32000** and sends it as `max_tokens` on every
request. Against `--max-model-len 32768` that leaves 768 tokens for the prompt,
and opencode's own system prompt is bigger than that, so vLLM rejects every
request before it ever reaches the model:

```
This model's maximum context length is 32768 tokens. However, you requested
32000 output tokens and your prompt contains at least 769 input tokens, for a
total of at least 32769 tokens.
```

opencode retries forever — 2901 rejected requests before the run was killed at
25 minutes, zero progress, GPU allocation burning throughout. It is pure arithmetic, it is
deterministic, and it has nothing to do with the checkpoint: the 1.5B and the 7B
fail identically.

The fix is in `tools/opencode_home.sh`, which now declares
`limit: {context, output}` in the provider config; `foresight.sbatch` passes
`--context` = the vLLM job's `--max-model-len` and `--output` =
`FORESIGHT_AUX_MAX_OUTPUT` (default 4096). If you raise `--max-model-len`, the
context declaration follows automatically. **If you ever write an opencode
provider config by hand, set these two fields**, or you will reproduce the storm.

**Model tool-call quality is a real but separate problem.** With the parser
enabled and the `max_tokens` collision fixed, `Qwen2.5-Coder-7B-Instruct` still
does not emit tool calls in the hermes `<tool_call>` shape vLLM's parser expects
— it answers with a fenced ```json block or `<function_call>` XML instead, so
`tool_calls` comes back `null` and the agent never uses a tool. Verified this is
not a template fault: the model's own chat template does inject the correct
`<tool_call>` instruction (checked by rendering it directly with
`apply_chat_template`), and the model simply does not follow it. Plain chat
completions and `GenericAdapter`'s tool-free aux calls both work cleanly.

So a real aux **agent** needs a checkpoint that is reliable at tool calling. The
`gpu-h100-killable` / `gpu-h200-killable` / `gpu-b200-killable` partitions make a
32B practical, which is the next thing to try; `--adapter generic` remains
available meanwhile and needs no tool calling at all.

### How to tell whether aux actually read the repo

**This is the difference between a reportable condition and a weaker one**, and
the trace answers it with no extra tooling. Every model call the aux agent makes
comes back through foresight as a `role: "aux"` row, because the agent is
pointed at `aux-model` rather than at vLLM directly. Count them:

```sh
python -c "
import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r['role'] == 'aux':
        print('messages=%-3d tools_offered=%d' % (r['message_count'], r['tool_count']))
" $RUN/trace.jsonl
```

A grounded run shows **three or more** aux calls with `message_count` growing:
the agent asks, the model answers with a tool call, opencode runs it and sends
the result back. An ungrounded run shows exactly two and stops:

```
messages=3   tools_offered=0     <- opencode's title-generation call
messages=2   tools_offered=10    <- the one agent turn; no tool round-trip followed
```

That is a real measurement from `Qwen2.5-Coder-7B-Instruct`: `tools_offered=10`
means opencode advertised its tools, and no third call means none of them ever
ran. The resulting future-task list was plausible but generic — "add unit
tests", "update the documentation" — with no reference to a single file in the
workspace, which is exactly what an answer derived from the task text alone
looks like.

**The row count tells you the run was ungrounded; it does not tell you why.**
Two different failures produce the same two rows, and they point at different
fixes, so read the aux answer text before concluding anything:

| aux text looks like | what happened | where to look |
|---|---|---|
| a clean prose answer | the model never attempted a tool | the prompt, or the model |
| a JSON or XML blob naming a tool | it attempted one and the format was rejected | the model's wire format, or `--tool-call-parser` |

The 7B produced *both*, depending on the prompt — see the next section.

`aux.usable: true` in the trace does **not** contradict this: the quality gate
counts enumerated items, not groundedness. A fluent, well-formatted, entirely
ungrounded answer passes it. Check the aux-row count as well.

**It is the model, not the prompt — tested.** The obvious suspect was the aux
prompt, which ended "Answer with the numbered list and nothing else" and could
plausibly have been read as *do not explore*. Replacing that with an explicit
"USE YOUR TOOLS to inspect the repository, do not answer from the task
description alone" changed the model's behaviour but not the outcome. It tried:

```json
{"name": "glob", "arguments": {"pattern": "**/*.{js,ts}"}}
```

A fenced JSON block, not the `<tool_call>` tags the hermes parser needs — so
`tool_calls` came back null, opencode had no tool call to execute, treated the
text as the assistant's final answer, and stopped at two aux calls again.

So the model does not *decline* the tools: under an explicit instruction it
reaches for one and cannot emit it in a form anything downstream will accept.
That is a wire-format failure, not a reasoning one, which is why a better prompt
cannot fix it and a more capable model plausibly can. (It also globbed for
JavaScript in a Python repo, so its tool *choice* was poor too.)

Note the tool-forcing prompt made the *output* worse with this model: that raw
JSON blob became the "future tasks" injected into the target prompt, with
`items: 0, usable: false`. The gentler prompt at least yields a well-formed if
ungrounded list. Revisit the stronger prompt when the model can actually call
tools; it is the better prompt for a model that can.

## 3. Where things land

Every run gets its own directory:

```
$FORESIGHT_ROOT/foresight-runs/<user>-<timestamp>/
    target.endpoint      "http://<node>:<port>/v1"   -- written once that vLLM answers 200
    aux.endpoint          (same content as target.endpoint in --single mode)
    foresight.endpoint    "http://<node>:<port>/v1"  -- the proxy itself, once /health is up
    config.yaml            rendered from configs/vllm-local.yaml.tmpl (or vllm.yaml.tmpl)
    trace.jsonl            one record per model call -- the experiment's evidence
    logs/
        vllm-target-<jobid>.out / .err
        vllm-aux-<jobid>.out / .err        (--split only)
        proxy-<jobid>.out / .err
```

**`trace.jsonl` is the thing to read**, not the logs. One row per model call, on
shared storage so you can watch it from the login node while the run continues.
The keys that matter: `prompt_in` vs `prompt_out` (did the enhancement apply, and
did it re-apply on *every* request of the session?), `aux.usable` (did aux
actually produce a task list, or just chatter?), `guard.workspace_intact`, and
`error`. Rows with `role: "aux"` are the aux agent's own model calls, which reach
the same file because aux is pointed back at foresight rather than at vLLM.

```sh
# did the enhancement reach the model, on every turn?
python -c "
import json,sys
for l in open('$RUN/trace.jsonl'):
    r = json.loads(l)
    if r['role'] != 'target': continue
    print(r['is_session_start'], r['enhanced'], r['prompt_in_chars'], '->', r['prompt_out_chars'], r['aux'] and r['aux']['usable'])
"
```

**One thing that is not in the run directory: the aux agent's opencode home.** It
lives at `/tmp/foresight-aux-home-<jobid>` on the *proxy* node, deliberately —
opencode materializes a `node_modules` tree on first use in a fresh home, which
took over 6 minutes onto this cluster's NFS versus under 20 seconds on local
`/tmp`. It is not cleaned up on exit, so its session DB survives for debugging a
failed aux run; `killable` nodes reclaim `/tmp` on their own.

`launch.sh` prints this path, plus a ready-made `tail -f` and `scancel` command, when it
submits. Nothing appears in `target.endpoint` / `aux.endpoint` until that vLLM has
actually answered `/v1/models` with 200 — not just once the Slurm job starts running —
so "the file isn't there yet" while a job shows `PD` in `squeue` is normal, not a bug.

```sh
squeue --me
tail -f $RUN/logs/*.out
cat $RUN/foresight.endpoint      # once ready, e.g. http://n-601:8000/v1
curl "$(cat $RUN/foresight.endpoint | sed 's#/v1$#/health#')"
```

### How long startup takes, and why the answer is a range

**15 to 50 minutes for a 14 GB checkpoint, and the variance is the filesystem,
not your config.** Weights are read from `$HF_HOME` over NFS on storage shared
far beyond this project — the volume these numbers came from was ~88% full at
the time — so throughput depends on which node you land on and what else is
hitting the volume:

| node | GPU | outcome |
|---|---|---|
| n-301 | rtx_3090 | 2.6 min/shard, serving in ~15 min — **twice** |
| n-306 | rtx_3090 | **wedged**: 0 bytes of log in 29 min, killed |
| n-601 | a6000 | 12.5 min/shard (~4.7 MB/s), never reached serving in 1 h |
| n-803 | l40s | failed at GPU init — driver too old (see below) |

Same checkpoint, same code, same evening. Budget for the slow case and do
something else meanwhile.

**Read that table carefully before reaching for `--constraint`.** n-301 and
n-306 are the *same GPU model* and produced the best and worst outcomes of the
night. GPU type predicts **driver compatibility** — which is real and worth
constraining on — but it does **not** predict throughput. Pinning
`geforce_rtx_3090` because n-301 was fast would have had a 50% chance of landing
on n-306 and hanging. Per-node variance on a shared filesystem dominates, and
one measurement per node is not a basis for a rule.

What this means in practice:

- **A long `Prefetching checkpoint files` phase is normal**, not a hang. Confirm
  rather than assume — see "Is it stuck, or just quiet?" below.
- **Don't restart a slow load.** Shards already read are in the node's page
  cache, so continuing is cheaper than starting over, and a restart re-pays the
  full cost on a fresh node lottery.
- **Use `--constraint` to dodge broken drivers, not to chase speed.** The one
  thing GPU type reliably predicts is whether the node's NVIDIA driver can run
  the CUDA-13 build pip installs for vLLM. Measured drivers: rtx_3090 595.84,
  a6000 595.84, a5000 580.17, h100 610.57 — all fine; **l40s 535.18 — too old**,
  fails at GPU init. If you constrain, constrain away from l40s.
- **A smaller checkpoint is the real fix for plumbing tests.** The 1.5B is ~3 GB
  and loads in a few minutes. Use it to test the pipeline; save the big model
  for when the model itself is what you are testing.

### Is it stuck, or just quiet?

**vLLM's log block-buffers.** `EngineCore` is a child process writing to a file,
not a TTY, so a healthy job can print nothing for 10+ minutes. Observed here: the
log sat at `Prefetching...` for 11 minutes while the weights were already
resident on the GPU. That is indistinguishable, from the log alone, from a node
that has genuinely wedged (also observed — n-306 held a GPU for 29 minutes
having written zero bytes).

Look inside the allocation. `srun --overlap` attaches to a job already running —
no new reservation, and not subject to the `srun --pty bash` ban:

```sh
srun --overlap --jobid=<vllm-jobid> --ntasks=1 sh -c \
  'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3
   ps -o pid,etime,pcpu,rss,comm -u $USER --sort=-pcpu | head -5'
```

GPU memory climbing, or `VLLM::EngineCore` burning CPU with growing RSS, means
progress the log has not flushed. Flat memory and an idle process means actually
stuck: `scancel` and resubmit, ideally with a different `--constraint`.

### The proxy waits on the vLLM job, not on a clock

`foresight.sbatch` used to give up after a fixed `FORESIGHT_WAIT_TIMEOUT`, and
killed a healthy run at exactly 1800s while vLLM was still loading. It now polls
the vLLM job's Slurm state instead: while any producer is `PENDING`/`RUNNING` the
endpoint may still arrive, so it keeps waiting; once they have all left the queue
without publishing one, it fails immediately and says so rather than serving out
the clock. `FORESIGHT_WAIT_TIMEOUT` (now 7200s) survives only as a backstop for
an alive-but-hung server.

Consequence worth knowing: **a stuck vLLM will hold the proxy job indefinitely.**
That is the intended trade — the alternative killed good runs — but it means you,
not the timeout, are responsible for noticing. Watch the run rather than
assuming it will fail on its own.

## 4. Paths and quota

**Where model weights go.** Every script here resolves `HF_HOME` once, the same
way, and never passes weight paths to each other — so there is exactly one thing
to get right. In precedence order:

| if… | `HF_HOME` becomes | why you might want this |
|---|---|---|
| `HF_HOME` is already set in your environment | left exactly as it is | you already have a populated cache and do not want a second copy |
| it is unset | `$FORESIGHT_ROOT/.cache/huggingface` | the normal case; nothing to configure |
| you pass `--root DIR` to a setup script | `DIR/.cache/huggingface` | moves the cache, the runs and the opencode install together |

Check where yours will land before a big download:

```sh
export FORESIGHT_ROOT=...            # or rely on $WORK
echo "${HF_HOME:-$FORESIGHT_ROOT/.cache/huggingface}"
```

- **There is no shared team cache.** Each of you downloads your own copy of whatever
  model you serve, into your own quota. Fine for the 1.5B bring-up model (~3 GB);
  worth a conversation before anyone downloads something in the tens of GB. If you do
  want to share one, agree a directory and export `HF_HOME` to it — the precedence
  above already supports that with no code change.
- **`$HOME` is a small, separate filesystem from your working area** — on this cluster
  it has a couple of GB of quota, while the working area has terabytes. Model weights,
  conda envs, npm installs and run directories all belong in the working area. Every
  script here refuses to write `FORESIGHT_ROOT` or `HF_HOME` under `$HOME` and aborts
  with an explicit message rather than silently filling it — if you see that abort, you
  passed (or defaulted to) the wrong path, not a bug in the guard.
- Models are named by **HF id only, never by path** — `prefetch_model.sh` and
  `serve_vllm.sbatch` agree on where weights are purely through `HF_HOME` and HF's own
  cache layout (`$HF_HOME/hub/models--<org>--<name>/...`). Swapping which checkpoint
  backs a role is a `launch.sh` argument, never a file to edit.

## 5. Cluster constraints that surprise people

- **`srun --pty bash` is blocked**: *"Running shells is not permitted. Use containers
  instead."* Everything here is `sbatch`, not an interactive session — there is no
  interactive-debugging equivalent of `launch.sh`, only logs.
- **`killable`/`studentkillable`/`cpu-killable` are all preemptible.** A preemption
  kills whichever job it hits — if it's a vLLM job, the proxy job's wait loop will
  eventually time out (`FORESIGHT_WAIT_TIMEOUT`, default 1800s); if it's the proxy, the
  vLLM job(s) keep running uselessly until you notice and `scancel` them.
- Compute nodes **do** have outbound internet (verified from `s-002`) — this deployment
  still prefetches models from the login node anyway, because spending GPU allocation
  on a multi-GB download is wasteful and a preempted job restarts the transfer from
  scratch.

## 6. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `sbatch: error: Invalid account or account/partition combination` | wrong `--account` for that `--partition` — see §2, check `launch.sh --help`'s `sacctmgr` output |
| `foresight: cannot resolve a working root` | neither `--root`, `$FORESIGHT_ROOT` nor `$WORK` is set |
| `foresight: refusing to use FORESIGHT_ROOT=... -- it is under $HOME` | you pointed at the small home filesystem; point it at your working area instead |
| job stuck in `squeue` as `PD` | queue is busy, or the partition/account combo has no headroom right now — `.endpoint` files simply won't appear yet; this is not a hang |
| `*.endpoint` never appears, job is `R`unning | check `logs/vllm-*.err` — usually a bad `--max-model-len` for the GPU's memory, or the model failing to load |
| `curl ... 400` mid opencode-session | context grew past `--max-model-len`; raise it with `launch.sh --max-model-len N` |
| `foresight: no env at .../conda-envs/foresight-vllm` | run `install_vllm_env.sh` first |
| `foresight: opencode binary not found` (from `tools/opencode_probe.sh`) | run `install_opencode.sh` first |
| `df -h "$HF_HOME"` shows little headroom | check whether you're about to duplicate a large checkpoint someone else already has — see §4 |
| target vLLM log fills with `POST /v1/chat/completions" 400 Bad Request`, opencode never finishes | a real agent harness always sends a `tools` array; confirm `--enable-auto-tool-choice`/`--tool-call-parser` actually reached `vllm serve` (check the job's `.out` log) — see §2. If the flags are present and it still 400s, this is model tool-call-quality, not a flag — see §2 |
| `tools/opencode_probe.sh` process alive but 0 requests reaching either vLLM, high CPU for minutes | `--home` is on NFS-mounted storage; opencode's first-run plugin install (`npm install`, thousands of small files) is catastrophically slow there — use local disk, see the script's own header |
| `502` with `aux_failure`, proxy log shows the aux agent exited non-zero | the aux agent (opencode) failed. Read `logs/proxy-*.out`: the `AuxFailure` message carries opencode's stderr tail. Most often the model cannot drive tool calls — see §2 |
| `502` with `aux_failure`, "aux agent exceeded 1800s" | aux never converged. Usually the model retrying a malformed tool call; check the target vLLM log for a 400 storm |
| `500` with `workspace_contaminated` | aux wrote to `--workspace`. The run is correctly refused — the prompt says read-only, the model ignored it. The error names the changed paths; re-clone the workspace before retrying |
| trace rows show `aux.usable: false` | aux answered but produced fewer than 3 enumerated items. Not fatal by design — the run continues and these instances are excluded at analysis time. If it is most rows, the aux model is too weak |
| `--adapter local` fails but `--adapter generic` works | the fault is in the aux agent, not the topology: model tool-calling, the opencode home, or the workspace. That is exactly what the `generic` rung is for |

## About `tools/opencode_probe.sh`

**Half retired.** Its setup half — writing opencode's `auth.json` + `opencode.json` —
is no longer temporary: it moved to `tools/opencode_home.sh`, which the real aux path
uses too, so the probe now calls that instead of duplicating it and cannot drift from
what `LocalAdapter`'s harness actually sees.

What remains temporary is the **caller** half. The probe stands in for a *target*
harness (SWE-CI, mini, your own agent) driving foresight from the outside. Milestone 2
made the aux side real; the target side is still a benchmark this repo does not launch.
So the probe is fine for hand-debugging and for measuring the aux-hop latency against
opencode's own client timeout (design risk 7) — never part of a measured run.

**Always pass `--home` on local disk**, never under `$FORESIGHT_ROOT` — see the
troubleshooting table above and the script's own header for why. `foresight.sbatch`
applies the same rule to the aux agent's home, for the same reason.

**Tool-calling against a real model is a known open gap**, not this script's bug —
see §2's "Model tool-call quality" note. Protocol conformance and `HOME` isolation are
both confirmed working (this script was run successfully against both a mock upstream
and a real vLLM backend for plain, tool-free completions); a full opencode agent
session with real tool calls has not yet succeeded against the 1.5B bring-up model,
which is why the examples in §2 now name a 7B.
