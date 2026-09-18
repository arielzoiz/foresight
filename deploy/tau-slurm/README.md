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

### Fitting a model that does not fit one GPU

Weights are roughly **2 bytes per parameter at bf16, 1 byte at FP8**, plus room for
KV cache and activations. So a 30B needs ~61 GB at bf16 and ~31 GB at FP8, against
48 GB on an a6000 and 80/141 GB on an H100/H200.

**`--gpus N` alone is not enough — it never was.** vLLM defaults
`--tensor-parallel-size` to 1, so reserving four GPUs and saying nothing else gets
you four GPUs with the model crammed onto *one*, OOMing while three sit idle.
`launch.sh` now defaults `--tensor-parallel` to whatever `--gpus` is, which is what
you almost always want; set it explicitly only to shard across fewer cards than you
reserved. It must divide the model's attention-head count, so prefer powers of two.

That gives three routes to a 30B on this cluster, and the scarcest hardware is not
automatically the right answer:

| Route | Fits? | Catch |
|---|---|---|
| FP8 on one H100/H200 | 31 GB of 80/141 GB, easily | `gpu-h200-killable` queues behind the non-killable `gpu-h200` partition, whose jobs run for **days** |
| bf16 on 2× a6000, `--gpus 2` | 61 GB of 96 GB | needs tensor parallelism; `killable` usually has capacity immediately |
| FP8 on one a6000 | 31 GB of 48 GB on paper | **unverified.** FP8 is nominally allowed from compute capability 7.5, but this checkpoint is *block*-quantized (128×128) and vLLM's Marlin path has no `weight_block_size` handling. If vLLM dequantizes to bf16 instead, it needs 61 GB and OOMs |

**Prefer the middle row when the big partitions are busy.** bf16 on two a6000s
sidesteps the FP8-on-Ampere question entirely — bf16 is fully supported on Ampere —
at the cost of downloading twice the weights and reading twice as much off NFS. A
queue you can enter now beats an ideal GPU you cannot.

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

**A node can be healthy to Slurm and broken in fact — `serve_vllm.sbatch` now checks.**
Observed on `n-102` (`gpu-h100-killable`): `sinfo` reported `mix` with no drain
reason, the cgroup exposed all eight `/dev/nvidia*`, and yet bare `nvidia-smi -L`
could not enumerate two of the eight GPUs —

```
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-286e2580-...)
Unable to determine the device handle for gpu 0000:18:00.0: Unknown Error
GPU 2: ...                        <- GPUs 1 and 7 never appear
```

— and Slurm allocated the job `CUDA_VISIBLE_DEVICES=7`, one of the dead ones. vLLM
then aborted inside `import vllm` with `NVMLError_Unknown`, because
`CudaPlatform.log_warnings()` enumerates **every physical device on the node** at
import time. So one sick GPU anywhere on the box kills the job even when the GPU you
were given is fine, and the traceback points at `pynvml` and reads like a
vLLM-or-driver version problem. It is neither.

The preflight runs `nvidia-smi -L` before `vllm serve` and aborts in about five
seconds with the node name if any device fails to enumerate, instead of discovering
it after a queue wait plus a 30 GB checkpoint load. On hitting it, resubmit — you
will usually land elsewhere — or add that node to `--exclude`. Worth reporting to
the cluster admins too: Slurm cannot see this fault, so it will keep scheduling work
onto it until someone says so.

**Tool calling is enabled by default, with no flag needed.** `serve_vllm.sbatch`
always passes `--enable-auto-tool-choice --tool-call-parser <name>`. Without this,
vLLM 400s on *any* request carrying a `tools` array — which every message from a real
agent harness does, since it always advertises its own tools. Confirmed directly:
pointed at a server started without these flags, an opencode session retried the same
400 forever — 1215 requests in 45 minutes, never making progress, silently burning
the GPU allocation the whole time. Disable it with `--tool-call-parser ""` only for
`--adapter generic`, whose aux call carries no tools at all; any real target or aux
**agent** will reproduce the retry storm.

**The parser must match the checkpoint, and a mismatch is silent — so it is
detected, not defaulted.** `--tool-call-parser` defaults to `auto`, and
`serve_vllm.sbatch` reads the answer off the checkpoint's own chat template
(`chat_template.jinja`, or `tokenizer_config.json` for checkpoints like
Qwen2.5-Coder-7B that carry the template inline). The chosen parser is echoed into
`logs/vllm-*.err`. Pass a name to force one; pass `""` to disable tool calling.

This is detected rather than defaulted because guessing by model name does not
work. Both families wrap tool calls in the same `<tool_call>` tag, so the mistake
is invisible from the outside — the body inside differs:

| Checkpoint family | `--tool-call-parser` | What the chat template emits |
|---|---|---|
| `Qwen2.5-*-Instruct` | `hermes` | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` (JSON) |
| `Qwen3-Coder-*` | `qwen3_coder` | `<tool_call><function=NAME><parameter=X>v</parameter></function></tool_call>` (XML) |

Point `hermes` at Qwen3-Coder and it matches the opening tag, fails to JSON-parse the
XML body, and returns **HTTP 200 with `tool_calls: null`** — no error anywhere. The
harness then reads the raw text as the assistant's final answer and the session ends
after one turn, which is *pixel-identical to the Qwen2.5-Coder-7B failure documented
below* despite having a completely different cause. Do not diagnose one as the other.

Detection tests `<function=` **before** `<tool_call>`, which is the whole trick:
Qwen3-Coder's template contains both, since it wraps XML in the tag Qwen2.5 wraps
JSON in. Checking the other order would classify every Qwen3-Coder as `hermes`.
Confirm what was chosen with:

```sh
grep "tool-call parser" $RUN/logs/vllm-*.err
```

Some repos (Qwen3-Coder's among them) also ship their own
`qwen3coder_tool_parser.py` next to the weights, which is the same hint by another
route. vLLM ships dozens of parsers — `vllm/tool_parsers/__init__.py` in the installed
package lists the registered names (this env has `qwen3_coder` and `qwen3_xml`
alongside `hermes`). The parser actually used is echoed near the top of
`logs/vllm-target-*.err`, so an ungrounded run can be checked against it in one grep.

**Don't guess a parser on a GPU — replay the output offline.** A tool parser is a
pure function from the model's output text to a list of tool calls: no GPU, no
weights, just a tokenizer. So every candidate can be tried in seconds against the
exact text the model produced, instead of paying a queue wait plus a checkpoint load
per guess:

```sh
python tools/probe_tool_parsers.py --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct
python tools/probe_tool_parsers.py --tokenizer <id> --text-file captured.txt
```

It sweeps every registered parser and reports which recognised the text. Measured
results on this cluster, which is where the table above comes from:

| Model output | Recognised by |
|---|---|
| `<tool_call>{"name":…,"arguments":…}</tool_call>` | `hermes` |
| `<tool_call><function=…><parameter=…>…` | `qwen3_coder`, `qwen3_xml` |
| Qwen3-Coder's XML fed to **`hermes`** | **nothing** — `JSONDecodeError`, swallowed |
| ```` ```json {"name":…,"arguments":…} ``` ```` (single object) | **nothing** |
| ```` ```json [{"name":…,"arguments":…}] ``` ```` (array) | `xlam` |

Two things worth keeping from that table. The Qwen3-Coder-under-hermes row is the
silent failure spelled out above, reproduced deliberately — it logs a server-side
`ERROR` and still returns 200 with `tool_calls: null`. And the last two rows are the
Qwen2.5-Coder-7B case: what it actually emitted parses under **no** parser, and the
reason is not the markdown fence — `xlam` strips fences happily — but that it emits a
single object where `xlam` requires an array. `xlam` is therefore *not* a safe
drop-in: it rejects the correct single-object `hermes` form, so switching to it would
lose the turns where the model gets the format right.

**The control sample is the point.** The sweep includes one input that must parse
(`hermes` tags under `hermes`). The first run of this script reported `NONE` for
every input including that one — because parsers register lazily and the registry
dict is empty until each module is imported, so it had silently tested nothing. A
sweep with no positive control cannot tell "no parser matches" from "no parser ran".

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

### Two stacked CUDA packaging bugs break JIT-compiled kernels (B200, and possibly other Hopper/Blackwell cards)

**Symptom:** `vllm serve` gets all the way through loading a checkpoint's
weights into GPU memory — a real full-sized load, not a quick failure — then
crashes at engine-core init. It looks like a driver or hardware problem, and
reads a lot like the l40s/rack-bgw-dgx1 driver issues elsewhere in this doc,
but is neither: both bugs below are pip packaging gaps, present regardless of
which node or driver you land on, and both are invisible until a
JIT-compiled kernel (`DeepGEMM`, or `flashinfer`'s `fused_moe_trtllm_*`
fallback) actually gets built — which only happens after a full checkpoint
load, costing 10-20+ minutes per attempt to even see the error.

#### Bug 1 — `nvcc` and `nvidia-cuda-runtime` land on different CUDA point releases

```
RuntimeError: Assertion error (deepgemm-src/.../compiler.hpp:234): false and "NVCC compilation failed"
```

or, one layer down, if `DeepGEMM` is disabled:

```
/…/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:8:
error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

**Root cause.** `pip install vllm` resolves `nvidia-cuda-nvcc` (the compiler)
and `nvidia-cuda-runtime` (its headers) as independent wheels — not a matched
bundle — so they can land on different CUDA point releases within the same
13.x line. Measured directly: `nvidia-cuda-nvcc==13.4.59` against
`nvidia-cuda-runtime==13.0.96`. `cccl`'s `cuda_toolkit.h` asserts these are
equal at major.minor granularity before letting any JIT-compiled kernel build:

```c
#if !_CCCL_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#  error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
#endif
```

`CUDART_VERSION` comes from `nvidia-cuda-runtime`'s `cuda_runtime_api.h`; the
compiler's own version is read directly from `nvcc`. The check is a pure
host-side preprocessor comparison — it does not depend on which GPU
architecture is being targeted, only on whether these two packages agree.

**Confirmed by direct compile-only reproduction**, no GPU needed: a minimal
`.cu` file including `cuda_toolkit.h`, compiled with the mismatched
`nvidia-cuda-runtime==13.0.96` headers, reproduces the exact error; the same
file compiled against `nvidia-cuda-runtime==13.4.92` (same major.minor as the
installed `nvcc`) compiles clean for `-arch=sm_100`. `nvcc`'s own
`--list-gpu-arch` already lists `compute_100` — the compiler fully supports
Blackwell; only the header/runtime version skew was blocking it.

#### Bug 2 — pip's CUDA wheels ship no `lib64/` and no unversioned `.so`, masked by bug 1 until it's fixed

Fixing bug 1 alone gets further, then hits a second, unrelated failure at the
link step:

```
/usr/bin/ld: cannot find -lcudart: No such file or directory
collect2: error: ld returned 1 exit status
```

**Root cause.** Every `nvidia-cuXX` pip wheel (`cu13`'s `cudart`, plus
`cudnn`, `nccl`, `cusparselt`, ...) ships its libraries under a plain `lib/`,
never the `lib64/` a traditional system CUDA toolkit install uses, and ships
only the versioned `.so` (`libcudart.so.13`) — no unversioned `libcudart.so`
for a linker's `-lcudart` flag to resolve against. `flashinfer`'s JIT linker
step is written for the traditional layout (`-L .../lib64 ... -lcudart`) and
fails outright against the pip layout. This is pure packaging, present in
every `nvidia-cuXX` wheel — **not specific to B200, not specific to this
checkpoint, and not new** in the sense of something this session introduced;
it was always going to fail here, for any JIT-compiled kernel on any GPU.
Bug 1 simply failed *first*, every time, which is the only reason bug 2 had
never been seen before tonight.

`-lcuda` (the driver stub, separate from `-lcudart`) is unaffected — it
resolves fine via the node's real NVIDIA driver install
(`/usr/lib/x86_64-linux-gnu/libcuda.so`, already on the default system
linker path via `ldconfig`), which pip has nothing to do with.

#### Fix for both

`install_vllm_env.sh` now, immediately after installing vllm:
1. pins `nvidia-cuda-runtime` to match `nvidia-cuda-nvcc`'s major.minor line
   (fixes bug 1), and
2. creates `lib64 -> lib` and `libcudart.so -> libcudart.so.<N>` compatibility
   symlinks inside the resolved `nvidia/cuXX` package directory (fixes bug 2).

`serve_vllm.sbatch` also checks both at job start, next to the existing
GPU-health preflight, and aborts in seconds with the exact fix command rather
than discovering either after a full checkpoint load — the same "fail fast,
name the cause" philosophy as the unhealthy-GPU preflight above.

**Why neither fix touches the driver floor.** `nvidia-cuda-runtime` also
ships the actual `libcudart.so` loaded at runtime by every CUDA call, not
just JIT-compiled ones — a version bump here could in principle raise the
minimum driver version needed on every node, not just the ones hitting the
JIT-kernel bug, which was the real remaining question before trusting bug 1's
fix broadly. It doesn't: the driver floor is set by `torch`'s own build tag
(`torch==2.13.0+cu130`), confirmed directly — `t-100`'s *"NVIDIA driver on
your system is too old (found version 12070)"* failure happened with the
**old**, unpatched `nvidia-cuda-runtime==13.0.96` already installed, so that
check was never gated by this package's exact minor version. The fix stays
within the same CUDA 13.x family `torch` already required from the start.
The symlinks fixing bug 2 don't touch versions at all, only paths/naming.

**Confirmed end to end**, 2026-09-19, on a B200 node (`n-b200`,
`gpu-b200`): checkpoint loaded, `DeepGEMM` compiled and linked successfully,
server reached `Application startup complete`, and a real
`/v1/chat/completions` request returned a real completion
(`Qwen3-Coder-30B-A3B-Instruct-FP8`, `finish_reason: stop`).

**What this means per GPU option:**

| Partition / GPU | Hits these bugs? | Why | Status after the fix |
|---|---|---|---|
| `killable` (a5000/a6000/rtx_3090/rtx_2080/v100/quadro_rtx_8000) | No | vLLM auto-selects the precompiled `MARLIN` FP8 kernel on Ampere-class cards, which needs no JIT compile at all | Unaffected either way. a6000 (48GB) still cannot fit this 30B FP8 checkpoint's weights + KV cache — that OOM is a real VRAM limit, unrelated to these bugs |
| `gpu-h100-killable` (H100) | Unconfirmed — never reached this code path | Both attempts tonight failed earlier for unrelated reasons: `n-102` has a physically unhealthy GPU (caught by the existing preflight), `t-100`'s driver is too old for the CUDA-13 build regardless of these fixes | Untested; the fix should apply if H100 selects `DeepGEMM`/`flashinfer`'s fused-MoE path the way B200 does — both bugs are architecture-independent packaging issues, not Blackwell-specific |
| `gpu-h200-killable` (H200) | Unconfirmed as of this writing | Prior attempts were preempted before reaching engine init (`killable` partitions are preemptible, unrelated to these bugs); a retry (`909240`) was still loading when this was written | Should benefit the same way as B200 if it hits the same kernel-selection path; check `results/` or re-run for a confirmed answer |
| `gpu-b200` / `gpu-b200-killable` (B200) | **Yes, both — confirmed** | Blackwell (`sm_100`); vLLM selects `DeepGEMM` (and falls back to `flashinfer`'s `fused_moe_trtllm_sm100`, which JIT-compiles too) for its FP8 MoE kernels here | **Fixed and confirmed end-to-end**, including a real completion — see above |
| `l40s`, `rack-bgw-dgx1`/`rack-gww-dgx1`/`rack-omerl-g01` | N/A | Pre-existing, unrelated driver/toolchain issues (driver too old for any CUDA-13 build; excluded by default) | Unaffected by these fixes either way |

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

### Reading the aux agent's own transcript

**The row count gives the verdict; opencode's transcript gives the diagnosis.**
Every run with `--adapter local` now writes the aux agent's own session export to:

```
$RUN/aux-agent/
    ses_<id>.json     one per aux session -- opencode's full transcript
    opencode.log      the harness's own log
```

This is not new logging — opencode always recorded it. Its session DB holds one
`step-start`/`step-finish` pair per model call and a `type: "tool"` part per
invocation, carrying the tool name, its input and its output; `opencode export`
renders that as JSON. What was missing was *retrieval*: the transcripts never reached
the run directory, so they were neither attributable to a particular run nor readable
by anyone without cluster access. `foresight.sbatch` now exports them into
`$RUN/aux-agent/` **when the job ends**.

The session DB itself does **not** live under `/tmp/foresight-aux-home-<jobid>` — only
opencode's two config files do. The DB is on shared storage and outlives the job, so
the risk was never losing it; the risk is that every run's sessions pile into the same
one (see below).

**`OPENCODE_HOME` does not isolate opencode's session database — measured, not
assumed.** `OPENCODE_HOME` is this project's own wrapper convention, not something
opencode reads directly: `setup/install_opencode.sh` generates a wrapper script that
does `export HOME="${OPENCODE_HOME:-$BASE/home}"` before exec'ing the real binary,
precisely because there is no container here to isolate homes the way
`docker exec -e HOME=...` does. So `HOME` genuinely *was* set per job -- this is not
"the wrong variable was set" -- and it still does not isolate the session database.
Sessions from every run accumulate in one shared DB regardless of the per-job home,
and `opencode session list` returns all of them: a run whose workspace was `tinyrepo`
exported two sessions belonging to `tinyrepo-7b`, from a different job on a different
node. The collector therefore keeps only sessions whose own `directory` field matches
this run's `--workspace`, since `LocalAdapter` always spawns the agent with `cwd` =
the workspace.

Two consequences worth knowing. Give every concurrent run its **own workspace clone**
— it is what makes the filter able to tell runs apart, on top of keeping the guard
from tripping on a neighbour's writes. And the design's assumption that a separate
`HOME` isolates opencode's state (`foresight-design-plan.md`, `SweCiAdapter`
specifics) does **not** hold here on the host; anything relying on it, such as
SWE-CI's own token accounting via `docker exec -e HOME=...` inside a container, needs
verifying with a container runtime rather than assumed from this host-side result --
see `README.md`, "For the M3 / SWE-CI PR", item 4.

**Deliberately on exit only, not on a timer.** Exporting runs `opencode` against the
same `OPENCODE_HOME` the aux agent uses, and two opencode processes sharing one home
deadlock its SQLite (`database is locked`) — which surfaces as a `502 aux_failure`
and kills the caller's request. A periodic export would risk breaking the very runs
it documents. To look at a run in progress, attach to the job instead:

```sh
srun --overlap --jobid=<proxy-jobid> --ntasks=1 \
  ls /tmp/foresight-aux-home-<jobid>/.local/share/opencode/
```

Read it with:

```sh
python tools/aux_transcript.py "$RUN/aux-agent/ses_*.json"
python tools/aux_transcript.py --full "$RUN/aux-agent/ses_*.json"   # no truncation
```

It prints each turn and ends with the verdict:

```
SUMMARY  model-calls(steps)=4  tool-calls=3
         tools used: glob, read, read
         GROUNDED -- the agent ran tools against the workspace.
```

`model-calls(steps)` should equal that session's `role: "aux"` row count in
`trace.jsonl` — two independent records of the same thing, so a mismatch means one
of them is lying and is worth chasing.

**This is what separates the two ungrounded failure modes** that the row count alone
cannot. A `tool` part present means opencode received a parsed tool call and ran it;
no `tool` part at all, with a JSON or XML blob sitting in the assistant text, means
the model emitted a call that the server's `--tool-call-parser` never recognised.
The first is a model-capability problem, the second is a configuration problem —
see the parser table in §2, which is the first thing to check.

A proxy-side dump of the raw response bytes was considered instead and is strictly
worse: it would show what the model *said* without showing how opencode *interpreted*
it, and the interpretation is the half that failed on the 7B.

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
`/tmp`. It is not cleaned up on exit; `killable` nodes reclaim `/tmp` on their own.

It holds opencode's `auth.json` and `opencode.json` and **not** its session DB —
that turns out to live on shared storage, outside the home, whatever
`OPENCODE_HOME` says. For a failed aux run, read `$RUN/aux-agent/` (exported at job
exit) rather than going looking on the node.

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

**But check the prefetch percentage first — it is the only real progress signal,
and it is in `.out`, not `.err`.** Before the shard loader runs at all, vLLM
pulls the whole checkpoint into page cache and reports that separately:

```sh
grep -E "Prefetching checkpoint files|Filesystem type" $RUN/logs/vllm-target-*.out
```

```
Filesystem type for checkpoints: NFS. Checkpoint size: 56.87 GiB. Available RAM: 981.19 GiB.
18:41:40  Prefetching checkpoint files into page cache started (num_threads=8, ...)
20:48:54  Prefetching checkpoint files: 10% (1/8)
```

**`Loading safetensors checkpoint shards: 0/N` in `.err` sits at zero for this
entire phase, by design.** A frozen shard counter is therefore *not* evidence of
a hang, and neither is a worker in `D` state with growing RSS — both are exactly
what a healthy-but-slow prefetch looks like. Only the percentage distinguishes
"slow" from "stuck". This was learned the hard way: a 30B run was called wedged
on those two signals while it was in fact reading at 0.76 MiB/s, and the correct
diagnosis arrived two hours late (`results/qwen3-coder-30b-bf16-FAILED/`).

Do the arithmetic once the first percentage lands — it tells you whether to wait
or resubmit. 10% in 2 h means ~21 h total: kill it.

**Tensor parallelism multiplies this cost.** Each `Worker_TP<n>` prefetches the
*entire* checkpoint with its own 8 threads, so `--gpus 2` means twice the bytes
and twice the concurrent streams against the same NFS volume. Measured on the
same evening, same partition, same hardware: a single-worker 7B sustained
~10 MiB/s, while a two-worker 30B managed ~0.76 MiB/s. That is not a controlled
comparison — different checkpoints, different nodes — but it is reason enough to
**prefer a quantized checkpoint that fits one GPU over a bf16 one that needs
two**, when the choice exists.

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

## 3a. Recording a finished run

Run directories live on shared storage, outside git, and are eventually cleaned
up. To keep a result — and let someone without cluster access review it:

```sh
tools/collect_run.sh $RUN <label>          # e.g. qwen3-coder-30b-fp8
$EDITOR results/<label>/findings.md        # the one part no script can write
git add results/<label> && git commit
```

That is the whole workflow. `collect_run.sh` copies the small, durable part of
the run into `results/<label>/` — config, trace, aux transcripts, the prompts,
the Slurm resource request, and the vLLM startup lines — then prints these three
steps back at you. It is idempotent, so re-run it if a run is still producing
output, and it works on runs that finished long ago (job details come from
`sacct`, which outlives `squeue`).

Missing pieces are warnings, not errors: a run that died before writing a config
is exactly the run whose evidence is worth keeping. See `results/README.md` for
what each file is, and `tools/collect_run.sh --help`.

**Write `findings.md` even for a failure** — especially for a failure. State what
the run showed *and what it does not show*, so nobody later reads a broken run as
evidence about a model. `results/qwen3-coder-30b-bf16-FAILED/` is the worked
example.

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
