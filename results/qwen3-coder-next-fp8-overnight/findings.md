# Qwen3-Coder-Next-FP8 overnight run — status

**Model confirmed serving end-to-end as of 15:17, and the local-caller
foresight test PASSED with the first fully-grounded aux result in this
project's history.** Job `909753` (`n-h200:8000`) left running
intentionally, per direct instruction, for further use -- not cancelled.

## Local-caller foresight test — PASSED, grounded

Ran foresight locally (`configs/vllm-local.yaml.tmpl` rendered by hand,
`LocalAdapter`, real `opencode` aux agent, no Docker/SWE-CI needed) against
the live H200 endpoint, workspace = `tinyrepo` (psf/cachecontrol). One
simulated target request ("Add a debug logging statement to the cache
lookup path in this repository", with a `tools` array so it's treated as
an agent turn):

- **`aux.source: "agent"`, `aux.usable: true`, `aux.items: 5`** -- 5 future
  tasks, all citing real files and line numbers actually read from the
  workspace: `adapter.py:67`, `controller.py:290`, `controller.py:275`,
  `controller.py:166`, `controller.py:204/269/293/393`.
- **`guard.workspace_intact: true`** -- no contamination.
- **`timings.aux_s: 101.4`** -- one real agent run, well within budget.
- **The trace shows real tool-call round-trips**, not a single ungrounded
  answer: aux `message_count` grew 2 -> 8 -> 12 -> 14 across successive
  calls, `tool_count: 10` throughout (opencode's own title-generation
  request, `message_count: 3, tool_count: 0`, was correctly skipped from
  contaminating this by the `is_agent_turn` gate).
- **The prompt was correctly enhanced**: `enhanced: true`, `prompt_out`
  contains the 5-item future-task list ahead of the original task, matching
  the `template` builder's design exactly.
- **The target model (same checkpoint) also handled tool calls correctly**:
  responded with a well-formed `bash` tool call
  (`find ... -name "*.py" | head -20`), `finish_reason: "tool_calls"`.

**This is the first aux result in this project to clear the grounding bar.**
`results/qwen2.5-coder-7b`, `results/qwen3-8b-ollama-local`, and
`results/qwen3-14b-ollama-local` all documented ungrounded or failed aux
runs from smaller models. `Qwen3-Coder-Next-FP8` is a genuine, measured win
as an aux model, not just as a target.

Next: regenerate this directory properly with `tools/collect_run.sh` for
the full standard shape (`config.yaml`, `aux-agent/` transcripts, etc.) per
`results/README.md`.

## What's running

- Job: Slurm job `909532`, serving `Qwen/Qwen3-Coder-Next-FP8` on
  `gpu-h200-killable` (`n-h200`, the only currently-viable H100/H200 hardware
  — both `gpu-h100-killable` nodes are down for unrelated reasons: `n-102`
  has a physically unhealthy GPU, `t-100`'s driver is too old for the
  CUDA-13 build).
- `--enforce-eager` (skips CUDA-graph capture — see PR
  `fix/b200-cuda-jit-packaging-bugs` for why: a prior H200 attempt on the
  smaller 30B checkpoint got to 88% through graph capture, genuinely still
  progressing, before a since-fixed hardcoded 60-minute readiness cap killed
  it), `--max-model-len 131072`.
- Run dir: `/home/dcor/arielzoizner/foresight-runs/overnight-qwen3-coder-next/`
- Both CUDA packaging bugs from the PR above are already fixed in the live
  `foresight-vllm` conda env, so this job doesn't need to rediscover them.

## Plan once it's serving

Per direct instruction: run **foresight with a local caller, not SWE-CI**,
from this same Slurm submit host (not the private laptop) — SWE-CI needs
`docker exec` into a live task container, and there is no Docker here or
anywhere on this cluster. The local-caller path (`LocalAdapter`,
`--adapter local` in `deploy/tau-slurm/launch.sh`'s terms) needs no
container: aux is a real `opencode` subprocess over a real repo checkout on
this host, which is exactly what M2 was built for.

Reusing job `909532`'s already-running vLLM directly (not `launch.sh`, which
would submit its own *second* copy of this ~80GB model) — construct a
`local`-adapter foresight config pointed at `909532`'s endpoint, run
`python -m foresight.server` here, and exercise it with a real request
against a scratch workspace checkout.

## How to check status (from a shell on the submit host, e.g. c-006)

```sh
squeue --me
cat /home/dcor/arielzoizner/foresight-runs/overnight-qwen3-coder-next/target.endpoint 2>&1
curl -sS "$(cat /home/dcor/arielzoizner/foresight-runs/overnight-qwen3-coder-next/target.endpoint)/models"
```

## For the morning, if this session doesn't get to the foresight test itself

Once `target.endpoint` contains e.g. `http://n-h200:8000/v1`, point
foresight at it **from your laptop** instead (over VPN, no SSH tunnel needed
— confirmed directly reachable earlier tonight for both a CPU and a GPU/H100
node from this exact VPN setup, though worth a quick `nc -zv <node-ip> 8000`
first):

```yaml
# configs/swe_ci.yaml (or a local-adapter config) on your laptop
models:
  target:
    base_url: "http://<node-ip>:8000/v1"
  aux:
    base_url: "http://<node-ip>:8000/v1"   # same server, single-role job backs both
```

Full steps for the Docker+SWE-CI-on-laptop path (once Docker is actually
wanted) in `deploy/local-swe-ci-slurm-models/README.md`.

## Live status log (appended overnight)

- 01:53 — job submitted (`909532`), queued for a `gpu-h200-killable` node.
- 02:02 — still `PENDING@(Resources)`.
- 02:22 — still `PENDING@(Resources)` — `n-h200` is the only node in this
  partition and evidently occupied by someone else's job; nothing to do but
  wait for it to free up.
- 02:23 — `RUNNING@n-h200`, GPU preflight OK.
- 02:25 — checkpoint prefetch started (74.86 GiB, 40 shards — bigger and in
  more files than anything else tonight, expect this phase alone to take a
  while).
- 02:45 — no prefetch percentage line yet (20 min in; every earlier run
  tonight had its first tick within 1.5-8 min). Checked live via `srun
  --overlap`: `VLLM::EngineCore` is in `D` state (active I/O wait) with
  growing RSS -- genuinely working, not hung. Plausibly just this
  checkpoint's shape (40 files vs. 4 for the 30B model means the first
  "shard complete" event legitimately takes longer even at a similar read
  rate). Watching for it to actually tick or to clearly stall past ~40 min.
- 02:53 — first tick: 10% (4/40) -- confirmed genuinely progressing, just
  slow (28.5 min for the first 10%). 20% (8/40) at 03:02, ~9 min later.
  Extrapolating, full prefetch may take on the order of 90-100 min total --
  slow but real, consistent with "budget for the slow case" in the main
  `deploy/tau-slurm/README.md`.
- 03:27 — 40% (16/40), pace holding steady.
- 03:51 — 60% (24/40), still steady (~8-17 min/10%).
- 04:04 — 70% (28/40).
- 04:18 — `909532` FAILED: **not a hang** -- shard loading (a separate phase
  after prefetch) was still genuinely climbing (62% of 40 shards, still
  increasing) when `FORESIGHT_READY_TIMEOUT_S=5400` (90 min, already raised
  once tonight from the old hardcoded 60 min) ran out. This checkpoint is
  having an exceptionally slow NFS night -- prefetch alone took ~95 min,
  and shard loading needed well over 100 min more without finishing.
  Resubmitted as `909656` with `FORESIGHT_READY_TIMEOUT_S=18000` (5h).
  `gpu-h200-killable` has exactly one node (`n-h200`), so the resubmit is
  guaranteed to land on the same physical node -- the page cache warmed by
  this failed attempt should still be resident (503GiB RAM on that node,
  nowhere near full) and make the retry much faster than a cold start.
- 04:19-05:39 — `909656` queued `PENDING@(Resources)` for over an hour --
  longer than any other queue wait tonight. `n-h200` is the only node in
  this partition, shown `mix` (shared with other tenants), so this is
  ordinary contention, not a bug -- nothing to do but wait.
- 05:51 — `RUNNING@n-h200`, GPU preflight OK. This attempt selects `TRITON`
  for its MoE backend (not `DeepGEMM`) -- likely `--enforce-eager` changes
  vLLM's backend auto-selection. Also flags a JIT-compiled kernel specific
  to this architecture's Gated DeltaNet linear-attention layers
  (`FlashInfer GDN prefill`), separate from anything fixed tonight --
  worth watching.
- 06:15 — first prefetch tick, 10% (4/40), in 22.5 min -- slightly faster
  than the first attempt's 28.5 min. Mild sign the page cache partially
  survived the ~1.5h queue wait, not dramatic.
- 06:36 — 30% (12/40), pace roughly comparable to the first attempt.
- 06:56 — 50% (20/40), still steady.
- 07:14 — 70% (28/40).
- 07:33 — prefetch 100% (40/40), took 6035s (~1h40m) total.
- 07:39 — `DeepGEMM warmup: 100%` completed cleanly, no crash -- further
  confirms both CUDA packaging fixes from the PR hold for this model's
  kernels too, not just the 30B's. Compile artifacts are landing under
  `/home/dcor/arielzoizner/.cache/vllm/...` (the redirected
  `VLLM_CACHE_ROOT`, not the tiny per-node `$HOME`) -- that fix is working
  too, and future runs against this checkpoint should reuse it. Waiting on
  the final readiness check now.
- 07:41 — `909656` FAILED right after the warmup: **new** instance of the
  same missing-unversioned-`.so` bug, this time `cannot find -lnvrtc`
  (flashinfer's JIT linker needing a *different* library than cudart).
  Checked the package directly: literally every `nvidia-cuXX` pip wheel
  ships only versioned `.so` files, none has an unversioned symlink -- this
  was never cudart-specific. Generalized the fix in `install_vllm_env.sh`
  and `serve_vllm.sbatch` to cover every library, applied it live, and
  resubmitted as `909697`.
- 08:08-08:20 — `909697` (`n-h200`, shared with another user's multi-day
  job) got stuck: my OWN new preflight check (`conda run -p ... python -c
  "import importlib.metadata..."`) sat at 0% CPU for 11+ minutes without
  returning -- env activation + interpreter start + a metadata scan of the
  whole site-packages tree, over NFS, apparently vulnerable to contention
  from the other heavy job sharing this node. Cancelled, replaced the check
  with a plain directory-name glob (no process spawn -- confirmed 0.017s
  locally, same information), resubmitted as `909753`.
- 09:16 — `909753` `RUNNING`, all preflight checks passed in seconds
  (confirms the glob-based fix). Preempted again 1 minute later, before
  prefetch even started -- pure bad luck on this preemptible partition,
  unrelated to any of tonight's fixes. Auto-requeued by Slurm to `PENDING`.
- 09:36 — `909753` preempted again, this time after only 41s. `squeue -p
  gpu-h200-killable` shows heavy contention -- ~9 other users' jobs queued
  for this same single-node partition tonight, various priority reasons.
  Genuine fairshare contention, nothing to fix on our end.
- 09:40 — rechecked H100: `n-102` now has another user's job successfully
  `RUNNING` on it. Tried `909834` there in parallel, hoping this meant the
  known-unhealthy-GPU issue was index-specific and we might get lucky. It
  wasn't: our own preflight (correctly) failed in ~5s, same as before --
  vLLM enumerates ALL 8 physical GPUs on the node at import time regardless
  of which one is actually allocated to a job, so n-102 will fail for any
  of *our* jobs unconditionally until the hardware is actually fixed. The
  other user's success there must be a different, non-vLLM workload. Not
  worth retrying n-102 again; back to waiting on `909753`/h200.
- 09:55 — `909753` `RUNNING` again. Preempted a third time at 10:05, ~10
  min in, before the first prefetch tick. Contention on this single-node
  partition is getting worse over the course of the night, not better --
  now the dominant obstacle to a serving model, not any remaining software
  bug. Nothing actionable except keep letting Slurm auto-requeue.
- 10:26 — `909753` `RUNNING` again, and stable this time -- no preemption
  for 55+ min and counting, the longest stretch tonight. Prefetch tick 1
  (10%) at 36.5 min (slowest yet, but genuine), tick 2 (20%) 12 min later.
- 12:50 — still `RUNNING@n-h200`, no preemption for 2h35m and counting --
  by far the most stable stretch tonight. Prefetch at 60% (24/40). Side
  note: `t-100` cu126 probe (`910001`) is still queued -- not down, just
  fully occupied (all 8 GPUs allocated to another job), nothing to do but
  wait for that separately.

## Side investigation: can we also unlock `t-100` (H100) and l40s via an older CUDA build?

`creativity-measure` (a different project on this account) successfully runs
on l40s using `torch==2.4.1+cu121` -- direct proof that pinning an older CUDA
build is a real, working strategy for old-driver nodes in general, not
something to dismiss outright.

- **`t-100`** (driver reports max CUDA 12.7): `torch==2.13.0+cu126` exists
  on PyTorch's own index -- *same* torch/vllm version as the main env, so
  `Qwen3NextForCausalLM` support should be unaffected, just an older CUDA
  build. Built an isolated `foresight-vllm-cu126` env to test this (kept
  fully separate from the main env -- B200 needs CUDA >=12.8 for `sm_100`,
  so downgrading the main env would break what's already proven working).
  **Caveat found while verifying, not yet resolved**: `vllm==0.29.0` pulls
  `nvidia-cuda-nvcc` (unsuffixed package name) directly, and that name only
  exists as a CUDA-13.x package on PyPI at all -- no 12.x version published
  under it (the CUDA-12.x line uses differently-suffixed package names
  instead, e.g. `nvidia-cuda-nvcc-cu12`, confirmed in `creativity-measure`'s
  own working env). So even with torch forced to cu126, `pip install
  vllm==0.29.0 --extra-index-url .../cu126` still resolved
  `nvidia-cuda-nvcc==13.4.92` from default PyPI -- meaning any JIT-compiled
  kernel (which `qwen3_next` needs) would likely still be built with CUDA-13
  tooling regardless, probably still requiring a CUDA-13 driver and
  defeating the purpose. **Testing this directly** (basic `torch.cuda`
  init first, cheap) via `probe-t100-cu126` -- queued, `t-100` is currently
  unavailable to Slurm (occupied by someone/something else).
- **l40s** (driver reports max CUDA 12.2, confirmed directly via a probe
  job): below cu126's requirement, so the `t-100` fix doesn't extend here.
  `creativity-measure`'s working `cu121` build has no equivalent for
  `torch==2.13.0` at all (index only offers cu126/cu129/cu130/cu132) --
  fixing l40s for real would need an old enough `vllm` release that it
  almost certainly predates `qwen3_next` support entirely. Not worth
  pursuing for tonight's actual goal; l40s remains viable only as a
  fallback for smaller/older checkpoints.
- **`n-102`**: unaffected by any of this -- confirmed a genuine hardware
  fault (one physical GPU un-enumerable via NVML), not a driver/CUDA
  version question. No dependency change fixes it.
- **`t-100` conclusion, tested directly once it freed up**: basic
  `torch.cuda` init with the `foresight-vllm-cu126` env passes cleanly
  (`cuda available: True`, `torch.version.cuda: 12.6`, correctly sees the
  H100). But a real `vllm serve` attempt (`Qwen3-Coder-30B-A3B-Instruct-FP8`,
  already-cached checkpoint) fails at plain **import time**, before any
  model loading or JIT compilation: `ImportError: libcudart.so.13: cannot
  open shared object file`. `vllm==0.29.0`'s own precompiled C extension
  (`vllm._C_stable_libtorch`) is hard-linked against CUDA 13's runtime
  library specifically, independent of which `torch` CUDA variant is
  installed alongside it. **`t-100` is a dead end for this vLLM version,
  full stop** -- not just the JIT-kernel path originally suspected. Fixing
  it for real would need an old enough `vllm` release to predate this
  baseline, which almost certainly means predating `qwen3_next` support too
  (same tension as l40s). Not pursuing further.
- 16:14 — **post-commit update**: `909753` was preempted after ~1h of
  serving (had already produced the grounded local-caller result above
  before this happened). Auto-requeued by Slurm, `PENDING@BeginTime`.
  `gpu-h200-killable` remains preemptible regardless of how long a job has
  already been up -- this is expected, not a regression of anything.
