# Server side setup (as reported by the server side)

Provided by whoever ran the model servers, 2026-09-20, after the caller side collected this run. Recorded
as reported: the caller side has not verified the job ids, the vLLM version or the partition details.
Still missing from this side: the sampling defaults the servers applied, the exact `sbatch` lines, and
the vLLM logs (see `TEMPLATE.md`).

## Server side (TAU Slurm cluster), A/B test setup

**Model:** `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` for both roles. Needed a fast NFS load: chosen over
`Qwen3-Coder-Next-FP8`, which took 4.5 h on a bad night; this one loads in about 10-35 min.

Two independent vLLM (v0.29.0) servers, same checkpoint, on the same node, `n-b200`, one B200 GPU each, no
tensor-parallel needed (31 GB weights against 143 GB per GPU). `--max-model-len 262144` (native context, no
rope scaling needed).

- **TARGET:** `http://n-b200:8001/v1`, served name `target-model`. Used directly by SWE-CI in the CONTROL
  arm (no foresight).
- **AUX:** `http://n-b200:8002/v1`, served name `aux-model`. Only used by foresight's aux-agent exploration
  in the TREATMENT arm; SWE-CI never talks to it directly.

Both: `--tool-call-parser qwen3_coder`, `api_key` unset (no auth), 12 h wall-clock jobs (Slurm jobs
911023 / 911146).

**Partition:** `gpu-b200` (non-killable, cs_dcor-owned, PriorityTier=20) rather than the shared
`gpu-b200-killable` / `killable` pools. Deliberate, so these two jobs cannot get preempted mid-experiment
(unlike a currently-running unrelated job, 909753 on `n-h200`, which has been preempted repeatedly).
Tradeoff: submitting here forcibly requeues other users' lower-tier jobs on `n-b200` to free GPUs;
accepted for run stability.

**Hardware note:** B200 / H200 are the only GPU types confirmed to run this vLLM build (CUDA-13 wheels);
a6000 OOMs on this checkpoint's KV cache at single-GPU, l40s and older cards have drivers too old for
CUDA 13, and both H100 nodes are individually broken or blocked.

foresight itself (the proxy that does aux-enrichment for the TREATMENT arm) is NOT running on the server:
it runs on the calling machine, pointed at both URLs above.
