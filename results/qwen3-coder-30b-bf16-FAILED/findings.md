# Qwen3-Coder-30B-A3B bf16 — FAILED to load, no result

**Run:** `arielzoizner-20260914-181041`, 2026-09-14
**Attempted:** `Qwen/Qwen3-Coder-30B-A3B-Instruct` (bf16, ~61 GB, 16 shards) on
2× a6000 (n-602), `--tensor-parallel-size 2`, `--tool-call-parser qwen3_coder`
**Outcome:** the model never loaded. **Zero of sixteen shards** completed in
2 h 52 m, the server never bound its port, no request was ever served.

**There is no aux result here, and nothing about the 30B's tool-calling ability
can be concluded from this run.** It is recorded because the failure mode is
worth not rediscovering.

## What happened

Everything up to the weight load was correct, and `vllm.txt` shows it:

```
GPU preflight OK on n-602 (CUDA_VISIBLE_DEVICES=4,5)
tool-call parser: qwen3_coder
tensor-parallel-size: 2
```

Both GPUs were granted and healthy, tensor parallelism was active, the right
parser was selected. Then:

- **18:10** job starts.
- **18:10 – 18:41** vLLM startup and TP worker init. 31 minutes with nothing on
  the GPU — normal for TP=2, which spawns workers before touching weights.
- **18:41** both workers begin prefetching the checkpoint into page cache.
  `Loading safetensors checkpoint shards: 0/16` appears and stays there — the
  shard loader does not run until prefetch finishes.
- **20:48** `Worker_TP1` reports prefetch at **10%**, 2 h 07 m in.
- **21:03** `serve_vllm.sbatch`'s readiness poll gives up: *"vllm did not become
  ready in time"*. The proxy job had already failed at its own 7200 s wait.

## It was not wedged — it was reading at 0.76 MiB/s

**An earlier version of this file called n-602 a wedged node. That was wrong**,
and the mistake is worth keeping because it was an artifact of watching the
wrong file.

vLLM prefetches the whole checkpoint into page cache **before** the shard loader
starts, and reports that separately:

```
Filesystem type for checkpoints: NFS. Checkpoint size: 56.87 GiB. Available RAM: 981.19 GiB.
Prefetching checkpoint files into page cache started (in background, num_threads=8, ...)
18:41:40  Worker_TP0/TP1  prefetch started
20:48:54  Worker_TP1      Prefetching checkpoint files: 10% (1/8)
```

So `Loading safetensors checkpoint shards: 0/16` was **correct and expected** the
whole time: the shard loader had not begun. The job was making progress, just
catastrophically slowly. Against the sister node the same evening:

| | checkpoint | prefetch progress | effective rate |
|---|---|---|---|
| 7B on n-601 | 14.19 GiB | 10% at 6 m, 20% at 21 m, 30% at 23 m | **~10 MiB/s** |
| 30B on n-602 | 56.87 GiB | **10% at 2 h 07 m** | **~0.76 MiB/s** |

Roughly **13× slower per byte**. Extrapolated, the full prefetch needed about
**21 hours** — so no realistic `--time` would have saved it.

**Tensor parallelism is a plausible aggravating factor, not a bystander.** Both
`Worker_TP0` and `Worker_TP1` start their own prefetch of the *entire* 56.87 GiB
with `num_threads=8` each. That is ~113 GiB of reads and 16 concurrent streams
against one NFS volume, where the 7B ran a single worker. TP was added so the
weights would fit across two 48 GB cards; it appears to have doubled the I/O
that then sank the run. Unverified — one observation — but it should be measured
before the next multi-GPU attempt.

## What to do differently

1. **Watch `.out`, not just `.err`.** The prefetch percentage — the only real
   progress signal during this phase — goes to stdout. The monitoring for this
   run watched `logs/vllm-target-*.err`, which carries the shard counter, and
   that counter is pinned at `0/N` by design until prefetch completes. Watching
   the wrong stream turned "slow but progressing" into "apparently stuck", and
   led to a two-hour-late diagnosis.
2. **Liveness is not progress, but neither is a stalled shard counter proof of a
   stall.** A `D`-state worker with growing RSS says the process is blocked on
   I/O, which is equally consistent with healthy-but-slow. The prefetch
   percentage distinguishes them; nothing else here does.
3. **`serve_vllm.sbatch`'s readiness cap is a fixed clock.** It polls a fixed
   720 iterations, unlike `foresight.sbatch`, which was deliberately changed to
   poll the producer job's Slurm state after a fixed timeout killed a healthy
   run. Here it terminated a job that could not have finished anyway, so it did
   no harm — but it is the same shape of bug.
4. **FP8 is the better route, and by more than half.**
   `Qwen3-Coder-30B-A3B-Instruct-FP8` is ~31 GB in 4 shards rather than ~57 GiB
   in 16, is already downloaded, and fits one H100/H200 — so it needs **no
   tensor parallelism**, and therefore only one prefetcher rather than two. Both
   the halved volume and the halved concurrency work in its favour.

## Reproducing (and the retry to prefer)

```sh
# what failed
deploy/tau-slurm/launch.sh --single Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --workspace $WORK/workspaces/tinyrepo-30b \
    --partition killable --account gpu-research --foresight-account gpu-research \
    --tool-call-parser qwen3_coder --gpus 2 --constraint a6000 \
    --mem 128000 --time 360

# prefer, if the H200/H100 queue has cleared -- 4 shards, one GPU, no TP
deploy/tau-slurm/launch.sh --single Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --workspace $WORK/workspaces/tinyrepo-30b \
    --partition gpu-h200-killable --account gpu-research \
    --foresight-account gpu-research \
    --tool-call-parser qwen3_coder --mem 128000 --time 300
```

Add `--exclude n-602` to the bf16 form to avoid the node this wedged on.
