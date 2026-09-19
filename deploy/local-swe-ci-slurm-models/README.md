# Running SWE-CI + foresight locally, models on Slurm

**The split:** Docker, foresight, and SWE-CI run on *your own machine* (over the
TAU VPN). Only the target/aux models run as Slurm GPU jobs. This is the answer
to the open question in the main `README.md` ("run foresight locally with
Docker while serving the model(s) from Slurm as jobs") — confirmed working
2026-09-18: a VPN-connected client reaches a Slurm compute node's IP:port
**directly**, no SSH tunnel needed, on both a CPU (`cpu-killable`) and a GPU
(`gpu-h100-killable`) node.

## Why foresight runs locally, not as a Slurm CPU job — this isn't a preference

`configs/swe_ci.yaml` requires `adapter.foresight_base_url` to be the Docker
bridge gateway (`http://172.17.0.1:8000/v1`) — the address a SWE-CI *container*
uses to reach foresight for its own aux-model calls. That address is only
meaningful on the machine actually running the Docker daemon those containers
sit on. If foresight ran elsewhere (e.g. a Slurm CPU job), the containers could
never reach it. Since Docker can't run on Slurm at all (no Docker install on
this cluster, checked repeatedly — see main `README.md`), foresight has to sit
wherever Docker sits: your machine. It's also a lightweight pure-HTTP proxy —
no GPU, trivial CPU — so there's no upside to a Slurm job even if the
constraint above didn't exist.

## One-time setup

**Slurm submit host** (`ssh` in, e.g. via VS Code):
```sh
export FORESIGHT_ROOT=$WORK   # or wherever you want weights/envs to live
deploy/tau-slurm/setup/install_vllm_env.sh
deploy/tau-slurm/setup/prefetch_model.sh Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
```
(That FP8 checkpoint — ~30 GB, one GPU, no tensor parallelism needed — is the
one to use; see `results/qwen3-coder-30b-bf16-FAILED/findings.md` for why the
plain bf16 checkpoint is NOT worth retrying: it needs 2 GPUs and tensor
parallelism roughly doubles the NFS read load, which sank a 2h52m attempt at
under 1% loaded.)

**Your machine:** Docker Desktop running, the `foresight` conda env installed
(`foresight/README.md`), SWE-CI cloned.

## Every run

**1. Start the model server on Slurm.** Submit `serve_vllm.sbatch` **directly**
— not `deploy/tau-slurm/launch.sh`, which also submits its own foresight-proxy
Slurm job you don't want here:

```sh
sbatch --job-name=vllm-target \
  --partition=gpu-h200-killable --account=gpu-research \
  --gpus=1 --mem=64000 --time=90 \
  --output=$WORK/logs/vllm-%j.out --error=$WORK/logs/vllm-%j.err \
  --export=FORESIGHT_ROOT=$WORK,FORESIGHT_RUN_DIR=$WORK/foresight-runs/manual,\
FORESIGHT_ROLE=target,FORESIGHT_MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8,\
FORESIGHT_ALIAS="target-upstream aux-upstream",FORESIGHT_PORT=8000,\
FORESIGHT_TOOL_CALL_PARSER=qwen3_coder \
  deploy/tau-slurm/serve_vllm.sbatch
```

**Not `gpu-h100-killable`** — as of 2026-09-19 both its nodes are unusable
for us: `n-102` has a physically unhealthy GPU (vLLM's own import-time check
aborts unconditionally, regardless of which GPU a job is actually allocated),
`t-100`'s driver is too old for the CUDA-13 build `pip install vllm`
resolves. `gpu-h200-killable` is the confirmed-working single-node partition;
see `results/qwen3-coder-next-fp8-overnight/findings.md` for the full trail,
including three CUDA/JIT packaging bugs found and fixed there (now merged —
a fresh `install_vllm_env.sh` picks them up automatically, nothing extra to
do). That result also confirms `Qwen3-Coder-Next-FP8` (bigger, 256K native
context) as the first aux model in this project to pass the grounding bar —
worth it over the 30B above if the extra load time is acceptable
(`FORESIGHT_MAX_MODEL_LEN=131072 FORESIGHT_ENFORCE_EAGER=1` recommended; see
that same findings.md for why).

`FORESIGHT_ALIAS="target-upstream aux-upstream"` registers one loaded model
under both names — one GPU job backs both roles (`--single` topology). Use two
separate submissions with distinct `FORESIGHT_ROLE`/`FORESIGHT_PORT` for
`--split` (different checkpoints per role) instead.

Watch for readiness (this can take 15–50 min on a cold NFS cache — see
`deploy/tau-slurm/README.md`, "How long startup takes"):
```sh
mkdir -p $WORK/foresight-runs/manual
cat $WORK/foresight-runs/manual/target.endpoint   # appears once vLLM answers 200
```

**2. Point foresight at it.** Edit `configs/swe_ci.yaml` on your machine:
`models.target.base_url` and `models.aux.base_url` → the contents of
`target.endpoint` (e.g. `http://n-102:8000/v1`) — the node IP resolves fine
over the VPN, confirmed directly. Leave `adapter.foresight_base_url` as
`http://172.17.0.1:8000/v1` — that one's about your own Docker bridge, unrelated
to where the models live.

**3. Run foresight locally:**
```sh
python -m foresight.server --config configs/swe_ci.yaml
```

**4. Point SWE-CI's own `config.toml` at foresight** (not at vLLM directly —
see `configs/swe_ci.yaml`'s header comment), and run the benchmark as usual.
Docker spins up task containers on your machine; they call foresight over the
bridge; foresight forwards to the Slurm-hosted vLLM over the VPN.

## Cleanup

The vLLM job doesn't stop itself — `scancel` it when you're done, or it burns
your GPU allocation idle:
```sh
squeue --me
scancel <jobid>
```

## If something's unreachable

Re-run the probe from `results/` (or ask whoever last did this) before
assuming the model job itself is broken — VPN routing/firewall behaviour
was confirmed once, not guaranteed forever:
```sh
nc -zv <node> <port>
```
