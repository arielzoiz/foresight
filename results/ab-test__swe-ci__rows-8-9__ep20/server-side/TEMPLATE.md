# Server side: what to add to this run

The caller side (the machine running SWE-CI and foresight) collected this folder. The people who ran the
model servers add what only they know, here, after that collection. The collector never overwrites
anything in `server-side/`, so re-collecting is safe. Replace or keep this file; add your own files
beside it (e.g. `setup.md`, `vllm-<role>.log`).

Fields, in order of how much a reader needs them:

- **Models**: checkpoint, served name per role, and the port each is on
- **vLLM**: version, launch flags (`--max-model-len`, `--tool-call-parser`, tensor parallel size,
  `--enforce-eager`), and the sampling defaults the server applies when a caller sends none
  (temperature, top_p, top_k, max tokens): these change results and are visible only on this side
- **Hardware**: partition, node, GPU type and count, memory
- **Jobs**: Slurm job ids, submit/start/end times, wall-clock limit, how long the model took to load
- **Events during the run**: preemptions, restarts, OOMs, errors in the vLLM log
- **Shared capacity**: other jobs on the same node or GPUs, since timing between arms depends on it
- **Logs**: the vLLM log (or its tail and any errors) for each server
