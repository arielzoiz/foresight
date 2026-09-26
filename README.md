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

## Reproducing the paper's results

**The data.** The paper analyses rows 6-21 of the SWE-CI benchmark csv, 20 epochs per arm, one run per
task and arm, with `Qwen3-Coder-30B-A3B-Instruct-FP8` as both target and aux. Each batch of rows has
its own folder:

| Folder | Role in the paper |
|---|---|
| `results/ab-test__swe-ci__rows-6-7__ep20` ... `rows-21__ep20` | the paper's data (16 tasks) |
| `results/ab-test__swe-ci__rows-22__ep20` | excluded: the foresight arm stopped at epoch 12 of 20 |
| `results/ab-test__swe-ci__rows-1-5__ep5` | pilot: 5 tasks, 5 epochs |
| `results/qwen*`, `results/swe-ci-mechanics-m1-docker` | earlier attempts, see "Earlier attempts" below |

Each A/B folder holds `RUN.md` (a generated summary), `manifest.json` (tasks, commits, settings of both
arms), `metrics.csv` (per task and epoch), the configs both arms ran with, `server-side/` (vLLM
settings and logs) and `data/<arm>/<task_id>/` (per-epoch failing tests, requirements, code diffs;
foresight's trace and the aux agent's sessions under `data/foresight/`). `results/README.md` describes
the layout in full.

**The figures and statistics.** Everything in the Results section comes from `paper_analysis/`, which
reads only the `results/` folders above:

```bash
cd paper_analysis && python make_all.py
```

The scripts import each other as sibling modules, so run them from inside `paper_analysis/`.
`make_all.py` runs the hypothesis tests (`stats_tests.py`: paired t-test and Wilcoxon on EvoScore,
permutation test on lines changed per epoch) and draws every figure into `paper_analysis/out/`:

| Output | In the paper |
|---|---|
| `stats.txt`, `evoscore_per_task.csv` | the numbers in the Results section |
| `fig4_2_progress_curve_all_rows.png` | Figure 4.1, left: EvoScore per epoch |
| `fig4_1_evoscore_paired.png` | Figure 4.1, right: EvoScore per task, paired |
| `fig4_3_changes_per_epoch_all_rows.png` | Figure 4.2: lines changed per epoch |
| `figA_1_diff_hist.png`, `figA_2_diff_qq.png` | Appendix A: normality of the paired differences |

The other `fig4_*` files are variants not used in the paper.

## Setup

```bash
conda create -y -n foresight python=3.13
conda activate foresight
pip install -r requirements.txt
```

## Running an A/B test on SWE-CI

**The setup used for the paper.** Two vLLM servers on Slurm, one job each, same checkpoint
(`Qwen3-Coder-30B-A3B-Instruct-FP8`), one B200 GPU each, no tensor parallelism,
`--max-model-len 262144`, `--tool-call-parser qwen3_coder`. `target` (`:8001`, served name
`target-model`) is what SWE-CI calls directly in the control arm; `aux` (`:8002`, `aux-model`) is used
only by foresight's aux agent in the treatment arm. SWE-CI, Docker and foresight run on one machine
that reaches the servers over the VPN. foresight is a light HTTP proxy (no GPU), and it has to sit
where Docker sits, because SWE-CI's containers must reach it.

1. **Serve the models.** Submit `deploy/tau-slurm/serve_vllm.sbatch` once per role (setup, variables
   and the exact `sbatch` line: `deploy/local-swe-ci-slurm-models/README.md` and
   `deploy/tau-slurm/README.md`). Check the endpoints from the host *and* from a container, since the
   containers make the calls: `docker run --rm curlimages/curl -s http://<node-fqdn>:<port>/v1/models`.
2. **Pick the tasks.** A task is a 1-based data row of SWE-CI's `metadata/<splitting>.csv`. Write the
   header plus your rows into `default.csv` and fetch only those tasks with
   `swe_ci.download.download_hf_folder` (not `python -m swe_ci.download`, which overwrites the csv and
   fetches the whole ~50 GB dataset).
3. **Two SWE-CI configs that differ only in `experiment_name`, `base_url` and `model_name`.** Control
   points at the target endpoint; foresight at `http://<address containers can reach>:<port>/v1` with
   `model_name = "target-model"` (`host.docker.internal` on macOS, `172.17.0.1` on Linux). Everything
   else must match, including `agent_name = "opencode"` and `evolve.max_workers = 1`. SWE-CI's repo
   stays unmodified.
4. **A foresight config** copied from `configs/swe_ci.yaml`, with the model URLs,
   `adapter.foresight_base_url`, `adapter.swe_ci_config`, a unique `trace.path`,
   `trace.log_replies: true` and `adapter.aux_export_dir` set. The configs of the paper's runs are in
   `configs/swe_ci_ab-test__*.yaml` and in each results folder.
5. **Run the arms one after the other**, from the SWE-CI directory:

   ```bash
   PYTHONPATH=src .venv/bin/python -u -m swe_ci.evaluate --config_file config_control.toml > control.log 2>&1
   python -m foresight.server --config configs/<your>.yaml      # in the foresight directory
   PYTHONPATH=src .venv/bin/python -u -m swe_ci.evaluate --config_file config_foresight.toml > foresight.log 2>&1
   ```

   With `max_workers = 1` foresight finds the calling container as the only running one. With
   several containers and no IP match it degrades to a body-only aux call
   (`provenance.fallback: body_only`), which is not the condition under test; exclude such runs.
   A failed experiment folder must be renamed or removed before re-running under the same name,
   because SWE-CI resumes from its checkpoints.
6. **Collect the results** into `results/`, for both arms at once:

   ```bash
   python tools/collect_swe_ci_ab.py --swe-ci-dir ../SWE-CI --metadata-csv ../SWE-CI/metadata/<full>.csv \
       --arm control=<control experiment>:config_control.toml \
       --arm foresight=<foresight experiment>:config_foresight.toml \
       --trace foresight=traces/<foresight experiment>.jsonl \
       --aux-export-dir foresight=<aux_export_dir> --foresight-config configs/<your>.yaml \
       --stdout-log control=control.log --stdout-log foresight=foresight.log \
       --model-url target=http://<node-fqdn>:8001/v1 --model-url aux=http://<node-fqdn>:8002/v1
   ```

   Check that no aux run has a `fallback`. The server side then adds what only it knows (vLLM
   flags, sampling defaults, job ids, logs) to `results/<label>/server-side/`, following the
   `TEMPLATE.md` the collector left there.

### Naming conventions for A/B tests

Rows are 1-based data rows of the **full** `metadata/<splitting>.csv`: `6-10`, or `1-2_5` for a
non-contiguous set. The epoch cap is `evolve.max_epoch`.

| What | Pattern | Example |
|---|---|---|
| results folder | `results/ab-test__swe-ci__rows-<rows>__ep<epochs>` | `ab-test__swe-ci__rows-6-10__ep20` |
| SWE-CI `experiment_name` | `ab-test__swe-ci__<arm>__rows-<rows>__ep<epochs>`, `<arm>` is `control` or `foresight` | `ab-test__swe-ci__foresight__rows-6-10__ep20` |
| foresight trace | `traces/<experiment_name>.jsonl` | `traces/ab-test__swe-ci__foresight__rows-6-10__ep20.jsonl` |
| aux export dir | `traces/<experiment_name>__aux-agent/` | |

Batches with the same epoch cap, arms, models and settings can be merged by copying their
`data/<arm>/<task_id>/` folders. Recompute EvoScore over all merged tasks from the `iteration.jsonl`
files rather than averaging the batches' averages.

## Try it locally (no GPU)

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
counts; they belong to no session. See "Design notes" below.

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
| `configs/local.yaml` | treatment, aux is a real agent over a local repo | `template`, with `guard: manifest` |
| `configs/swe_ci.yaml` | treatment on SWE-CI, aux runs inside the live task container | `template` -- the template for an A/B run |
| `configs/swe_ci_ab-test__*.yaml` | treatment on SWE-CI, the paper's runs | as `swe_ci.yaml`, one per batch of rows |

The local configs (`local.yaml`, `local-fake.yaml`, `vllm-local.yaml.tmpl`,
`fake.yaml.tmpl`) also set `adapter.max_concurrent_aux: 1`: two `opencode run`
processes sharing one on-disk session database deadlock each other.

`canary.yaml` answers a question no upstream inspection can, once a real
harness sits between us and the visible output: *did the enhancement reach the
model, or did it merely get sent?* Point a real harness at it and grep its
transcript for "what day is it" -- reach for this when the enhanced arm
(`naive.yaml`) behaves indistinguishably from the control, to rule out "never
arrived" before concluding "no effect", and to catch a silent revert of the
re-application logic mid-session.

## Design notes

See `foresight-design-plan.md` for the full reasoning. In short:

- **Requests that are not agent turns are forwarded untouched.** A request
  without `tools` (e.g. opencode's title-generation request at the start of
  each session) skips both aux and the builder, so it reaches the model exactly
  as in the control arm. Handling it as a session start raced the real aux run
  (`database is locked`, see `results/qwen2.5-coder-7b/findings.md`). For a
  caller that never sends tools, set `adapter.require_tools: false`; `/health`
  counts `enhanced` and `skipped_not_agent` requests, so a run that silently
  enhances nothing is visible.
- **`ManifestGuard` detects aux's writes.**

## Tests

```bash
pytest tests/ -q
```

Needs neither GPU nor Docker, and no agent harness -- `tools/fake_agent.py`,
`tools/fake_upstream.py` and `tools/fake_docker.py` stand in for the harness,
the model and Docker. The suite covers the proxy pipeline end to end, both
builders, session keying, the aux agent and its guard, and each adapter.
`deploy/local-ollama/README.md` has the setup for checking `SweCiAdapter`
against real Docker.

## Earlier attempts

Before the paper's runs, smaller models served locally through Ollama were tried as aux; none was
good enough:

| Model | Result |
|---|---|
| `qwen2.5-coder:7b` / `:14b` | can't emit well-formed tool calls at all via Ollama |
| `qwen3:8b` | tool-call format is fine, but 90% of aux invocations produced empty output outright |
| `qwen3:14b` | stopped failing outright, but still couldn't find the one file in a two-file toy repo after 7.5 minutes of searching -- grounded activity, ungrounded result |
| `qwen3-coder:30b` | doesn't fit a 16 GB machine alongside Docker |

The findings are in `results/qwen2.5-coder-7b/`, `results/qwen3-8b-ollama-local/` and
`results/qwen3-14b-ollama-local/`. On Slurm, `results/qwen3-coder-next-fp8-overnight/` is the first aux
model that passed the grounding bar (too slow to load to be used), `results/qwen3-coder-30b-bf16-FAILED/`
records why the bf16 checkpoint was dropped for the FP8 one, and `results/swe-ci-mechanics-m1-docker/`
is the first check of `SweCiAdapter` against real Docker, with mock models.

These led to the model used for all the paper's runs: `Qwen3-Coder-30B-A3B-Instruct-FP8`, served by
vLLM on Slurm as both target and aux (see "Running an A/B test on SWE-CI").
