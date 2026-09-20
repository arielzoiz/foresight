# Results

One directory per measured run, copied out of the cluster's run directory by
`tools/collect_run.sh` so a result can be committed, cited from a commit, and
read by someone with no cluster access.

```
<label>/
    findings.md        what the run showed, and what it does not show
    summary.txt        tools/aux_transcript.py's verdict -- read this first
    prompts.txt        the task, aux and builder prompts -- the condition itself
    run-settings.txt   partition, node, GPUs, memory, time limit (from sacct)
    aux-agent/         opencode's session exports: every aux turn and tool call
    trace.jsonl        one foresight record per model call
    config.yaml        adapter, builder, aux prompt, guard -- the full config
    vllm.txt           served model, parser, TP size; load timeline; error tail
```

**Read `findings.md`, then `summary.txt`.** The rest is the evidence behind them.

`prompts.txt` and `run-settings.txt` are what make a result reproducible rather
than anecdotal. The three prompts *are* the experimental condition, so two arms
can be diffed by diffing that one file; `run-settings.txt` is recovered from
`sacct`, which remembers a job's resource request long after `squeue` has
forgotten the job existed.

Regenerate any of it with:

```sh
tools/collect_run.sh $WORK/foresight-runs/<run-dir> <label>
```

For a run against a local OpenAI-compatible server (Ollama, llama.cpp, LM
Studio) instead of a Slurm run directory, use `tools/collect_local_run.sh`
instead -- same shape, but `run-settings.txt` records the local server's own
version and this host's hardware instead of `sacct`/vLLM, and there is no
`vllm.txt`:

```sh
tools/collect_local_run.sh --config PATH --trace PATH --workspace DIR \
    --aux-home DIR --label <label>
```

For a SWE-CI A/B run (control vs foresight, with the models on Slurm and SWE-CI
+ foresight on a laptop), use `tools/collect_swe_ci_ab.py` instead. It gathers
SWE-CI's `experiments/` folder, foresight's trace and the aux transcripts into
one layout, and adds what only SWE-CI's archive folders still hold: per epoch,
the failing-test summary the architect saw, the `requirement.xml` it wrote, and
the code diff the programmer made. It writes a generated `RUN.md` beside your
hand-written `findings.md`. See the docstring of the script for the layout and
the exact command, and `ab-test__swe-ci__rows-1-5__ep5/` for an example.

### SWE-CI A/B folder layout

`results/ab-test__swe-ci__rows-<rows>__ep<epochs>/`. Files marked *generated* are written by
`collect_swe_ci_ab.py` and are overwritten by a re-run; `findings.md` is yours.

```
findings.md            hand-written: what the run showed and does not show
RUN.md                 generated: run summary, results, code change and quality per epoch,
                       timing, SWE-CI cost, aux provenance, incidents
manifest.json          generated: rows, each task's repo and both commits (from the benchmark csv),
                       settings, and whether the arms' settings matched
metrics.csv            generated: one row per arm, task and epoch end (0 = starting state):
                       gap, passed, accepted, lines changed (this epoch, cumulative), mi, pylint
environment.txt        generated: commits, versions, Docker resources, model endpoints (a model
                       line is kept from an earlier collection if the server has since expired)
configs/               the SWE-CI configs and the foresight config used, keys redacted;
                       tasks.txt lists the task ids
server-side/           NOT generated: added by whoever ran the model servers, after the caller side has
                       collected. TEMPLATE.md is the checklist (models, vLLM flags, sampling defaults,
                       hardware, job ids, events, shared load, logs); setup.md is their description.
                       The collector creates the folder if absent and never overwrites it.
plots/                 gap, EvoScore, cumulative lines changed, maintainability, pylint, per epoch
data/<arm>/            <arm> is control or foresight
    main.log             SWE-CI's own log
    swe_ci_stdout.log    console output of swe_ci.evaluate: SWE-CI's effective config and the final
                         table (not the agent's log); a crash of the process itself would show here
    summary.txt          the swe_ci.summarize table: EvoScore, resolved, zero-regression
    <task_id>/
        iteration.jsonl    one line per epoch, 0 = starting state: gap, pytest counts, agent tokens/time
        task.log           SWE-CI's per-task log: timestamps, retries, errors
        epoch_<N>/
            non-passed/summary.jsonl   failing tests at the START of epoch N
            requirement.xml            the architect's output for epoch N
            edit.diff                  the code change made in epoch N (a/ = before, b/ = after)
        final/non-passed/summary.jsonl   failing tests AFTER the last epoch (the last accepted state)
    trace.jsonl.gz       foresight's trace (foresight arm only); rows carry the target model's reply
                         (text and tool calls) when the run set trace.log_replies
    aux_sessions.json    one entry per aux run: task, epoch, phase, answer, provenance
    AUX_PAIRS.md         each aux answer next to its base task
    aux-agent/           ses_*.json, the aux agent's own reads and tool calls (only when
                         adapter.aux_export_dir was set; absent for the first run)
    examples/            foresight arm only: readable samples of aux runs (a full architect and
                         programmer example, and two real exported aux sessions in aux-export-sample/)
foresight_server.log   foresight's own server log for the run
plot_results.py        the original plotting script, from before the collector drew plots
```

How to read it:

- **The base task.** A SWE-CI task is a repo evolved from `current_sha` until the tests of
  `target_sha` pass; each epoch is an architect step, then a programmer step. Aux names plausible
  future tasks after the *current* step, so its base task is that step's input: the failing tests
  for an architect session, that epoch's `requirement.xml` for a programmer session. The commits
  are in `manifest.json`.
- **`requirement.xml` is an output, not an input.** In the foresight arm the architect wrote it
  after foresight added aux's future-task list to the prompt, so it depends on aux. Epoch 1 is the
  cleanest place to compare arms: both start from identical code and failing tests.
- **Epoch folders.** SWE-CI archives each epoch's starting state under a timestamp; the collector
  maps them to `epoch_<N>` using the log. An epoch whose pytest could not run (`accepted` is false
  in `metrics.csv`, marked `!` in `RUN.md`) is not accepted: SWE-CI keeps the previous code, so
  `gap` is empty and the quality scores repeat the previous epoch, while `edit.diff` still shows the
  attempted change.
- **Aux runs to tasks.** `aux_sessions.json` joins each aux run to a task and epoch by timestamp
  against `task.log`. A run inside two epoch windows (`max_workers > 1`) stays unjoined, marked
  `ambiguous`. Exclude any run whose provenance has `fallback: body_only`.
- **Code-quality scores.** `mi` is SWE-CI's `mi_score` (radon); `pylint` is a corrected pylint run,
  because SWE-CI's own `pylint_score` returns 0 on these repos. Both exclude `tests/` and skip the
  macOS `._*` sidecar files. Only the change over epochs means anything, not the absolute value.

## The question every run here is trying to answer

Does the auxiliary agent actually **read the repository** before naming plausible
future tasks, or does it answer from the task description alone?

It matters because the whole hypothesis is that anticipating future work produces
more maintainable code — and an aux that never opened a file cannot judge what
would make future work easier. An ungrounded aux run still produces a fluent,
well-formatted answer, so this is not visible without looking.

Two independent records answer it, and they should agree:

| Record | Verdict it gives |
|---|---|
| `trace.jsonl`, `role: "aux"` rows | **how many** model calls the aux agent made |
| `aux-agent/ses_*.json` | **what happened inside** each one — tool calls, inputs, outputs |

A grounded run shows three or more aux calls with tool parts present. Two calls
and no tool part means ungrounded — and only the transcript says *why*: a clean
prose answer means the model never reached for a tool, while a JSON or XML blob
means it tried and nothing downstream parsed it (suspect `--tool-call-parser`).

**`aux.usable: true` in the trace is not evidence of grounding.** The quality gate
counts enumerated items, not whether any of them reference a real file.
