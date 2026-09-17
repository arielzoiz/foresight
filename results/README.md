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
