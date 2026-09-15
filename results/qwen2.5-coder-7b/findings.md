# Qwen2.5-Coder-7B-Instruct — aux is ungrounded, and the model never tries

**Run:** `arielzoizner-20260914-174800`, 2026-09-14
**Serving:** vLLM on one a6000 (n-601), `--tool-call-parser hermes`, `--max-model-len 32768`
**Adapter:** `local` (M2) — a real opencode aux agent over a scratch clone of
[`cachecontrol`](https://github.com/psf/cachecontrol), guard armed
**Task prompt:** *"Add an option to the CacheControl adapter that lets the caller
override the max-age used for cached responses."*

## Verdict

**Ungrounded.** Both aux sessions made exactly **one model call and zero tool
calls**, answering in clean prose without opening a single file.

This is not a wire-format problem. `hermes` is the correct parser for this
checkpoint — its chat template emits `<tool_call>{json}</tool_call>`, which
`hermes` parses (verified offline, see below). The model simply does not reach
for a tool.

The content proves it independently of any counter. Against a **Python** library,
aux named:

```
src/features/newFeature.js
src/utils/helpers.js
public/index.html
CHANGELOG.md
```

JavaScript and web files, in 5.9 seconds, none of which exist in the repository.
The target model failed identically, replying with a fenced JSON `edit` block
against `src/adapters/cache-control.js` — also nonexistent, also JavaScript.

`aux.usable: true` in the trace does not contradict this: the gate counts
enumerated items, and a fluent hallucination passes it.

## Why no parser change would have helped

Before running this, every registered vLLM tool parser was replayed offline
against the output shapes this model was previously observed to emit
(`tools/probe_tool_parsers.py`, no GPU required):

| Output | Recognised by |
|---|---|
| `<tool_call>{"name":…,"arguments":…}</tool_call>` | `hermes` (the control) |
| ```` ```json {"name":…,"arguments":…} ``` ```` single object | **nothing** |
| ```` ```json [{…}] ``` ```` array | `xlam` |

`xlam` is not a safe substitute: it strips fences happily but requires an
**array**, and it rejects the correct single-object `hermes` form — so adopting
it would lose the turns where a model gets the format right. `llama3_json` would
accept bare JSON but refuses to initialise without `<|python_tag|>` in the
tokenizer, which Qwen's lacks.

In this run the question was moot anyway: the model produced prose, not a
malformed tool call.

## Two pipeline bugs this run exposed

Both are independent of the model and affect every `local`-adapter run,
including the earlier one described in `deploy/tau-slurm/README.md`.

**1. opencode's title-generation request is treated as a task prompt.** The first
aux session's prompt reads, verbatim:

> The developer is about to work on this task:
>
> Generate a title for this conversation:

A whole aux agent was spawned to brainstorm future work for *naming a chat*. Its
output — the JavaScript hallucination above — was then injected into that
request's prompt. Every session pays for this second, meaningless aux run.

**2. Two aux agents run concurrently and deadlock opencode's database.** opencode
issues the title call and the real agent call **1.8 s apart**
(`15:30:23.499` and `15:30:25.261`). They hash to different session keys
(`8595f970135d1b96`, `97a4813495312818`), so both count as session starts and
both spawn `opencode run` against the *same* `OPENCODE_HOME`. The result:

```
Error: Unexpected error

database is locked
```

and the first request failed with `502 aux_failure`. This violates the design's
own constraint 3, *"One agent at a time."*

**Giving each aux run its own `OPENCODE_HOME` would not fix it.** Measured
afterwards: `OPENCODE_HOME` does not isolate opencode's session database at all —
sessions from different jobs, nodes and workspaces all accumulate in one shared DB.
So the fix has to be serialising aux runs, not separating their homes.

## What this run does not show

- Nothing about whether future-task prompting helps. That needs a grounded aux
  and both experimental arms; this is a capability check on the aux side only.
- Nothing about larger models. A 30B run is the point of comparison.
- The ungroundedness here is *model-specific*, not evidence that the M2
  topology is wrong — the proxy, guard, trace, session keying and transcript
  capture all behaved correctly.

## Reproducing

```sh
deploy/tau-slurm/launch.sh --single Qwen/Qwen2.5-Coder-7B-Instruct \
    --workspace $WORK/workspaces/tinyrepo-7b \
    --partition killable --account gpu-research --foresight-account gpu-research \
    --tool-call-parser hermes --mem 64000 --time 240 --constraint a6000
```

Then read `summary.txt`, or regenerate it:

```sh
python tools/aux_transcript.py "results/qwen2.5-coder-7b/aux-agent/ses_*.json"
```
