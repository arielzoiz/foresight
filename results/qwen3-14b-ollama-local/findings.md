# Qwen3-14B via Ollama, local M1 Mac — pipeline is reliable, aux still doesn't find the file

**Run:** local (no Slurm, no GPU node), 2026-09-16, Apple M1 Pro / 16GB
**Serving:** Ollama 0.34.0, `qwen3:14b` (Q4_K_M), OpenAI-compatible endpoint
**Adapter:** `local` (M2) — a real opencode aux agent over the same scratch
repo used for `results/qwen3-8b-ollama-local/`
**Task prompt:** *"Add a multiply(a, b) function to calc.py, matching the
style of add and subtract."*
**foresight commit:** `81bd178`

## Verdict

**Mechanically reliable, semantically still ungrounded.** Unlike `qwen3:8b`
(90% empty-stdout failure rate, see the sibling results directory), aux
succeeded on its **first** attempt here — five model calls, real tool usage
throughout (`tool_count` 8-10 per call), zero `aux_failure` rows, and every
subsequent target request in the session correctly got `enhanced: true`. The
pipeline itself — session gating, aux dispatch, quality flagging, prompt
re-application — worked exactly as designed, same as the 8B run.

**But aux never found the file it was looking for.** Its full answer:

```
The search for files containing "calc" didn't yield any results. Here are some options to try next:

1. Try alternative search terms like "calculate" or "calculator"
2. Adjust file type filters - currently searching all files, but you could limit to specific extensions (e.g., .ts, .js, .py)
3. Check for partial matches or variations of the term

Would you like to try any of these options?
```

`calc.py` is the only source file in the workspace besides `README.md`. Aux
made 5 tool calls (`tool_count` 8-10 each — real activity, not a no-op) over
**462 seconds** and still failed to locate it, then gave up with a
clarifying question rather than reporting "no matching files found" or
retrying with a broader search. `aux.usable: true`, `items: 3` — the gate
passed this for the same reason it passed qwen3:8b's non-answer: it counts
numbered lines, not whether the aux actually read anything real. This is the
same failure *shape* as qwen3:8b's "asking for help with my own broken tool
call" (a fluent, well-formatted non-answer that satisfies the item-count
check), just reached by search-fu failure instead of a malformed tool-call
schema.

**The target model separately mishandled its own tool call while solving the
task**, unrelated to aux: opencode's CLI output shows the target invoking a
subagent-style call named `find-calc-file` where opencode expected a real
session ID (`Expected a string starting with "ses", got "find-calc-file"`),
and the session terminated on that error rather than recovering. This
happened after the prompt was already enhanced (trace rows 7-9 all show
`enhanced: true`), so it is independent of aux's own failure above — two
separate models-fumbling-tool-schemas problems in one run, one in aux, one in
target.

**462 seconds for one aux call is a real cost.** At that rate, a real SWE-CI
task's architect/programmer timeout (3600s in `config.toml`) has room for at
most a handful of aux attempts before the phase itself times out, even before
counting the target's own generation time.

## Comparison with `qwen3:8b`

| | `qwen3:8b` | `qwen3:14b` |
|---|---|---|
| aux success rate (this task) | 1 of 10 attempts (across 2 sessions) | 1 of 1 |
| aux tool usage | mostly none / malformed args | real, substantial (8-10 tool calls/attempt) |
| aux answer quality | non-answer (broken tool-call plea) | non-answer (gave up after failed search) |
| aux latency | not separately timed | 462s for the one successful call |
| pipeline correctness | correct (gate, retry, guard all worked) | correct (gate, guard all worked, no retries needed) |

Bigger fixed the *reliability* problem (aux stopped silently failing) but not
the *grounding* problem (aux still doesn't produce a usable answer) — and
cost roughly 5-8x the wall-clock time doing it, in this one sample.

## What this run does not show

- Whether a different search approach (a plainer aux prompt, or one that
  starts with `glob` rather than a text search) would have found the file —
  not tested here.
- The target's own `find-calc-file` failure is not diagnosed further; it may
  be unrelated to model size, since it happened only once and this run has no
  qwen3:8b equivalent to compare against for the target's own tool use.
- Whether either model does better against a real SWE-CI task's harder,
  larger codebase rather than this two-file toy repo.

## Reproducing

```sh
ollama pull qwen3:14b
tools/opencode_home.sh --home /tmp/foresight-aux-home --model aux-model \
    --base-url http://127.0.0.1:8000/v1 --context 8192 --output 2048
tools/opencode_home.sh --home /tmp/foresight-target-home --model target-model \
    --base-url http://127.0.0.1:8000/v1 --context 8192 --output 2048
python -m foresight.server --config results/qwen3-14b-ollama-local/config.yaml
cd /tmp/foresight-scratch
HOME=/tmp/foresight-target-home opencode run --model custom/target-model \
    "Add a multiply(a, b) function to calc.py, matching the style of add and subtract."
```

Then regenerate this directory's evidence with `tools/collect_local_run.sh`
(see `results/README.md`).
