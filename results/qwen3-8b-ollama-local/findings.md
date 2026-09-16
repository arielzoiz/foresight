# Qwen3-8B via Ollama, local M1 Mac — aux invocation is unreliable, and the
# quality gate does not catch it

**Run:** local (no Slurm, no GPU node), 2026-09-16, Apple M1 Pro / 16GB
**Serving:** Ollama 0.34.0, `qwen3:8b` (Q4_K_M), OpenAI-compatible endpoint
**Adapter:** `local` (M2) — a real opencode aux agent over a scratch repo
(`/tmp/foresight-scratch`, a two-file toy calculator), guard armed
**Task prompt:** *"Add a multiply(a, b) function to calc.py, matching the
style of add and subtract."*
**foresight commit:** `db3a5bd` (includes the `$PWD` fix described below)

## Verdict

**Unreliable, not merely ungrounded.** Across the two agent sessions this one
task produced, aux's underlying `opencode run` subprocess exited 0 with
**empty stdout in 9 of 10 invocations** (90%):

| Session | Attempts | Failed (empty stdout) | Succeeded |
|---|---|---|---|
| top-level target session | 4 | 3 | 1 |
| nested "subagent" session (opencode's own delegation, see below) | 6 | 6 | 0 |

The nested session's task ("Update calc.py with multiply function") never
recovered and opencode surfaced foresight's own `aux_failure` error text
directly into the target model's context, which then gave up and asked the
user to troubleshoot instead of finishing the task.

**The one "successful" aux answer is not usable either, and the quality gate
passed it anyway.** Its full text:

```
I need to fix the task tool call by adding the required "description" field. Could you please provide:
1. A short 3-5 word description of the task
2. The type of subagent to use (explore/general)
3. The specific instructions/prompt for the agent

This will help me create a properly formatted task tool call with all required parameters.
```

`aux.usable: true`, `items: 3` in the trace — the gate counts three numbered
lines and calls it a plausible future-task list. It is actually aux asking
*us* for help filling out a broken tool call. This got injected into the
target's prompt as if it were real future-work context.

## What aux is actually doing wrong

The one case where we captured the underlying content (a separate run,
before the `$PWD` fix below, same model/config) shows the same shape: aux
identified the right *intent* — read a file, or spawn a task — but omitted a
required argument (`read`'s `filePath`, `task`'s `description`), got a tool
error back, and answered the error message instead of retrying or falling
back to a direct tool call. Both the top-level and nested sessions show aux
reaching for `task` (opencode's own subagent-delegation tool) rather than
reading files directly — a consistent behavioral pattern across attempts, not
a one-off. This is a different failure shape from Qwen2.5-Coder-7B's
(`results/qwen2.5-coder-7b/`), which never attempted a tool at all; this
model attempts tools constantly and gets their schemas wrong.

We cannot confirm the exact error text for most of the 9 failed attempts:
see the opencode-version caveat below.

## A pipeline bug this run exposed and fixed

**`LocalAdapter`'s aux subprocess inherited a stale `$PWD`.** `_child_env()`
passed `cwd=self._workspace` to the subprocess (correctly changing its real
working directory) but only inherited the *server's* environment otherwise,
leaving `$PWD` pointing at wherever the foresight server itself was launched
from. opencode's own project/session bookkeeping reads `$PWD` rather than
calling `getcwd()` — measured directly with a minimal repro against a fresh
`HOME` — so every aux session got silently recorded under the *server's*
launch directory instead of the workspace aux actually explored. File tool
calls were unaffected (those resolve against the real cwd), but anything
that identifies a session by its recorded directory — `collect_run.sh`'s and
the new `collect_local_run.sh`'s aux-agent export filter included — silently
found nothing. Fixed in `foresight/adapters/local.py` (commit `db3a5bd`):
`env["PWD"]` is now set to match `self._workspace`.

Confirmed fixed: after the fix, all 10 sessions in this run's aux home
correctly show `directory = /private/tmp/foresight-scratch` in opencode's own
`opencode.db`. This is *not* the same bug documented in
`deploy/tau-slurm/README.md` about `OPENCODE_HOME` not isolating the session
database across unrelated runs on shared NFS — that is about the database
being shared; this is about one session correctly written to its own
database under the wrong directory value.

## A known gap we hit and stopped chasing: opencode 1.18.30 cannot export its own sessions

`opencode session list` and `opencode export <id>` — the mechanism
`collect_run.sh` and `collect_local_run.sh` both use to pull per-tool-call
transcripts — return nothing for headless `opencode run` sessions on this
opencode version, even after confirming the `$PWD` fix above and even when
passed a session ID read directly out of the database (`export` answers
"Session not found" for an ID visible in `SELECT id FROM session`). This
appears to be an opencode CLI limitation for `run`-created sessions in
1.18.30, not a foresight or `collect_local_run.sh` bug — `session`,
`project` and other tables in `opencode.db` are populated correctly, but the
CLI's own list/export path evidently queries through something else (a
`workspace` table that stays empty for every session in this run is one
candidate) that headless `run` never populates. `results/qwen2.5-coder-7b/`
proves this mechanism *did* work on whatever opencode version the Slurm
cluster had; it does not on this machine's 1.18.30. Consequence: this run's
`aux-agent/` has `opencode.log` only, no `ses_*.json`, and no `summary.txt`.
Anyone hitting empty `aux-agent/` exports locally should check the installed
opencode version before assuming the collection tooling is broken.

## What this run does not show

- Nothing about `qwen3:14b`: its raw tool-calling format was verified
  separately (correct, structured `tool_calls`, no schema errors) but it has
  not yet been run as an actual aux/target harness end to end.
- Nothing about the target model's own code-writing quality — the top-level
  session got an enhanced (garbage) prompt and never got far enough to matter
  here.
- Whether this is a Q4_K_M quantization artifact or a `qwen3:8b`-at-any-precision
  limitation. Not tested at higher precision.
- The per-attempt root cause for 8 of the 9 failures (only one, from a
  separate run, was directly captured) — see the opencode-export caveat
  above.

## Reproducing

```sh
ollama pull qwen3:8b
tools/opencode_home.sh --home /tmp/foresight-aux-home --model aux-model \
    --base-url http://127.0.0.1:8000/v1 --context 8192 --output 2048
tools/opencode_home.sh --home /tmp/foresight-target-home --model target-model \
    --base-url http://127.0.0.1:8000/v1 --context 8192 --output 2048
python -m foresight.server --config results/qwen3-8b-ollama-local/config.yaml
# separately, as the target caller:
cd /tmp/foresight-scratch
HOME=/tmp/foresight-target-home opencode run --model custom/target-model \
    "Add a multiply(a, b) function to calc.py, matching the style of add and subtract."
```

Then regenerate this directory's evidence with:

```sh
tools/collect_local_run.sh --config <path> --trace <path> \
    --workspace /tmp/foresight-scratch --aux-home /tmp/foresight-aux-home \
    --label qwen3-8b-ollama-local
```
