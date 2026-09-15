#!/bin/sh
# Copy one Slurm run's evidence out of $FORESIGHT_ROOT into results/ in the repo,
# so it can be committed and reviewed by someone without cluster access.
#
# Run directories live under $FORESIGHT_ROOT/foresight-runs/ on shared storage,
# outside the repo and outside git. That is right for a run in flight -- they are
# large and numerous -- but it makes a finished result impossible to share or to
# cite from a commit. This copies the small, durable part in.
#
# What it takes, and what it deliberately leaves behind:
#   config.yaml        the rendered config -- which adapter, builder, prompt
#   trace.jsonl        one record per model call; the experiment's own evidence
#   aux-agent/         opencode's session exports: every aux tool call
#   vllm.txt           per vLLM role: the served model, tool-call parser and
#                       tensor-parallel size; the checkpoint load timeline; and
#                       the tail of stderr, which for a failed run is the result
#   summary.txt        tools/aux_transcript.py's verdict, rendered once here so
#                       a reviewer does not need the repo checked out to read it
#   prompts.txt        the three prompts that define the condition, lifted out of
#                       config.yaml and trace.jsonl so they can be read and
#                       diffed without parsing either: the TASK prompt the target
#                       agent was given, the AUX prompt template, and the BUILDER
#                       template that injects aux's answer
#   run-settings.txt   how the run was resourced -- partition, node, GPUs, memory
#                       and time limit, recovered from sacct via the job ids in
#                       the log filenames, so it works retroactively on runs that
#                       predate this script
#
# The last two exist because a result nobody can reproduce is an anecdote. The
# prompts ARE the experimental condition, and the Slurm settings are what decide
# whether a rerun behaves the same.
# NOT the full vLLM/proxy logs: tens of MB of progress bars that say nothing the
# trace does not.
#
# No secrets are involved -- vLLM runs without an API key and the configs carry
# none -- but the copied files DO contain absolute workspace paths and the task
# prompt, so read them before pushing to anywhere public.
#
# Works on any run directory produced by deploy/tau-slurm/launch.sh, in any of
# its modes (--single, --split, --fake; --adapter local or generic), and on runs
# that finished long ago -- nothing here needs the jobs to still exist. Missing
# pieces are reported as warnings, never errors: a run that died before writing
# a config is exactly the run whose evidence you most want kept.
#
# Usage:
#   tools/collect_run.sh <run-dir> <label>
#   tools/collect_run.sh --help
# Example:
#   tools/collect_run.sh $WORK/foresight-runs/arielzoizner-20260914-181041 qwen3-coder-30b-bf16
#
# Then write results/<label>/findings.md by hand -- what the run showed, and what
# it does NOT show -- and commit the directory. That file is the only part no
# script can produce.

set -eu

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <run-dir> <label>   (see --help)" >&2
    exit 1
fi

run_dir="$1"
label="$2"

if [ ! -d "$run_dir" ]; then
    echo "collect_run: not a directory: $run_dir" >&2
    exit 1
fi

# A label becomes a directory name under results/. Rejected rather than
# sanitised: silently rewriting someone's label would put the output somewhere
# they did not ask for and then report success.
case "$label" in
    */*|.*|"")
        echo "collect_run: label must be a plain directory name (no '/', no leading '.'): $label" >&2
        exit 1
        ;;
esac

# `python` is not guaranteed to exist -- many systems ship only `python3`, and
# this script should not require the project's conda env just to copy files.
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=""
    echo "collect_run: WARNING -- no python found; prompts.txt and summary.txt will be skipped" >&2
fi

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
out="$repo_dir/results/$label"
mkdir -p "$out"

for f in config.yaml trace.jsonl; do
    if [ -f "$run_dir/$f" ]; then
        cp -f "$run_dir/$f" "$out/$f"
    else
        echo "collect_run: WARNING -- no $f in $run_dir" >&2
    fi
done

if [ -d "$run_dir/aux-agent" ]; then
    mkdir -p "$out/aux-agent"
    # The session exports and opencode's log.
    # `|| true` is load-bearing under `set -e`: an unmatched glob stays literal
    # in sh, the [ -f ] then fails, and that failure would be the loop's exit
    # status -- aborting the whole collection for a run that simply has no
    # transcripts yet.
    for f in "$run_dir"/aux-agent/*.json "$run_dir"/aux-agent/opencode.log; do
        [ -f "$f" ] && cp -f "$f" "$out/aux-agent/" || true
    done
else
    echo "collect_run: WARNING -- no aux-agent/ in $run_dir (generic adapter, or aux never ran)" >&2
fi

# One file per run, not three: head/tail/load written separately meant three
# near-identical copies on a short log.
#
# Per vLLM role -- every role, since --split serves aux from a SECOND job with
# its own checkpoint, and dropping it would lose the identity of the model that
# produced the aux answer:
#
#   identity   served model, --tool-call-parser, --tensor-parallel-size
#   load       the PREFETCH percentages, which are on stdout while the shard
#              counter on stderr sits at 0/N for the whole phase by design --
#              read only stderr and a healthy slow load looks wedged
#   tail       the end of stderr, which for a failed run is the whole result
: > "$out/vllm.txt"
for role_log in "$run_dir"/logs/vllm-*.err; do
    [ -f "$role_log" ] || continue
    role_out="${role_log%.err}.out"
    {
        echo "########## $(basename "$role_log")"
        echo "--- identity"
        grep -E '^foresight: \[' "$role_log" 2>/dev/null || echo "(none)"
        if [ -f "$role_out" ]; then
            echo "--- load timeline (from .out)"
            tr '\r' '\n' < "$role_out" \
                | grep -E 'Prefetching checkpoint files|Filesystem type for checkpoints|KV cache|graph capturing' \
                || echo "(none)"
        fi
        echo "--- tail of .err"
        tr '\r' '\n' < "$role_log" \
            | grep -vE '^(Loading safetensors|Capturing CUDA graph)|^$' \
            | tail -25
        echo
    } >> "$out/vllm.txt" 2>/dev/null || true
done
[ -s "$out/vllm.txt" ] || rm -f "$out/vllm.txt"

# -- how the run was resourced -------------------------------------------
# Job ids are recoverable from the log filenames (logs/vllm-target-<id>.err),
# so this works on runs recorded before this script existed. sacct keeps the
# submission's resource request after the job is gone, which is the point --
# by the time anyone reviews a result, squeue has long forgotten it.
{
    echo "# Run settings for $(basename "$run_dir")"
    echo "# Recovered from sacct; see config.yaml for the pipeline's own settings."
    echo
    # Which code produced this. Without it, a result cannot be tied to the
    # pipeline version that made it -- and this pipeline changes between runs.
    echo "## foresight"
    echo "  commit=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "  dirty=$(test -n "$(git -C "$repo_dir" status --porcelain 2>/dev/null)" && echo yes || echo no)"
    echo "  collected=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    for log in "$run_dir"/logs/*-*.err; do
        [ -f "$log" ] || continue
        jobid=$(basename "$log" | sed -n 's/.*-\([0-9][0-9]*\)\.err$/\1/p')
        [ -n "$jobid" ] || continue
        echo "## job $jobid  ($(basename "$log" | sed 's/-[0-9]*\.err$//'))"
        sacct -j "$jobid" --parsable2 --noheader \
            --format=JobName,Partition,NodeList,ReqTRES,Timelimit,Elapsed,State 2>/dev/null \
            | head -1 \
            | awk -F'|' '{printf "  name=%s\n  partition=%s\n  node=%s\n  requested=%s\n  timelimit=%s\n  elapsed=%s\n  state=%s\n", $1,$2,$3,$4,$5,$6,$7}' \
            || echo "  (sacct unavailable)"
        echo
    done
} > "$out/run-settings.txt" 2>/dev/null || true

# -- the prompts, which ARE the experimental condition --------------------
# config.yaml holds the aux and builder templates but buries them in commentary,
# and the task prompt only exists in the trace. Lift all three into one file so
# a reviewer can read the condition, and so two arms can be diffed directly.
[ -n "$PY" ] && "$PY" - "$run_dir" "$out" <<'PYPROMPTS' 2>/dev/null || true
import json, pathlib, sys

run, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
parts = []

task = None
trace = run / "trace.jsonl"
if trace.exists():
    for line in trace.open(encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        # The first target row that carries a real task prompt. Rows for
        # opencode's title-generation call are targets too, so prefer the
        # longest prompt_in seen rather than simply the first.
        if row.get("role") == "target" and row.get("prompt_in"):
            if task is None or len(row["prompt_in"]) > len(task):
                task = row["prompt_in"]
if task:
    parts.append("=== TASK PROMPT (what the target agent was asked to do) ===\n\n" + task)

cfg = run / "config.yaml"
if cfg.exists():
    text = cfg.read_text(encoding="utf-8", errors="replace")
    # Deliberately crude: these are top-level YAML block scalars, and a real
    # parser would pull in a dependency for two keys.
    # `template:` sits two spaces in, under `builder:` -- matched with its exact
    # indentation so the search cannot wander into some other mapping.
    for key, label in (("aux_prompt:", "AUX PROMPT (what the aux agent was asked)"),
                       ("  template:", "BUILDER TEMPLATE (how aux's answer is injected)")):
        start = text.find("\n" + key)
        if start == -1:
            continue
        block, indent = [], None
        for line in text[start + 1:].splitlines()[1:]:
            if not line.strip():
                block.append("")
                continue
            lead = len(line) - len(line.lstrip())
            if indent is None:
                indent = lead
            if lead < indent:
                break
            block.append(line[indent:])
        parts.append(f"=== {label} ===\n\n" + "\n".join(block).rstrip())

# An empty prompts.txt is worse than none: it reads as "no prompt was used"
# rather than "this run died before a prompt existed".
if parts:
    (out / "prompts.txt").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
PYPROMPTS

# For a FAILED run the log is the entire result, so keep its tail. Carriage
# returns are rewritten to newlines and tqdm progress bars dropped, because
# vLLM's loader writes thousands of \r-updates that otherwise bury the one
# line that says what went wrong.
#
# BOTH streams, and that is not belt-and-braces. vLLM splits the two signals
# that matter during startup: the shard counter goes to stderr, while the
# PREFETCH percentage -- the only progress indicator while the checkpoint is
# being pulled into page cache -- goes to stdout. The shard counter sits at
# `0/N` by design for the whole prefetch phase, so a reader with only stderr
# sees a frozen counter and concludes the job is wedged. That mistake was made
# on this very run; see qwen3-coder-30b-bf16-FAILED/findings.md.
if [ -n "$PY" ] && [ -n "$(ls "$out"/aux-agent/*.json 2>/dev/null)" ]; then
    "$PY" "$repo_dir/tools/aux_transcript.py" "$out"/aux-agent/*.json \
        > "$out/summary.txt" 2>&1 || true
fi

echo "collect_run: wrote $out"
ls -la "$out"
cat >&2 <<EOF

collect_run: next steps
  1. write $out/findings.md -- what the run showed, and what it does NOT show
  2. skim the files above: they carry absolute paths and the task prompt
  3. git add results/$label && git commit
EOF
