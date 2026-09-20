#!/usr/bin/env python
"""Collect a SWE-CI A/B run (control vs foresight) into ``results/<label>/``.

``tools/collect_run.sh`` does this for a Slurm run directory. A SWE-CI run has no
such directory: its evidence is scattered over SWE-CI's ``experiments/`` folder,
foresight's trace, an aux export directory and a few logs. This gathers it in
one command, in one layout, so every run is documented the same way::

    python tools/collect_swe_ci_ab.py --swe-ci-dir ../SWE-CI \\
        --metadata-csv ../SWE-CI/metadata/default.csv \\
        --arm control=<experiment>:config_control.toml \\
        --arm foresight=<experiment>:config_foresight.toml \\
        --trace foresight=traces/<experiment>.jsonl \\
        --aux-export-dir foresight=traces/<experiment>__aux-agent \\
        --foresight-config configs/swe_ci_expt.yaml \\
        --model-url target=http://host:8001/v1 --model-url aux=http://host:8002/v1

The results folder is named from the data, ``results/ab-test__swe-ci__rows-<rows>__ep<epochs>``
(``--label`` overrides it): rows are 1-based data rows of the benchmark csv, so pass
``--metadata-csv`` with the untrimmed csv if ``metadata/default.csv`` was cut down to
your tasks; epochs are ``evolve.max_epoch``. See the main README, "Naming conventions".

Only the standard library is required. ``matplotlib`` (plots) and SWE-CI's own
python (the ``swe_ci.summarize`` table and the code-quality scores) are used when
present and skipped, with a note in ``RUN.md``, when not. Scoring runs pylint on every
epoch snapshot (about a minute for a large repo); ``--no-pylint`` keeps only the
maintainability index, ``--no-score`` skips both.

Layout written under ``results/<label>/``::

    RUN.md                        generated: run summary, results, code change and
                                  quality per epoch, timing, aux provenance, incidents,
                                  cost. findings.md stays hand-written.
    manifest.json                 rows, each task's commits (from the benchmark csv), settings,
                                  and whether the arms' settings matched
    metrics.csv                   per arm, task and epoch end: gap, passed, lines changed
                                  (this epoch and cumulative), maintainability index, pylint
    environment.txt               commits, versions, hardware, model endpoints
    server-side/                  for the people who ran the model servers, to fill in AFTER this
                                  collection (vLLM flags, sampling, partition, job ids, logs).
                                  Created with a checklist if absent, never overwritten
    configs/                      SWE-CI configs and the foresight config (keys redacted)
    plots/                        gap per epoch, EvoScore per task
    data/<arm>/main.log           SWE-CI's main log
    data/<arm>/swe_ci_stdout.log  console output of swe_ci.evaluate (--stdout-log): the effective
                                  config and final table; a crash traceback would be here
    data/<arm>/summary.txt        the swe_ci.summarize table
    data/<arm>/<task>/iteration.jsonl, task.log
    data/<arm>/<task>/epoch_<N>/  what the architect saw and what the model did:
        non-passed/summary.jsonl    the failing-test list at the START of epoch N
        requirement.xml             the architect's output for epoch N
        edit.diff                   the code change made in epoch N (a/ = start, b/ = end)
    data/<arm>/<task>/final/non-passed/summary.jsonl
                                  the failing tests AFTER the last epoch (the last accepted state)
    data/<arm>/trace.jsonl.gz     foresight's trace (arms that have one); rows carry the target
                                  model's reply when the run set trace.log_replies
    data/<arm>/aux_sessions.json  one entry per aux run: task, epoch, phase, the
                                  aux answer, provenance, exported session ids
    data/<arm>/AUX_PAIRS.md       each aux answer next to its base task
    data/<arm>/aux-agent/         ses_*.json, opencode's own record of each aux run

What "the base task" means. A SWE-CI task is a repo evolved from ``current_sha`` until the
tests of ``target_sha`` pass; each epoch is one architect step and one programmer step.
Aux is asked about plausible FUTURE tasks after the CURRENT step, so its base task is the
step's input, not the whole goal: for an architect session the failing tests, for a
programmer session that epoch's ``requirement.xml``. The commits give the long-run goal
and are in ``manifest.json``.

Code quality. ``mi`` is SWE-CI's own ``mi_score`` (radon maintainability index). ``pylint``
is a corrected pylint run: SWE-CI's ``pylint_score`` never passes ``--recursive=y`` and
returns 0 on these repos. Scores exclude ``tests/`` and describe the state AFTER each epoch;
an epoch whose pytest could not run is not accepted (SWE-CI keeps the previous code), so its
state repeats the previous one. Only the change over epochs is meaningful, not the absolute
value. See ``tools/swe_ci_score_helper.py``.

Read ``requirement.xml`` with care: it is the architect's OUTPUT. In the foresight
arm it was written after foresight enriched the prompt with aux's future-task
list, so it depends on aux, not the other way round. The base task of an
architect session is the failing-test summary; the base task of a programmer
session is that epoch's ``requirement.xml``.

How the epoch folders map. SWE-CI archives ``current/`` into a folder named for
the END of each epoch, and that folder holds the epoch's STARTING state (code,
``non-passed/``, ``requirement.xml``). The state after epoch N is the starting
state of epoch N+1, or the final folder (named a minute in the future) after the
last epoch. An epoch whose pytest could not run archives ``tmp/`` instead, which
holds the broken code; its starting state is then the final folder.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ARCHIVE_RE = re.compile(r"^\d{4}-\d\d-\d\d-\d\d-\d\d-\d\d$")
LOG_TS = "%Y-%m-%d %H:%M:%S"
_TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
EPOCH_RE = re.compile(rf"^{_TS} \| \w+ \| \S+ \| =+Epoch (\d+)=+")
STEP6_RE = re.compile(rf"^{_TS} \| \w+ \| \S+ \| \(6/7\) .*?Archived directory '(current|tmp)'")
INCIDENT_RE = re.compile(r"\| (WARNING|ERROR) \||\(Attempt [23]/3\)")

#: Matching an archive folder to a log time. Folder names come from the same
#: clock read as the log line, so they differ by at most a second or two.
ARCHIVE_TOLERANCE_S = 3.0
#: The final folder is named 60 s ahead of the moment it was made.
FINAL_MIN_LEAD_S = 20.0
#: An epoch diff bigger than this is truncated, with a note; a reformat of a
#: whole repository is not evidence anyone will read.
DIFF_CAP_BYTES = 2_000_000

_SECRET_RE = re.compile(
    r"""(?im)^(\s*(?:api_key|hf_token|token|secret|password)\s*[:=]\s*)(["']?)"""
    r"""(?!dummy\b|none\b|\*\*\*)([^"'\s#]+)\2"""
)


# -- small helpers -----------------------------------------------------------


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def redact(text: str) -> str:
    """Config text with literal secrets masked. `dummy`/`none` are placeholders."""
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***{m.group(2)}", text)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def parse_kv(values: list[str], what: str) -> dict[str, str]:
    out = {}
    for value in values or []:
        name, sep, rest = value.partition("=")
        if not sep or not name or not rest:
            raise SystemExit(f"{what} expects NAME=VALUE, got {value!r}")
        out[name] = rest
    return out


def run(argv: list[str], cwd: Path | None = None, timeout: float = 60.0) -> str:
    try:
        done = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


def local_naive(iso_utc: str, tz_offset_hours: float | None) -> dt.datetime:
    """A trace timestamp (UTC, ``...Z``) as the naive local time SWE-CI logs in."""
    moment = dt.datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    if tz_offset_hours is not None:
        return moment.replace(tzinfo=None) + dt.timedelta(hours=tz_offset_hours)
    return moment.astimezone().replace(tzinfo=None)


# -- reading one SWE-CI task -------------------------------------------------


def parse_task_log(path: Path) -> list[dict]:
    """``[{epoch, start, end, broken}]``. ``end`` is None for an unfinished epoch."""
    epochs: list[dict] = []
    current = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = EPOCH_RE.match(line)
        if m:
            current = {
                "epoch": int(m.group(2)),
                "start": dt.datetime.strptime(m.group(1), LOG_TS),
                "end": None,
                "broken": False,
            }
            epochs.append(current)
            continue
        m = STEP6_RE.match(line)
        if m and current is not None and current["end"] is None:
            current["end"] = dt.datetime.strptime(m.group(1), LOG_TS)
            current["broken"] = m.group(2) == "tmp"
    return epochs


def archives(task_dir: Path) -> list[tuple[dt.datetime, Path]]:
    found = []
    for path in task_dir.iterdir():
        if path.is_dir() and ARCHIVE_RE.match(path.name):
            found.append((dt.datetime.strptime(path.name, "%Y-%m-%d-%H-%M-%S"), path))
    return sorted(found)


def nearest(found: list[tuple[dt.datetime, Path]], when: dt.datetime) -> Path | None:
    best, best_gap = None, None
    for stamp, path in found:
        gap = abs((stamp - when).total_seconds())
        if gap <= ARCHIVE_TOLERANCE_S and (best_gap is None or gap < best_gap):
            best, best_gap = path, gap
    return best


def final_folder(found: list[tuple[dt.datetime, Path]], finished: list[dict]) -> Path | None:
    """The state after the last accepted epoch: SWE-CI archives ``current/`` a minute ahead at the end."""
    if not found or not finished:
        return None
    last_end = max(e["end"] for e in finished)
    if (found[-1][0] - last_end).total_seconds() >= FINAL_MIN_LEAD_S:
        return found[-1][1]
    return None


def epoch_states(task_dir: Path, epochs: list[dict]) -> dict[int, dict]:
    """``{epoch: {"start": dir|None, "result": dir|None}}``. See the module docstring."""
    found = archives(task_dir)
    finished = [e for e in epochs if e["end"] is not None]
    final = final_folder(found, finished)
    by_number = {e["epoch"]: e for e in finished}
    states: dict[int, dict] = {}
    for number, epoch in by_number.items():
        at_end = nearest(found, epoch["end"])
        if epoch["broken"]:
            states[number] = {"start": final, "result": at_end}
            continue
        following = by_number.get(number + 1)
        if following is not None and not following["broken"]:
            result = nearest(found, following["end"])
        else:
            result = final
        states[number] = {"start": at_end, "result": result}
    return states


def code_diff(start: Path, result: Path) -> str:
    """``diff -ruN`` of the two ``code/`` trees, with ``a/`` and ``b/`` prefixes."""
    a, b = start / "code", result / "code"
    if not a.is_dir() or not b.is_dir():
        return ""
    done = subprocess.run(
        # "._*" are macOS AppleDouble sidecars: on an external (non-HFS) volume, moving a
        # folder makes macOS write one beside every file, and they are not code.
        ["diff", "-ruN", "-x", ".git", "-x", "__pycache__", "-x", "*.pyc", "-x", "._*", str(a), str(b)],
        capture_output=True,
        text=True,
        errors="replace",
    )
    text = done.stdout.replace(str(a) + "/", "a/").replace(str(b) + "/", "b/")
    if len(text) > DIFF_CAP_BYTES:
        text = text[:DIFF_CAP_BYTES] + f"\n\n[truncated: diff exceeded {DIFF_CAP_BYTES} bytes]\n"
    return text


# -- collecting --------------------------------------------------------------


def collect_arm(
    arm: str, experiment: str, swe_ci_dir: Path, out: Path, *, with_tracebacks: bool
) -> dict:
    """Copy one arm's SWE-CI evidence. Returns ``{tasks: {task_id: [epoch, ...]}}``."""
    src = swe_ci_dir / "experiments" / experiment
    if not src.is_dir():
        raise SystemExit(f"no experiment folder {src}")
    dest = out / "data" / arm
    dest.mkdir(parents=True, exist_ok=True)
    if (src / "main.log").is_file():
        shutil.copy2(src / "main.log", dest / "main.log")
    tasks: dict[str, list[dict]] = {}
    all_states: dict[str, dict] = {}
    for task_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        log = task_dir / "task.log"
        if not log.is_file():
            continue
        tdest = dest / task_dir.name
        tdest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, tdest / "task.log")
        if (task_dir / "iteration.jsonl").is_file():
            shutil.copy2(task_dir / "iteration.jsonl", tdest / "iteration.jsonl")
        epochs = parse_task_log(log)
        tasks[task_dir.name] = epochs
        states = epoch_states(task_dir, epochs) if archives(task_dir) else {}
        all_states[task_dir.name] = states
        # The failing tests AFTER the last epoch: the epoch folders hold the state at the
        # START of each epoch, so without this the end state would be unrecorded. After
        # an epoch that could not run pytest it is the last accepted state.
        final = final_folder(archives(task_dir), [e for e in epochs if e["end"] is not None])
        if final is not None and (final / "non-passed").is_dir():
            (tdest / "final" / "non-passed").mkdir(parents=True, exist_ok=True)
            summary = final / "non-passed" / "summary.jsonl"
            if summary.is_file():
                shutil.copy2(summary, tdest / "final" / "non-passed" / "summary.jsonl")
            if with_tracebacks:
                for extra in (final / "non-passed").iterdir():
                    if extra.is_file() and extra.name != "summary.jsonl":
                        shutil.copy2(extra, tdest / "final" / "non-passed" / extra.name)
        for number, state in states.items():
            edest = tdest / f"epoch_{number}"
            edest.mkdir(exist_ok=True)
            start, result = state["start"], state["result"]
            if start is not None:
                req = start / "requirement.xml"
                if req.is_file():
                    shutil.copy2(req, edest / "requirement.xml")
                nonpassed = start / "non-passed"
                if nonpassed.is_dir():
                    (edest / "non-passed").mkdir(exist_ok=True)
                    if (nonpassed / "summary.jsonl").is_file():
                        shutil.copy2(nonpassed / "summary.jsonl", edest / "non-passed" / "summary.jsonl")
                    if with_tracebacks:
                        for extra in nonpassed.iterdir():
                            if extra.is_file() and extra.name != "summary.jsonl":
                                shutil.copy2(extra, edest / "non-passed" / extra.name)
            if start is not None and result is not None:
                (edest / "edit.diff").write_text(code_diff(start, result), encoding="utf-8")
    return {"tasks": tasks, "states": all_states}


def copy_summary(
    arm: str, experiment: str, config_file: str | None, swe_ci_dir: Path, python: str | None,
    out: Path, notes: list[str],
) -> None:
    """The ``swe_ci.summarize`` table, via SWE-CI's own python."""
    if not config_file or not python or not Path(python).exists():
        notes.append(f"{arm}: swe_ci.summarize was not run (no SWE-CI python or config given)")
        return
    config_text = (swe_ci_dir / config_file).read_text(encoding="utf-8", errors="replace")
    m = re.search(r'(?m)^splitting\s*=\s*"([^"]+)"', config_text)
    splitting = m.group(1) if m else "default"
    env = dict(os.environ, PYTHONPATH="src")
    done = subprocess.run(
        [python, "-m", "swe_ci.summarize", "--config_file", config_file,
         "--experiment_name", experiment, "--splitting", splitting],
        cwd=swe_ci_dir, env=env, capture_output=True, text=True, timeout=300, errors="replace",
    )
    text = strip_ansi(done.stdout + done.stderr)
    if "EVOSCORE" not in text:
        notes.append(f"{arm}: swe_ci.summarize produced no table (exit {done.returncode})")
        return
    (out / "data" / arm / "summary.txt").write_text(text, encoding="utf-8")


def parse_summary(path: Path, task_ids: list[str]) -> dict:
    """``{task_id: EvoScore, ..., "AVERAGE": ..., "_zero_reg": ...}`` from summary.txt."""
    scores: dict = {}
    if not path.is_file():
        return scores
    row = re.compile(
        r"^\s*[║|]?\s*(\S+?)…?\s*[│|]\s*(-?\d+\.\d+)\s*[│|]\s*(-?\d+\.\d+)\s*[│|]\s*(-?\d+\.\d+)"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = row.match(line)
        if not m:
            continue
        name = m.group(1)
        if name == "AVERAGE":
            scores["AVERAGE"] = float(m.group(2))
            scores["_resolved"] = float(m.group(3))
            scores["_zero_reg"] = float(m.group(4))
            continue
        for tid in task_ids:
            if tid.startswith(name):
                scores[tid] = float(m.group(2))
    return scores


def gap_series(path: Path) -> list[int | None]:
    """Gap per line of iteration.jsonl; None when pytest did not run that epoch."""
    gaps: list[int | None] = []
    for row in read_jsonl(path):
        valid = row.get("pytest", {}).get("passed") is not None and row.get("gap", -1) >= 0
        gaps.append(row["gap"] if valid else None)
    return gaps


def collect_trace(
    arm: str, trace: Path, tasks: dict[str, list[dict]], out: Path, tz: float | None
) -> dict:
    """Gzip the trace, and write one joined record per aux run."""
    dest = out / "data" / arm
    with trace.open("rb") as src, gzip.open(dest / "trace.jsonl.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    rows = read_jsonl(trace)
    windows = []
    for task_id, epochs in tasks.items():
        for e in epochs:
            if e["end"] is not None:
                windows.append((e["start"], e["end"] + dt.timedelta(seconds=2), task_id, e["epoch"]))
    sessions = []
    for row in sorted(rows, key=lambda r: r.get("received_at", "")):
        if not (row.get("role") == "target" and row.get("is_session_start") and row.get("phase")):
            continue
        aux = row.get("aux") or {}
        when = local_naive(row["received_at"], tz)
        # Exactly one window, or nothing. Two tasks running at once (SWE-CI
        # with max_workers > 1) put a session inside two windows, and picking
        # either would attach the aux answer to a task it was not written for --
        # worse than leaving it unjoined and saying why.
        matches = [(tid, number) for start, end, tid, number in windows if start <= when <= end]
        task_id, epoch = matches[0] if len(matches) == 1 else (None, None)
        provenance = dict(aux.get("provenance") or {})
        provenance.pop("stdout_tail", None)
        sessions.append({
            "received_at": row["received_at"],
            "task_id": task_id,
            "epoch": epoch,
            "join": "ok" if task_id else ("ambiguous" if len(matches) > 1 else "no epoch window"),
            "phase": row["phase"],
            "session_key": row.get("session_key"),
            "aux_text": aux.get("text"),
            "usable": aux.get("usable"),
            "items": aux.get("items"),
            "provenance": provenance,
            "guard": row.get("guard"),
            "prompt_in_chars": row.get("prompt_in_chars"),
            "prompt_out_chars": row.get("prompt_out_chars"),
        })
    (dest / "aux_sessions.json").write_text(json.dumps(sessions, indent=1), encoding="utf-8")
    tokens = {"aux": Counter(), "target": Counter()}
    for row in rows:
        role = row.get("role")
        upstream = (row.get("usage") or {}).get("upstream") or {}
        if role in tokens:
            tokens[role]["calls"] += 1
            tokens[role]["prompt_tokens"] += upstream.get("prompt_tokens") or 0
            tokens[role]["completion_tokens"] += upstream.get("completion_tokens") or 0
    target_rows = [r for r in rows if r.get("role") == "target"]
    return {"sessions": sessions, "tokens": tokens, "rows": len(rows),
            "target_rows": len(target_rows), "replies": sum(1 for r in target_rows if "reply" in r)}


SERVER_SIDE_TEMPLATE = """# Server side: what to add to this run

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
"""


def ensure_server_side(out: Path) -> dict:
    """Create ``server-side/`` with a checklist if it is absent, and report what it holds.

    Written only when the folder does not exist, and never touched afterwards: the
    server side adds files here after the caller side has collected, and a re-run
    of this script must not lose them.
    """
    folder = out / "server-side"
    if not folder.exists():
        folder.mkdir(parents=True)
        (folder / "TEMPLATE.md").write_text(SERVER_SIDE_TEMPLATE, encoding="utf-8")
    files = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name != "TEMPLATE.md")
    return {"filled": bool(files), "files": files}


def copy_stdout_log(arm: str, path: Path, out: Path) -> bool:
    """The console output of ``swe_ci.evaluate`` (nohup's file): SWE-CI's effective config
    (secrets already shown as ``***``) and the final summary table. It is NOT the agent's
    log. Small, and the only place a crash of the evaluate process itself would show."""
    if not path.is_file():
        return False
    dest = out / "data" / arm
    dest.mkdir(parents=True, exist_ok=True)
    text = strip_ansi(path.read_text(encoding="utf-8", errors="replace"))
    (dest / "swe_ci_stdout.log").write_text(redact(text), encoding="utf-8")
    return True


def copy_aux_export(arm: str, export_dir: Path, out: Path) -> int:
    dest = out / "data" / arm / "aux-agent"
    count = 0
    for path in sorted(export_dir.glob("ses_*")):
        if path.suffix in {".json", ".txt"}:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)
            count += 1
    return count


def copy_configs(swe_ci_dir: Path, arms: dict, foresight_config: Path | None, out: Path) -> None:
    dest = out / "configs"
    dest.mkdir(parents=True, exist_ok=True)
    for spec in arms.values():
        config = spec.get("config")
        if config and (swe_ci_dir / config).is_file():
            text = (swe_ci_dir / config).read_text(encoding="utf-8", errors="replace")
            (dest / Path(config).name).write_text(redact(text), encoding="utf-8")
    if foresight_config and foresight_config.is_file():
        text = foresight_config.read_text(encoding="utf-8", errors="replace")
        (dest / foresight_config.name).write_text(redact(text), encoding="utf-8")


def describe_repo(path: Path, rev_override: str | None = None) -> str:
    if rev_override:
        # A revision given by hand describes the run's tree, not this checkout,
        # so this checkout's branch and dirty flag would be about the wrong thing.
        return f"{rev_override} (given, not read from the tree)"
    rev = run(["git", "-C", str(path), "rev-parse", "HEAD"])
    branch = run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"])
    return f"{rev or 'unknown'}  branch={branch or '?'}  tracked-changes={'yes' if dirty else 'no'}"


def model_info(url: str) -> str:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as reply:
            data = json.load(reply)["data"][0]
        return (f"served={data.get('id')} root={data.get('root')} "
                f"max_model_len={data.get('max_model_len')}")
    except Exception as exc:  # noqa: BLE001 -- an unreachable endpoint is a note, not a failure
        return f"unreachable ({type(exc).__name__})"


def write_environment(
    out: Path, swe_ci_dir: Path, urls: dict[str, str], foresight_rev: str | None, argv: list[str]
) -> None:
    lines = [
        f"collected_at        {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"collector_command   {' '.join(argv)}",
        f"foresight           {describe_repo(REPO, foresight_rev)}",
        f"swe_ci              {describe_repo(swe_ci_dir)}",
        f"python (collector)  {platform.python_version()}",
        f"os                  {platform.platform()}",
        f"machine             {platform.machine()}",
    ]
    docker = run(["docker", "version", "--format", "{{.Server.Version}}"])
    info = run(["docker", "info", "--format", "cpus={{.NCPU}} mem_bytes={{.MemTotal}}"])
    lines.append(f"docker              {docker or 'unavailable'}  {info}")
    # The model servers are Slurm jobs that expire, so a later collection often finds
    # them gone. Losing the served model, checkpoint and context length to
    # "unreachable" would erase the one record of what was actually served, so an
    # earlier good line for the same endpoint is kept and labelled as such.
    previous = {}
    if (out / "environment.txt").is_file():
        for line in (out / "environment.txt").read_text(encoding="utf-8").splitlines():
            if line.startswith("model[") and "served=" in line:      # a line that carries real info
                previous[line.split(maxsplit=1)[0]] = line
    for name, url in urls.items():
        key, info = f"model[{name}]", model_info(url)
        if info.startswith("unreachable") and key in previous:
            lines.append(previous[key].split(" (kept from")[0] + "  (kept from an earlier collection; "
                         "unreachable now)")
        else:
            lines.append(key.ljust(20) + f"{url}  {info}")
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- manifest, churn and maintainability ------------------------------------

#: Per-arm plumbing. Everything else in a SWE-CI config must match between arms
#: for an A/B to mean anything, and the manifest checks that.
ARM_SPECIFIC_KEYS = {"experiment_name", "base_url", "model_name", "api_key", "hf_token"}
#: The settings copied into the manifest for a reader, in a stable order.
SETTING_KEYS = (
    "agent_name", "mode", "splitting", "evolve.max_epoch", "evolve.max_workers", "init.max_workers",
    "evolve.architect.max_try", "evolve.programmer.max_try", "evolve.architect.timeout",
    "evolve.programmer.timeout", "pytest.timeout",
)
#: Run naming: ``ab-test__swe-ci__rows-<rows>__ep<epochs>``. Rows are 1-based data
#: rows of the benchmark's ``metadata/<splitting>.csv``.
LABEL_TEMPLATE = "ab-test__swe-ci__rows-{rows}__ep{epochs}"

METRIC_COLUMNS = [
    "arm", "task_id", "epoch", "accepted", "gap", "passed", "edit_files", "edit_added",
    "edit_removed", "edit_test_files", "cum_added", "cum_removed", "cum_lines", "mi", "pylint",
]


def read_config_values(text: str) -> dict[str, str]:
    """A SWE-CI ``config.toml`` as flat ``section.key -> text``. Scalars only, which is all it has."""
    section, values = "", {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^\[([^\]]+)\]$", line)
        if m:
            section = m.group(1).strip()
            continue
        m = re.match(r"^([A-Za-z_][\w\-]*)\s*=\s*(.+)$", line)
        if m:
            value = m.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[f"{section}.{m.group(1)}" if section else m.group(1)] = value
    return values


def load_task_rows(csv_path: Path) -> dict[str, dict]:
    """``{task_id: {row, repo, url, current_sha, target_sha, test_gap}}``; ``row`` is 1-based."""
    rows: dict[str, dict] = {}
    if not csv_path.is_file():
        return rows
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for index, record in enumerate(csv.DictReader(handle), start=1):
            task_id = record.get("task_id")
            if task_id:
                rows[task_id] = {
                    "row": index, "repo": record.get("repo_name"), "url": record.get("url"),
                    "current_sha": record.get("current_sha"), "target_sha": record.get("target_sha"),
                    "test_gap": record.get("test_gap"),
                }
    return rows


def range_str(rows: list[int]) -> str:
    """``[6,7,8,9,10]`` -> ``6-10``; ``[1,2,5]`` -> ``1-2_5``."""
    rows = sorted(set(rows))
    parts, start, prev = [], rows[0], rows[0]
    for row in rows[1:] + [None]:
        if row is not None and row == prev + 1:
            prev = row
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        if row is not None:
            start = prev = row
    return "_".join(parts)


def discover_tasks(swe_ci_dir: Path, experiment: str) -> list[str]:
    src = swe_ci_dir / "experiments" / experiment
    if not src.is_dir():
        return []
    return sorted(p.name for p in src.iterdir() if p.is_dir() and (p / "task.log").is_file())


def build_manifest(
    label: str, arms: dict, collected: dict, task_rows: dict, config_values: dict, csv_name: str,
    foresight_rev: str | None, swe_ci_dir: Path, argv: list[str], server_side: dict | None = None,
) -> dict:
    task_ids = sorted({t for c in collected.values() for t in c["tasks"]},
                      key=lambda t: (task_rows.get(t, {}).get("row") or 10**9, t))
    tasks = [{"task_id": t, **task_rows.get(t, {"row": None})} for t in task_ids]
    rows = [t["row"] for t in tasks if t["row"]]
    settings = {a: {k: v for k, v in config_values[a].items() if k in SETTING_KEYS}
                for a in arms if a in config_values}
    differences: dict[str, dict] = {}
    if len(settings) > 1:
        comparable = {a: {k: v for k, v in config_values[a].items() if k not in ARM_SPECIFIC_KEYS}
                      for a in settings}
        for key in sorted(set().union(*[set(c) for c in comparable.values()])):
            values = {a: comparable[a].get(key) for a in comparable}
            if len(set(values.values())) > 1:
                differences[key] = values
    first = next(iter(settings.values()), {})
    return {
        "label": label,
        "naming": "ab-test__swe-ci__rows-<rows>__ep<epochs>; rows are 1-based data rows of metadata/<splitting>.csv",
        "collected_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "splitting": first.get("splitting"),
        "metadata_csv": csv_name,
        "rows": sorted(rows),
        "rows_range": range_str(rows) if rows else None,
        "max_epoch": int(first["evolve.max_epoch"]) if first.get("evolve.max_epoch", "").isdigit() else None,
        "arms": {a: {"experiment": spec["experiment"], "config": spec["config"]} for a, spec in arms.items()},
        "settings": settings,
        "settings_identical_across_arms": (not differences) if len(settings) > 1 else None,
        "settings_differences": differences,
        "tasks": tasks,
        "foresight_rev": foresight_rev or run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "swe_ci_rev": run(["git", "-C", str(swe_ci_dir), "rev-parse", "HEAD"]),
        "server_side": server_side or {"filled": False, "files": []},
        "collector_command": ["collect_swe_ci_ab.py"] + argv,
    }


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (bool({"tests", "test", "testing"} & set(parts[:-1]))
            or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def diffstat(text: str) -> dict:
    """Files changed and lines added/removed in a ``diff -ruN``, counted inside hunks only.

    Hunk-aware on purpose: a removed line whose text starts with ``--`` appears as
    ``---...`` and would be mistaken for a file header by a line-prefix count.
    """
    files: dict[str, list[int]] = {}
    current, in_hunk = None, False
    for line in text.splitlines():
        if line.startswith("diff "):
            current, in_hunk = line.rsplit(" b/", 1)[-1] if " b/" in line else line, False
            files[current] = [0, 0]
        elif line.startswith("@@") and current is not None:
            in_hunk = True
        elif in_hunk and current is not None:
            if line.startswith("+"):
                files[current][0] += 1
            elif line.startswith("-"):
                files[current][1] += 1
    tests = [f for f in files if is_test_path(f)]
    return {
        "files": len(files),
        "added": sum(a for a, _ in files.values()),
        "removed": sum(r for _, r in files.values()),
        "test_files": len(tests),
    }


def build_metrics(arm: str, task_id: str, out: Path, epochs: list[dict], states: dict, scores: dict) -> list[dict]:
    """One row per epoch end (0 = the starting state): churn, gap, and code-quality scores.

    Churn is the size of the edit made IN that epoch. The scores describe the state
    AFTER it. An epoch whose pytest could not run is not accepted (SWE-CI keeps the
    previous code), so its state is the previous one; its attempted edit is still
    counted in ``edit_*`` but not in ``cum_*``.
    """
    task_out = out / "data" / arm / task_id
    iteration = read_jsonl(task_out / "iteration.jsonl") if (task_out / "iteration.jsonl").is_file() else []
    broken = {e["epoch"]: e["broken"] for e in epochs}

    def gap_passed(n: int):
        row = iteration[n] if n < len(iteration) else None
        if row and row.get("pytest", {}).get("passed") is not None and row.get("gap", -1) >= 0:
            return row["gap"], row["pytest"]["passed"]
        return None, None

    rows: list[dict] = []
    first_start = (states.get(1) or {}).get("start")
    if first_start is None:
        return rows
    state_dir = first_start
    gap, passed = gap_passed(0)
    cum_added = cum_removed = 0

    def make_row(epoch, accepted, edit, dir_):
        sc = scores.get(dir_) or {}
        return {
            "arm": arm, "task_id": task_id, "epoch": epoch, "accepted": accepted, "gap": gap,
            "passed": passed, "edit_files": edit.get("files"), "edit_added": edit.get("added"),
            "edit_removed": edit.get("removed"), "edit_test_files": edit.get("test_files"),
            "cum_added": cum_added, "cum_removed": cum_removed, "cum_lines": cum_added + cum_removed,
            "mi": sc.get("mi"), "pylint": sc.get("pylint"),
        }

    rows.append(make_row(0, True, {}, state_dir))
    for number in sorted(states):
        accepted = not broken.get(number, False)
        diff_path = task_out / f"epoch_{number}" / "edit.diff"
        edit = diffstat(diff_path.read_text(encoding="utf-8", errors="replace")) if diff_path.is_file() else {}
        if accepted:
            cum_added += edit.get("added", 0)
            cum_removed += edit.get("removed", 0)
            state_dir = states[number]["result"] or state_dir
        gap, passed = gap_passed(number)
        rows.append(make_row(number, accepted, edit, state_dir))
    return rows


def score_snapshots(
    dirs: list[Path], python: str, helper: Path, score_py: Path, exclude: list[str],
    pylint: bool, timeout: float, jobs: int,
) -> dict[Path, dict]:
    """Run ``swe_ci_score_helper.py`` on each snapshot's ``code/`` under SWE-CI's python."""
    def one(directory: Path):
        command = [python, str(helper), str(score_py), str(directory / "code")]
        for item in exclude:
            command += ["--exclude", item]
        if not pylint:
            command.append("--no-pylint")
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, errors="replace")
        except subprocess.TimeoutExpired:
            return directory, {"error": f"timed out after {timeout:.0f}s"}
        except OSError as exc:
            return directory, {"error": str(exc)}
        for line in reversed(done.stdout.strip().splitlines()):
            try:
                return directory, json.loads(line)
            except ValueError:
                continue
        return directory, {"error": ((done.stderr or done.stdout).strip()[-200:]) or f"exit {done.returncode}"}

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return dict(pool.map(one, dirs))


def write_metrics_csv(out: Path, rows: list[dict]) -> None:
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in METRIC_COLUMNS})


def write_aux_pairs(out: Path, arm: str, sessions: list[dict], task_info: dict) -> None:
    """Each aux answer next to the base task it was written for.

    Architect session: the base task is the failing-test list at the start of the
    epoch. Programmer session: it is that epoch's ``requirement.xml``. The task's
    commits give the long-run goal (evolve from ``current_sha`` until the tests of
    ``target_sha`` pass); the aux tasks are about what could plausibly follow the
    CURRENT step, not about that whole goal.
    """
    lines = [
        f"# Aux answers paired with their base task ({arm})", "",
        "Architect session: base task = failing tests at the start of the epoch. Programmer session: "
        "base task = that epoch's `requirement.xml`. Generated by `tools/collect_swe_ci_ab.py`.", "",
    ]
    for s in sessions:
        task_id, epoch = s.get("task_id"), s.get("epoch")
        if not task_id:
            lines += [f"## unjoined session at {s.get('received_at')} ({s.get('phase')}): {s.get('join')}", ""]
            continue
        info = task_info.get(task_id, {})
        lines += [f"## {short(task_id)}, epoch {epoch}, {s['phase']}", ""]
        if info.get("current_sha"):
            lines.append(f"Task goal: evolve `{info.get('repo')}` from `{info['current_sha'][:10]}` "
                         f"until the tests of `{(info.get('target_sha') or '?')[:10]}` pass.")
        base = out / "data" / arm / task_id / f"epoch_{epoch}"
        if s["phase"] == "architect":
            summary = base / "non-passed" / "summary.jsonl"
            tests = [r.get("test") for r in read_jsonl(summary)] if summary.is_file() else []
            shown = ", ".join(f"`{t}`" for t in tests[:8]) + (f" (+{len(tests) - 8} more)" if len(tests) > 8 else "")
            lines += ["", f"**Base task: {len(tests)} failing test(s):** {shown or 'not available'}"]
        else:
            requirement = base / "requirement.xml"
            lines += ["", "**Base task: requirement.xml**"]
            try:
                for item in ET.parse(requirement).getroot().iter("requirement"):
                    where = (item.findtext("location") or "").strip()
                    what = " ".join((item.findtext("description") or "").split())[:300]
                    lines.append(f"- `{where}`: {what}")
            except (ET.ParseError, OSError):
                lines.append("- not available")
        lines += ["", "**Aux's future tasks:**", ""]
        lines += [f"> {t}" if t else ">" for t in (s.get("aux_text") or "").strip().splitlines()]
        lines.append("")
    (out / "data" / arm / "AUX_PAIRS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cells(rows: list[dict], fmt) -> str:
    return ", ".join(fmt(r) for r in sorted(rows, key=lambda r: r["epoch"]))


def metrics_section(manifest: dict, metrics: list[dict], arms: dict, scored: bool) -> list[str]:
    names = list(arms)
    by = {}
    for row in metrics:
        by.setdefault((row["task_id"], row["arm"]), []).append(row)
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    lines = ["", "## Code change per epoch", "",
             "`+added/-removed` lines of the edit made in each epoch (epoch 1 first). `!` = pytest could not run, "
             "so the edit was not accepted. Cumulative counts are in `metrics.csv`.", "",
             "| Task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def edit_cell(r):
        if r["epoch"] == 0:
            return None
        mark = "" if r["accepted"] else "!"
        return f"+{r['edit_added'] or 0}/-{r['edit_removed'] or 0}{mark}"

    for tid in task_ids:
        cells = []
        for a in names:
            rows = [r for r in by.get((tid, a), []) if r["epoch"] > 0]
            cells.append(_cells(rows, edit_cell) or "-")
        lines.append(f"| {short(tid)} | " + " | ".join(cells) + " |")
    for key, title, digits in (("mi", "Maintainability index after each epoch (0 = start; higher is better)", 1),
                               ("pylint", "pylint note after each epoch (0 = start; 0-10, higher is better)", 2)):
        if not any(r.get(key) is not None for r in metrics):
            continue
        lines += ["", f"## {title}", "", "| Task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for tid in task_ids:
            cells = []
            for a in names:
                cells.append(_cells(by.get((tid, a), []),
                                    lambda r, k=key, d=digits: "-" if r.get(k) is None else f"{r[k]:.{d}f}") or "-")
            lines.append(f"| {short(tid)} | " + " | ".join(cells) + " |")
    if scored:
        lines += ["", "Scores: `mi` is SWE-CI's `mi_score` (radon), `pylint` a corrected pylint run; both exclude "
                      "`tests/`. Only the change between epochs is meaningful, not the absolute value "
                      "(see `tools/swe_ci_score_helper.py`)."]
    return lines


def manifest_section(manifest: dict) -> list[str]:
    lines = ["## Run", "",
             f"- rows {manifest['rows_range'] or '?'} of `{manifest.get('metadata_csv')}`, "
             f"splitting `{manifest.get('splitting')}`, up to {manifest.get('max_epoch')} epochs",
             f"- settings identical across arms: {manifest['settings_identical_across_arms']}"]
    side = manifest.get("server_side") or {}
    lines.append("- server side (`server-side/`, added by whoever ran the model servers): "
                 + (f"{len(side['files'])} file(s): {', '.join(side['files'])}" if side.get("filled")
                    else "not added yet, see `server-side/TEMPLATE.md`"))
    for key, values in manifest["settings_differences"].items():
        lines.append(f"  - DIFFERS `{key}`: {values}")
    lines += ["", "| Row | Task | Repo | current -> target commit |", "|---|---|---|---|"]
    for t in manifest["tasks"]:
        cur, tgt = (t.get("current_sha") or "?")[:10], (t.get("target_sha") or "?")[:10]
        lines.append(f"| {t.get('row') or '?'} | `{short(t['task_id'])}` | {t.get('repo') or '?'} | `{cur}` -> `{tgt}` |")
    lines.append("")
    return lines


# -- report and plots --------------------------------------------------------


def short(task_id: str) -> str:
    parts = task_id.split("__")
    return f"{parts[1]} {parts[2][:6]}" if len(parts) >= 3 else task_id[:24]


def wall_clock(main_log: Path) -> tuple[str, str, str]:
    stamps = []
    for line in main_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(rf"^{_TS} ", line)
        if m:
            stamps.append(dt.datetime.strptime(m.group(1), LOG_TS))
    if len(stamps) < 2:
        return "?", "?", "?"
    return stamps[0].strftime(LOG_TS), stamps[-1].strftime(LOG_TS), str(stamps[-1] - stamps[0])


def build_report(
    label: str, out: Path, arms: dict, collected: dict, traces: dict, exports: dict, notes: list[str],
    manifest: dict, metrics: list[dict], scored: bool,
) -> str:
    names = list(arms)
    task_ids = sorted({t for c in collected.values() for t in c["tasks"]})
    scores = {a: parse_summary(out / "data" / a / "summary.txt", task_ids) for a in names}
    lines = [f"# {label}: generated run report", "",
             "Generated by `tools/collect_swe_ci_ab.py`. The interpretation lives in `findings.md`.", ""]
    lines += manifest_section(manifest)
    lines += ["## Results", "", "Gap per epoch (starting state first; lower is better; `x` = pytest could not run).", ""]
    lines += ["| Task | " + " | ".join(f"{a} gaps" for a in names) + " |",
              "|---|" + "---|" * len(names)]
    for tid in task_ids:
        cells = []
        for a in names:
            path = out / "data" / a / tid / "iteration.jsonl"
            gaps = gap_series(path) if path.is_file() else []
            cells.append(", ".join("x" if g is None else str(g) for g in gaps) or "-")
        lines.append(f"| {short(tid)} | " + " | ".join(cells) + " |")
    if any(scores.values()):
        lines += ["", "EvoScore (`swe_ci.summarize`):", "",
                  "| Task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for tid in task_ids + ["AVERAGE"]:
            vals = [scores[a].get(tid) for a in names]
            lines.append(f"| {short(tid) if tid != 'AVERAGE' else '**AVERAGE**'} | "
                         + " | ".join("-" if v is None else f"{v:.4f}" for v in vals) + " |")
        for a in names:
            if "_zero_reg" in scores[a]:
                lines.append("")
                lines.append(f"{a}: resolved {scores[a]['_resolved']:.2f}, zero-regression rate {scores[a]['_zero_reg']:.2f}")
    lines += metrics_section(manifest, metrics, arms, scored)
    lines += ["", "## Timing (indicative: arms may share model capacity)", "",
              "| Arm | Start | End | Elapsed |", "|---|---|---|---|"]
    for a in names:
        log = out / "data" / a / "main.log"
        s, e, d = wall_clock(log) if log.is_file() else ("?", "?", "?")
        lines.append(f"| {a} | {s} | {e} | {d} |")
    lines += ["", "## SWE-CI cost (from iteration.jsonl)", "",
              "| Arm | architect in/out tokens | programmer in/out tokens | agent seconds |", "|---|---|---|---|"]
    for a in names:
        tot = Counter()
        for tid in collected[a]["tasks"]:
            path = out / "data" / a / tid / "iteration.jsonl"
            for row in read_jsonl(path) if path.is_file() else []:
                for role in ("architect", "programmer"):
                    r = row.get(role) or {}
                    tot[f"{role}_in"] += r.get("input_tokens") or 0
                    tot[f"{role}_out"] += r.get("output_tokens") or 0
                    tot["seconds"] += r.get("execution_time") or 0
        lines.append(f"| {a} | {tot['architect_in']:,} / {tot['architect_out']:,} | "
                     f"{tot['programmer_in']:,} / {tot['programmer_out']:,} | {tot['seconds']:.0f} |")
    for a, info in traces.items():
        sessions = info["sessions"]
        prov = [s["provenance"] for s in sessions]
        lines += ["", f"## Foresight / aux provenance ({a})", "",
                  f"- trace rows: {info['rows']:,}; aux runs: {len(sessions)}; "
                  f"joined to a task and epoch: {sum(1 for s in sessions if s['task_id'])}",
                  f"- resolved_by: {dict(Counter(p.get('resolved_by') for p in prov))}",
                  f"- fallback: {dict(Counter(p.get('fallback') for p in prov))} "
                  "(any `body_only` run must be excluded from analysis)",
                  f"- usable: {dict(Counter(s.get('usable') for s in sessions))}; "
                  f"guard: {dict(Counter((s.get('guard') or {}).get('verdict') for s in sessions))}"]
        durations = [p["duration_s"] for p in prov if p.get("duration_s") is not None]
        if durations:
            lines.append(f"- aux run time: mean {sum(durations)/len(durations):.1f} s, max {max(durations):.1f} s")
        growth = [s["prompt_out_chars"] - s["prompt_in_chars"] for s in sessions
                  if s.get("prompt_in_chars") is not None and s.get("prompt_out_chars") is not None]
        if growth:
            lines.append(f"- prompt growth: mean {sum(growth)/len(growth):.0f} chars, max {max(growth)}")
        for role, tok in info["tokens"].items():
            lines.append(f"- {role} model calls in trace: {tok['calls']:,}, "
                         f"prompt tokens {tok['prompt_tokens']:,}, completion tokens {tok['completion_tokens']:,}")
        if info.get("replies"):
            lines.append(f"- target replies logged: {info['replies']:,} of {info['target_rows']:,} target rows "
                         "(text and tool calls, in the trace)")
        else:
            lines.append("- target replies: not logged in this run (`trace.log_replies` in the foresight config)")
        exported = sum(len(p.get("aux_export", {}).get("sessions", [])) for p in prov)
        errors = [e for p in prov for e in p.get("aux_export", {}).get("errors", [])]
        if any("aux_export" in p for p in prov):
            lines.append(f"- aux transcripts exported: {exported} session(s), {len(errors)} export error(s)")
        elif exports.get(a):
            lines.append(f"- aux transcripts copied: {exports[a]} file(s)")
        else:
            lines.append("- aux transcripts: none recorded for this run (see `adapter.aux_export_dir`)")
    lines += ["", "## Incidents (warnings, errors, retries, early exits from task.log)", ""]
    any_incident = False
    for a in names:
        for tid in collected[a]["tasks"]:
            log = out / "data" / a / tid / "task.log"
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if INCIDENT_RE.search(line):
                    any_incident = True
                    lines.append(f"- `{a}` {short(tid)}: {line[:200]}")
    if not any_incident:
        lines.append("- none")
    if notes:
        lines += ["", "## Collector notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def make_plots(out: Path, arms: dict, collected: dict, notes: list[str], metrics: list[dict] | None = None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        notes.append("plots skipped: matplotlib is not installed")
        return
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    names = list(arms)
    task_ids = sorted({t for c in collected.values() for t in c["tasks"]})
    if not task_ids:
        return
    (out / "plots").mkdir(exist_ok=True)

    def style(ax):
        ax.set_facecolor(surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=muted)

    cols = 3
    rows = (len(task_ids) + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows), facecolor=surface, squeeze=False)
    flat = list(axes.flat)
    for ax, tid in zip(flat, task_ids):
        style(ax)
        for color, arm in zip(palette, names):
            path = out / "data" / arm / tid / "iteration.jsonl"
            gaps = gap_series(path) if path.is_file() else []
            xs = [i for i, g in enumerate(gaps) if g is not None]
            ys = [g for g in gaps if g is not None]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, mec=surface, mew=2, zorder=3)
            if None in gaps and ys:
                ax.plot([gaps.index(None)], [ys[-1]], marker="X", ms=11, color=color, mec=surface, zorder=4)
        ax.set_title(short(tid), loc="left", fontsize=11, fontweight="bold", color=ink)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("epoch (0 = starting state)", fontsize=9, color=muted)
    for ax in flat[len(task_ids):]:
        ax.axis("off")
    legend = [Line2D([0], [0], color=c, lw=2, marker="o", label=a) for c, a in zip(palette, names)]
    legend.append(Line2D([0], [0], color=muted, lw=0, marker="X", ms=10, label="pytest not executed"))
    flat[len(task_ids)].legend(handles=legend, loc="upper left", frameon=False, labelcolor=ink)
    fig.suptitle("Remaining test gap per epoch (lower is better; own y-scale per task)",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "plots" / "results_gap_per_epoch.png", dpi=150, facecolor=surface)
    plt.close(fig)

    def metric_figure(key: str, title: str, filename: str) -> None:
        series: dict = {}
        for row in metrics or []:
            if row.get(key) is not None:
                series.setdefault((row["task_id"], row["arm"]), []).append((row["epoch"], row[key]))
        if not series:
            return
        fig2, axes2 = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows), facecolor=surface, squeeze=False)
        flat2 = list(axes2.flat)
        for ax, tid in zip(flat2, task_ids):
            style(ax)
            for color, arm in zip(palette, names):
                points = sorted(series.get((tid, arm), []))
                if points:
                    ax.plot([x for x, _ in points], [y for _, y in points], color=color, lw=2,
                            marker="o", ms=6, mec=surface, mew=2, zorder=3)
            ax.set_title(short(tid), loc="left", fontsize=11, fontweight="bold", color=ink)
            ax.set_xlabel("epoch (0 = starting state)", fontsize=9, color=muted)
        for ax in flat2[len(task_ids):]:
            ax.axis("off")
        flat2[len(task_ids)].legend(handles=legend[:len(names)], loc="upper left", frameon=False, labelcolor=ink)
        fig2.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=ink)
        fig2.tight_layout(rect=(0, 0, 1, 0.95))
        fig2.savefig(out / "plots" / filename, dpi=150, facecolor=surface)
        plt.close(fig2)

    metric_figure("cum_lines", "Cumulative lines changed (added + removed), state after each epoch",
                  "results_churn_per_epoch.png")
    metric_figure("mi", "Maintainability index after each epoch (higher is better)", "results_mi_per_epoch.png")
    metric_figure("pylint", "pylint note after each epoch (higher is better)", "results_pylint_per_epoch.png")

    scores = {a: parse_summary(out / "data" / a / "summary.txt", task_ids) for a in names}
    if not all(scores.values()):
        notes.append("EvoScore plot skipped: a summary.txt is missing")
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(task_ids) + 3), 5), facecolor=surface)
    style(ax)
    labels = [short(t) for t in task_ids] + ["AVERAGE"]
    width = 0.8 / len(names)
    for k, (color, arm) in enumerate(zip(palette, names)):
        vals = [scores[arm].get(t, 0.0) for t in task_ids] + [scores[arm].get("AVERAGE", 0.0)]
        offset = (k - (len(names) - 1) / 2) * width
        ax.bar([i + offset for i in range(len(vals))], vals, width * 0.95, color=color, label=arm, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i + offset, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color=ink)
    ax.axhline(0, color=muted, lw=1, zorder=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("EvoScore (higher is better)", color=muted)
    ax.legend(frameon=False, loc="upper left", labelcolor=ink)
    ax.set_title("EvoScore per task", loc="left", fontsize=14, fontweight="bold", color=ink, pad=12)
    fig.tight_layout()
    fig.savefig(out / "plots" / "results_evoscore.png", dpi=150, facecolor=surface)
    plt.close(fig)


# -- entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label", default=None,
                   help="results/<label>/. Default: ab-test__swe-ci__rows-<rows>__ep<epochs>, from the data")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--swe-ci-dir", type=Path, default=REPO.parent / "SWE-CI")
    p.add_argument("--arm", action="append", required=True, metavar="NAME=EXPERIMENT[:CONFIG]",
                   help="repeatable; CONFIG is a file name inside --swe-ci-dir")
    p.add_argument("--trace", action="append", default=[], metavar="ARM=PATH")
    p.add_argument("--aux-export-dir", action="append", default=[], metavar="ARM=DIR")
    p.add_argument("--stdout-log", action="append", default=[], metavar="ARM=PATH",
                   help="the nohup output of `swe_ci.evaluate` for that arm (small; keeps a crash traceback)")
    p.add_argument("--foresight-config", type=Path, default=None)
    p.add_argument("--model-url", action="append", default=[], metavar="NAME=URL")
    p.add_argument("--metadata-csv", type=Path, default=None,
                   help="the FULL benchmark csv, for row numbers and commits; default "
                        "<swe-ci-dir>/metadata/<splitting>.csv (pass the untrimmed copy if you trimmed it)")
    p.add_argument("--swe-ci-python", default=None, help="default: <swe-ci-dir>/.venv/bin/python")
    p.add_argument("--foresight-rev", default=None,
                   help="commit the run used, when this checkout has moved on since")
    p.add_argument("--tz-offset-hours", type=float, default=None,
                   help="local time offset from UTC of the machine that ran SWE-CI (default: this machine)")
    p.add_argument("--with-tracebacks", action="store_true", help="also copy each epoch's per-test tracebacks")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-score", action="store_true", help="skip maintainability index and pylint")
    p.add_argument("--no-pylint", action="store_true", help="maintainability index only (much faster)")
    p.add_argument("--score-jobs", type=int, default=2, help="snapshots scored in parallel")
    p.add_argument("--score-timeout", type=float, default=1800.0, help="seconds per snapshot")
    p.add_argument("--score-exclude", action="append", default=None,
                   help="top-level dir to leave out of the scores (default: tests); repeatable")
    args = p.parse_args(argv)

    swe_ci_dir = args.swe_ci_dir.resolve()
    python = args.swe_ci_python or str(swe_ci_dir / ".venv" / "bin" / "python")
    arms: dict[str, dict] = {}
    for value in args.arm:
        name, sep, rest = value.partition("=")
        experiment, _, config = rest.partition(":")
        if not sep or not name or not experiment:
            raise SystemExit(f"--arm expects NAME=EXPERIMENT[:CONFIG], got {value!r}")
        arms[name] = {"experiment": experiment, "config": config or None}
    traces, exports = parse_kv(args.trace, "--trace"), parse_kv(args.aux_export_dir, "--aux-export-dir")
    stdout_logs = parse_kv(args.stdout_log, "--stdout-log")
    for name in list(traces) + list(exports) + list(stdout_logs):
        if name not in arms:
            raise SystemExit(f"--trace/--aux-export-dir/--stdout-log names an arm that is not given: {name!r}")

    # What the run was, before anything is copied: the label is derived from it.
    config_values = {}
    for name, spec in arms.items():
        if spec["config"] and (swe_ci_dir / spec["config"]).is_file():
            config_values[name] = read_config_values(
                (swe_ci_dir / spec["config"]).read_text(encoding="utf-8", errors="replace"))
    first_cfg = next(iter(config_values.values()), {})
    splitting = first_cfg.get("splitting", "default")
    csv_path = args.metadata_csv or (swe_ci_dir / "metadata" / f"{splitting}.csv")
    task_rows = load_task_rows(csv_path)
    all_tasks = sorted({t for spec in arms.values() for t in discover_tasks(swe_ci_dir, spec["experiment"])})
    rows = [task_rows[t]["row"] for t in all_tasks if t in task_rows]
    label = args.label
    if not label:
        max_epoch = first_cfg.get("evolve.max_epoch")
        if not rows or not max_epoch:
            raise SystemExit("cannot derive a label (no matching rows in the csv, or no config with "
                             "evolve.max_epoch): pass --label, and --metadata-csv if the csv was trimmed")
        label = LABEL_TEMPLATE.format(rows=range_str(rows), epochs=max_epoch)
    out = args.out or (REPO / "results" / label)

    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    if all_tasks and len(rows) < len(all_tasks):
        notes.append(f"{len(all_tasks) - len(rows)} task(s) not found in {csv_path.name}: row numbers "
                     "unknown (pass --metadata-csv with the untrimmed benchmark csv)")
    collected, trace_info, export_counts = {}, {}, {}
    for name, spec in arms.items():
        collected[name] = collect_arm(name, spec["experiment"], swe_ci_dir, out,
                                      with_tracebacks=args.with_tracebacks)
        copy_summary(name, spec["experiment"], spec["config"], swe_ci_dir, python, out, notes)
        if name in traces:
            path = Path(traces[name])
            if path.is_file():
                trace_info[name] = collect_trace(name, path, collected[name]["tasks"], out, args.tz_offset_hours)
            else:
                notes.append(f"{name}: trace {path} not found")
        if name in stdout_logs and not copy_stdout_log(name, Path(stdout_logs[name]), out):
            notes.append(f"{name}: stdout log {stdout_logs[name]} not found")
        if name in exports:
            path = Path(exports[name])
            export_counts[name] = copy_aux_export(name, path, out) if path.is_dir() else 0
            if not path.is_dir():
                notes.append(f"{name}: aux export dir {path} not found")
    copy_configs(swe_ci_dir, arms, args.foresight_config, out)
    write_environment(out, swe_ci_dir, parse_kv(args.model_url, "--model-url"), args.foresight_rev,
                      ["collect_swe_ci_ab.py"] + argv)

    manifest = build_manifest(label, arms, collected, task_rows, config_values, csv_path.name,
                              args.foresight_rev, swe_ci_dir, argv, ensure_server_side(out))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if manifest["settings_identical_across_arms"] is False:
        notes.append("the arms' SWE-CI settings differ beyond experiment name, base_url and model: "
                     f"{sorted(manifest['settings_differences'])}")

    # Per-epoch code change and code quality, from the epoch archive folders.
    scores: dict[Path, dict] = {}
    score_py = swe_ci_dir / "src" / "swe_ci" / "benchmark" / "utils" / "score.py"
    scored = False
    if not args.no_score:
        dirs = sorted({d for c in collected.values() for st in c["states"].values()
                       for e in st.values() for d in (e["start"], e["result"]) if d is not None})
        if Path(python).exists() and score_py.is_file() and dirs:
            scores = score_snapshots(dirs, python, REPO / "tools" / "swe_ci_score_helper.py", score_py,
                                     args.score_exclude or ["tests"], not args.no_pylint,
                                     args.score_timeout, args.score_jobs)
            scored = True
            failed = [d.parent.name + "/" + d.name for d, r in scores.items() if r.get("error")]
            if failed:
                notes.append(f"{len(failed)} snapshot(s) could not be scored, e.g. {failed[0]}: "
                             f"{scores[next(d for d, r in scores.items() if r.get('error'))]['error']}")
        else:
            notes.append("maintainability scores skipped: no SWE-CI python / score.py, or no epoch snapshots")
    metrics: list[dict] = []
    for name in arms:
        for task_id, states in collected[name]["states"].items():
            metrics += build_metrics(name, task_id, out, collected[name]["tasks"][task_id], states, scores)
    write_metrics_csv(out, metrics)
    for name, info in trace_info.items():
        write_aux_pairs(out, name, info["sessions"], {t["task_id"]: t for t in manifest["tasks"]})

    if not args.no_plots:
        make_plots(out, arms, collected, notes, metrics)
    (out / "RUN.md").write_text(
        build_report(label, out, arms, collected, trace_info, export_counts, notes, manifest, metrics, scored),
        encoding="utf-8",
    )
    print(f"collected {len(arms)} arm(s) into {out}")
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
