#!/usr/bin/env python
"""Collect a SWE-CI A/B run (control vs foresight) into ``results/<label>/``.

``tools/collect_run.sh`` does this for a Slurm run directory. A SWE-CI run has no
such directory: its evidence is scattered over SWE-CI's ``experiments/`` folder,
foresight's trace, an aux export directory and a few logs. This gathers it in
one command, in one layout, so every run is documented the same way::

    python tools/collect_swe_ci_ab.py --label swe-ci-ab-<name> \\
        --swe-ci-dir ../SWE-CI \\
        --arm control=control_t1-5:config_control.toml \\
        --arm foresight=foresight_t1-5:config_foresight.toml \\
        --trace foresight=traces/swe_ci_foresight_t1-5.jsonl \\
        --aux-export-dir foresight=traces/aux-agent-swe-ci-t1-5 \\
        --foresight-config configs/swe_ci_expt.yaml \\
        --model-url target=http://host:8001/v1 --model-url aux=http://host:8002/v1

Only the standard library is required. ``matplotlib`` (plots) and SWE-CI's own
python (the ``swe_ci.summarize`` table) are used when present and skipped, with a
note in ``RUN.md``, when not.

Layout written under ``results/<label>/``::

    RUN.md                        generated: results, timing, aux provenance,
                                  incidents, cost. findings.md stays hand-written.
    environment.txt               commits, versions, hardware, model endpoints
    configs/                      SWE-CI configs and the foresight config (keys redacted)
    plots/                        gap per epoch, EvoScore per task
    data/<arm>/main.log           SWE-CI's main log
    data/<arm>/summary.txt        the swe_ci.summarize table
    data/<arm>/<task>/iteration.jsonl, task.log
    data/<arm>/<task>/epoch_<N>/  what the architect saw and what the model did:
        non-passed/summary.jsonl    the failing-test list at the START of epoch N
        requirement.xml             the architect's output for epoch N
        edit.diff                   the code change made in epoch N (a/ = start, b/ = end)
    data/<arm>/trace.jsonl.gz     foresight's trace (arms that have one)
    data/<arm>/aux_sessions.json  one entry per aux run: task, epoch, phase, the
                                  aux answer, provenance, exported session ids
    data/<arm>/aux-agent/         ses_*.json, opencode's own record of each aux run

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
from collections import Counter
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


def epoch_states(task_dir: Path, epochs: list[dict]) -> dict[int, dict]:
    """``{epoch: {"start": dir|None, "result": dir|None}}``. See the module docstring."""
    found = archives(task_dir)
    finished = [e for e in epochs if e["end"] is not None]
    final = None
    if found and finished:
        last_end = max(e["end"] for e in finished)
        if (found[-1][0] - last_end).total_seconds() >= FINAL_MIN_LEAD_S:
            final = found[-1][1]
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
        ["diff", "-ruN", "-x", ".git", "-x", "__pycache__", "-x", "*.pyc", str(a), str(b)],
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
    return {"tasks": tasks}


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
    return {"sessions": sessions, "tokens": tokens, "rows": len(rows)}


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
    for name, url in urls.items():
        lines.append(f"model[{name}]".ljust(20) + f"{url}  {model_info(url)}")
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    label: str, out: Path, arms: dict, collected: dict, traces: dict, exports: dict, notes: list[str]
) -> str:
    names = list(arms)
    task_ids = sorted({t for c in collected.values() for t in c["tasks"]})
    scores = {a: parse_summary(out / "data" / a / "summary.txt", task_ids) for a in names}
    lines = [f"# {label}: generated run report", "",
             "Generated by `tools/collect_swe_ci_ab.py`. The interpretation lives in `findings.md`.", ""]
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


def make_plots(out: Path, arms: dict, collected: dict, notes: list[str]) -> None:
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
    p.add_argument("--label", required=True, help="results/<label>/ unless --out is given")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--swe-ci-dir", type=Path, default=REPO.parent / "SWE-CI")
    p.add_argument("--arm", action="append", required=True, metavar="NAME=EXPERIMENT[:CONFIG]",
                   help="repeatable; CONFIG is a file name inside --swe-ci-dir")
    p.add_argument("--trace", action="append", default=[], metavar="ARM=PATH")
    p.add_argument("--aux-export-dir", action="append", default=[], metavar="ARM=DIR")
    p.add_argument("--foresight-config", type=Path, default=None)
    p.add_argument("--model-url", action="append", default=[], metavar="NAME=URL")
    p.add_argument("--swe-ci-python", default=None, help="default: <swe-ci-dir>/.venv/bin/python")
    p.add_argument("--foresight-rev", default=None,
                   help="commit the run used, when this checkout has moved on since")
    p.add_argument("--tz-offset-hours", type=float, default=None,
                   help="local time offset from UTC of the machine that ran SWE-CI (default: this machine)")
    p.add_argument("--with-tracebacks", action="store_true", help="also copy each epoch's per-test tracebacks")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    out = args.out or (REPO / "results" / args.label)
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
    for name in list(traces) + list(exports):
        if name not in arms:
            raise SystemExit(f"--trace/--aux-export-dir names an arm that is not given: {name!r}")

    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
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
        if name in exports:
            path = Path(exports[name])
            export_counts[name] = copy_aux_export(name, path, out) if path.is_dir() else 0
            if not path.is_dir():
                notes.append(f"{name}: aux export dir {path} not found")
    copy_configs(swe_ci_dir, arms, args.foresight_config, out)
    write_environment(out, swe_ci_dir, parse_kv(args.model_url, "--model-url"), args.foresight_rev,
                      ["collect_swe_ci_ab.py"] + argv)
    if not args.no_plots:
        make_plots(out, arms, collected, notes)
    (out / "RUN.md").write_text(
        build_report(args.label, out, arms, collected, trace_info, export_counts, notes), encoding="utf-8"
    )
    print(f"collected {len(arms)} arm(s) into {out}")
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
