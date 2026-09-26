"""Load the SWE-CI A/B results (rows 6-21, 20 epochs) into plain Python structures.

Everything the statistics and the figures need comes from here:

- EvoScore per task and arm, computed exactly as SWE-CI does
  (SWE-CI/src/swe_ci/benchmark/summarize.py:metrics_func): an epoch whose pytest
  could not run is recorded with an empty pytest summary, read as passed = 0, and
  therefore scores a relative change of -1.
- The per-epoch relative change series (EvoScore is its mean over the 20 epochs).
- Lines changed per epoch (added + removed in that epoch's edit.diff, from metrics.csv),
  including edits that were rejected because pytest could not run.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ROWS = range(6, 22)          # rows 6-21 of the benchmark csv (row 22's foresight arm is truncated)
MAX_EPOCH = 20               # evolve.max_epoch in every run
ARMS = ("control", "foresight")


@dataclass
class ArmRun:
    init_pass: int
    target_pass: int
    passed: list          # passed tests after epochs 1..n, None where pytest could not run
    rel_change: list      # SWE-CI's relative change per epoch, padded to MAX_EPOCH
    evo_score: float
    broken_epochs: int    # epochs where pytest could not run (edit rejected)
    churn: list           # lines added + removed per epoch 1..MAX_EPOCH (0 after the task stopped)
    accepted: list        # per epoch 1..MAX_EPOCH: True, False (rejected) or None (no epoch run)


@dataclass
class Task:
    row: int
    task_id: str
    label: str            # short name, e.g. "yarl 576b5e"
    arms: dict = field(default_factory=dict)


def relative_changes(init_pass, target_pass, evo_seq, seq_len=MAX_EPOCH):
    """SWE-CI's metrics_func, returning the per-epoch relative changes (EvoScore = their mean)."""
    evo_seq = [min(max(0, p), target_pass) for p in evo_seq]
    if len(evo_seq) < seq_len:
        fill = evo_seq[-1] if evo_seq else init_pass
        evo_seq = evo_seq + [fill] * (seq_len - len(evo_seq))
    else:
        evo_seq = evo_seq[:seq_len]
    total_gap = target_pass - init_pass
    changes = []
    for p in evo_seq:
        if p > init_pass:
            changes.append((p - init_pass) / total_gap if total_gap > 0 else 1.0)
        elif p < init_pass:
            changes.append((p - init_pass) / init_pass if init_pass > 0 else -1.0)
        else:
            changes.append(0.0)
    return changes


def _batch_dirs():
    return sorted(RESULTS_DIR.glob("ab-test__swe-ci__rows-*__ep20"))


def _short_label(task_id):
    # "aio-libs__yarl__576b5e__857930" -> "yarl 576b5e"
    _, repo, cur, _ = task_id.rsplit("__", 3)
    return f"{repo} {cur}"


def load_tasks():
    """Return the paired tasks of rows 6-21, ordered by row."""
    tasks = {}
    for batch in _batch_dirs():
        # raw_decode: rows-12-13's committed manifest.json has stray bytes after the JSON object
        manifest, _ = json.JSONDecoder().raw_decode((batch / "manifest.json").read_text())
        rows_of = {t["task_id"]: t["row"] for t in manifest["tasks"]}
        metrics = {}
        with open(batch / "metrics.csv", newline="") as f:
            for r in csv.DictReader(f):
                metrics.setdefault((r["arm"], r["task_id"]), {})[int(r["epoch"])] = r
        for task_id, row in rows_of.items():
            if row not in ROWS:
                continue
            task = tasks.setdefault(row, Task(row, task_id, _short_label(task_id)))
            for arm in ARMS:
                it_file = batch / "data" / arm / task_id / "iteration.jsonl"
                records = [json.loads(l) for l in it_file.read_text().splitlines() if l.strip()]
                init_pass = records[0]["pytest"]["passed"]
                target_pass = init_pass + records[0]["gap"]
                passed = [r["pytest"].get("passed") for r in records[1:]]
                # SWE-CI reads a missing pytest summary as passed = 0
                rel = relative_changes(init_pass, target_pass, [p if p is not None else 0 for p in passed])

                per_epoch = metrics[(arm, task_id)]
                churn, accepted = [], []
                for e in range(1, MAX_EPOCH + 1):
                    m = per_epoch.get(e)
                    if m is None:
                        churn.append(0)
                        accepted.append(None)
                    else:
                        churn.append(int(m["edit_added"] or 0) + int(m["edit_removed"] or 0))
                        accepted.append(m["accepted"] == "True")

                task.arms[arm] = ArmRun(
                    init_pass=init_pass,
                    target_pass=target_pass,
                    passed=passed,
                    rel_change=rel,
                    evo_score=sum(rel) / MAX_EPOCH,
                    broken_epochs=sum(p is None for p in passed),
                    churn=churn,
                    accepted=accepted,
                )
    missing = [r for r in ROWS if r not in tasks or len(tasks[r].arms) != 2]
    if missing:
        raise RuntimeError(f"rows without both arms: {missing}")
    return [tasks[r] for r in sorted(tasks)]


def differences(tasks):
    """Paired differences d = EvoScore(foresight) - EvoScore(control), in task order."""
    return [t.arms["foresight"].evo_score - t.arms["control"].evo_score for t in tasks]


def load_churn_all_rows(rows=range(1, 23)):
    """Lines changed per epoch for every A/B task, including rows 1-5 (5-epoch cap) and row 22
    (foresight stopped after 12 epochs), which the tests leave out.

    Returns {row: {arm: (churn, accepted)}}, both lists of length MAX_EPOCH. An epoch that was never
    run because of the epoch cap or truncation is NaN / None; an epoch after the task was resolved
    (gap reached 0) counts as 0 lines changed, as in load_tasks.
    """
    out = {}
    for batch in sorted(RESULTS_DIR.glob("ab-test__swe-ci__rows-*")):
        manifest, _ = json.JSONDecoder().raw_decode((batch / "manifest.json").read_text())
        rows_of = {t["task_id"]: t["row"] for t in manifest["tasks"]}
        metrics = {}
        with open(batch / "metrics.csv", newline="") as f:
            for r in csv.DictReader(f):
                metrics.setdefault((r["arm"], r["task_id"]), {})[int(r["epoch"])] = r
        for task_id, row in rows_of.items():
            if row not in rows:
                continue
            for arm in ARMS:
                records = (batch / "data" / arm / task_id / "iteration.jsonl").read_text().splitlines()
                resolved = json.loads(records[-1])["gap"] == 0
                per_epoch = metrics[(arm, task_id)]
                churn, accepted = [], []
                for e in range(1, MAX_EPOCH + 1):
                    m = per_epoch.get(e)
                    if m is not None:
                        churn.append(int(m["edit_added"] or 0) + int(m["edit_removed"] or 0))
                        accepted.append(m["accepted"] == "True")
                    elif resolved:
                        churn.append(0)
                        accepted.append(None)
                    else:
                        churn.append(float("nan"))
                        accepted.append(None)
                out.setdefault(row, {})[arm] = (churn, accepted)
    return out


def load_rel_change_all_rows(rows=range(1, 23)):
    """SWE-CI's relative change per epoch for every A/B task, including rows 1-5 (5-epoch cap) and
    row 22 (foresight stopped after 12 epochs).

    Returns {row: {arm: list of MAX_EPOCH floats}}. After a resolved task the last value is carried
    forward, as SWE-CI does; an epoch never run because of the cap or truncation is NaN.
    """
    out = {}
    for batch in sorted(RESULTS_DIR.glob("ab-test__swe-ci__rows-*")):
        manifest, _ = json.JSONDecoder().raw_decode((batch / "manifest.json").read_text())
        for t in manifest["tasks"]:
            if t["row"] not in rows:
                continue
            for arm in ARMS:
                it_file = batch / "data" / arm / t["task_id"] / "iteration.jsonl"
                records = [json.loads(l) for l in it_file.read_text().splitlines() if l.strip()]
                init_pass = records[0]["pytest"]["passed"]
                target_pass = init_pass + records[0]["gap"]
                passed = [r["pytest"].get("passed", 0) for r in records[1:]]
                rel = relative_changes(init_pass, target_pass, passed, seq_len=len(passed))
                if records[-1]["gap"] == 0:
                    rel += [rel[-1]] * (MAX_EPOCH - len(rel))
                else:
                    rel += [float("nan")] * (MAX_EPOCH - len(rel))
                out.setdefault(t["row"], {})[arm] = rel
    return out
