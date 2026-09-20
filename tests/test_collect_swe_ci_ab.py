"""tools/collect_swe_ci_ab.py: gathering a SWE-CI A/B run into results/<label>/.

The fixture is a small fake ``SWE-CI/experiments/<name>/`` tree laid out the way
SWE-CI writes it -- a ``task.log``, an ``iteration.jsonl`` and one archive folder
per epoch, named for the END of the epoch and holding its STARTING state -- with
two tasks: one that ran two normal epochs, and one whose second epoch ended with
pytest unable to run (SWE-CI then archives ``tmp/`` instead of ``current/``).
That second shape is the one that is easy to get backwards, so it is the one the
diff assertions are about.

What this deliberately does not establish: that the folder-to-epoch mapping
matches every SWE-CI version. It matches the log lines and folder names the
checked-in copy writes, and the real A/B run this was built from.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import collect_swe_ci_ab as collector  # noqa: E402

TASK_A = "owner__alpha__aaaaaa__bbbbbb"
TASK_B = "owner__beta__cccccc__dddddd"


def log_line(when: str, task: str, message: str, level: str = "INFO") -> str:
    return f"2026-09-19 {when} | {level} | {task} | {message}"


def make_archive(task_dir: Path, name: str, code: str, *, failing: str = "", requirement: str = "") -> None:
    folder = task_dir / name
    (folder / "code").mkdir(parents=True)
    (folder / "code" / "a.py").write_text(code)
    if failing:
        (folder / "non-passed").mkdir()
        (folder / "non-passed" / "summary.jsonl").write_text(failing)
        (folder / "non-passed" / "tests_a.py__test_x").write_text("traceback text\n")
    if requirement:
        (folder / "requirement.xml").write_text(requirement)


def iteration(*rows: tuple[int, int | None]) -> str:
    """(gap, passed); passed=None -> an epoch where pytest did not run."""
    out = []
    for gap, passed in rows:
        out.append(json.dumps({"gap": gap, "pytest": {} if passed is None else {"passed": passed}}))
    return "\n".join(out) + "\n"


@pytest.fixture
def swe_ci(tmp_path: Path) -> Path:
    root = tmp_path / "SWE-CI"
    exp = root / "experiments" / "exp1"
    exp.mkdir(parents=True)
    (exp / "main.log").write_text(
        "2026-09-19 10:00:00 | INFO | main | Initializing tasks...\n"
        "2026-09-19 10:03:00 | INFO | main | Evolution complete(2/2)\n"
    )

    # Task A: two normal epochs. State v1 -> v2 (epoch 1) -> v3 (epoch 2).
    a = exp / TASK_A
    a.mkdir()
    (a / "task.log").write_text("\n".join([
        log_line("10:00:00", TASK_A, "====================Epoch 1===================="),
        log_line("10:00:05", TASK_A, "(6/7) ✅ Archived directory 'current', and rename directory 'tmp' to 'current'."),
        log_line("10:00:05", TASK_A, "====================Epoch 2===================="),
        log_line("10:00:30", TASK_A, "(2/7) (Attempt 2/3) ✅ The architect agent has generated the requirements."),
        log_line("10:00:40", TASK_A, "pytest odd", level="WARNING"),
        log_line("10:01:10", TASK_A, "(6/7) ✅ Archived directory 'current', and rename directory 'tmp' to 'current'."),
        log_line("10:01:10", TASK_A, "Reached the exit condition. Archived directory 'current'."),
    ]) + "\n")
    (a / "iteration.jsonl").write_text(iteration((5, 10), (4, 11), (2, 13)))
    make_archive(a, "2026-09-19-10-00-05", "x = 1\n", failing='{"test": "t1"}\n', requirement="<req>one</req>")
    make_archive(a, "2026-09-19-10-01-10", "x = 2\n", failing='{"test": "t2"}\n', requirement="<req>two</req>")
    make_archive(a, "2026-09-19-10-02-10", "x = 3\n", failing='{"test": "t3"}\n')  # final: +60 s

    # Task B: epoch 1 normal (v1 -> v2); epoch 2 broke pytest (tmp = broken code).
    b = exp / TASK_B
    b.mkdir()
    (b / "task.log").write_text("\n".join([
        log_line("10:10:00", TASK_B, "====================Epoch 1===================="),
        log_line("10:10:05", TASK_B, "(6/7) ✅ Archived directory 'current', and rename directory 'tmp' to 'current'."),
        log_line("10:10:05", TASK_B, "====================Epoch 2===================="),
        log_line("10:10:40", TASK_B, "(4/7) ⚠️ pytest was not executed correctly. returncode=2", level="WARNING"),
        log_line("10:10:40", TASK_B, "(6/7) ✅ Archived directory 'tmp'."),
        log_line("10:10:40", TASK_B, "Reached the exit condition. Archived directory 'current'."),
    ]) + "\n")
    (b / "iteration.jsonl").write_text(iteration((3, 20), (2, 21), (-1, None)))
    make_archive(b, "2026-09-19-10-10-05", "x = 1\n", failing='{"test": "b1"}\n', requirement="<req>b-one</req>")
    make_archive(b, "2026-09-19-10-10-40", "x = (\n")                       # the broken tmp
    make_archive(b, "2026-09-19-10-11-40", "x = 2\n", failing='{"test": "b2"}\n', requirement="<req>b-two</req>")

    (root / "cfg.toml").write_text('splitting = "default"\napi_key = "sk-secret123"\nhf_token = "none"\n')
    return root


@pytest.fixture
def trace(tmp_path: Path) -> Path:
    def session(when: str, phase: str, text: str) -> dict:
        return {
            "request_id": "r-" + when, "received_at": when, "role": "target", "is_session_start": True,
            "phase": phase, "session_key": "k-" + when, "prompt_in_chars": 100, "prompt_out_chars": 180,
            "guard": {"verdict": "intact"}, "usage": {"upstream": {"prompt_tokens": 50, "completion_tokens": 5}},
            "aux": {"text": text, "usable": True, "items": 3, "provenance": {
                "resolved_by": "sole_container", "fallback": None, "duration_s": 7.5, "stdout_tail": "noise",
                "aux_export": {"dir": "d", "sessions": ["ses_" + "a" * 26], "errors": []}}},
        }

    rows = [
        session("2026-09-19T10:00:02.000Z", "architect", "epoch one architect future tasks"),
        session("2026-09-19T10:00:40.000Z", "programmer", "epoch two programmer future tasks"),
        {"role": "aux", "received_at": "2026-09-19T10:00:03.000Z",
         "usage": {"upstream": {"prompt_tokens": 1000, "completion_tokens": 10}}},
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def run_collector(swe_ci: Path, out: Path, *extra: str) -> int:
    return collector.main([
        "--label", "t", "--out", str(out), "--swe-ci-dir", str(swe_ci),
        "--arm", "control=exp1", "--no-plots", "--swe-ci-python", "/nonexistent",
        "--foresight-rev", "abc123", "--tz-offset-hours", "0", *extra,
    ])


# -- what is copied for each epoch -------------------------------------------


def test_each_epoch_gets_its_requirement_failing_tests_and_edit(swe_ci, tmp_path):
    out = tmp_path / "out"
    assert run_collector(swe_ci, out) == 0

    e1 = out / "data" / "control" / TASK_A / "epoch_1"
    assert (e1 / "requirement.xml").read_text() == "<req>one</req>"
    assert (e1 / "non-passed" / "summary.jsonl").read_text() == '{"test": "t1"}\n'
    edit = (e1 / "edit.diff").read_text()
    assert "-x = 1" in edit and "+x = 2" in edit
    assert "a/a.py" in edit and "b/a.py" in edit          # neutral prefixes, no host paths
    assert str(swe_ci) not in edit

    # Epoch 2's start is the folder named for ITS end, and its result the final folder.
    e2 = out / "data" / "control" / TASK_A / "epoch_2"
    assert (e2 / "requirement.xml").read_text() == "<req>two</req>"
    assert "-x = 2" in (e2 / "edit.diff").read_text() and "+x = 3" in (e2 / "edit.diff").read_text()


def test_tracebacks_are_only_copied_on_request(swe_ci, tmp_path):
    run_collector(swe_ci, tmp_path / "plain")
    run_collector(swe_ci, tmp_path / "full", "--with-tracebacks")
    plain = tmp_path / "plain" / "data" / "control" / TASK_A / "epoch_1" / "non-passed"
    full = tmp_path / "full" / "data" / "control" / TASK_A / "epoch_1" / "non-passed"
    assert not (plain / "tests_a.py__test_x").exists()
    assert (full / "tests_a.py__test_x").read_text() == "traceback text\n"


def test_an_epoch_where_pytest_could_not_run_diffs_the_broken_code(swe_ci, tmp_path):
    """The broken epoch archives tmp/, so its start state is the FINAL folder, not its own."""
    out = tmp_path / "out"
    run_collector(swe_ci, out)

    e1 = out / "data" / "control" / TASK_B / "epoch_1"
    assert "+x = 2" in (e1 / "edit.diff").read_text()      # result is the final folder (v2)
    e2 = out / "data" / "control" / TASK_B / "epoch_2"
    assert (e2 / "requirement.xml").read_text() == "<req>b-two</req>"
    broken = (e2 / "edit.diff").read_text()
    assert "-x = 2" in broken and "+x = (" in broken       # v2 -> the broken tmp


def test_the_logs_and_iteration_files_are_copied_per_task(swe_ci, tmp_path):
    out = tmp_path / "out"
    run_collector(swe_ci, out)
    for task in (TASK_A, TASK_B):
        assert (out / "data" / "control" / task / "task.log").is_file()
        assert (out / "data" / "control" / task / "iteration.jsonl").is_file()
    assert (out / "data" / "control" / "main.log").is_file()


# -- the trace and aux sessions ----------------------------------------------


def test_aux_runs_are_joined_to_their_task_and_epoch(swe_ci, trace, tmp_path):
    out = tmp_path / "out"
    run_collector(swe_ci, out, "--trace", f"control={trace}")

    sessions = json.loads((out / "data" / "control" / "aux_sessions.json").read_text())
    assert [(s["task_id"], s["epoch"], s["phase"]) for s in sessions] == [
        (TASK_A, 1, "architect"),
        (TASK_A, 2, "programmer"),
    ]
    assert sessions[0]["aux_text"] == "epoch one architect future tasks"
    assert "stdout_tail" not in sessions[0]["provenance"]      # noise dropped
    assert sessions[0]["provenance"]["aux_export"]["sessions"] == ["ses_" + "a" * 26]
    with gzip.open(out / "data" / "control" / "trace.jsonl.gz", "rt") as f:
        assert f.read() == trace.read_text()


def test_a_session_outside_every_epoch_is_kept_but_unjoined(swe_ci, tmp_path):
    path = tmp_path / "late.jsonl"
    path.write_text(json.dumps({
        "role": "target", "is_session_start": True, "phase": "architect",
        "received_at": "2026-09-19T15:00:00.000Z", "aux": {"text": "t", "provenance": {}},
    }) + "\n")
    out = tmp_path / "out"
    run_collector(swe_ci, out, "--trace", f"control={path}")
    (session,) = json.loads((out / "data" / "control" / "aux_sessions.json").read_text())
    assert session["task_id"] is None and session["epoch"] is None


def test_a_session_inside_two_epoch_windows_is_not_guessed(tmp_path):
    """max_workers > 1: two tasks at once. Attaching the wrong one is worse than none."""
    import datetime as dt

    def window(n):
        return {"epoch": n, "start": dt.datetime(2026, 9, 19, 10, 0, 0),
                "end": dt.datetime(2026, 9, 19, 10, 5, 0), "broken": False}

    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({
        "role": "target", "is_session_start": True, "phase": "architect",
        "received_at": "2026-09-19T10:01:00.000Z", "aux": {"text": "t", "provenance": {}},
    }) + "\n")
    (tmp_path / "data" / "control").mkdir(parents=True)
    info = collector.collect_trace("control", path, {"task1": [window(1)], "task2": [window(1)]}, tmp_path, 0)
    (session,) = info["sessions"]
    assert session["task_id"] is None and session["join"] == "ambiguous"


def test_a_trace_for_an_unknown_arm_is_refused(swe_ci, trace, tmp_path):
    with pytest.raises(SystemExit, match="arm"):
        run_collector(swe_ci, tmp_path / "out", "--trace", f"nope={trace}")


def test_the_aux_export_directory_is_copied(swe_ci, tmp_path):
    export = tmp_path / "aux-agent"
    export.mkdir()
    (export / ("ses_" + "b" * 26 + ".json")).write_text("{}")
    (export / "ignored.txt").write_text("x")
    out = tmp_path / "out"
    run_collector(swe_ci, out, "--aux-export-dir", f"control={export}")
    assert sorted(p.name for p in (out / "data" / "control" / "aux-agent").iterdir()) == [
        "ses_" + "b" * 26 + ".json"
    ]


# -- configs, report, environment --------------------------------------------


def test_secrets_are_redacted_but_placeholders_are_not(swe_ci, tmp_path):
    out = tmp_path / "out"
    collector.main([
        "--label", "t", "--out", str(out), "--swe-ci-dir", str(swe_ci),
        "--arm", "control=exp1:cfg.toml", "--no-plots", "--swe-ci-python", "/nonexistent",
        "--foresight-rev", "abc", "--tz-offset-hours", "0",
    ])
    text = (out / "configs" / "cfg.toml").read_text()
    assert "sk-secret123" not in text and 'api_key = "***"' in text
    assert 'hf_token = "none"' in text


@pytest.mark.parametrize("line, kept", [
    ('api_key = "dummy"', True),
    ('api_key = "sk-abc"', False),
    ("api_key_env: TARGET_API_KEY", True),      # an env var NAME is not a secret
    ('hf_token = "hf_realtoken"', False),
])
def test_redact_cases(line, kept):
    redacted = collector.redact(line)
    assert (redacted == line) if kept else ("***" in redacted)


def test_the_report_lists_gaps_incidents_and_cost(swe_ci, trace, tmp_path):
    out = tmp_path / "out"
    run_collector(swe_ci, out, "--trace", f"control={trace}")
    report = (out / "RUN.md").read_text()
    assert "5, 4, 2" in report                       # task A's gaps
    assert "3, 2, x" in report                       # task B: the epoch pytest could not run
    assert "(Attempt 2/3)" in report and "pytest was not executed correctly" in report
    assert "resolved_by: {'sole_container': 2}" in report
    assert "prompt tokens 1,000" in report           # aux cost from the trace
    assert "swe_ci.summarize was not run" in report  # no SWE-CI python: noted, not a failure


def test_the_summary_table_is_parsed(tmp_path):
    table = (
        " Task ID  │ EVOSCORE(Γ=1) │ RESOLVED │ ZERO_REG. │ ZRR\n"
        f" {TASK_A[:20]}… │ -0.0625 │ 0.0000 │ 1.0000 │ 0.0000\n"
        " AVERAGE │ 0.1785 │ 0.0000 │ 0.6000 │ 0.0000\n"
    )
    path = tmp_path / "summary.txt"
    path.write_text(table)
    scores = collector.parse_summary(path, [TASK_A, TASK_B])
    assert scores[TASK_A] == -0.0625 and scores["AVERAGE"] == 0.1785
    assert scores["_zero_reg"] == 0.6
    assert TASK_B not in scores


def test_the_environment_records_the_given_revision(swe_ci, tmp_path):
    out = tmp_path / "out"
    run_collector(swe_ci, out)
    env = (out / "environment.txt").read_text()
    assert "abc123 (given, not read from the tree)" in env
    assert "collector_command" in env


def test_a_missing_experiment_is_a_clear_error(swe_ci, tmp_path):
    with pytest.raises(SystemExit, match="no experiment folder"):
        collector.main(["--label", "t", "--out", str(tmp_path / "o"), "--swe-ci-dir", str(swe_ci),
                        "--arm", "control=missing", "--no-plots"])
