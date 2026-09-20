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
REQ_TWO = ("<requirements><requirement><location>/app/code/a.py</location>"
           "<description>make x three</description></requirement></requirements>")


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
    make_archive(a, "2026-09-19-10-01-10", "x = 2\n", failing='{"test": "t2"}\n', requirement=REQ_TWO)
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

    (root / "cfg.toml").write_text(
        'splitting = "default"\napi_key = "sk-secret123"\nhf_token = "none"\n'
        '[evolve]\nmax_epoch = 2\nmax_workers = 1\n[evolve.architect]\nmax_try = 3\n'
    )
    # The benchmark csv: A and B are rows 6 and 7 of it.
    (root / "metadata").mkdir()
    lines = ["task_id,repo_name,url,licence,current_sha,target_sha,test_gap,image_sha,code_sha"]
    for n in range(1, 6):
        lines.append(f"filler{n},o/f{n},https://x/f{n}.git,MIT,{'0' * 40},{'1' * 40},3,i,c")
    lines.append(f"{TASK_A},owner/alpha,https://github.com/owner/alpha.git,MIT,{'a' * 40},{'b' * 40},5,i,c")
    lines.append(f"{TASK_B},owner/beta,https://github.com/owner/beta.git,MIT,{'c' * 40},{'d' * 40},3,i,c")
    (root / "metadata" / "default.csv").write_text("\n".join(lines) + "\n")
    # Stands in for SWE-CI's score.py: only its PATH is checked, the scorer itself is faked below.
    score_dir = root / "src" / "swe_ci" / "benchmark" / "utils"
    score_dir.mkdir(parents=True)
    (score_dir / "score.py").write_text("")
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
    assert (e2 / "requirement.xml").read_text() == REQ_TWO
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


# -- rows, commits, settings: the manifest -----------------------------------


def test_range_str():
    assert collector.range_str([6, 7, 8, 9, 10]) == "6-10"
    assert collector.range_str([1, 2, 5]) == "1-2_5"
    assert collector.range_str([3]) == "3"
    assert collector.range_str([10, 6, 8, 7, 9, 6]) == "6-10"


def test_config_values_are_read_flat_with_their_section():
    values = collector.read_config_values(
        'experiment_name = "x"   # a comment\nmode = "tdd"\n[evolve]\nmax_epoch = 20\n'
        '[evolve.architect]\nmax_try = 3\n'
    )
    assert values["experiment_name"] == "x" and values["mode"] == "tdd"
    assert values["evolve.max_epoch"] == "20" and values["evolve.architect.max_try"] == "3"


def test_the_label_is_derived_from_the_rows_and_epochs(swe_ci, tmp_path):
    out = tmp_path / "out"
    collector.main(["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
                    "--no-plots", "--no-score", "--foresight-rev", "abc"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["label"] == "ab-test__swe-ci__rows-6-7__ep2"
    assert manifest["rows"] == [6, 7] and manifest["rows_range"] == "6-7" and manifest["max_epoch"] == 2
    assert (out / "RUN.md").read_text().startswith("# ab-test__swe-ci__rows-6-7__ep2")


def test_the_manifest_carries_each_tasks_commits_from_the_benchmark(swe_ci, tmp_path):
    out = tmp_path / "out"
    collector.main(["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
                    "--no-plots", "--no-score", "--foresight-rev", "abc"])
    manifest = json.loads((out / "manifest.json").read_text())
    alpha = manifest["tasks"][0]
    assert alpha["task_id"] == TASK_A and alpha["row"] == 6 and alpha["repo"] == "owner/alpha"
    assert alpha["current_sha"] == "a" * 40 and alpha["target_sha"] == "b" * 40
    report = (out / "RUN.md").read_text()
    assert "`aaaaaaaaaa` -> `bbbbbbbbbb`" in report and "owner/alpha" in report


def test_a_label_cannot_be_derived_without_rows_so_one_must_be_given(swe_ci, tmp_path):
    (swe_ci / "metadata" / "default.csv").write_text("task_id,repo_name\n")     # a trimmed-away csv
    with pytest.raises(SystemExit, match="--label"):
        collector.main(["--out", str(tmp_path / "o"), "--swe-ci-dir", str(swe_ci),
                        "--arm", "control=exp1:cfg.toml", "--no-plots", "--no-score"])


def test_settings_that_differ_between_arms_are_flagged(swe_ci, tmp_path):
    import shutil

    shutil.copytree(swe_ci / "experiments" / "exp1", swe_ci / "experiments" / "exp2")
    (swe_ci / "cfg2.toml").write_text(       # top-level keys first, as in a real SWE-CI config
        'base_url = "http://elsewhere"\n' + (swe_ci / "cfg.toml").read_text().replace("max_try = 3", "max_try = 5")
    )
    out = tmp_path / "out"
    collector.main(["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
                    "--arm", "foresight=exp2:cfg2.toml", "--no-plots", "--no-score", "--foresight-rev", "a"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["settings_identical_across_arms"] is False
    # base_url is per-arm plumbing and must not count; max_try must.
    assert list(manifest["settings_differences"]) == ["evolve.architect.max_try"]
    assert "DIFFERS `evolve.architect.max_try`" in (out / "RUN.md").read_text()


def test_identical_settings_are_confirmed(swe_ci, tmp_path):
    import shutil

    shutil.copytree(swe_ci / "experiments" / "exp1", swe_ci / "experiments" / "exp2")
    (swe_ci / "cfg2.toml").write_text('experiment_name = "other"\n' + (swe_ci / "cfg.toml").read_text())
    out = tmp_path / "out"
    collector.main(["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
                    "--arm", "foresight=exp2:cfg2.toml", "--no-plots", "--no-score", "--foresight-rev", "a"])
    assert json.loads((out / "manifest.json").read_text())["settings_identical_across_arms"] is True


# -- code change and code quality per epoch ----------------------------------

DIFF = """diff -ruN a/x.py b/x.py
--- a/x.py\t2026-01-01
+++ b/x.py\t2026-01-02
@@ -1,3 +1,3 @@
 keep
--- a comment that begins with two dashes
+added one
+added two
-plain removal
diff -ruN a/tests/test_x.py b/tests/test_x.py
--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -1 +1 @@
-old
+new
"""


def test_diffstat_counts_inside_hunks_and_is_not_fooled_by_dashes():
    stat = collector.diffstat(DIFF)
    # "--- a comment" is a REMOVED line (its text starts with "-- "), not a file header.
    assert stat == {"files": 2, "added": 3, "removed": 3, "test_files": 1}


@pytest.mark.parametrize("path, expected", [
    ("tests/test_a.py", True), ("pkg/tests/helpers.py", True), ("conftest.py", True),
    ("pkg/test_thing.py", True), ("pkg/thing_test.py", True), ("pkg/core.py", False),
    ("src/testing_utils.py", False),
])
def test_test_paths(path, expected):
    assert collector.is_test_path(path) is expected


def fake_scores(directory_scores):
    def score(dirs, *args, **kwargs):
        return {d: directory_scores.get(d.name, {"mi": 50.0, "pylint": 5.0}) for d in dirs}
    return score


def test_metrics_follow_every_epoch_end_and_hold_state_over_a_broken_epoch(swe_ci, tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "score_snapshots", fake_scores({
        "2026-09-19-10-00-05": {"mi": 40.0, "pylint": 4.0},     # A: start of epoch 1 (state 0)
        "2026-09-19-10-01-10": {"mi": 44.0, "pylint": 4.5},     # A: after epoch 1
        "2026-09-19-10-02-10": {"mi": 47.0, "pylint": 5.0},     # A: after epoch 2 (final)
    }))
    out = tmp_path / "out"
    collector.main(["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
                    "--no-plots", "--swe-ci-python", sys.executable, "--foresight-rev", "abc"])

    import csv

    rows = list(csv.DictReader((out / "metrics.csv").open()))
    a = [r for r in rows if r["task_id"] == TASK_A]
    assert [r["epoch"] for r in a] == ["0", "1", "2"]
    assert [r["mi"] for r in a] == ["40.0", "44.0", "47.0"]           # the state AFTER each epoch
    assert (a[1]["edit_added"], a[1]["edit_removed"]) == ("1", "1")   # x = 1 -> x = 2
    assert (a[2]["cum_added"], a[2]["cum_removed"], a[2]["cum_lines"]) == ("2", "2", "4")
    assert [r["gap"] for r in a] == ["5", "4", "2"]

    b = [r for r in rows if r["task_id"] == TASK_B]
    assert [r["accepted"] for r in b] == ["True", "True", "False"]
    assert b[2]["gap"] == ""                                           # pytest did not run
    assert b[2]["mi"] == b[1]["mi"]                                    # SWE-CI kept the previous code
    assert b[2]["edit_added"] != ""                                    # the attempted edit is still counted
    assert b[2]["cum_lines"] == b[1]["cum_lines"]                      # ... but not accumulated
    report = (out / "RUN.md").read_text()
    assert "+1/-1, +1/-1" in report and "!" in report                 # churn table, broken epoch marked


def test_scoring_can_be_skipped(swe_ci, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not score")

    monkeypatch.setattr(collector, "score_snapshots", boom)
    out = tmp_path / "out"
    collector.main(["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
                    "--no-plots", "--no-score", "--foresight-rev", "abc"])
    import csv

    assert all(r["mi"] == "" for r in csv.DictReader((out / "metrics.csv").open()))


# -- the aux / base task pairing table ---------------------------------------


def test_each_aux_answer_is_paired_with_its_base_task(swe_ci, trace, tmp_path):
    out = tmp_path / "out"
    collector.main(["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
                    "--trace", f"control={trace}", "--tz-offset-hours", "0", "--no-plots", "--no-score",
                    "--foresight-rev", "abc"])
    text = (out / "data" / "control" / "AUX_PAIRS.md").read_text()
    # Architect session (epoch 1): the failing tests it started from.
    assert "alpha aaaaaa, epoch 1, architect" in text and "`t1`" in text
    assert "evolve `owner/alpha` from `aaaaaaaaaa` until the tests of `bbbbbbbbbb` pass" in text
    assert "> epoch one architect future tasks" in text
    # Programmer session (epoch 2): the requirement.xml it was working from.
    assert "alpha aaaaaa, epoch 2, programmer" in text and "make x three" in text
    assert "> epoch two programmer future tasks" in text


def test_a_model_line_recorded_while_the_server_was_up_survives_a_later_collection(swe_ci, tmp_path, monkeypatch):
    """Slurm model jobs expire; a re-collection must not erase what was served."""
    out = tmp_path / "out"
    args = ["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
            "--no-plots", "--no-score", "--foresight-rev", "abc", "--model-url", "target=http://gone/v1"]
    monkeypatch.setattr(collector, "model_info", lambda url: "served=target-model root=Qwen/X max_model_len=262144")
    collector.main(args)
    monkeypatch.setattr(collector, "model_info", lambda url: "unreachable (URLError)")
    collector.main(args)
    collector.main(args)                                     # a third time must not stack the note
    line = next(l for l in (out / "environment.txt").read_text().splitlines() if l.startswith("model[target]"))
    assert "root=Qwen/X" in line and "max_model_len=262144" in line
    assert line.count("kept from an earlier collection") == 1


def test_an_endpoint_never_seen_up_is_reported_unreachable(swe_ci, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(collector, "model_info", lambda url: "unreachable (URLError)")
    collector.main(["--out", str(out), "--swe-ci-dir", str(swe_ci), "--arm", "control=exp1:cfg.toml",
                    "--no-plots", "--no-score", "--foresight-rev", "abc", "--model-url", "target=http://gone/v1"])
    assert "unreachable (URLError)" in (out / "environment.txt").read_text()


# -- macOS AppleDouble sidecars ("._name") -----------------------------------


def test_sidecar_files_are_left_out_of_the_code_diff(tmp_path):
    """On an external volume macOS writes ._x beside every moved file; they are not code."""
    def snapshot(name, code, sidecar):
        folder = tmp_path / name
        (folder / "code").mkdir(parents=True)
        (folder / "code" / "a.py").write_text(code)
        if sidecar:
            (folder / "code" / "._a.py").write_bytes(b"\x00\x05\x16\x07 appledouble")
        return folder

    diff = collector.code_diff(snapshot("before", "x = 1\n", False), snapshot("after", "x = 2\n", True))
    assert "+x = 2" in diff
    assert "._a.py" not in diff
    assert collector.diffstat(diff)["files"] == 1


def test_pylint_is_told_to_ignore_sidecars_and_the_excluded_dirs():
    sys.path.insert(0, str(REPO / "tools"))
    import swe_ci_score_helper as helper

    args = helper.pylint_args(Path("/snap/code"), ["tests"])
    assert "--recursive=y" in args and "-j1" in args          # SWE-CI's own call scans nothing
    assert r"--ignore-patterns=^\._" in args
    assert any(a.startswith("--ignore-paths=") and "/snap/code/tests" in a for a in args)


# -- the failing tests after the last epoch, stdout logs, logged replies ------


def test_the_failing_tests_after_the_last_epoch_are_saved(swe_ci, tmp_path):
    out = tmp_path / "out"
    run_collector(swe_ci, out)
    # Task A: the final folder holds the state after epoch 2 (t3), not epoch 2's start (t2).
    final = out / "data" / "control" / TASK_A / "final" / "non-passed" / "summary.jsonl"
    assert final.read_text() == '{"test": "t3"}\n'
    assert (out / "data" / "control" / TASK_A / "epoch_2" / "non-passed" / "summary.jsonl").read_text() == '{"test": "t2"}\n'


def test_after_an_epoch_that_could_not_run_the_final_state_is_the_last_accepted_one(swe_ci, tmp_path):
    out = tmp_path / "out"
    run_collector(swe_ci, out)
    final = out / "data" / "control" / TASK_B / "final" / "non-passed" / "summary.jsonl"
    assert final.read_text() == '{"test": "b2"}\n'      # not the broken tmp/, which has no failing-test list


def test_tracebacks_of_the_final_state_follow_the_flag(swe_ci, tmp_path):
    run_collector(swe_ci, tmp_path / "plain")
    run_collector(swe_ci, tmp_path / "full", "--with-tracebacks")
    plain = tmp_path / "plain" / "data" / "control" / TASK_A / "final" / "non-passed"
    full = tmp_path / "full" / "data" / "control" / TASK_A / "final" / "non-passed"
    assert not (plain / "tests_a.py__test_x").exists() and (full / "tests_a.py__test_x").is_file()


def test_the_stdout_log_is_copied_without_colour_codes_or_secrets(swe_ci, tmp_path):
    log = tmp_path / "control.log"
    log.write_text("\x1b[1mAVERAGE\x1b[0m 0.1\n api_key = \"sk-real-secret-value\"\n")
    out = tmp_path / "out"
    run_collector(swe_ci, out, "--stdout-log", f"control={log}")
    text = (out / "data" / "control" / "swe_ci_stdout.log").read_text()
    assert "AVERAGE 0.1" in text and "\x1b" not in text and "sk-real-secret-value" not in text


def test_a_missing_stdout_log_is_a_note_not_a_failure(swe_ci, tmp_path, capsys):
    assert run_collector(swe_ci, tmp_path / "out", "--stdout-log", f"control={tmp_path / 'nope.log'}") == 0
    assert "stdout log" in capsys.readouterr().err


def test_the_report_says_whether_target_replies_were_logged(swe_ci, trace, tmp_path):
    with_replies = tmp_path / "with.jsonl"
    rows = [json.loads(l) for l in trace.read_text().splitlines()]
    for row in rows:
        if row["role"] == "target":
            row["reply"] = {"content": "hi", "tool_calls": [], "finish_reason": "stop", "truncated": False}
    with_replies.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    run_collector(swe_ci, out1, "--trace", f"control={with_replies}")
    run_collector(swe_ci, out2, "--trace", f"control={trace}")
    assert "target replies logged: 2 of 2 target rows" in (out1 / "RUN.md").read_text()
    assert "target replies: not logged in this run" in (out2 / "RUN.md").read_text()


# -- server-side/: filled in by the model-server side after collection ----------


def test_a_server_side_checklist_is_created_and_reported_as_not_filled(swe_ci, tmp_path):
    out = tmp_path / "out"
    run_collector(swe_ci, out)
    assert "sampling defaults" in (out / "server-side" / "TEMPLATE.md").read_text()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["server_side"] == {"filled": False, "files": []}
    assert "server side" in (out / "RUN.md").read_text() and "not added yet" in (out / "RUN.md").read_text()


def test_what_the_server_side_adds_survives_a_recollection(swe_ci, tmp_path):
    """The whole point: they add files AFTER our collection, and a re-run must not lose them."""
    out = tmp_path / "out"
    run_collector(swe_ci, out)
    (out / "server-side" / "setup.md").write_text("vLLM 0.29.0, gpu-b200, jobs 911023 / 911146\n")
    (out / "server-side" / "TEMPLATE.md").write_text("edited by them\n")
    run_collector(swe_ci, out)
    assert (out / "server-side" / "setup.md").read_text().startswith("vLLM 0.29.0")
    assert (out / "server-side" / "TEMPLATE.md").read_text() == "edited by them\n"   # not regenerated
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["server_side"] == {"filled": True, "files": ["setup.md"]}
    assert "1 file(s): setup.md" in (out / "RUN.md").read_text()
