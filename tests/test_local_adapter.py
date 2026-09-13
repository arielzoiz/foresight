"""LocalAdapter: aux as a real subprocess over a real workspace.

These drive the actual harness contract -- spawn a command, explore a directory,
read an answer file back, and catch a write -- using tools/fake_agent.py as the
harness. Nothing is mocked, because every failure mode worth having a test for
lives in the plumbing: argv substitution, cwd, environment, timeouts, stale
answers, and the guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import StubBackend, user_only
from foresight.adapters import ADAPTERS
from foresight.adapters.base import AdapterOptions
from foresight.adapters.local import LocalAdapter
from foresight.context import InboundRequest
from foresight.errors import AuxFailure, ConfigError, GuardViolation

REPO = Path(__file__).resolve().parents[1]
FAKE_AGENT = str(REPO / "tools" / "fake_agent.py")

AUX_PROMPT = (
    "Repo at {{ workspace }}. READ ONLY.\nTask:\n{{ prompt }}\n"
    "Write your answer to {{ answer_file }}.\n"
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A scratch repo with enough in it for the agent to have read something."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "sync.py").write_text("def sync(src, dst):\n    copy(src, dst)\n")
    (root / "pkg" / "util.py").write_text("def helper():\n    return 1\n")
    (root / "README.md").write_text("# scratch repo\n")
    return root


def build(
    workspace: Path,
    aux_spec,
    *,
    extra_args: list[str] | None = None,
    answer_file: str | None = None,
    answer_from: str = "file",
    guard: str | None = "manifest",
    timeout_s: float = 60.0,
    aux_prompt: str = AUX_PROMPT,
) -> LocalAdapter:
    argv = [sys.executable, FAKE_AGENT]
    if answer_from == "file":
        argv += ["--answer-file", "{answer_file}"]
    argv += list(extra_args or [])
    argv += ["{prompt}"]

    return LocalAdapter(
        aux_backend=StubBackend(),
        aux_spec=aux_spec,
        aux_prompt=aux_prompt,
        options=AdapterOptions(
            name="local",
            workspace=str(workspace),
            agent_cmd=argv,
            answer_file=answer_file,
            answer_from=answer_from,
            timeout_s=timeout_s,
        ),
        guard_name=guard,
    )


def request_for(task: str) -> InboundRequest:
    return InboundRequest(body=user_only(task))


# -- registration and construction ----------------------------------------


def test_local_adapter_is_registered():
    assert ADAPTERS["local"] is LocalAdapter


def test_workspace_is_required(aux_spec):
    with pytest.raises(ConfigError, match="adapter.workspace is required"):
        LocalAdapter(
            aux_backend=StubBackend(),
            aux_spec=aux_spec,
            aux_prompt=AUX_PROMPT,
            options=AdapterOptions(name="local", agent_cmd=["echo"]),
        )


def test_missing_workspace_directory_is_a_config_error(aux_spec, tmp_path):
    with pytest.raises(ConfigError, match="not a directory"):
        LocalAdapter(
            aux_backend=StubBackend(),
            aux_spec=aux_spec,
            aux_prompt=AUX_PROMPT,
            options=AdapterOptions(
                name="local", workspace=str(tmp_path / "nope"), agent_cmd=["echo"]
            ),
        )


def test_agent_cmd_is_required(aux_spec, workspace):
    """No default harness: we define no tools, so the command must be named."""
    with pytest.raises(ConfigError, match="agent_cmd is required"):
        LocalAdapter(
            aux_backend=StubBackend(),
            aux_spec=aux_spec,
            aux_prompt=AUX_PROMPT,
            options=AdapterOptions(name="local", workspace=str(workspace)),
        )


def test_unknown_guard_is_a_config_error(aux_spec, workspace):
    with pytest.raises(ConfigError, match="unknown guard"):
        build(workspace, aux_spec, guard="telepathy")


# -- the happy path --------------------------------------------------------


@pytest.mark.asyncio
async def test_aux_reads_the_workspace_and_answers_from_a_file(
    workspace, aux_spec, tmp_path
):
    answer = tmp_path / "answer.md"
    adapter = build(workspace, aux_spec, answer_file=str(answer))

    result = await adapter.run_aux(request_for("Add --dry-run to sync"), aux_spec)

    assert result.source == "agent"
    assert result.provenance["answer_from"] == "file"
    assert result.provenance["returncode"] == 0
    assert result.provenance["guard"] == "intact"
    assert result.provenance["workspace"] == str(workspace)
    # Proof it actually looked at the repo rather than reasoning from task text:
    # the answer names a file that only exists in this fixture.
    assert "sync.py" in result.text


@pytest.mark.asyncio
async def test_the_agent_runs_with_cwd_set_to_the_workspace(workspace, aux_spec):
    """cwd is how the harness finds the code; nothing is passed as a path."""
    adapter = build(workspace, aux_spec, answer_file=None)
    result = await adapter.run_aux(request_for("x"), aux_spec)
    assert str(workspace) in result.text


@pytest.mark.asyncio
async def test_a_stdout_harness_is_supported_when_declared(workspace, aux_spec):
    """opencode answers on stdout, so the mode is configurable -- not guessed."""
    adapter = build(workspace, aux_spec, answer_from="stdout")
    result = await adapter.run_aux(request_for("x"), aux_spec)
    assert result.provenance["answer_from"] == "stdout"
    assert "FUTURE tasks" in result.text


@pytest.mark.asyncio
async def test_file_mode_never_falls_back_to_stdout_chatter(workspace, aux_spec, tmp_path):
    """The bug this mode exists to prevent.

    A harness that ignores the write instruction still prints progress lines.
    Accepting those as the future-task list would enhance the prompt with a
    banner, putting the instance in neither experimental arm.
    """
    adapter = build(
        workspace,
        aux_spec,
        answer_file=str(tmp_path / "answer.md"),
        extra_args=["--silent"],
        guard=None,
    )
    with pytest.raises(AuxFailure, match="no answer"):
        await adapter.run_aux(request_for("x"), aux_spec)


@pytest.mark.asyncio
async def test_braces_in_the_task_survive_substitution(workspace, aux_spec, tmp_path):
    """str.replace, not str.format: prompts carry code, and code carries braces."""
    answer = tmp_path / "answer.md"
    adapter = build(workspace, aux_spec, answer_file=str(answer))
    task = 'Fix f"{user}" handling in {a: 1} dict literals {'

    # str.format() would raise KeyError('user') or ValueError here.
    result = await adapter.run_aux(request_for(task), aux_spec)

    assert result.text
    # fake_agent echoes the prompt length, so the whole braced task got through.
    assert "prompt was" in result.text


# -- session keying --------------------------------------------------------


def test_session_key_includes_the_workspace(workspace, aux_spec, tmp_path):
    """Same prompt against a different checkout is a different task."""
    other = tmp_path / "other"
    (other / "pkg").mkdir(parents=True)

    a = build(workspace, aux_spec)
    b = build(other, aux_spec)
    req = request_for("same task text")

    assert a.session_key(req) != b.session_key(req)


def test_session_key_is_stable_for_the_same_workspace_and_prompt(workspace, aux_spec):
    adapter = build(workspace, aux_spec)
    req = request_for("same task text")
    assert adapter.session_key(req) == adapter.session_key(request_for("same task text"))


# -- the guard -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_write_to_the_workspace_fails_the_request(workspace, aux_spec, tmp_path):
    """The milestone-2 reason a guard exists: nothing here can be thrown away."""
    adapter = build(
        workspace,
        aux_spec,
        answer_file=str(tmp_path / "answer.md"),
        extra_args=["--write-file", "aux_should_not_write.txt"],
    )

    with pytest.raises(GuardViolation) as caught:
        await adapter.run_aux(request_for("x"), aux_spec)

    assert "aux_should_not_write.txt" in str(caught.value)
    assert caught.value.changed
    assert caught.value.status_code == 500
    assert caught.value.error_type == "workspace_contaminated"


@pytest.mark.asyncio
async def test_a_same_size_edit_is_still_caught(workspace, aux_spec, tmp_path):
    adapter = build(
        workspace,
        aux_spec,
        answer_file=str(tmp_path / "answer.md"),
        extra_args=["--touch", "README.md"],
    )
    with pytest.raises(GuardViolation, match="README.md"):
        await adapter.run_aux(request_for("x"), aux_spec)


@pytest.mark.asyncio
async def test_contamination_beats_a_missing_answer(workspace, aux_spec, tmp_path):
    """If aux both wrote to the repo and produced nothing, report the write.

    The contamination is what invalidates the instance; a missing answer is only
    a failed request.
    """
    adapter = build(
        workspace,
        aux_spec,
        answer_file=str(tmp_path / "answer.md"),
        extra_args=["--write-file", "oops.txt", "--silent"],
    )
    with pytest.raises(GuardViolation):
        await adapter.run_aux(request_for("x"), aux_spec)


@pytest.mark.asyncio
async def test_guard_can_be_disabled(workspace, aux_spec, tmp_path):
    """guard: null is the opt-out, per guards.py -- not a NoopGuard class."""
    adapter = build(
        workspace,
        aux_spec,
        answer_file=str(tmp_path / "answer.md"),
        guard=None,
        extra_args=["--write-file", "allowed.txt"],
    )
    result = await adapter.run_aux(request_for("x"), aux_spec)
    assert result.provenance["guard"] is None


@pytest.mark.asyncio
async def test_the_answer_file_is_outside_the_workspace_so_it_is_not_a_violation(
    workspace, aux_spec, tmp_path
):
    """Aux must write its answer somewhere; that somewhere cannot be the repo."""
    adapter = build(workspace, aux_spec, answer_file=str(tmp_path / "outside.md"))
    result = await adapter.run_aux(request_for("x"), aux_spec)
    assert result.provenance["guard"] == "intact"


# -- failure paths ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_harness_is_an_aux_failure(workspace, aux_spec):
    adapter = LocalAdapter(
        aux_backend=StubBackend(),
        aux_spec=aux_spec,
        aux_prompt=AUX_PROMPT,
        options=AdapterOptions(
            name="local",
            workspace=str(workspace),
            agent_cmd=["definitely-not-a-real-harness-xyz", "{prompt}"],
        ),
        guard_name=None,
    )
    with pytest.raises(AuxFailure, match="not found"):
        await adapter.run_aux(request_for("x"), aux_spec)


@pytest.mark.asyncio
async def test_a_timeout_is_an_aux_failure(workspace, aux_spec, tmp_path):
    adapter = build(
        workspace,
        aux_spec,
        answer_file=str(tmp_path / "answer.md"),
        extra_args=["--sleep", "5"],
        timeout_s=0.5,
        guard=None,
    )
    with pytest.raises(AuxFailure, match="exceeded"):
        await adapter.run_aux(request_for("x"), aux_spec)


@pytest.mark.asyncio
async def test_a_stale_answer_file_is_not_read_as_this_runs_output(
    workspace, aux_spec, tmp_path
):
    """The hazard of a fixed answer_file: last run's text passing as this one's."""
    answer = tmp_path / "answer.md"
    answer.write_text("STALE ANSWER FROM AN EARLIER RUN\n")

    adapter = build(
        workspace,
        aux_spec,
        answer_file=str(answer),
        extra_args=["--silent"],
        guard=None,
    )
    with pytest.raises(AuxFailure, match="no answer"):
        await adapter.run_aux(request_for("x"), aux_spec)
