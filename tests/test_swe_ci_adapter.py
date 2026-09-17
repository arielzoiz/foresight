"""SweCiAdapter: aux as a second harness inside the task container.

These drive the real contract -- resolve a container from a client IP, build a
`docker exec` argv, run a harness "in" it, read the answer back out, and catch a
write -- through ``tools/fake_docker.py`` and ``tools/fake_agent.py``. Nothing
is mocked, because every failure mode worth a test lives in the plumbing:
IP->ID resolution, container-ID session keying, the HOME override, argv
substitution, timeouts, the guard, and the body-only degradation.

What these tests deliberately do NOT establish: that any of it works against a
real container runtime. ``fake_docker.py`` implements the four verbs the
adapter uses; a real Docker install is required to confirm the rest -- see
``configs/swe_ci.yaml`` and the M3 verification steps in ``README.md``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import StubBackend, user_only
from foresight.adapters import ADAPTERS
from foresight.adapters.base import AdapterOptions
from foresight.adapters.swe_ci import SWE_CI_HOME, SweCiAdapter
from foresight.context import InboundRequest
from foresight.errors import AuxFailure, ConfigError, GuardViolation

REPO = Path(__file__).resolve().parents[1]
FAKE_DOCKER = str(REPO / "tools" / "fake_docker.py")
FAKE_AGENT = str(REPO / "tools" / "fake_agent.py")

AUX_PROMPT = (
    "Project at {{ workspace }} in a container. READ ONLY.\n"
    "Task:\n{{ prompt }}\nWrite your answer to {{ answer_file }}.\n"
)

# SWE-CI's rendered prompts, reduced to the sentences that identify the phase.
# Both mention both roles, which is exactly why the marker is the identity line.
ARCHITECT_PROMPT = (
    "<identity>You are a senior software architect proficient in Python.</identity>"
    "<scene>You are collaborating closely with a senior programmer.</scene>"
)
PROGRAMMER_PROMPT = (
    "<identity>You are a senior programmer proficient in Python.</identity>"
    "<scene>You are collaborating closely with a senior software architect.</scene>"
)


class FakeDocker:
    """A ``docker`` on disk, plus the helpers a test needs to drive it."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self.path = state / "docker"
        # A shell wrapper, so the adapter can be given ONE string as
        # docker_cmd, exactly as a real deployment gives it the real `docker`.
        self.path.write_text(
            "#!/bin/sh\n"
            f'FORESIGHT_FAKE_DOCKER_STATE="{state / "containers"}" '
            f'exec "{sys.executable}" "{FAKE_DOCKER}" "$@"\n'
        )
        self.path.chmod(0o755)

    def create(self, name: str, ip: str = "", *, running: bool = True) -> str:
        argv = [str(self.path), "create", name, ip]
        if not running:
            argv.append("--not-running")
        out = subprocess.run(argv, capture_output=True, text=True, check=True)
        return out.stdout.strip()

    def rootfs(self, container_id: str) -> Path:
        return self.state / "containers" / container_id / "rootfs"

    def seed(self, container_id: str, workspace: str = "/app") -> Path:
        """A /app tree with enough in it for the agent to have read something."""
        root = self.rootfs(container_id) / workspace.lstrip("/")
        (root / "code" / "pkg").mkdir(parents=True)
        (root / "code" / "pkg" / "sync.py").write_text("def sync(a, b):\n    pass\n")
        (root / "code" / "pkg" / "util.py").write_text("def helper():\n    return 1\n")
        (root / "non-passed").mkdir(parents=True)
        (root / "non-passed" / "summary.jsonl").write_text('{"test": "t"}\n')
        return root


@pytest.fixture
def docker(tmp_path: Path) -> FakeDocker:
    return FakeDocker(tmp_path)


DEFAULT_FORESIGHT_BASE_URL = "http://172.17.0.1:8000/v1"


def build(
    docker: FakeDocker,
    aux_spec,
    *,
    extra_args: list[str] | None = None,
    answer_file: str | None = None,
    answer_from: str = "file",
    guard: str | None = "manifest",
    timeout_s: float = 60.0,
    workspace: str = "/app",
    env: dict[str, str] | None = None,
    aux_prompt: str = AUX_PROMPT,
    swe_ci_config: str | None = None,
    foresight_base_url: str | None = DEFAULT_FORESIGHT_BASE_URL,
    session_header_names: list[str] | None = None,
) -> SweCiAdapter:
    argv = [sys.executable, FAKE_AGENT]
    if answer_from == "file":
        argv += ["--answer-file", "{answer_file}"]
    argv += list(extra_args or [])
    argv += ["{prompt}"]

    kwargs = {}
    if session_header_names is not None:
        kwargs["session_header_names"] = session_header_names

    return SweCiAdapter(
        aux_backend=StubBackend(reply_text=_body_only_reply()),
        aux_spec=aux_spec,
        aux_prompt=aux_prompt,
        options=AdapterOptions(
            name="swe_ci",
            docker_cmd=str(docker.path),
            workspace=workspace,
            agent_cmd=argv,
            answer_file=answer_file,
            answer_from=answer_from,
            timeout_s=timeout_s,
            env=env or {},
            swe_ci_config=swe_ci_config,
            foresight_base_url=foresight_base_url,
            **kwargs,
        ),
        guard_name=guard,
    )


def _body_only_reply() -> str:
    """Shaped like a usable aux answer, so the fallback is not ALSO weak-aux."""
    return "1. first\n2. second\n3. third\n"


def request_from(ip: str, task: str = ARCHITECT_PROMPT) -> InboundRequest:
    return InboundRequest(body=user_only(task), client_ip=ip)


def continuation_from(ip: str, task: str = ARCHITECT_PROMPT) -> InboundRequest:
    body = user_only(task)
    body["messages"].append({"role": "assistant", "content": "thinking"})
    return InboundRequest(body=body, client_ip=ip)


# -- registration and construction ----------------------------------------


def test_swe_ci_adapter_is_registered():
    assert ADAPTERS["swe_ci"] is SweCiAdapter


def test_agent_cmd_is_required(aux_spec, docker):
    with pytest.raises(ConfigError, match="adapter.agent_cmd is required"):
        SweCiAdapter(
            aux_backend=StubBackend(),
            aux_spec=aux_spec,
            aux_prompt=AUX_PROMPT,
            options=AdapterOptions(name="swe_ci", docker_cmd=str(docker.path)),
        )


def test_sharing_swe_cis_home_is_refused(aux_spec, docker):
    """The one config error that would silently corrupt the benchmark's numbers."""
    with pytest.raises(ConfigError, match="must not be"):
        build(docker, aux_spec, env={"HOME": SWE_CI_HOME})


def test_home_defaults_to_an_isolated_directory(aux_spec, docker):
    adapter = build(docker, aux_spec)
    assert adapter._env["HOME"] == "/tmp/aux-home"


def test_agent_name_mismatch_fails_at_startup(aux_spec, docker, tmp_path):
    """The image ships exactly one harness; naming the other one cannot work."""
    config = tmp_path / "config.toml"
    config.write_text('agent_name = "opencode"\nmode = "tdd"\n')
    with pytest.raises(ConfigError, match="agent_name is 'opencode'"):
        SweCiAdapter(
            aux_backend=StubBackend(),
            aux_spec=aux_spec,
            aux_prompt=AUX_PROMPT,
            options=AdapterOptions(
                name="swe_ci",
                docker_cmd=str(docker.path),
                agent_cmd=["iflow", "--prompt", "{prompt}"],
                swe_ci_config=str(config),
                foresight_base_url=DEFAULT_FORESIGHT_BASE_URL,
            ),
        )


def test_a_matching_agent_name_is_accepted(aux_spec, docker, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('agent_name = "opencode"\n')
    adapter = SweCiAdapter(
        aux_backend=StubBackend(),
        aux_spec=aux_spec,
        aux_prompt=AUX_PROMPT,
        options=AdapterOptions(
            name="swe_ci",
            docker_cmd=str(docker.path),
            agent_cmd=["/usr/local/bin/opencode", "run", "{prompt}"],
            swe_ci_config=str(config),
            foresight_base_url=DEFAULT_FORESIGHT_BASE_URL,
        ),
    )
    assert adapter._agent_cmd[0].endswith("opencode")


def test_a_command_naming_no_known_harness_fails(aux_spec, docker, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('agent_name = "opencode"\n')
    with pytest.raises(ConfigError, match="names none of"):
        build(docker, aux_spec, swe_ci_config=str(config))


def test_the_real_swe_ci_config_is_readable_if_present(aux_spec, docker):
    """Pins the regex against the actual file, not a fixture of our own making."""
    real = REPO.parents[0] / "SWE-CI" / "config.toml"
    if not real.is_file():
        pytest.skip("SWE-CI is not checked out next to foresight")
        return
    from foresight.adapters.swe_ci import _AGENT_NAME_RE

    match = _AGENT_NAME_RE.search(real.read_text())
    assert match is not None
    assert match.group(1) in {"opencode", "iflow"}


# -- container resolution and session keying -------------------------------


def test_the_client_ip_resolves_to_its_container(aux_spec, docker):
    """The mechanism the whole adapter rests on, and it was never executed."""
    other = docker.create("swe-ci-other", "172.17.0.4")
    mine = docker.create("swe-ci-mine", "172.17.0.7")
    adapter = build(docker, aux_spec)

    assert adapter.session_key(request_from("172.17.0.7")) == f"swe_ci:{mine}"
    assert adapter.session_key(request_from("172.17.0.4")) == f"swe_ci:{other}"


def test_the_session_key_is_the_container_id_not_a_prompt_hash(aux_spec, docker):
    """SWE-CI's prompts are byte-identical across tasks, so this is the point.

    Two containers sending the SAME prompt must not share an aux result; under
    the ABC's default keying they would.
    """
    first = docker.create("swe-ci-1", "172.17.0.2")
    second = docker.create("swe-ci-2", "172.17.0.3")
    adapter = build(docker, aux_spec)

    keys = {
        adapter.session_key(request_from("172.17.0.2", ARCHITECT_PROMPT)),
        adapter.session_key(request_from("172.17.0.3", ARCHITECT_PROMPT)),
    }
    assert keys == {f"swe_ci:{first}", f"swe_ci:{second}"}


def test_a_recycled_ip_does_not_alias_the_previous_epochs_container(aux_spec, docker):
    """Docker reuses 172.17.0.x per epoch; caching on IP forever would break.

    This is the failure the container-ID key exists to prevent, so it is
    checked directly rather than trusted.
    """
    adapter = build(docker, aux_spec)

    epoch1 = docker.create("swe-ci-epoch1", "172.17.0.2")
    assert adapter.session_key(request_from("172.17.0.2")) == f"swe_ci:{epoch1}"

    # The epoch ends: SWE-CI removes the container and the next one takes the
    # address over.
    (docker.state / "containers" / epoch1 / "meta.json").unlink()
    epoch2 = docker.create("swe-ci-epoch2", "172.17.0.2")

    assert epoch2 != epoch1
    assert adapter.session_key(request_from("172.17.0.2")) == f"swe_ci:{epoch2}"


def test_a_continuation_request_does_not_re_resolve(aux_spec, docker):
    """session_key runs on EVERY request, including the re-application path.

    A continuation must therefore cost no `docker ps`, and -- more importantly
    -- must keep returning the same key even if resolution would now fail,
    because a mid-session key change silently stops the enhancement
    re-applying.
    """
    container = docker.create("swe-ci-1", "172.17.0.2")
    adapter = build(docker, aux_spec)
    key = adapter.session_key(request_from("172.17.0.2"))

    # Every container disappears; a continuation still maps to the same session.
    (docker.state / "containers" / container / "meta.json").unlink()
    assert adapter.session_key(continuation_from("172.17.0.2")) == key


def test_the_sole_running_container_is_used_when_no_ip_matches(aux_spec, docker):
    """The fallback for a `docker_cmd` with no per-container IP at all.

    Exact only at evolve.max_workers = 1, which is why the container COUNT
    gates it and not a config flag.
    """
    only = docker.create("swe-ci-1", "")  # no address at all
    adapter = build(docker, aux_spec)

    assert adapter.session_key(request_from("10.0.0.9")) == f"swe_ci:{only}"
    assert adapter._resolved["10.0.0.9"] == (only, "sole_container", "ip:10.0.0.9")


# -- session identity via headers -------------------------------------------


def test_a_session_header_is_used_when_present(aux_spec, docker):
    """Measured (tools/probe_opencode_headers.sh): opencode sends x-session-id
    on every request. When present it answers session_key directly -- no
    docker involvement at all, unlike every other case in this file."""
    adapter = build(docker, aux_spec)
    adapter._docker = "definitely-not-a-real-docker-xyz"  # must not be needed

    req = InboundRequest(body=user_only(ARCHITECT_PROMPT), headers={"x-session-id": "ses_abc123"})
    assert adapter.session_key(req) == "swe_ci:hdr:ses_abc123"


def test_session_header_names_are_checked_in_order(aux_spec, docker):
    """Configurable and ordered: opencode's header is undocumented behaviour
    tied to a version, not a guaranteed contract, and iFlow may send
    something else entirely."""
    adapter = build(
        docker, aux_spec, session_header_names=["x-foresight-session", "x-session-id"]
    )

    only_second = InboundRequest(
        body=user_only(ARCHITECT_PROMPT), headers={"x-session-id": "ses_1"}
    )
    assert adapter.session_key(only_second) == "swe_ci:hdr:ses_1"

    both = InboundRequest(
        body=user_only(ARCHITECT_PROMPT),
        headers={"x-foresight-session": "hdr_1", "x-session-id": "ses_1"},
    )
    assert adapter.session_key(both) == "swe_ci:hdr:hdr_1"


def test_no_matching_header_falls_back_to_container_resolution(aux_spec, docker):
    mine = docker.create("swe-ci-mine", "172.17.0.7")
    adapter = build(docker, aux_spec)

    req = InboundRequest(body=user_only(ARCHITECT_PROMPT), client_ip="172.17.0.7")
    assert adapter.session_key(req) == f"swe_ci:{mine}"


@pytest.mark.asyncio
async def test_a_session_header_does_not_replace_container_targeting(aux_spec, docker):
    """A session key is not a container address: run_aux must still resolve
    and `docker exec` into the container the client IP names, even when the
    session KEY itself came from a header, not from that resolution."""
    container = docker.create("swe-ci-1", "172.17.0.7")
    docker.seed(container)
    adapter = build(docker, aux_spec, answer_from="stdout")

    req = InboundRequest(
        body=user_only(ARCHITECT_PROMPT),
        client_ip="172.17.0.7",
        headers={"x-session-id": "ses_xyz"},
    )
    assert adapter.session_key(req) == "swe_ci:hdr:ses_xyz"

    result = await adapter.run_aux(req, aux_spec)
    assert result.provenance["container_id"] == container


@pytest.mark.asyncio
async def test_a_new_session_does_not_reuse_a_stale_container_id_from_the_same_ip(
    aux_spec, docker
):
    """Docker recycling an IP across sessions must not recycle the container ID.

    session_key() only re-resolves on the IP-keyed path; when a session
    header is present (the common case, opencode always sends one) it
    returns before ever calling _resolve(). Measured against real Docker:
    SWE-CI recreates a container per retry/phase, and the replacement can
    land on the SAME bridge IP the just-removed one had -- so a naive
    IP-only cache would hand session 2 session 1's already-gone container
    ID, and every docker exec into it would fail as "no such container".
    """
    first = docker.create("swe-ci-1", "172.17.0.7")
    docker.seed(first)
    adapter = build(docker, aux_spec, answer_from="stdout")

    req1 = InboundRequest(
        body=user_only(ARCHITECT_PROMPT),
        client_ip="172.17.0.7",
        headers={"x-session-id": "ses_1"},
    )
    assert adapter.session_key(req1) == "swe_ci:hdr:ses_1"
    result1 = await adapter.run_aux(req1, aux_spec)
    assert result1.provenance["container_id"] == first

    # session 1's container is gone; a new one for session 2 reuses its IP.
    (docker.state / "containers" / first / "meta.json").unlink()
    second = docker.create("swe-ci-2", "172.17.0.7")
    docker.seed(second)

    req2 = InboundRequest(
        body=user_only(ARCHITECT_PROMPT),
        client_ip="172.17.0.7",
        headers={"x-session-id": "ses_2"},
    )
    assert adapter.session_key(req2) == "swe_ci:hdr:ses_2"
    result2 = await adapter.run_aux(req2, aux_spec)
    assert result2.provenance["container_id"] == second


def test_several_containers_and_no_ip_match_is_unresolved(aux_spec, docker):
    """With nothing to disambiguate on, guessing would corrupt the measurement."""
    docker.create("swe-ci-1", "")
    docker.create("swe-ci-2", "")
    adapter = build(docker, aux_spec)

    assert adapter.session_key(request_from("10.0.0.9")).startswith(
        "swe_ci:unresolved:"
    )


def test_no_containers_at_all_is_unresolved_and_does_not_raise(aux_spec, docker):
    """session_key is on the hot path: a failure here must not become a 502."""
    adapter = build(docker, aux_spec)
    assert adapter.session_key(request_from("172.17.0.2")) == (
        "swe_ci:unresolved:172.17.0.2"
    )


def test_a_missing_docker_client_does_not_raise_from_session_key(aux_spec, docker):
    adapter = build(docker, aux_spec)
    adapter._docker = "definitely-not-a-real-docker-xyz"
    assert adapter.session_key(request_from("172.17.0.2")).startswith(
        "swe_ci:unresolved:"
    )


def test_inspect_output_that_is_not_json_is_survivable(aux_spec, docker):
    """A `docker_cmd` whose `inspect` is not Docker-shaped -- a real risk for
    any non-Docker client named there, not something to crash on.

    Resolution by IP is then impossible, but the sole-container path must
    still work.
    """
    only = docker.create("swe-ci-1", "172.17.0.2")
    adapter = build(docker, aux_spec)

    broken = docker.state / "broken-docker"
    broken.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "inspect" ]; then echo "CONTAINER ID   IMAGE"; exit 0; fi\n'
        f'exec "{docker.path}" "$@"\n'
    )
    broken.chmod(0o755)
    adapter._docker = str(broken)

    assert adapter.session_key(request_from("172.17.0.2")) == f"swe_ci:{only}"
    assert adapter._resolved["172.17.0.2"] == (only, "sole_container", "ip:172.17.0.2")


# -- phase classification --------------------------------------------------


def test_the_architect_prompt_is_classified_as_architect(aux_spec, docker):
    adapter = build(docker, aux_spec)
    assert adapter.phase(request_from("172.17.0.2", ARCHITECT_PROMPT)) == "architect"


def test_the_programmer_prompt_is_classified_as_programmer(aux_spec, docker):
    """Both prompts contain both words; only the identity line separates them.

    A bare grep for "architect" matches the programmer prompt too, via
    "collaborating closely with a senior software architect" -- which is what
    the first assertion pins.
    """
    adapter = build(docker, aux_spec)
    assert "architect" in PROGRAMMER_PROMPT.lower()
    assert adapter.phase(request_from("172.17.0.2", PROGRAMMER_PROMPT)) == "programmer"


def test_an_unrecognised_prompt_has_no_phase(aux_spec, docker):
    adapter = build(docker, aux_spec)
    assert adapter.phase(request_from("172.17.0.2", "fix the bug")) is None


# -- the docker exec argv --------------------------------------------------


def test_the_exec_argv_carries_the_workspace_and_an_isolated_home(aux_spec, docker):
    """`-w /app` then `-e HOME=...`, the shape SWE-CI's own call_opencode uses.

    HOME is asserted because sharing SWE-CI's would fold aux's tokens into the
    benchmark's own accounting -- reasoning that is cheap to check and expensive
    to discover wrong.
    """
    adapter = build(docker, aux_spec)
    argv = adapter._exec_argv("CID", ["opencode", "run"])

    assert argv[:4] == [str(docker.path), "exec", "-w", "/app"]
    assert "-e" in argv and "HOME=/tmp/aux-home" in argv
    # The container comes after every flag and before the command.
    assert argv[argv.index("CID") + 1 :] == ["opencode", "run"]


def test_exec_argv_adds_dash_i_only_when_stdin_is_needed(aux_spec, docker):
    """`docker exec` without `-i` never attaches the child's stdin to ours --

    measured against real Docker, not `fake_docker.py`, which pipes stdin
    through regardless of this flag and so cannot catch its absence itself.
    `_bootstrap_provider` pipes its payload through exactly this path; without
    `-i` the write silently produces a zero-byte file (`cat` sees immediate
    EOF) instead of failing, and aux's first request inside the container then
    fails with no indication the provider config was ever wrong.
    """
    adapter = build(docker, aux_spec)

    assert "-i" not in adapter._exec_argv("CID", ["opencode", "run"])
    assert "-i" in adapter._exec_argv("CID", ["sh", "-c", "cat > f"], stdin=True)


def test_placeholders_are_substituted_by_replace_not_by_format(aux_spec, docker):
    """SWE-CI's prompts are XML full of braces; format() would eat or raise.

    AdapterOptions documents str.replace deliberately, and the draft this
    replaced rendered each element through Jinja -- which leaves {prompt}
    untouched and would have passed the literal placeholder to the harness.
    """
    adapter = build(docker, aux_spec)
    prompt = 'a {brace} and a {{ jinja }} and a "quote"'
    argv = adapter._argv(prompt, "/tmp/answer.md")

    assert prompt in argv
    assert "{prompt}" not in argv
    assert "/tmp/answer.md" in argv


def test_the_workspace_is_configurable(aux_spec, docker):
    adapter = build(docker, aux_spec, workspace="/workspace/task")
    assert adapter._exec_argv("CID", ["x"])[:4] == [
        str(docker.path),
        "exec",
        "-w",
        "/workspace/task",
    ]


# -- the happy path --------------------------------------------------------


@pytest.mark.asyncio
async def test_aux_runs_in_the_container_and_the_answer_comes_back(aux_spec, docker):
    """End to end: resolve, exec, read the file back out of the container."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec)

    result = await adapter.run_aux(request_from("172.17.0.2"), aux_spec)

    assert result.source == "agent"
    # fake_agent reports what it actually read, so this proves aux saw /app.
    assert "code/pkg/sync.py" in result.text
    assert result.provenance["container_id"] == container
    assert result.provenance["resolved_by"] == "client_ip"
    assert result.provenance["fallback"] is None
    assert result.provenance["valid_condition"] is True
    assert result.provenance["guard"] == "intact"


@pytest.mark.asyncio
async def test_the_answer_can_come_from_stdout(aux_spec, docker):
    """opencode answers on stdout; the config declares which stream to trust."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec, answer_from="stdout")

    result = await adapter.run_aux(request_from("172.17.0.2"), aux_spec)

    assert "Plausible FUTURE tasks" in result.text
    assert result.provenance["answer_from"] == "stdout"


@pytest.mark.asyncio
async def test_a_fixed_answer_file_is_cleared_before_the_run(aux_spec, docker):
    """The stale-answer hazard: last run's text read as this run's output."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    root = docker.seed(container)
    stale = docker.rootfs(container) / "tmp" / "foresight-aux-answer.md"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("STALE ANSWER FROM AN EARLIER EPOCH\n")

    adapter = build(
        docker,
        aux_spec,
        answer_file="/tmp/foresight-aux-answer.md",
        extra_args=["--silent"],
        guard=None,
    )
    with pytest.raises(AuxFailure, match="no answer"):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)
    assert root.is_dir()  # the workspace itself was untouched


# -- provider bootstrap ------------------------------------------------------


def _opencode_home_files(docker: FakeDocker, container_id: str) -> tuple[Path, Path]:
    root = docker.rootfs(container_id) / "tmp" / "aux-home"
    return (
        root / ".local" / "share" / "opencode" / "auth.json",
        root / ".config" / "opencode" / "opencode.json",
    )


@pytest.mark.asyncio
async def test_provider_config_is_bootstrapped_under_auxs_home(aux_spec, docker):
    """Without this, aux has no provider at all and fails on its first
    request -- not a degraded mode, a missing feature. Mirrors SWE-CI's own
    setup_opencode (agents/opencode.py:16-67): same two files, same shape."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec)

    await adapter.run_aux(request_from("172.17.0.2"), aux_spec)

    auth_path, cfg_path = _opencode_home_files(docker, container)
    assert auth_path.is_file()
    assert cfg_path.is_file()

    auth = json.loads(auth_path.read_text())
    assert auth == {"custom": {"type": "api", "key": "dummy"}}

    cfg = json.loads(cfg_path.read_text())
    provider = cfg["provider"]["custom"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == DEFAULT_FORESIGHT_BASE_URL
    assert "aux-model" in provider["models"]


@pytest.mark.asyncio
async def test_bootstrap_respects_a_configured_home(aux_spec, docker):
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec, env={"HOME": "/tmp/a-different-aux-home"})

    await adapter.run_aux(request_from("172.17.0.2"), aux_spec)

    root = docker.rootfs(container) / "tmp" / "a-different-aux-home"
    assert (root / ".local" / "share" / "opencode" / "auth.json").is_file()
    assert (root / ".config" / "opencode" / "opencode.json").is_file()


def test_missing_foresight_base_url_is_a_config_error(aux_spec, docker):
    """Construction-time, not first-run-time: the whole point of checking it
    is that a config which cannot work should never reach a real container."""
    with pytest.raises(ConfigError, match="foresight_base_url"):
        build(docker, aux_spec, foresight_base_url=None)


@pytest.mark.asyncio
async def test_a_bootstrap_failure_raises_aux_failure(aux_spec, docker):
    """Isolated from resolution failure: ps/inspect succeed normally (so the
    container IS found), only exec -- the bootstrap's own mechanism -- fails.
    """
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec)

    refusing = docker.state / "refusing-docker"
    refusing.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "exec" ]; then echo "exec refused" >&2; exit 1; fi\n'
        f'exec "{docker.path}" "$@"\n'
    )
    refusing.chmod(0o755)
    adapter._docker = str(refusing)

    with pytest.raises(AuxFailure, match="bootstrap"):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)


# -- the guard -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_write_under_the_workspace_is_a_guard_violation(aux_spec, docker):
    """An edit below /app/code is copied out when the epoch ends.

    That is what makes the guard mandatory here and not merely prudent: the
    contamination would become part of the measured result silently.
    """
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec, extra_args=["--write-file", "code/oops.py"])

    with pytest.raises(GuardViolation, match="oops.py"):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)


@pytest.mark.asyncio
async def test_a_modification_without_a_size_change_is_still_caught(aux_spec, docker):
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec, extra_args=["--touch", "code/pkg/sync.py"])

    with pytest.raises(GuardViolation, match="sync.py"):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)


@pytest.mark.asyncio
async def test_contamination_beats_a_missing_answer(aux_spec, docker):
    """The write is what invalidates the instance; no answer is a failed request."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(
        docker, aux_spec, extra_args=["--write-file", "code/oops.py", "--silent"]
    )

    with pytest.raises(GuardViolation):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)


@pytest.mark.asyncio
async def test_the_answer_file_under_tmp_is_not_contamination(aux_spec, docker):
    """/tmp is never copied out of the container, so the answer belongs there."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec, answer_file="/tmp/foresight-aux-answer.md")

    result = await adapter.run_aux(request_from("172.17.0.2"), aux_spec)
    assert result.provenance["guard"] == "intact"


@pytest.mark.asyncio
async def test_the_guard_can_be_disabled(aux_spec, docker):
    """guard: null is the opt-out, per guards.py -- not a NoopGuard class."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(
        docker, aux_spec, guard=None, extra_args=["--write-file", "code/allowed.py"]
    )

    result = await adapter.run_aux(request_from("172.17.0.2"), aux_spec)
    assert result.provenance["guard"] is None


def test_an_unknown_guard_is_a_config_error(aux_spec, docker):
    adapter = build(docker, aux_spec, guard="nonexistent")
    with pytest.raises(ConfigError, match="unknown guard"):
        adapter._build_guard("CID")


# -- failure paths ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_timeout_is_an_aux_failure(aux_spec, docker):
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(
        docker, aux_spec, extra_args=["--sleep", "5"], timeout_s=0.5, guard=None
    )

    with pytest.raises(AuxFailure, match="exceeded"):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)


@pytest.mark.asyncio
async def test_an_empty_answer_is_an_aux_failure(aux_spec, docker):
    """A harness that printed only chatter must not pass as a future-task list."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec, extra_args=["--silent"], guard=None)

    with pytest.raises(AuxFailure, match="no answer"):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)


@pytest.mark.asyncio
async def test_a_missing_docker_client_is_an_aux_failure(aux_spec, docker):
    docker.create("swe-ci-1", "172.17.0.2")
    adapter = build(docker, aux_spec, guard=None)
    adapter.session_key(request_from("172.17.0.2"))  # resolve while it still works
    adapter._docker = "definitely-not-a-real-docker-xyz"

    with pytest.raises(AuxFailure, match="docker client not found"):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)


@pytest.mark.asyncio
async def test_a_container_that_died_mid_session_is_an_aux_failure(aux_spec, docker):
    """`docker exec` into a removed container: a real mid-run failure."""
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec, guard=None)
    adapter.session_key(request_from("172.17.0.2"))
    (docker.state / "containers" / container / "meta.json").unlink()

    with pytest.raises(AuxFailure):
        await adapter.run_aux(request_from("172.17.0.2"), aux_spec)


# -- the body-only degradation --------------------------------------------


@pytest.mark.asyncio
async def test_an_unresolved_container_degrades_to_a_body_only_call(aux_spec, docker):
    """The run stays alive, and the instance is marked unusable.

    Raising would abort SWE-CI's session and, after max_try, the task -- which
    would bias which tasks finish. The flag is what excludes it at analysis
    time; the plan is explicit that these must never be averaged in.
    """
    docker.create("swe-ci-1", "")
    docker.create("swe-ci-2", "")
    adapter = build(docker, aux_spec)

    result = await adapter.run_aux(request_from("10.0.0.9"), aux_spec)

    assert result.source == "chat_call"
    assert result.provenance["fallback"] == "body_only"
    assert result.provenance["valid_condition"] is False
    assert result.provenance["guard"] is None


@pytest.mark.asyncio
async def test_the_fallback_renders_the_adapters_own_prompt_variables(
    aux_spec, docker
):
    """StrictUndefined would otherwise turn the fallback itself into a crash.

    The configured aux_prompt mentions {{ workspace }} and {{ answer_file }},
    which the body-only helper knows nothing about -- so the adapter has to
    supply them even on the path where they are meaningless.
    """
    adapter = build(docker, aux_spec)
    result = await adapter.run_aux(request_from("10.0.0.9"), aux_spec)

    assert result.provenance["fallback"] == "body_only"
    # The prompt the stub backend was actually asked, not a claim about it.
    sent = adapter._aux_backend.calls[-1]["messages"][0]["content"]
    assert "/app" in sent


# -- the event loop --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_event_loop_keeps_serving_while_aux_runs(aux_spec, docker):
    """The defect that would have made M3 fail 100% of the time.

    Aux runs INSIDE the container and is pointed back at foresight as
    aux-model, so its own model calls must be served by this same loop while
    run_aux awaits it. A blocking subprocess.run deadlocks by construction:
    aux waits for a reply that cannot arrive until aux returns. This asserts
    the loop still makes progress -- with a blocking spawn, `ticks` stays at 0
    and the test fails.
    """
    container = docker.create("swe-ci-1", "172.17.0.2")
    docker.seed(container)
    adapter = build(docker, aux_spec, extra_args=["--sleep", "1"], guard=None)

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    beat = asyncio.ensure_future(heartbeat())
    try:
        result = await adapter.run_aux(request_from("172.17.0.2"), aux_spec)
    finally:
        beat.cancel()

    assert result.source == "agent"
    assert ticks > 3, "the event loop was blocked for the whole aux run"


# -- config wiring ---------------------------------------------------------


def test_the_shipped_config_builds_a_runtime(tmp_path):
    """configs/swe_ci.yaml must survive config.py, including _check_template.

    The aux_prompt is validated at startup against this adapter's declared
    aux_prompt_vars, so a template naming a variable the adapter does not
    supply fails here rather than on the first request.
    """
    from foresight.config import load_config

    source = (REPO / "configs" / "swe_ci.yaml").read_text()
    # The shipped file carries a deliberate placeholder for the SWE-CI checkout
    # and a harness that is not installed here; both are pointed at real things
    # so that what is under test is the config's shape.
    swe_ci_config = tmp_path / "config.toml"
    swe_ci_config.write_text('agent_name = "opencode"\n')
    source = source.replace(
        "/ABSOLUTE/PATH/TO/SWE-CI/config.toml", str(swe_ci_config)
    )

    path = tmp_path / "swe_ci.yaml"
    path.write_text(source)
    config = load_config(path)

    assert config.adapter.name == "swe_ci"
    assert config.guard == "manifest"

    runtime = _runtime_for(config)
    assert isinstance(runtime.adapter, SweCiAdapter)
    assert runtime.adapter._env["HOME"] == "/tmp/aux-home"


def _runtime_for(config):
    """Build a Runtime without letting it open a real trace file."""
    import httpx

    from foresight.config import Runtime

    config = config.model_copy(update={"trace": config.trace.model_copy(update={"path": None})})
    return Runtime(config, http_client=httpx.AsyncClient())
