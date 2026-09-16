"""SweCiAdapter -- aux is a second harness, `docker exec`'d into the live container.

Milestone 3. The target harness runs *inside* SWE-CI's task container, so the
only way aux can read the same code is to enter the same container. That is the
whole of this file: resolve which container a request came from, `docker exec` a
second harness into it, read the answer back out, and check the workspace was
not touched.

Why the container ID is the session key
---------------------------------------
SWE-CI renders both of its prompts from one static Jinja template whose only
variables are ``role`` and ``mode`` (``benchmark/run.py:78-79``). Every architect
call, for all 100 tasks, is byte-identical -- so the ABC's prompt hash would
collapse every concurrent session onto one key and hand epoch 7 of task B the
aux answer computed for epoch 1 of task A. The container ID is unique per
container *instance*, and SWE-CI starts a fresh container per session
(``run.py:101``, ``run.py:126``, removed in a ``finally``), so container ID and
session are one-to-one even when Docker recycles the ``172.17.0.x`` address.

How the container is resolved, and the two ways it can go wrong
---------------------------------------------------------------
SWE-CI passes no ``--network`` flag (``utils/docker.py:113-118``), so its
containers sit on Docker's default bridge and each has its own ``172.17.0.x``,
which foresight sees as the client address. ``_resolve`` walks ``docker ps -q``
and one batched ``docker inspect`` and matches ``NetworkSettings.IPAddress``.

*One inspect, not N.* ``docker inspect a b c`` returns a JSON array, so
resolution costs exactly two process spawns however many containers are running.
That matters because it happens on the synchronous ``session_key`` path.

*The sole-container path.* Not every `docker_cmd` sits on a real Docker bridge
with per-container IPs (a udocker translation shim was built and tested against
this cluster's Slurm nodes and found not to support the pattern this adapter
needs at all -- see ``foresight-design-plan.md`` risk 1; not shipped, since
``SweCiAdapter`` targets real Docker). When IP matching finds nothing and
exactly **one** container is running, that container is the caller by
elimination and is used, with ``resolved_by: "sole_container"`` in the
provenance. This is exact only when SWE-CI runs with ``evolve.max_workers = 1``
and wrong above it -- which is why the number of running containers is what
gates it, not a config flag. With several containers running and no IP match
there is nothing to disambiguate on, and ``run_aux`` degrades to the body-only
call below.

*The fallback is not a valid condition.* A body-only aux call for SWE-CI sees a
request body that is byte-identical across all 100 tasks, so it has no
task-specific content to work from. It keeps the run alive and makes the failure
visible; ``provenance["fallback"] == "body_only"`` is how those instances are
**excluded** at analysis time, never averaged in.

Why nothing here may block the event loop
-----------------------------------------
The aux agent runs inside the container and is pointed back at foresight as
``aux-model``, so its own model calls must be served by this same event loop
*while* ``run_aux`` is awaiting it. A blocking ``subprocess.run`` for the agent
would therefore deadlock by construction, not merely be slow: aux would hang
until its own timeout fired, every time. The agent and the answer read go
through ``asyncio.create_subprocess_exec``; the guard's runner is synchronous by
Protocol and is called under ``asyncio.to_thread``.

The one exception is ``session_key``, which the ABC defines as synchronous and
which runs *before* any aux agent exists -- two short-lived spawns, at most
twice per session.

The HOME override is mandatory, and by itself is not sufficient
-----------------------------------------------------------------
opencode keeps its session DB under ``$HOME/.local/share/opencode``, and SWE-CI
copies ``opencode.db`` out of its own ``HOME=/opt/agent/home`` to bill the
target's tokens (``agents/opencode.py:71-145``). Sharing that HOME would fold
aux's tokens into the benchmark's own accounting, so a config that asks for it
is refused at startup.

That override only works, though, if ``adapter.agent_cmd`` names the real
binary directly. ``Dockerfile.opencode`` generates ``/usr/local/bin/opencode``
as a wrapper that *unconditionally* does ``export HOME="$BASE/home"`` (plus
``XDG_CACHE_HOME``, ``NPM_CONFIG_PREFIX``, ``PATH``), with no
``${OPENCODE_HOME:-...}`` escape hatch -- so ``docker exec -e HOME=/tmp/aux-home
... opencode`` silently lands back in ``/opt/agent/home`` regardless of the
``-e`` flag. ``configs/swe_ci.yaml`` points ``agent_cmd`` at
``/opt/agent/npm-global/bin/opencode`` -- the real binary, bypassing the
wrapper -- and sets the whole environment the wrapper would have (``env`` in
that config), which requires no code change here: ``_exec_argv`` already
forwards every ``adapter.env`` entry as its own ``-e``.

The second, separate requirement: aux has no provider config until this
adapter writes one. ``_bootstrap_provider`` replicates SWE-CI's own
``setup_opencode`` (``agents/opencode.py:16-67``) under aux's ``$HOME`` --
``auth.json`` and ``opencode.json`` pointing the ``custom`` provider at
``adapter.foresight_base_url`` (this server, reachable from inside the
container -- typically the Docker bridge gateway) with ``aux.served_name`` as
the model. Without it aux fails on its very first request: not a degraded
mode, a missing feature, which is why construction raises ``ConfigError`` if
``foresight_base_url`` is unset rather than deferring the failure to the first
real run.

Timeout, and what it can and cannot kill
----------------------------------------
A timeout kills the local ``docker exec`` client. The harness *inside* the
container may survive it, and there is deliberately no ``pkill`` chaser: the
target agent runs the same binary under the same name in the same container, so
matching on the process name risks killing the thing being measured. The orphan
is bounded instead -- SWE-CI removes the container in a ``finally`` at the end
of the session (``run.py:119-120``), and the request has already failed with
``AuxFailure``, so the instance is excluded either way.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import ClassVar

from jinja2 import StrictUndefined, Template

from ..context import InboundRequest
from ..errors import AuxFailure, ConfigError, GuardViolation
from ..guards import ManifestGuard
from ..llm import ModelSpec
from .base import AuxResult, CallerAdapter

log = logging.getLogger(__name__)

GUARDS = {"manifest": ManifestGuard}

#: SWE-CI's own HOME inside the task container (``agents/opencode.py::HOME_DIR``).
#: Aux must never share it; see above.
SWE_CI_HOME = "/opt/agent/home"

#: Where aux's harness keeps its state instead.
DEFAULT_AUX_HOME = "/tmp/aux-home"

#: Where SWE-CI copies the task into (``run.py:102``). Configurable via
#: ``adapter.workspace``, which for this adapter is a path *inside* the
#: container rather than on the host.
DEFAULT_WORKSPACE = "/app"

#: The harnesses SWE-CI can drive (``benchmark/tools.py:176-179``). The task
#: image ships exactly one of them, chosen by ``agent_name``.
KNOWN_HARNESSES = ("opencode", "iflow")

#: Phase markers, read off the rendered prompts rather than guessed: both
#: prompts mention both roles ("collaborating closely with a senior
#: programmer"), so a bare grep for "architect" matches the programmer prompt
#: too. The *identity* sentence is the only discriminator, and it is stable
#: across both the tdd and rdd renderings.
ARCHITECT_MARKER = "You are a senior software architect"
PROGRAMMER_MARKER = "You are a senior programmer"

#: ``agent_name = "opencode"`` out of SWE-CI's config.toml. A regex rather than
#: a TOML parser because ``tomllib`` is 3.11+, this reads exactly one top-level
#: string key, and taking a dependency for a startup sanity check would be out
#: of proportion.
_AGENT_NAME_RE = re.compile(r"""(?m)^[ \t]*agent_name[ \t]*=[ \t]*["']([^"']+)["']""")

#: How long to wait, after killing a timed-out `docker exec`, for its pipes to
#: close. Only the local client is killed, so the in-container harness may
#: still hold the write end -- this bounds the wait rather than leaking the
#: transport to the garbage collector.
_DRAIN_S = 5.0


class SweCiAdapter(CallerAdapter):
    """Aux explores SWE-CI's live task container via a second harness."""

    name: ClassVar[str] = "swe_ci"

    #: workspace and answer_file are container paths here, and are substituted
    #: into the prompt so the agent can be told where it is and where to write.
    aux_prompt_vars: ClassVar[set[str]] = {"prompt", "system", "workspace", "answer_file"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        options = self._options

        if not options.agent_cmd:
            raise ConfigError(
                "adapter.agent_cmd is required by the 'swe_ci' adapter: aux reaches "
                "the codebase through the same harness the image ships, so give the "
                "command to run inside the container"
            )
        self._agent_cmd = list(options.agent_cmd)

        if not options.foresight_base_url:
            raise ConfigError(
                "adapter.foresight_base_url is required by the 'swe_ci' adapter: "
                "aux's harness runs inside the container and needs this server's "
                "own address to bootstrap its provider config, or it has nothing "
                "to call"
            )
        self._foresight_base_url = options.foresight_base_url

        # A container path, not a host path -- so unlike LocalAdapter there is
        # nothing on this filesystem to check it against.
        self._workspace = (options.workspace or DEFAULT_WORKSPACE).rstrip("/") or "/"

        self._timeout_s = options.timeout_s
        self._answer_file = options.answer_file
        self._answer_from = options.answer_from
        self._guard_ignore = list(options.guard_ignore)
        self._docker = options.docker_cmd
        self._session_header_names = list(options.session_header_names)

        env = dict(options.env)
        env.setdefault("HOME", DEFAULT_AUX_HOME)
        if env["HOME"].rstrip("/") == SWE_CI_HOME:
            raise ConfigError(
                f"adapter.env.HOME must not be {SWE_CI_HOME!r}: that is SWE-CI's own "
                "agent HOME, and it reads opencode.db out of it to bill the target's "
                "tokens -- sharing it folds aux's tokens into the benchmark's results"
            )
        self._env = env

        self._check_agent_name(options.swe_ci_config)

        # ip -> (container_id, how_it_was_resolved). Re-resolved at every
        # session start rather than trusted forever: Docker recycles
        # 172.17.0.x across epochs, so a permanent IP cache would hand epoch 2
        # epoch 1's container ID and defeat the reason for keying on the ID.
        self._resolved: dict[str, tuple[str, str]] = {}

    # -- startup checks ----------------------------------------------------

    def _check_agent_name(self, config_path: str | None) -> None:
        """Fail loudly if agent_cmd names a harness the image does not ship.

        Binding constraint 2: we define no tools, so aux must use the harness
        already in the image -- ``Dockerfile.opencode`` installs whichever
        ``AGENT_NPM_PKG`` matched ``agent_name``, and nothing else is there.
        Getting this wrong produces "command not found" hundreds of requests
        into a run; checking it costs one file read at startup.
        """
        if not config_path:
            log.warning(
                "adapter.swe_ci_config is unset: cannot confirm adapter.agent_cmd "
                "matches SWE-CI's agent_name, so a harness mismatch will only "
                "surface as a failed aux run"
            )
            return

        try:
            text = Path(config_path).expanduser().read_text()
        except OSError as exc:
            raise ConfigError(
                f"adapter.swe_ci_config {config_path!r} is not readable: {exc}"
            ) from exc

        match = _AGENT_NAME_RE.search(text)
        if match is None:
            raise ConfigError(
                f"no top-level agent_name in {config_path!r}; it should be SWE-CI's "
                "config.toml"
            )
        declared = match.group(1).strip()

        named = [
            harness
            for element in self._agent_cmd
            for harness in KNOWN_HARNESSES
            if Path(element).name == harness
        ]
        if not named:
            raise ConfigError(
                f"adapter.agent_cmd names none of {list(KNOWN_HARNESSES)}, but the "
                f"SWE-CI image at {config_path!r} ships only {declared!r}; aux must "
                "use the harness that is already in the container"
            )
        if declared not in named:
            raise ConfigError(
                f"adapter.agent_cmd runs {named[0]!r} but SWE-CI's agent_name is "
                f"{declared!r} ({config_path}); the task image ships exactly one "
                "harness, so aux cannot run the other one"
            )

    # -- caller-specific behaviour ----------------------------------------

    def session_key(self, req: InboundRequest) -> str:
        """A session header if opencode sends one; the container ID otherwise.

        Called on *every* request, including the builder's re-application path,
        so it must be cheap and must not raise -- a resolution failure mid
        session would turn a working session into a 502.

        Measured (``tools/probe_opencode_headers.sh``): opencode sends
        ``x-session-id`` on every request, stable within one ``opencode run``
        invocation and distinct across invocations, with zero configuration.
        When present this answers the question directly -- no subprocess at
        all, on the hottest path in the adapter. ``adapter.session_header_names``
        lists candidates in order, since this is undocumented behaviour tied to
        an opencode version, not a guaranteed contract; the task image may ship
        a different one, or iFlow, which may not send anything comparable.

        Falling back to container-ID resolution when no listed header is
        present: shells out only at session start (two spawns), reads the
        remembered value otherwise, and returns a stable ``unresolved:`` key
        when resolution fails so the body-only fallback still re-applies for
        the rest of that session.
        """
        for name in self._session_header_names:
            value = req.headers.get(name)
            if value:
                return f"swe_ci:hdr:{value}"

        ip = req.client_ip or ""
        entry = (
            self._resolve(ip) if self.is_session_start(req) else self._resolved.get(ip)
        )
        if entry is None:
            return f"swe_ci:unresolved:{ip or 'no-client-ip'}"
        return f"swe_ci:{entry[0]}"

    def phase(self, req: InboundRequest) -> str | None:
        """architect or programmer -- trace metadata only, never control flow.

        Expect the effect to differ between the two: the architect prompt
        explicitly constrains scope ("1 to 5 items", "core contradictions"), so
        future-task prompting pushes against its own instructions. Because the
        phase is on every trace row, that can be tested by splitting the
        results afterwards at no extra cost.
        """
        text = f"{req.system_text()}\n{req.first_user_content()}"
        if ARCHITECT_MARKER in text:
            return "architect"
        if PROGRAMMER_MARKER in text:
            return "programmer"
        return super().phase(req)

    async def run_aux(self, req: InboundRequest, aux: ModelSpec) -> AuxResult:
        ip = req.client_ip or ""
        entry = self._resolved.get(ip)
        if entry is None:
            # session_key already tried; retry here (off the loop) because a
            # container that was still starting a moment ago may be up now.
            entry = await asyncio.to_thread(self._resolve, ip)
        if entry is None:
            return await self._body_only_fallback(req, aux, ip)

        container_id, resolved_by = entry
        await self._bootstrap_provider(container_id, aux)
        answer_path = self._answer_path()
        prompt = self._render_prompt(req, answer_path)

        # Stale answers are a real hazard with a fixed answer_file: last run's
        # text would be read as this run's output, silently. Unnecessary on the
        # default per-run path, which is a fresh name every time.
        if self._answer_file:
            await self._exec(container_id, ["rm", "-f", answer_path], timeout_s=30.0)

        guard = self._build_guard(container_id)
        before = None
        if guard is not None:
            before = await asyncio.to_thread(guard.snapshot)

        started = time.monotonic()
        argv = self._exec_argv(container_id, self._argv(prompt, answer_path))
        code, stdout, stderr, timed_out = await self._spawn(argv, self._timeout_s)
        duration = time.monotonic() - started

        # Checked before the answer is even read: if aux both contaminated the
        # workspace and produced text, the contamination is the thing that
        # invalidates the instance. Under SWE-CI an edit below /app/code is
        # copied out when the epoch ends and silently becomes part of the
        # measured result.
        if guard is not None:
            changed = await asyncio.to_thread(guard.verify, before)
            if changed:
                raise GuardViolation(changed, f"{container_id[:12]}:{self._workspace}")

        if timed_out:
            raise AuxFailure(
                f"aux agent exceeded {self._timeout_s}s in container "
                f"{container_id[:12]} (command: {self._agent_cmd[0]}); {_tail(stderr)}"
            )

        text = await self._read_answer(container_id, answer_path, stdout)
        if not text.strip():
            where = "stdout" if self._answer_from == "stdout" else answer_path
            raise AuxFailure(
                f"aux agent produced no answer on {where} in container "
                f"{container_id[:12]} (exit {code}); "
                f"stdout: {_tail(stdout, 200)} stderr: {_tail(stderr, 200)}"
            )

        return AuxResult(
            text=text,
            source="agent",
            provenance={
                "adapter": self.name,
                "container_id": container_id,
                "resolved_by": resolved_by,
                "workspace": self._workspace,
                "command": self._agent_cmd[0],
                "returncode": code,
                "duration_s": round(duration, 3),
                "answer_from": self._answer_from,
                "guard": None if guard is None else "intact",
                "fallback": None,
                "valid_condition": True,
                "stdout_tail": _tail(stdout),
            },
        )

    # -- container resolution ---------------------------------------------

    def _resolve(self, ip: str) -> tuple[str, str] | None:
        """(container_id, how) for this client IP, or None. Never raises.

        Two spawns: ``docker ps -q``, then one batched ``docker inspect``.
        """
        ids = self._running_container_ids()
        if not ids:
            log.warning("swe_ci: no running containers, cannot resolve %r", ip)
            self._resolved.pop(ip, None)
            return None

        if ip:
            for container_id, address in self._container_addresses(ids):
                if address and address == ip:
                    return self._remember(ip, container_id, "client_ip")

        if len(ids) == 1:
            # udocker and any host-network runtime land here: no per-container
            # IP exists to match on, but with one container there is nothing to
            # disambiguate. Exact under evolve.max_workers = 1.
            log.info(
                "swe_ci: no IP match for %s; using the sole running container %s",
                ip or "(no client ip)",
                ids[0][:12],
            )
            return self._remember(ip, ids[0], "sole_container")

        log.warning(
            "swe_ci: %d containers running and none has address %r; aux will fall "
            "back to a body-only call, which is not a valid condition",
            len(ids),
            ip,
        )
        self._resolved.pop(ip, None)
        return None

    def _remember(self, ip: str, container_id: str, how: str) -> tuple[str, str]:
        """Keep the verdict for the rest of this session, and no longer.

        Keyed on IP, but every session start overwrites it, so a recycled
        172.17.0.x cannot alias epoch 1's container onto epoch 2.
        """
        entry = (container_id, how)
        self._resolved[ip] = entry
        return entry

    def _running_container_ids(self) -> list[str]:
        code, out, err = self._run([self._docker, "ps", "-q"], timeout_s=30.0)
        if code != 0:
            log.error(
                "swe_ci: `%s ps -q` failed (exit %s): %s",
                self._docker,
                code,
                _tail(err, 200),
            )
            return []
        return out.split()

    def _container_addresses(self, ids: list[str]) -> list[tuple[str, str]]:
        """[(id, ip)] from one ``docker inspect`` over every running container."""
        code, out, err = self._run([self._docker, "inspect", *ids], timeout_s=30.0)
        if code != 0:
            log.error(
                "swe_ci: `%s inspect` failed (exit %s): %s",
                self._docker,
                code,
                _tail(err, 200),
            )
            return []
        try:
            records = json.loads(out)
        except json.JSONDecodeError as exc:
            # A warning, not a crash: a `docker_cmd` pointed at something whose
            # `inspect` is not Docker-shaped leaves the sole-container path as
            # the only way through, which is better than no aux at all.
            log.warning("swe_ci: `%s inspect` output is not JSON: %s", self._docker, exc)
            return []
        if not isinstance(records, list):
            return []

        found: list[tuple[str, str]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            container_id = record.get("Id") or record.get("ID") or ""
            settings = record.get("NetworkSettings")
            address = settings.get("IPAddress", "") if isinstance(settings, dict) else ""
            if container_id:
                found.append((container_id, address or ""))
        return found

    # -- the body-only degradation ----------------------------------------

    async def _body_only_fallback(
        self, req: InboundRequest, aux: ModelSpec, ip: str
    ) -> AuxResult:
        """Keep the run alive, and mark the instance as unusable.

        Recorded rather than raised because a 502 here aborts SWE-CI's whole
        session and, after ``max_try`` retries, the task -- which would bias
        *which* tasks finish. The flag is what excludes it at analysis time:
        the same "record a validity flag, filter later" policy as the weak-aux
        verdict and the guard verdict.
        """
        log.warning(
            "swe_ci: unresolved container for client %s -- degrading to a body-only "
            "aux call. SWE-CI's request body carries no task-specific content, so "
            "this instance must be excluded from results.",
            ip or "(no client ip)",
        )
        result = await self._body_only_aux(
            req, aux, workspace=self._workspace, answer_file=""
        )
        result.provenance.update(
            {
                "adapter": self.name,
                "fallback": "body_only",
                "container_resolution": "failed",
                "client_ip": ip or None,
                "guard": None,
                "valid_condition": False,
            }
        )
        return result

    # -- mechanics ---------------------------------------------------------

    async def _bootstrap_provider(self, container_id: str, aux: ModelSpec) -> None:
        """Write aux's own opencode provider config under aux's ``$HOME``.

        Mirrors SWE-CI's own ``setup_opencode`` (``agents/opencode.py:16-67``)
        exactly -- same two files, same JSON shape, same
        ``docker exec -i -u root <cid> sh -c "mkdir -p D && cat > D/F"`` stdin
        trick -- but for aux's home, not SWE-CI's, and pointed at this server
        (``adapter.foresight_base_url``) rather than at a model endpoint
        directly, so aux's own model calls bypass the pipeline and land in the
        same trace (design plan, "Server"). Without this aux has no provider
        at all and fails on its very first request.

        Runs once per resolved container: ``run_aux`` is cached per session by
        the pipeline, and a container is one-to-one with a session, so this
        never repeats work.
        """
        home = self._env.get("HOME", DEFAULT_AUX_HOME).rstrip("/") or "/"
        auth_dir = f"{home}/.local/share/opencode"
        cfg_dir = f"{home}/.config/opencode"

        auth_payload = json.dumps({"custom": {"type": "api", "key": "dummy"}}, indent=4) + "\n"
        cfg_payload = (
            json.dumps(
                {
                    "$schema": "https://opencode.ai/config.json",
                    "permission": "allow",
                    "provider": {
                        "custom": {
                            "npm": "@ai-sdk/openai-compatible",
                            "name": "custom",
                            "options": {"baseURL": self._foresight_base_url},
                            "models": {aux.served_name: {"name": aux.served_name}},
                        }
                    },
                },
                indent=4,
            )
            + "\n"
        )

        for target_dir, filename, payload in (
            (auth_dir, "auth.json", auth_payload),
            (cfg_dir, "opencode.json", cfg_payload),
        ):
            argv = self._exec_argv(
                container_id,
                ["sh", "-c", f"mkdir -p {target_dir} && cat > {target_dir}/{filename}"],
                user="root",
            )
            code, _out, err, timed_out = await self._spawn(
                argv, timeout_s=30.0, input_data=payload.encode()
            )
            if timed_out or code != 0:
                raise AuxFailure(
                    f"could not bootstrap aux's opencode provider config in "
                    f"container {container_id[:12]} ({filename}, exit {code}): "
                    f"{_tail(err, 200)}"
                )

    def _answer_path(self) -> str:
        """A container path. Fresh per run unless configured.

        Per-run by default, as with LocalAdapter: SWE-CI runs up to 16 tasks
        concurrently, and a fixed path would let one run read another's answer.
        Directly under /tmp so the harness needs no mkdir, and because /tmp is
        never copied out of the container -- writing the answer into ``/app``
        instead would make the guard correctly flag aux's own answer as
        contamination.
        """
        if self._answer_file:
            return self._answer_file
        return f"/tmp/foresight-aux-{uuid.uuid4().hex[:12]}.md"

    def _render_prompt(self, req: InboundRequest, answer_path: str) -> str:
        return Template(self._aux_prompt, undefined=StrictUndefined).render(
            prompt=req.first_user_content(),
            system=req.system_text(),
            workspace=self._workspace,
            answer_file=answer_path,
        )

    def _argv(self, prompt: str, answer_path: str) -> list[str]:
        """Substitute placeholders positionally.

        str.replace, never str.format and never Jinja: SWE-CI's prompts are XML
        full of braces, and both alternatives would either raise or silently
        eat them. Same rule as LocalAdapter, and the one ``AdapterOptions``
        documents.
        """
        substitutions = {
            "{prompt}": prompt,
            "{answer_file}": answer_path,
            "{workspace}": self._workspace,
        }
        argv = []
        for element in self._agent_cmd:
            for placeholder, value in substitutions.items():
                element = element.replace(placeholder, value)
            argv.append(element)
        return argv

    def _exec_argv(
        self, container_id: str, command: list[str], *, user: str | None = None
    ) -> list[str]:
        """``docker exec [-u USER] -w <workspace> -e K=V ... <cid> <command>``.

        Shaped like SWE-CI's own ``call_opencode`` (``-w``, then ``-e``, then
        the container), so the aux harness starts in the same working directory
        the target's does and reads the same tree. ``user="root"`` matches
        SWE-CI's own ``setup_opencode`` (``agents/opencode.py:60,65``), needed
        to write config files under a fresh ``$HOME`` before aux's own user
        exists in the container's passwd database.
        """
        flags: list[str] = []
        if user is not None:
            flags += ["-u", user]
        flags += ["-w", self._workspace]
        for key, value in self._env.items():
            flags += ["-e", f"{key}={value}"]
        return [self._docker, "exec", *flags, container_id, *command]

    async def _read_answer(
        self, container_id: str, answer_path: str, stdout: str
    ) -> str:
        """Whichever stream the config declared. No fallback between them.

        A harness told to write a file and failing to still prints its own
        progress chatter, and accepting that as the future-task list would
        enhance the prompt with a banner -- an instance in neither arm. The
        file comes back through ``docker exec cat`` rather than ``docker cp``:
        one spawn, no host temp file, nothing left behind.
        """
        if self._answer_from == "stdout":
            return stdout
        code, out, _err, timed_out = await self._spawn(
            self._exec_argv(container_id, ["cat", answer_path]), timeout_s=60.0
        )
        if timed_out or code != 0:
            # Missing file, most often. Treated as "no answer" rather than as a
            # distinct failure so the empty-answer path reports it once, with
            # the harness's own stdout and stderr attached.
            return ""
        return out

    def _build_guard(self, container_id: str):
        """A guard whose reach is a ``docker exec`` into *this* container.

        guards.py takes a runner precisely because what varies between callers
        is not the algorithm but where the workspace lives. Built per run
        rather than per adapter: the container ID is only known here, and every
        session gets a new container.
        """
        name = self._guard_name
        if name is None:
            return None
        try:
            cls = GUARDS[name]
        except KeyError:
            raise ConfigError(
                f"unknown guard {name!r}; known: {sorted(GUARDS)}"
            ) from None
        return cls(
            self._workspace,
            functools.partial(self._run_in_container, container_id),
            self._guard_ignore,
        )

    def _run_in_container(self, container_id: str, command: str) -> tuple[int, str]:
        """The guard's reach into this workspace: a shell inside the container.

        Synchronous by Protocol; callers wrap it in ``asyncio.to_thread`` so a
        large tree does not stall the event loop.
        """
        code, out, _err = self._run(
            self._exec_argv(container_id, ["sh", "-c", command]),
            timeout_s=max(60.0, min(self._timeout_s, 600.0)),
        )
        return code, out

    def _run(self, argv: list[str], *, timeout_s: float) -> tuple[int, str, str]:
        """A short, bounded, blocking spawn. Never raises.

        Only for the two paths that cannot be async: ``session_key``, which the
        ABC defines as synchronous, and the guard's runner, which the Protocol
        defines as synchronous and which is always called under
        ``asyncio.to_thread``. The aux agent itself must never come through
        here -- see the module docstring.
        """
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s
            )
        except FileNotFoundError:
            log.error(
                "swe_ci: %r not found on PATH (adapter.docker_cmd names the client)",
                argv[0],
            )
            return 127, "", f"{argv[0]}: not found"
        except subprocess.TimeoutExpired:
            log.error("swe_ci: %r timed out after %ss", " ".join(argv[:3]), timeout_s)
            return 124, "", "timed out"
        except OSError as exc:
            log.error("swe_ci: could not run %r: %s", argv[0], exc)
            return 126, "", str(exc)
        return completed.returncode, completed.stdout, completed.stderr

    async def _spawn(
        self, argv: list[str], timeout_s: float, *, input_data: bytes | None = None
    ) -> tuple[int | None, str, str, bool]:
        """(returncode, stdout, stderr, timed_out), without blocking the loop.

        ``input_data`` feeds the process's stdin then closes it -- the same
        stdin-JSON pattern SWE-CI's own ``setup_opencode`` uses to write
        ``auth.json``/``opencode.json`` (``docker exec -i ... sh -c "cat >
        file"``), needed by ``_bootstrap_provider`` below.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if input_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AuxFailure(
                f"docker client not found: {argv[0]!r} (adapter.docker_cmd names it)"
            ) from exc
        except OSError as exc:
            raise AuxFailure(f"could not run {argv[0]!r}: {exc}") from exc

        # communicate() is driven through its own task so that on a timeout it
        # can be allowed to finish after the kill. Awaiting it is what closes
        # the pipes: `docker exec` leaves the in-container harness holding the
        # write end, so without this the transport is only reclaimed at GC --
        # which on a long run means leaked file descriptors, one per timeout.
        communicate = asyncio.ensure_future(process.communicate(input_data))
        try:
            out, err = await asyncio.wait_for(asyncio.shield(communicate), timeout_s)
        except asyncio.TimeoutError:
            # Kills the local client only; see the module docstring on why there
            # is no in-container chaser.
            process.kill()
            try:
                await asyncio.wait_for(communicate, timeout=_DRAIN_S)
            except (asyncio.TimeoutError, asyncio.CancelledError, OSError):
                # The orphan is still holding the pipe. Bounded and rare, and
                # the request is failing anyway.
                communicate.cancel()
            return None, "", "", True

        return process.returncode, _decode(out), _decode(err), False

    async def _exec(
        self, container_id: str, command: list[str], *, timeout_s: float
    ) -> None:
        """Best-effort housekeeping inside the container."""
        await self._spawn(self._exec_argv(container_id, command), timeout_s)


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text
