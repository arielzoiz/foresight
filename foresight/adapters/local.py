"""LocalAdapter -- aux is a real agent, exploring a real repo on this host.

Milestone 2. The first adapter where aux *sees the codebase*: instead of one
chat call over the request body, ``run_aux`` spawns a harness as a subprocess
with ``cwd`` set to the configured repo, lets it explore, and reads back what it
wrote.

Why a configured command and not an integration
-----------------------------------------------
Binding constraint: *we define no tools*. Aux reaches the codebase through a
real agent harness, never through machinery we invent -- otherwise the aux
condition is not comparable to the target condition, which always runs through
a harness. So ``agent_cmd`` is argv, and the harness is whatever the config
names. ``tools/fake_agent.py`` stands in for one in tests, the same way
``tools/fake_upstream.py`` stands in for a model.

Where the answer comes from, and why it is declared
---------------------------------------------------
By default the agent is told to write to ``{answer_file}`` and we read it back,
mirroring SWE-CI's own requirement.xml pattern. Stdout carries the harness's
chatter -- tool calls, progress, banners -- so scraping an answer out of it is
guesswork that breaks whenever the harness changes its output.

``adapter.answer_from`` says which stream to trust, and there is deliberately no
fallback between them. An earlier version fell back from file to stdout, which
looked forgiving and was not: a harness that ignored the instruction still
prints progress lines, so the fallback quietly turned "aux produced nothing" into
"aux produced two lines of banner text", and that text then went into the target
prompt as the future-task list. An instance enhanced with a banner is in neither
experimental arm. Stdout is still captured into provenance either way, because it
is what you need to see when a run fails.

Why this is the milestone that needs a guard
--------------------------------------------
No container. The aux agent runs directly on the filesystem the target agent
will then edit, and it is read-only *by prompt instruction only*. This is the
one case where the read-only guarantee protects something that cannot be thrown
away, so a detected modification fails the request -- see errors.GuardViolation
and the argument in guards.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import ClassVar

from jinja2 import StrictUndefined, Template

from ..context import InboundRequest
from ..errors import AuxFailure, ConfigError, GuardViolation
from ..guards import ManifestGuard
from ..llm import ModelSpec
from .base import AuxResult, CallerAdapter

GUARDS = {"manifest": ManifestGuard}


class LocalAdapter(CallerAdapter):
    """Aux explores a local repo via a harness subprocess."""

    name: ClassVar[str] = "local"

    #: workspace and answer_file are substituted into the prompt so the agent
    #: can be told where it is and where to write.
    aux_prompt_vars: ClassVar[set[str]] = {"prompt", "system", "workspace", "answer_file"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        options = self._options

        if not options.workspace:
            raise ConfigError("adapter.workspace is required by the 'local' adapter")
        workspace = Path(options.workspace).expanduser()
        if not workspace.is_dir():
            raise ConfigError(
                f"adapter.workspace {str(workspace)!r} is not a directory"
            )
        self._workspace = workspace.resolve()

        if not options.agent_cmd:
            raise ConfigError(
                "adapter.agent_cmd is required by the 'local' adapter: aux reaches "
                "the codebase through a real harness, so give the command to run"
            )
        self._agent_cmd = list(options.agent_cmd)
        self._timeout_s = options.timeout_s
        self._env_overrides = dict(options.env)
        self._answer_file = options.answer_file
        self._answer_from = options.answer_from
        self._guard = self._build_guard(self._guard_name)

    def _build_guard(self, name: str | None):
        """The adapter builds its own guard, because it owns the reach.

        guards.py is explicit that what varies is not the algorithm but where the
        workspace lives, so the guard takes an "execute this in the workspace"
        callable -- which only the adapter can supply.
        """
        if name is None:
            return None
        try:
            cls = GUARDS[name]
        except KeyError:
            raise ConfigError(
                f"unknown guard {name!r}; known: {sorted(GUARDS)}"
            ) from None
        return cls(str(self._workspace), self._run_in_workspace)

    # -- caller-specific behaviour ----------------------------------------

    def session_key(self, req: InboundRequest) -> str:
        """Workspace path plus prompt hash.

        The workspace is part of the key because the same prompt against a
        different repo is a different task, and two foresight instances pointed
        at different checkouts must not share a cached aux result.
        """
        material = "\x00".join(
            [str(self._workspace), req.system_text(), req.first_user_content()]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    async def run_aux(self, req: InboundRequest, aux: ModelSpec) -> AuxResult:
        answer_path = self._answer_path()
        prompt = self._render_prompt(req, answer_path)

        # Stale answers are a real hazard with a fixed answer_file: last run's
        # text would be read as this run's output, silently.
        _unlink(answer_path)

        before = None
        if self._guard is not None:
            before = await asyncio.to_thread(self._guard.snapshot)

        started = time.monotonic()
        code, stdout, stderr, timed_out = await self._spawn(prompt, answer_path)
        duration = time.monotonic() - started

        # Checked before the answer is even read: if aux both contaminated the
        # workspace and produced text, the contamination is the thing that
        # invalidates the instance.
        if self._guard is not None:
            changed = await asyncio.to_thread(self._guard.verify, before)
            if changed:
                raise GuardViolation(changed, str(self._workspace))

        if timed_out:
            raise AuxFailure(
                f"aux agent exceeded {self._timeout_s}s "
                f"(command: {self._agent_cmd[0]}); {_tail(stderr)}"
            )

        text = self._read_answer(answer_path, stdout)
        if not text.strip():
            where = (
                "stdout" if self._answer_from == "stdout" else str(answer_path)
            )
            raise AuxFailure(
                f"aux agent produced no answer on {where} (exit {code}); "
                f"stdout: {_tail(stdout, 200)} stderr: {_tail(stderr, 200)}"
            )

        return AuxResult(
            text=text,
            source="agent",
            provenance={
                "adapter": self.name,
                "workspace": str(self._workspace),
                "command": self._agent_cmd[0],
                "returncode": code,
                "duration_s": round(duration, 3),
                "answer_from": self._answer_from,
                "guard": None if self._guard is None else "intact",
                "stdout_tail": _tail(stdout),
            },
        )

    # -- mechanics ---------------------------------------------------------

    def _answer_path(self) -> Path:
        """A configured path, or a fresh temp file per run.

        Per-run by default: a fixed path shared by concurrent sessions would let
        one run read another's answer.
        """
        if self._answer_file:
            return Path(self._answer_file).expanduser()
        directory = tempfile.mkdtemp(prefix="foresight-aux-")
        return Path(directory) / "answer.md"

    def _render_prompt(self, req: InboundRequest, answer_path: Path) -> str:
        return Template(self._aux_prompt, undefined=StrictUndefined).render(
            prompt=req.first_user_content(),
            system=req.system_text(),
            workspace=str(self._workspace),
            answer_file=str(answer_path),
        )

    def _argv(self, prompt: str, answer_path: Path) -> list[str]:
        """Substitute placeholders positionally.

        str.replace, never str.format: prompts and code are full of braces, and
        format() would either raise or silently eat them.
        """
        substitutions = {
            "{prompt}": prompt,
            "{answer_file}": str(answer_path),
            "{workspace}": str(self._workspace),
        }
        argv = []
        for element in self._agent_cmd:
            for placeholder, value in substitutions.items():
                element = element.replace(placeholder, value)
            argv.append(element)
        return argv

    def _child_env(self) -> dict[str, str]:
        """Inherit, then override.

        HOME belongs in adapter.env: opencode keeps its session DB under $HOME,
        so an aux run sharing it would pollute the user's own history and, under
        SWE-CI, the benchmark's token accounting.
        """
        env = dict(os.environ)
        env.update(self._env_overrides)
        return env

    async def _spawn(
        self, prompt: str, answer_path: Path
    ) -> tuple[int | None, str, str, bool]:
        argv = self._argv(prompt, answer_path)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self._workspace),
                env=self._child_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AuxFailure(
                f"aux agent command not found: {argv[0]!r} "
                "(adapter.agent_cmd names the harness to run)"
            ) from exc
        except OSError as exc:
            raise AuxFailure(f"could not start aux agent {argv[0]!r}: {exc}") from exc

        try:
            out, err = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout_s
            )
        except asyncio.TimeoutError:
            # Leaving an orphaned agent behind would keep touching the workspace
            # while the guard is verifying it.
            process.kill()
            await process.wait()
            return None, "", "", True

        return process.returncode, _decode(out), _decode(err), False

    def _read_answer(self, answer_path: Path, stdout: str) -> str:
        """Whichever stream the config declared. No fallback between them."""
        if self._answer_from == "stdout":
            return stdout
        try:
            return answer_path.read_text(errors="replace")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            raise AuxFailure(f"could not read aux answer at {answer_path}: {exc}") from exc

    def _run_in_workspace(self, command: str) -> tuple[int, str]:
        """The guard's reach into this workspace: a local shell.

        Synchronous by Protocol; callers wrap it in asyncio.to_thread so a large
        tree does not stall the event loop.
        """
        import subprocess

        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(self._workspace),
            capture_output=True,
            text=True,
        )
        return completed.returncode, completed.stdout


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _tail(text: str, limit: int = 400) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass
