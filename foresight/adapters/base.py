"""CallerAdapter -- everything caller-specific, one file per caller.

The rule this abstraction exists to enforce: there is no ``if caller == ...``
anywhere else in the system. Adding a fourth caller means writing one file, not
editing the pipeline.

The ABC carries three concrete defaults that most callers never override, and
one abstract method. What differs between callers is not how the *target* gets
context -- it always goes through a harness -- but how *aux* reaches the same
code. That is what ``run_aux`` exists to absorb:

    GenericAdapter  one chat call over the request body       (M1, mini opt. A)
    LocalAdapter    host subprocess, cwd = the repo           (M2)
    SweCiAdapter    docker exec a second harness into the
                    live task container                       (M3)
    MiniAdapter     docker run a throwaway container from
                    the per-instance image                    (M4, option B)
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from ..backends import Backend
from ..context import InboundRequest
from ..errors import AuxFailure
from ..llm import ModelSpec


class AuxResult(BaseModel):
    """What the aux step produced, plus how it was obtained."""

    text: str
    """Non-empty by construction -- AuxStage raises AuxFailure otherwise, so
    downstream code never handles a missing-aux case."""

    source: Literal["chat_call", "agent"]
    """How aux reached its answer. "agent" arrives with milestone 2."""

    provenance: dict = Field(default_factory=dict)
    """Model, usage, duration. Kept populated even with no trace writer: tests
    assert on it, and it is what a trace would serialise if one is added."""


class AdapterOptions(BaseModel):
    """The ``adapter:`` config block.

    Lives here rather than in config.py so that adapters own the shape of their
    own configuration, and config.py can use it directly instead of maintaining
    a parallel model. Most fields past ``name`` are used only by adapters that
    launch a harness (GenericAdapter ignores them); ``require_tools`` and
    ``max_concurrent_aux`` apply to every adapter.
    """

    name: str = "generic"

    require_tools: bool | None = None
    """Treat a session-start request with no ``tools`` array as not a turn of
    the agent session, so no aux run and no prompt rewrite touch it.

    ``None`` means "ask the adapter" (``CallerAdapter.require_tools_default``).
    Measured: opencode opens every session with a title-generation request
    ("Generate a title for this conversation:", zero tools) roughly two seconds
    before the real agent turn (which carries the harness's full tool list).
    Treating that request as a task prompt spends a whole aux agent run naming
    a chat, and -- because it hashes to a different session key than the real
    turn and so takes a different per-key lock -- races it into the harness's
    own on-disk state.

    Set ``false`` for a caller that legitimately never advertises tools (e.g.
    mini-swe-agent's text-based model classes, which parse actions out of plain
    text). Foresight cannot tell "never advertises tools" apart from "every
    request is being skipped" until it has seen a few -- see the startup
    warning that exists so a run enhancing nothing is never silent.
    """

    max_concurrent_aux: int = Field(default=0, ge=0)
    """Cap on aux runs in flight across ALL sessions. 0 = unlimited.

    Aux runs that share on-disk state can corrupt each other if run
    concurrently -- measured with LocalAdapter/opencode, two `opencode run`
    processes against one SQLite session DB produced `database is locked`.
    Set to 1 for a caller whose aux subprocess shares state across sessions.
    Not needed where each aux run is isolated (e.g. SWE-CI's aux runs in its
    own container, with its own filesystem, per session)."""

    workspace: str | None = None
    """Directory the aux agent explores. Required by LocalAdapter."""

    agent_cmd: list[str] = Field(default_factory=list)
    """The harness invocation, as argv. We define no tools and invent no
    protocol -- aux reaches the codebase through a real harness, so this is a
    command, not a plugin. Placeholders ``{prompt}``, ``{answer_file}`` and
    ``{workspace}`` are substituted per element."""

    answer_file: str | None = None
    """Where the agent is told to write its answer. Reading a file is more
    reliable than scraping stdout, which carries the harness's own chatter --
    the same pattern SWE-CI uses for requirement.xml. Omit for a fresh temp file
    per run, which is what concurrent sessions need."""

    answer_from: Literal["file", "stdout"] = "file"
    """Which stream carries aux's answer.

    Declared, never guessed. A harness told to write a file and then failing to
    prints only its own progress chatter, and silently accepting that as the
    future-task list would inject text carrying no future tasks -- the same
    "instance in neither arm" contamination the aux gate exists to catch. Set
    "stdout" for a harness that answers on stdout (opencode does)."""

    timeout_s: float = 900.0
    """Wall clock for the whole agent run."""

    guard_ignore: list[str] = Field(default_factory=list)
    """Paths the workspace guard should not treat as contamination, as fnmatch
    globs relative to ``workspace``.

    For harness bookkeeping only. opencode snapshots the tree for its own undo
    feature and writes the object id to ``.git/opencode`` on every run against a
    git repo, so without ``[".git/opencode"]`` here aux fails on any git
    workspace, every time, for a reason unrelated to the experiment. Keep it
    narrow: ignoring all of ``.git`` would also hide aux rewriting refs, which
    is real damage to a local user's repo."""

    env: dict[str, str] = Field(default_factory=dict)
    """Added to the subprocess environment. Set HOME here: opencode keeps its
    session DB under $HOME, and sharing it would corrupt both the user's own
    state and (in SWE-CI) the benchmark's token accounting."""

    docker_cmd: str = "docker"
    """The container client SweCiAdapter drives. Named rather than hardcoded so
    a deployment can point it at an absolute path instead of relying on PATH
    order."""

    swe_ci_config: str | None = None
    """Path to SWE-CI's ``config.toml``. Read once at startup so that an
    ``agent_cmd`` naming a harness the task image does not ship fails there
    rather than on the first aux run -- binding constraint 2 says aux must use
    the harness already in the image, and ``Dockerfile.opencode`` installs
    exactly the one ``agent_name`` selected."""

    foresight_base_url: str | None = None
    """This server's own address, as reachable from inside a SWE-CI container
    -- typically the Docker bridge gateway (``http://172.17.0.1:8000/v1``),
    never ``localhost``. Required by ``SweCiAdapter`` to bootstrap aux's own
    opencode provider config (``auth.json``/``opencode.json``, mirroring
    SWE-CI's own ``setup_opencode``): aux's harness runs *inside* the
    container and needs to know where to send its own model calls, which is
    this address, not the aux model's real upstream -- aux is pointed at this
    server precisely so its traffic bypasses the pipeline and lands in the
    same trace as everything else."""

    session_header_names: list[str] = Field(default_factory=lambda: ["x-session-id"])
    """Request headers, checked in order, that identify a caller's session on
    their own -- no ``docker inspect`` needed. Measured
    (``tools/probe_opencode_headers.sh``): opencode sends ``x-session-id`` on
    every request, stable within one ``opencode run`` invocation and distinct
    across invocations, with zero configuration. When present this replaces a
    blocking subprocess pair on every request, not only session starts;
    container-ID resolution remains the fallback when no listed header is
    present, since a session key alone does not say which container to
    ``docker exec`` into."""


class CallerAdapter(ABC):
    """One caller's idiosyncrasies, and how aux reaches its codebase."""

    name: ClassVar[str] = "adapter"

    aux_prompt_vars: ClassVar[set[str]] = {"prompt", "system"}
    """Variables this adapter renders ``aux_prompt`` with. config._check_template
    validates against exactly this set at startup, so a template referencing a
    variable its adapter does not supply fails before the port is bound rather
    than on the first request."""

    require_tools_default: ClassVar[bool] = True
    """This adapter's default for AdapterOptions.require_tools when unset.

    True for every adapter shipped today: every supported caller's real agent
    turn advertises tools. A caller that never does (e.g. a text-based,
    non-tool-calling harness) overrides this to False in its adapter class."""

    def __init__(
        self,
        *,
        aux_backend: Backend,
        aux_spec: ModelSpec,
        aux_prompt: str,
        options: AdapterOptions | None = None,
        guard_name: str | None = None,
    ) -> None:
        # Collaborators are injected at construction rather than passed per
        # call, so run_aux keeps the signature the design specifies.
        self._aux_backend = aux_backend
        self._aux_spec = aux_spec
        self._aux_prompt = aux_prompt
        self._options = options or AdapterOptions(name=self.name)
        self._guard_name = guard_name
        # Resolved once, not per request: the tri-state config value means
        # "ask the adapter" only until construction settles it.
        self._require_tools = (
            self.require_tools_default
            if self._options.require_tools is None
            else self._options.require_tools
        )

    # -- concrete defaults ------------------------------------------------

    def is_agent_turn(self, req: InboundRequest) -> bool:
        """Is this request a turn of the agent session we are here to enhance?

        Agent harnesses do bookkeeping against the same model on the same
        endpoint: opencode opens every session with a title-generation request
        ("Generate a title for this conversation:") shortly before the real
        turn. SWE-CI's target agent is invoked the same way (``opencode run
        --model ... "<prompt>"``), so this is not a local-only quirk. Treating
        the title request as a task prompt spends a whole aux agent run naming
        a chat and -- because it hashes to a different session key than the
        real turn and so takes a different per-key lock -- runs concurrently
        with it, which is what deadlocked opencode's shared session database
        in the M2 runs (see results/qwen2.5-coder-7b/findings.md).

        The discriminator is caller-agnostic: the OpenAI protocol requires
        tools to be resent on every request, so a real agent turn always
        advertises them and a bookkeeping call does not. Checked on every
        request, not only at session start, so a title regenerated mid-session
        (which would carry assistant messages) is skipped too, rather than
        being handed the session's real aux text.
        """
        if not self._require_tools:
            return True
        return req.advertises_tools()

    def is_session_start(self, req: InboundRequest) -> bool:
        """True on the first request of an agent session.

        No assistant message means the model has not replied yet, which means
        the harness is opening a conversation. Every later request of that
        session carries at least one.
        """
        return not req.has_assistant_message()

    def session_key(self, req: InboundRequest) -> str:
        """Scope for the cached aux result.

        Hashing system + first user message works wherever the task prompt is
        unique per task. It is *not* universal: SWE-CI's prompts are
        byte-identical across all 100 of its tasks, which is exactly why
        SweCiAdapter overrides this with the container ID.
        """
        material = f"{req.system_text()}\x00{req.first_user_content()}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def phase(self, req: InboundRequest) -> str | None:
        """Trace metadata only. Never branches control flow.

        SWE-CI's architect and programmer are treated identically -- same
        pipeline, same aux prompt, twice per epoch. Recording the phase lets the
        results be split afterwards at zero extra cost.
        """
        return None

    # -- the one thing every caller must answer ---------------------------

    @abstractmethod
    async def run_aux(self, req: InboundRequest, aux: ModelSpec) -> AuxResult:
        """Ask aux about plausible future work for this request's task."""

    # -- shared helper ----------------------------------------------------

    async def _body_only_aux(
        self, req: InboundRequest, aux: ModelSpec, **extra_vars: str
    ) -> AuxResult:
        """One chat call over the request body.

        The generic way to ask a model a question about a request, with no
        workspace access at all. GenericAdapter *is* this helper; it lives here
        because later adapters may want the same primitive -- SweCiAdapter
        degrades to it when it cannot resolve the caller's container.

        ``extra_vars`` exists for that degraded path: the aux prompt was
        validated at startup against ``aux_prompt_vars``, so an adapter that
        declares more of them must still supply them all here, or
        StrictUndefined turns the fallback itself into a crash. The values are
        necessarily hollow (there is no workspace to name), which is one more
        reason such an instance is excluded at analysis time.
        """
        from jinja2 import StrictUndefined, Template

        rendered = Template(self._aux_prompt, undefined=StrictUndefined).render(
            prompt=req.first_user_content(),
            system=req.system_text(),
            **extra_vars,
        )
        started = time.monotonic()
        response = await self._aux_backend.complete(
            aux,
            {
                "model": aux.model,
                "messages": [{"role": "user", "content": rendered}],
                "stream": False,
            },
        )
        duration = time.monotonic() - started

        if response.status != 200:
            raise AuxFailure(
                f"aux failed: upstream returned {response.status} ({_brief(response.json)})"
            )

        try:
            text = response.json["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise AuxFailure(f"aux failed: unreadable response ({exc})") from exc

        return AuxResult(
            text=text,
            source="chat_call",
            provenance={
                "model": aux.model,
                "served_name": aux.served_name,
                "duration_s": round(duration, 3),
                "usage": (response.json or {}).get("usage"),
            },
        )


def _brief(payload: object, limit: int = 200) -> str:
    return str(payload)[:limit]
