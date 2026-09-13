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
    a parallel model. Fields past ``name`` are used only by adapters that launch
    a harness; GenericAdapter ignores all of them.
    """

    name: str = "generic"

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


class CallerAdapter(ABC):
    """One caller's idiosyncrasies, and how aux reaches its codebase."""

    name: ClassVar[str] = "adapter"

    aux_prompt_vars: ClassVar[set[str]] = {"prompt", "system"}
    """Variables this adapter renders ``aux_prompt`` with. config._check_template
    validates against exactly this set at startup, so a template referencing a
    variable its adapter does not supply fails before the port is bound rather
    than on the first request."""

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

    # -- concrete defaults ------------------------------------------------

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

    async def _body_only_aux(self, req: InboundRequest, aux: ModelSpec) -> AuxResult:
        """One chat call over the request body.

        The generic way to ask a model a question about a request, with no
        workspace access at all. GenericAdapter *is* this helper; it lives here
        because later adapters may want the same primitive.
        """
        from jinja2 import StrictUndefined, Template

        rendered = Template(self._aux_prompt, undefined=StrictUndefined).render(
            prompt=req.first_user_content(),
            system=req.system_text(),
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
