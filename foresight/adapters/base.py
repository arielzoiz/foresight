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


class CallerAdapter(ABC):
    """One caller's idiosyncrasies, and how aux reaches its codebase."""

    name: ClassVar[str] = "adapter"

    def __init__(self, *, aux_backend: Backend, aux_spec: ModelSpec, aux_prompt: str) -> None:
        # Collaborators are injected at construction rather than passed per
        # call, so run_aux keeps the signature the design specifies.
        self._aux_backend = aux_backend
        self._aux_spec = aux_spec
        self._aux_prompt = aux_prompt

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
