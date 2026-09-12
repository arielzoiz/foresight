"""The request as foresight sees it, and the state accumulated while handling it.

Two objects, both deliberately plain:

``InboundRequest`` wraps the parsed request body *plus transport facts* (client
IP, path, headers). The transport facts are not decoration: SweCiAdapter
resolves which container a request came from by its client IP, because SWE-CI's
prompts are byte-identical across all its tasks and carry no task identity at
all.

``RequestContext`` is the mutable scratch space for one request -- stages write
into ``extras``, the builder rewrites ``body_out``.

Note what is *not* here: the body is an opaque ``dict``, never a Pydantic model.
See ``server.py`` for why that is a correctness constraint rather than a style
choice.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InboundRequest:
    """One POST /v1/chat/completions arriving at foresight.

    The only unit we observe. A caller's *session* is a sequence of these over a
    growing message list; a caller's *task* may span many sessions.
    """

    body: dict
    client_ip: str | None = None
    path: str = "/v1/chat/completions"
    headers: Mapping[str, str] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    received_at: float = field(default_factory=time.time)

    # -- accessors -------------------------------------------------------
    # Everything reads the body through these, so no code path is ever tempted
    # to build a typed request model (which would drop unmodelled fields).

    @property
    def messages(self) -> list[dict]:
        messages = self.body.get("messages")
        return messages if isinstance(messages, list) else []

    @property
    def model(self) -> str | None:
        model = self.body.get("model")
        return model if isinstance(model, str) else None

    @property
    def stream(self) -> bool:
        return bool(self.body.get("stream", False))

    @property
    def tools(self) -> list[dict]:
        tools = self.body.get("tools")
        return tools if isinstance(tools, list) else []

    def has_assistant_message(self) -> bool:
        """True once the model has replied at least once in this session."""
        return any(m.get("role") == "assistant" for m in self.messages)

    def first_user_index(self) -> int | None:
        """Index of the task-prompt message.

        The task prompt is the first user message. True for mini (its
        ``instance_template`` is the first user message) and for SWE-CI (a single
        prompt argument to ``opencode run``). Revisit if a caller violates it.
        """
        for i, message in enumerate(self.messages):
            if message.get("role") == "user":
                return i
        return None

    def first_user_content(self) -> str:
        index = self.first_user_index()
        if index is None:
            return ""
        return _as_text(self.messages[index].get("content"))

    def system_text(self) -> str:
        """All system messages concatenated, in order."""
        return "\n".join(
            _as_text(m.get("content")) for m in self.messages if m.get("role") == "system"
        )


def _as_text(content: Any) -> str:
    """Flatten a message ``content`` to text.

    OpenAI allows content to be either a string or a list of typed parts. We
    only ever *read* through this -- rewrites go back as plain strings, and the
    builder refuses to touch a message whose content is not a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


@dataclass
class RequestContext:
    """Everything accumulated while handling one request.

    A plain dataclass, not a Pydantic model: it holds live objects (the adapter)
    and a body that is mutated in flight.

    ``body_out`` is a deep copy made once, at construction. The inbound body is
    never mutated, which is what keeps the before/after comparison honest.
    """

    req: InboundRequest
    body_out: dict
    adapter_name: str = ""
    builder_name: str = ""
    session_key: str = ""
    is_session_start: bool = False
    phase: str | None = None
    enhanced: bool = False
    extras: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def aux(self):
        """The session's AuxResult, or None on the passthrough/bypass paths."""
        return self.extras.get("aux")

    def note(self, text: str) -> None:
        self.notes.append(text)
