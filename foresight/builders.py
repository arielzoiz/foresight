"""Builders -- the only component permitted to rewrite ``messages``.

Each experimental arm is a builder:

    passthrough   control   -- forwards unchanged, byte for byte
    template      treatment -- injects aux's future-task context

Switching arms is one config value, and both arms traverse identical code, so
the only difference between them is the prompt. That is what makes the
comparison a paired one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from jinja2 import StrictUndefined, Template

from .context import RequestContext

#: Improbable in any real prompt; used only to locate where the template puts
#: the task text, never sent anywhere.
_SENTINEL = "\x00\x00FORESIGHT_PROMPT_SLOT\x00\x00"


class Builder(ABC):
    name: ClassVar[str] = "builder"

    @abstractmethod
    def build(self, ctx: RequestContext) -> None:
        """Rewrite ``ctx.body_out`` in place. The inbound body is never touched."""


class PassthroughBuilder(Builder):
    """The control arm. Forwards the conversation unchanged.

    Deliberately a real class rather than a config flag: the control arm must
    traverse the same pipeline, adapter and backend as the treatment arm, so
    that any difference in results is attributable to the prompt and not to a
    different code path.
    """

    name: ClassVar[str] = "passthrough"

    def build(self, ctx: RequestContext) -> None:
        return None


class TemplateBuilder(Builder):
    """The treatment arm. Wraps the task prompt in aux's future-task context.

    Re-application, not caching
    ---------------------------
    This runs on EVERY request of a session, not only the first. The harness
    keeps its own message list and never learns we rewrote anything, so it
    resends the original task prompt every time. Enhancing only the first
    request would put the enhanced text in front of the model exactly once,
    leaving the assistant's own earlier turns replying to text no longer in
    context. For SWE-CI's architect -- which reads files for dozens of requests
    and writes its output only at the end -- the instruction would be long gone
    by the time it mattered.

    Because the rewrite is idempotent, the conversation the model sees is
    identical to what it would have seen had the harness been handed the
    enhanced prompt in the first place. Re-applying is not "enhancing every
    request"; it is what stops a single enhancement from silently reverting.

    Idempotency
    -----------
    Detected without a marker and without assuming anything about the template's
    structure: render once with a sentinel in the prompt slot, then split the
    *rendered* output on it. That yields the exact prefix and suffix this
    session wraps the prompt in, whatever conditionals or loops the template
    contains. Deriving them from the template *source* would break the moment a
    ``{% if %}`` appeared.

    This is also why the aux result is cached per session: stable aux means a
    stable prefix and suffix for the session's lifetime.
    """

    name: ClassVar[str] = "template"

    def __init__(self, template: str) -> None:
        self._source = template
        # StrictUndefined: a mistyped variable in a config fails here, at
        # startup, rather than silently rendering empty on every request.
        self._template = Template(template, undefined=StrictUndefined)
        self._template.render(prompt=_SENTINEL, aux="")

    def build(self, ctx: RequestContext) -> None:
        index = ctx.req.first_user_index()
        if index is None:
            ctx.note("no user message; nothing to enhance")
            return

        message = ctx.body_out["messages"][index]
        content = message.get("content")
        if not isinstance(content, str):
            # Structured content parts: rewriting would mean guessing which part
            # is the task. Leave it alone and say so rather than corrupt it.
            ctx.note("first user message has non-string content; not enhanced")
            return

        aux = ctx.aux
        aux_text = aux.text if aux is not None else ""

        prefix, suffix = self._wrapping(aux_text)
        if content.startswith(prefix) and content.endswith(suffix) and prefix:
            # Already carries this session's enhancement -- the caller echoed
            # back our own text. Re-applying would double the prefix.
            ctx.enhanced = True
            ctx.note("already enhanced; left as is")
            return

        message["content"] = self._template.render(prompt=content, aux=aux_text)
        ctx.enhanced = True

    def _wrapping(self, aux_text: str) -> tuple[str, str]:
        rendered = self._template.render(prompt=_SENTINEL, aux=aux_text)
        prefix, _, suffix = rendered.partition(_SENTINEL)
        return prefix, suffix


BUILDERS: dict[str, type[Builder]] = {
    PassthroughBuilder.name: PassthroughBuilder,
    TemplateBuilder.name: TemplateBuilder,
}
