"""Shared fixtures.

StubBackend replaces network I/O with an in-memory, call-counting fake, so
tests can assert "aux fired exactly once" without a real server. FakeUpstream
(used only by test_e2e.py) is the real thing, run in-process over ASGI.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from foresight.adapters import ADAPTERS
from foresight.backends import Backend, UpstreamResponse
from foresight.builders import BUILDERS
from foresight.context import InboundRequest
from foresight.llm import ModelSpec
from foresight.pipeline import Pipeline, SessionStore
from foresight.stages import AuxStage


class StubBackend(Backend):
    """An in-memory Backend. Records every call; replies are canned or computed."""

    name = "stub"

    def __init__(self, reply_text: str | list[str] = "stub reply", status: int = 200) -> None:
        self.calls: list[dict] = []
        self._reply_text = reply_text
        self._status = status
        self._call_index = 0

    def _next_reply(self) -> str:
        if isinstance(self._reply_text, list):
            text = self._reply_text[min(self._call_index, len(self._reply_text) - 1)]
        else:
            text = self._reply_text
        self._call_index += 1
        return text

    async def complete(self, spec: ModelSpec, body: dict) -> UpstreamResponse:
        # A real backend always awaits network I/O; yielding here makes the
        # stub faithful enough that asyncio.gather over two calls actually
        # interleaves them, instead of running the first to completion before
        # the second is ever scheduled -- which is what genuine lock
        # contention (concurrent session starts) needs to test anything.
        await asyncio.sleep(0)
        self.calls.append(body)
        text = self._next_reply()
        return UpstreamResponse(
            status=self._status,
            json={
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            headers={},
        )

    async def stream(self, spec: ModelSpec, body: dict) -> AsyncIterator[bytes]:
        self.calls.append(body)
        yield b'data: {"choices":[{"delta":{"content":"stub"}}]}\n\n'
        yield b"data: [DONE]\n\n"


@pytest.fixture
def aux_spec() -> ModelSpec:
    return ModelSpec(served_name="aux-model", model="fake-aux", base_url="http://unused/v1")


@pytest.fixture
def target_spec() -> ModelSpec:
    return ModelSpec(served_name="target-model", model="fake-target", base_url="http://unused/v1")


AUX_PROMPT = "quote a detail, then list future tasks.\n\n{{ prompt }}"

TEMPLATE = (
    "FUTURE TASKS (not your assignment):\n{{ aux }}\n\n"
    "Solve only the main task below.\n\n{{ prompt }}"
)


def make_adapter(aux_backend: StubBackend, aux_spec: ModelSpec, *, name: str = "generic"):
    cls = ADAPTERS[name]
    return cls(aux_backend=aux_backend, aux_spec=aux_spec, aux_prompt=AUX_PROMPT)


def make_pipeline(
    aux_backend: StubBackend,
    aux_spec: ModelSpec,
    *,
    builder_name: str = "template",
    template: str = TEMPLATE,
    ttl_s: float = 3600.0,
) -> Pipeline:
    adapter = make_adapter(aux_backend, aux_spec)
    builder_cls = BUILDERS[builder_name]
    builder = builder_cls(template) if builder_name == "template" else builder_cls()
    store = SessionStore(ttl_s=ttl_s)
    stage = AuxStage(adapter, aux_spec)
    return Pipeline(adapter=adapter, stages=[stage], builder=builder, store=store)


def user_only(text: str) -> dict:
    return {"model": "target-model", "messages": [{"role": "user", "content": text}]}


def with_history(text: str, *turns: dict) -> dict:
    return {
        "model": "target-model",
        "messages": [{"role": "user", "content": text}, *turns],
    }
