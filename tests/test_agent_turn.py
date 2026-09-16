"""The non-agent-turn gate: which requests foresight is allowed to touch.

Pinned contract: a request that advertises no ``tools`` array is not a turn of
an agent session. foresight forwards it exactly as the control arm would --
no session key resolved, no lock taken, no aux run, no builder. Measured
against opencode, which opens every session with a title-generation request
("Generate a title for this conversation:", zero tools) shortly before the
real agent turn (which carries the harness's full tool list, ten for
opencode) -- see results/qwen2.5-coder-7b/findings.md. Treating the title
request as a task prompt spends a whole aux agent run naming a chat and, once
keyed on something that collapses the two requests to the same session
(container ID, under M3), risks handing the real session's future-task list
to it instead.

Configurable via ``adapter.require_tools`` (tri-state: null asks the adapter's
``require_tools_default``), and loud when a whole run enhances nothing, since
silently skipping every request is functionally the control arm.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import AUX_PROMPT, TEMPLATE, StubBackend, user_only, with_history
from foresight.adapters.base import AdapterOptions
from foresight.adapters.generic import GenericAdapter
from foresight.builders import BUILDERS
from foresight.context import InboundRequest
from foresight.pipeline import SILENT_RUN_WARNING_AFTER, Pipeline, SessionStore
from foresight.stages import AuxStage
from foresight.trace import target_record


def _adapter(aux_backend, aux_spec, *, require_tools=None, cls=GenericAdapter):
    return cls(
        aux_backend=aux_backend,
        aux_spec=aux_spec,
        aux_prompt=AUX_PROMPT,
        options=AdapterOptions(name=cls.name, require_tools=require_tools),
    )


def _pipeline(adapter, aux_spec, *, store=None, **kwargs):
    builder = BUILDERS["template"](TEMPLATE)
    store = store if store is not None else SessionStore()
    stage = AuxStage(adapter, aux_spec)
    return Pipeline(adapter=adapter, stages=[stage], builder=builder, store=store, **kwargs)


# -- is_agent_turn itself ----------------------------------------------------


def test_a_session_start_without_tools_is_not_an_agent_turn(aux_spec):
    adapter = _adapter(StubBackend(), aux_spec)
    req = InboundRequest(body=user_only("Generate a title for this conversation:", tools=[]))
    assert adapter.is_agent_turn(req) is False


def test_a_request_advertising_tools_is_an_agent_turn(aux_spec):
    adapter = _adapter(StubBackend(), aux_spec)
    req = InboundRequest(body=user_only("fix parse_date()"))
    assert adapter.is_agent_turn(req) is True


# -- the measured pair --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_title_request_is_skipped_while_the_real_turn_is_enhanced(aux_spec):
    aux_backend = StubBackend(reply_text="future: X, Y, Z")
    adapter = _adapter(aux_backend, aux_spec)
    pipeline = _pipeline(adapter, aux_spec)

    title = InboundRequest(body=user_only("Generate a title for this conversation:", tools=[]))
    real_turn = InboundRequest(body=user_only("fix parse_date()"))

    title_ctx, turn_ctx = await asyncio.gather(
        pipeline.handle(title),
        pipeline.handle(real_turn),
    )

    assert len(aux_backend.calls) == 1
    assert title_ctx.is_agent_turn is False
    assert title_ctx.body_out == title.body
    assert turn_ctx.is_agent_turn is True
    assert turn_ctx.enhanced is True
    assert "future: X, Y, Z" in turn_ctx.body_out["messages"][0]["content"]


@pytest.mark.asyncio
async def test_a_skipped_request_is_forwarded_unchanged(aux_spec):
    aux_backend = StubBackend()
    adapter = _adapter(aux_backend, aux_spec)
    pipeline = _pipeline(adapter, aux_spec)

    req = InboundRequest(body=user_only("Generate a title for this conversation:", tools=[]))
    ctx = await pipeline.handle(req)

    assert ctx.body_out == req.body
    assert ctx.enhanced is False
    assert len(aux_backend.calls) == 0


@pytest.mark.asyncio
async def test_a_skipped_request_never_writes_the_session_store(aux_spec):
    aux_backend = StubBackend(reply_text="future: X")
    adapter = _adapter(aux_backend, aux_spec)
    store = SessionStore()
    pipeline = _pipeline(adapter, aux_spec, store=store)

    title = InboundRequest(body=user_only("Generate a title for this conversation:", tools=[]))
    real_turn = InboundRequest(body=user_only("fix parse_date()"))

    await asyncio.gather(pipeline.handle(title), pipeline.handle(real_turn))

    assert len(store) == 1


@pytest.mark.asyncio
async def test_a_skipped_request_never_resolves_a_session_key(aux_spec):
    """Protects the M3 PR: SweCiAdapter's session_key resolves the container ID
    via a subprocess (docker inspect). A skipped request must never reach it."""

    class ExplodingKeyAdapter(GenericAdapter):
        name = "exploding-key"

        def session_key(self, req):
            raise AssertionError("session_key must not be called for a skipped request")

    aux_backend = StubBackend()
    adapter = _adapter(aux_backend, aux_spec, cls=ExplodingKeyAdapter)
    pipeline = _pipeline(adapter, aux_spec)

    req = InboundRequest(body=user_only("Generate a title for this conversation:", tools=[]))
    ctx = await pipeline.handle(req)  # must not raise

    assert ctx.is_agent_turn is False
    assert ctx.session_key == ""


@pytest.mark.asyncio
async def test_a_continuation_without_tools_is_also_skipped(aux_spec):
    """A title regenerated mid-session still carries no tools. Under
    container-ID keying it would share a key with the real session -- this
    proves it never reaches the store lookup that would hand it that
    session's real aux text."""
    aux_backend = StubBackend(reply_text="future: real session content")
    adapter = _adapter(aux_backend, aux_spec)
    pipeline = _pipeline(adapter, aux_spec)

    await pipeline.handle(InboundRequest(body=user_only("fix parse_date()")))

    continuation = InboundRequest(
        body=with_history(
            "fix parse_date()",
            {"role": "assistant", "content": "looking"},
            tools=[],
        )
    )
    ctx = await pipeline.handle(continuation)

    assert ctx.is_agent_turn is False
    assert ctx.enhanced is False
    assert "future: real session content" not in str(ctx.body_out)


# -- config surface ------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_tools_false_restores_unconditional_enhancement(aux_spec):
    aux_backend = StubBackend(reply_text="future: X")
    adapter = _adapter(aux_backend, aux_spec, require_tools=False)
    pipeline = _pipeline(adapter, aux_spec)

    req = InboundRequest(body=user_only("Generate a title for this conversation:", tools=[]))
    ctx = await pipeline.handle(req)

    assert ctx.is_agent_turn is True
    assert ctx.enhanced is True


def test_an_adapter_may_declare_that_its_caller_sends_no_tools(aux_spec):
    class TextbasedAdapter(GenericAdapter):
        name = "textbased"
        require_tools_default = False

    adapter = _adapter(StubBackend(), aux_spec, cls=TextbasedAdapter)
    req = InboundRequest(body=user_only("no tools here", tools=[]))
    assert adapter.is_agent_turn(req) is True


# -- trace visibility ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_trace_row_distinguishes_a_skipped_request_from_a_joined_session(aux_spec):
    aux_backend = StubBackend()
    adapter = _adapter(aux_backend, aux_spec)
    pipeline = _pipeline(adapter, aux_spec)

    skipped_ctx = await pipeline.handle(
        InboundRequest(body=user_only("Generate a title for this conversation:", tools=[]))
    )
    joined_ctx = await pipeline.handle(
        InboundRequest(
            body=with_history("never seen before", {"role": "assistant", "content": "x"})
        )
    )

    skipped_row = target_record(
        skipped_ctx, served_model="target-model", upstream_status=None, usage_upstream=None
    )
    joined_row = target_record(
        joined_ctx, served_model="target-model", upstream_status=None, usage_upstream=None
    )

    assert skipped_row["is_agent_turn"] is False
    assert skipped_row["session_key"] == ""
    assert any("not_agent_turn" in n for n in skipped_row["notes"])

    assert joined_row["is_agent_turn"] is True
    assert joined_row["session_key"] != ""
    assert "no_session_entry" in joined_row["notes"]


# -- observability: the never-enhanced warning and counters -----------------


@pytest.mark.asyncio
async def test_a_run_that_enhances_nothing_warns_once(aux_spec, capsys):
    aux_backend = StubBackend()
    adapter = _adapter(aux_backend, aux_spec)
    pipeline = _pipeline(adapter, aux_spec)

    for i in range(SILENT_RUN_WARNING_AFTER + 5):
        req = InboundRequest(
            body=user_only(f"Generate a title for this conversation: {i}", tools=[])
        )
        await pipeline.handle(req)

    err = capsys.readouterr().err
    assert err.count("requests skipped as non-agent turns") == 1
    assert "adapter.require_tools: false" in err


@pytest.mark.asyncio
async def test_a_healthy_run_never_warns(aux_spec, capsys):
    """One skipped title request followed by real, enhanced turns must never
    trip the warning -- that is every correct opencode session."""
    aux_backend = StubBackend(reply_text="future: X")
    adapter = _adapter(aux_backend, aux_spec)
    pipeline = _pipeline(adapter, aux_spec)

    await pipeline.handle(
        InboundRequest(body=user_only("Generate a title for this conversation:", tools=[]))
    )
    for i in range(SILENT_RUN_WARNING_AFTER + 5):
        await pipeline.handle(InboundRequest(body=user_only(f"task {i}")))

    err = capsys.readouterr().err
    assert "requests skipped as non-agent turns" not in err


@pytest.mark.asyncio
async def test_health_reports_the_enhanced_and_skipped_counters():
    import httpx

    from test_server import build_test_runtime

    from foresight.server import create_app

    runtime, _, _ = build_test_runtime()
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json=user_only("Generate a title for this conversation:", tools=[]),
        )
        await client.post("/v1/chat/completions", json=user_only("fix bug"))

        health = await client.get("/health")

    body = health.json()
    assert body["enhanced"] == 1
    assert body["skipped_not_agent"] == 1
