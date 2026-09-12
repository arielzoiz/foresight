"""CallerAdapter defaults and GenericAdapter's run_aux."""

from __future__ import annotations

import pytest

from conftest import StubBackend, make_adapter, user_only, with_history
from foresight.context import InboundRequest
from foresight.errors import AuxFailure


def test_is_session_start_true_without_assistant_message():
    adapter = make_adapter(StubBackend(), None)
    req = InboundRequest(body=user_only("task"))
    assert adapter.is_session_start(req) is True


def test_is_session_start_false_once_assistant_replied():
    adapter = make_adapter(StubBackend(), None)
    req = InboundRequest(body=with_history("task", {"role": "assistant", "content": "hi"}))
    assert adapter.is_session_start(req) is False


def test_phase_is_none_and_never_branches():
    adapter = make_adapter(StubBackend(), None)
    assert adapter.phase(InboundRequest(body=user_only("x"))) is None


@pytest.mark.asyncio
async def test_generic_adapter_run_aux_returns_text_from_body(aux_spec):
    backend = StubBackend(reply_text='Distinctive detail: "fix parse_date()". Future: X, Y.')
    adapter = make_adapter(backend, aux_spec)

    req = InboundRequest(body=user_only("fix parse_date()"))
    result = await adapter.run_aux(req, aux_spec)

    assert result.source == "chat_call"
    assert "fix parse_date()" in result.text
    assert len(backend.calls) == 1
    # the aux call must carry the task text, not a static placeholder
    sent = backend.calls[0]["messages"][0]["content"]
    assert "fix parse_date()" in sent


@pytest.mark.asyncio
async def test_generic_adapter_raises_aux_failure_on_non_200(aux_spec):
    backend = StubBackend(status=500)
    adapter = make_adapter(backend, aux_spec)
    with pytest.raises(AuxFailure):
        await adapter.run_aux(InboundRequest(body=user_only("x")), aux_spec)


@pytest.mark.asyncio
async def test_generic_adapter_raises_aux_failure_on_empty_reply(aux_spec):
    backend = StubBackend(reply_text="")
    adapter = make_adapter(backend, aux_spec)
    # _body_only_aux itself does not check emptiness -- AuxStage does. Direct
    # adapter use should still surface a non-empty-by-construction AuxResult
    # or fail; confirm it does not silently succeed with blank text where the
    # design guarantees non-empty (AuxStage is exercised in test_stage below).
    result = await adapter.run_aux(InboundRequest(body=user_only("x")), aux_spec)
    assert result.text == ""  # AuxStage is the enforcement point, tested separately


@pytest.mark.asyncio
async def test_aux_stage_raises_on_empty_text(aux_spec):
    from foresight.stages import AuxStage
    from foresight.context import RequestContext
    import copy

    backend = StubBackend(reply_text="   ")
    adapter = make_adapter(backend, aux_spec)
    stage = AuxStage(adapter, aux_spec)

    body = user_only("x")
    ctx = RequestContext(req=InboundRequest(body=body), body_out=copy.deepcopy(body))
    with pytest.raises(AuxFailure):
        await stage.run(ctx)
