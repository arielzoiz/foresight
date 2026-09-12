"""Builder rewriting: both arms, idempotency, and edge cases."""

from __future__ import annotations

import copy

import pytest

from foresight.adapters.base import AuxResult
from foresight.builders import BUILDERS, PassthroughBuilder, TemplateBuilder
from foresight.context import InboundRequest, RequestContext


def _ctx(body: dict, aux_text: str | None = None) -> RequestContext:
    req = InboundRequest(body=body)
    ctx = RequestContext(req=req, body_out=copy.deepcopy(body))
    if aux_text is not None:
        ctx.extras["aux"] = AuxResult(text=aux_text, source="chat_call")
    return ctx


def test_registry_has_both_arms():
    assert set(BUILDERS) == {"passthrough", "template"}


def test_passthrough_leaves_messages_byte_identical():
    body = {
        "model": "target-model",
        "messages": [{"role": "user", "content": "fix parse_date()"}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
    }
    ctx = _ctx(body)
    original = copy.deepcopy(ctx.body_out)

    PassthroughBuilder().build(ctx)

    assert ctx.body_out == original
    assert ctx.enhanced is False


def test_template_rewrites_first_user_message_with_aux_ahead_of_prompt():
    body = {"model": "target-model", "messages": [{"role": "user", "content": "fix parse_date()"}]}
    ctx = _ctx(body, aux_text="watch for timezones")
    builder = TemplateBuilder("FUTURE:\n{{ aux }}\n\nTASK:\n{{ prompt }}")

    builder.build(ctx)

    rewritten = ctx.body_out["messages"][0]["content"]
    assert rewritten == "FUTURE:\nwatch for timezones\n\nTASK:\nfix parse_date()"
    assert rewritten.index("watch for timezones") < rewritten.index("fix parse_date()")
    assert ctx.enhanced is True
    # inbound body must never be mutated
    assert ctx.req.body["messages"][0]["content"] == "fix parse_date()"


def test_template_is_idempotent_across_a_session():
    """Re-applying to already-enhanced content must not double the wrapping."""
    template = "FUTURE:\n{{ aux }}\n\nTASK:\n{{ prompt }}"
    builder = TemplateBuilder(template)

    body = {"model": "target-model", "messages": [{"role": "user", "content": "fix parse_date()"}]}
    ctx1 = _ctx(body, aux_text="watch for timezones")
    builder.build(ctx1)
    enhanced_once = ctx1.body_out["messages"][0]["content"]

    # Simulate request 2: the harness resends its own original copy, but if it
    # somehow echoed back our enhanced text (or if we ever re-ran on our own
    # output), a second pass must not double-wrap it.
    body2 = {"model": "target-model", "messages": [{"role": "user", "content": enhanced_once}]}
    ctx2 = _ctx(body2, aux_text="watch for timezones")
    builder.build(ctx2)

    assert ctx2.body_out["messages"][0]["content"] == enhanced_once
    assert ctx2.body_out["messages"][0]["content"].count("FUTURE:") == 1


def test_template_startup_validation_rejects_unknown_variable():
    with pytest.raises(Exception):
        TemplateBuilder("{{ nonexistent_variable }}")


def test_template_skips_non_string_content_without_raising():
    body = {
        "model": "target-model",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    ctx = _ctx(body, aux_text="future stuff")
    TemplateBuilder("{{ aux }}\n{{ prompt }}").build(ctx)

    assert ctx.body_out["messages"][0]["content"] == [{"type": "text", "text": "hi"}]
    assert ctx.enhanced is False
    assert any("non-string" in n for n in ctx.notes)


def test_template_no_user_message_notes_and_does_not_raise():
    body = {"model": "target-model", "messages": [{"role": "system", "content": "sys"}]}
    ctx = _ctx(body, aux_text="future stuff")
    TemplateBuilder("{{ aux }}\n{{ prompt }}").build(ctx)

    assert ctx.enhanced is False
    assert any("nothing to enhance" in n for n in ctx.notes)
