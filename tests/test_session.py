"""Pipeline + SessionStore: the two-decisions-per-request contract.

This is the design's central correctness requirement: aux runs once per
session, but the builder re-applies on every request using the cached result.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import StubBackend, make_pipeline, user_only, with_history


@pytest.mark.asyncio
async def test_aux_fires_exactly_once_per_session(aux_spec):
    aux_backend = StubBackend(reply_text="future: X, Y, Z")
    pipeline = make_pipeline(aux_backend, aux_spec)

    req1 = user_only("fix parse_date()")
    req2 = {
        **req1,
        "messages": [
            *req1["messages"],
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "output"},
        ],
    }
    req3 = {**req2, "messages": [*req2["messages"], {"role": "assistant", "content": "more"}]}

    from foresight.context import InboundRequest

    await pipeline.handle(InboundRequest(body=req1))
    await pipeline.handle(InboundRequest(body=req2))
    await pipeline.handle(InboundRequest(body=req3))

    assert len(aux_backend.calls) == 1


@pytest.mark.asyncio
async def test_enhanced_prompt_present_in_every_request_of_the_session(aux_spec):
    aux_backend = StubBackend(reply_text="future: timezones")
    pipeline = make_pipeline(aux_backend, aux_spec)

    from foresight.context import InboundRequest

    body1 = user_only("fix parse_date()")
    ctx1 = await pipeline.handle(InboundRequest(body=body1))
    assert "future: timezones" in ctx1.body_out["messages"][0]["content"]

    body2 = with_history("fix parse_date()", {"role": "assistant", "content": "looking"})
    ctx2 = await pipeline.handle(InboundRequest(body=body2))
    assert "future: timezones" in ctx2.body_out["messages"][0]["content"]

    body3 = with_history(
        "fix parse_date()",
        {"role": "assistant", "content": "looking"},
        {"role": "user", "content": "still going"},
    )
    ctx3 = await pipeline.handle(InboundRequest(body=body3))
    assert "future: timezones" in ctx3.body_out["messages"][0]["content"]


@pytest.mark.asyncio
async def test_concurrent_session_starts_run_aux_once(aux_spec):
    """Two concurrent requests opening the same session must not double-call aux."""
    aux_backend = StubBackend(reply_text="future: A")
    pipeline = make_pipeline(aux_backend, aux_spec)

    from foresight.context import InboundRequest

    body = user_only("fix parse_date()")
    await asyncio.gather(
        pipeline.handle(InboundRequest(body=body)),
        pipeline.handle(InboundRequest(body=body)),
    )

    assert len(aux_backend.calls) == 1


@pytest.mark.asyncio
async def test_session_key_stable_within_session_distinct_across_prompts(aux_spec):
    aux_backend = StubBackend()
    pipeline = make_pipeline(aux_backend, aux_spec)

    from foresight.context import InboundRequest

    ctx_a1 = await pipeline.handle(InboundRequest(body=user_only("task A")))
    ctx_a2 = await pipeline.handle(
        InboundRequest(body=with_history("task A", {"role": "assistant", "content": "x"}))
    )
    ctx_b1 = await pipeline.handle(InboundRequest(body=user_only("task B")))

    assert ctx_a1.session_key == ctx_a2.session_key
    assert ctx_a1.session_key != ctx_b1.session_key


@pytest.mark.asyncio
async def test_identical_prompt_collision_is_documented_known_behaviour(aux_spec):
    """Two concurrent *sessions* with an identical prompt share a key with the
    default prompt-hash session_key. Not a bug for GenericAdapter (mini's issue
    text is unique per instance); it is exactly why SweCiAdapter (M3) keys on
    the container ID instead. Pinned here so a future change is deliberate.
    """
    aux_backend = StubBackend()
    pipeline = make_pipeline(aux_backend, aux_spec)

    from foresight.context import InboundRequest

    same_prompt = user_only("identical task text")
    ctx1 = await pipeline.handle(InboundRequest(body=same_prompt))
    ctx2 = await pipeline.handle(InboundRequest(body=same_prompt))

    assert ctx1.session_key == ctx2.session_key


@pytest.mark.asyncio
async def test_session_start_overwrites_existing_entry(aux_spec):
    """A fresh session start (no assistant message) always re-runs aux, even
    if a cached entry exists under the same key -- in-flight state, not a
    long-lived cache.
    """
    aux_backend = StubBackend(reply_text=["first aux text", "second aux text"])
    pipeline = make_pipeline(aux_backend, aux_spec)

    from foresight.context import InboundRequest

    body = user_only("same task text")
    ctx1 = await pipeline.handle(InboundRequest(body=body))
    ctx2 = await pipeline.handle(InboundRequest(body=body))  # same key, session-start again

    assert len(aux_backend.calls) == 2
    assert "first aux text" in ctx1.body_out["messages"][0]["content"]
    assert "second aux text" in ctx2.body_out["messages"][0]["content"]


@pytest.mark.asyncio
async def test_no_session_entry_notes_instead_of_crashing(aux_spec):
    """Joining a session with no cached entry (TTL expiry, restart) must not
    crash -- it notes and forwards without an aux block rather than 500ing.
    """
    aux_backend = StubBackend()
    pipeline = make_pipeline(aux_backend, aux_spec, ttl_s=3600.0)

    from foresight.context import InboundRequest

    # A "continuation" request with no matching prior session-start.
    body = with_history("never seen before", {"role": "assistant", "content": "x"})
    ctx = await pipeline.handle(InboundRequest(body=body))

    assert "no_session_entry" in ctx.notes
    assert len(aux_backend.calls) == 0
