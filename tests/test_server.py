"""Server routing: by served model name, decided before any pipeline logic.

Uses a lightweight stand-in for Runtime -- create_app only reads a handful of
attributes off it (adapter, builder, store, pipeline, tracer, by_served_name),
so tests build those directly with StubBackend instead of going through
config.Runtime's real httpx.AsyncClient.

``build_test_runtime`` is shared with test_trace.py, which passes a real
TraceWriter where these tests pass None.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from conftest import AUX_PROMPT, TEMPLATE, StubBackend, make_adapter
from foresight.builders import BUILDERS
from foresight.llm import ModelSpec
from foresight.pipeline import Pipeline, SessionStore
from foresight.server import create_app
from foresight.stages import AuxStage


def build_test_runtime(
    *,
    aux_status: int = 200,
    aux_reply: str = "future: X, Y, Z",
    builder_name: str = "template",
    tracer=None,
    target_status: int = 200,
):
    target_spec = ModelSpec(served_name="target-model", model="fake-target", base_url="http://x/v1")
    aux_spec = ModelSpec(served_name="aux-model", model="fake-aux", base_url="http://x/v1")

    target_backend = StubBackend(reply_text="target reply", status=target_status)
    aux_backend = StubBackend(reply_text=aux_reply, status=aux_status)

    adapter = make_adapter(aux_backend, aux_spec)
    builder = BUILDERS[builder_name](TEMPLATE) if builder_name == "template" else BUILDERS[builder_name]()
    store = SessionStore()
    pipeline = Pipeline(
        adapter=adapter, stages=[AuxStage(adapter, aux_spec)], builder=builder, store=store
    )

    runtime = SimpleNamespace(
        adapter=adapter,
        builder=builder,
        store=store,
        pipeline=pipeline,
        tracer=tracer,
        by_served_name={
            "target-model": ("target", target_spec, target_backend),
            "aux-model": ("aux", aux_spec, aux_backend),
        },
    )
    return runtime, target_backend, aux_backend


async def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_and_models():
    runtime, _, _ = build_test_runtime()
    app = create_app(runtime)
    async with await _client(app) as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["adapter"] == "generic"

        models = await client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
        assert ids == {"target-model", "aux-model"}


@pytest.mark.asyncio
async def test_unknown_model_returns_400():
    runtime, _, _ = build_test_runtime()
    app = create_app(runtime)
    async with await _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
async def test_target_request_is_enhanced_via_pipeline():
    runtime, target_backend, aux_backend = build_test_runtime()
    app = create_app(runtime)
    async with await _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "target-model", "messages": [{"role": "user", "content": "fix bug"}]},
        )

    assert resp.status_code == 200
    assert len(aux_backend.calls) == 1
    assert len(target_backend.calls) == 1
    sent_to_target = target_backend.calls[0]["messages"][0]["content"]
    assert "future: X, Y, Z" in sent_to_target
    assert "fix bug" in sent_to_target
    # Rewriting `model` to the upstream id is OpenAICompatBackend's job, not
    # the server's or the pipeline's -- StubBackend deliberately does not
    # replicate it. See test_backends.py::test_prepare_rewrites_model.


@pytest.mark.asyncio
async def test_aux_model_bypass_never_enhances_and_never_touches_store():
    runtime, target_backend, aux_backend = build_test_runtime()
    app = create_app(runtime)
    async with await _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "aux-model", "messages": [{"role": "user", "content": "raw aux traffic"}]},
        )

    assert resp.status_code == 200
    assert len(aux_backend.calls) == 1
    sent = aux_backend.calls[0]["messages"][0]["content"]
    assert sent == "raw aux traffic"  # relayed unchanged, not run through the builder
    assert sent != "future: X, Y, Z"  # never enhanced
    assert len(runtime.store) == 0  # bypass never touches session state


@pytest.mark.asyncio
async def test_aux_failure_returns_502_and_target_is_never_called():
    runtime, target_backend, aux_backend = build_test_runtime(aux_status=500)
    app = create_app(runtime)
    async with await _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "target-model", "messages": [{"role": "user", "content": "fix bug"}]},
        )

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "aux_failure"
    assert len(target_backend.calls) == 0


@pytest.mark.asyncio
async def test_passthrough_builder_forwards_unenhanced_and_still_runs_aux():
    """Control arm: aux still runs (so a run configured as control never
    silently becomes cheaper than treatment), but the target sees the
    original prompt."""
    runtime, target_backend, aux_backend = build_test_runtime(builder_name="passthrough")
    app = create_app(runtime)
    async with await _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "target-model", "messages": [{"role": "user", "content": "fix bug"}]},
        )

    assert resp.status_code == 200
    assert len(aux_backend.calls) == 1
    assert target_backend.calls[0]["messages"][0]["content"] == "fix bug"


@pytest.mark.asyncio
async def test_streaming_request_relays_sse_from_backend():
    runtime, target_backend, _ = build_test_runtime()
    app = create_app(runtime)
    async with await _client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "target-model",
                "stream": True,
                "messages": [{"role": "user", "content": "fix bug"}],
            },
        )
    assert resp.status_code == 200
    assert b"data: " in resp.content
    assert b"[DONE]" in resp.content
