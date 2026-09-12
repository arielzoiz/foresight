"""OpenAICompatBackend: the wire-format translation layer.

This is where `model` gets rewritten from the served name to the upstream id,
and where everything else in the body must pass through untouched. Uses
httpx.MockTransport rather than real sockets.
"""

from __future__ import annotations

import json

import httpx
import pytest

from foresight.backends import OpenAICompatBackend
from foresight.llm import ModelSpec


def _echo_transport(captured: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append({"body": body, "headers": dict(request.headers)})
        return httpx.Response(200, json={"echo": body})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_prepare_rewrites_model_to_upstream_id():
    captured: list[dict] = []
    client = httpx.AsyncClient(transport=_echo_transport(captured))
    backend = OpenAICompatBackend(client)
    spec = ModelSpec(served_name="target-model", model="fake-target", base_url="http://x/v1")

    await backend.complete(spec, {"model": "target-model", "messages": []})

    assert captured[0]["body"]["model"] == "fake-target"
    await client.aclose()


@pytest.mark.asyncio
async def test_prepare_forwards_unknown_fields_untouched():
    captured: list[dict] = []
    client = httpx.AsyncClient(transport=_echo_transport(captured))
    backend = OpenAICompatBackend(client)
    spec = ModelSpec(served_name="target-model", model="fake-target", base_url="http://x/v1")

    body = {
        "model": "target-model",
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "parallel_tool_calls": True,
        "reasoning_effort": "medium",
        "provider_options": {"custom": {"weird_field": 1}},
    }
    await backend.complete(spec, body)

    sent = captured[0]["body"]
    assert sent["tools"] == body["tools"]
    assert sent["parallel_tool_calls"] is True
    assert sent["reasoning_effort"] == "medium"
    assert sent["provider_options"] == body["provider_options"]
    await client.aclose()


@pytest.mark.asyncio
async def test_no_api_key_env_means_no_authorization_header():
    captured: list[dict] = []
    client = httpx.AsyncClient(transport=_echo_transport(captured))
    backend = OpenAICompatBackend(client)
    spec = ModelSpec(served_name="target-model", model="fake-target", base_url="http://x/v1")

    await backend.complete(spec, {"model": "target-model", "messages": []})

    assert "authorization" not in captured[0]["headers"]
    await client.aclose()


@pytest.mark.asyncio
async def test_api_key_env_adds_bearer_header(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-123")
    captured: list[dict] = []
    client = httpx.AsyncClient(transport=_echo_transport(captured))
    backend = OpenAICompatBackend(client)
    spec = ModelSpec(
        served_name="target-model",
        model="fake-target",
        base_url="http://x/v1",
        api_key_env="TEST_KEY",
    )

    await backend.complete(spec, {"model": "target-model", "messages": []})

    assert captured[0]["headers"]["authorization"] == "Bearer secret-123"
    await client.aclose()


@pytest.mark.asyncio
async def test_non_200_relays_status_faithfully():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(client)
    spec = ModelSpec(served_name="target-model", model="fake-target", base_url="http://x/v1")

    result = await backend.complete(spec, {"model": "target-model", "messages": []})

    assert result.status == 429
    assert result.json["error"]["message"] == "rate limited"
    await client.aclose()
