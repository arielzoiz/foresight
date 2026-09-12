"""End-to-end: foresight ASGI app -> fake_upstream ASGI app, no sockets.

Two real FastAPI apps wired together via httpx.ASGITransport, so a request
travels through the actual routing, pipeline, builder and backend code -- the
only thing not real is the network socket. fake_upstream's --record file is
what lets us assert what the target model actually received.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foresight.config import build_runtime
from foresight.server import create_app
from tools.fake_upstream import create_app as create_fake_app

FIXTURES = Path(__file__).parent / "fixtures"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest_asyncio.fixture
async def wired_apps(tmp_path):
    """A real foresight app backed by a real fake-upstream app, in-process."""
    record_path = tmp_path / "upstream.jsonl"
    fake_app = create_fake_app(record_path)
    fake_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app))

    runtime = build_runtime("configs/naive.yaml", http_client=fake_client)
    foresight_app = create_app(runtime)

    transport = httpx.ASGITransport(app=foresight_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, record_path

    await runtime.aclose()


@pytest.mark.asyncio
async def test_aux_block_reaches_target_with_exactly_one_aux_call(wired_apps):
    client, record_path = wired_apps

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "target-model",
            "messages": [{"role": "user", "content": "Fix the bug in parse_date()"}],
        },
    )
    assert resp.status_code == 200

    records = _read_jsonl(record_path)
    aux_calls = [r for r in records if r["body"]["model"] == "fake-aux"]
    target_calls = [r for r in records if r["body"]["model"] == "fake-target"]

    assert len(aux_calls) == 1
    assert len(target_calls) == 1

    target_prompt = target_calls[0]["body"]["messages"][0]["content"]
    assert "FUTURE tasks" in target_prompt
    assert "Fix the bug in parse_date()" in target_prompt
    assert target_prompt.index("FUTURE tasks") < target_prompt.index("Fix the bug in parse_date()")


@pytest.mark.asyncio
async def test_streaming_relays_raw_sse_unmodified(wired_apps):
    client, record_path = wired_apps

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "target-model",
            "stream": True,
            "messages": [{"role": "user", "content": "Fix the bug in parse_date()"}],
        },
    )
    assert resp.status_code == 200
    body = resp.content.decode()

    # Raw SSE: multiple "data: {...}" chunks, terminated by [DONE]. If foresight
    # ever reassembled and re-serialised deltas, this shape (and tool-call
    # argument fragments split across chunks) would be exactly what breaks.
    data_lines = [line for line in body.splitlines() if line.startswith("data: ")]
    assert len(data_lines) >= 3
    assert body.strip().endswith("data: [DONE]")

    non_done = [json.loads(line[len("data: ") :]) for line in data_lines[:-1]]
    for chunk in non_done:
        assert chunk["object"] == "chat.completion.chunk"


@pytest.mark.asyncio
async def test_opencode_style_tool_definitions_survive_untouched(wired_apps):
    client, record_path = wired_apps
    fixture = json.loads((FIXTURES / "opencode_request.json").read_text())

    resp = await client.post("/v1/chat/completions", json=fixture)
    assert resp.status_code == 200

    records = _read_jsonl(record_path)
    target_call = next(r for r in records if r["body"]["model"] == "fake-target")
    sent = target_call["body"]

    # Protocol conformance: tool definitions must be byte-for-byte (structurally
    # identical) to what came in -- foresight rewrites messages, never tools.
    assert sent["tools"] == fixture["tools"]
    assert sent["parallel_tool_calls"] == fixture["parallel_tool_calls"]

    # Regression guard for the "never parse the proxied body into a Pydantic
    # model" decision: unmodelled provider fields must survive untouched.
    assert sent["reasoning_effort"] == fixture["reasoning_effort"]
    assert sent["provider_options"] == fixture["provider_options"]

    # But the task prompt (first user message) must have been enhanced.
    assert "FUTURE tasks" in sent["messages"][1]["content"]
    # The prior tool round-trip must be forwarded unchanged.
    assert sent["messages"][2]["tool_calls"] == fixture["messages"][2]["tool_calls"]
    assert sent["messages"][3] == fixture["messages"][3]


@pytest.mark.asyncio
async def test_aux_model_bypass_e2e_never_enhances(wired_apps):
    client, record_path = wired_apps

    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "aux-model", "messages": [{"role": "user", "content": "raw text"}]},
    )
    assert resp.status_code == 200

    records = _read_jsonl(record_path)
    assert len(records) == 1
    assert records[0]["body"]["messages"][0]["content"] == "raw text"
