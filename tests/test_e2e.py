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

from foresight.config import Runtime, load_config
from foresight.server import create_app
from tools.fake_upstream import create_app as create_fake_app

FIXTURES = Path(__file__).parent / "fixtures"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest_asyncio.fixture
async def wired_apps(tmp_path):
    """A real foresight app backed by a real fake-upstream app, in-process."""
    record_path = tmp_path / "upstream.jsonl"
    trace_path = tmp_path / "traces" / "naive.jsonl"
    fake_app = create_fake_app(record_path)
    fake_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app))

    # The real shipped config, with only its trace path redirected: otherwise a
    # test run would append to the repo's own traces/naive.jsonl for ever.
    config = load_config("configs/naive.yaml")
    config.trace.path = str(trace_path)
    runtime = Runtime(config, http_client=fake_client)
    foresight_app = create_app(runtime)

    transport = httpx.ASGITransport(app=foresight_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, record_path, trace_path

    await runtime.aclose()


@pytest.mark.asyncio
async def test_aux_block_reaches_target_with_exactly_one_aux_call(wired_apps):
    client, record_path, trace_path = wired_apps

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

    # And the same exchange, as foresight recorded it: one trace row whose
    # prompt_out is what the upstream actually received.
    traces = _read_jsonl(trace_path)
    assert len(traces) == 1
    row = traces[0]
    assert row["role"] == "target"
    assert row["enhanced"] is True
    assert row["is_session_start"] is True
    assert row["error"] is None
    assert row["upstream_status"] == 200
    assert row["prompt_in"] == "Fix the bug in parse_date()"
    assert row["prompt_out"] == target_prompt
    assert row["usage"]["aux"]["prompt_tokens"] > 0
    assert row["usage"]["upstream"]["prompt_tokens"] > 0
    assert row["timings"]["aux_s"] >= 0


@pytest.mark.asyncio
async def test_streaming_relays_raw_sse_unmodified(wired_apps):
    client, record_path, trace_path = wired_apps

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

    # The trace is written after the stream drains, and records what a stream
    # structurally cannot supply: no HTTP status, and no usage unless the caller
    # asked for it via stream_options (this request did not).
    traces = _read_jsonl(trace_path)
    assert len(traces) == 1
    assert traces[0]["stream"] is True
    assert traces[0]["upstream_status"] is None
    assert traces[0]["usage"]["upstream"] is None
    assert traces[0]["enhanced"] is True
    assert "stream_incomplete" not in traces[0]["notes"]


@pytest.mark.asyncio
async def test_streaming_usage_is_recovered_when_caller_opts_in(wired_apps):
    """stream_options passes through untouched, and its usage lands in the trace.

    The only way target token counts exist on a streamed request: foresight must
    not inject include_usage itself (that would modify the caller's body), so
    this is the caller-opted-in path, recovered by UsageSniffer watching the
    raw bytes go past.
    """
    client, record_path, trace_path = wired_apps

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "target-model",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Fix the bug in parse_date()"}],
        },
    )
    assert resp.status_code == 200

    # Forwarded verbatim -- foresight added nothing and removed nothing.
    target_call = next(r for r in _read_jsonl(record_path) if r["body"]["model"] == "fake-target")
    assert target_call["body"]["stream_options"] == {"include_usage": True}

    traces = _read_jsonl(trace_path)
    assert len(traces) == 1
    assert traces[0]["usage"]["upstream"]["prompt_tokens"] > 0
    assert traces[0]["usage"]["aux"]["prompt_tokens"] > 0


@pytest.mark.asyncio
async def test_opencode_style_tool_definitions_survive_untouched(wired_apps):
    client, record_path, trace_path = wired_apps
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
    client, record_path, trace_path = wired_apps

    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "aux-model", "messages": [{"role": "user", "content": "raw text"}]},
    )
    assert resp.status_code == 200

    records = _read_jsonl(record_path)
    assert len(records) == 1
    assert records[0]["body"]["messages"][0]["content"] == "raw text"

    # Traced anyway: routing aux through foresight is what puts every model call
    # of the experiment in one file. No pipeline fields, because there was no
    # RequestContext.
    traces = _read_jsonl(trace_path)
    assert len(traces) == 1
    assert traces[0]["role"] == "aux"
    assert traces[0]["served_model"] == "aux-model"
    assert traces[0]["usage"]["aux"] is None
    assert traces[0]["usage"]["upstream"]["prompt_tokens"] > 0
    assert "session_key" not in traces[0]
    assert "enhanced" not in traces[0]
