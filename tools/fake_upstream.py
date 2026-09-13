#!/usr/bin/env python
"""A mock OpenAI-compatible server, standing in for both target and aux models.

Serves fake-target and fake-aux at /v1/chat/completions, non-streaming and SSE.
Replies are deterministic echoes, so tests can assert exact content instead of
pattern-matching a real model's output.

With --record <path>, every request body received is appended as one JSON line
-- this is what lets a test (or a human) see exactly what the target model was
sent. Note this is the *upstream's* view: it records whole bodies, including
every tool result of a session. foresight's own traces (foresight/trace.py) are
the experiment's record; these two are complementary, and this one is the only
way to see a body byte for byte as the model received it.

Streaming replies carry a usage object only when the request asks for one via
``stream_options: {"include_usage": true}``, which is what real OpenAI-compatible
servers do. foresight never injects that option -- raw passthrough is the whole
guarantee -- so this is what exercises the caller-opted-in path.

Usage:
    python tools/fake_upstream.py --port 8001 --record /tmp/upstream.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _reply_text(model: str, messages: list[dict]) -> str:
    """A deterministic, content-bearing reply.

    Echoes the model name and the length of the last message, plus (for the
    aux model) a canned future-tasks list -- concrete enough that tests can
    assert on it, and that GenericAdapter's "quote one distinctive detail"
    check has something real to quote.
    """
    last = messages[-1] if messages else {}
    content = last.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content)

    if model == "fake-aux":
        return (
            f'Distinctive detail: "{content[:60]}". '
            "Future tasks: 1) add input validation 2) support additional "
            "formats 3) add retry logic 4) improve error messages."
        )
    return f"fake-target saw {len(content)} chars, last role={last.get('role')}"


def _usage(messages: list[dict], text: str) -> dict[str, int]:
    """A plausible usage object. Deterministic, so tests can assert on it."""
    return {
        "prompt_tokens": sum(len(str(m.get("content", ""))) for m in messages) // 4,
        "completion_tokens": len(text) // 4,
        "total_tokens": 0,
    }


def _wants_stream_usage(body: dict) -> bool:
    options = body.get("stream_options")
    return bool(isinstance(options, dict) and options.get("include_usage"))


def create_app(record_path: Path | None) -> FastAPI:
    app = FastAPI(title="fake-upstream")

    def _record(body: dict) -> None:
        if record_path is None:
            return
        with record_path.open("a") as f:
            f.write(json.dumps({"received_at": time.time(), "body": body}) + "\n")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        _record(body)

        model = body.get("model", "fake-target")
        messages = body.get("messages", [])
        text = _reply_text(model, messages)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if body.get("stream"):
            return StreamingResponse(
                _sse_chunks(
                    completion_id,
                    created,
                    model,
                    text,
                    usage=_usage(messages, text) if _wants_stream_usage(body) else None,
                ),
                media_type="text/event-stream",
            )

        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(messages, text),
            }
        )

    return app


def _sse_chunks(
    completion_id: str,
    created: int,
    model: str,
    text: str,
    usage: dict | None = None,
):
    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    async def gen():
        yield chunk({"role": "assistant", "content": ""})
        # Split into a few pieces so a real SSE client sees multiple chunks --
        # this is what a re-serialising relay would be most likely to corrupt.
        step = max(1, len(text) // 4)
        for i in range(0, len(text), step):
            yield chunk({"content": text[i : i + step]})
        yield chunk({}, finish_reason="stop")
        if usage is not None:
            # The shape real servers use for include_usage: an extra event after
            # the last delta, carrying no choices, immediately before [DONE].
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return gen()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="fake_upstream")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--record", default=None, help="JSONL path to append received bodies")
    args = parser.parse_args()

    record_path = Path(args.record) if args.record else None
    app = create_app(record_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
