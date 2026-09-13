"""The OpenAI-compatible endpoint.

POST /v1/chat/completions, GET /v1/models, GET /health.

Routing is by served model name, decided before any pipeline logic runs:

    model == target.served_name  ->  full pipeline  ->  target model
    model == aux.served_name     ->  bypass         ->  aux model
    anything else                ->  400, loudly

Aux agents are pointed at foresight rather than directly at their model. That
costs one local hop and buys three things: the Backend layer serves aux as well
as target, aux token counts land in our accounting for free, and every model
call in the experiment appears in one place. The bypass is one branch at the top
of the handler, with a test asserting it never enhances.

Why the request body is never a Pydantic model
----------------------------------------------
Declaring a request model would make FastAPI validate and rebuild the body from
that model, silently dropping every field we did not declare -- opencode's
provider options, reasoning_effort, cache_control blocks. That collides with the
requirement that a harness cannot distinguish foresight from vLLM. So the route
takes a raw Request, and three rules keep the dict intact:

* read through InboundRequest helpers, never a typed model;
* one mutation point -- deepcopy into ctx.body_out, then only the builder
  (message content) and the backend (model) assign into it;
* never round-trip: ``Model(**body).model_dump()`` is the exact line that drops
  fields, and the conformance test exists to pin its absence.

Precision about the guarantee: request.json() -> httpx(json=...) is still a
re-serialisation. Key order survives; whitespace and unicode escaping may not.
The guarantee is FIELD-PRESERVING, not byte-identical -- which is the one that
matters, since the upstream parses JSON and nothing here signs over raw bytes.

Where the trace is written
--------------------------
After the upstream exchange finishes, not after the pipeline: the record carries
the target model's token usage, which does not exist until the reply arrives.
``_forward`` therefore takes an ``on_complete`` callback and is the single place
that knows the exchange is over -- immediately for a JSON reply, and after the
last chunk has been relayed for a stream.

Three paths reach a model, and all three are recorded: the pipeline path, the
aux bypass, and a request that failed in the pipeline. The 400 for an unknown
served name is not, because no model was called.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import trace
from .backends import Backend
from .config import Runtime, build_runtime
from .context import InboundRequest
from .errors import ForesightError, error_body
from .llm import ModelSpec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    #: Called once the upstream exchange is finished: (status, usage, notes).
    #: ``status`` and ``usage`` are None on a stream -- see trace.py.
    #:
    #: Kept behind TYPE_CHECKING deliberately. A type alias is an ordinary
    #: runtime expression, so `from __future__ import annotations` does not
    #: defer it, and `int | None` inside a subscript is a TypeError before 3.10.
    #: Annotations *referring* to it stay lazy strings, and nothing introspects
    #: these two functions -- FastAPI only reads the decorated route signatures.
    OnComplete = Callable[[int | None, dict | None, list[str]], None]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "foresight"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: str
    adapter: str
    builder: str
    sessions: int


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="foresight", version="0.1.0")
    app.state.runtime = runtime

    @app.exception_handler(ForesightError)
    async def _foresight_error(request: Request, exc: ForesightError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=error_body(exc))

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            adapter=runtime.adapter.name,
            builder=runtime.builder.name,
            sessions=len(runtime.store),
        )

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        return ModelsResponse(
            data=[ModelCard(id=name) for name in sorted(runtime.by_served_name)]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        served = body.get("model")

        route = runtime.by_served_name.get(served)
        if route is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": (
                            f"foresight: unknown model {served!r}; "
                            f"served models are {sorted(runtime.by_served_name)}"
                        ),
                        "type": "model_not_found",
                    }
                },
            )

        role, spec, backend = route
        req = InboundRequest(
            body=body,
            client_ip=request.client.host if request.client else None,
            path=request.url.path,
            headers=dict(request.headers),
        )

        if role == "aux":
            # Bypass: aux traffic is relayed, never enhanced. Enhancing it would
            # mean asking aux about future work for its own aux prompt.
            def on_bypass_complete(
                status: int | None, usage: dict | None, notes: list[str]
            ) -> None:
                _trace(
                    runtime,
                    trace.bypass_record(
                        req,
                        served_model=served,
                        upstream_status=status,
                        usage_upstream=usage,
                        extra_notes=notes,
                    ),
                )

            return await _forward(
                backend, spec, dict(body), stream=req.stream, on_complete=on_bypass_complete
            )

        started = time.monotonic()
        try:
            ctx = await runtime.pipeline.handle(req)
        except ForesightError as exc:
            _trace_error(runtime, req, role=role, served_model=served, exc=exc)
            raise
        ctx.timings["pipeline_s"] = round(time.monotonic() - started, 4)

        def on_target_complete(
            status: int | None, usage: dict | None, notes: list[str]
        ) -> None:
            _trace(
                runtime,
                trace.target_record(
                    ctx,
                    served_model=served,
                    upstream_status=status,
                    usage_upstream=usage,
                    extra_notes=notes,
                ),
            )

        return await _forward(
            backend, spec, ctx.body_out, stream=req.stream, on_complete=on_target_complete
        )

    return app


async def _forward(
    backend: Backend,
    spec: ModelSpec,
    body: dict,
    *,
    stream: bool,
    on_complete: OnComplete,
):
    if stream:
        return StreamingResponse(
            _traced_stream(backend, spec, body, on_complete),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    upstream = await backend.complete(spec, body)
    usage = upstream.json.get("usage") if isinstance(upstream.json, dict) else None
    # Before the response, so a TraceFailure here surfaces as a 500 rather than
    # riding along behind a 200 that has already been committed.
    on_complete(upstream.status, usage if isinstance(usage, dict) else None, [])
    return JSONResponse(status_code=upstream.status, content=upstream.json)


async def _traced_stream(
    backend: Backend, spec: ModelSpec, body: dict, on_complete: OnComplete
) -> AsyncIterator[bytes]:
    """Relay the stream untouched, then record it.

    Every chunk is yielded before the sniffer sees it, so nothing in the trace
    path can alter or delay what reaches the caller.

    The record is written once the stream ends, which is necessarily after the
    status and headers went out. A TraceFailure on the success path therefore
    aborts the stream instead of becoming a 500 -- loud, but as loud as HTTP
    allows. On a stream that was already failing (an upstream error, or the
    client hanging up, which arrives as CancelledError) the trace is best-effort:
    masking the original exception would cost more than the record is worth.
    """
    sniffer = trace.UsageSniffer()
    notes: list[str] = []
    try:
        async for chunk in backend.stream(spec, body):
            yield chunk
            sniffer.feed(chunk)
    except BaseException:
        notes.append("stream_incomplete")
        try:
            on_complete(None, sniffer.usage(), notes)
        except Exception as exc:  # noqa: BLE001 -- never mask the stream failure
            print(f"foresight: trace write failed on aborted stream: {exc}", file=sys.stderr)
        raise
    else:
        on_complete(None, sniffer.usage(), notes)


def _trace(runtime: Runtime, record: dict) -> None:
    """Write a record, or do nothing when tracing is disabled.

    Deliberately not guarded: a TraceFailure propagates. See errors.TraceFailure
    for why losing traces silently is the worse outcome.
    """
    if runtime.tracer is None:
        return
    runtime.tracer.write(record)


def _trace_error(
    runtime: Runtime,
    req: InboundRequest,
    *,
    role: str,
    served_model: str | None,
    exc: BaseException,
) -> None:
    """Record a request that failed inside the pipeline, then let it raise.

    The one place a trace failure does not win. The caller is already getting an
    error; turning a 502 aux_failure into a 500 trace_failure would replace the
    diagnosis with a symptom.
    """
    if runtime.tracer is None:
        return

    try:
        session_key = runtime.adapter.session_key(req)
    except Exception:  # noqa: BLE001 -- an unkeyed record still beats no record
        session_key = None

    try:
        runtime.tracer.write(
            trace.error_record(
                req,
                role=role,
                served_model=served_model,
                session_key=session_key,
                exc=exc,
            )
        )
    except Exception as write_exc:  # noqa: BLE001
        print(
            f"foresight: trace write failed while recording {exc!r}: {write_exc}",
            file=sys.stderr,
        )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="foresight")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    runtime = build_runtime(args.config)
    app = create_app(runtime)
    uvicorn.run(
        app,
        host=args.host or runtime.config.server.host,
        port=args.port or runtime.config.server.port,
    )


if __name__ == "__main__":
    main()
