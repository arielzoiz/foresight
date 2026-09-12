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
"""

from __future__ import annotations

import argparse
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .backends import Backend
from .config import Runtime, build_runtime
from .context import InboundRequest
from .errors import ForesightError, error_body
from .llm import ModelSpec


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
            return await _forward(backend, spec, dict(body), stream=req.stream)

        started = time.monotonic()
        ctx = await runtime.pipeline.handle(req)
        ctx.timings["pipeline_s"] = round(time.monotonic() - started, 4)
        return await _forward(backend, spec, ctx.body_out, stream=req.stream)

    return app


async def _forward(backend: Backend, spec: ModelSpec, body: dict, *, stream: bool):
    if stream:
        return StreamingResponse(
            backend.stream(spec, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    upstream = await backend.complete(spec, body)
    return JSONResponse(status_code=upstream.status, content=upstream.json)


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
