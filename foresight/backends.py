"""Backends -- which provider a model call goes to, and in what wire format.

Inbound is always OpenAI chat-completions; the callers decide that, not us.
*Outbound* varies by provider, and absorbing that variation is this layer's
whole job.

``OpenAICompatBackend`` is the default and the only one milestone 1 needs: vLLM,
Ollama, any OpenAI-shaped endpoint, and Claude via Anthropic's compatibility
layer. It is raw passthrough with zero translation, which is precisely what
keeps tool calls intact on the main experimental path.

Later, if a provider's wire format demands it: LiteLLMBackend (Gemini, Bedrock)
and AnthropicBackend (only if Claude becomes a real condition *and* prompt
caching cost binds -- the compatibility layer supports no cache_control).
Pick a backend by what the provider's format demands, not by how many providers
one could theoretically cover: making a translation layer the default would put
it in front of vLLM, which needs none.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .llm import ModelSpec


@dataclass
class UpstreamResponse:
    """A non-streaming reply, carried with its status so errors relay faithfully.

    An upstream 429 or 400 must reach the caller as a 429 or 400, not as a 500
    from us -- a harness's retry logic reads those.
    """

    status: int
    json: Any
    headers: dict[str, str]


class Backend(ABC):
    """One provider's wire format."""

    name: str = "backend"

    @abstractmethod
    async def complete(self, spec: ModelSpec, body: dict) -> UpstreamResponse: ...

    @abstractmethod
    def stream(self, spec: ModelSpec, body: dict) -> AsyncIterator[bytes]: ...


class OpenAICompatBackend(Backend):
    """Raw passthrough to an OpenAI-compatible endpoint.

    Everything in the body is forwarded untouched -- tools, tool_choice,
    parallel_tool_calls, stream_options, and any provider extension we have
    never heard of. The only field we rewrite is ``model``, which must become
    the upstream's id because the served name is ours alone.
    """

    name = "openai_compat"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    @staticmethod
    def _prepare(spec: ModelSpec, body: dict) -> tuple[dict, dict[str, str]]:
        out = dict(body)
        out.update(spec.extra_body)
        out["model"] = spec.model
        headers = {"Content-Type": "application/json"}
        if spec.api_key:
            headers["Authorization"] = f"Bearer {spec.api_key}"
        return out, headers

    async def complete(self, spec: ModelSpec, body: dict) -> UpstreamResponse:
        out, headers = self._prepare(spec, body)
        response = await self._client.post(
            spec.completions_url, json=out, headers=headers, timeout=spec.timeout_s
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.text, "type": "upstream_error"}}
        return UpstreamResponse(
            status=response.status_code, json=payload, headers=dict(response.headers)
        )

    async def stream(self, spec: ModelSpec, body: dict) -> AsyncIterator[bytes]:
        """Relay SSE as raw bytes.

        Never reassembled and never re-serialised: rebuilding deltas reliably
        breaks tool calls, whose arguments arrive as JSON fragments split across
        chunks at arbitrary boundaries.
        """
        out, headers = self._prepare(spec, body)
        async with self._client.stream(
            "POST", spec.completions_url, json=out, headers=headers, timeout=spec.timeout_s
        ) as response:
            async for chunk in response.aiter_raw():
                yield chunk


BACKENDS: dict[str, type[Backend]] = {
    OpenAICompatBackend.name: OpenAICompatBackend,
}
