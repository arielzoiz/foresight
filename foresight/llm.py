"""ModelSpec -- one model endpoint foresight can call.

A spec pairs the name foresight *serves* (what callers ask for) with the model
it forwards to upstream. Routing is by served name:

    "target-model" -> full pipeline -> the target model
    "aux-model"    -> bypass        -> the aux model

The two are deliberately symmetric. Provider choice is a property of the spec,
not of the role, so target and aux may each be a local vLLM, a hosted API, or
anything else OpenAI-shaped.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Where a model call goes, and under what name callers know it."""

    served_name: str
    """The name callers put in the request body, e.g. "target-model"."""

    model: str
    """The upstream model id, e.g. "Qwen3-Coder-30B-A3B". Replaces served_name
    on the way out -- the upstream has never heard of our served names."""

    base_url: str
    """OpenAI-compatible base, including /v1. Trailing slash optional."""

    backend: str = "openai_compat"
    """Registry key in backends.BACKENDS."""

    api_key_env: str | None = None
    """Environment variable holding the key. Never the key itself -- $WORK is
    group-readable, so secrets must not land in files under it."""

    timeout_s: float = 3600.0
    """Generous by default. From milestone 2 the aux step is a whole agent run
    that holds the target's request open for minutes."""

    extra_body: dict = Field(default_factory=dict)
    """Merged into the outbound body. For provider knobs the caller does not set."""

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None

    @property
    def completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"
