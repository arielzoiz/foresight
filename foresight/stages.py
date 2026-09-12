"""Stages -- work done once per session, before the builder rewrites anything.

A stage enriches a RequestContext by writing into ``ctx.extras``. Stages are a
list in config, so later milestones can add more without touching Pipeline.

Only one exists today: AuxStage, which delegates to ``adapter.run_aux``. That
delegation is the whole point of the adapter layer -- the stage does not know
whether aux was a chat call, a subprocess, or a container.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar

import httpx

from .adapters.base import AuxResult, CallerAdapter
from .context import RequestContext
from .errors import AuxFailure
from .llm import ModelSpec


class Stage(ABC):
    name: ClassVar[str] = "stage"

    @abstractmethod
    async def run(self, ctx: RequestContext) -> None: ...


class AuxStage(Stage):
    """Consult aux about plausible future work.

    The expensive part of the system: from milestone 2 this is an entire agent
    run. It executes only at session start -- Pipeline decides that, not us.

    Every failure mode converges on AuxFailure, which the server renders as a
    502. Milestone 1 has no fallback by design: degrading to an unenhanced
    prompt would put an instance into the control arm while the config still
    says treatment.
    """

    name: ClassVar[str] = "aux"

    def __init__(self, adapter: CallerAdapter, aux_spec: ModelSpec) -> None:
        self._adapter = adapter
        self._aux_spec = aux_spec

    async def run(self, ctx: RequestContext) -> None:
        started = time.monotonic()
        try:
            result = await self._adapter.run_aux(ctx.req, self._aux_spec)
        except AuxFailure:
            raise
        except httpx.TimeoutException as exc:
            raise AuxFailure(f"aux timed out after {self._aux_spec.timeout_s}s ({exc})") from exc
        except httpx.HTTPError as exc:
            raise AuxFailure(f"aux transport error ({exc})") from exc
        except Exception as exc:  # noqa: BLE001 -- any adapter failure is an aux failure
            raise AuxFailure(f"aux raised {type(exc).__name__}: {exc}") from exc

        if not result.text.strip():
            raise AuxFailure("aux returned empty text")

        ctx.extras["aux"] = result
        ctx.timings["aux_s"] = round(time.monotonic() - started, 3)
