"""GenericAdapter -- aux sees the request body, nothing more.

Serves milestone 1, mini-SWE-agent option A, and the tests. Every ABC default is
already correct for it, and ``run_aux`` is the ABC's shared body-only helper, so
this file adds no behaviour -- only the documentation of what it costs.

What it cannot do
-----------------
Aux never sees the codebase. It reasons about future tasks from the task text
alone, which makes any result obtained this way weaker evidence than one where
aux explored the real repo. The project's hypothesis is that anticipating future
work produces code that makes that work easier -- which aux cannot judge from a
description. So:

* For mini, option A is a bring-up step to prove config, routing and traces
  before adding container machinery. It is not a reportable condition.
* For SWE-CI it is not a condition at all: SWE-CI's request bodies are
  byte-identical across all 100 tasks, so a body-only call has no task-specific
  content to work from.

Session keying, and its one known limit
---------------------------------------
The ABC default hashes system + first user message. Two *concurrent* sessions
carrying an identical prompt therefore share a key and would share one aux
result.

Harmless where this adapter is used: mini's issue text is unique per instance,
and milestone 1 runs against a mock. It is precisely why SweCiAdapter keys on
the container ID instead. Documented as known behaviour, with a test that pins
it, so a future reader does not mistake it for an accident.

Milestone 2 note
----------------
The configured aux prompt asks the model to quote a distinctive detail of the
task before listing future tasks -- that echo is what makes milestone 1's check
meaningful rather than merely agreeable, and it rides along harmlessly into the
target prompt while nothing is being measured. DROP THE QUOTE REQUIREMENT before
milestone 2, where the injected block becomes experimentally meaningful and the
quote would be noise in the treatment prompt.
"""

from __future__ import annotations

from typing import ClassVar

from ..context import InboundRequest
from ..llm import ModelSpec
from .base import AuxResult, CallerAdapter


class GenericAdapter(CallerAdapter):
    name: ClassVar[str] = "generic"

    async def run_aux(self, req: InboundRequest, aux: ModelSpec) -> AuxResult:
        return await self._body_only_aux(req, aux)
