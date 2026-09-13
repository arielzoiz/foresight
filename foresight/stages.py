"""Stages -- work done once per session, before the builder rewrites anything.

A stage enriches a RequestContext by writing into ``ctx.extras``. Stages are a
list in config, so later milestones can add more without touching Pipeline.

Only one exists today: AuxStage, which delegates to ``adapter.run_aux``. That
delegation is the whole point of the adapter layer -- the stage does not know
whether aux was a chat call, a subprocess, or a container.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import ClassVar

import httpx

from .adapters.base import AuxResult, CallerAdapter
from .context import RequestContext
from .errors import AuxFailure, GuardViolation
from .llm import ModelSpec

#: Fewest enumerated items an aux answer must carry to count as usable.
#:
#: Measured, not guessed: 15 runs of qwen2.5-coder:7b over 3 tasks produced 14
#: answers with exactly 5 items (713-1809 chars) and one with 0 items (52
#: chars) that quoted a filename and stopped. An earlier failure was 145 chars
#: and also 0 items. The two populations do not overlap anywhere near 3, so this
#: threshold cannot plausibly misfire in either direction.
MIN_FUTURE_TASK_ITEMS = 3

#: "1. ", "2) ", and the "1. **Bold**" form the model actually favours.
_NUMBERED = re.compile(r"(?m)^\s*(?:\*\*)?(\d+)[.)]\s")
#: "- item" / "* item" / "+ item". Markdown bold ("**Detail:**") does not match:
#: the second character is another star, not a space.
_BULLET = re.compile(r"(?m)^\s*[-*+]\s+\S")


def assess_aux(text: str) -> dict:
    """Does this aux answer actually carry a list of future tasks?

    A pure function of the text, deliberately. That is what lets the verdict
    appear on EVERY trace row of a session: on continuation requests the aux
    result comes back from the SessionStore and this stage never runs, so a
    verdict computed once and stored here would be missing exactly where the
    re-application invariant is checked.

    Counts distinct numbers rather than matches, so a model that restarts its
    numbering (or writes "1." twice) is not credited twice.
    """
    numbered = len({m.group(1) for m in _NUMBERED.finditer(text)})
    bullets = len(_BULLET.findall(text))
    items = max(numbered, bullets)
    return {"items": items, "usable": items >= MIN_FUTURE_TASK_ITEMS}


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

    GuardViolation is the one exception that is not an aux failure: aux worked,
    and then did more than it was allowed to. It propagates unchanged.

    Weak aux is flagged, not raised
    -------------------------------
    A third case sits between success and failure: aux answers, but the answer
    carries no future tasks (measured at ~7% with a 7B -- it quotes a detail,
    then stops). Such an instance is in neither arm; it is noise.

    It is recorded rather than raised, which is the opposite of the choice above,
    for a reason guards.py already sets out: at experiment scale, failing a
    request aborts the harness's whole session, so a 7% rate would abort 7% of
    TASKS and bias which ones finish. The mechanical failures above cannot be
    filtered afterwards -- there is no result. This one can: the verdict is on
    every trace row, so weak instances are excluded at analysis time. That is
    the same "record a validity flag, filter later" pattern the design uses for
    the body-only fallback and for workspace contamination.
    """

    name: ClassVar[str] = "aux"

    def __init__(self, adapter: CallerAdapter, aux_spec: ModelSpec) -> None:
        self._adapter = adapter
        self._aux_spec = aux_spec

    async def run(self, ctx: RequestContext) -> None:
        started = time.monotonic()
        try:
            result = await self._adapter.run_aux(ctx.req, self._aux_spec)
        except (AuxFailure, GuardViolation):
            # GuardViolation must pass through intact. Collapsing it into
            # AuxFailure would report contamination as "aux broke" -- a 502
            # rather than a 500, and error_type "aux_failure" rather than
            # "workspace_contaminated". Those are filtered differently at
            # analysis time: a failed aux call is a lost instance, a
            # contaminated workspace is a corrupted measurement.
            raise
        except httpx.TimeoutException as exc:
            raise AuxFailure(f"aux timed out after {self._aux_spec.timeout_s}s ({exc})") from exc
        except httpx.HTTPError as exc:
            raise AuxFailure(f"aux transport error ({exc})") from exc
        except Exception as exc:  # noqa: BLE001 -- any adapter failure is an aux failure
            raise AuxFailure(f"aux raised {type(exc).__name__}: {exc}") from exc

        if not result.text.strip():
            raise AuxFailure("aux returned empty text")

        quality = assess_aux(result.text)
        if not quality["usable"]:
            # Noted here, where aux actually ran, so the session-start row says
            # so in prose. The verdict itself is recomputed per row in trace.py,
            # which is what puts it on continuation rows too.
            ctx.note(
                f"weak_aux: {quality['items']} enumerated items "
                f"(< {MIN_FUTURE_TASK_ITEMS}), {len(result.text)} chars; "
                "exclude this instance at analysis time"
            )

        ctx.extras["aux"] = result
        ctx.timings["aux_s"] = round(time.monotonic() - started, 3)
