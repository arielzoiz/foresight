"""The pipeline: the gate, then stages, then the builder.

Three decisions per request, not two:

* Is this a turn of the agent session at all? A harness's own bookkeeping
  calls (opencode's title-generation request) look like a fresh session but
  carry no tools -- see CallerAdapter.is_agent_turn. If not, the request is
  forwarded exactly as the control arm would forward it: no key resolved, no
  aux, no builder.
* Run the aux step? Only at session start. It is the expensive part -- from
  milestone 2, a whole agent run -- and its output does not change mid-session.
* Apply the builder? On EVERY request, using the stored result.

The third is required for correctness, not an optimisation, and it is the
single easiest thing in this system to get wrong. See SessionStore below.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import sys
import time
from collections.abc import Sequence

from .adapters.base import AuxResult, CallerAdapter
from .builders import Builder
from .context import InboundRequest, RequestContext
from .stages import Stage


class SessionStore:
    """Holds one aux result per in-flight session, so the builder can re-inject it.

    Not a performance cache. After a request returns, the aux text exists
    nowhere else, and the builder needs it again on every later request of the
    session.

    The model has no memory: each request is a complete, fresh input. The
    harness owns the conversation and manages it correctly -- but its copy holds
    the ORIGINAL prompt, because we rewrite on the wire, after it hands us the
    body. So it re-sends the unenhanced prompt every time, and we cannot write
    into its state (we get base_url and model_name, nothing more).

        req 1  [system, user(TASK)]                        -> assistant(tool_call)
        req 2  [system, user(TASK), assistant, tool(out)]  -> ...
                 ^ still the ORIGINAL text, every time

    Drop the store and the alternatives are: stop rewriting after request 1 (the
    enhancement reaches the model for one turn out of dozens, and the agent does
    its real work in the rest), or re-run aux per request (minutes per call from
    milestone 2 on, and non-deterministic -- the injected text would change each
    turn, the assistant's earlier replies would answer text no longer present,
    and the idempotency check would break, since it renders the prefix from this
    aux).

    This is target-side state. Aux's own conversation is owned by us, starts and
    ends inside a single target request, and leaves behind only the string kept
    here. Milestone 1 has no aux agent at all -- one chat call -- and still
    needs it.

    Lifetime: in-flight state, not a long-lived cache. A session start always
    overwrites any existing entry under its key, and entries expire on a TTL.
    """

    def __init__(self, ttl_s: float = 3600.0) -> None:
        self._ttl_s = ttl_s
        self._entries: dict[str, tuple[AuxResult, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def lock(self, key: str) -> asyncio.Lock:
        """Per-key lock, so two concurrent session starts run aux once."""
        return self._locks.setdefault(key, asyncio.Lock())

    def get(self, key: str) -> AuxResult | None:
        self._sweep()
        entry = self._entries.get(key)
        return entry[0] if entry else None

    def get_if_created_after(self, key: str, since: float) -> AuxResult | None:
        """The cached result, but only if it was written after ``since``.

        Lets a session-start request tell "a concurrent duplicate just produced
        this while I was waiting for the lock" (reuse it) apart from "this
        entry is left over from an earlier, unrelated session that happened to
        collide on this key" (recompute and overwrite, per the class's own
        overwrite policy). ``since`` should be a timestamp taken before the
        caller started waiting for the per-key lock.
        """
        self._sweep()
        entry = self._entries.get(key)
        if entry is None:
            return None
        result, created_at = entry
        return result if created_at >= since else None

    def put(self, key: str, result: AuxResult) -> None:
        self._entries[key] = (result, time.monotonic())

    def _sweep(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, at) in self._entries.items() if now - at > self._ttl_s]
        for key in expired:
            del self._entries[key]
            self._locks.pop(key, None)

    def __len__(self) -> int:
        return len(self._entries)


#: Requests skipped as non-agent turns before a run enhancing nothing is
#: called out. A healthy opencode session always skips its title request and
#: then enhances the real turn within seconds, so a warning on the first skip
#: would fire on every correct run and be ignored. This threshold only trips
#: when N requests have gone by with zero enhancements -- e.g. a caller whose
#: config never advertises tools (mini-swe-agent's text-based model classes).
SILENT_RUN_WARNING_AFTER = 10


class Pipeline:
    """Gate non-agent-turn requests, decide whether to run stages, then always
    run the builder on what remains."""

    def __init__(
        self,
        adapter: CallerAdapter,
        stages: Sequence[Stage],
        builder: Builder,
        store: SessionStore,
        max_concurrent_aux: int = 0,
    ) -> None:
        self._adapter = adapter
        self._stages = list(stages)
        self._builder = builder
        self._store = store
        self._aux_gate = asyncio.Semaphore(max_concurrent_aux) if max_concurrent_aux > 0 else None
        self.counters = {"enhanced": 0, "skipped_not_agent": 0, "no_session_entry": 0}
        self._warned = False

    async def handle(self, req: InboundRequest) -> RequestContext:
        if not self._adapter.is_agent_turn(req):
            return self._skip(req)

        ctx = RequestContext(
            req=req,
            body_out=copy.deepcopy(req.body),
            adapter_name=self._adapter.name,
            builder_name=self._builder.name,
            session_key=self._adapter.session_key(req),
            is_session_start=self._adapter.is_session_start(req),
            phase=self._adapter.phase(req),
        )

        # Stages run whenever configured to, independent of which builder is
        # selected: passthrough (control) still pays the aux cost, so a run
        # configured as control never becomes cheaper than treatment for
        # reasons unrelated to the prompt. `stages: []` is the way to opt out.
        if self._stages:
            await self._resolve_aux(ctx)

        started = time.monotonic()
        self._builder.build(ctx)
        ctx.timings["build_s"] = round(time.monotonic() - started, 4)
        if ctx.enhanced:
            self.counters["enhanced"] += 1
        return ctx

    def _skip(self, req: InboundRequest) -> RequestContext:
        """A request that is not a turn of the agent session: forward as-is.

        No session key is resolved -- deliberately. For SweCiAdapter that
        would mean a docker inspect for a request we have already decided not
        to touch, and a container that fails to resolve could then fail a
        request that was going to be forwarded unchanged anyway.
        """
        ctx = RequestContext(
            req=req,
            body_out=copy.deepcopy(req.body),
            adapter_name=self._adapter.name,
            builder_name=self._builder.name,
            is_session_start=self._adapter.is_session_start(req),
            is_agent_turn=False,
            phase=self._adapter.phase(req),
        )
        ctx.note("not_agent_turn: no tools advertised; forwarded unenhanced")
        self.counters["skipped_not_agent"] += 1
        self._maybe_warn()
        return ctx

    def _maybe_warn(self) -> None:
        if self._warned or self.counters["enhanced"] > 0:
            return
        if self.counters["skipped_not_agent"] < SILENT_RUN_WARNING_AFTER:
            return
        self._warned = True
        print(
            f"foresight: {self.counters['skipped_not_agent']} requests skipped as "
            "non-agent turns and none enhanced yet. If this caller never advertises "
            "a `tools` array, set adapter.require_tools: false -- otherwise every "
            "request of this run is in the control arm.",
            file=sys.stderr,
        )

    async def _resolve_aux(self, ctx: RequestContext) -> None:
        key = ctx.session_key

        if not ctx.is_session_start:
            # The common case, and lock-free: a continuation request only
            # reads, so it must not serialise behind other requests of the
            # same session.
            cached = self._store.get(key)
            if cached is None:
                # We joined a session already in progress -- foresight
                # restarted, or the TTL expired under a very long session.
                # Re-running aux here would inject different text than earlier
                # turns carried.
                ctx.note("no_session_entry")
                self.counters["no_session_entry"] += 1
                return
            ctx.extras["aux"] = cached
            return

        # Session start: only this path writes, so only this path needs the
        # lock. Taken before acquiring it, `wait_started` is what lets two
        # concurrent session-starts for the same key -- e.g. a harness retrying
        # a request that timed out while our (slow) aux call was in flight --
        # collapse into one aux run instead of two. See SessionStore.
        wait_started = time.monotonic()
        async with self._store.lock(key):
            fresh = self._store.get_if_created_after(key, wait_started)
            if fresh is not None:
                ctx.extras["aux"] = fresh
                ctx.note("reused_concurrent_aux")
                return

            # The concurrency gate sits INSIDE the key lock and after the
            # dedupe check above, not around the whole method. A concurrent
            # duplicate session start must be discovered by
            # get_if_created_after before it ever queues for a slot --
            # otherwise it would consume a slot, block on the key lock, then
            # find `fresh` and release having done nothing: serialising the
            # exact case the store already collapses for free. Every task
            # acquires in the same fixed order (key lock, then semaphore) and
            # holds at most one key lock, so there is no cycle; aux is bounded
            # by adapter.timeout_s, so every slot is eventually released.
            gate = self._aux_gate or contextlib.nullcontext()
            queued = time.monotonic()
            async with gate:
                ctx.timings["aux_wait_s"] = round(time.monotonic() - queued, 3)
                for stage in self._stages:
                    await stage.run(ctx)
                result = ctx.extras.get("aux")
                if result is not None:
                    self._store.put(key, result)
