"""JSONL trace writer -- one record per request that reached a model.

Why this exists
---------------
``RequestContext`` accumulates everything worth knowing about a request --
session key, whether aux ran, what the prompt looked like before and after,
timings, token counts -- and until this module existed, threw all of it away.
Without a record of it, an enhancement *effect* is indistinguishable from a
plumbing bug: a run where the rewrite silently stopped re-applying at turn 2
looks exactly like a run where future-task prompting simply did not help.

The trace is the experiment's primary evidence. ``tools/fake_upstream.py
--record`` was the stand-in, and it only works against the mock.

What a record holds, and what it deliberately does not
-----------------------------------------------------
``prompt_in`` and ``prompt_out`` are the *first user message* only -- the task
prompt, the one message the builder rewrites. The full ``messages`` array is not
recorded: it carries every tool result of the session, grows without bound, and
its bulk is not attributable to anything we did. Comparing in against out per
request is what verifies the re-application invariant (aux ran once, but the
enhanced prompt is present in *every* outbound request of the session).

``error`` is present on every record, ``None`` on success, so one key filters
failed instances out at analysis time.

``guard`` carries the workspace-contamination verdict from milestone 2 on:
``{"verdict": ..., "workspace_intact": ...}``, or ``None`` where no guard is
configured. Like the aux verdict it appears on every row of a session, not only
the one where the guard ran -- see ``_guard_block``.

Token counts
------------
``usage.aux`` is the aux consultation made *during* this request; ``usage.
upstream`` is the model this request was forwarded to. On a target record those
are the aux and target models. On an aux-bypass record ``upstream`` is the aux
model and ``aux`` is ``None`` -- ``role`` disambiguates.

``usage.upstream`` is ``None`` on streamed requests unless the caller set
``stream_options: {"include_usage": true}``. Nothing here can change that:
injecting the option would modify the body the harness sent, and raw
passthrough is the guarantee this project is built on. UsageSniffer recovers it
when the caller *did* ask, by watching the bytes go past without touching them.

``upstream_status`` is likewise ``None`` on streamed requests: Backend.stream
yields bytes and never exposes the HTTP status.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Private, deliberately: prompt_out must be flattened by exactly the same rule
# as InboundRequest applies to prompt_in, or the two are not comparable.
from .context import InboundRequest, RequestContext, _as_text
from .errors import TraceFailure
from .stages import assess_aux

#: How much of the end of a stream to keep while looking for a usage object.
#: The usage chunk is emitted immediately before ``data: [DONE]``, so this only
#: has to span the last few events.
USAGE_TAIL_BYTES = 16384


class TraceWriter:
    """Appends one JSON object per line to a file.

    Opens per record rather than holding a handle, which is what
    ``fake_upstream._record`` already does: no lifecycle to manage, no shutdown
    hook, and nothing to flush if the process dies. A syscall per request is
    irrelevant next to a model call.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def preflight(self) -> None:
        """Create the parent directory and prove the file is writable.

        Raises ``OSError``, which ``config._build_tracer`` turns into a
        ConfigError. Called at startup so that an unwritable path fails before
        the port is bound, rather than killing the first request of a run --
        which is what TraceFailure would otherwise do.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a"):
            pass

    def write(self, record: dict) -> None:
        # default=str: provenance and usage come from upstream JSON and are not
        # ours to trust. A record that cannot be serialised is still a trace
        # failure, but an unexpected type must not be the cause.
        line = json.dumps(record, default=str, ensure_ascii=False)
        try:
            with self.path.open("a") as f:
                f.write(line + "\n")
        except OSError as exc:
            raise TraceFailure(f"could not write trace to {self.path}: {exc}") from exc


class UsageSniffer:
    """Recovers a usage object from raw SSE bytes, without altering them.

    Read-only by construction: ``feed`` is handed each chunk *after* it has been
    yielded downstream, and keeps only a bounded tail. Trimming can behead the
    oldest line in the buffer; ``usage`` scans from the end and skips anything
    that does not parse, so a half line is simply ignored.
    """

    def __init__(self, tail_bytes: int = USAGE_TAIL_BYTES) -> None:
        self._tail = bytearray()
        self._limit = tail_bytes

    def feed(self, chunk: bytes) -> None:
        self._tail.extend(chunk)
        excess = len(self._tail) - self._limit
        if excess > 0:
            del self._tail[:excess]

    def usage(self) -> dict | None:
        for raw in reversed(bytes(self._tail).split(b"\n")):
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("usage"):
                return event["usage"]
        return None


# -- record builders ------------------------------------------------------
# Pure functions: they read state and return a dict. Keeping them separate from
# the writer is what lets tests assert on record shape without touching a disk.


def target_record(
    ctx: RequestContext,
    *,
    served_model: str | None,
    upstream_status: int | None,
    usage_upstream: dict | None,
    extra_notes: list[str] | None = None,
) -> dict:
    """The full record: a request that went through the pipeline."""
    req = ctx.req
    prompt_in = req.first_user_content()
    prompt_out = _prompt_out(ctx)
    return {
        "request_id": req.request_id,
        "received_at": _iso(req.received_at),
        "role": "target",
        "served_model": served_model,
        "adapter": ctx.adapter_name,
        "builder": ctx.builder_name,
        "session_key": ctx.session_key,
        "is_session_start": ctx.is_session_start,
        "phase": ctx.phase,
        "enhanced": ctx.enhanced,
        "prompt_in": prompt_in,
        "prompt_out": prompt_out,
        "prompt_in_chars": len(prompt_in),
        "prompt_out_chars": len(prompt_out),
        "aux": _aux_block(ctx.aux),
        "guard": _guard_block(ctx.aux),
        "message_count": len(req.messages),
        "tool_count": len(req.tools),
        "stream": req.stream,
        "upstream_status": upstream_status,
        "usage": {"aux": _aux_usage(ctx.aux), "upstream": usage_upstream},
        "timings": dict(ctx.timings),
        "notes": [*ctx.notes, *(extra_notes or [])],
        "error": None,
    }


def bypass_record(
    req: InboundRequest,
    *,
    served_model: str | None,
    upstream_status: int | None,
    usage_upstream: dict | None,
    extra_notes: list[str] | None = None,
) -> dict:
    """An aux-model request, relayed without enhancement.

    Recorded because the reason aux is pointed at foresight at all is so that
    every model call in the experiment lands in one file. The pipeline fields
    are absent rather than null: this request never had a RequestContext.

    Rare in milestone 1, where aux is a direct chat call from the adapter and
    never re-enters the server. It is the M2 aux *agent* that exercises this.
    """
    return {
        "request_id": req.request_id,
        "received_at": _iso(req.received_at),
        "role": "aux",
        "served_model": served_model,
        "message_count": len(req.messages),
        "tool_count": len(req.tools),
        "stream": req.stream,
        "upstream_status": upstream_status,
        "usage": {"aux": None, "upstream": usage_upstream},
        "notes": [*(extra_notes or [])],
        "error": None,
    }


def error_record(
    req: InboundRequest,
    *,
    role: str,
    served_model: str | None,
    session_key: str | None,
    exc: BaseException,
) -> dict:
    """A request that failed before reaching the target model.

    Thin by necessity, not by choice: AuxFailure propagates out of
    Pipeline.handle before a RequestContext is ever returned, so there are no
    timings and no notes to record. What matters is here -- which session
    failed, and why -- because excluding failed instances at analysis time is
    the alternative to averaging them in.
    """
    return {
        "request_id": req.request_id,
        "received_at": _iso(req.received_at),
        "role": role,
        "served_model": served_model,
        "session_key": session_key,
        "enhanced": False,
        "prompt_in": req.first_user_content(),
        "prompt_in_chars": len(req.first_user_content()),
        "message_count": len(req.messages),
        "tool_count": len(req.tools),
        "stream": req.stream,
        "error": {
            "type": getattr(exc, "error_type", type(exc).__name__),
            "message": str(exc),
        },
    }


def _prompt_out(ctx: RequestContext) -> str:
    """The task prompt as it will leave, read out of ``body_out``.

    Indexed by the *inbound* first-user position: the builder rewrites content
    in place and never reorders messages, so the index is shared.
    """
    index = ctx.req.first_user_index()
    if index is None:
        return ""
    messages = ctx.body_out.get("messages")
    if not isinstance(messages, list) or index >= len(messages):
        return ""
    message = messages[index]
    if not isinstance(message, dict):
        return ""
    return _as_text(message.get("content"))


def _aux_block(aux: Any) -> dict | None:
    """The aux result, plus a verdict on whether it is usable.

    ``items``/``usable`` come from ``stages.assess_aux``, recomputed here rather
    than read off the result. It is a pure function of the text, and computing
    it per row is what puts the verdict on CONTINUATION rows too -- those reuse
    a stored AuxResult and never re-enter AuxStage. Filtering an experiment on
    ``aux.usable`` therefore works on every row of a session, not just its
    first.
    """
    if aux is None:
        return None
    return {
        "source": aux.source,
        "text": aux.text,
        "provenance": aux.provenance,
        **assess_aux(aux.text),
    }


def _guard_block(aux: Any) -> dict | None:
    """The workspace-contamination verdict, on every row of the session.

    Same argument as ``_aux_block``: continuation rows reuse a stored AuxResult
    and never re-enter the adapter, so a verdict recorded only where the guard
    actually ran would be absent from most rows of the very session it
    invalidates. Read off the provenance rather than recomputed, because unlike
    ``assess_aux`` this is not a pure function of anything the row still holds --
    the snapshot it compared against is gone.

    ``None`` means no verdict: either no aux result yet, or ``guard: null`` in
    config (a legitimate setting for any adapter that launches no aux agent).

    ``workspace_intact`` is spelled out, per the design's "record a validity
    flag, filter at analysis time" policy, so one key filters. In practice it is
    only ever true here: a detected modification raises GuardViolation out of
    AuxStage and lands in ``error_record`` with error type
    "workspace_contaminated" instead. That is deliberate -- contamination fails
    the request rather than producing a result that could be averaged in -- but
    the field is written either way so that analysis code needs one rule, not
    two.
    """
    if aux is None:
        return None
    verdict = aux.provenance.get("guard")
    if verdict is None:
        return None
    return {"verdict": verdict, "workspace_intact": verdict == "intact"}


def _aux_usage(aux: Any) -> dict | None:
    if aux is None:
        return None
    usage = aux.provenance.get("usage")
    return usage if isinstance(usage, dict) else None


def _iso(epoch_s: float) -> str:
    return (
        datetime.fromtimestamp(epoch_s, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
