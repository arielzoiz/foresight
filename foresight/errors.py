"""Errors that foresight surfaces to its caller.

Milestone 1 assumes the aux step succeeds. When it does not there is no
fallback and no degraded path -- the request fails loudly. The alternative,
degrading to an unenhanced prompt, would silently move an instance into the
control arm of the experiment while the config still says "treatment", which is
the failure mode worth engineering against.
"""

from __future__ import annotations


class ForesightError(Exception):
    """Base class for errors foresight itself raises."""

    #: HTTP status used when this surfaces to the caller.
    status_code = 500
    #: Value of ``error.type`` in the OpenAI-shaped error body.
    error_type = "foresight_error"


class AuxFailure(ForesightError):
    """The aux step did not produce usable text.

    Raised when the aux call errors, returns a non-200, times out, or returns
    text that is empty after stripping. Rendered as a 502 with an OpenAI-shaped
    body, so the caller sees exactly what it would see from any upstream that is
    down -- no harness needs to know foresight exists.
    """

    status_code = 502
    error_type = "aux_failure"


class TraceFailure(ForesightError):
    """A trace record could not be written.

    Fatal by choice, not by necessity. A run whose traces silently stopped
    produces results nobody can interpret afterwards -- the same failure mode
    AuxFailure exists to prevent, one layer out. Most causes (an unwritable
    path, a missing parent directory) are caught at startup instead; see
    ``config._build_tracer``, which raises ConfigError there so this stays rare.

    Two places deliberately do not honour it:

    * The request is already failing. Converting a 502 aux_failure into a 500
      trace_failure would hide the cause the operator needs; server.py logs the
      trace failure and re-raises the original.
    * The response is a stream. By the time the last chunk has been relayed the
      status and body are already on the wire, and HTTP offers no way to
      retract them. Raising there aborts the stream mid-flight, which is loud
      but is not a 500.
    """

    status_code = 500
    error_type = "trace_failure"


class GuardViolation(ForesightError):
    """The aux agent modified the target's workspace.

    Aux is read-only by prompt instruction only; WorkspaceGuard detects
    violations rather than preventing them. When one is detected the measured
    artefact is already corrupt, so the request fails -- guards.py argues the
    policy at length. The changed paths are named in the message, because a
    manifest diff can name them and a hash could not.

    Distinct from AuxFailure: aux did its job, and then did more than its job.
    """

    status_code = 500
    error_type = "workspace_contaminated"

    def __init__(self, changed: list[str], root: str) -> None:
        self.changed = changed
        self.root = root
        shown = ", ".join(changed[:5])
        more = f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""
        super().__init__(
            f"aux modified {len(changed)} path(s) under {root}: {shown}{more}"
        )


class ConfigError(ForesightError):
    """The configuration is unusable. Raised at startup, never per request."""


def error_body(exc: ForesightError) -> dict:
    """Render an exception as an OpenAI-shaped error body."""
    return {"error": {"message": f"foresight: {exc}", "type": exc.error_type}}
