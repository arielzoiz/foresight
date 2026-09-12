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


class ConfigError(ForesightError):
    """The configuration is unusable. Raised at startup, never per request."""


def error_body(exc: ForesightError) -> dict:
    """Render an exception as an OpenAI-shaped error body."""
    return {"error": {"message": f"foresight: {exc}", "type": exc.error_type}}
