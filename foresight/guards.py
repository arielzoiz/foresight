"""WorkspaceGuard -- detect aux contamination of the target's workspace.

MILESTONE 1 SHIPS NO IMPLEMENTATION. This file exists to mark the seam and
record, for whoever builds milestones 2 and 3, why it is mandatory there.

Why nothing is needed yet
-------------------------
GenericAdapter launches no aux agent -- ``run_aux`` is a single chat call over
the request body. With no agent there is no workspace to contaminate, so the
milestone 1 configs set ``guard: null``. Opting out is a config value, not a
NoopGuard class.

Why milestones 2 and 3 must have one
------------------------------------
From milestone 2, aux is a real agent harness exploring a real codebase. It is
read-only *by prompt instruction only* -- nothing enforces it. The guard does
not prevent violations, it detects them:

* LocalAdapter (M2) -- aux runs directly on the user's filesystem. Highest
  stakes: there is nothing to throw away.
* SweCiAdapter (M3) -- aux runs inside the live task container. An edit under
  ``/app/code`` is copied out when the epoch ends and silently becomes part of
  the measured result.

Not needed for mini option B, where aux gets its own throwaway container and
physically cannot reach the target's workspace.

The single planned implementation, ManifestGuard:

    find <root> -type f -printf '%s %T@ %p\\n' | sort

Catches modifications, additions and deletions. Keeping the manifest rather than
hashing it means ``verify`` can *name* the changed files. A git-based guard
could not serve SWE-CI anyway -- its task setup strips ``.git*`` from the
workspace.

This is a Protocol, not an ABC: only one implementation is planned, so there is
no contract to enforce across a hierarchy. What varies is not the algorithm but
*where the workspace lives* -- inside a container for SWE-CI, on the host for
local use -- so the implementation takes an "execute this command in the
workspace" callable from the adapter, which already owns that capability.

On policy, which matters more than the detector: milestone 1 chose to raise
rather than degrade (see errors.AuxFailure). Carry that here -- on a detected
mismatch, prefer failing the request over recording a flag and continuing. A
contaminated workspace has already corrupted the measured result, and silent
contamination averaged into results is the failure mode worth engineering
against. Weigh it against the cost at experiment scale, where failing may abort
a whole task rather than one instance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkspaceGuard(Protocol):
    """Snapshot a workspace, then report what changed."""

    def snapshot(self) -> list[str]:
        """A sorted manifest of the tree: one line per file."""
        ...

    def verify(self, before: list[str]) -> list[str]:
        """Compare against an earlier snapshot.

        Returns the changed paths; an empty list means the workspace is intact.
        """
        ...


# TODO(milestone 2): ManifestGuard, per the module docstring.
