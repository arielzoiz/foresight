"""WorkspaceGuard -- detect aux contamination of the target's workspace.

Milestone 2 implements ManifestGuard. The Protocol stays, because what varies is
not the algorithm but *where the workspace lives*.

When it is not needed
---------------------
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

The single implementation, ManifestGuard:

    find <root> -type f -printf '%s %T@ %p\\n' | sort

Catches modifications, additions and deletions. Keeping the manifest rather than
hashing it means ``verify`` can *name* the changed files. A git-based guard
could not serve SWE-CI anyway -- its task setup strips ``.git*`` from the
workspace.

``-printf`` is a GNU extension, and BSD find (macOS) rejects it. The command is
therefore chosen once, by probing, with a BSD equivalent as the fallback:

    find <root> -type f -exec stat -f '%z %m %N' {} + | sort

Both emit "size mtime path" per line, so ``verify`` does not care which ran.
The Linux cluster takes the GNU path, which is the command the design specifies;
the fallback exists so the guard is developable on a Mac.

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

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    #: Runs a shell command in the workspace and returns (exit_code, stdout).
    #: Supplied by the adapter, which already owns "how do I reach this
    #: workspace" -- a subprocess for LocalAdapter, `docker exec` for SweCiAdapter.
    #: Behind TYPE_CHECKING because a type alias is a runtime expression and
    #: `tuple[int, str]` subscripting must not be evaluated on Python 3.9.
    Runner = Callable[[str], "tuple[int, str]"]

#: GNU find. What the design specifies; what the Linux cluster will use.
_GNU_MANIFEST = "find {root} -type f -printf '%s %T@ %p\\n'"
#: BSD find (macOS). Same three fields, so verify() cannot tell them apart.
_BSD_MANIFEST = "find {root} -type f -exec stat -f '%z %m %N' {{}} +"
#: Cheap probe: GNU accepts -printf, BSD exits non-zero on it.
_PROBE = "find {root} -maxdepth 0 -printf '' >/dev/null 2>&1"


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


class ManifestGuard:
    """Detect contamination by diffing a size+mtime manifest of the tree.

    Takes a ``runner`` rather than shelling out itself: for SweCiAdapter the
    workspace is inside a container, and only the adapter knows how to reach it.
    The guard supplies the algorithm, the adapter supplies the reach.

    Catches modifications (size or mtime moves), additions and deletions, and
    ``verify`` can *name* what changed -- which is the reason for keeping the
    manifest rather than hashing it into a single digest.

    Not a security boundary. A sufficiently determined agent could restore an
    mtime; nothing here prevents a write. It exists so that contamination is
    loud instead of silent, and silent contamination averaged into results is
    the failure mode the design engineers against.
    """

    def __init__(self, root: str, runner: Runner) -> None:
        self.root = root
        self._runner = runner
        self._command: str | None = None

    def _manifest_command(self) -> str:
        """Pick GNU or BSD form once, by probing the workspace's own find."""
        if self._command is None:
            code, _ = self._runner(_PROBE.format(root=_quote(self.root)))
            template = _GNU_MANIFEST if code == 0 else _BSD_MANIFEST
            self._command = template.format(root=_quote(self.root))
        return self._command

    def snapshot(self) -> list[str]:
        code, out = self._runner(self._manifest_command())
        if code != 0:
            # An unusable guard must not read as "workspace intact". This is why
            # the command carries no `| sort`: a shell pipeline reports the exit
            # status of its LAST stage, so `find ... | sort` over a missing
            # workspace returns 0 with no output -- indistinguishable from an
            # empty-but-fine tree, and verify() would then see nothing wrong.
            raise RuntimeError(
                f"workspace manifest failed for {self.root} (exit {code}): {out[:300]}"
            )
        return sorted(line for line in out.splitlines() if line.strip())

    def verify(self, before: list[str]) -> list[str]:
        """Paths whose size/mtime changed, or that appeared or disappeared."""
        after = self.snapshot()
        gone = set(before) - set(after)
        new = set(after) - set(before)
        return sorted({_path_of(line) for line in gone | new})


def _path_of(manifest_line: str) -> str:
    """The path from a "size mtime path" line.

    Split from the left exactly twice: paths may contain spaces, sizes and
    mtimes may not.
    """
    parts = manifest_line.split(None, 2)
    return parts[2] if len(parts) == 3 else manifest_line


def _quote(path: str) -> str:
    from shlex import quote

    return quote(path)
