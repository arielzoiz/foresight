#!/usr/bin/env python
"""A mock agent harness, standing in for opencode while it is not installed.

What LocalAdapter needs from a harness is narrow: be a command, run in a
workspace, look at the code, write an answer to a path. This does exactly that
and nothing else -- no model call, no tool loop -- so milestone 2 is testable
with no harness installed and no tokens spent, the same way fake_upstream.py
makes the model layer testable with no GPU.

It reads real files, so a test can prove aux actually saw the workspace rather
than guessing from the prompt.

    python tools/fake_agent.py --answer-file /tmp/a.md "<prompt>"

Flags that exist to be misused, deliberately:

    --write-file NAME   create a file in the workspace
    --touch NAME        modify an existing file's mtime

Those are how the WorkspaceGuard tests get a real violation to detect. A guard
that has never been shown a genuine modification is not a guard, it is a
comment. Nothing else in the system can produce one on demand.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}


def survey(root: Path, limit: int = 12) -> list[Path]:
    """The files a real agent would plausibly open first."""
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(found) >= limit:
            break
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in {".py", ".md", ".txt", ".yaml", ".toml"}:
            found.append(path)
    return found


def build_answer(root: Path, prompt: str, files: list[Path]) -> str:
    """A deterministic answer that quotes the workspace, so tests can assert.

    Shaped like what the real aux prompt asks for -- a quoted detail then an
    enumerated list -- so it passes stages.assess_aux and exercises the same
    path a real answer would.
    """
    names = [str(p.relative_to(root)) for p in files]
    quoted = names[0] if names else "(no files found)"
    listing = "\n".join(f"- {n}" for n in names[:8])
    return (
        f"Distinctive detail: `{quoted}`\n\n"
        f"I read {len(names)} file(s) under {root}:\n{listing}\n\n"
        "Plausible FUTURE tasks:\n"
        "1. Extend the affected module with a second implementation.\n"
        "2. Add error handling around the new code path.\n"
        "3. Cover the change with tests.\n"
        "4. Document the behaviour for callers.\n"
        "5. Make the behaviour configurable.\n\n"
        f"(prompt was {len(prompt)} chars)\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="fake_agent")
    parser.add_argument("prompt", nargs="?", default="")
    parser.add_argument("--answer-file", default=None)
    parser.add_argument("--write-file", default=None, help="contaminate: create this")
    parser.add_argument("--touch", default=None, help="contaminate: bump this mtime")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0, help="to test timeouts")
    parser.add_argument("--silent", action="store_true", help="write no answer at all")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    print(f"fake_agent: exploring {root}", flush=True)

    if args.sleep:
        time.sleep(args.sleep)

    files = survey(root)
    print(f"fake_agent: read {len(files)} file(s)", flush=True)

    if args.write_file:
        (root / args.write_file).write_text("written by aux, which should not write\n")
        print(f"fake_agent: WROTE {args.write_file}", flush=True)

    if args.touch:
        target = root / args.touch
        if target.exists():
            stamp = time.time() + 10
            os.utime(target, (stamp, stamp))
            print(f"fake_agent: TOUCHED {args.touch}", flush=True)

    if not args.silent:
        answer = build_answer(root, args.prompt, files)
        if args.answer_file:
            path = Path(args.answer_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(answer)
            print(f"fake_agent: wrote answer to {path}", flush=True)
        else:
            # No answer file configured: LocalAdapter falls back to stdout.
            print(answer, flush=True)

    return args.exit_code


if __name__ == "__main__":
    sys.exit(main())
