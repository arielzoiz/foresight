#!/usr/bin/env python
"""Score one code tree: SWE-CI's maintainability index and a pylint note.

Run under SWE-CI's python (it needs ``pylint`` and ``radon``); ``collect_swe_ci_ab.py``
calls it once per code snapshot::

    python tools/swe_ci_score_helper.py <swe-ci>/src/swe_ci/benchmark/utils/score.py \\
        <code dir> [--exclude tests] [--no-pylint]

Prints one JSON object: ``{"mi": float|null, "pylint": float|null, "statements": int}``.

* ``mi`` is SWE-CI's own ``mi_score`` (radon maintainability index, weighted by
  source lines), loaded from its ``score.py`` and used unchanged. Higher is
  better; roughly 0-100.
* ``pylint`` is the pylint global note (0-10, higher is better), computed here
  and NOT with SWE-CI's ``pylint_score``. That function never passes
  ``--recursive=y``, so on a plain source directory pylint reports "No module
  named ..." and scans zero statements -- measured on the copyparty and
  inline-snapshot tasks, it returns 0 whatever the code. SWE-CI does not call it
  anywhere, so nothing else depends on that behaviour. Run outside the task's
  own environment, pylint also flags every unresolved import; the absolute
  number is therefore not meaningful, only its change from one epoch to the next.
* Both exclude the given top-level directories (default ``tests``): the tests are
  the fixed benchmark, not the code being evolved. pylint also skips macOS
  AppleDouble sidecar files (``._*``); ``mi_score`` already skips them, since they
  do not decode as text.

Single process on purpose: pylint's ``-j0`` uses multiprocessing, which on macOS
re-imports the ``__main__`` module in every worker.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import sys
from pathlib import Path


def load_score(path: str):
    spec = importlib.util.spec_from_file_location("swe_ci_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pylint_args(code: Path, exclude: list[str]) -> list[str]:
    args = [str(code), "--rcfile=", "--persistent=n", "--exit-zero", "--recursive=y", "-j1",
            # macOS AppleDouble sidecars ("._name.py"): binary files that only look like
            # python. On an external volume every moved folder gets one beside each file,
            # and pylint scores each as a syntax error (measured: it made every later
            # epoch look worse than the starting state, identically in both arms).
            r"--ignore-patterns=^\._"]
    if exclude:
        patterns = [f"^{re.escape(str(code / item))}(/.*|$)" for item in exclude]
        args.append("--ignore-paths=" + ",".join(patterns))
    return args


def pylint_note(code: Path, exclude: list[str]) -> tuple[float | None, int]:
    from pylint.lint import Run
    from pylint.reporters.text import TextReporter

    run = Run(pylint_args(code, exclude), reporter=TextReporter(io.StringIO()), exit=False)
    stats = getattr(run.linter, "stats", None)
    statements = int(getattr(stats, "statement", 0) or 0)
    note = getattr(stats, "global_note", None)
    return (float(note) if statements and note is not None else None), statements


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("score_py", help="SWE-CI's benchmark/utils/score.py")
    parser.add_argument("code_dir")
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--no-pylint", action="store_true")
    args = parser.parse_args(argv)
    code = Path(args.code_dir).resolve()
    exclude = args.exclude if args.exclude is not None else ["tests"]

    result: dict = {"mi": None, "pylint": None, "statements": 0}
    mi = load_score(args.score_py).mi_score(code, exclude)
    result["mi"] = None if mi is None or mi < 0 else float(mi)
    if not args.no_pylint:
        result["pylint"], result["statements"] = pylint_note(code, exclude)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
