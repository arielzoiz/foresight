#!/usr/bin/env python3
"""A mock ``opencode`` CLI for the two verbs the aux transcript export uses.

    fake_opencode.py session list
    fake_opencode.py export <session id>

Sessions are plain files, one per id, under aux's HOME inside the fake
container -- ``<rootfs>/<HOME>/.local/share/opencode/sessions/<id>.json`` -- so a
test seeds one by writing a file and the adapter finds it the way it would find
a real one: through ``docker exec``. ``FORESIGHT_FAKE_DOCKER_ROOTFS`` is set by
``fake_docker.py exec``; ``HOME`` comes from the adapter's own ``-e HOME=...``,
and is a container path, so it is mapped onto the rootfs here.

``FAKE_OPENCODE_MODE`` exists to be misused, like ``fake_agent.py``'s flags:

    list_fail    ``session list`` exits non-zero
    export_fail  ``export`` exits non-zero
    bad_json     ``export`` prints something that is not JSON
    banner       ``export`` prints a banner line, then the JSON
    pipe_cut     ``export`` cuts its output at 64 KiB when stdout is a pipe, and is
                 whole when stdout is a file -- what the real opencode does (it
                 exits before the pipe drains)
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


def sessions_dir() -> Path:
    root = Path(os.environ.get("FORESIGHT_FAKE_DOCKER_ROOTFS", "/"))
    home = os.environ.get("HOME", "/")
    return root / home.lstrip("/") / ".local" / "share" / "opencode" / "sessions"


def main(argv: list[str]) -> int:
    mode = os.environ.get("FAKE_OPENCODE_MODE", "")
    if argv[:2] == ["session", "list"]:
        if mode == "list_fail":
            sys.stderr.write("fake_opencode: session list failed\n")
            return 3
        # opencode prints a table; the adapter greps ids out of it.
        print("Session ID                        Title      Updated")
        for path in sorted(sessions_dir().glob("ses_*.json")):
            print(f"{path.stem}  some title  just now")
        return 0
    if len(argv) == 2 and argv[0] == "export":
        if mode == "export_fail":
            sys.stderr.write("fake_opencode: export failed\n")
            return 4
        if mode == "bad_json":
            print("this is not json")
            return 0
        if mode == "banner":
            print(f"Exporting session: {argv[1]}")
        path = sessions_dir() / f"{argv[1]}.json"
        if not path.is_file():
            sys.stderr.write(f"fake_opencode: no such session {argv[1]}\n")
            return 1
        text = path.read_text()
        if mode == "pipe_cut" and not stat.S_ISREG(os.fstat(1).st_mode):
            text = text[:65536]
        sys.stdout.write(text)
        return 0
    sys.stderr.write(f"fake_opencode: unsupported: {' '.join(argv)}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
