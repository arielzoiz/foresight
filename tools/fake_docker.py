#!/usr/bin/env python
"""A mock ``docker`` CLI, standing in for a container runtime there is none of.

What SweCiAdapter needs from Docker is narrow: list running containers, report
their IP addresses as JSON, and run a command "inside" one. This implements
exactly that subset and nothing else -- so milestone 3's plumbing (IP -> ID
resolution, `docker exec` argv shape, HOME isolation, the guard's reach, the
timeout) is testable on a laptop with no Docker, no root and no images, the same
way ``fake_agent.py`` makes the harness testable with no harness and
``fake_upstream.py`` makes the model layer testable with no GPU.

It is *not* a Docker emulator. A "container" is a directory on the host plus a
JSON file describing it, and ``exec`` runs the command on the host with ``cwd``
set to that directory. That is enough to be faithful about the things the
adapter can get wrong, and it deliberately proves nothing about whether real
containers -- or udocker, which is the actual open question on the cluster --
behave this way.

    export FORESIGHT_FAKE_DOCKER_STATE=/tmp/fd
    fake_docker.py create <name> <ip>     # not a docker verb; test setup
    fake_docker.py ps -q
    fake_docker.py inspect <id> [<id>...]
    fake_docker.py exec [-w DIR] [-e K=V]... <id> <cmd> [args...]

State lives under $FORESIGHT_FAKE_DOCKER_STATE, one directory per container:

    <state>/<id>/meta.json     {"Id": ..., "IPAddress": ..., "running": true}
    <state>/<id>/rootfs/       the container filesystem, so -w resolves

``exec`` maps a container path onto ``rootfs`` by prefixing it, which is what
lets a test point ``adapter.workspace`` at ``/app`` and have the guard's
``find /app ...`` actually find the seeded tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def state_root() -> Path:
    root = os.environ.get("FORESIGHT_FAKE_DOCKER_STATE")
    if not root:
        sys.stderr.write("fake_docker: FORESIGHT_FAKE_DOCKER_STATE is not set\n")
        raise SystemExit(125)
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def containers() -> list[dict]:
    found = []
    for meta in sorted(state_root().glob("*/meta.json")):
        try:
            found.append(json.loads(meta.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return found


def find(ref: str) -> dict | None:
    """By full ID or by the 12-char prefix Docker prints."""
    for container in containers():
        if container["Id"] == ref or container["Id"].startswith(ref):
            return container
    return None


def rootfs(container: dict) -> Path:
    return state_root() / container["Id"] / "rootfs"


def cmd_create(argv: list[str]) -> int:
    """Test-only: bring a fake container into existence.

    ``--not-running`` makes it invisible to ``ps``, which is how a test
    reproduces "the container died between requests".
    """
    positional = [a for a in argv if not a.startswith("-")]
    name = positional[0] if positional else uuid.uuid4().hex
    ip = positional[1] if len(positional) > 1 else ""
    running = "--not-running" not in argv

    container_id = uuid.uuid4().hex + uuid.uuid4().hex[:32]  # 64 hex, like Docker
    home = state_root() / container_id
    (home / "rootfs").mkdir(parents=True)
    (home / "meta.json").write_text(
        json.dumps(
            {"Id": container_id, "Name": name, "IPAddress": ip, "running": running}
        )
    )
    print(container_id)
    return 0


def cmd_ps(argv: list[str]) -> int:
    quiet = "-q" in argv
    for container in containers():
        if not container.get("running", True):
            continue
        print(
            container["Id"]
            if quiet
            else f"{container['Id'][:12]}  {container['Name']}"
        )
    return 0


def cmd_inspect(argv: list[str]) -> int:
    """Docker returns a JSON *array*, and exits 1 if any ref is unknown."""
    records = []
    missing = []
    for ref in [a for a in argv if not a.startswith("-")]:
        container = find(ref)
        if container is None:
            missing.append(ref)
            continue
        records.append(
            {
                "Id": container["Id"],
                "Name": f"/{container['Name']}",
                "State": {"Running": bool(container.get("running", True))},
                "NetworkSettings": {"IPAddress": container.get("IPAddress", "")},
            }
        )
    print(json.dumps(records, indent=2))
    for ref in missing:
        sys.stderr.write(f"Error: No such object: {ref}\n")
    return 1 if missing else 0


def cmd_exec(argv: list[str]) -> int:
    """``exec [-w DIR] [-e K=V]... <id> <cmd>...`` -- run it on the host.

    Flags are parsed in the order the adapter emits them, and the container
    must be running, because "exec into a container that just died" is a real
    mid-run failure and the adapter has to report it as no answer rather than
    as a crash.
    """
    workdir = None
    env_overrides = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "-w":
            workdir = argv[index + 1]
            index += 2
        elif token == "-e":
            key, _, value = argv[index + 1].partition("=")
            env_overrides[key] = value
            index += 2
        elif token in {"-i", "-t", "-it", "-d", "--detach"}:
            index += 1
        elif token == "-u":
            index += 2
        else:
            break

    if index >= len(argv):
        sys.stderr.write("fake_docker: exec needs a container and a command\n")
        return 125
    ref, command = argv[index], argv[index + 1 :]
    container = find(ref)
    if container is None or not container.get("running", True):
        sys.stderr.write(f"Error: No such container: {ref}\n")
        return 1
    if not command:
        sys.stderr.write("fake_docker: exec needs a command\n")
        return 125

    root = rootfs(container)
    cwd = root / (workdir or "/").lstrip("/")
    cwd.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(env_overrides)
    # So an in-"container" command can rewrite absolute paths onto the fake
    # rootfs itself. Real docker needs no such thing.
    env["FORESIGHT_FAKE_DOCKER_ROOTFS"] = str(root)

    command = [_rewrite(element, root) for element in command]
    try:
        completed = subprocess.run(command, cwd=str(cwd), env=env)
    except FileNotFoundError:
        sys.stderr.write(f"exec: {command[0]}: not found\n")
        return 127
    return completed.returncode


def _rewrite(element: str, root: Path) -> str:
    """Map a container-absolute path onto the fake rootfs.

    Only the prefixes the adapter itself passes (``cat /tmp/foresight-aux-...``,
    ``sh -c 'find /app ...'``) are rewritten, so a prompt travels through
    untouched -- which is what keeps the argv-substitution assertions honest.
    """
    for prefix in ("/app", "/tmp/foresight-aux", "/tmp/aux-home"):
        if prefix in element:
            element = element.replace(prefix, str(root) + prefix)
    return element


VERBS = {
    "create": cmd_create,
    "ps": cmd_ps,
    "inspect": cmd_inspect,
    "exec": cmd_exec,
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        sys.stderr.write(f"fake_docker: expected one of {sorted(VERBS)}\n")
        return 125
    verb, rest = argv[0], argv[1:]
    # `docker container inspect` / `docker image inspect` are the forms SWE-CI
    # itself uses, so accept them too and this stays usable beyond the adapter.
    if verb in {"container", "image"} and rest:
        verb, rest = rest[0], rest[1:]
    handler = VERBS.get(verb)
    if handler is None:
        sys.stderr.write(f"fake_docker: unsupported verb {verb!r}\n")
        return 125
    return handler(rest)


if __name__ == "__main__":
    sys.exit(main())
