"""ManifestGuard: detect that aux touched the workspace.

Runs against real directories and a real shell, because the whole value of this
component is whether the manifest command actually works on the machine it runs
on -- and the design's specified command uses GNU `find -printf`, which BSD find
(macOS) rejects. A mocked runner would prove nothing about that.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from foresight.guards import ManifestGuard, WorkspaceGuard


def local_runner(root: Path):
    def run(command: str) -> tuple[int, str]:
        done = subprocess.run(
            command, shell=True, cwd=str(root), capture_output=True, text=True
        )
        return done.returncode, done.stdout

    return run


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def f():\n    return 1\n")
    (tmp_path / "README.md").write_text("# scratch\n")
    (tmp_path / "with space.txt").write_text("paths may contain spaces\n")
    return tmp_path


def guard_for(root: Path) -> ManifestGuard:
    return ManifestGuard(str(root), local_runner(root))


def test_manifest_guard_satisfies_the_protocol(workspace):
    assert isinstance(guard_for(workspace), WorkspaceGuard)


def test_snapshot_lists_every_file(workspace):
    lines = guard_for(workspace).snapshot()
    assert len(lines) == 3
    # "size mtime path" -- three whitespace-separated fields, path last.
    for line in lines:
        size, mtime, path = line.split(None, 2)
        assert size.isdigit()
        assert float(mtime) > 0
        assert path


def test_untouched_workspace_verifies_clean(workspace):
    guard = guard_for(workspace)
    assert guard.verify(guard.snapshot()) == []


def test_detects_a_new_file(workspace):
    guard = guard_for(workspace)
    before = guard.snapshot()
    (workspace / "pkg" / "sneaky.py").write_text("x = 1\n")

    changed = guard.verify(before)
    assert len(changed) == 1
    assert changed[0].endswith("sneaky.py")


def test_detects_a_deletion(workspace):
    guard = guard_for(workspace)
    before = guard.snapshot()
    (workspace / "README.md").unlink()

    changed = guard.verify(before)
    assert len(changed) == 1
    assert changed[0].endswith("README.md")


def test_detects_a_modification_that_changes_size(workspace):
    guard = guard_for(workspace)
    before = guard.snapshot()
    (workspace / "pkg" / "mod.py").write_text("def f():\n    return 2  # edited\n")

    changed = guard.verify(before)
    assert any(c.endswith("mod.py") for c in changed)


def test_detects_a_same_size_edit_via_mtime(workspace):
    """A one-character swap keeps the size identical; mtime is what catches it."""
    import os
    import time

    target = workspace / "pkg" / "mod.py"
    guard = guard_for(workspace)
    before = guard.snapshot()

    target.write_text("def f():\n    return 9\n")  # same length as "return 1"
    stamp = time.time() + 5
    os.utime(target, (stamp, stamp))

    assert any(c.endswith("mod.py") for c in guard.verify(before))


def test_names_every_changed_path(workspace):
    """Naming them is the reason a manifest is kept instead of one digest."""
    guard = guard_for(workspace)
    before = guard.snapshot()
    (workspace / "a.py").write_text("1\n")
    (workspace / "b.py").write_text("2\n")
    (workspace / "README.md").unlink()

    changed = guard.verify(before)
    assert len(changed) == 3
    assert sorted(Path(c).name for c in changed) == ["README.md", "a.py", "b.py"]


def test_paths_with_spaces_survive_the_split(workspace):
    """_path_of splits twice from the left, so a spaced filename stays whole."""
    guard = guard_for(workspace)
    before = guard.snapshot()
    (workspace / "with space.txt").write_text("edited, and longer than before\n")

    changed = guard.verify(before)
    assert any(c.endswith("with space.txt") for c in changed)


def test_an_unusable_manifest_raises_rather_than_reading_as_intact(tmp_path):
    """A broken guard must never be mistaken for a clean workspace."""
    guard = ManifestGuard(str(tmp_path / "does-not-exist"), local_runner(tmp_path))
    with pytest.raises(RuntimeError, match="manifest failed"):
        guard.snapshot()
