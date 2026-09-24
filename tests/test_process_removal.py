"""Removal of AgentFlow-owned folders, including sandbox-locked leftovers."""

import os
import shutil
import stat
from pathlib import Path

import pytest

from agentflow.process import removal
from agentflow.process.executor import ProcessExecutor
from agentflow.process.removal import remove_tree


@pytest.mark.asyncio
async def test_refuses_paths_outside_managed_roots(tmp_path: Path):
    managed = tmp_path / "worktrees"
    outside = tmp_path / "project"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside AgentFlow-managed"):
        await remove_tree(outside, [managed], ProcessExecutor())
    with pytest.raises(ValueError):
        await remove_tree(managed, [managed], ProcessExecutor())  # never the root itself
    assert outside.exists()


@pytest.mark.asyncio
async def test_removes_tree_with_read_only_files(tmp_path: Path):
    root = tmp_path / "worktrees"
    target = root / "demo" / "RUN-1.tmp" / "pytest-of-user"
    target.mkdir(parents=True)
    locked_file = target / "cache.bin"
    locked_file.write_text("x", encoding="utf-8")
    os.chmod(locked_file, stat.S_IREAD)

    result = await remove_tree(root / "demo" / "RUN-1.tmp", [root], ProcessExecutor())

    assert result.removed
    assert not (root / "demo" / "RUN-1.tmp").exists()


@pytest.mark.asyncio
async def test_missing_path_counts_as_removed(tmp_path: Path):
    root = tmp_path / "worktrees"
    assert (await remove_tree(root / "gone", [root], ProcessExecutor())).removed


@pytest.mark.asyncio
async def test_locked_folder_needs_admin_and_only_innermost_is_taken_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "worktrees"
    worktree = root / "demo" / "RUN-1"
    locked = worktree / "ai-engine" / ".pytest-repair-temp"
    locked.mkdir(parents=True)

    def denied(path: str) -> None:
        raise PermissionError(path)

    def fake_rmtree(path, onexc):
        # A locked dir fails, and so do its ancestors because they are not empty.
        for failed in (locked, locked.parent, worktree):
            onexc(denied, str(failed), PermissionError("denied"))

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(removal, "is_elevated", lambda: False)

    assert removal._rmtree_collecting_failures(worktree) == [locked]
    result = await remove_tree(worktree, [root], ProcessExecutor())

    assert not result.removed
    assert result.needs_admin == (os.name == "nt")
    assert str(locked) in result.error
