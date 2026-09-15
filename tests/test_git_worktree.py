"""Fixture-repo tests for the Git worktree manager (Phase 5)."""

import subprocess
from pathlib import Path

import pytest

from agentflow.errors import WorktreeError
from agentflow.git.worktree import WorktreeManager


@pytest.mark.asyncio
async def test_create_worktree_and_branch(git_repo: Path, tmp_path: Path):
    """create() creates the dedicated agentflow/<run-id> branch and worktree directory."""
    manager = WorktreeManager(tmp_path / "worktrees")
    handle = await manager.create("demo-project", "RUN-001", git_repo)

    assert handle.path.exists()
    assert handle.branch_name == "agentflow/RUN-001"
    assert await manager.branch_exists(git_repo, "agentflow/RUN-001")


@pytest.mark.asyncio
async def test_worktree_path_follows_naming_convention(git_repo: Path, tmp_path: Path):
    """Worktree path follows <root>/<project-name>/<run-id>/."""
    root = tmp_path / "worktrees"
    manager = WorktreeManager(root)
    handle = await manager.create("demo-project", "RUN-002", git_repo)
    assert handle.path == root / "demo-project" / "RUN-002"


@pytest.mark.asyncio
async def test_primary_working_tree_unchanged(git_repo: Path, tmp_path: Path):
    """Creating a worktree and writing inside it never modifies the primary working tree."""
    manager = WorktreeManager(tmp_path / "worktrees")
    handle = await manager.create("demo-project", "RUN-003", git_repo)

    (handle.path / "new_file.txt").write_text("hello", encoding="utf-8")

    assert not (git_repo / "new_file.txt").exists()
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=git_repo, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


@pytest.mark.asyncio
async def test_capture_changes_detects_new_and_modified_files(git_repo: Path, tmp_path: Path):
    """capture_changes() reports both new and modified files with a non-empty diff."""
    manager = WorktreeManager(tmp_path / "worktrees")
    handle = await manager.create("demo-project", "RUN-004", git_repo)

    (handle.path / "new_file.txt").write_text("hello\n", encoding="utf-8")
    (handle.path / "README.md").write_text("# Updated\n", encoding="utf-8")

    diff = await manager.capture_changes(handle.path)

    assert not diff.is_empty
    assert "new_file.txt" in diff.changed_files
    assert "README.md" in diff.changed_files
    assert "new_file.txt" in diff.diff_text


@pytest.mark.asyncio
async def test_capture_changes_empty_when_nothing_changed(git_repo: Path, tmp_path: Path):
    """capture_changes() reports an empty diff when nothing was modified."""
    manager = WorktreeManager(tmp_path / "worktrees")
    handle = await manager.create("demo-project", "RUN-005", git_repo)

    diff = await manager.capture_changes(handle.path)
    assert diff.is_empty
    assert diff.changed_files == []


@pytest.mark.asyncio
async def test_duplicate_branch_conflict_raises(git_repo: Path, tmp_path: Path):
    """create() refuses to create a worktree if the dedicated branch already exists."""
    manager = WorktreeManager(tmp_path / "worktrees")
    await manager.create("demo-project", "RUN-006", git_repo)

    with pytest.raises(WorktreeError, match="already exists"):
        await manager.create("demo-project", "RUN-006", git_repo)


@pytest.mark.asyncio
async def test_duplicate_worktree_path_raises(git_repo: Path, tmp_path: Path):
    """create() refuses to create a worktree at a path that already exists."""
    manager = WorktreeManager(tmp_path / "worktrees")
    worktree_path = manager.worktree_path_for("demo-project", "RUN-007")
    worktree_path.mkdir(parents=True)

    with pytest.raises(WorktreeError, match="already exists"):
        await manager.create("demo-project", "RUN-007", git_repo)


@pytest.mark.asyncio
async def test_create_fails_for_invalid_branch_name(git_repo: Path, tmp_path: Path):
    """A run_id that produces an invalid Git ref name surfaces as a WorktreeError."""
    manager = WorktreeManager(tmp_path / "worktrees")
    with pytest.raises(WorktreeError, match="Failed to create worktree"):
        await manager.create("demo-project", "RUN..008", git_repo)


@pytest.mark.asyncio
async def test_cleanup_removes_partial_worktree_directory(git_repo: Path, tmp_path: Path):
    """_cleanup_failed_creation removes a partially-created worktree directory."""
    manager = WorktreeManager(tmp_path / "worktrees")
    partial_path = manager.worktree_path_for("demo-project", "RUN-CLEANUP")
    partial_path.mkdir(parents=True)
    (partial_path / "partial.txt").write_text("leftover", encoding="utf-8")

    await manager._cleanup_failed_creation(git_repo, partial_path)

    assert not partial_path.exists()


@pytest.mark.asyncio
async def test_remove_worktree(git_repo: Path, tmp_path: Path):
    """remove() deletes the worktree directory and unregisters it from Git."""
    manager = WorktreeManager(tmp_path / "worktrees")
    handle = await manager.create("demo-project", "RUN-009", git_repo)
    assert handle.path.exists()

    await manager.remove(handle)

    assert not handle.path.exists()
