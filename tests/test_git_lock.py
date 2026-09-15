"""Unit tests for the single-writer worktree lock."""

from pathlib import Path

import pytest

from agentflow.errors import WorktreeLockedError
from agentflow.git.lock import WorktreeLock


def test_acquire_and_release(tmp_path: Path):
    """Acquiring then releasing a lock leaves no lock file behind."""
    lock = WorktreeLock(tmp_path)
    lock.acquire()
    assert lock.is_locked()
    lock.release()
    assert not lock.is_locked()


def test_double_acquire_raises(tmp_path: Path):
    """A second writer cannot acquire a lock already held by another."""
    lock1 = WorktreeLock(tmp_path)
    lock1.acquire()
    lock2 = WorktreeLock(tmp_path)
    with pytest.raises(WorktreeLockedError):
        lock2.acquire()
    lock1.release()


def test_release_when_not_locked_is_noop(tmp_path: Path):
    """Releasing a lock that was never acquired does not raise."""
    lock = WorktreeLock(tmp_path)
    lock.release()


def test_context_manager_acquires_and_releases(tmp_path: Path):
    """Using WorktreeLock as a context manager acquires on enter and releases on exit."""
    lock = WorktreeLock(tmp_path)
    with lock:
        assert lock.is_locked()
    assert not lock.is_locked()


def test_context_manager_releases_on_exception(tmp_path: Path):
    """The lock is released even if the protected block raises."""
    lock = WorktreeLock(tmp_path)
    with pytest.raises(ValueError, match="boom"), lock:
        raise ValueError("boom")
    assert not lock.is_locked()


def test_lock_creates_parent_directory_but_not_the_worktree_itself(tmp_path: Path):
    """Acquiring a lock ensures its parent exists, without creating the worktree directory.

    The lock file is a sibling of the worktree (never inside it), so it must not interfere
    with `git add -A` / `git diff` run inside a not-yet-created or freshly-created worktree.
    """
    target = tmp_path / "nested" / "worktree"
    lock = WorktreeLock(target)
    lock.acquire()
    assert lock.lock_path.parent.exists()
    assert not target.exists()
    lock.release()


def test_after_release_a_new_writer_can_acquire(tmp_path: Path):
    """Once released, a different WorktreeLock instance can acquire the same path."""
    lock1 = WorktreeLock(tmp_path)
    lock1.acquire()
    lock1.release()

    lock2 = WorktreeLock(tmp_path)
    lock2.acquire()
    assert lock2.is_locked()
    lock2.release()
