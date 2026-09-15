"""Unit tests for the generic stale-aware file lock underlying WorktreeLock and RunLock."""

import os
from pathlib import Path

import pytest

from agentflow.concurrency.lockfile import LockFile, LockHeldError, is_process_alive

DEAD_PID = 999999


def test_acquire_and_release(tmp_path: Path):
    """Acquiring then releasing a lock leaves no lock file behind."""
    lock = LockFile(tmp_path / "a.lock")
    lock.acquire()
    assert lock.is_locked()
    lock.release()
    assert not lock.is_locked()


def test_double_acquire_by_live_process_raises(tmp_path: Path):
    """A second holder cannot acquire a lock already held by a live process."""
    path = tmp_path / "a.lock"
    lock1 = LockFile(path)
    lock1.acquire()
    lock2 = LockFile(path)
    with pytest.raises(LockHeldError):
        lock2.acquire()
    lock1.release()


def test_release_when_not_locked_is_noop(tmp_path: Path):
    """Releasing a lock that was never acquired does not raise."""
    LockFile(tmp_path / "a.lock").release()


def test_non_owner_release_never_removes_an_existing_lock(tmp_path: Path):
    """A second lock object cannot accidentally release the live owner's lock file."""
    path = tmp_path / "a.lock"
    owner = LockFile(path)
    owner.acquire()
    LockFile(path).release()
    assert owner.is_locked()
    owner.release()


def test_acquire_creates_parent_directory(tmp_path: Path):
    """Acquiring creates the lock file's parent directory if it does not yet exist."""
    path = tmp_path / "nested" / "dir" / "a.lock"
    lock = LockFile(path)
    lock.acquire()
    assert path.parent.exists()
    lock.release()


def test_read_info_roundtrip(tmp_path: Path):
    """Owner and pid recorded at acquire time can be read back."""
    lock = LockFile(tmp_path / "a.lock")
    lock.acquire(owner="tester")
    info = lock.read_info()
    assert info is not None
    assert info.owner == "tester"
    assert info.pid == os.getpid()
    lock.release()


def test_read_info_when_unlocked_is_none(tmp_path: Path):
    """Reading lock info for a path with no lock file returns None."""
    assert LockFile(tmp_path / "a.lock").read_info() is None


def test_read_info_ignores_malformed_lock_file(tmp_path: Path):
    """A corrupted lock file is treated as unreadable rather than raising."""
    path = tmp_path / "a.lock"
    path.write_text("not json", encoding="utf-8")
    assert LockFile(path).read_info() is None


def test_is_stale_false_for_live_owner(tmp_path: Path):
    """A lock held by the current (live) process is never considered stale."""
    lock = LockFile(tmp_path / "a.lock")
    lock.acquire()
    assert not lock.is_stale()
    lock.release()


def test_is_stale_false_when_unlocked(tmp_path: Path):
    """An unlocked path is not stale -- staleness only applies to a held lock."""
    assert not LockFile(tmp_path / "a.lock").is_stale()


def test_is_stale_true_for_dead_pid(tmp_path: Path):
    """A lock file naming a pid that is not running is detected as stale."""
    path = tmp_path / "a.lock"
    path.write_text(
        f'{{"owner": "ghost", "pid": {DEAD_PID}, "acquired_at": "x"}}', encoding="utf-8"
    )
    assert LockFile(path).is_stale()


def test_acquire_break_stale_reclaims_a_dead_owners_lock(tmp_path: Path):
    """`break_stale=True` clears a stale lock and reacquires it for the current process."""
    path = tmp_path / "a.lock"
    path.write_text(
        f'{{"owner": "ghost", "pid": {DEAD_PID}, "acquired_at": "x"}}', encoding="utf-8"
    )
    lock = LockFile(path)
    lock.acquire(break_stale=True)
    assert lock.is_locked()
    info = lock.read_info()
    assert info is not None and info.pid == os.getpid()
    lock.release()


def test_acquire_break_stale_never_breaks_a_live_owners_lock(tmp_path: Path):
    """`break_stale=True` still refuses to acquire when the existing owner is alive."""
    path = tmp_path / "a.lock"
    lock1 = LockFile(path)
    lock1.acquire()
    lock2 = LockFile(path)
    with pytest.raises(LockHeldError):
        lock2.acquire(break_stale=True)
    lock1.release()


def test_is_process_alive_true_for_current_process():
    """The running test process is, definitionally, alive."""
    assert is_process_alive(os.getpid())


def test_is_process_alive_false_for_nonexistent_pid():
    """A pid nothing is using is reported as not alive."""
    assert not is_process_alive(DEAD_PID)


def test_is_process_alive_false_for_non_positive_pid():
    """Pid 0 and negative pids are never treated as alive."""
    assert not is_process_alive(0)
    assert not is_process_alive(-1)
