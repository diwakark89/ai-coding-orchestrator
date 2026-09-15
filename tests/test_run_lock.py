"""Unit tests for RunLock: single-controller guard preventing a run from being driven by two
processes at once (phased-implementation-plan.md Phase 10 Step 4)."""

import os
from pathlib import Path

import pytest

from agentflow.concurrency.run_lock import RunLock
from agentflow.errors import RunLockedError

DEAD_PID = 999999


def test_acquire_and_release(tmp_path: Path):
    """Acquiring then releasing a run lock leaves no lock file behind."""
    lock = RunLock("RUN-1", tmp_path)
    lock.acquire()
    assert lock.is_locked()
    lock.release()
    assert not lock.is_locked()


def test_double_acquire_by_live_process_raises(tmp_path: Path):
    """The same run cannot be controlled by two live processes at once."""
    lock1 = RunLock("RUN-1", tmp_path)
    lock1.acquire()
    lock2 = RunLock("RUN-1", tmp_path)
    with pytest.raises(RunLockedError):
        lock2.acquire()
    lock1.release()


def test_stale_lock_is_broken_automatically_by_default(tmp_path: Path):
    """A run lock left by a dead process is reclaimed automatically -- crash recovery is the
    entire point of RunLock, so a stale leftover must never block `agentflow resume`."""
    lock_path = tmp_path / "RUN-1.lock"
    lock_path.write_text(
        f'{{"owner": "ghost", "pid": {DEAD_PID}, "acquired_at": "x"}}', encoding="utf-8"
    )
    lock = RunLock("RUN-1", tmp_path)
    lock.acquire()
    assert lock.is_locked()
    info = lock.read_info()
    assert info is not None and info.pid == os.getpid()
    lock.release()


def test_break_stale_can_be_disabled(tmp_path: Path):
    """Callers that want strict behavior can opt out of automatic stale-lock recovery."""
    lock_path = tmp_path / "RUN-1.lock"
    lock_path.write_text(
        f'{{"owner": "ghost", "pid": {DEAD_PID}, "acquired_at": "x"}}', encoding="utf-8"
    )
    lock = RunLock("RUN-1", tmp_path)
    with pytest.raises(RunLockedError):
        lock.acquire(break_stale=False)


def test_a_live_owners_lock_is_never_broken(tmp_path: Path):
    """Even with break_stale defaulted on, a live owner's lock is refused, not broken."""
    lock1 = RunLock("RUN-1", tmp_path)
    lock1.acquire()
    lock2 = RunLock("RUN-1", tmp_path)
    with pytest.raises(RunLockedError):
        lock2.acquire()
    assert lock1.is_locked()
    lock1.release()


def test_different_run_ids_do_not_collide(tmp_path: Path):
    """Locks are scoped per run_id: two different runs may be controlled simultaneously."""
    lock_a = RunLock("RUN-A", tmp_path)
    lock_b = RunLock("RUN-B", tmp_path)
    lock_a.acquire()
    lock_b.acquire()
    assert lock_a.is_locked()
    assert lock_b.is_locked()
    lock_a.release()
    lock_b.release()
