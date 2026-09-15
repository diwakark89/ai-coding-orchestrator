"""Run-level lock: ensures only one process controls a given run at a time.

Complements `WorktreeLock` (single-writer guard on a worktree's files): `RunLock` guards the
run's *orchestration* -- its state machine and DB records -- so two `agentflow` processes
(for example, an original `implement`/`complete` invocation and a later `resume` of the same
run) can never drive the same run concurrently (phased-implementation-plan.md Phase 10 Step 4).
"""

from pathlib import Path

from agentflow.concurrency.lockfile import LockFile, LockHeldError, LockInfo
from agentflow.errors import RunLockedError

LOCK_SUFFIX = ".lock"


class RunLock:
    """Exclusive, filesystem-based lock guarding orchestration control of a single run."""

    def __init__(self, run_id: str, locks_root: Path) -> None:
        self.run_id = run_id
        self.locks_root = Path(locks_root)
        self._lock_file = LockFile(self.lock_path)

    @property
    def lock_path(self) -> Path:
        """Path to the lock file for this run, under the shared global locks directory."""
        return self.locks_root / f"{self.run_id}{LOCK_SUFFIX}"

    def acquire(self, owner: str = "agentflow", break_stale: bool = True) -> None:
        """Acquire the lock, raising RunLockedError if already held by a live process.

        Unlike `WorktreeLock`, a stale run lock (owning process no longer alive) is broken
        automatically by default: crash recovery is the entire purpose of this lock, so
        refusing to resume a run because of its own dead process's leftover lock file would
        defeat that purpose. A lock held by a live process is never broken.
        """
        try:
            self._lock_file.acquire(owner=owner, break_stale=break_stale)
        except LockHeldError as e:
            raise RunLockedError(
                f"Run '{self.run_id}' is already controlled by another process: {self.lock_path}"
            ) from e

    def release(self) -> None:
        """Release the lock if held; a no-op if it is not."""
        self._lock_file.release()

    def is_locked(self) -> bool:
        """Return True if the lock is currently held."""
        return self._lock_file.is_locked()

    def is_stale(self) -> bool:
        """Return True if the lock is held but its owning process is no longer running."""
        return self._lock_file.is_stale()

    def read_info(self) -> LockInfo | None:
        """Read the lock's owner/pid metadata, or None if unlocked or unreadable."""
        return self._lock_file.read_info()

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()
