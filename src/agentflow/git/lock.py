"""File-based single-writer lock for a worktree.

Only one AI agent may modify a worktree at any given moment (AGENTS.md invariant #5).

The lock file is deliberately kept OUTSIDE the worktree directory (as a sibling file), never
inside it: a lock file living inside the worktree would show up as an untracked change in
`git add -A` / `git diff`, corrupting no-op detection and the captured implementation diff.
"""

from pathlib import Path

from agentflow.concurrency.lockfile import LockFile, LockHeldError, LockInfo
from agentflow.errors import WorktreeLockedError

LOCK_SUFFIX = ".agentflow-writer.lock"


class WorktreeLock:
    """Exclusive, filesystem-based lock guarding writer access to a single worktree."""

    def __init__(self, worktree_path: Path) -> None:
        self.worktree_path = Path(worktree_path)
        self._lock_file = LockFile(self.lock_path)

    @property
    def lock_path(self) -> Path:
        """Path to the lock file, a sibling of the worktree directory (never inside it)."""
        return self.worktree_path.parent / f"{self.worktree_path.name}{LOCK_SUFFIX}"

    def acquire(self, owner: str = "agentflow", break_stale: bool = False) -> None:
        """Acquire the lock, raising WorktreeLockedError if already held by a live process.

        `break_stale` is False by default: a worktree lock is broken automatically only when
        the caller has explicitly determined (via `is_stale()`) that recovery is appropriate,
        never as a silent side effect of a normal acquire.
        """
        try:
            self._lock_file.acquire(owner=owner, break_stale=break_stale)
        except LockHeldError as e:
            raise WorktreeLockedError(
                f"Worktree is already locked by another writer: {self.lock_path}"
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

    def __enter__(self) -> "WorktreeLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()
