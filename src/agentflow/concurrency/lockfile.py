"""Generic file-based lock with owner/pid metadata and stale-owner detection.

Both `WorktreeLock` (single-writer worktree guard) and `RunLock` (single-controller run guard)
build on this: a lock file storing `{owner, pid, acquired_at}` as JSON, created atomically via
`O_CREAT | O_EXCL`. Staleness is determined by checking whether the recorded pid is still
running -- a lock is never silently deleted just because it exists; only a dead owner's lock
may be treated as stale (AGENTS.md invariant #5 / phased-implementation-plan.md Phase 10 Step 7).
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LockInfo:
    """Metadata recorded inside a lock file."""

    owner: str
    pid: int
    acquired_at: str


class LockHeldError(Exception):
    """Raised when a lock is already held by a live process."""


def is_process_alive(pid: int) -> bool:
    """Best-effort, cross-platform check for whether a process ID is currently running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user -- still alive.
        return True
    return True


class LockFile:
    """Exclusive, filesystem-based lock with owner/pid metadata and staleness detection."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._owned_payload: str | None = None

    def acquire(self, owner: str = "agentflow", break_stale: bool = False) -> None:
        """Acquire the lock, raising LockHeldError if already held by a live process.

        If `break_stale` is True and an existing lock's owning process is no longer alive,
        the stale lock is removed first and acquisition proceeds. A live owner's lock is
        never broken automatically, regardless of `break_stale`.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if break_stale and self.is_stale():
            self._force_remove()
        payload = json.dumps(
            {
                "owner": owner,
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise LockHeldError(f"Lock already held: {self.path}") from e
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        self._owned_payload = payload

    def release(self) -> None:
        """Release this instance's lock without removing another owner's lock."""
        payload = self._owned_payload
        self._owned_payload = None
        if payload is None:
            return
        try:
            if self.path.read_text(encoding="utf-8") == payload:
                self.path.unlink()
        except FileNotFoundError:
            pass

    def is_locked(self) -> bool:
        """Return True if the lock file currently exists."""
        return self.path.exists()

    def read_info(self) -> LockInfo | None:
        """Read the lock file's owner/pid metadata, or None if unlocked or unreadable."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return LockInfo(
                owner=data["owner"], pid=int(data["pid"]), acquired_at=data["acquired_at"]
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def is_stale(self) -> bool:
        """Return True if the lock is held but its owning process is no longer running."""
        info = self.read_info()
        if info is None:
            return False
        return not is_process_alive(info.pid)

    def _force_remove(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "LockFile":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()
