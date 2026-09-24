"""Removal of AgentFlow-owned folders, including ones a sandboxed agent left locked.

Coding-agent sandboxes (e.g. Codex on Windows) can create files owned by a separate sandbox
account that the user cannot even list. Such folders can only be taken over from an elevated
(Administrator) process. Ownership is taken only for the specific folders that refused
deletion -- never recursively over a whole worktree, whose node_modules junctions could
point outside it -- and only inside directories AgentFlow manages.
"""

import os
import shutil
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentflow.process.executor import ProcessExecutor

_ADMINISTRATORS_SID = "*S-1-5-32-544"


@dataclass
class RemovalResult:
    """Outcome of removing one folder."""

    removed: bool
    needs_admin: bool = False
    error: str = ""


def is_elevated() -> bool:
    """True when running as Administrator (Windows) or root (POSIX)."""
    if os.name == "nt":
        try:
            import ctypes

            windll: Any = getattr(ctypes, "windll")  # absent from non-Windows type stubs
            return bool(windll.shell32.IsUserAnAdmin())
        except (AttributeError, OSError):
            return False
    geteuid: Callable[[], int] | None = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


def _is_managed(path: Path, allowed_roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(
        resolved != root.resolve() and resolved.is_relative_to(root.resolve())
        for root in allowed_roots
    )


def _rmtree_collecting_failures(path: Path) -> list[Path]:
    """Delete what we can; clear read-only flags and retry once; return what still refused."""
    failures: list[Path] = []

    def on_error(func: Callable[..., Any], failed: str, exc: BaseException) -> None:
        try:
            os.chmod(failed, stat.S_IWRITE)
            func(failed)
        except OSError:
            failures.append(Path(failed))

    shutil.rmtree(path, onexc=on_error)
    # Ancestors of a locked item fail only because they are not empty; keep just the
    # innermost failures, so ownership is never taken over a whole tree.
    unique = set(failures)
    return sorted(
        failed
        for failed in unique
        if not any(other != failed and other.is_relative_to(failed) for other in unique)
    )


async def remove_tree(
    path: Path, allowed_roots: Sequence[Path], executor: ProcessExecutor
) -> RemovalResult:
    """Remove an AgentFlow-managed folder, taking over sandbox-locked parts when elevated."""
    if not _is_managed(path, allowed_roots):
        raise ValueError(f"Refusing to remove {path}: it is outside AgentFlow-managed folders.")
    if not os.path.lexists(path):
        return RemovalResult(removed=True)

    locked = _rmtree_collecting_failures(path)
    if not os.path.lexists(path):
        return RemovalResult(removed=True)

    elevated = is_elevated()
    if os.name == "nt" and elevated:
        for folder in locked:
            if not _is_managed(folder, allowed_roots) or folder.is_symlink():
                continue
            target = str(folder)
            # takeown answers its per-folder prompt with /d Y; icacls covers locales where
            # that answer letter differs. Failures are judged by the final existence check.
            await executor.run(["takeown", "/f", target, "/a", "/r", "/d", "Y"])
            await executor.run(["icacls", target, "/setowner", _ADMINISTRATORS_SID, "/t", "/c"])
            await executor.run(["icacls", target, "/grant", f"{_ADMINISTRATORS_SID}:F", "/t", "/c"])
        _rmtree_collecting_failures(path)
        if not os.path.lexists(path):
            return RemovalResult(removed=True)

    return RemovalResult(
        removed=False,
        needs_admin=os.name == "nt" and not elevated,
        error=f"{len(locked)} locked item(s), e.g. {locked[0]}" if locked else "not removed",
    )
