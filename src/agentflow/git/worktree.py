"""Git worktree manager: isolated, disposable workspaces for AI-driven implementation.

The user's primary working tree is never touched. Every implementation run gets its own
worktree under ~/.agentflow/worktrees/<project-name>/<run-id>/ on a dedicated branch
agentflow/<run-id>, created via plain `git worktree` / `git diff` — never `shell=True`.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agentflow.errors import WorktreeError
from agentflow.process.executor import ProcessExecutor


@dataclass
class WorktreeHandle:
    """A created worktree: its filesystem path, branch, and originating repository."""

    path: Path
    branch_name: str
    repository_path: Path


@dataclass
class WorktreeDiff:
    """The full staged changeset (tracked + untracked) captured from a worktree."""

    diff_text: str
    changed_files: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Return True if no files were changed."""
        return not self.changed_files


class WorktreeManager:
    """Creates, inspects, and removes isolated Git worktrees under a shared root."""

    def __init__(self, worktrees_root: Path | str, executor: ProcessExecutor | None = None) -> None:
        self.worktrees_root = Path(worktrees_root).expanduser()
        self.executor = executor or ProcessExecutor()

    def branch_name_for(self, run_id: str) -> str:
        """The dedicated branch name for a run: agentflow/<run-id>."""
        return f"agentflow/{run_id}"

    def worktree_path_for(self, project_name: str, run_id: str) -> Path:
        """The worktree path for a project+run: <root>/<project-name>/<run-id>/."""
        return self.worktrees_root / project_name / run_id

    async def branch_exists(self, repository_path: Path, branch_name: str) -> bool:
        """Return True if branch_name already exists in the primary repository."""
        result = await self.executor.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
            cwd=repository_path,
        )
        return result.exit_code == 0

    async def create(self, project_name: str, run_id: str, repository_path: Path) -> WorktreeHandle:
        """Create a new worktree + dedicated branch for a run. Never touches the primary tree."""
        branch_name = self.branch_name_for(run_id)
        worktree_path = self.worktree_path_for(project_name, run_id)

        if await self.branch_exists(repository_path, branch_name):
            raise WorktreeError(
                f"Branch '{branch_name}' already exists; refusing to create a duplicate worktree."
            )
        if worktree_path.exists():
            raise WorktreeError(f"Worktree path already exists: {worktree_path}")

        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        result = await self.executor.run(
            ["git", "worktree", "add", str(worktree_path), "-b", branch_name],
            cwd=repository_path,
        )
        if result.exit_code != 0:
            await self._cleanup_failed_creation(repository_path, worktree_path)
            raise WorktreeError(
                f"Failed to create worktree at {worktree_path}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        return WorktreeHandle(
            path=worktree_path, branch_name=branch_name, repository_path=Path(repository_path)
        )

    async def _cleanup_failed_creation(self, repository_path: Path, worktree_path: Path) -> None:
        """Best-effort cleanup of a partially-created worktree, never touching the primary tree."""
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        try:
            await self.executor.run(["git", "worktree", "prune"], cwd=repository_path)
        except Exception:
            pass

    async def remove(self, handle: WorktreeHandle, force: bool = True) -> None:
        """Remove a worktree (the disposable copy only — never the primary working tree)."""
        args = ["git", "worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(handle.path))
        result = await self.executor.run(args, cwd=handle.repository_path)
        if result.exit_code != 0:
            raise WorktreeError(f"Failed to remove worktree {handle.path}: {result.stderr.strip()}")

    async def delete_branch(self, repository_path: Path, branch_name: str) -> None:
        """Force-delete a run's dedicated branch after its worktree has been removed.

        Safe by construction: implementation/repair/review/documentation stages only ever
        `git add -A` inside the worktree to capture a diff (never `git commit`), so an
        agentflow/<run-id> branch never carries commits that could be lost. Best-effort: a
        missing branch (e.g. already deleted) is not treated as an error.
        """
        await self.executor.run(["git", "branch", "-D", branch_name], cwd=repository_path)

    async def capture_changes(self, worktree_path: Path) -> WorktreeDiff:
        """Stage and capture the full changeset (tracked + untracked) inside a worktree."""
        await self.executor.run(["git", "add", "-A"], cwd=worktree_path)
        diff_result = await self.executor.run(["git", "diff", "--cached"], cwd=worktree_path)
        files_result = await self.executor.run(
            ["git", "diff", "--cached", "--name-only"], cwd=worktree_path
        )
        changed_files = [line.strip() for line in files_result.stdout.splitlines() if line.strip()]
        return WorktreeDiff(diff_text=diff_result.stdout, changed_files=changed_files)
