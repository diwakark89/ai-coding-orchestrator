"""Git worktree manager: isolated, disposable workspaces for AI-driven implementation.

Agents never touch the user's primary working tree. Every implementation run gets its own
worktree under ~/.agentflow/worktrees/<project-name>/<run-id>/ on a dedicated branch
agentflow/<run-id>, created via plain `git worktree` / `git diff` — never `shell=True`.
Only an explicitly approved merge applies a run's single commit to the primary checkout.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agentflow.errors import WorktreeError
from agentflow.process.executor import ProcessExecutor


@dataclass
class WorktreeHandle:
    """A created worktree: its filesystem path, branch, and originating repository.

    base_branch/base_commit record what the primary checkout had at creation, so an
    approved run can later be merged back into the branch it started from.
    """

    path: Path
    branch_name: str
    repository_path: Path
    base_branch: str | None = None
    base_commit: str | None = None


@dataclass
class CherryPickResult:
    """Outcome of applying a run's commit to the primary checkout."""

    success: bool
    commit_sha: str | None = None
    conflicts: list[str] = field(default_factory=list)
    error: str = ""


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

        base_branch = await self.current_branch(repository_path)
        base_commit = await self._rev_parse(repository_path, "HEAD")

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
            path=worktree_path,
            branch_name=branch_name,
            repository_path=Path(repository_path),
            base_branch=base_branch,
            base_commit=base_commit,
        )

    async def _rev_parse(self, cwd: Path, ref: str) -> str | None:
        result = await self.executor.run(["git", "rev-parse", "-q", "--verify", ref], cwd=cwd)
        if result.exit_code != 0:
            return None
        return result.stdout.strip() or None

    async def current_branch(self, repository_path: Path) -> str | None:
        """The checked-out branch's short name, or None when HEAD is detached."""
        result = await self.executor.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repository_path
        )
        if result.exit_code != 0:
            return None
        return result.stdout.strip() or None

    async def unsafe_checkout_reason(self, repository_path: Path) -> str | None:
        """Why the primary checkout must not receive a merge now, or None if it is safe.

        Untracked files are allowed: Git itself refuses a cherry-pick that would
        overwrite one, without modifying anything.
        """
        for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REBASE_HEAD", "REVERT_HEAD"):
            if await self._rev_parse(repository_path, marker):
                return f"a {marker.split('_')[0].lower()} is in progress"
        status = await self.executor.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository_path
        )
        if status.exit_code != 0:
            return f"git status failed: {status.stderr.strip()}"
        if status.stdout.strip():
            return "it has uncommitted changes to tracked files"
        return None

    async def commits_since(self, worktree_path: Path, base_commit: str) -> int:
        """Number of commits on the worktree's branch after base_commit."""
        result = await self.executor.run(
            ["git", "rev-list", "--count", f"{base_commit}..HEAD"], cwd=worktree_path
        )
        if result.exit_code != 0:
            raise WorktreeError(f"Could not inspect worktree history: {result.stderr.strip()}")
        return int(result.stdout.strip() or "0")

    async def commit(
        self, worktree_path: Path, subject: str, body: str, amend: bool = False
    ) -> str:
        """Commit everything staged in a run's worktree; hooks run normally. Returns the SHA.

        amend folds new staged changes into a run commit left by an earlier merge attempt,
        so a run always lands as exactly one commit.
        """
        args = ["git", "commit", *(["--amend"] if amend else []), "-m", subject, "-m", body]
        result = await self.executor.run(args, cwd=worktree_path)
        if result.exit_code != 0:
            raise WorktreeError(
                "Commit in worktree failed: "
                + (result.stderr.strip() or result.stdout.strip())[-2000:]
            )
        sha = await self._rev_parse(worktree_path, "HEAD")
        if sha is None:
            raise WorktreeError("Commit in worktree succeeded but HEAD could not be resolved.")
        return sha

    async def cherry_pick(self, repository_path: Path, commit_sha: str) -> CherryPickResult:
        """Apply one commit onto the primary checkout; on failure, restore it exactly."""
        result = await self.executor.run(["git", "cherry-pick", commit_sha], cwd=repository_path)
        if result.exit_code == 0:
            return CherryPickResult(
                success=True, commit_sha=await self._rev_parse(repository_path, "HEAD")
            )
        conflicts_result = await self.executor.run(
            ["git", "diff", "--name-only", "--diff-filter=U"], cwd=repository_path
        )
        conflicts = [line.strip() for line in conflicts_result.stdout.splitlines() if line.strip()]
        # A refused pick (e.g. an untracked file would be overwritten) leaves no state to abort.
        if await self._rev_parse(repository_path, "CHERRY_PICK_HEAD"):
            await self.executor.run(["git", "cherry-pick", "--abort"], cwd=repository_path)
        return CherryPickResult(
            success=False,
            conflicts=conflicts,
            error=(result.stderr.strip() or result.stdout.strip())[-2000:],
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

        Workflow stages never commit; only an approved merge commits on the branch, and it
        deletes the branch after its commit was applied to the base branch. Callers must not
        delete the branch of a completed-but-unmerged run. Best-effort: a missing branch
        (e.g. already deleted) is not treated as an error.
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
