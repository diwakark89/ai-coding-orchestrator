"""Merge an approved run back into the branch it started from.

Runs only after explicit human approval at the final gate (or `agentflow merge`). The run's
changes are committed once inside its worktree (hooks run normally) and that single commit
is cherry-picked onto the primary checkout, which must already be on the run's base branch
with no tracked changes. A failed pick is aborted, leaving the primary checkout exactly as
it was and the worktree intact for a retry. Nothing is ever pushed.
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agentflow.errors import WorktreeError
from agentflow.git.worktree import WorktreeHandle, WorktreeManager
from agentflow.persistence.database import DatabaseManager

_SUBJECT_LIMIT = 72


@dataclass
class MergePreflight:
    """Whether an approved run can be merged right now, and what it would look like."""

    ok: bool
    reason: str = ""
    base_branch: str | None = None
    subject: str = ""


@dataclass
class MergeOutcome:
    """Result of an attempted merge."""

    merged: bool
    base_branch: str | None = None
    commit_sha: str | None = None
    reason: str = ""
    conflicts: list[str] = field(default_factory=list)
    cleanup_warning: str | None = None


def latest_decision(db_manager: DatabaseManager, run_id: str, question: str) -> str | None:
    """The most recent recorded answer to `question` for a run, if any."""
    answers = [d.answer for d in db_manager.list_decisions(run_id) if d.question == question]
    return answers[-1] if answers else None


def commit_subject(plan_markdown: str, task_description: str) -> str:
    """First sentence of the plan's Objective (markdown stripped), else the task's first line."""
    candidate = ""
    match = re.search(
        r"^#+\s*Objective\s*\n+(.+?)(?:\n\s*\n|\n#|\Z)",
        plan_markdown,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if match:
        candidate = " ".join(match.group(1).split())
        candidate = re.split(r"(?<=[.!?])\s", candidate, maxsplit=1)[0]
    if not candidate:
        lines = [line.strip() for line in task_description.splitlines() if line.strip()]
        candidate = lines[0] if lines else "AgentFlow change"
    candidate = re.sub(r"[*_`]+", "", candidate).lstrip("#-> ").rstrip(" .")
    if len(candidate) > _SUBJECT_LIMIT:
        candidate = candidate[:_SUBJECT_LIMIT].rsplit(" ", 1)[0].rstrip(" ,;:")
    return candidate or "AgentFlow change"


def commit_body(run_id: str, task_description: str) -> str:
    return f"AgentFlow run {run_id}\n\nTask: {task_description.strip()}"


async def preflight_merge(
    db_manager: DatabaseManager,
    manager: WorktreeManager,
    run_id: str,
    repository_path: Path,
    worktree_path: Path,
    subject: str,
) -> MergePreflight:
    """Check that a merge would be safe. Never modifies the primary checkout; it only
    stages the worktree's changes, as every stage's diff capture already does."""
    base = latest_decision(db_manager, run_id, "base_branch")

    def refuse(reason: str) -> MergePreflight:
        return MergePreflight(ok=False, reason=reason, base_branch=base, subject=subject)

    if latest_decision(db_manager, run_id, "merged_commit"):
        return refuse("this run was already merged")
    if base is None:
        return refuse(
            "the branch this run started from was not recorded (detached HEAD, or the run "
            "predates automatic merging)"
        )
    if not worktree_path.exists():
        return refuse(f"its worktree no longer exists: {worktree_path}")
    current = await manager.current_branch(repository_path)
    if current != base:
        return refuse(
            f"your checkout is on {current or 'a detached HEAD'!r}, not {base!r}; "
            f"switch to {base!r} and run `agentflow merge {run_id}`"
        )
    unsafe = await manager.unsafe_checkout_reason(repository_path)
    if unsafe is not None:
        return refuse(f"your checkout is not safe to merge into: {unsafe}")
    diff = await manager.capture_changes(worktree_path)
    base_commit = latest_decision(db_manager, run_id, "base_commit")
    prior = await manager.commits_since(worktree_path, base_commit) if base_commit else 0
    if diff.is_empty and prior == 0:
        return refuse("the run has no changes to merge")
    if prior > 1:
        return refuse(f"the run branch has {prior} commits; expected at most one")
    return MergePreflight(ok=True, base_branch=base, subject=subject)


async def merge_run(
    db_manager: DatabaseManager,
    manager: WorktreeManager,
    run_id: str,
    repository_path: Path,
    worktree_path: Path,
    subject: str,
    body: str,
) -> MergeOutcome:
    """Squash the run into one commit on its base branch, then remove its worktree/branch."""
    preflight = await preflight_merge(
        db_manager, manager, run_id, repository_path, worktree_path, subject
    )
    if not preflight.ok:
        return MergeOutcome(
            merged=False, base_branch=preflight.base_branch, reason=preflight.reason
        )
    base = preflight.base_branch

    base_commit = latest_decision(db_manager, run_id, "base_commit")
    amend = bool(base_commit) and await manager.commits_since(worktree_path, str(base_commit)) > 0
    try:
        run_commit = await manager.commit(worktree_path, subject, body, amend=amend)
    except WorktreeError as e:
        return MergeOutcome(merged=False, base_branch=base, reason=str(e))

    picked = await manager.cherry_pick(repository_path, run_commit)
    if not picked.success:
        reason = (
            f"conflicts with {base!r} in: {', '.join(picked.conflicts)}"
            if picked.conflicts
            else f"git could not apply the run to {base!r}: {picked.error}"
        )
        return MergeOutcome(
            merged=False, base_branch=base, reason=reason, conflicts=picked.conflicts
        )

    merged_sha = picked.commit_sha or run_commit
    db_manager.record_decision(str(uuid.uuid4()), run_id, "merged_commit", merged_sha)
    db_manager.record_event(run_id=run_id, stage="COMPLETED", event="RUN_MERGED")

    branch_name = manager.branch_name_for(run_id)
    warning: str | None = None
    try:
        await manager.remove(
            WorktreeHandle(worktree_path, branch_name, repository_path), force=True
        )
    except WorktreeError as e:
        warning = str(e)
    await manager.delete_branch(repository_path, branch_name)
    if worktree_path.exists():
        warning = (
            f"Merged, but the worktree folder could not be fully deleted: {worktree_path}. "
            "Remove it manually." + (f" ({warning})" if warning else "")
        )
    elif await manager.branch_exists(repository_path, branch_name):
        warning = f"Merged, but branch {branch_name!r} could not be deleted."
    return MergeOutcome(
        merged=True, base_branch=base, commit_sha=merged_sha, cleanup_warning=warning
    )
