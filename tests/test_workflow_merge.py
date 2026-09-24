"""Approved-run merge: squash into the base branch, with every refusal leaving checkouts intact."""

import subprocess
from pathlib import Path

import pytest
from rich.prompt import Prompt

from agentflow.application import Application
from agentflow.config.models import GlobalConfig, LoggingConfig, StorageConfig, WorktreesConfig
from agentflow.errors import MergeError, WorktreeError
from agentflow.git.worktree import WorktreeHandle, WorktreeManager, scratch_dir_for
from agentflow.persistence.database import DatabaseManager
from agentflow.process.removal import RemovalResult
from agentflow.project.discovery import discover_project
from agentflow.ui.approval import FinalApprovalDecision, ask_final_approval
from agentflow.workflow import merge as merge_module
from agentflow.workflow.merge import commit_subject, latest_decision, merge_run, preflight_merge
from agentflow.workflow.states import WorkflowState

RUN_ID = "RUN-MERGE01"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, name: str, content: str) -> None:
    (repo / name).write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", f"change {name}")


async def _setup(git_repo: Path, tmp_path: Path) -> tuple[DatabaseManager, WorktreeManager, Path]:
    db = DatabaseManager(tmp_path / "merge.db")
    db.initialize()
    db.upsert_project("project", "demo", str(git_repo))
    db.create_run(RUN_ID, "project", "Add a feature")
    manager = WorktreeManager(tmp_path / "worktrees")
    handle = await manager.create("demo", RUN_ID, git_repo)
    assert handle.base_branch and handle.base_commit
    db.record_decision("d1", RUN_ID, "base_branch", handle.base_branch)
    db.record_decision("d2", RUN_ID, "base_commit", handle.base_commit)
    (handle.path / "feature.py").write_text("value = 1\n", encoding="utf-8")
    return db, manager, handle.path


async def _merge(db: DatabaseManager, manager: WorktreeManager, repo: Path, worktree: Path):
    return await merge_run(db, manager, RUN_ID, repo, worktree, "Add feature", "AgentFlow run")


@pytest.mark.asyncio
async def test_create_records_base_branch_and_none_for_detached_head(
    git_repo: Path, tmp_path: Path
):
    manager = WorktreeManager(tmp_path / "worktrees")
    branch = _git(git_repo, "symbolic-ref", "--short", "HEAD")
    handle = await manager.create("demo", "RUN-A", git_repo)
    assert handle.base_branch == branch
    assert handle.base_commit == _git(git_repo, "rev-parse", "HEAD")

    _git(git_repo, "checkout", "--detach")
    detached = await manager.create("demo", "RUN-B", git_repo)
    assert detached.base_branch is None


@pytest.mark.asyncio
async def test_merge_squashes_into_base_branch_and_removes_worktree(git_repo: Path, tmp_path: Path):
    db, manager, worktree = await _setup(git_repo, tmp_path)
    before = _git(git_repo, "rev-parse", "HEAD")

    outcome = await _merge(db, manager, git_repo, worktree)

    assert outcome.merged, outcome.reason
    assert outcome.cleanup_warning is None
    assert _git(git_repo, "rev-parse", "HEAD~1") == before
    assert _git(git_repo, "log", "-1", "--format=%s") == "Add feature"
    assert (git_repo / "feature.py").read_text(encoding="utf-8") == "value = 1\n"
    assert _git(git_repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert not worktree.exists()
    assert not await manager.branch_exists(git_repo, f"agentflow/{RUN_ID}")
    assert latest_decision(db, RUN_ID, "merged_commit") == outcome.commit_sha

    again = await preflight_merge(db, manager, RUN_ID, git_repo, worktree, "x")
    assert not again.ok and "already merged" in again.reason


@pytest.mark.asyncio
async def test_merge_applies_on_top_of_base_branch_that_moved_on(git_repo: Path, tmp_path: Path):
    db, manager, worktree = await _setup(git_repo, tmp_path)
    _commit(git_repo, "other.py", "other = 1\n")
    moved = _git(git_repo, "rev-parse", "HEAD")

    outcome = await _merge(db, manager, git_repo, worktree)

    assert outcome.merged, outcome.reason
    assert _git(git_repo, "rev-parse", "HEAD~1") == moved
    assert (git_repo / "other.py").exists() and (git_repo / "feature.py").exists()


@pytest.mark.asyncio
async def test_merge_refused_when_checkout_is_on_another_branch(git_repo: Path, tmp_path: Path):
    db, manager, worktree = await _setup(git_repo, tmp_path)
    _git(git_repo, "switch", "-c", "elsewhere")
    before = _git(git_repo, "rev-parse", "HEAD")

    outcome = await _merge(db, manager, git_repo, worktree)

    assert not outcome.merged
    assert "elsewhere" in outcome.reason
    assert _git(git_repo, "rev-parse", "HEAD") == before
    assert (worktree / "feature.py").exists()


@pytest.mark.asyncio
async def test_merge_refused_when_checkout_has_uncommitted_changes(git_repo: Path, tmp_path: Path):
    db, manager, worktree = await _setup(git_repo, tmp_path)
    (git_repo / "README.md").write_text("local edit\n", encoding="utf-8")

    outcome = await _merge(db, manager, git_repo, worktree)

    assert not outcome.merged
    assert "uncommitted changes" in outcome.reason
    assert (git_repo / "README.md").read_text(encoding="utf-8") == "local edit\n"
    assert (worktree / "feature.py").exists()


@pytest.mark.asyncio
async def test_conflict_is_aborted_and_retry_still_lands_one_commit(git_repo: Path, tmp_path: Path):
    db, manager, worktree = await _setup(git_repo, tmp_path)
    (worktree / "README.md").write_text("from the run\n", encoding="utf-8")
    _commit(git_repo, "README.md", "from the user\n")
    before = _git(git_repo, "rev-parse", "HEAD")

    outcome = await _merge(db, manager, git_repo, worktree)

    assert not outcome.merged
    assert outcome.conflicts == ["README.md"]
    assert _git(git_repo, "rev-parse", "HEAD") == before
    assert _git(git_repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert (git_repo / "README.md").read_text(encoding="utf-8") == "from the user\n"
    assert worktree.exists()

    # User resolves by reverting their change; the run also gained another edit meanwhile.
    _git(git_repo, "revert", "--no-edit", "HEAD")
    (worktree / "extra.py").write_text("extra = 1\n", encoding="utf-8")
    retry = await _merge(db, manager, git_repo, worktree)

    assert retry.merged, retry.reason
    changed = _git(git_repo, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(changed) == ["README.md", "extra.py", "feature.py"]


@pytest.mark.asyncio
async def test_rejecting_commit_hook_leaves_checkout_untouched(git_repo: Path, tmp_path: Path):
    db, manager, worktree = await _setup(git_repo, tmp_path)
    hook = git_repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho blocked by hook >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    before = _git(git_repo, "rev-parse", "HEAD")

    outcome = await _merge(db, manager, git_repo, worktree)

    assert not outcome.merged
    assert "blocked by hook" in outcome.reason
    assert _git(git_repo, "rev-parse", "HEAD") == before
    assert worktree.exists()


@pytest.mark.asyncio
async def test_failed_worktree_removal_is_a_warning_not_a_failure(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db, manager, worktree = await _setup(git_repo, tmp_path)

    async def locked_remove(handle: WorktreeHandle, force: bool = True) -> None:
        raise WorktreeError("Directory not empty")

    async def cannot_remove(path: Path, allowed_roots: object, executor: object) -> RemovalResult:
        return RemovalResult(removed=path != worktree, needs_admin=True)

    monkeypatch.setattr(manager, "remove", locked_remove)
    monkeypatch.setattr(merge_module, "remove_tree", cannot_remove)
    outcome = await _merge(db, manager, git_repo, worktree)

    assert outcome.merged
    assert outcome.cleanup_warning is not None
    assert str(worktree) in outcome.cleanup_warning
    assert "Administrator" in outcome.cleanup_warning


@pytest.mark.asyncio
async def test_merge_clears_leftovers_when_git_worktree_remove_fails(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`git worktree remove` failing (as with locked temp dirs) still ends fully cleaned."""
    db, manager, worktree = await _setup(git_repo, tmp_path)
    scratch = scratch_dir_for(worktree)
    (scratch / "pytest-of-user").mkdir(parents=True)

    async def failing_remove(handle: WorktreeHandle, force: bool = True) -> None:
        raise WorktreeError("Directory not empty")

    monkeypatch.setattr(manager, "remove", failing_remove)
    outcome = await _merge(db, manager, git_repo, worktree)

    assert outcome.merged and outcome.cleanup_warning is None
    assert not worktree.exists() and not scratch.exists()
    assert not await manager.branch_exists(git_repo, f"agentflow/{RUN_ID}")


def test_commit_subject_from_plan_objective_or_task():
    plan = (
        "## Objective\nAdd a **Replace Model** action to the `Model Catalog` tab. "
        "When an admin confirms, policies update.\n\n## Other\n"
    )
    assert commit_subject(plan, "task") == "Add a Replace Model action to the Model Catalog tab"
    long_plan = plan.replace("tab.", "tab on the `/admin/ai-management` page.")
    assert commit_subject(long_plan, "task") == (
        "Add a Replace Model action to the Model Catalog tab on the"
    )
    assert commit_subject("# Plan\n\nObjective: x.\n", "Fix login\nmore") == "Fix login"
    long_subject = commit_subject("", "word " * 40)
    assert len(long_subject) <= 72 and not long_subject.endswith(" ")


def test_final_prompt_omits_merge_when_unavailable(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_ask(prompt: str, choices: list[str], default: str, console: object) -> str:
        seen.update(choices=choices, default=default)
        return default

    monkeypatch.setattr(Prompt, "ask", fake_ask)
    assert ask_final_approval(merge_available=False) == FinalApprovalDecision.KEEP_WORKTREE
    assert "merge" not in seen["choices"]  # type: ignore[operator]
    assert ask_final_approval(merge_available=True) == FinalApprovalDecision.MERGE


def _app(tmp_path: Path) -> Application:
    config = GlobalConfig(
        worktrees=WorktreesConfig(root=tmp_path / "worktrees"),
        storage=StorageConfig(database=tmp_path / "merge.db"),
        logging=LoggingConfig(root=tmp_path / "logs"),
    )
    return Application(config=config, db_manager=DatabaseManager(tmp_path / "merge.db"))


@pytest.mark.asyncio
async def test_agentflow_merge_command_merges_kept_run(git_repo: Path, tmp_path: Path):
    app_instance = _app(tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    db = app_instance.db_manager
    db.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    db.create_run(RUN_ID, ctx.project_id, "Add a feature")
    manager = WorktreeManager(app_instance.config.worktrees.root)

    with pytest.raises(MergeError, match="only completed runs"):
        await app_instance.merge_completed_run(RUN_ID, project_path=git_repo)

    db.update_run_state(RUN_ID, WorkflowState.COMPLETED.value, "kept worktree")
    handle = await manager.create(ctx.project_name, RUN_ID, git_repo)
    assert handle.base_branch
    db.record_decision("d1", RUN_ID, "base_branch", handle.base_branch)
    db.record_decision("d2", RUN_ID, "base_commit", str(handle.base_commit))
    (handle.path / "feature.py").write_text("value = 1\n", encoding="utf-8")

    outcome = await app_instance.merge_completed_run(
        RUN_ID, project_path=git_repo, confirm=lambda: True
    )

    assert outcome.merged, outcome.reason
    assert (git_repo / "feature.py").exists()
    assert not handle.path.exists()
