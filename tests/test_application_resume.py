"""Integration tests for Application.run_resume (Milestone 7: Phase 10 crash recovery).

Exercises resume against a real Git fixture repository and real SQLite persistence. Only the
AI agent CLIs are scripted; verification, Git worktrees, and the database are real.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

import agentflow.workflow.planning as planning_module
from agentflow.agents.base import AdapterCapabilities, AgentRequest, AgentResult, Provider
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.application import Application
from agentflow.concurrency.run_lock import RunLock
from agentflow.config.models import GlobalConfig, LoggingConfig, StorageConfig, WorktreesConfig
from agentflow.errors import ResumeError, RunLockedError
from agentflow.git.lock import WorktreeLock
from agentflow.git.worktree import WorktreeManager
from agentflow.persistence.database import DatabaseManager
from agentflow.persistence.models import RunStatus
from agentflow.project.discovery import discover_project
from agentflow.routing.rules import ModelRef, RoleOverride
from agentflow.task.profile import Stage
from agentflow.ui.approval import ApprovalDecision, FinalApprovalDecision
from agentflow.workflow.states import WorkflowState

PY = sys.executable
DEAD_PID = 999999


class ScriptedAdapter:
    """Fake AgentAdapter running a scripted sequence of (response, worktree side-effect) steps."""

    def __init__(
        self,
        provider: Provider,
        steps: list[tuple[AgentResult, Callable[[Path], None] | None]],
    ) -> None:
        self._provider = provider
        self._steps = list(steps)
        self.calls = 0

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=True,
            supports_read_only_mode=True,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    async def start(self, request: AgentRequest) -> AgentResult:
        self.calls += 1
        response, side_effect = self._steps.pop(0)
        if side_effect is not None:
            assert request.working_directory is not None
            side_effect(request.working_directory)
        return response

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        return await self.start(request)


def make_agent_result(provider: Provider, model: str, text: str = "done") -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=provider,
        model=model,
        session_id="sess-1",
        exit_code=0,
        text=text,
        started_at=now,
        completed_at=now,
    )


def plan_ready_payload() -> str:
    return json.dumps(
        {
            "status": "plan_ready",
            "plan_markdown": "# Plan\n\nObjective: add a health check.\n",
            "task_profile": {
                "stage": "IMPLEMENTATION",
                "technologies": ["python"],
                "affected_layers": ["backend"],
                "estimated_files": 1,
            },
        }
    )


def review_payload(status: str = "APPROVED", findings=None) -> str:
    return json.dumps({"status": status, "findings": findings or []})


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _commit_file(repo: Path, name: str, content: str) -> None:
    (repo / name).write_text(content, encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", f"add {name}"], repo)


def _write_routing_yaml(repo: Path) -> None:
    orch_dir = repo / ".ai-orchestrator"
    orch_dir.mkdir(exist_ok=True)
    (orch_dir / "routing.yaml").write_text(
        "version: 1\n\nproject:\n  name: demo\n\n"
        "verification:\n  py:\n    detect:\n      - check_ok.py\n"
        f"    commands:\n      - '{PY} check_ok.py'\n",
        encoding="utf-8",
    )


def make_app(registry: AgentAdapterRegistry, tmp_path: Path) -> Application:
    config = GlobalConfig(
        worktrees=WorktreesConfig(root=tmp_path / "worktrees"),
        storage=StorageConfig(database=tmp_path / "test.db"),
        logging=LoggingConfig(root=tmp_path / "logs"),
    )
    return Application(
        config=config,
        db_manager=DatabaseManager(tmp_path / "test.db"),
        agent_registry=registry,
    )


def make_registry(review_cycles: int = 1):
    def write_feature(worktree: Path) -> None:
        (worktree / "feature.py").write_text("print('hi')\n", encoding="utf-8")

    claude = ScriptedAdapter(
        Provider.ANTHROPIC,
        [(make_agent_result(Provider.ANTHROPIC, "sonnet", plan_ready_payload()), None)],
    )
    codex = ScriptedAdapter(
        Provider.OPENAI, [(make_agent_result(Provider.OPENAI, "luna"), write_feature)]
    )
    gemini = ScriptedAdapter(
        Provider.GOOGLE,
        [
            (make_agent_result(Provider.GOOGLE, "flash", review_payload()), None)
            for _ in range(review_cycles)
        ],
    )
    registry = AgentAdapterRegistry()
    registry.register(Provider.ANTHROPIC, claude)
    registry.register(Provider.OPENAI, codex)
    registry.register(Provider.GOOGLE, gemini)
    return registry, claude, codex, gemini


def _prepare_repo(git_repo: Path) -> None:
    _commit_file(git_repo, "check_ok.py", "import sys\nsys.exit(0)\n")
    _write_routing_yaml(git_repo)


@pytest.mark.asyncio
async def test_resume_unknown_run_raises(git_repo: Path, tmp_path: Path):
    """Resuming a run_id that was never created is rejected."""
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    with pytest.raises(ResumeError, match="not found"):
        await app_instance.run_resume("RUN-DOES-NOT-EXIST", project_path=git_repo)


@pytest.mark.asyncio
async def test_resume_new_run_raises(git_repo: Path, tmp_path: Path):
    """A run that never advanced past NEW cannot be resumed."""
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    app_instance.db_manager.upsert_project("proj_1", "demo", str(git_repo))
    app_instance.db_manager.create_run(run_id="RUN-NEW", project_id="proj_1", task="Add a feature")

    with pytest.raises(ResumeError, match="never advanced"):
        await app_instance.run_resume("RUN-NEW", project_path=git_repo)


@pytest.mark.asyncio
async def test_resume_terminal_run_raises(git_repo: Path, tmp_path: Path):
    """A run that already reached a terminal state has nothing left to resume."""
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    app_instance.db_manager.upsert_project("proj_1", "demo", str(git_repo))
    app_instance.db_manager.create_run(run_id="RUN-DONE", project_id="proj_1", task="Add a feature")
    app_instance.db_manager.update_run_state(
        "RUN-DONE", WorkflowState.COMPLETED.value, "test setup"
    )

    with pytest.raises(ResumeError, match="already ended"):
        await app_instance.run_resume("RUN-DONE", project_path=git_repo)


@pytest.mark.asyncio
async def test_resume_during_planning_reuses_persisted_cli_session(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A planning crash resumes the persisted CLI session and then continues the full run."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)
    registry, claude, codex, gemini = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    app_instance.db_manager.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    app_instance.db_manager.create_run("RUN-PLAN", ctx.project_id, "Add a feature")
    app_instance.db_manager.record_agent_session(
        "session-record", "RUN-PLAN", "PLANNING", "anthropic", "Claude Sonnet 5", "prior-sess"
    )
    app_instance.db_manager.update_run_state("RUN-PLAN", WorkflowState.PLANNING.value, "crashed")

    outcome = await app_instance.run_resume(
        "RUN-PLAN",
        project_path=git_repo,
        final_approval_prompt=lambda: FinalApprovalDecision.APPROVE,
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert claude.calls == 1
    assert codex.calls == 1
    assert gemini.calls == 1
    recoveries = app_instance.db_manager.list_decisions("RUN-PLAN")
    assert any(decision.question == "planning_recovery" for decision in recoveries)


@pytest.mark.asyncio
async def test_resume_during_planning_reconstructs_context_when_session_is_unavailable(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An expired planner session falls back to a fresh session with persisted task context."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)
    failed = make_agent_result(Provider.ANTHROPIC, "sonnet", "session expired")
    failed.exit_code = 1
    registry, claude, codex, gemini = make_registry()
    claude._steps.insert(0, (failed, None))
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    app_instance.db_manager.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    app_instance.db_manager.create_run("RUN-PLAN-FALLBACK", ctx.project_id, "Add a feature")
    app_instance.db_manager.record_agent_session(
        "session-record",
        "RUN-PLAN-FALLBACK",
        "PLANNING",
        "anthropic",
        "Claude Sonnet 5",
        "expired-sess",
    )
    app_instance.db_manager.update_run_state(
        "RUN-PLAN-FALLBACK", WorkflowState.PLANNING.value, "crashed"
    )

    outcome = await app_instance.run_resume(
        "RUN-PLAN-FALLBACK",
        project_path=git_repo,
        final_approval_prompt=lambda: FinalApprovalDecision.APPROVE,
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert claude.calls == 2
    assert codex.calls == 1
    assert gemini.calls == 1
    recoveries = app_instance.db_manager.list_decisions("RUN-PLAN-FALLBACK")
    assert any("unavailable" in decision.answer for decision in recoveries)


@pytest.mark.asyncio
async def test_resume_with_no_worktree_yet_implements_from_persisted_plan(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A crash right after TASK_CLASSIFIED (no worktree yet) resumes by implementing from the
    persisted plan/TaskProfile artifacts -- without re-invoking the planner."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)
    registry, claude, codex, gemini = make_registry()
    app_instance = make_app(registry, tmp_path)

    planning_outcome = await app_instance.run_planning("Add a health check", project_path=git_repo)
    assert planning_outcome.state == WorkflowState.TASK_CLASSIFIED
    assert claude.calls == 1

    outcome = await app_instance.run_resume(
        planning_outcome.run_id,
        project_path=git_repo,
        final_approval_prompt=lambda: FinalApprovalDecision.APPROVE,
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert claude.calls == 1  # planning was NOT re-invoked
    assert codex.calls == 1
    assert gemini.calls == 1


@pytest.mark.asyncio
async def test_resume_falls_back_to_db_task_when_task_md_is_missing(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Resume must not hard-fail just because the human-readable task.md export is gone --
    the task description is durably persisted in the `runs.task` DB column too."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)
    registry, claude, codex, gemini = make_registry()
    app_instance = make_app(registry, tmp_path)

    planning_outcome = await app_instance.run_planning("Add a health check", project_path=git_repo)
    assert planning_outcome.state == WorkflowState.TASK_CLASSIFIED

    run_dir = git_repo / ".ai-orchestrator" / "runs" / planning_outcome.run_id
    (run_dir / "task.md").unlink()

    outcome = await app_instance.run_resume(
        planning_outcome.run_id,
        project_path=git_repo,
        final_approval_prompt=lambda: FinalApprovalDecision.APPROVE,
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert claude.calls == 1  # planning was NOT re-invoked


@pytest.mark.asyncio
async def test_resume_with_empty_worktree_recreates_and_implements(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A worktree that exists but holds no changes yet (crash before any edit) is recreated
    rather than reused, then implementation proceeds normally."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)
    registry, claude, codex, gemini = make_registry()
    app_instance = make_app(registry, tmp_path)

    planning_outcome = await app_instance.run_planning("Add a health check", project_path=git_repo)
    run_id = planning_outcome.run_id

    worktree_manager = WorktreeManager(
        app_instance.config.worktrees.root, executor=app_instance.executor
    )
    await worktree_manager.create("demo", run_id, git_repo)
    app_instance.db_manager.update_run_state(
        run_id, WorkflowState.WORKTREE_READY.value, "simulated crash before any edit"
    )

    outcome = await app_instance.run_resume(
        run_id, project_path=git_repo, final_approval_prompt=lambda: FinalApprovalDecision.APPROVE
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert codex.calls == 1
    assert gemini.calls == 1


@pytest.mark.asyncio
async def test_resume_with_existing_changes_skips_reimplementation(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A worktree that already holds implementation changes is reused: resume re-verifies and
    proceeds straight to review, without invoking the implementation agent again."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)
    registry, claude, codex, gemini = make_registry()
    app_instance = make_app(registry, tmp_path)

    impl_outcome = await app_instance.run_implementation(
        "Add a health check", project_path=git_repo
    )
    assert impl_outcome.verification is not None and impl_outcome.verification.success
    run_id = impl_outcome.planning_outcome.run_id
    assert codex.calls == 1

    # Simulate a crash that happened right after implementation, before review started.
    app_instance.db_manager.update_run_state(
        run_id, WorkflowState.IMPLEMENTING.value, "simulated crash"
    )
    app_instance.db_manager.update_run_status(run_id, RunStatus.RUNNING.value)

    outcome = await app_instance.run_resume(
        run_id, project_path=git_repo, final_approval_prompt=lambda: FinalApprovalDecision.APPROVE
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert claude.calls == 1  # planning not repeated
    assert codex.calls == 1  # implementation not repeated -- existing diff was reused
    assert gemini.calls == 1


@pytest.mark.asyncio
async def test_resume_clears_stale_worktree_lock_and_continues(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A writer lock left by a dead process is cleared automatically, and resume proceeds."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)
    registry, claude, codex, gemini = make_registry()
    app_instance = make_app(registry, tmp_path)

    impl_outcome = await app_instance.run_implementation(
        "Add a health check", project_path=git_repo
    )
    run_id = impl_outcome.planning_outcome.run_id
    assert impl_outcome.implementation is not None and impl_outcome.implementation.worktree

    worktree_path = impl_outcome.implementation.worktree.path
    lock = WorktreeLock(worktree_path)
    lock.lock_path.write_text(
        f'{{"owner": "ghost", "pid": {DEAD_PID}, "acquired_at": "x"}}', encoding="utf-8"
    )
    assert lock.is_locked()

    app_instance.db_manager.update_run_state(
        run_id, WorkflowState.VERIFYING.value, "simulated crash while holding a stale lock"
    )
    app_instance.db_manager.update_run_status(run_id, RunStatus.RUNNING.value)

    outcome = await app_instance.run_resume(
        run_id, project_path=git_repo, final_approval_prompt=lambda: FinalApprovalDecision.APPROVE
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert not lock.is_locked()


@pytest.mark.asyncio
async def test_resume_refuses_when_worktree_lock_is_live(git_repo: Path, tmp_path: Path):
    """A worktree lock owned by a still-running process blocks resume rather than being broken."""
    _prepare_repo(git_repo)
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    app_instance.db_manager.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    run_id = "RUN-LIVE"
    app_instance.db_manager.create_run(
        run_id=run_id, project_id=ctx.project_id, task="Add a feature"
    )

    run_dir = git_repo / ".ai-orchestrator" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "task.md").write_text("# Task\n\nAdd a feature\n", encoding="utf-8")
    (run_dir / "approved-plan.md").write_text("# Plan\n", encoding="utf-8")
    (run_dir / "task-profile.json").write_text(
        json.dumps(
            {
                "stage": "IMPLEMENTATION",
                "technologies": [],
                "affected_layers": [],
                "estimated_files": 1,
            }
        ),
        encoding="utf-8",
    )

    worktree_manager = WorktreeManager(
        app_instance.config.worktrees.root, executor=app_instance.executor
    )
    handle = await worktree_manager.create(ctx.project_name, run_id, git_repo)
    (handle.path / "partial.py").write_text("x = 1\n", encoding="utf-8")

    live_lock = WorktreeLock(handle.path)
    live_lock.acquire(owner="a-live-process")
    try:
        app_instance.db_manager.update_run_state(
            run_id, WorkflowState.IMPLEMENTING.value, "simulated crash while a writer is still up"
        )
        with pytest.raises(ResumeError, match="actively locked"):
            await app_instance.run_resume(run_id, project_path=git_repo)
    finally:
        live_lock.release()


@pytest.mark.asyncio
async def test_resume_refuses_when_run_is_controlled_by_a_live_process(
    git_repo: Path, tmp_path: Path
):
    """Two processes may never drive the same run concurrently."""
    _prepare_repo(git_repo)
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    app_instance.db_manager.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    run_id = "RUN-CONTROLLED"
    app_instance.db_manager.create_run(
        run_id=run_id, project_id=ctx.project_id, task="Add a feature"
    )
    app_instance.db_manager.update_run_state(run_id, WorkflowState.IMPLEMENTING.value, "in flight")

    run_lock = RunLock(run_id, app_instance._locks_root())
    run_lock.acquire(owner="another-agentflow-process")
    try:
        with pytest.raises(RunLockedError):
            await app_instance.run_resume(run_id, project_path=git_repo)
    finally:
        run_lock.release()


@pytest.mark.asyncio
async def test_different_projects_can_create_isolated_worktrees_concurrently(
    git_repo: Path, tmp_path: Path
):
    """Independent projects use separate run locks and worktree roots without collisions."""
    second_repo = tmp_path / "second-repo"
    second_repo.mkdir()
    _git(["init"], second_repo)
    _git(["config", "user.email", "test@example.com"], second_repo)
    _git(["config", "user.name", "AgentFlow Test"], second_repo)
    (second_repo / "README.md").write_text("# Second Repo\n", encoding="utf-8")
    _git(["add", "-A"], second_repo)
    _git(["commit", "-m", "Initial commit"], second_repo)

    manager = WorktreeManager(tmp_path / "worktrees")
    first_lock = RunLock("RUN-FIRST", tmp_path / "locks")
    second_lock = RunLock("RUN-SECOND", tmp_path / "locks")
    first_lock.acquire()
    second_lock.acquire()
    try:
        first_handle = await manager.create("first", "RUN-FIRST", git_repo)
        second_handle = await manager.create("second", "RUN-SECOND", second_repo)
    finally:
        first_lock.release()
        second_lock.release()

    assert first_handle.path != second_handle.path
    assert first_handle.path.exists()
    assert second_handle.path.exists()
    assert first_handle.repository_path == git_repo
    assert second_handle.repository_path == second_repo


@pytest.mark.asyncio
async def test_cleanup_removes_terminal_unlocked_worktree_and_logs(git_repo: Path, tmp_path: Path):
    """Cleanup removes only a terminal run's disposable worktree, branch, and verification logs."""
    _prepare_repo(git_repo)
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    app_instance.db_manager.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    run_id = "RUN-CLEAN"
    app_instance.db_manager.create_run(run_id, ctx.project_id, "Add a feature")
    app_instance.db_manager.update_run_state(run_id, WorkflowState.COMPLETED.value, "test setup")

    manager = WorktreeManager(app_instance.config.worktrees.root, executor=app_instance.executor)
    handle = await manager.create(ctx.project_name, run_id, git_repo)
    (handle.path / "feature.py").write_text("value = 1\n", encoding="utf-8")
    log_dir = git_repo / ".ai-orchestrator" / "runs" / run_id / "verification"
    log_dir.mkdir(parents=True)
    (log_dir / "stdout.log").write_text("ok\n", encoding="utf-8")

    report = await app_instance.run_cleanup(project_path=git_repo)

    assert report.worktrees_removed == [run_id]
    assert report.logs_cleared == [run_id]
    assert not handle.path.exists()
    assert not await manager.branch_exists(git_repo, handle.branch_name)
    assert not log_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_preserves_worktree_guarded_by_live_writer_lock(
    git_repo: Path, tmp_path: Path
):
    """Cleanup leaves even a terminal worktree intact while its writer lock is live."""
    _prepare_repo(git_repo)
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    app_instance.db_manager.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    run_id = "RUN-KEEP"
    app_instance.db_manager.create_run(run_id, ctx.project_id, "Add a feature")
    app_instance.db_manager.update_run_state(run_id, WorkflowState.COMPLETED.value, "test setup")

    manager = WorktreeManager(app_instance.config.worktrees.root, executor=app_instance.executor)
    handle = await manager.create(ctx.project_name, run_id, git_repo)
    lock = WorktreeLock(handle.path)
    lock.acquire(owner="active-writer")
    try:
        report = await app_instance.run_cleanup(project_path=git_repo)
        assert report.worktrees_removed == []
        assert report.skipped_active == [run_id]
        assert handle.path.exists()
        assert await manager.branch_exists(git_repo, handle.branch_name)
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_cleanup_preserves_nonterminal_worktree(git_repo: Path, tmp_path: Path):
    """Cleanup never removes a worktree for a run that may still be resumed."""
    _prepare_repo(git_repo)
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    app_instance.db_manager.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    run_id = "RUN-ACTIVE"
    app_instance.db_manager.create_run(run_id, ctx.project_id, "Add a feature")
    app_instance.db_manager.update_run_state(run_id, WorkflowState.IMPLEMENTING.value, "test setup")

    manager = WorktreeManager(app_instance.config.worktrees.root, executor=app_instance.executor)
    handle = await manager.create(ctx.project_name, run_id, git_repo)

    report = await app_instance.run_cleanup(project_path=git_repo)

    assert report.worktrees_removed == []
    assert report.skipped_active == [run_id]
    assert handle.path.exists()


@pytest.mark.asyncio
async def test_resume_blocked_run_retries_from_recovered_stage(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A BLOCKED run (agent CLI failure) is resumed by retrying the stage it blocked in."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)
    registry, claude, codex, gemini = make_registry()
    app_instance = make_app(registry, tmp_path)

    planning_outcome = await app_instance.run_planning("Add a health check", project_path=git_repo)
    run_id = planning_outcome.run_id
    assert planning_outcome.state == WorkflowState.TASK_CLASSIFIED

    # Simulate implementation starting, then its coding-agent CLI failing (e.g. a usage limit).
    app_instance.db_manager.update_run_state(
        run_id, WorkflowState.IMPLEMENTING.value, "implementation started"
    )
    app_instance.db_manager.update_run_state(
        run_id, WorkflowState.BLOCKED.value, "Codex CLI hit a usage limit"
    )
    app_instance.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)

    outcome = await app_instance.run_resume(
        run_id, project_path=git_repo, final_approval_prompt=lambda: FinalApprovalDecision.APPROVE
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert claude.calls == 1  # planning was not re-invoked
    assert codex.calls == 1
    assert gemini.calls == 1


@pytest.mark.asyncio
async def test_resume_blocked_run_with_no_recoverable_stage_raises(git_repo: Path, tmp_path: Path):
    """A BLOCKED run whose only recorded prior state is NEW cannot be resumed."""
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    app_instance.initialize()
    ctx = discover_project(explicit_path=git_repo)
    app_instance.db_manager.upsert_project(ctx.project_id, ctx.project_name, str(git_repo))
    run_id = "RUN-NOWHERE"
    app_instance.db_manager.create_run(
        run_id=run_id, project_id=ctx.project_id, task="Add a feature"
    )
    app_instance.db_manager.update_run_state(
        run_id, WorkflowState.BLOCKED.value, "blocked immediately"
    )

    with pytest.raises(ResumeError, match="no recoverable prior stage"):
        await app_instance.run_resume(run_id, project_path=git_repo)


@pytest.mark.asyncio
async def test_resume_blocked_run_honors_role_override_for_recovered_stage(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A role_override scoped to IMPLEMENTATION reroutes the retried stage away from its
    original provider entirely."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _prepare_repo(git_repo)

    def write_feature(worktree: Path) -> None:
        (worktree / "feature.py").write_text("print('hi')\n", encoding="utf-8")

    claude = ScriptedAdapter(
        Provider.ANTHROPIC,
        [
            (make_agent_result(Provider.ANTHROPIC, "sonnet", plan_ready_payload()), None),
            (make_agent_result(Provider.ANTHROPIC, "sonnet"), write_feature),
        ],
    )
    codex = ScriptedAdapter(Provider.OPENAI, [])  # must never be invoked
    gemini = ScriptedAdapter(
        Provider.GOOGLE, [(make_agent_result(Provider.GOOGLE, "flash", review_payload()), None)]
    )
    registry = AgentAdapterRegistry()
    registry.register(Provider.ANTHROPIC, claude)
    registry.register(Provider.OPENAI, codex)
    registry.register(Provider.GOOGLE, gemini)
    app_instance = make_app(registry, tmp_path)

    planning_outcome = await app_instance.run_planning("Add a health check", project_path=git_repo)
    run_id = planning_outcome.run_id

    app_instance.db_manager.update_run_state(
        run_id, WorkflowState.IMPLEMENTING.value, "implementation started"
    )
    app_instance.db_manager.update_run_state(
        run_id, WorkflowState.BLOCKED.value, "Codex CLI hit a usage limit"
    )
    app_instance.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)

    role_override = RoleOverride(
        overrides={Stage.IMPLEMENTATION: ModelRef(provider="anthropic", model="Claude Sonnet 5")}
    )
    outcome = await app_instance.run_resume(
        run_id,
        project_path=git_repo,
        role_override=role_override,
        final_approval_prompt=lambda: FinalApprovalDecision.APPROVE,
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert claude.calls == 2  # planning, then the overridden re-implementation
    assert codex.calls == 0  # never routed to -- override replaced it entirely
    assert gemini.calls == 1
