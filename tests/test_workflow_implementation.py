"""Fixture-repo tests for the Codex-driven implementation workflow (Phase 5)."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.agents.base import AdapterCapabilities, AgentRequest, AgentResult, Provider
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.git.lock import WorktreeLock
from agentflow.git.worktree import WorktreeManager
from agentflow.persistence.database import DatabaseManager
from agentflow.project.context import ProjectContext
from agentflow.routing.complexity import ComplexityLevel
from agentflow.routing.decision import RoutingDecision
from agentflow.task.profile import Stage, TaskProfile
from agentflow.workflow.implementation import ImplementationWorkflow
from agentflow.workflow.states import WorkflowState


class FileWritingAdapter:
    """Fake AgentAdapter that writes a file into the worktree, simulating a real code change."""

    def __init__(
        self, provider: Provider, response: AgentResult, filename: str = "new_feature.py"
    ) -> None:
        self._provider = provider
        self._response = response
        self._filename = filename
        self.received_request: AgentRequest | None = None

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=False,
            supports_read_only_mode=False,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    async def start(self, request: AgentRequest) -> AgentResult:
        self.received_request = request
        assert request.working_directory is not None
        (request.working_directory / self._filename).write_text("print('hi')\n", encoding="utf-8")
        return self._response

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        raise NotImplementedError


class NoOpAdapter:
    """Fake AgentAdapter that reports success without changing anything (no-op)."""

    def __init__(self, provider: Provider, response: AgentResult) -> None:
        self._provider = provider
        self._response = response

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=False,
            supports_read_only_mode=False,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    async def start(self, request: AgentRequest) -> AgentResult:
        return self._response

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        raise NotImplementedError


def make_agent_result(
    exit_code: int = 0, stderr: str = "", text: str = "Implemented."
) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=Provider.OPENAI,
        model="gpt-5.6-luna",
        session_id="sess-codex-1",
        exit_code=exit_code,
        text=text,
        stderr=stderr,
        started_at=now,
        completed_at=now,
    )


def make_routing_decision(
    role: str = "implementation.lightweight", model: str = "GPT-5.6 Luna"
) -> RoutingDecision:
    return RoutingDecision(
        stage=Stage.IMPLEMENTATION,
        provider=Provider.OPENAI,
        model=model,
        role=role,
        matched_rule="low-complexity",
        reason="test",
        complexity_score=0,
        complexity=ComplexityLevel.LOW,
        risk_flags=[],
    )


def make_environment(git_repo: Path, tmp_path: Path):
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(git_repo))
    db.create_run(run_id="run_1", project_id="proj_1", task="Add a feature")
    db.update_run_state("run_1", WorkflowState.TASK_CLASSIFIED.value)

    project_context = ProjectContext(root_path=git_repo, project_id="proj_1", project_name="demo")
    task_profile = TaskProfile(
        stage=Stage.IMPLEMENTATION, technologies=set(), affected_layers=set(), estimated_files=1
    )
    worktree_manager = WorktreeManager(tmp_path / "worktrees")
    return db, project_context, task_profile, worktree_manager


@pytest.mark.asyncio
async def test_successful_implementation_reaches_verifying(git_repo: Path, tmp_path: Path):
    """A successful implementation captures the diff and transitions to VERIFYING."""
    db, ctx, task_profile, worktree_manager = make_environment(git_repo, tmp_path)
    registry = AgentAdapterRegistry()
    adapter = FileWritingAdapter(Provider.OPENAI, make_agent_result())
    registry.register(Provider.OPENAI, adapter)

    workflow = ImplementationWorkflow(db, registry, worktree_manager)
    decision = make_routing_decision()

    outcome = await workflow.run("run_1", "Add a feature", ctx, "# Plan", task_profile, decision)

    assert outcome.state == WorkflowState.VERIFYING
    assert outcome.worktree is not None
    assert outcome.diff is not None
    assert "new_feature.py" in outcome.diff.changed_files
    assert outcome.summary_path is not None and outcome.summary_path.exists()

    run = db.get_run("run_1")
    assert run is not None
    assert run.state == "VERIFYING"

    # Primary working tree remains untouched.
    assert not (git_repo / "new_feature.py").exists()

    # Implementation happens only inside the worktree, and the request carried it as
    # the working directory (repository_path stays the original repo root).
    assert adapter.received_request is not None
    assert adapter.received_request.repository_path == git_repo
    assert adapter.received_request.working_directory == outcome.worktree.path


@pytest.mark.asyncio
async def test_implementation_only_writes_inside_worktree(git_repo: Path, tmp_path: Path):
    """Source changes from the implementation agent appear only in the AgentFlow worktree."""
    db, ctx, task_profile, worktree_manager = make_environment(git_repo, tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(Provider.OPENAI, FileWritingAdapter(Provider.OPENAI, make_agent_result()))

    workflow = ImplementationWorkflow(db, registry, worktree_manager)
    outcome = await workflow.run(
        "run_1", "Add a feature", ctx, "# Plan", task_profile, make_routing_decision()
    )

    assert outcome.worktree is not None
    assert (outcome.worktree.path / "new_feature.py").exists()
    assert not (git_repo / "new_feature.py").exists()


@pytest.mark.asyncio
async def test_noop_implementation_is_blocked(git_repo: Path, tmp_path: Path):
    """An agent that reports success but changes nothing is treated as blocked, not successful."""
    db, ctx, task_profile, worktree_manager = make_environment(git_repo, tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(Provider.OPENAI, NoOpAdapter(Provider.OPENAI, make_agent_result()))

    workflow = ImplementationWorkflow(db, registry, worktree_manager)
    outcome = await workflow.run(
        "run_1", "Add a feature", ctx, "# Plan", task_profile, make_routing_decision()
    )

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.blocker_reason is not None
    assert "no-op" in outcome.blocker_reason.lower()


@pytest.mark.asyncio
async def test_agent_failure_is_blocked(git_repo: Path, tmp_path: Path):
    """A non-zero exit code from the implementation agent blocks the run."""
    db, ctx, task_profile, worktree_manager = make_environment(git_repo, tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.OPENAI, NoOpAdapter(Provider.OPENAI, make_agent_result(exit_code=1, stderr="boom"))
    )

    workflow = ImplementationWorkflow(db, registry, worktree_manager)
    outcome = await workflow.run(
        "run_1", "Add a feature", ctx, "# Plan", task_profile, make_routing_decision()
    )

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.blocker_reason is not None
    assert "boom" in outcome.blocker_reason


@pytest.mark.asyncio
async def test_worktree_lock_released_after_implementation(git_repo: Path, tmp_path: Path):
    """The single-writer lock is released once implementation completes."""
    db, ctx, task_profile, worktree_manager = make_environment(git_repo, tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(Provider.OPENAI, FileWritingAdapter(Provider.OPENAI, make_agent_result()))

    workflow = ImplementationWorkflow(db, registry, worktree_manager)
    outcome = await workflow.run(
        "run_1", "Add a feature", ctx, "# Plan", task_profile, make_routing_decision()
    )

    assert outcome.worktree is not None
    assert not WorktreeLock(outcome.worktree.path).is_locked()


@pytest.mark.asyncio
async def test_agent_session_is_recorded(git_repo: Path, tmp_path: Path):
    """The implementation agent invocation is recorded as an agent_sessions row."""
    db, ctx, task_profile, worktree_manager = make_environment(git_repo, tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(Provider.OPENAI, FileWritingAdapter(Provider.OPENAI, make_agent_result()))

    workflow = ImplementationWorkflow(db, registry, worktree_manager)
    await workflow.run(
        "run_1", "Add a feature", ctx, "# Plan", task_profile, make_routing_decision()
    )

    sessions = db.list_agent_sessions("run_1")
    assert len(sessions) == 1
    assert sessions[0].model == "GPT-5.6 Luna"
    assert sessions[0].cli_session_id == "sess-codex-1"
