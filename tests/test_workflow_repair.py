"""Tests for deterministic failure classification and the bounded repair loop (Phase 6)."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.agents.base import AdapterCapabilities, AgentRequest, AgentResult, Provider
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.config.models import LimitsConfig
from agentflow.git.lock import WorktreeLock
from agentflow.persistence.database import DatabaseManager
from agentflow.project.context import ProjectContext
from agentflow.routing.rules import DEFAULT_MODELS_CONFIG
from agentflow.task.profile import Stage, TaskProfile
from agentflow.workflow.repair import FailureCategory, RepairTier, RepairWorkflow, classify_failure
from agentflow.workflow.states import WorkflowState
from agentflow.workflow.verification import CommandResult, VerificationResult, VerificationStatus

# --- 1. classify_failure ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "stdout", "stderr", "expected"),
    [
        ("ruff check .", "", "E501 line too long", FailureCategory.LINT),
        ("black --check .", "", "would reformat file.py", FailureCategory.FORMATTING),
        (
            "pytest",
            "",
            "ModuleNotFoundError: No module named 'foo'",
            FailureCategory.MISSING_IMPORT,
        ),
        ("mypy src", "", "error: Incompatible type", FailureCategory.TYPE_MISMATCH),
        ("./mvnw compile", "", "error: expected ';'", FailureCategory.COMPILATION),
        ("./mvnw test", "", "deadlock detected", FailureCategory.TRANSACTION),
        ("pytest", "", "psycopg2.OperationalError: connection refused", FailureCategory.DATABASE),
        ("npm run test:integration", "", "some failure", FailureCategory.INTEGRATION_TEST),
        ("pytest tests/", "FAILED test_foo", "AssertionError", FailureCategory.UNIT_TEST),
        ("some-custom-tool", "", "an unrecognizable failure", FailureCategory.UNKNOWN),
    ],
)
def test_classify_failure_patterns(command, stdout, stderr, expected):
    """Deterministic keyword/command patterns classify each canonical failure type."""
    assert classify_failure(command, stdout, stderr) == expected


# --- 2. RepairWorkflow scaffolding -------------------------------------------------------


class ScriptedAdapter:
    """Fake AgentAdapter returning a pre-scripted sequence of AgentResults."""

    def __init__(self, provider: Provider, responses: list[AgentResult]) -> None:
        self._provider = provider
        self._responses = list(responses)
        self.calls = 0

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
        self.calls += 1
        return self._responses.pop(0)

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        raise NotImplementedError


class ScriptedVerificationRunner:
    """Fake VerificationRunner returning a pre-scripted sequence of VerificationResults."""

    def __init__(self, results: list[VerificationResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def run(
        self, run_id, worktree_path, verification_config, log_dir=None, timeout_seconds=None
    ):
        self.calls += 1
        return self._results.pop(0)


def make_agent_result(session_id: str = "sess-1") -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=Provider.OPENAI,
        model="gpt-6-luna",
        session_id=session_id,
        exit_code=0,
        text="fixed it",
        started_at=now,
        completed_at=now,
    )


def make_command_result(
    command: str, stdout: str = "", stderr: str = "", exit_code: int = 1
) -> CommandResult:
    now = datetime.now(timezone.utc)
    return CommandResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        started_at=now,
        completed_at=now,
    )


def failed_result(command_result: CommandResult) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.FAILED, groups_run=["grp"], command_results=[command_result]
    )


def passed_result() -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.PASSED,
        groups_run=["grp"],
        command_results=[make_command_result("pytest", exit_code=0)],
    )


def make_environment(tmp_path: Path):
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(tmp_path))
    db.create_run(run_id="run_1", project_id="proj_1", task="Do a thing")
    # RepairWorkflow is only ever invoked after implementation reaches VERIFYING.
    db.update_run_state("run_1", WorkflowState.VERIFYING.value)

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    project_context = ProjectContext(root_path=tmp_path, project_id="proj_1", project_name="demo")
    task_profile = TaskProfile(
        stage=Stage.IMPLEMENTATION, technologies=set(), affected_layers=set(), estimated_files=1
    )
    return db, worktree_path, project_context, task_profile


# --- 3. RepairWorkflow loop scenarios -----------------------------------------------------


@pytest.mark.asyncio
async def test_repair_succeeds_on_first_lightweight_attempt(tmp_path: Path):
    """A simple (lint) failure is repaired by Luna and verification passes immediately."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(Provider.OPENAI, ScriptedAdapter(Provider.OPENAI, [make_agent_result()]))
    registry.register(Provider.ANTHROPIC, ScriptedAdapter(Provider.ANTHROPIC, []))

    verifier = ScriptedVerificationRunner([passed_result()])
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG)

    initial = failed_result(make_command_result("ruff check .", stderr="E501 line too long"))
    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.VERIFYING
    assert outcome.attempts == 1
    assert outcome.final_tier == RepairTier.LIGHTWEIGHT
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_lightweight_fails_twice_then_escalates_to_standard(tmp_path: Path):
    """Two lightweight (Luna) failures escalate the third attempt to the standard (Sol) tier."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.OPENAI,
        ScriptedAdapter(
            Provider.OPENAI, [make_agent_result(), make_agent_result(), make_agent_result()]
        ),
    )
    registry.register(Provider.ANTHROPIC, ScriptedAdapter(Provider.ANTHROPIC, []))

    lint_failure = failed_result(make_command_result("ruff check .", stderr="E501 line too long"))
    verifier = ScriptedVerificationRunner([lint_failure, lint_failure, passed_result()])
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG)

    initial = failed_result(make_command_result("ruff check .", stderr="E501 line too long"))
    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.VERIFYING
    assert outcome.attempts == 3
    assert outcome.final_tier == RepairTier.STANDARD


@pytest.mark.asyncio
async def test_standard_fails_repeatedly_then_escalates_to_sonnet(tmp_path: Path):
    """Repeated complex (unit test) failures at the standard tier escalate to Claude Sonnet 5."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.OPENAI,
        ScriptedAdapter(Provider.OPENAI, [make_agent_result(), make_agent_result()]),
    )
    registry.register(
        Provider.ANTHROPIC, ScriptedAdapter(Provider.ANTHROPIC, [make_agent_result()])
    )

    unit_failure = failed_result(make_command_result("pytest", stdout="FAILED test_foo"))
    verifier = ScriptedVerificationRunner([unit_failure, unit_failure, passed_result()])
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG)

    initial = failed_result(make_command_result("pytest", stdout="FAILED test_foo"))
    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.VERIFYING
    assert outcome.attempts == 3
    assert outcome.final_tier == RepairTier.ESCALATION


@pytest.mark.asyncio
async def test_escalation_failure_ends_blocked(tmp_path: Path):
    """If even Claude Sonnet 5 cannot fix it, the run ends BLOCKED with a clear reason."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.OPENAI,
        ScriptedAdapter(Provider.OPENAI, [make_agent_result(), make_agent_result()]),
    )
    registry.register(
        Provider.ANTHROPIC, ScriptedAdapter(Provider.ANTHROPIC, [make_agent_result()])
    )

    unit_failure = failed_result(make_command_result("pytest", stdout="FAILED test_foo"))
    verifier = ScriptedVerificationRunner([unit_failure, unit_failure, unit_failure])
    limits = LimitsConfig(standard_failures=2, lightweight_verification_failures=2)
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG, limits)

    initial = failed_result(make_command_result("pytest", stdout="FAILED test_foo"))
    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.final_tier == RepairTier.ESCALATION
    assert outcome.blocker_reason is not None
    assert "architecture" in outcome.blocker_reason.lower()

    run = db.get_run("run_1")
    assert run is not None
    assert run.state == "BLOCKED"


@pytest.mark.asyncio
async def test_repair_releases_worktree_lock_after_each_attempt(tmp_path: Path):
    """The worktree lock is released after each repair invocation, not held across attempts."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(Provider.OPENAI, ScriptedAdapter(Provider.OPENAI, [make_agent_result()]))
    registry.register(Provider.ANTHROPIC, ScriptedAdapter(Provider.ANTHROPIC, []))

    verifier = ScriptedVerificationRunner([passed_result()])
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG)

    initial = failed_result(make_command_result("ruff check .", stderr="E501 line too long"))
    await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert not WorktreeLock(worktree_path).is_locked()


@pytest.mark.asyncio
async def test_agent_sessions_are_recorded_for_each_repair_attempt(tmp_path: Path):
    """Each repair attempt records an agent_sessions row with the tier's model."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(Provider.OPENAI, ScriptedAdapter(Provider.OPENAI, [make_agent_result()]))
    registry.register(Provider.ANTHROPIC, ScriptedAdapter(Provider.ANTHROPIC, []))

    verifier = ScriptedVerificationRunner([passed_result()])
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG)

    initial = failed_result(make_command_result("ruff check .", stderr="E501 line too long"))
    await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    sessions = db.list_agent_sessions("run_1")
    assert len(sessions) == 1
    assert sessions[0].model == "GPT-6 Luna"
    assert sessions[0].stage == "REPAIRING"
