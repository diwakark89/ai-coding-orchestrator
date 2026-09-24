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
from agentflow.workflow.repair import (
    FailureCategory,
    RepairTier,
    RepairWorkflow,
    classify_failure,
    is_environment_failure,
    is_potentially_intermittent,
)
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


def make_agent_result(
    session_id: str = "sess-1", exit_code: int = 0, stderr: str = ""
) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=Provider.OPENAI,
        model="gpt-6-luna",
        session_id=session_id,
        exit_code=exit_code,
        text="fixed it",
        stderr=stderr,
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


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("ModuleNotFoundError: No module named 'prometheus_client'", True),
        ("Cannot find module 'jest'", True),
        ('npm error Missing script: "test"', True),
        ("Executable not found: mvn", True),
        ("AssertionError: expected 42", False),
    ],
)
def test_environment_failure_classification(stderr: str, expected: bool):
    result = failed_result(make_command_result("test", stderr=stderr))
    assert is_environment_failure(result) is expected


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("Exceeded timeout of 5000 ms for a test", True),
        ("Async callback was not invoked within the timeout", True),
        ("AssertionError: expected 42", False),
    ],
)
def test_intermittent_failure_classification(stderr: str, expected: bool):
    result = failed_result(make_command_result("npm test", stderr=stderr))
    assert is_potentially_intermittent(result) is expected


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
async def test_verification_environment_error_blocks_without_repair(tmp_path: Path):
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    adapter = ScriptedAdapter(Provider.OPENAI, [])
    registry.register(Provider.OPENAI, adapter)
    verifier = ScriptedVerificationRunner([])
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG)
    initial = VerificationResult(
        status=VerificationStatus.ERROR,
        groups_run=["ai-engine"],
        command_results=[
            make_command_result(
                "python -m pytest",
                exit_code=127,
                stderr="Python virtualenv interpreter for group 'ai-engine' is missing or unusable",
            )
        ],
    )

    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.attempts == 0
    assert "Python virtualenv interpreter" in (outcome.blocker_reason or "")
    assert adapter.calls == 0
    assert verifier.calls == 0


@pytest.mark.asyncio
async def test_missing_dependency_blocks_without_agent_repair(tmp_path: Path):
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    adapter = ScriptedAdapter(Provider.OPENAI, [])
    registry.register(Provider.OPENAI, adapter)
    workflow = RepairWorkflow(db, registry, ScriptedVerificationRunner([]), DEFAULT_MODELS_CONFIG)
    initial = failed_result(
        make_command_result(
            "python -m pytest",
            stderr="ModuleNotFoundError: No module named 'prometheus_client'",
        )
    )

    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.attempts == 0
    assert "missing module 'prometheus_client'" in (outcome.blocker_reason or "")
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_timeout_reproduction_pass_blocks_without_agent_repair(tmp_path: Path):
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    adapter = ScriptedAdapter(Provider.OPENAI, [])
    registry.register(Provider.OPENAI, adapter)

    class ReproducingVerifier(ScriptedVerificationRunner):
        seen_timeout: float | None = None

        async def reproduce_failure(
            self,
            run_id,
            worktree_path,
            verification_config,
            result,
            log_dir=None,
            timeout_seconds=None,
        ):
            self.seen_timeout = timeout_seconds
            result.reproduction_result = make_command_result("npm test", exit_code=0)
            result.status = VerificationStatus.INTERMITTENT
            return result

    verifier = ReproducingVerifier([])
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG)
    initial = failed_result(
        make_command_result(
            "npm test",
            stderr="Exceeded timeout of 5000 ms",
        )
    )

    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.attempts == 0
    assert "did not reproduce" in (outcome.blocker_reason or "")
    assert adapter.calls == 0
    assert verifier.seen_timeout == 300.0


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
async def test_repair_model_rejection_blocks_without_reverification(tmp_path: Path):
    """A repair CLI model rejection is surfaced instead of being ignored by the loop."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.OPENAI,
        ScriptedAdapter(
            Provider.OPENAI,
            [make_agent_result(exit_code=1, stderr="Error: Unknown model gpt-6-luna")],
        ),
    )
    verifier = ScriptedVerificationRunner([])
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG)
    initial = failed_result(make_command_result("ruff check .", stderr="E501 line too long"))

    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.blocker_reason is not None
    assert "codex update" in outcome.blocker_reason
    assert verifier.calls == 0
    assert db.get_run("run_1").state == WorkflowState.BLOCKED.value
    assert not WorktreeLock(worktree_path).is_locked()


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
    """Exhausted repairs report the actual failing command and saved log."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.OPENAI,
        ScriptedAdapter(Provider.OPENAI, [make_agent_result(), make_agent_result()]),
    )
    registry.register(
        Provider.ANTHROPIC, ScriptedAdapter(Provider.ANTHROPIC, [make_agent_result()])
    )

    failing_command = make_command_result(
        "pytest",
        stdout=(
            "FAILED ai-engine/tests/test_worker.py::test_run - AssertionError\n1 failed, 120 passed"
        ),
        exit_code=2,
    )
    unit_failure = failed_result(failing_command)
    log_path = tmp_path / "verification" / "13.stdout.log"
    log_path.parent.mkdir()
    log_path.write_text(failing_command.stdout, encoding="utf-8")
    now = datetime.now(timezone.utc)
    db.record_verification_run(
        "verification-13", "run_1", "pytest", 2, now, now, stdout_path=str(log_path)
    )
    verifier = ScriptedVerificationRunner([unit_failure, unit_failure, unit_failure])
    limits = LimitsConfig(standard_failures=2, lightweight_verification_failures=0)
    workflow = RepairWorkflow(db, registry, verifier, DEFAULT_MODELS_CONFIG, limits)

    outcome = await workflow.run(
        "run_1", worktree_path, ctx, "# Plan", task_profile, None, unit_failure
    )

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.final_tier == RepairTier.ESCALATION
    assert outcome.blocker_reason is not None
    assert "pytest" in outcome.blocker_reason
    assert "exit code 2" in outcome.blocker_reason
    assert "failed test 'ai-engine/tests/test_worker.py::test_run'" in outcome.blocker_reason
    assert str(log_path) in outcome.blocker_reason
    assert "architecture" not in outcome.blocker_reason.lower()

    run = db.get_run("run_1")
    assert run is not None
    assert run.state == "BLOCKED"


@pytest.mark.asyncio
async def test_verification_without_failed_command_reports_status(tmp_path: Path):
    """An abnormal verification result still gives a reason when no command is available."""
    db, worktree_path, ctx, task_profile = make_environment(tmp_path)
    workflow = RepairWorkflow(
        db,
        AgentAdapterRegistry(),
        ScriptedVerificationRunner([]),
        DEFAULT_MODELS_CONFIG,
    )
    initial = VerificationResult(status=VerificationStatus.ERROR)

    outcome = await workflow.run("run_1", worktree_path, ctx, "# Plan", task_profile, None, initial)

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.blocker_reason is not None
    assert "state ERROR" in outcome.blocker_reason


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


def test_agent_prompts_defer_full_verification_to_the_orchestrator():
    from agentflow.workflow.implementation import _build_implementation_prompt
    from agentflow.workflow.repair import FailureCategory, _build_repair_prompt
    from agentflow.workflow.review import _build_review_fix_prompt
    from agentflow.workflow.verification import AGENT_VERIFICATION_GUIDANCE

    profile = TaskProfile(
        stage=Stage.IMPLEMENTATION, technologies=set(), affected_layers=set(), estimated_files=1
    )
    now = datetime.now(timezone.utc)
    failing = CommandResult(
        command="mvn test", exit_code=1, stdout="", stderr="", started_at=now, completed_at=now
    )
    prompts = [
        _build_implementation_prompt("task", "plan", profile, Path("wt")),
        _build_repair_prompt("plan", profile, failing, FailureCategory.UNIT_TEST),
        _build_review_fix_prompt("plan", profile, []),
    ]
    for prompt in prompts:
        assert AGENT_VERIFICATION_GUIDANCE in prompt
    assert "not 'mvn test' in full" in prompts[1]
