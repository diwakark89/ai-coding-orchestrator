"""Fixture-repo tests for the documentation workflow (Phase 8)."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.agents.base import AdapterCapabilities, AgentRequest, AgentResult, Provider
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.config.models import DocumentationConfig, VerificationGroup
from agentflow.git.worktree import WorktreeManager
from agentflow.persistence.database import DatabaseManager
from agentflow.project.context import ProjectContext
from agentflow.routing.rules import DEFAULT_MODELS_CONFIG, DEFAULT_ROUTING_RULES
from agentflow.task.profile import Stage, TaskProfile
from agentflow.workflow.documentation import DocumentationWorkflow
from agentflow.workflow.states import WorkflowState
from agentflow.workflow.verification import VerificationResult, VerificationStatus


class ScriptedAdapter:
    """Fake AgentAdapter that writes a file into the worktree and returns a scripted result."""

    def __init__(
        self,
        provider: Provider,
        response: AgentResult,
        filename: str | None,
        content: str = "docs\n",
    ) -> None:
        self._provider = provider
        self._response = response
        self._filename = filename
        self._content = content
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
        if self._filename is not None:
            assert request.working_directory is not None
            target = request.working_directory / self._filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self._content, encoding="utf-8")
        return self._response

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


def make_result(exit_code: int = 0, text: str = "Updated docs.") -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=Provider.GOOGLE,
        model="gemini-3.8-flash",
        session_id="sess-doc-1",
        exit_code=exit_code,
        text=text,
        started_at=now,
        completed_at=now,
    )


def passed_verification() -> VerificationResult:
    return VerificationResult(status=VerificationStatus.PASSED, groups_run=[], command_results=[])


def failed_verification() -> VerificationResult:
    return VerificationResult(status=VerificationStatus.FAILED, groups_run=[], command_results=[])


async def make_environment(git_repo: Path, tmp_path: Path):
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(git_repo))
    db.create_run(run_id="run_1", project_id="proj_1", task="Add a feature")
    db.update_run_state("run_1", WorkflowState.REVIEW_APPROVED.value)

    worktree_manager = WorktreeManager(tmp_path / "worktrees")
    handle = await worktree_manager.create("demo", "run_1", git_repo)

    ctx = ProjectContext(root_path=git_repo, project_id="proj_1", project_name="demo")
    task_profile = TaskProfile(
        stage=Stage.IMPLEMENTATION, technologies=set(), affected_layers=set(), estimated_files=1
    )
    return db, ctx, task_profile, worktree_manager, handle.path


def make_workflow(db, registry, worktree_manager, verifier=None) -> DocumentationWorkflow:
    return DocumentationWorkflow(
        db,
        registry,
        worktree_manager,
        verifier or ScriptedVerificationRunner([]),
        DEFAULT_MODELS_CONFIG,
        DEFAULT_ROUTING_RULES,
    )


@pytest.mark.asyncio
async def test_documentation_disabled_skips(git_repo: Path, tmp_path: Path):
    """A disabled documentation config skips the workflow entirely."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()  # no adapter registered -- must never be invoked
    workflow = make_workflow(db, registry, worktree_manager)

    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        "diff",
        passed_verification(),
        [],
        DocumentationConfig(enabled=False, candidate_files=["docs/architecture.md"]),
        None,
    )

    assert outcome.enabled is False
    assert outcome.updated_files == []


@pytest.mark.asyncio
async def test_documentation_no_candidate_files_skips(git_repo: Path, tmp_path: Path):
    """Documentation with no configured candidate files is a no-op."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    workflow = make_workflow(db, registry, worktree_manager)

    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        "diff",
        passed_verification(),
        [],
        DocumentationConfig(enabled=True, candidate_files=[]),
        None,
    )

    assert outcome.enabled is False


@pytest.mark.asyncio
async def test_documentation_enabled_updates_candidate_file(git_repo: Path, tmp_path: Path):
    """An enabled documentation run updates the configured candidate file."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    adapter = ScriptedAdapter(Provider.GOOGLE, make_result(), "architecture.md")
    registry.register(Provider.GOOGLE, adapter)
    workflow = make_workflow(db, registry, worktree_manager)

    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        "diff",
        passed_verification(),
        [],
        DocumentationConfig(enabled=True, candidate_files=["architecture.md"]),
        None,
    )

    assert outcome.enabled is True
    assert not outcome.blocked
    assert "architecture.md" in outcome.updated_files
    assert outcome.summary_path is not None and outcome.summary_path.exists()
    assert (worktree_path / "architecture.md").exists()


@pytest.mark.asyncio
async def test_documentation_missing_candidate_file_is_handled(git_repo: Path, tmp_path: Path):
    """A candidate file that does not yet exist can be created without crashing the workflow."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    assert not (worktree_path / "new-doc.md").exists()
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.GOOGLE, ScriptedAdapter(Provider.GOOGLE, make_result(), "new-doc.md")
    )
    workflow = make_workflow(db, registry, worktree_manager)

    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        "diff",
        passed_verification(),
        [],
        DocumentationConfig(enabled=True, candidate_files=["new-doc.md"]),
        None,
    )

    assert outcome.enabled is True
    assert not outcome.blocked
    assert "new-doc.md" in outcome.updated_files


@pytest.mark.asyncio
async def test_documentation_agent_failure_blocks(git_repo: Path, tmp_path: Path):
    """A non-zero exit code from the documentation agent blocks the workflow."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.GOOGLE, ScriptedAdapter(Provider.GOOGLE, make_result(exit_code=1), None)
    )
    workflow = make_workflow(db, registry, worktree_manager)

    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        "diff",
        passed_verification(),
        [],
        DocumentationConfig(enabled=True, candidate_files=["architecture.md"]),
        None,
    )

    assert outcome.enabled is True
    assert outcome.blocked is True
    assert outcome.blocker_reason is not None


@pytest.mark.asyncio
async def test_documentation_reruns_verification_and_blocks_on_failure(
    git_repo: Path, tmp_path: Path
):
    """Documentation changes that break verification block the workflow."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.GOOGLE, ScriptedAdapter(Provider.GOOGLE, make_result(), "architecture.md")
    )
    verifier = ScriptedVerificationRunner([failed_verification()])
    workflow = make_workflow(db, registry, worktree_manager, verifier)

    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        "diff",
        passed_verification(),
        [],
        DocumentationConfig(enabled=True, candidate_files=["architecture.md"]),
        {"py": VerificationGroup(detect=[], commands=[])},  # non-empty -> triggers a re-verify
    )

    assert outcome.blocked is True
    assert "verification" in (outcome.blocker_reason or "").lower()
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_documentation_prompt_includes_final_context(git_repo: Path, tmp_path: Path):
    """The documentation prompt includes the plan, TaskProfile, final diff, and review findings."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    adapter = ScriptedAdapter(Provider.GOOGLE, make_result(), "architecture.md")
    registry.register(Provider.GOOGLE, adapter)
    workflow = make_workflow(db, registry, worktree_manager)

    await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# The Plan Objective",
        task_profile,
        "diff --git a/x b/x\n+added line",
        passed_verification(),
        [],
        DocumentationConfig(enabled=True, candidate_files=["architecture.md"]),
        None,
    )

    assert adapter.received_request is not None
    prompt = adapter.received_request.prompt
    assert "The Plan Objective" in prompt
    assert "added line" in prompt
    assert "architecture.md" in prompt
