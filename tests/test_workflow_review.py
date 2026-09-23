"""Fixture-repo tests for the independent review workflow (Phase 7)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.agents.base import AdapterCapabilities, AgentRequest, AgentResult, Provider
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.config.models import LimitsConfig
from agentflow.git.worktree import WorktreeManager
from agentflow.persistence.database import DatabaseManager
from agentflow.project.context import ProjectContext
from agentflow.routing.complexity import ComplexityLevel
from agentflow.routing.decision import RoutingDecision
from agentflow.routing.rules import (
    DEFAULT_MODELS_CONFIG,
    DEFAULT_ROUTING_RULES,
    ModelRef,
    RoleOverride,
)
from agentflow.task.profile import Stage, TaskProfile
from agentflow.workflow.review import ReviewFinding, ReviewWorkflow, mandatory_findings
from agentflow.workflow.states import WorkflowState
from agentflow.workflow.verification import VerificationResult, VerificationStatus


class ScriptedAdapter:
    """Fake AgentAdapter returning a pre-scripted sequence of AgentResults."""

    def __init__(self, provider: Provider, responses: list[AgentResult]) -> None:
        self._provider = provider
        self._responses = list(responses)
        self.calls: list[AgentRequest] = []

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=False,
            supports_read_only_mode=True,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    async def start(self, request: AgentRequest) -> AgentResult:
        self.calls.append(request)
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


def make_result(text: str, session_id: str = "sess-1") -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=Provider.GOOGLE,
        model="gemini-3.8-flash",
        session_id=session_id,
        exit_code=0,
        text=text,
        started_at=now,
        completed_at=now,
    )


def review_payload(status: str, findings: list[dict[str, object]] | None = None) -> str:
    return json.dumps({"status": status, "findings": findings or []})


def make_implementation_decision() -> RoutingDecision:
    return RoutingDecision(
        stage=Stage.IMPLEMENTATION,
        provider=Provider.OPENAI,
        model="GPT-6 Luna",
        role="implementation.lightweight",
        matched_rule="low-complexity",
        reason="test",
        complexity_score=0,
        complexity=ComplexityLevel.LOW,
        risk_flags=[],
    )


def passed_verification() -> VerificationResult:
    return VerificationResult(status=VerificationStatus.PASSED, groups_run=[], command_results=[])


async def make_environment(git_repo: Path, tmp_path: Path, **task_profile_flags: object):
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(git_repo))
    db.create_run(run_id="run_1", project_id="proj_1", task="Add a feature")
    # ReviewWorkflow is only ever invoked after verification passes and the run enters REVIEWING.
    db.update_run_state("run_1", WorkflowState.REVIEWING.value)

    worktree_manager = WorktreeManager(tmp_path / "worktrees")
    handle = await worktree_manager.create("demo", "run_1", git_repo)
    (handle.path / "feature.py").write_text("print('hi')\n", encoding="utf-8")

    ctx = ProjectContext(root_path=git_repo, project_id="proj_1", project_name="demo")
    task_profile = TaskProfile(
        stage=Stage.IMPLEMENTATION,
        technologies=set(),
        affected_layers=set(),
        estimated_files=1,
        **task_profile_flags,  # type: ignore[arg-type]
    )
    return db, ctx, task_profile, worktree_manager, handle.path


# --- 1. mandatory_findings -----------------------------------------------------------------


def _finding(severity: str) -> dict[str, object]:
    return {
        "severity": severity,
        "category": "test",
        "file": "a.py",
        "line": 1,
        "problem": "x",
        "recommendation": "y",
    }


def test_critical_and_high_are_always_mandatory():
    """CRITICAL and HIGH findings are always mandatory, regardless of configuration."""
    findings = [
        ReviewFinding(**_finding("CRITICAL")),  # type: ignore[arg-type]
        ReviewFinding(**_finding("HIGH")),  # type: ignore[arg-type]
    ]
    assert mandatory_findings(findings, medium_is_mandatory=False) == findings


def test_low_is_never_mandatory():
    """LOW findings never block approval."""
    findings = [ReviewFinding(**_finding("LOW"))]  # type: ignore[arg-type]
    assert mandatory_findings(findings, medium_is_mandatory=True) == []


def test_medium_is_configurable():
    """MEDIUM findings are mandatory only when explicitly configured."""
    findings = [ReviewFinding(**_finding("MEDIUM"))]  # type: ignore[arg-type]
    assert mandatory_findings(findings, medium_is_mandatory=False) == []
    assert mandatory_findings(findings, medium_is_mandatory=True) == findings


# --- 2. Reviewer selection -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_reviewer_is_gemini_with_no_findings(git_repo: Path, tmp_path: Path):
    """A task with no risk flags routes to the default reviewer (Gemini) and approves cleanly."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.GOOGLE, ScriptedAdapter(Provider.GOOGLE, [make_result(review_payload("APPROVED"))])
    )

    workflow = ReviewWorkflow(
        db,
        registry,
        ScriptedVerificationRunner([]),
        worktree_manager,
        DEFAULT_MODELS_CONFIG,
        DEFAULT_ROUTING_RULES,
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.state == WorkflowState.REVIEW_APPROVED
    assert outcome.reviewer_model == "Gemini 3.8 Flash"
    assert outcome.cycles == 1
    assert outcome.review_path is not None and outcome.review_path.exists()
    assert outcome.findings_path is not None and outcome.findings_path.exists()


@pytest.mark.asyncio
async def test_deep_review_selected_for_authentication_risk(git_repo: Path, tmp_path: Path):
    """A task touching authentication routes to the deep reviewer (Claude Sonnet 5)."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path, authentication=True
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.ANTHROPIC,
        ScriptedAdapter(Provider.ANTHROPIC, [make_result(review_payload("APPROVED"))]),
    )

    workflow = ReviewWorkflow(
        db,
        registry,
        ScriptedVerificationRunner([]),
        worktree_manager,
        DEFAULT_MODELS_CONFIG,
        DEFAULT_ROUTING_RULES,
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.state == WorkflowState.REVIEW_APPROVED
    assert outcome.reviewer_model == "Claude Sonnet 5"


@pytest.mark.asyncio
async def test_role_override_for_review_bypasses_default_routing(git_repo: Path, tmp_path: Path):
    """A RoleOverride targeting REVIEW routes there regardless of task risk flags."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.OPENAI,
        ScriptedAdapter(Provider.OPENAI, [make_result(review_payload("APPROVED"))]),
    )
    override = RoleOverride(
        overrides={Stage.REVIEW: ModelRef(provider="openai", model="GPT-6 Sol")}
    )

    workflow = ReviewWorkflow(
        db,
        registry,
        ScriptedVerificationRunner([]),
        worktree_manager,
        DEFAULT_MODELS_CONFIG,
        DEFAULT_ROUTING_RULES,
        role_override=override,
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.state == WorkflowState.REVIEW_APPROVED
    assert outcome.reviewer_model == "GPT-6 Sol"


@pytest.mark.asyncio
async def test_architecture_review_selected_for_architecture_change(git_repo: Path, tmp_path: Path):
    """A task with architecture_change=true routes to the architecture reviewer (Opus 5.5)."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path, architecture_change=True
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.ANTHROPIC,
        ScriptedAdapter(Provider.ANTHROPIC, [make_result(review_payload("APPROVED"))]),
    )

    workflow = ReviewWorkflow(
        db,
        registry,
        ScriptedVerificationRunner([]),
        worktree_manager,
        DEFAULT_MODELS_CONFIG,
        DEFAULT_ROUTING_RULES,
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.reviewer_model == "Claude Opus 5.5"


# --- 3. Mandatory-fix cycle ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critical_finding_triggers_fix_then_reapproves(git_repo: Path, tmp_path: Path):
    """A CRITICAL finding sends a fix to the implementation worker, re-verifies, and re-reviews."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    google = ScriptedAdapter(
        Provider.GOOGLE,
        [
            make_result(review_payload("CHANGES_REQUIRED", [_finding("CRITICAL")])),
            make_result(review_payload("APPROVED")),
        ],
    )
    openai = ScriptedAdapter(Provider.OPENAI, [make_result(review_payload("APPROVED"))])
    registry.register(Provider.GOOGLE, google)
    registry.register(Provider.OPENAI, openai)

    verifier = ScriptedVerificationRunner([passed_verification()])
    workflow = ReviewWorkflow(
        db, registry, verifier, worktree_manager, DEFAULT_MODELS_CONFIG, DEFAULT_ROUTING_RULES
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.state == WorkflowState.REVIEW_APPROVED
    assert outcome.cycles == 2
    assert len(openai.calls) == 1
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_low_finding_does_not_block_approval(git_repo: Path, tmp_path: Path):
    """A LOW-severity finding is informational only and does not trigger a fix cycle."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.GOOGLE,
        ScriptedAdapter(
            Provider.GOOGLE, [make_result(review_payload("APPROVED", [_finding("LOW")]))]
        ),
    )

    workflow = ReviewWorkflow(
        db,
        registry,
        ScriptedVerificationRunner([]),
        worktree_manager,
        DEFAULT_MODELS_CONFIG,
        DEFAULT_ROUTING_RULES,
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.state == WorkflowState.REVIEW_APPROVED
    assert outcome.cycles == 1
    assert len(outcome.findings) == 1


@pytest.mark.asyncio
async def test_max_review_cycles_enforced(git_repo: Path, tmp_path: Path):
    """When mandatory findings never clear, the review loop is BLOCKED after the fix budget."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    always_critical = review_payload("CHANGES_REQUIRED", [_finding("CRITICAL")])
    google = ScriptedAdapter(
        Provider.GOOGLE, [make_result(always_critical), make_result(always_critical)]
    )
    openai = ScriptedAdapter(Provider.OPENAI, [make_result(review_payload("APPROVED"))])
    registry.register(Provider.GOOGLE, google)
    registry.register(Provider.OPENAI, openai)

    verifier = ScriptedVerificationRunner([passed_verification()])
    limits = LimitsConfig(review_fix_cycles=1)
    workflow = ReviewWorkflow(
        db,
        registry,
        verifier,
        worktree_manager,
        DEFAULT_MODELS_CONFIG,
        DEFAULT_ROUTING_RULES,
        limits,
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.blocker_reason is not None
    assert len(openai.calls) == 1

    run = db.get_run("run_1")
    assert run is not None
    assert run.state == "BLOCKED"


@pytest.mark.asyncio
async def test_review_fix_breaking_verification_blocks(git_repo: Path, tmp_path: Path):
    """If a review fix breaks verification, the run is blocked rather than re-reviewed."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.GOOGLE,
        ScriptedAdapter(
            Provider.GOOGLE, [make_result(review_payload("CHANGES_REQUIRED", [_finding("HIGH")]))]
        ),
    )
    registry.register(Provider.OPENAI, ScriptedAdapter(Provider.OPENAI, [make_result("fixed")]))

    failing_verification = VerificationResult(
        status=VerificationStatus.FAILED, groups_run=[], command_results=[]
    )
    verifier = ScriptedVerificationRunner([failing_verification])
    workflow = ReviewWorkflow(
        db, registry, verifier, worktree_manager, DEFAULT_MODELS_CONFIG, DEFAULT_ROUTING_RULES
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.state == WorkflowState.BLOCKED
    assert "broke verification" in (outcome.blocker_reason or "")


# --- 4. Malformed output retry ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_reviewer_output_is_retried(git_repo: Path, tmp_path: Path):
    """Malformed reviewer output triggers a bounded retry before succeeding."""
    db, ctx, task_profile, worktree_manager, worktree_path = await make_environment(
        git_repo, tmp_path
    )
    registry = AgentAdapterRegistry()
    registry.register(
        Provider.GOOGLE,
        ScriptedAdapter(
            Provider.GOOGLE, [make_result("not json"), make_result(review_payload("APPROVED"))]
        ),
    )

    workflow = ReviewWorkflow(
        db,
        registry,
        ScriptedVerificationRunner([]),
        worktree_manager,
        DEFAULT_MODELS_CONFIG,
        DEFAULT_ROUTING_RULES,
    )
    outcome = await workflow.run(
        "run_1",
        worktree_path,
        ctx,
        "# Plan",
        task_profile,
        None,
        make_implementation_decision(),
        passed_verification(),
    )

    assert outcome.state == WorkflowState.REVIEW_APPROVED
