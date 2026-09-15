"""Unit tests for the interactive Claude planning workflow (Phase 3)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.agents.base import (
    AdapterCapabilities,
    AgentRequest,
    AgentResult,
    Provider,
)
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.persistence.database import DatabaseManager
from agentflow.persistence.models import RunStatus
from agentflow.project.context import ProjectContext
from agentflow.task.profile import Stage
from agentflow.ui.approval import ApprovalDecision
from agentflow.workflow.planning import PlanningWorkflow
from agentflow.workflow.states import WorkflowState


class ScriptedAdapter:
    """Fake AgentAdapter returning a pre-scripted sequence of AgentResults."""

    def __init__(self, responses: list[AgentResult]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, AgentRequest, str | None]] = []

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=True,
            supports_read_only_mode=True,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    async def start(self, request: AgentRequest) -> AgentResult:
        self.calls.append(("start", request, None))
        return self._responses.pop(0)

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        self.calls.append(("resume", request, session_id))
        return self._responses.pop(0)


def make_result(
    text: str,
    session_id: str | None = "sess-1",
    exit_code: int = 0,
    stderr: str = "",
) -> AgentResult:
    """Build a minimal AgentResult carrying the given planner turn text."""
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=Provider.ANTHROPIC,
        model="sonnet",
        session_id=session_id,
        exit_code=exit_code,
        text=text,
        started_at=now,
        completed_at=now,
        stderr=stderr,
    )


def plan_ready_payload(**task_profile_overrides: object) -> str:
    """Build JSON text for a plan_ready planner turn."""
    task_profile = {
        "stage": "IMPLEMENTATION",
        "technologies": ["python"],
        "affected_layers": ["backend"],
        "estimated_files": 3,
        **task_profile_overrides,
    }
    return json.dumps(
        {
            "status": "plan_ready",
            "plan_markdown": "# Plan\n\nObjective: Do the thing.\n",
            "task_profile": task_profile,
        }
    )


def make_workflow(
    tmp_path: Path,
    adapter: ScriptedAdapter,
    approval_prompt=None,
    question_prompt=None,
    feedback_prompt=None,
    max_malformed_retries: int = 2,
    max_turns: int = 12,
) -> tuple[PlanningWorkflow, DatabaseManager, ProjectContext]:
    """Construct a PlanningWorkflow wired to an in-memory DB and a scripted adapter."""
    db = DatabaseManager(":memory:")
    db.initialize()
    registry = AgentAdapterRegistry()
    registry.register(Provider.ANTHROPIC, adapter)
    ctx = ProjectContext(root_path=tmp_path, project_id="proj-1", project_name="demo-project")
    workflow = PlanningWorkflow(
        db_manager=db,
        agent_registry=registry,
        question_prompt=question_prompt or (lambda q: "unused"),
        approval_prompt=approval_prompt or (lambda: ApprovalDecision.APPROVE),
        feedback_prompt=feedback_prompt or (lambda: "unused"),
        max_malformed_retries=max_malformed_retries,
        max_turns=max_turns,
    )
    return workflow, db, ctx


@pytest.mark.asyncio
async def test_plan_ready_on_first_turn_reaches_task_classified(tmp_path: Path):
    """A planner that produces a valid plan on the first turn reaches TASK_CLASSIFIED."""
    adapter = ScriptedAdapter([make_result(plan_ready_payload())])
    workflow, db, ctx = make_workflow(tmp_path, adapter)

    outcome = await workflow.run("Add a health check endpoint", ctx)

    assert outcome.state == WorkflowState.TASK_CLASSIFIED
    assert outcome.task_profile is not None
    assert outcome.task_profile.stage == Stage.IMPLEMENTATION
    assert outcome.plan_path is not None and outcome.plan_path.exists()
    assert outcome.task_profile_path is not None and outcome.task_profile_path.exists()

    run = db.get_run(outcome.run_id)
    assert run is not None
    assert run.state == "TASK_CLASSIFIED"
    assert run.status == RunStatus.COMPLETED.value

    transitions = [t.to_state for t in db.list_state_transitions(outcome.run_id)]
    assert transitions == [
        "PROJECT_READY",
        "PLANNING",
        "PLAN_READY",
        "PLAN_APPROVED",
        "TASK_CLASSIFIED",
    ]

    sessions = db.list_agent_sessions(outcome.run_id)
    assert len(sessions) == 1
    assert sessions[0].model == "Claude Sonnet 5"

    task_md = (ctx.root_path / ".ai-orchestrator" / "runs" / outcome.run_id / "task.md").read_text()
    assert "Add a health check endpoint" in task_md


@pytest.mark.asyncio
async def test_question_loop_records_decision_and_resumes_session(tmp_path: Path):
    """Planner questions are surfaced to the user and answers are persisted then relayed back."""
    question_payload = json.dumps(
        {
            "status": "questions",
            "questions": [
                {
                    "question": "REST or GraphQL?",
                    "options": ["REST", "GraphQL"],
                    "recommended": "REST",
                }
            ],
        }
    )
    adapter = ScriptedAdapter(
        [
            make_result(question_payload, session_id="sess-abc"),
            make_result(plan_ready_payload(), session_id="sess-abc"),
        ]
    )
    workflow, db, ctx = make_workflow(
        tmp_path, adapter, question_prompt=lambda q: q.recommended or q.options[0]
    )

    outcome = await workflow.run("Build the public API", ctx)

    assert outcome.state == WorkflowState.TASK_CLASSIFIED

    decisions = db.list_decisions(outcome.run_id)
    assert len(decisions) == 1
    assert decisions[0].question == "REST or GraphQL?"
    assert decisions[0].answer == "REST"

    assert adapter.calls[0][0] == "start"
    assert adapter.calls[1] == ("resume", adapter.calls[1][1], "sess-abc")

    transitions = [t.to_state for t in db.list_state_transitions(outcome.run_id)]
    assert "WAITING_FOR_USER" in transitions


@pytest.mark.asyncio
async def test_opus_escalation_switches_model_for_next_turn(tmp_path: Path):
    """An escalation signal switches subsequent planner turns to Claude Opus 5."""
    escalation_payload = json.dumps(
        {
            "status": "questions",
            "questions": [{"question": "Confirm new service?", "options": ["Yes", "No"]}],
            "escalation": {
                "flags": ["architecture_change", "new_service"],
                "reason": "New billing service",
            },
        }
    )
    adapter = ScriptedAdapter(
        [
            make_result(escalation_payload, session_id="sess-esc"),
            make_result(plan_ready_payload(architecture_change=True), session_id="sess-esc"),
        ]
    )
    workflow, db, ctx = make_workflow(tmp_path, adapter, question_prompt=lambda q: "Yes")

    outcome = await workflow.run("Split billing into its own service", ctx)

    assert outcome.state == WorkflowState.TASK_CLASSIFIED
    second_request = adapter.calls[1][1]
    assert second_request.model == "Claude Opus 5"

    decisions = db.list_decisions(outcome.run_id)
    reasons = [d.answer for d in decisions if d.question == "architecture_escalation"]
    assert len(reasons) == 1
    assert "new_service" in reasons[0] or "architecture_change" in reasons[0]

    sessions = db.list_agent_sessions(outcome.run_id)
    assert sessions[-1].model == "Claude Opus 5"


@pytest.mark.asyncio
async def test_malformed_output_is_retried_then_succeeds(tmp_path: Path):
    """Malformed structured output triggers a bounded retry before succeeding."""
    adapter = ScriptedAdapter(
        [
            make_result("not json at all", session_id="sess-retry"),
            make_result(plan_ready_payload(), session_id="sess-retry"),
        ]
    )
    workflow, db, ctx = make_workflow(tmp_path, adapter, max_malformed_retries=2)

    outcome = await workflow.run("Add caching", ctx)

    assert outcome.state == WorkflowState.TASK_CLASSIFIED
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_malformed_output_exhausts_retries_and_blocks(tmp_path: Path):
    """Persistently malformed output blocks the run after bounded retries are exhausted."""
    adapter = ScriptedAdapter(
        [
            make_result("garbage 1", session_id="sess-bad"),
            make_result("garbage 2", session_id="sess-bad"),
        ]
    )
    workflow, db, ctx = make_workflow(tmp_path, adapter, max_malformed_retries=1)

    outcome = await workflow.run("Add caching", ctx)

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.blocker_reason is not None
    assert "malformed" in outcome.blocker_reason.lower()

    run = db.get_run(outcome.run_id)
    assert run is not None
    assert run.status == RunStatus.BLOCKED.value


@pytest.mark.asyncio
async def test_planner_blocked_status_ends_run_blocked(tmp_path: Path):
    """An explicit 'blocked' planner status ends the run in the BLOCKED workflow state."""
    blocked_payload = json.dumps({"status": "blocked", "blocker_reason": "Task is impossible."})
    adapter = ScriptedAdapter([make_result(blocked_payload)])
    workflow, db, ctx = make_workflow(tmp_path, adapter)

    outcome = await workflow.run("Do the impossible", ctx)

    assert outcome.state == WorkflowState.BLOCKED
    assert outcome.blocker_reason == "Task is impossible."


@pytest.mark.asyncio
async def test_nonzero_exit_code_blocks_run(tmp_path: Path):
    """A non-zero planner CLI exit code blocks the run rather than being treated as a plan."""
    adapter = ScriptedAdapter([make_result("", exit_code=1, stderr="claude: not authenticated")])
    workflow, db, ctx = make_workflow(tmp_path, adapter)

    outcome = await workflow.run("Add a feature", ctx)

    assert outcome.state == WorkflowState.BLOCKED
    assert "not authenticated" in (outcome.blocker_reason or "")


@pytest.mark.asyncio
async def test_cancel_decision_ends_run_cancelled(tmp_path: Path):
    """Cancelling at plan approval ends the run in CANCELLED without producing artifacts."""
    adapter = ScriptedAdapter([make_result(plan_ready_payload())])
    workflow, db, ctx = make_workflow(
        tmp_path, adapter, approval_prompt=lambda: ApprovalDecision.CANCEL
    )

    outcome = await workflow.run("Add a feature", ctx)

    assert outcome.state == WorkflowState.CANCELLED
    assert outcome.plan_path is None
    run = db.get_run(outcome.run_id)
    assert run is not None
    assert run.status == RunStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_continue_planning_requests_feedback_then_approves(tmp_path: Path):
    """Choosing 'continue' sends feedback back to the planner before final approval."""
    decisions_iter = iter([ApprovalDecision.CONTINUE_PLANNING, ApprovalDecision.APPROVE])
    adapter = ScriptedAdapter(
        [
            make_result(plan_ready_payload(), session_id="sess-cont"),
            make_result(plan_ready_payload(estimated_files=5), session_id="sess-cont"),
        ]
    )
    workflow, db, ctx = make_workflow(
        tmp_path,
        adapter,
        approval_prompt=lambda: next(decisions_iter),
        feedback_prompt=lambda: "Please also add tests.",
    )

    outcome = await workflow.run("Add a feature", ctx)

    assert outcome.state == WorkflowState.TASK_CLASSIFIED
    assert outcome.task_profile is not None
    assert outcome.task_profile.estimated_files == 5

    decisions = db.list_decisions(outcome.run_id)
    assert any(d.question == "Plan feedback requested" for d in decisions)
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_plan_ready_missing_task_profile_blocks_run(tmp_path: Path):
    """A plan_ready turn missing task_profile is treated as malformed and blocks the run."""
    incomplete_payload = json.dumps({"status": "plan_ready", "plan_markdown": "# Plan\n"})
    adapter = ScriptedAdapter([make_result(incomplete_payload)])
    workflow, db, ctx = make_workflow(tmp_path, adapter, max_malformed_retries=0)

    outcome = await workflow.run("Add a feature", ctx)

    assert outcome.state == WorkflowState.BLOCKED
