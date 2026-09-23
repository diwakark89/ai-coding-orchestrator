"""Interactive planning workflow.

Drives a multi-turn conversation with the configured planner CLI (Claude by default, but any
provider `routing.yaml`'s `models.planner.*` resolves to -- see `PlanningWorkflow`), escalating
to a stronger model for architecture-sensitive work, persists every question/answer and state
transition, and produces the two planner output artifacts: a human-readable `approved-plan.md`
and a machine-readable `task-profile.json` (validated against `TaskProfile`).
"""

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agentflow.agents.base import (
    AgentAdapter,
    AgentRequest,
    AgentResult,
    AgentRole,
    describe_agent_failure,
    is_authentication_failure,
)
from agentflow.agents.parser import parse_model_into_schema
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.errors import (
    AdapterAuthenticationError,
    PlanningBlockedError,
    StructuredParsingError,
)
from agentflow.observability.events import record_agent_completed, record_agent_started
from agentflow.persistence.database import DatabaseManager
from agentflow.persistence.models import DecisionRecord, RunStatus
from agentflow.project.context import ProjectContext
from agentflow.routing.rules import ModelRef, ModelsConfig, RoutingRulesConfig
from agentflow.task.profile import TaskProfile
from agentflow.ui.approval import ApprovalDecision, ask_plan_approval, ask_plan_feedback
from agentflow.ui.console import ConsoleUI
from agentflow.ui.questions import ask_question
from agentflow.workflow.states import WorkflowState, validate_transition

# TDD §14.1 / phased-implementation-plan.md Phase 3 Step 6.
PLAN_REQUIRED_SECTIONS: tuple[str, ...] = (
    "Objective",
    "Existing Behavior",
    "Required Behavior",
    "Architecture Impact",
    "Data Changes",
    "API Changes",
    "Implementation Steps",
    "Validation",
    "Authorization",
    "Concurrency",
    "Idempotency",
    "Tests",
    "Documentation",
    "Risks",
    "Acceptance Criteria",
)


class PlannerStatus(str, Enum):
    """Status reported by the planner in each structured turn."""

    QUESTIONS = "questions"
    PLAN_READY = "plan_ready"
    BLOCKED = "blocked"


class PlannerQuestion(BaseModel):
    """A single clarifying question raised by the planner."""

    model_config = ConfigDict(extra="ignore")

    question: str
    options: list[str] = Field(default_factory=list)
    recommended: str | None = None


class PlannerEscalation(BaseModel):
    """Architecture escalation signal raised by the planner."""

    model_config = ConfigDict(extra="ignore")

    flags: list[str] = Field(default_factory=list)
    reason: str = ""


class PlannerTurn(BaseModel):
    """Structured wire-protocol response expected from the planner CLI on every turn."""

    model_config = ConfigDict(extra="forbid")

    status: PlannerStatus
    questions: list[PlannerQuestion] = Field(default_factory=list)
    plan_markdown: str | None = None
    task_profile: TaskProfile | None = None
    escalation: PlannerEscalation | None = None
    blocker_reason: str | None = None


@dataclass
class PlanningOutcome:
    """Terminal result of a planning workflow run."""

    run_id: str
    state: WorkflowState
    plan_markdown: str | None = None
    task_profile: TaskProfile | None = None
    plan_path: Path | None = None
    task_profile_path: Path | None = None
    blocker_reason: str | None = None


def _build_initial_prompt(
    task_description: str, project_context: ProjectContext, architecture_flags: list[str]
) -> str:
    """Build the first-turn prompt instructing the planner on its role and output contract."""
    schema_json = json.dumps(TaskProfile.model_json_schema())
    sections = ", ".join(PLAN_REQUIRED_SECTIONS)
    escalation_flags = ", ".join(sorted(architecture_flags))
    return (
        "You are AgentFlow's planning agent. Operate strictly read-only: do not edit any files.\n\n"
        f"Repository root: {project_context.root_path}\n"
        f"Project: {project_context.project_name}\n\n"
        f"Requested change:\n{task_description}\n\n"
        "Inspect the repository as needed to understand existing architecture before answering. "
        "Identify ambiguity that materially affects implementation and ask only material "
        "questions. When multiple valid approaches exist, present the alternatives and "
        "recommend one. Do not produce an implementation yourself.\n\n"
        "Respond with a single JSON object and nothing else (no prose outside the JSON), "
        "matching exactly one of the following shapes:\n\n"
        "1) To ask clarifying questions before proceeding:\n"
        '{"status": "questions", "questions": [{"question": "...", '
        '"options": ["...", "..."], "recommended": "..."}]}\n\n'
        "2) To report that planning cannot proceed (e.g. the request is impossible or "
        "contradicts the repository):\n"
        '{"status": "blocked", "blocker_reason": "..."}\n\n'
        "3) Once you have enough information, to produce the final plan:\n"
        '{"status": "plan_ready", '
        f'"plan_markdown": "<markdown with required sections: {sections}>", '
        '"task_profile": {<TaskProfile JSON matching the schema below>}, '
        '"escalation": {"flags": [...], "reason": "..."}}\n\n'
        'The "escalation" field must be omitted unless the task involves one or more of: '
        f"{escalation_flags}.\n\n"
        "TaskProfile JSON schema (boolean fields default to false when omitted; stage, "
        "technologies, affected_layers, and estimated_files are required):\n"
        f"{schema_json}\n"
    )


def _build_answers_prompt(qa_pairs: list[tuple[str, str]]) -> str:
    """Build a follow-up prompt relaying the user's answers back to the planner."""
    lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
    return (
        f"The user answered your question(s):\n\n{lines}\n\n"
        "Continue planning. Respond again with a single JSON object in one of the three shapes "
        "described previously (questions / blocked / plan_ready)."
    )


def _build_feedback_prompt(feedback: str) -> str:
    """Build a follow-up prompt relaying user feedback on a not-yet-approved plan."""
    return (
        f"The user reviewed the plan and requested changes before approving:\n{feedback}\n\n"
        "Revise your plan accordingly. Respond again with a single JSON object in one of the "
        "three shapes described previously (questions / blocked / plan_ready)."
    )


def _build_recovery_prompt(
    task_description: str,
    decisions: list[DecisionRecord],
    continuing_session: bool,
) -> str:
    """Reconstruct enough planning context to recover after a local process crash."""
    prior_decisions = (
        "\n".join(f"- {decision.question}: {decision.answer}" for decision in decisions)
        or "- (no prior answers or decisions were persisted)"
    )
    mode = (
        "Continue the existing CLI session."
        if continuing_session
        else "The prior CLI session is unavailable; reconstruct the plan from this context."
    )
    return (
        "AgentFlow's local process was interrupted while planning. "
        f"{mode}\n\n"
        f"Requested change:\n{task_description}\n\n"
        f"Persisted planning decisions:\n{prior_decisions}\n\n"
        "Inspect the repository again if needed. Continue with the next appropriate structured "
        "planning response: questions, blocked, or plan_ready. Return only the required JSON "
        "object; do not edit any files."
    )


def _build_retry_prompt(error: str) -> str:
    """Build a follow-up prompt asking the planner to correct malformed structured output."""
    return (
        "Your previous response could not be parsed as valid JSON matching the required "
        f"schema. Parsing error: {error}\n\n"
        "Respond again with ONLY a single valid JSON object matching one of the three shapes "
        "described previously. Do not include any explanation outside the JSON."
    )


class PlanningWorkflow:
    """Coordinates interactive planning: question loop, architecture escalation, plan approval,
    and TaskProfile generation. The planner's provider/model come from `models_config`/
    `routing_rules` (routing.yaml's `models.planner.*`/`routing.planning.*`), not a hardcoded
    provider."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        agent_registry: AgentAdapterRegistry,
        models_config: ModelsConfig,
        routing_rules: RoutingRulesConfig,
        console_ui: ConsoleUI | None = None,
        question_prompt: Callable[[PlannerQuestion], str] | None = None,
        approval_prompt: Callable[[], ApprovalDecision] | None = None,
        feedback_prompt: Callable[[], str] | None = None,
        max_malformed_retries: int = 2,
        max_turns: int = 12,
        retired_models: Mapping[str, str] | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.agent_registry = agent_registry
        self.models_config = models_config
        self.routing_rules = routing_rules
        self.retired_models = retired_models
        self.ui = console_ui or ConsoleUI()
        self.question_prompt: Callable[[PlannerQuestion], str] = (
            question_prompt or self._default_question_prompt
        )
        self.approval_prompt: Callable[[], ApprovalDecision] = (
            approval_prompt or self._default_approval_prompt
        )
        self.feedback_prompt: Callable[[], str] = feedback_prompt or self._default_feedback_prompt
        self.max_malformed_retries = max_malformed_retries
        self.max_turns = max_turns

    @property
    def _default_model_ref(self) -> ModelRef:
        """The configured planner model/provider for a fresh (non-escalated) turn."""
        return self.models_config.resolve(
            self.routing_rules.planning.default, retired=self.retired_models
        )

    @property
    def _architecture_model_ref(self) -> ModelRef:
        """The configured planner model/provider to escalate to for architecture-sensitive work."""
        return self.models_config.resolve(
            self.routing_rules.planning.architecture, retired=self.retired_models
        )

    def _default_question_prompt(self, question: PlannerQuestion) -> str:
        """Default terminal prompt for a planner question."""
        return ask_question(question, self.ui.console)

    def _default_approval_prompt(self) -> ApprovalDecision:
        """Default terminal prompt for plan approval."""
        return ask_plan_approval(self.ui.console)

    def _default_feedback_prompt(self) -> str:
        """Default terminal prompt for plan revision feedback."""
        return ask_plan_feedback(self.ui.console)

    async def run(
        self,
        task_description: str,
        project_context: ProjectContext,
        run_id: str | None = None,
    ) -> PlanningOutcome:
        """Execute planning end to end: create the run, plan interactively, and classify it."""
        run_id = run_id or f"RUN-{uuid.uuid4().hex[:8].upper()}"

        self.db_manager.upsert_project(
            project_id=project_context.project_id,
            name=project_context.project_name,
            repository_path=str(project_context.root_path),
        )
        self.db_manager.create_run(run_id, project_context.project_id, task_description)
        self.db_manager.update_run_status(run_id, RunStatus.RUNNING.value)
        self._write_artifact(project_context, run_id, "task.md", f"# Task\n\n{task_description}\n")

        self._transition(run_id, WorkflowState.PROJECT_READY, "Project discovered")
        self._transition(run_id, WorkflowState.PLANNING, "Planning started")

        try:
            return await self._planning_loop(run_id, task_description, project_context)
        except PlanningBlockedError as e:
            self._transition(run_id, WorkflowState.BLOCKED, str(e))
            self.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)
            self.ui.print_error(f"Planning blocked: {e}")
            return PlanningOutcome(
                run_id=run_id, state=WorkflowState.BLOCKED, blocker_reason=str(e)
            )

    async def resume(
        self,
        run_id: str,
        task_description: str,
        project_context: ProjectContext,
        current_state: WorkflowState,
    ) -> PlanningOutcome:
        """Resume an interrupted planning run from its persisted planner session when possible.

        If the prior CLI session is unavailable, reconstruct the conversation from the task and
        persisted user decisions, then start a fresh planner session and record that recovery.
        """
        if current_state not in {
            WorkflowState.PROJECT_READY,
            WorkflowState.PLANNING,
            WorkflowState.WAITING_FOR_USER,
            WorkflowState.PLAN_READY,
        }:
            raise PlanningBlockedError(
                f"Planning recovery is not valid from state {current_state.value}."
            )

        self._write_artifact(project_context, run_id, "task.md", f"# Task\n\n{task_description}\n")
        if current_state != WorkflowState.PLANNING:
            self._transition(run_id, WorkflowState.PLANNING, "Resuming interrupted planning")

        prior_sessions = self.db_manager.list_agent_sessions(run_id)
        prior_planner_sessions = [
            session
            for session in prior_sessions
            if session.stage == WorkflowState.PLANNING.value and session.cli_session_id
        ]
        prior_session = prior_planner_sessions[-1] if prior_planner_sessions else None
        initial_ref = (
            ModelRef(provider=prior_session.provider, model=prior_session.model)
            if prior_session is not None
            else self._default_model_ref
        )
        adapter = self.agent_registry.get(initial_ref.provider_enum)
        session_id = (
            prior_session.cli_session_id
            if prior_session is not None and adapter.capabilities.supports_resume
            else None
        )
        role = (
            AgentRole.ARCHITECTURE_PLANNER
            if initial_ref == self._architecture_model_ref
            else AgentRole.DEFAULT_PLANNER
        )
        decisions = self.db_manager.list_decisions(run_id)
        prompt = _build_recovery_prompt(task_description, decisions, session_id is not None)
        recovery_kind = (
            "resumed CLI session" if session_id else "reconstructed context in new CLI session"
        )
        self.db_manager.record_decision(
            str(uuid.uuid4()), run_id, "planning_recovery", recovery_kind
        )

        try:
            return await self._planning_loop(
                run_id,
                task_description,
                project_context,
                initial_prompt=prompt,
                initial_session_id=session_id,
                initial_ref=initial_ref,
                initial_role=role,
            )
        except PlanningBlockedError as error:
            if session_id is None:
                self._transition(run_id, WorkflowState.BLOCKED, str(error))
                self.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)
                return PlanningOutcome(
                    run_id=run_id, state=WorkflowState.BLOCKED, blocker_reason=str(error)
                )

            self.db_manager.record_decision(
                str(uuid.uuid4()),
                run_id,
                "planning_recovery",
                "prior CLI session unavailable; reconstructing context in new session",
            )
            try:
                return await self._planning_loop(
                    run_id,
                    task_description,
                    project_context,
                    initial_prompt=_build_recovery_prompt(task_description, decisions, False),
                )
            except PlanningBlockedError as fallback_error:
                self._transition(run_id, WorkflowState.BLOCKED, str(fallback_error))
                self.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)
                return PlanningOutcome(
                    run_id=run_id,
                    state=WorkflowState.BLOCKED,
                    blocker_reason=str(fallback_error),
                )

    async def _planning_loop(
        self,
        run_id: str,
        task_description: str,
        project_context: ProjectContext,
        initial_prompt: str | None = None,
        initial_session_id: str | None = None,
        initial_ref: ModelRef | None = None,
        initial_role: AgentRole = AgentRole.DEFAULT_PLANNER,
    ) -> PlanningOutcome:
        current_ref = initial_ref or self._default_model_ref
        role = initial_role
        session_id = initial_session_id
        escalated = current_ref == self._architecture_model_ref
        architecture_flags = self.routing_rules.planning.architecture_if_any
        prompt = initial_prompt or _build_initial_prompt(
            task_description, project_context, architecture_flags
        )

        for _turn in range(self.max_turns):
            adapter = self.agent_registry.get(current_ref.provider_enum)
            result = await self._invoke_planner(
                run_id, adapter, current_ref.model, role, project_context, prompt, session_id
            )
            session_id = result.session_id or session_id

            planner_turn, result = await self._parse_turn_with_retry(
                run_id, adapter, current_ref.model, role, project_context, result, session_id
            )
            session_id = result.session_id or session_id

            if planner_turn.escalation and not escalated:
                flags = [f for f in planner_turn.escalation.flags if f in architecture_flags]
                if flags:
                    escalated = True
                    new_ref = self._architecture_model_ref
                    if new_ref.provider_enum != current_ref.provider_enum:
                        # A CLI session ID from the old provider means nothing to the new one.
                        session_id = None
                    current_ref = new_ref
                    role = AgentRole.ARCHITECTURE_PLANNER
                    reason = planner_turn.escalation.reason or ", ".join(flags)
                    self.db_manager.record_decision(
                        str(uuid.uuid4()),
                        run_id,
                        "architecture_escalation",
                        f"flags={flags}; reason={reason}",
                    )
                    self.ui.print_info(
                        f"Escalating planning to {current_ref.model} ({', '.join(flags)}): {reason}"
                    )

            if planner_turn.status == PlannerStatus.BLOCKED:
                raise PlanningBlockedError(
                    planner_turn.blocker_reason or "Planner reported a blocker."
                )

            if planner_turn.status == PlannerStatus.QUESTIONS:
                if not planner_turn.questions:
                    raise PlanningBlockedError(
                        "Planner reported 'questions' status with no questions attached."
                    )
                self._transition(run_id, WorkflowState.WAITING_FOR_USER, "Awaiting user answers")
                qa_pairs: list[tuple[str, str]] = []
                for question in planner_turn.questions:
                    answer = self.question_prompt(question)
                    self.db_manager.record_decision(
                        str(uuid.uuid4()), run_id, question.question, answer
                    )
                    qa_pairs.append((question.question, answer))
                self._transition(
                    run_id, WorkflowState.PLANNING, "Resuming planning with user answers"
                )
                prompt = _build_answers_prompt(qa_pairs)
                continue

            # status == PLAN_READY
            if not planner_turn.plan_markdown or not planner_turn.task_profile:
                raise PlanningBlockedError(
                    "Planner reported 'plan_ready' without both plan_markdown and task_profile."
                )

            self._transition(run_id, WorkflowState.PLAN_READY, "Plan ready for approval")
            self.ui.print_header("Proposed Plan")
            self.ui.console.print(planner_turn.plan_markdown)
            decision = self.approval_prompt()

            if decision == ApprovalDecision.CANCEL:
                self._transition(run_id, WorkflowState.CANCELLED, "User cancelled plan approval")
                self.db_manager.update_run_status(run_id, RunStatus.CANCELLED.value)
                return PlanningOutcome(run_id=run_id, state=WorkflowState.CANCELLED)

            if decision == ApprovalDecision.CONTINUE_PLANNING:
                feedback = self.feedback_prompt()
                self.db_manager.record_decision(
                    str(uuid.uuid4()), run_id, "Plan feedback requested", feedback
                )
                self._transition(
                    run_id, WorkflowState.PLANNING, "Continuing planning after feedback"
                )
                prompt = _build_feedback_prompt(feedback)
                continue

            # APPROVE
            self._transition(run_id, WorkflowState.PLAN_APPROVED, "Plan approved by user")
            plan_path = self._write_artifact(
                project_context, run_id, "approved-plan.md", planner_turn.plan_markdown
            )
            task_profile_path = self._write_artifact(
                project_context,
                run_id,
                "task-profile.json",
                planner_turn.task_profile.model_dump_json(indent=2),
            )
            self._transition(run_id, WorkflowState.TASK_CLASSIFIED, "TaskProfile generated")
            self.db_manager.update_run_status(run_id, RunStatus.COMPLETED.value)
            self.ui.print_success(f"Run {run_id} classified. Plan and TaskProfile persisted.")
            return PlanningOutcome(
                run_id=run_id,
                state=WorkflowState.TASK_CLASSIFIED,
                plan_markdown=planner_turn.plan_markdown,
                task_profile=planner_turn.task_profile,
                plan_path=plan_path,
                task_profile_path=task_profile_path,
            )

        raise PlanningBlockedError(f"Planning did not converge within {self.max_turns} turns.")

    async def _invoke_planner(
        self,
        run_id: str,
        adapter: AgentAdapter,
        model: str,
        role: AgentRole,
        project_context: ProjectContext,
        prompt: str,
        session_id: str | None,
    ) -> AgentResult:
        """Invoke the planner CLI for one turn and record the resulting agent session."""
        request = AgentRequest(
            role=role,
            prompt=prompt,
            repository_path=project_context.root_path,
            model=model,
            read_only=True,
        )
        record_agent_started(
            self.db_manager, run_id, WorkflowState.PLANNING.value, adapter.provider.value, model
        )
        async with self.ui.animate_stage(f"Planning: {model}"):
            result = (
                await adapter.resume(session_id, request)
                if session_id
                else await adapter.start(request)
            )

        self.db_manager.record_agent_session(
            str(uuid.uuid4()),
            run_id,
            WorkflowState.PLANNING.value,
            adapter.provider.value,
            model,
            result.session_id,
        )
        record_agent_completed(self.db_manager, run_id, WorkflowState.PLANNING.value, result)

        if not result.success:
            message = describe_agent_failure(result, "Planner CLI")
            if is_authentication_failure(result):
                raise PlanningBlockedError(message) from AdapterAuthenticationError(message)
            raise PlanningBlockedError(message)
        return result

    async def _parse_turn_with_retry(
        self,
        run_id: str,
        adapter: AgentAdapter,
        model: str,
        role: AgentRole,
        project_context: ProjectContext,
        result: AgentResult,
        session_id: str | None,
    ) -> tuple[PlannerTurn, AgentResult]:
        """Parse the planner's structured output, retrying with a correction prompt on failure."""
        current_result = result
        current_session_id = session_id
        last_error = ""

        for attempt in range(self.max_malformed_retries + 1):
            try:
                return parse_model_into_schema(current_result.text, PlannerTurn), current_result
            except StructuredParsingError as e:
                last_error = str(e)
                if attempt >= self.max_malformed_retries:
                    break
                current_result = await self._invoke_planner(
                    run_id,
                    adapter,
                    model,
                    role,
                    project_context,
                    _build_retry_prompt(last_error),
                    current_session_id,
                )
                current_session_id = current_result.session_id or current_session_id

        raise PlanningBlockedError(
            f"Planner returned malformed structured output after "
            f"{self.max_malformed_retries + 1} attempt(s): {last_error}"
        )

    def _transition(self, run_id: str, to_state: WorkflowState, reason: str) -> None:
        """Validate and persist a workflow state transition for a run."""
        current_run = self.db_manager.get_run(run_id)
        current_state = WorkflowState(current_run.state) if current_run else WorkflowState.NEW
        validate_transition(current_state, to_state)
        self.db_manager.update_run_state(run_id, to_state.value, reason)

    def _write_artifact(
        self,
        project_context: ProjectContext,
        run_id: str,
        filename: str,
        content: str,
    ) -> Path:
        """Persist a human/machine-readable run artifact under .ai-orchestrator/runs/<run-id>/."""
        run_dir = project_context.root_path / ".ai-orchestrator" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        path.write_text(content, encoding="utf-8")
        return path
