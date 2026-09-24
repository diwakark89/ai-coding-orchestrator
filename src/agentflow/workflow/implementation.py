"""Codex-driven implementation workflow: build the approved plan inside an isolated worktree.

Implementation occurs strictly inside a disposable Git worktree (AGENTS.md invariant),
under a single-writer lock, and is never considered successful merely because the agent
claims so — an empty diff after a claimed success is treated as a blocked no-op.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path

from agentflow.agents.base import AgentRequest, AgentRole, describe_agent_failure
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.git.lock import WorktreeLock
from agentflow.git.worktree import WorktreeDiff, WorktreeHandle, WorktreeManager
from agentflow.observability.events import record_agent_completed, record_agent_started
from agentflow.persistence.database import DatabaseManager
from agentflow.project.context import ProjectContext
from agentflow.routing.decision import RoutingDecision
from agentflow.task.profile import TaskProfile
from agentflow.ui.console import ConsoleUI
from agentflow.workflow.states import WorkflowState, transition_run_state
from agentflow.workflow.verification import AGENT_VERIFICATION_GUIDANCE

_TIER_ROLES: dict[str, AgentRole] = {
    "implementation.lightweight": AgentRole.LIGHTWEIGHT_CODER,
    "implementation.standard": AgentRole.STANDARD_CODER,
    "implementation.escalation": AgentRole.IMPLEMENTATION_ESCALATION,
}


@dataclass
class ImplementationOutcome:
    """Terminal result of the implementation workflow for a single attempt."""

    run_id: str
    state: WorkflowState
    worktree: WorktreeHandle | None = None
    diff: WorktreeDiff | None = None
    model: str | None = None
    summary_path: Path | None = None
    blocker_reason: str | None = None


def _build_implementation_prompt(
    task_description: str,
    plan_markdown: str,
    task_profile: TaskProfile,
    worktree_path: Path,
) -> str:
    """Build the implementation prompt: plan, TaskProfile, acceptance criteria, and scope."""
    return (
        "You are AgentFlow's implementation agent, operating inside an isolated Git worktree.\n\n"
        f"Worktree path (your only writable scope): {worktree_path}\n\n"
        f"Task:\n{task_description}\n\n"
        f"Approved plan:\n{plan_markdown}\n\n"
        f"TaskProfile:\n{task_profile.model_dump_json(indent=2)}\n\n"
        "Acceptance criteria: implement exactly what the approved plan describes, satisfying "
        "its Implementation Steps and Acceptance Criteria sections.\n\n"
        "Scope constraints:\n"
        "- Only modify files inside this worktree.\n"
        "- Do not redesign the approved architecture.\n"
        "- Do not modify files unrelated to this task.\n\n"
        f"{AGENT_VERIFICATION_GUIDANCE}\n\n"
        "If the plan is invalid or impossible to implement as written, stop and report the "
        "blocker instead of improvising a different design."
    )


class ImplementationWorkflow:
    """Creates an isolated worktree and drives Codex (or an escalated model) to implement it."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        agent_registry: AgentAdapterRegistry,
        worktree_manager: WorktreeManager,
        console_ui: ConsoleUI | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.agent_registry = agent_registry
        self.worktree_manager = worktree_manager
        self.ui = console_ui or ConsoleUI()

    async def run(
        self,
        run_id: str,
        task_description: str,
        project_context: ProjectContext,
        plan_markdown: str,
        task_profile: TaskProfile,
        routing_decision: RoutingDecision,
    ) -> ImplementationOutcome:
        """Create the worktree, invoke the routed implementation agent, and capture the diff."""
        try:
            handle = await self.worktree_manager.create(
                project_context.project_name, run_id, project_context.root_path
            )
        except Exception as e:
            transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, str(e))
            return ImplementationOutcome(
                run_id=run_id, state=WorkflowState.BLOCKED, blocker_reason=str(e)
            )

        transition_run_state(
            self.db_manager,
            run_id,
            WorkflowState.WORKTREE_READY,
            f"Worktree created at {handle.path}",
        )
        transition_run_state(
            self.db_manager, run_id, WorkflowState.IMPLEMENTING, "Implementation started"
        )

        lock = WorktreeLock(handle.path)
        try:
            lock.acquire()
        except Exception as e:
            transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, str(e))
            return ImplementationOutcome(
                run_id=run_id, state=WorkflowState.BLOCKED, worktree=handle, blocker_reason=str(e)
            )

        try:
            return await self._implement(
                run_id,
                task_description,
                project_context,
                plan_markdown,
                task_profile,
                routing_decision,
                handle,
            )
        finally:
            lock.release()

    async def _implement(
        self,
        run_id: str,
        task_description: str,
        project_context: ProjectContext,
        plan_markdown: str,
        task_profile: TaskProfile,
        routing_decision: RoutingDecision,
        handle: WorktreeHandle,
    ) -> ImplementationOutcome:
        adapter = self.agent_registry.get(routing_decision.provider)
        role = _TIER_ROLES.get(routing_decision.role, AgentRole.IMPLEMENTER)
        prompt = _build_implementation_prompt(
            task_description, plan_markdown, task_profile, handle.path
        )
        request = AgentRequest(
            role=role,
            prompt=prompt,
            repository_path=project_context.root_path,
            working_directory=handle.path,
            model=routing_decision.model,
            read_only=False,
        )
        record_agent_started(
            self.db_manager,
            run_id,
            WorkflowState.IMPLEMENTING.value,
            adapter.provider.value,
            routing_decision.model,
        )
        async with self.ui.animate_stage(f"Implementing: {routing_decision.model}"):
            result = await adapter.start(request)

        self.db_manager.record_agent_session(
            str(uuid.uuid4()),
            run_id,
            WorkflowState.IMPLEMENTING.value,
            adapter.provider.value,
            routing_decision.model,
            result.session_id,
        )
        record_agent_completed(self.db_manager, run_id, WorkflowState.IMPLEMENTING.value, result)

        if not result.success:
            reason = describe_agent_failure(result, "Implementation agent")
            transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, reason)
            return ImplementationOutcome(
                run_id=run_id,
                state=WorkflowState.BLOCKED,
                worktree=handle,
                model=routing_decision.model,
                blocker_reason=reason,
            )

        diff = await self.worktree_manager.capture_changes(handle.path)

        if diff.is_empty:
            reason = "Implementation agent reported success but produced no changes (no-op)."
            transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, reason)
            return ImplementationOutcome(
                run_id=run_id,
                state=WorkflowState.BLOCKED,
                worktree=handle,
                diff=diff,
                model=routing_decision.model,
                blocker_reason=reason,
            )

        summary_path = self._write_summary_artifact(project_context, run_id, result.text, diff)
        transition_run_state(
            self.db_manager,
            run_id,
            WorkflowState.VERIFYING,
            "Implementation produced changes; verifying",
        )
        return ImplementationOutcome(
            run_id=run_id,
            state=WorkflowState.VERIFYING,
            worktree=handle,
            diff=diff,
            model=routing_decision.model,
            summary_path=summary_path,
        )

    def _write_summary_artifact(
        self, project_context: ProjectContext, run_id: str, agent_text: str, diff: WorktreeDiff
    ) -> Path:
        """Persist implementation-summary.md describing what the agent changed."""
        run_dir = project_context.root_path / ".ai-orchestrator" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "implementation-summary.md"
        changed = "\n".join(f"- {f}" for f in diff.changed_files) or "- (none)"
        content = (
            f"# Implementation Summary\n\n"
            f"## Agent Output\n\n{agent_text}\n\n"
            f"## Changed Files\n\n{changed}\n"
        )
        path.write_text(content, encoding="utf-8")
        return path
