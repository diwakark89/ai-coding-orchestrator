"""Workflow state machine for AgentFlow runs."""

from enum import Enum

from agentflow.errors import InvalidStateTransitionError
from agentflow.persistence.database import DatabaseManager


class WorkflowState(str, Enum):
    """Granular workflow state for a run, persisted on every transition."""

    NEW = "NEW"
    PROJECT_READY = "PROJECT_READY"
    PLANNING = "PLANNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    PLAN_READY = "PLAN_READY"
    PLAN_APPROVED = "PLAN_APPROVED"
    TASK_CLASSIFIED = "TASK_CLASSIFIED"

    WORKTREE_READY = "WORKTREE_READY"
    IMPLEMENTING = "IMPLEMENTING"
    VERIFYING = "VERIFYING"
    REPAIRING = "REPAIRING"

    REVIEWING = "REVIEWING"
    REVIEW_FIXING = "REVIEW_FIXING"
    REVIEW_APPROVED = "REVIEW_APPROVED"

    READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
    COMPLETED = "COMPLETED"

    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ALLOWED_TRANSITIONS: dict[WorkflowState, frozenset[WorkflowState]] = {
    WorkflowState.NEW: frozenset(
        {WorkflowState.PROJECT_READY, WorkflowState.FAILED, WorkflowState.CANCELLED}
    ),
    WorkflowState.PROJECT_READY: frozenset(
        {WorkflowState.PLANNING, WorkflowState.FAILED, WorkflowState.CANCELLED}
    ),
    WorkflowState.PLANNING: frozenset(
        {
            WorkflowState.WAITING_FOR_USER,
            WorkflowState.PLAN_READY,
            WorkflowState.BLOCKED,
            WorkflowState.FAILED,
            WorkflowState.CANCELLED,
        }
    ),
    WorkflowState.WAITING_FOR_USER: frozenset(
        {WorkflowState.PLANNING, WorkflowState.BLOCKED, WorkflowState.CANCELLED}
    ),
    WorkflowState.PLAN_READY: frozenset(
        {
            WorkflowState.PLANNING,
            WorkflowState.PLAN_APPROVED,
            WorkflowState.BLOCKED,
            WorkflowState.CANCELLED,
        }
    ),
    WorkflowState.PLAN_APPROVED: frozenset({WorkflowState.TASK_CLASSIFIED, WorkflowState.FAILED}),
    WorkflowState.TASK_CLASSIFIED: frozenset(
        {WorkflowState.WORKTREE_READY, WorkflowState.BLOCKED, WorkflowState.FAILED}
    ),
    WorkflowState.WORKTREE_READY: frozenset(
        {WorkflowState.IMPLEMENTING, WorkflowState.BLOCKED, WorkflowState.FAILED}
    ),
    WorkflowState.IMPLEMENTING: frozenset(
        {WorkflowState.VERIFYING, WorkflowState.BLOCKED, WorkflowState.FAILED}
    ),
    WorkflowState.VERIFYING: frozenset(
        {
            WorkflowState.REPAIRING,
            WorkflowState.REVIEWING,
            WorkflowState.BLOCKED,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.REPAIRING: frozenset(
        {WorkflowState.VERIFYING, WorkflowState.BLOCKED, WorkflowState.FAILED}
    ),
    WorkflowState.REVIEWING: frozenset(
        {
            WorkflowState.REVIEW_FIXING,
            WorkflowState.REVIEW_APPROVED,
            WorkflowState.BLOCKED,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.REVIEW_FIXING: frozenset(
        {WorkflowState.VERIFYING, WorkflowState.BLOCKED, WorkflowState.FAILED}
    ),
    WorkflowState.REVIEW_APPROVED: frozenset(
        {WorkflowState.READY_FOR_APPROVAL, WorkflowState.BLOCKED, WorkflowState.FAILED}
    ),
    WorkflowState.READY_FOR_APPROVAL: frozenset(
        {
            WorkflowState.COMPLETED,
            WorkflowState.CANCELLED,
            WorkflowState.BLOCKED,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.COMPLETED: frozenset(),
    WorkflowState.BLOCKED: frozenset(),
    WorkflowState.FAILED: frozenset(),
    WorkflowState.CANCELLED: frozenset(),
}


def validate_transition(current: WorkflowState, target: WorkflowState) -> None:
    """Raise InvalidStateTransitionError unless the transition is explicitly allowed."""
    if target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
        raise InvalidStateTransitionError(
            f"Invalid workflow state transition: {current.value} -> {target.value}"
        )


def transition_run_state(
    db_manager: DatabaseManager, run_id: str, to_state: WorkflowState, reason: str
) -> None:
    """Validate and persist a workflow state transition for a run."""
    current_run = db_manager.get_run(run_id)
    current_state = WorkflowState(current_run.state) if current_run else WorkflowState.NEW
    validate_transition(current_state, to_state)
    db_manager.update_run_state(run_id, to_state.value, reason)
